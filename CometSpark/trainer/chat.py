"""交互式多轮对话会话（``ChatSession``）。

把「加载一次模型 + 多轮对话」封装成可复用的对象，供 ``run.py chat`` 与
``run.py generate --chat`` 共用，避免每次提问都重新读 checkpoint。

性能要点（相对逐次调用 ``pipeline.generate``）：

1. **模型只加载一次**：``pipeline.generate`` 每次都要 ``torch.load`` 一个数 GB
   的 checkpoint 并重建模型；REPL 里这是最贵的开销。
2. **设备自动选择**：默认 ``device="auto"``，在有昇腾/CUDA 时用加速器并套
   bf16 autocast。此前 ``generate`` 的默认设备是 ``cpu``，NPU 机器上也只跑
   CPU。
3. **``torch.inference_mode``**：关闭 autograd 版本计数与图构建。
4. **资源锁**：按 :class:`~verse_trainer.resources.ResourceConfig` 限制线程/显存
   （NPU 默认 70%），避免推理进程吃满整机。

多轮历史以 ChatML 消息列表保存，每轮把「历史 + 新提问」整体渲染后送模型。
``VerseTransformer.generate`` 不接受外部 KV 缓存，故每轮都会重新 prefill 历史；
对话轮数很多时可调小 ``max_history_turns`` 控制上下文长度。
"""

from __future__ import annotations

import torch

from verse_core import get_logger
from verse_nn.devices import detect_backend

from tokenizer import ENDOFTEXT

logger = get_logger("cometspark.chat")

__all__ = [
    "ChatSession",
    "build_messages",
    "extract_reply",
    "render_messages",
    "resolve_inference_dtype",
    "trim_history",
]

# ChatML 的结束标记：回答以 <|im_end|> 结尾，解码时要裁掉
_IM_END = "<|im_end|>"


class ChatSession:
    """多轮对话会话：常驻模型，维护消息历史。

    用法::

        session = ChatSession("checkpoints/xxx_merged.pt")
        print(session.ask("你好"))
        print(session.ask("再讲一个"))   # 自动带上上文
        session.reset()                # 清空历史

    Args:
        ckpt_path: checkpoint 路径（需含 ``model_state``；LoRA 适配器请用其
            ``*_merged.pt``）。
        device: ``auto`` / ``cpu`` / ``cuda`` / ``npu``；``auto`` 按设备优先级
            自动选（NPU > CUDA > CPU）。
        dtype: ``auto``（按设备能力选 bf16/fp16，CPU 上多为 fp32）/ ``fp32`` /
            ``bf16`` / ``fp16``。
        system: 可选的 system 提示，插入到历史最前。
        max_history_turns: 最多保留的历史对话轮数（一问一答算一轮），
            0 表示不限制。超出后丢弃最早的轮次。
        num_threads: CPU 线程数覆盖（0 = 按资源锁比例）。
        resources: :class:`~verse_trainer.resources.ResourceConfig`；None 用默认。
        **gen_kwargs: 生成参数（temperature/top_k/top_p/repetition_penalty/
            no_repeat_ngram/greedy/max_new_tokens）。
    """

    def __init__(
        self,
        ckpt_path: str,
        *,
        device: str = "auto",
        dtype: str = "auto",
        system: str | None = None,
        max_history_turns: int = 12,
        num_threads: int = 0,
        resources=None,
        **gen_kwargs,
    ) -> None:
        from .pipeline import load_for_inference

        self.backend = detect_backend(device)
        self.device = torch.device(self.backend.device_type)
        self.dtype = self._resolve_dtype(dtype, self.backend)
        self.autocast_enabled = self.dtype != torch.float32

        # 资源锁在 load_for_inference 内统一施加（generate 也走同一条路），
        # 这里只回读生效的线程数用于日志，避免重复加锁。
        self.model, self.tokenizer, self.model_config = load_for_inference(
            ckpt_path, device=self.backend.device_type,
            num_threads=num_threads, resources=resources,
        )
        self.num_threads = torch.get_num_threads()
        self.model = self.model.to(self.device)
        self.model.eval()

        self.template = self._build_template()
        self.system = system
        self.max_history_turns = max_history_turns
        self.gen_kwargs = {
            "max_new_tokens": 128,
            "temperature": 0.7,
            "top_k": 40,
            "top_p": 0.9,
            "repetition_penalty": 1.1,
            "no_repeat_ngram": 3,
            "greedy": False,
            **gen_kwargs,
        }
        self.history: list[dict] = []
        self._stop_ids = self._resolve_stop_ids()
        self._eos_id = self._resolve_eos_id()
        logger.info(
            "对话会话就绪：device=%s dtype=%s 线程=%d 模型=%s",
            self.backend.name, self.dtype, self.num_threads,
            type(self.model).__name__,
        )

    # ------------------------------------------------------------ 初始化辅助

    @staticmethod
    def _resolve_dtype(dtype: str, backend) -> torch.dtype:
        """解析推理 dtype（见模块级 :func:`resolve_inference_dtype`）。"""
        return resolve_inference_dtype(dtype, backend)

    def _build_template(self):
        """从 tokenizer 目录加载 ChatTemplate（缺失则回退手写 ChatML）。"""
        tokenizer_dir = getattr(self.tokenizer, "_tokenizer_dir", None)
        if tokenizer_dir:
            try:
                from verse_tokenizer import ChatTemplate

                return ChatTemplate.from_pretrained(tokenizer_dir)
            except Exception as exc:  # pragma: no cover - 模板缺失
                logger.warning("加载 ChatTemplate 失败，回退内置 ChatML: %s", exc)
        return _FallbackChatML()

    def _resolve_stop_ids(self) -> set[int]:
        special = getattr(self.tokenizer, "special_tokens", {}) or {}
        return {tid for tid in (special.get(_IM_END),) if tid is not None}

    def _resolve_eos_id(self) -> int | None:
        special = getattr(self.tokenizer, "special_tokens", {}) or {}
        # 对话模型以 <|im_end|> 结束回答；没有则用 endoftext
        return special.get(_IM_END) or special.get(ENDOFTEXT)

    # -------------------------------------------------------------- 对话

    def reset(self) -> None:
        """清空历史（保留 system 提示）。"""
        self.history = []

    def _messages(self, user_text: str) -> list[dict]:
        """构造本轮送模型的完整消息列表（含裁剪后的历史）。"""
        return build_messages(self.history, user_text, self.system)

    def _trim_history(self) -> None:
        self.history = trim_history(self.history, self.max_history_turns)

    @torch.inference_mode()
    def ask(self, user_text: str) -> str:
        """发起一轮对话：返回助手回复，并把本轮问答追加进历史。"""
        if not user_text.strip():
            raise ValueError("输入为空")
        messages = self._messages(user_text)
        prompt = render_messages(
            self.template, messages, getattr(self.tokenizer, "special_tokens", {}) or {}
        )
        ids = self.tokenizer.encode(prompt)
        if not ids:
            raise ValueError("prompt 分词结果为空")
        input_ids = torch.tensor([ids], dtype=torch.long, device=self.device)
        with self.backend.autocast(self.dtype, enabled=self.autocast_enabled):
            out = self.model.generate(
                input_ids,
                eos_token_id=self._eos_id,
                stop_token_ids=self._stop_ids,
                **self.gen_kwargs,
            )
        reply = extract_reply(self.tokenizer.decode(out[0].tolist()))
        self.history.append({"role": "user", "content": user_text})
        self.history.append({"role": "assistant", "content": reply})
        self._trim_history()
        return reply

    @torch.inference_mode()
    def complete(self, prompt: str) -> str:
        """自由续写（不加对话模板），返回完整解码文本（含 prompt）。"""
        ids = self.tokenizer.encode(prompt)
        if not ids:
            raise ValueError("prompt 分词结果为空")
        input_ids = torch.tensor([ids], dtype=torch.long, device=self.device)
        with self.backend.autocast(self.dtype, enabled=self.autocast_enabled):
            out = self.model.generate(
                input_ids,
                eos_token_id=self._eos_id,
                stop_token_ids=self._stop_ids,
                **self.gen_kwargs,
            )
        return self.tokenizer.decode(out[0].tolist(), skip_special_tokens=True)

    # -------------------------------------------------------------- 展示

    @property
    def turns(self) -> int:
        """已完成的对话轮数。"""
        return len(self.history) // 2


# --------------------------------------------------------------- 纯函数工具
#
# 这几件事与「模型怎么加载」无关，抽成模块级函数便于单测（无需真实 checkpoint）。

#: 显式 dtype 名 -> torch dtype。不能写 ``getattr(torch, "fp16")``——torch 只有
#: ``float16``/``bfloat16``，没有 ``fp16``/``bf16`` 别名（历史上这里踩过坑）。
_DTYPE_MAP = {
    "fp32": torch.float32,
    "float32": torch.float32,
    "bf16": torch.bfloat16,
    "bfloat16": torch.bfloat16,
    "fp16": torch.float16,
    "float16": torch.float16,
}


def resolve_inference_dtype(dtype: str, backend) -> torch.dtype:
    """解析推理 dtype：``fp32``/``bf16``/``fp16`` 显式优先，``auto`` 用设备推荐值。"""
    explicit = _DTYPE_MAP.get(str(dtype).lower())
    if explicit is not None:
        return explicit
    recommended = backend.default_autocast_dtype()
    if recommended is not None and backend.supports_dtype(recommended):
        return recommended
    return torch.float32


def build_messages(history: list[dict], user_text: str,
                   system: str | None = None) -> list[dict]:
    """拼出本轮完整消息列表：``[system?] + history + 本轮 user``。"""
    msgs: list[dict] = []
    if system:
        msgs.append({"role": "system", "content": system})
    msgs.extend(history)
    msgs.append({"role": "user", "content": user_text})
    return msgs


def trim_history(history: list[dict], max_turns: int) -> list[dict]:
    """按轮数裁剪历史：整轮丢弃（保持 user/assistant 配对），``max_turns<=0`` 不裁。"""
    if not max_turns:
        return history
    max_msgs = max_turns * 2
    return history[-max_msgs:] if len(history) > max_msgs else history


def render_messages(template, messages: list[dict], special_tokens: dict) -> str:
    """渲染消息列表为 prompt 文本。

    tokenizer 无 ``<|bos|>`` 时显式传空串，避免 BPE 兜底把未知 token 编成
    越界 id（历史事故：表现为开头乱码）。
    """
    kwargs = {} if "<|bos|>" in special_tokens else {"bos_token": ""}
    return template.render(messages, add_generation_prompt=True, **kwargs)


def extract_reply(text: str) -> str:
    """从解码文本里裁出助手回答（去掉模板前缀与结束标记）。"""
    marker = "assistant\n"
    if marker in text:
        text = text.rsplit(marker, 1)[-1]
    for stop in (_IM_END, ENDOFTEXT):
        idx = text.find(stop)
        if idx >= 0:
            text = text[:idx]
    return text.strip()


class _FallbackChatML:
    """``verse_tokenizer.ChatTemplate`` 不可用时的内置 ChatML 渲染。

    只实现最小子集：``system`` / ``user`` / ``assistant`` 三种角色 +
    ``add_generation_prompt``。正常路径不会走到这里。
    """

    def render(self, messages: list[dict], add_generation_prompt: bool = True,
               bos_token: str = "<|bos|>") -> str:
        parts = [bos_token] if bos_token else []
        for msg in messages:
            parts.append(
                f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>\n"
            )
        if add_generation_prompt:
            parts.append("<|im_start|>assistant\n")
        return "".join(parts)
