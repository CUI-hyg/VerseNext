"""CometSpark 交互式对话（``ChatSession``）测试。

分两层：

1. **纯逻辑**：历史裁剪 / 消息拼装 / 模板渲染 / 回答裁剪 / dtype 解析——这些
   与「模型怎么加载」无关，直接用假对象验证，任何设备上都能跑。
2. **端到端**：用一份极小的配置训出 checkpoint，再让 ``ChatSession`` 加载并
   完成一轮多轮对话（验证设备解析、autocast、模板、历史累积真的接通了）。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

_COMETSPARK = Path(__file__).resolve().parents[2] / "CometSpark"

# 与 test_cometspark_chain.py 同款极小配置：1 层、d_model 32、词表与真实
# tokenizer 对齐（否则 encode 出的 id 会被 generate 的 clamp 截断）。
_CONFIG = """\
model_name: cometspark
model:
  vocab_size: 248077
  d_model: 32
  num_layers: 1
  num_heads: 2
  max_seq_len: 64
  dropout: 0.0
  tie_weights: true
  gradient_checkpointing: false
  attn_type: sdpa
data:
  batch_size: 2
  seq_len: 16
  seed: 42
  drop_last: true
optim:
  name: adamw
  lr: 3.0e-4
  warmup_steps: 1
  max_steps: 2
  min_lr_ratio: 0.1
steps: 2
log_every: 1
eval_every: 0
ckpt_dir: checkpoints
device: cpu
dtype: fp32
compile: false
grad_accum_steps: 1
seed: 42
"""


@pytest.fixture(scope="module")
def chat_mod():
    """导入 ``trainer.chat``（需要 CometSpark 在 sys.path 上）。"""
    if not (_COMETSPARK / "trainer" / "chat.py").exists():
        pytest.skip("工作区无 CometSpark/trainer/chat.py")
    if str(_COMETSPARK) not in sys.path:
        sys.path.insert(0, str(_COMETSPARK))
    import trainer.chat as chat  # noqa: PLC0415

    return chat


# --------------------------------------------------------------- 纯逻辑

def test_build_messages_appends_user_after_history(chat_mod):
    history = [
        {"role": "user", "content": "你好"},
        {"role": "assistant", "content": "你好！"},
    ]
    msgs = chat_mod.build_messages(history, "再讲一个", system="你是助手")
    assert msgs[0] == {"role": "system", "content": "你是助手"}
    assert [m["role"] for m in msgs] == ["system", "user", "assistant", "user"]
    assert msgs[-1]["content"] == "再讲一个"
    # 历史本身不被修改
    assert len(history) == 2


def test_build_messages_without_system(chat_mod):
    msgs = chat_mod.build_messages([], "hi")
    assert msgs == [{"role": "user", "content": "hi"}]


def test_trim_history_drops_whole_turns(chat_mod):
    history = []
    for i in range(5):
        history += [{"role": "user", "content": f"q{i}"},
                    {"role": "assistant", "content": f"a{i}"}]
    trimmed = chat_mod.trim_history(history, 2)
    assert len(trimmed) == 4
    # 保留最近的 2 轮，且从 user 开始（配对不被打断）
    assert trimmed[0] == {"role": "user", "content": "q3"}
    assert trimmed[-1] == {"role": "assistant", "content": "a4"}


def test_trim_history_noop_when_within_limit_or_disabled(chat_mod):
    history = [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]
    assert chat_mod.trim_history(history, 4) == history
    assert chat_mod.trim_history(history, 0) is history


def test_extract_reply_strips_template_and_stop_markers(chat_mod):
    raw = "<|im_start|>user\n你好<|im_end|>\n<|im_start|>assistant\n我是助手<|im_end|>\n"
    assert chat_mod.extract_reply(raw) == "我是助手"
    # 没有 assistant 标记时原样返回（只裁停止符）
    assert chat_mod.extract_reply("答案<|endoftext|>尾巴") == "答案"
    # 模板标记在回答正文中出现时，只裁到第一个停止符
    assert chat_mod.extract_reply("assistant\n甲<|im_end|>乙") == "甲"


def test_render_messages_omits_bos_when_tokenizer_lacks_it(chat_mod):
    captured: dict = {}

    class _T:
        def render(self, messages, add_generation_prompt=True, **kw):
            captured.clear()
            captured.update(kw)
            captured["n"] = len(messages)
            return "PROMPT"

    msgs = [{"role": "user", "content": "hi"}]
    assert chat_mod.render_messages(_T(), msgs, {"<|endoftext|>": 0}) == "PROMPT"
    assert captured == {"bos_token": "", "n": 1}
    # tokenizer 有 <|bos|> 时不传该 kwarg，交给模板默认值
    chat_mod.render_messages(_T(), msgs, {"<|bos|>": 1})
    assert captured == {"n": 1}


def test_fallback_chatml_render(chat_mod):
    tpl = chat_mod._FallbackChatML()
    out = tpl.render([{"role": "user", "content": "hi"}], add_generation_prompt=True)
    assert out == "<|bos|><|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n"
    # 不要求生成提示时不以 assistant 前缀结尾
    assert tpl.render([{"role": "user", "content": "hi"}],
                      add_generation_prompt=False).endswith("<|im_end|>\n")
    # bos_token="" 时不应留下空串痕迹
    assert not tpl.render([], add_generation_prompt=False, bos_token="").startswith("<")


def test_resolve_inference_dtype(chat_mod):
    class _Backend:
        def __init__(self, rec, supports=True):
            self._rec, self._supports = rec, supports

        def default_autocast_dtype(self):
            return self._rec

        def supports_dtype(self, _dtype):
            return self._supports

    assert chat_mod.resolve_inference_dtype("fp32", _Backend(None)) is torch.float32
    assert chat_mod.resolve_inference_dtype("fp16", _Backend(None)) is torch.float16
    assert chat_mod.resolve_inference_dtype("bf16", _Backend(None)) is torch.bfloat16
    # auto：用设备推荐值
    assert chat_mod.resolve_inference_dtype(
        "auto", _Backend(torch.bfloat16)) is torch.bfloat16
    # auto：设备没有推荐值 -> fp32
    assert chat_mod.resolve_inference_dtype("auto", _Backend(None)) is torch.float32
    # auto：设备推荐但声明不支持 -> fp32
    assert chat_mod.resolve_inference_dtype(
        "auto", _Backend(torch.float16, supports=False)) is torch.float32


# --------------------------------------------------------------- 端到端

@pytest.fixture(scope="module")
def chat_ckpt(tmp_path_factory):
    """训一个极小模型，产出可被 ``load_for_inference`` 加载的 checkpoint。"""
    if not (_COMETSPARK / "trainer" / "pipeline.py").exists():
        pytest.skip("工作区无 CometSpark/trainer")
    if str(_COMETSPARK) not in sys.path:
        sys.path.insert(0, str(_COMETSPARK))
    import trainer.pipeline as pipeline  # noqa: PLC0415

    root = tmp_path_factory.mktemp("chat_root")
    (root / "config").mkdir()
    (root / "data").mkdir()
    (root / "config" / "cfg.yaml").write_text(_CONFIG, encoding="utf-8")
    # train() 的默认数据路径是 data/train.jsonl
    (root / "data" / "train.jsonl").write_text(
        "\n".join('{"text": "机器学习是人工智能的一个分支。"}' for _ in range(60)),
        encoding="utf-8",
    )
    summary = pipeline.train(str(root / "config" / "cfg.yaml"), root)
    return Path(summary["checkpoint"])


def test_chat_session_single_turn_and_history(chat_mod, chat_ckpt):
    session = chat_mod.ChatSession(
        str(chat_ckpt), device="cpu", dtype="fp32", max_new_tokens=2,
        max_history_turns=2,
    )
    assert session.device.type == "cpu"
    assert session.dtype is torch.float32
    assert session.turns == 0

    reply = session.ask("你好")
    assert isinstance(reply, str)
    assert session.turns == 1
    assert session.history[0] == {"role": "user", "content": "你好"}
    assert session.history[1]["role"] == "assistant"

    session.ask("再讲一个")
    assert session.turns == 2
    # 超过 max_history_turns=2 后仍只保留最近 2 轮
    session.ask("第三个问题")
    assert session.turns == 2
    assert session.history[0]["content"] == "再讲一个"

    session.reset()
    assert session.turns == 0 and session.history == []


def test_chat_session_rejects_empty_input(chat_mod, chat_ckpt):
    session = chat_mod.ChatSession(
        str(chat_ckpt), device="cpu", dtype="fp32", max_new_tokens=1,
    )
    with pytest.raises(ValueError):
        session.ask("   ")


# --------------------------------------------------------------- run.py REPL

@pytest.fixture(scope="module")
def run_mod():
    """导入 ``CometSpark/run.py``（REPL 逻辑与真实 checkpoint 无关）。"""
    if not (_COMETSPARK / "run.py").exists():
        pytest.skip("工作区无 CometSpark/run.py")
    if str(_COMETSPARK) not in sys.path:
        sys.path.insert(0, str(_COMETSPARK))
    import importlib

    return importlib.import_module("run")


class _FakeSession:
    """最小会话桩：只记录调用，不碰模型。"""

    def __init__(self) -> None:
        self.history: list[dict] = []
        self.turns = 0
        self.reset_called = 0

    def ask(self, text: str) -> str:
        self.turns += 1
        self.history += [{"role": "user", "content": text},
                         {"role": "assistant", "content": f"回复:{text}"}]
        return f"回复:{text}"

    def reset(self) -> None:
        self.reset_called += 1
        self.history = []
        self.turns = 0


def _feed(monkeypatch, run_mod, lines: list[str]) -> None:
    """把 console.input 换成按序吐出给定行的迭代器。"""
    it = iter(lines)

    def _input(*_a, **_k):
        try:
            return next(it)
        except StopIteration:
            raise EOFError from None

    monkeypatch.setattr(run_mod.console, "input", _input)


def test_chat_repl_asks_and_handles_slash_commands(run_mod, monkeypatch):
    import io

    from rich.console import Console

    session = _FakeSession()
    # rich 的 Console 持有自己的文件句柄，capsys 抓不到；换成一个写进 StringIO 的
    # Console，再把它的 input 换成脚本化的行序列。
    buf = io.StringIO()
    monkeypatch.setattr(run_mod, "console",
                        Console(file=buf, width=200, force_terminal=False))
    # 空行应被忽略；/history 打印表格；/reset 清空；未知命令只提示；/exit 退出
    # （/reset 放在提问前，否则会把 turns 清零）
    _feed(monkeypatch, run_mod,
          ["", "/history", "/reset", "你好", "/history", "/bogus", "/exit",
           "不应被读到"])
    rc = run_mod._chat_repl(session)

    assert rc == 0
    assert session.reset_called == 1
    assert session.turns == 1                  # 只问了一次
    assert session.history[-1] == {"role": "assistant", "content": "回复:你好"}
    out = buf.getvalue()
    assert "回复:你好" in out
    assert "未知命令" in out
    assert "暂无历史" in out                    # 提问前的 /history


def test_chat_repl_exits_on_eof_and_keyboard_interrupt(run_mod, monkeypatch):
    def _eof(*_a, **_k):
        raise EOFError

    monkeypatch.setattr(run_mod.console, "input", _eof)
    assert run_mod._chat_repl(_FakeSession()) == 0

    def _ctrl_c(*_a, **_k):
        raise KeyboardInterrupt

    monkeypatch.setattr(run_mod.console, "input", _ctrl_c)
    assert run_mod._chat_repl(_FakeSession()) == 0


def test_chat_command_returns_false_only_for_exit(run_mod):
    session = _FakeSession()
    assert run_mod._handle_chat_command(session, "/exit") is False
    assert run_mod._handle_chat_command(session, "/quit") is False
    assert run_mod._handle_chat_command(session, "/reset") is True
    assert session.reset_called == 1
    assert run_mod._handle_chat_command(session, "/help") is True
