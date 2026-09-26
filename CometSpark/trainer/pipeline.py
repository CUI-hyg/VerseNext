"""CometSpark 训练/推理管线。

- 数据：JSONL（``{"text": ...}``）-> 分词 -> 文档间以 <|endoftext|> 分隔
  的 token 流，交给 verse_trainer 的随机窗口迭代器做 next-token 预测。
- 训练：TrainerConfig（yaml 加载）-> Trainer -> checkpoint。
- 推理：从 checkpoint 恢复模型，支持 <|endoftext|> 停止。
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from verse_core import EventBus, get_logger
from verse_nn.transformer import build_model, count_parameters
from verse_tokenizer.bpe import BPETokenizer
from verse_trainer.trainer import Trainer, TrainerConfig

from model import CometSparkConfig, MODEL_NAME
from tokenizer import ENDOFTEXT, load_endoftext_id, load_tokenizer

logger = get_logger("cometspark.pipeline")


# ------------------------------------------------------------------- data

# 文本字段识别与多格式加载统一由 verse_trainer.data_formats 提供；
# 此处保留 extract_text 兼容导出（历史调用方可能直接引用）。
from verse_trainer.data import fingerprint_source
from verse_trainer.data_formats import (  # noqa: E402
    extract_text,
    is_pretokenized,
    load_texts_auto,
    load_token_stream,
)


def load_texts(
    path: str | Path,
    *,
    text_column: str | None = None,
    text_columns: list[str] | None = None,
    text_template: str | None = None,
    delimiter: str | None = None,
) -> list[str]:
    """按文件后缀加载文本（jsonl/json/txt/md/csv/tsv/parquet）。

    字段识别见 :func:`verse_trainer.data_formats.extract_text`；无法识别时
    jsonl 会回退到整行 JSON 并告警（避免静默把 JSON 结构当作语料训练）。
    """
    return load_texts_auto(
        path, text_column=text_column, text_columns=text_columns,
        text_template=text_template, delimiter=delimiter,
    )


def texts_to_tokens(tok: BPETokenizer, texts: list[str]) -> list[int]:
    """文本列表 -> 单条 token 流，文档间以 <|endoftext|> 分隔。"""
    eos_id = load_endoftext_id(tok)
    if eos_id is None:
        eos_id = tok.add_special_token(ENDOFTEXT)
    ids: list[int] = []
    # 批量编码复用词级 BPE 缓存（真实语料重复词多，命中率高）
    for encoded in tok.encode_batch(texts):
        ids.extend(encoded)
        ids.append(eos_id)
    logger.info("分词完成：%d 篇文档 -> %d tokens", len(texts), len(ids))
    return ids


def _file_sha12(path: str | Path) -> str:
    """文件内容 hash 前 12 位（token 缓存的失效键）。"""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:12]


def _cache_key(path: str | Path, text_kwargs: dict | None) -> str:
    """缓存键：文件内容 + 文本抽取参数（参数变化时缓存失效）。"""
    suffix = ""
    if text_kwargs:
        suffix = hashlib.sha256(
            json.dumps(text_kwargs, ensure_ascii=False, sort_keys=True).encode()
        ).hexdigest()[:8]
    return _file_sha12(path) + (f"_{suffix}" if suffix else "")


def _tokens_with_cache(
    tok: BPETokenizer, split: str, path: str | Path, cache_dir: Path | None,
    text_kwargs: dict | None = None,
    token_dtype: str = "int32",
    mmap_tokens: bool = False,
) -> np.ndarray:
    """编码 + 磁盘缓存：源文件内容不变时直接加载 .npy，跳过重复分词。

    预分词文件（``.npy``/``.bin``/``.tokens``）直接加载，跳过分词与缓存。
    返回 **int32 numpy 数组**（而非 Python list）：省去 ``.tolist()`` 的大
    列表构造与后续 int64 全量拷贝，token 流内存占用减半。
    """
    if is_pretokenized(path):
        ids = load_token_stream(path, dtype=token_dtype, mmap=mmap_tokens)
        logger.info("预分词文件直载: %s (%d tokens)", Path(path).name, len(ids))
        return ids
    if cache_dir is None:
        return np.asarray(texts_to_tokens(tok, load_texts(path, **(text_kwargs or {}))),
                          dtype=np.int32)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{split}_{_cache_key(path, text_kwargs)}.npy"
    if cache_path.exists():
        ids = np.load(cache_path)
        logger.info("token 缓存命中: %s (%d tokens)", cache_path.name, len(ids))
        return ids
    ids = np.asarray(texts_to_tokens(tok, load_texts(path, **(text_kwargs or {}))),
                     dtype=np.int32)
    np.save(cache_path, ids)
    return ids


def prepare_tokens(
    tok: BPETokenizer,
    train_path: str | Path,
    val_path: str | Path | None = None,
    cache_dir: str | Path | None = "auto",
    text_kwargs: dict | None = None,
    token_dtype: str = "int32",
    mmap_tokens: bool = False,
) -> tuple[np.ndarray, np.ndarray | None]:
    """加载训练/验证数据并编码为 token 流。

    支持文本格式（jsonl/json/txt/md/csv/tsv/parquet，按后缀自动识别）与
    预分词文件（npy/bin/tokens，直接加载）。

    Args:
        tok: 分词器。
        train_path: 训练数据路径。
        val_path: 验证数据路径（可选）。
        cache_dir: 缓存目录；``auto`` 时使用数据文件所在目录的 ``.token_cache``，
            传 None 关闭缓存（预分词文件不受影响）。
        text_kwargs: 文本字段抽取参数（``text_column``/``text_columns``/
            ``text_template``/``delimiter``）。
        token_dtype: 预分词 raw 二进制的元素类型（int32/uint16/...）。
        mmap_tokens: ``.npy`` 是否用内存映射加载。
    """
    if cache_dir == "auto":
        cache_dir = Path(train_path).parent / ".token_cache"
    cache_path = Path(cache_dir) if cache_dir else None
    train_tokens = _tokens_with_cache(
        tok, "train", train_path, cache_path, text_kwargs, token_dtype, mmap_tokens
    )
    eval_tokens = (
        _tokens_with_cache(
            tok, "val", val_path, cache_path, text_kwargs, token_dtype, mmap_tokens
        )
        if val_path else None
    )
    return train_tokens, eval_tokens


# ------------------------------------------------------------- resume（续训）

def _apply_resume(
    cfg: TrainerConfig,
    payload: dict,
    resume_path: str | Path,
    steps_override: int | None,
    allow_data_change: bool,
    new_fingerprint: dict | None,
    horizon_steps: int | None = None,
) -> None:
    """把断点续训语义写入 ``cfg``。

    规则（与用户确认的口径一致）：
    - 模型架构与 LoRA 结构以 checkpoint 固化配置为准；
    - ``--steps N`` 表示在断点基础上**新增 N 步**，不传则沿用原配置剩余步数；
    - LR **延续原 warmup-cosine**（不重开 warmup，地平线取 checkpoint 原值），
      保证 LR 轨迹连续、不出现回弹；如需重新拉伸衰减用 ``horizon_steps``；
    - 数据内容 / ``seq_len`` / ``batch_size`` 与 checkpoint 不一致时**直接报错**，
      需显式 ``allow_data_change`` 放行（用于「换数据继续训练」）。
    """
    ckpt_cfg = payload.get("config") or {}
    ckpt_optim = ckpt_cfg.get("optim", {})
    ckpt_data = ckpt_cfg.get("data", {})

    # 1) 架构以 checkpoint 为准（权重才能对上；参数量在 load 时二次校验）
    cfg.model = CometSparkConfig.from_dict(payload["model_config"])
    lora_cfg = ckpt_cfg.get("lora") or payload.get("lora_config")
    if lora_cfg:
        from verse_nn.lora import LoRAConfig

        cfg.lora = LoRAConfig.from_dict(lora_cfg)

    # 2) 严格一致性校验
    problems: list[str] = []
    old_fp = payload.get("data_fingerprint")
    if old_fp and new_fingerprint and old_fp.get("sha256") != new_fingerprint.get("sha256"):
        problems.append(
            f"训练数据内容变化: {old_fp.get('sha256')}(n={old_fp.get('n')}) -> "
            f"{new_fingerprint.get('sha256')}(n={new_fingerprint.get('n')})"
        )
    for key in ("seq_len", "batch_size"):
        old, new = ckpt_data.get(key), getattr(cfg.data, key)
        if old is not None and old != new:
            problems.append(f"data.{key} 不一致: {old} -> {new}")
    if problems and not allow_data_change:
        raise RuntimeError(
            "断点续训一致性校验失败（确需换数据/改形状请显式放行 --allow-data-change）:\n"
            "  - " + "\n  - ".join(problems)
        )
    if problems:
        logger.warning("续训放行数据/形状变化:\n  - %s", "\n  - ".join(problems))

    # 3) 步数：--steps = 新增步数；默认 = 原配置剩余步数
    ckpt_step = int(payload.get("global_step", 0))
    original_total = int(ckpt_cfg.get("steps") or cfg.steps)
    additional = steps_override if (steps_override and steps_override > 0) \
        else max(original_total - ckpt_step, 0)
    if additional <= 0:
        logger.warning(
            "断点 step %d 已达/超过原配置总步数 %d，且未指定 --steps：本次无新增步数",
            ckpt_step, original_total,
        )
    cfg.steps = ckpt_step + additional

    # 4) LR：延续原 warmup-cosine（不重开 warmup，地平线取 ckpt 原值）
    cfg.optim.warmup_steps = int(ckpt_optim.get("warmup_steps", cfg.optim.warmup_steps))
    cfg.optim.max_steps = int(horizon_steps) if (horizon_steps and horizon_steps > 0) \
        else int(ckpt_optim.get("max_steps", cfg.optim.max_steps))
    if cfg.steps > cfg.optim.max_steps:
        logger.warning(
            "新增后总步数(%d)超出 cosine 地平线(%d)，超出部分维持 min_lr；"
            "如需重新衰减请显式传 --horizon-steps",
            cfg.steps, cfg.optim.max_steps,
        )
    saved_lr = ckpt_optim.get("lr")
    if saved_lr is not None and abs(float(saved_lr) - cfg.optim.lr) > 1e-12:
        logger.warning("续训学习率被覆盖: %s -> %s（优化器动量状态仍完整保留）",
                       saved_lr, cfg.optim.lr)

    logger.info(
        "断点续训：step %d -> %d（新增 %d 步）| warmup %d | cosine 地平线 %d | 起点 %s",
        ckpt_step, cfg.steps, additional,
        cfg.optim.warmup_steps, cfg.optim.max_steps, resume_path,
    )


def _resumed_trainer(
    cfg: TrainerConfig,
    payload: dict,
    resume_path: str | Path,
    steps_override: int | None,
    allow_data_change: bool,
    fingerprint: dict | None,
    stage: str,
    event_bus: EventBus | None,
    horizon_steps: int | None = None,
) -> Trainer:
    """应用续训语义并构建已恢复完整训练状态的 Trainer。"""
    _apply_resume(cfg, payload, resume_path, steps_override, allow_data_change,
                  fingerprint, horizon_steps)
    cfg.validate()
    cfg.stage = stage
    trainer = Trainer(cfg, event_bus=event_bus)
    trainer.load_checkpoint(resume_path)
    logger.info(
        "%s 断点续训：%.2fM 参数，从 step %d 继续到 %d",
        MODEL_NAME, trainer.model.num_parameters() / 1e6,
        trainer.global_step, cfg.steps,
    )
    return trainer


# ------------------------------------------------------------------ train

def _resolve_data_paths(
    root: Path, data_path: str | Path | None, val_path: str | Path | None
) -> tuple[Path, Path | None]:
    """解析训练/验证数据路径：显式优先，否则用 ``data/`` 下的默认文件。"""
    train_path = Path(data_path) if data_path else root / "data" / "train.jsonl"
    if not train_path.exists():
        raise FileNotFoundError(f"训练数据不存在: {train_path}")
    if val_path:
        eval_path: Path | None = Path(val_path)
        if not eval_path.exists():
            raise FileNotFoundError(f"验证数据不存在: {eval_path}")
    else:
        default_val = root / "data" / "val.jsonl"
        eval_path = default_val if default_val.exists() else None
    return train_path, eval_path


def train(
    config_path: str | Path,
    root: Path,
    event_bus: EventBus | None = None,
    steps_override: int | None = None,
    threads_override: int | None = None,
    device_override: str | None = None,
    text_kwargs: dict | None = None,
    data_path: str | Path | None = None,
    val_path: str | Path | None = None,
    token_dtype: str = "int32",
    mmap_tokens: bool = False,
    resume_from: str | Path | None = None,
    allow_data_change: bool = False,
    horizon_steps: int | None = None,
) -> dict:
    """按 yaml 配置训练 CometSpark 模型，返回训练摘要。

    Args:
        config_path: TrainerConfig 兼容的 yaml 配置。
        root: 项目根目录（数据/checkpoint/配置均相对该目录解析）。
        event_bus: 可选事件总线，注入后可通过订阅 ``train/step_end`` 等
            事件驱动外部进度展示（CLI 用 rich 进度条）。
        steps_override: 覆盖训练步数；**断点续训时为「新增步数」**。
        threads_override: 覆盖配置中的 CPU 线程数。
        device_override: 覆盖配置中的设备（cpu/cuda/mps）。
        text_kwargs: 文本字段抽取参数（``text_column``/``text_columns``/
            ``text_template``/``delimiter``），透传给 :func:`prepare_tokens`。
        data_path: 训练数据路径（默认 ``data/train.jsonl``）；支持
            jsonl/json/txt/md/csv/tsv/parquet 与预分词 npy/bin/tokens。
        val_path: 验证数据路径（默认 ``data/val.jsonl``）。
        token_dtype: 预分词 raw 二进制元素类型。
        mmap_tokens: ``.npy`` 预分词文件是否内存映射加载。
        resume_from: 断点续训的 checkpoint（恢复权重+优化器+RNG+监控进度）。
            架构取该 checkpoint，``steps_override`` 变为新增步数。
        allow_data_change: 续训时数据内容/``seq_len``/``batch_size`` 与
            checkpoint 不一致需置 True 才放行（用于换数据继续训练）。

    checkpoint 保存为 ``checkpoints/{MODEL_NAME}_pretrain_final.pt``，
    同时把模型配置固化到 ``config/{MODEL_NAME}_resolved.yaml``。
    """
    cfg = TrainerConfig.load(config_path)
    cfg.model_name = "cometspark"
    cfg.ckpt_dir = str((root / cfg.ckpt_dir).resolve())
    if threads_override is not None and threads_override > 0:
        cfg.num_threads = threads_override
    if device_override:
        cfg.device = device_override

    tok = load_tokenizer()
    train_path, eval_path = _resolve_data_paths(root, data_path, val_path)
    train_tokens, eval_tokens = prepare_tokens(
        tok, train_path, eval_path,
        text_kwargs=text_kwargs, token_dtype=token_dtype, mmap_tokens=mmap_tokens,
    )

    if resume_from is not None:
        payload = _load_payload(resume_from)
        cfg.stage = "pretrain"
        trainer = _resumed_trainer(
            cfg, payload, resume_from, steps_override, allow_data_change,
            fingerprint_source(train_tokens), "pretrain", event_bus, horizon_steps,
        )
    else:
        if steps_override is not None and steps_override > 0:
            cfg.steps = steps_override
            cfg.optim.max_steps = steps_override
        cfg.validate()
        cfg.stage = "pretrain"
        trainer = Trainer(cfg, event_bus=event_bus)
        logger.info("%s 开始训练：%.2fM 参数，device=%s",
                    MODEL_NAME, trainer.model.num_parameters() / 1e6, trainer.device)

    if eval_tokens is not None and len(eval_tokens) < cfg.data.seq_len + 1:
        logger.warning(
            "验证集 tokens(%d) 不足 seq_len+1(%d)，本次训练跳过周期评估",
            len(eval_tokens), cfg.data.seq_len + 1,
        )
        eval_tokens = None

    model_params = trainer.model.num_parameters() / 1e6
    summary = trainer.train(train_tokens, eval_tokens)

    # 保存 final 并清理中间 checkpoint（只保留本阶段 final）
    ckpt_path = Path(cfg.ckpt_dir) / f"{MODEL_NAME}_pretrain_final.pt"
    trainer.finalize("pretrain", path=ckpt_path)

    # 固化本次训练使用的完整配置，便于复现与版本追溯
    resolved_path = root / "config" / f"{MODEL_NAME}_resolved.yaml"
    cfg.save(resolved_path)

    summary["stage"] = "pretrain"
    summary["checkpoint"] = str(ckpt_path)
    summary["resolved_config"] = str(resolved_path)
    summary["model"] = MODEL_NAME
    summary["model_params_m"] = round(model_params, 2)
    logger.info("训练完成: %s", json.dumps(summary, ensure_ascii=False))
    return summary


# --------------------------------------------------------------- generate

def load_for_inference(ckpt_path: str | Path, device: str = "cpu"):
    """从 checkpoint 恢复模型与分词器，返回 (model, tokenizer, config)。

    加载后校验参数量与 checkpoint 元数据一致，避免架构不匹配导致的
    「参数量减半/权重对不上」被静默接受。

    注意：final checkpoint 默认保留优化器状态（数 GB），加载完权重后会
    立即释放 ``model_state``/``optimizer_state`` 等大张量，避免推理进程
    额外驻留一份优化器状态。
    """
    payload = torch.load(ckpt_path, map_location=device, weights_only=False)
    if "model_state" not in payload:
        raise ValueError(
            f"{ckpt_path} 不含 model_state（可能是 LoRA 适配器）。"
            "请使用其 *_merged.pt 完整模型导出进行推理。"
        )
    model_cfg = CometSparkConfig.from_dict(payload["model_config"])
    expected = payload.get("param_count")
    model = build_model("cometspark", model_cfg).to(torch.device(device))
    model.load_state_dict(payload["model_state"])
    # 权重已拷入模型，尽快释放 payload 中的大张量（尤其优化器状态）
    payload.pop("model_state", None)
    payload.pop("optimizer_state", None)
    payload.pop("scaler_state", None)
    payload = None
    _check_param_count(model, expected, ckpt_path)
    model.eval()
    tok = load_tokenizer()
    return model, tok, model_cfg


@torch.no_grad()
def generate(prompt: str, ckpt_path: str | Path, max_new_tokens: int = 24,
             temperature: float = 0.7, top_k: int = 40, top_p: float = 0.9,
             repetition_penalty: float = 1.1, no_repeat_ngram: int = 3,
             greedy: bool = False, chat: bool = False) -> str:
    """从 checkpoint 加载模型并续写 prompt。

    Args:
        chat: True 时用 ChatML 模板包裹 prompt（SFT 后的模型按对话格式
            训练，推理需保持一致格式才能触发回答行为）。
        top_p / repetition_penalty / no_repeat_ngram: 抑制复读与乱码的
            采样控制；``greedy=True`` 时忽略 temperature/top_k/top_p。

    停止符同时包含 ``<|endoftext|>`` 与 ``<|im_end|>``，避免对话模型
    在回答结束后继续输出模板标记（表现为乱码）。
    """
    model, tok, model_cfg = load_for_inference(ckpt_path)
    eos_id = load_endoftext_id(tok)
    stop_ids = {tid for tid in (
        tok.special_tokens.get(ENDOFTEXT),
        tok.special_tokens.get("<|im_end|>"),
    ) if tid is not None}
    if chat:
        from verse_tokenizer import ChatTemplate

        template = ChatTemplate.from_pretrained(getattr(tok, "_tokenizer_dir", "."))
        im_end = tok.special_tokens.get("<|im_end|>")
        if im_end is not None:
            eos_id = im_end  # ChatML 回答以 <|im_end|> 结束
        render_kwargs = {}
        # tokenizer 无 <|bos|> 特殊 token 时传空串，避免 BPE 兜底产生越界 id
        if "<|bos|>" not in tok.special_tokens:
            render_kwargs["bos_token"] = ""
        prompt = template.render(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True, **render_kwargs,
        )
    ids = tok.encode(prompt)
    if not ids:
        raise ValueError("prompt 分词结果为空")
    input_ids = torch.tensor([ids], dtype=torch.long, device=model.device)
    out = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        no_repeat_ngram=no_repeat_ngram,
        greedy=greedy,
        eos_token_id=eos_id,
        stop_token_ids=stop_ids,
    )
    return tok.decode(out[0].tolist(), skip_special_tokens=True)


# -------------------------------------------------- finetune / sft（后训练）

def _load_payload(ckpt_path: str | Path) -> dict:
    payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if "model_state" not in payload or "model_config" not in payload:
        raise ValueError(f"{ckpt_path} 不是有效的 CometSpark checkpoint")
    return payload


def _check_param_count(model, expected: int | None, path: str | Path) -> None:
    """校验模型参数量与 checkpoint 元数据一致（旧 ckpt 无元数据则跳过）。"""
    if expected is None:
        return
    unique, total = count_parameters(model)
    if unique != expected:
        raise RuntimeError(
            f"checkpoint 参数量不匹配: {path}\n"
            f"  期望(唯一) {expected:,}，实际(唯一) {unique:,}，"
            f"state_dict 合计 {total:,}\n"
            "  模型架构与 checkpoint 不一致，请使用 checkpoint 自带的 "
            "model_config（不要用 --config 的 model 段覆盖）。"
        )


def _verify_payload_params(model, payload: dict, path: str | Path) -> None:
    """校验模型参数量与 checkpoint 元数据一致（旧 ckpt 无元数据则跳过）。"""
    _check_param_count(model, payload.get("param_count"), path)


def _load_base_weights(trainer: Trainer, payload: dict, ckpt_path: str | Path) -> None:
    """把 checkpoint 权重载入未应用 LoRA 的 Trainer 模型（LoRA 之后再加）。"""
    try:
        trainer.model.load_state_dict(payload["model_state"], strict=True)
    except RuntimeError as e:
        raise RuntimeError(
            f"checkpoint 权重与当前架构不匹配（{e}）。"
            "提示：若该 checkpoint 来自 LoRA 训练，请先使用其 merged 导出版本"
        ) from e
    _verify_payload_params(trainer.model, payload, ckpt_path)


def _maybe_apply_lora(trainer: Trainer, lora_rank: int | None, lora_alpha: float) -> None:
    """按需对已加载权重的模型应用 LoRA 并重建优化器（只训适配器）。"""
    if not lora_rank:
        return
    from verse_nn.lora import LoRAConfig, apply_lora, mark_only_lora_trainable
    from verse_trainer.optim import build_optimizer

    lora_cfg = LoRAConfig(r=lora_rank, lora_alpha=lora_alpha)
    apply_lora(trainer.model, lora_cfg)
    mark_only_lora_trainable(trainer.model)
    trainer.config.lora = lora_cfg
    trainer.optimizer = build_optimizer(trainer.model, trainer.config.optim)


def _export_merged_checkpoint(model, model_config, global_step: int, stage: str) -> dict:
    """LoRA 合并后的完整模型导出（键名剥离 ``.base.``，可被 generate 加载）。"""
    from verse_nn.lora import merge_lora

    merge_lora(model)
    state = {
        k.replace(".base.", "."): v.detach().cpu()
        for k, v in model.state_dict().items()
        if "lora_" not in k
    }
    unique, total = _state_param_counts(state, getattr(model_config, "tie_weights", False))
    return {
        "global_step": global_step,
        "model_state": state,
        "model_config": model_config.to_dict(),
        "model_name": "cometspark",
        "lora_config": None,
        "merged_from_lora": True,
        "stage": stage,
        "param_count": unique,
        "state_numel": total,
        "tie_weights": getattr(model_config, "tie_weights", None),
    }


def _state_param_counts(state: dict, tie_weights: bool) -> tuple[int, int]:
    """从 state_dict 计算 (unique, total)。绑定权重在 state 中占两个键，需去重。"""
    total = sum(v.numel() for v in state.values())
    unique = total
    if tie_weights and "lm_head.weight" in state and "token_emb.weight" in state:
        unique -= state["lm_head.weight"].numel()
    return unique, total


def finetune(
    config_path: str | Path,
    root: Path,
    ckpt_path: str | Path,
    event_bus: EventBus | None = None,
    steps_override: int | None = None,
    threads_override: int | None = None,
    lr_override: float | None = None,
    device_override: str | None = None,
    lora_rank: int | None = None,
    lora_alpha: float = 16.0,
    text_kwargs: dict | None = None,
    data_path: str | Path | None = None,
    val_path: str | Path | None = None,
    token_dtype: str = "int32",
    mmap_tokens: bool = False,
    resume_from: str | Path | None = None,
    allow_data_change: bool = False,
    horizon_steps: int | None = None,
) -> dict:
    """微调（后训练·继续训练）：从 checkpoint 出发，在文本流上继续
    next-token 训练。支持全参微调（默认）与 LoRA（``lora_rank`` 给定时）。

    模型架构以 checkpoint 内固化的配置为准（保证权重可加载），训练超参
    取自 yaml 配置并可被参数覆盖。数据支持 jsonl/json/txt/md/csv/tsv/
    parquet 与预分词 npy/bin/tokens。

    ``resume_from`` 给定时为断点续训：忽略 ``ckpt_path`` 基座，直接恢复该
    checkpoint 的权重+优化器+调度进度+数据指纹（``steps_override`` 变为新增
    步数；换数据需 ``allow_data_change``）。产物：
    - 全参：``checkpoints/{MODEL_NAME}_finetune_final.pt``
    - LoRA：``checkpoints/{MODEL_NAME}_finetune_adapter.pt``（适配器）+
      ``checkpoints/{MODEL_NAME}_finetune_merged.pt``（合并后完整模型）
    """
    cfg = TrainerConfig.load(config_path)
    cfg.model_name = "cometspark"
    cfg.ckpt_dir = str((root / cfg.ckpt_dir).resolve())
    if threads_override is not None and threads_override > 0:
        cfg.num_threads = threads_override
    if device_override:
        cfg.device = device_override
    if lr_override is not None and lr_override > 0:
        cfg.optim.lr = lr_override

    tok = load_tokenizer()
    train_path, eval_path = _resolve_data_paths(root, data_path, val_path)
    train_tokens, eval_tokens = prepare_tokens(
        tok, train_path, eval_path,
        text_kwargs=text_kwargs, token_dtype=token_dtype, mmap_tokens=mmap_tokens,
    )

    if resume_from is not None:
        payload = _load_payload(resume_from)
        trainer = _resumed_trainer(
            cfg, payload, resume_from, steps_override, allow_data_change,
            fingerprint_source(train_tokens), "finetune", event_bus, horizon_steps,
        )
        lora_rank = lora_rank or (cfg.lora.r if cfg.lora is not None else None)
        mode = f"LoRA(r={lora_rank})" if lora_rank else "full"
        ckpt_used = resume_from
    else:
        payload = _load_payload(ckpt_path)
        cfg.model = CometSparkConfig.from_dict(payload["model_config"])
        if steps_override is not None and steps_override > 0:
            cfg.steps = steps_override
            cfg.optim.max_steps = steps_override
        cfg.validate()
        cfg.stage = "finetune"
        trainer = Trainer(cfg, event_bus=event_bus)
        _load_base_weights(trainer, payload, ckpt_path)
        _maybe_apply_lora(trainer, lora_rank, lora_alpha)
        mode = f"LoRA(r={lora_rank})" if lora_rank else "full"
        ckpt_used = ckpt_path

    if eval_tokens is not None and len(eval_tokens) < cfg.data.seq_len + 1:
        eval_tokens = None

    params = trainer.model.num_parameters() / 1e6
    trainable = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad) / 1e6
    logger.info("%s 微调（%s）：%.2fM 参数（可训练 %.2fM），起点 %s",
                MODEL_NAME, mode, params, trainable, ckpt_used)

    summary = trainer.train(train_tokens, eval_tokens)
    summary["mode"] = mode
    summary["stage"] = "finetune"
    summary["base_checkpoint"] = str(ckpt_used)

    raw_model = getattr(trainer.model, "_orig_mod", trainer.model)
    if lora_rank:
        adapter_path = Path(cfg.ckpt_dir) / f"{MODEL_NAME}_finetune_adapter.pt"
        trainer.save_adapter(str(adapter_path))
        merged_path = Path(cfg.ckpt_dir) / f"{MODEL_NAME}_finetune_merged.pt"
        torch.save(_export_merged_checkpoint(raw_model, cfg.model, trainer.global_step, "finetune"),
                   merged_path)
        trainer.cleanup_intermediate(keep={adapter_path, merged_path})
        summary["adapter"] = str(adapter_path)
        summary["checkpoint"] = str(merged_path)
    else:
        ckpt_out = Path(cfg.ckpt_dir) / f"{MODEL_NAME}_finetune_final.pt"
        trainer.finalize("finetune", path=ckpt_out)
        summary["checkpoint"] = str(ckpt_out)

    resolved_path = root / "config" / f"{MODEL_NAME}_finetune_resolved.yaml"
    cfg.save(resolved_path)
    summary["model"] = MODEL_NAME
    summary["model_params_m"] = round(params, 2)
    summary["resolved_config"] = str(resolved_path)
    logger.info("微调完成: %s", json.dumps(summary, ensure_ascii=False))
    return summary


def sft(
    config_path: str | Path,
    root: Path,
    ckpt_path: str | Path,
    data_path: str | Path,
    val_path: str | Path | None = None,
    event_bus: EventBus | None = None,
    steps_override: int | None = None,
    threads_override: int | None = None,
    lr_override: float | None = None,
    device_override: str | None = None,
    lora_rank: int | None = None,
    lora_alpha: float = 16.0,
    batch_size: int = 4,
    max_seq_len: int | None = None,
    train_on_last_turns: int = 1,
    resume_from: str | Path | None = None,
    allow_data_change: bool = False,
    horizon_steps: int | None = None,
) -> dict:
    """SFT 指令微调（后训练）：对话数据 + answer-only loss 掩码。

    数据格式：JSON/JSONL/CSV/TSV/Parquet，``[{"messages": [...]}, ...]``。
    模型架构以 checkpoint 固化配置为准；loss 只计 assistant 回答（含
    ``<|im_end|>``）。产物与 :func:`finetune` 一致（LoRA 时额外导出 merged）。

    ``resume_from`` 给定时为断点续训（恢复优化器/RNG/监控进度；``--steps``
    变为新增步数；换数据需 ``allow_data_change``）。
    """
    from verse_trainer.sft import SFTConfig, SFTDataset

    cfg = TrainerConfig.load(config_path)
    cfg.model_name = "cometspark"
    cfg.ckpt_dir = str((root / cfg.ckpt_dir).resolve())
    if threads_override is not None and threads_override > 0:
        cfg.num_threads = threads_override
    if device_override:
        cfg.device = device_override
    if lr_override is not None and lr_override > 0:
        cfg.optim.lr = lr_override

    # 架构取续训 ckpt（若续训）否则取基座 ckpt
    source_ckpt = resume_from or ckpt_path
    payload = _load_payload(source_ckpt)
    cfg.model = CometSparkConfig.from_dict(payload["model_config"])

    tok = load_tokenizer()
    # tokenizer 缺少 <|bos|>/<|eos|>/<|pad|> 特殊 token 时传空串，
    # 避免模板渲染后 BPE 兜底编码产生超出 vocab 的 id（"<|bos|>" -> "bos_token"）
    encode_kwargs = {
        f"{name[2:-2]}_token": ""
        for name in ("<|bos|>", "<|eos|>", "<|pad|>")
        if name not in tok.special_tokens
    }
    sft_cfg = SFTConfig(
        max_seq_len=min(max_seq_len or cfg.model.max_seq_len, cfg.model.max_seq_len),
        train_on_last_turns=train_on_last_turns,
        batch_size=batch_size,
        seed=cfg.seed,
    )
    train_ds = SFTDataset.from_file(data_path, tok, None, sft_cfg, encode_kwargs)
    eval_ds = SFTDataset.from_file(val_path, tok, None, sft_cfg) if val_path else None

    if resume_from is not None:
        trainer = _resumed_trainer(
            cfg, payload, resume_from, steps_override, allow_data_change,
            fingerprint_source(train_ds), "sft", event_bus, horizon_steps,
        )
        lora_rank = lora_rank or (cfg.lora.r if cfg.lora is not None else None)
        ckpt_used = resume_from
    else:
        if steps_override is not None and steps_override > 0:
            cfg.steps = steps_override
            cfg.optim.max_steps = steps_override
        cfg.validate()
        cfg.stage = "sft"
        trainer = Trainer(cfg, event_bus=event_bus)
        _load_base_weights(trainer, payload, ckpt_path)
        _maybe_apply_lora(trainer, lora_rank, lora_alpha)
        ckpt_used = ckpt_path

    train_ds.device = trainer.device
    if eval_ds is not None:
        eval_ds.device = trainer.device

    mode = f"LoRA(r={lora_rank})" if lora_rank else "full"
    params = trainer.model.num_parameters() / 1e6
    trainable = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad) / 1e6
    logger.info("%s SFT（%s）：%.2fM 参数（可训练 %.2fM），样本 %d 条，起点 %s",
                MODEL_NAME, mode, params, trainable, len(train_ds), ckpt_used)
    summary = trainer.train(train_ds, eval_ds)
    summary["mode"] = mode
    summary["stage"] = "sft"
    summary["base_checkpoint"] = str(ckpt_used)

    raw_model = getattr(trainer.model, "_orig_mod", trainer.model)
    if lora_rank:
        adapter_path = Path(cfg.ckpt_dir) / f"{MODEL_NAME}_sft_adapter.pt"
        trainer.save_adapter(str(adapter_path))
        merged_path = Path(cfg.ckpt_dir) / f"{MODEL_NAME}_sft_merged.pt"
        torch.save(_export_merged_checkpoint(raw_model, cfg.model, trainer.global_step, "sft"),
                   merged_path)
        trainer.cleanup_intermediate(keep={adapter_path, merged_path})
        summary["adapter"] = str(adapter_path)
        summary["checkpoint"] = str(merged_path)
    else:
        ckpt_out = Path(cfg.ckpt_dir) / f"{MODEL_NAME}_sft_final.pt"
        trainer.finalize("sft", path=ckpt_out)
        summary["checkpoint"] = str(ckpt_out)

    resolved_path = root / "config" / f"{MODEL_NAME}_sft_resolved.yaml"
    cfg.save(resolved_path)
    summary["model"] = MODEL_NAME
    summary["model_params_m"] = round(params, 2)
    summary["resolved_config"] = str(resolved_path)
    logger.info("SFT 完成: %s", json.dumps(summary, ensure_ascii=False))
    return summary
