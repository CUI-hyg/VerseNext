"""评估入口：加载 best.pt → 生成示例文本。

5 条预设 prompt（中英混合 + 数字序列），用 greedy + top-k 生成，打印结果。
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np

from verse_torch import Tensor, no_grad

from model.config import CometSparkConfig, load_full_config
from model.model import CometSparkLM
from model.tokenizer import load_tokenizer
from src.utils import set_seed


# 5 条预设 prompt（与任务要求一致）
_DEFAULT_PROMPTS = [
    "床前明月光，",
    "白日依山尽，",
    "你好，",
    "1+1=",
    "春风",
]


def _resolve_path(base_dir: str, path_str: str) -> str:
    p = Path(path_str)
    if p.is_absolute():
        return str(p)
    return str((Path(base_dir) / p).resolve())


def _safe_encode(tok, text):
    """兼容 BPETokenizer / ByteTokenizer 的 encode 签名。"""
    try:
        return list(tok.encode(text, add_special_tokens=False))
    except TypeError:
        try:
            return list(tok.encode(text))
        except Exception:
            return []


def _safe_decode(tok, ids):
    """兼容不同 decode 签名。"""
    try:
        return tok.decode(list(ids))
    except TypeError:
        # ByteTokenizer.decode 有 strip_special 参数
        try:
            return tok.decode(list(ids), strip_special=True)
        except Exception:
            return tok.decode(list(ids))


def evaluate(
    config_path: str,
    base_dir: str = ".",
    prompts=None,
    max_new_tokens: int = 32,
    top_k: int = None,
    seed: int = 42,
) -> dict:
    """加载 best.pt 生成示例文本。

    Args:
        config_path: 配置文件路径
        base_dir: 相对路径基准
        prompts: 自定义 prompt 列表；None 用默认 5 条
        max_new_tokens: 每条 prompt 生成 token 数
        top_k: top-k 采样；None 表示 greedy
        seed: 随机种子
    Returns:
        dict 包含 results（list of {prompt, generated}）/ wall_clock
    """
    start_time = time.time()
    set_seed(seed)

    full_cfg = load_full_config(config_path)
    tok_cfg = full_cfg.get("tokenizer", {})
    ckpt_cfg = full_cfg.get("checkpoint", {})
    model_cfg = full_cfg.get("model", {})

    tok_kind = str(tok_cfg.get("kind", "byte"))
    save_dir = _resolve_path(base_dir, str(ckpt_cfg.get("save_dir", "checkpoints")))

    # 1. 加载 tokenizer
    tok_path = os.path.join(save_dir, "tokenizer.json")
    if not os.path.exists(tok_path):
        alt = _resolve_path(base_dir, "tokenizer.json")
        if os.path.exists(alt):
            tok_path = alt
    print(f"[evaluate] 加载 tokenizer ({tok_kind}) from {tok_path}", flush=True)
    tok = load_tokenizer(tok_path, kind=tok_kind)
    vocab_size = len(tok)

    # 2. 加载模型
    # 优先：完整模型 cometspark.pt（含 config）
    # 其次：best.pt（仅 state_dict）
    full_model_path = os.path.join(save_dir, "cometspark.pt")
    best_path = os.path.join(save_dir, "best.pt")

    if os.path.exists(full_model_path):
        print(f"[evaluate] 加载完整模型 {full_model_path}", flush=True)
        model = CometSparkLM.from_pretrained(full_model_path)
    elif os.path.exists(best_path):
        print(f"[evaluate] best.pt 存在但无完整模型，用 config 重建模型", flush=True)
        config_dict = dict(model_cfg)
        config_dict["vocab_size"] = vocab_size
        config = CometSparkConfig.from_dict(config_dict)
        model = CometSparkLM(config)
        # best.pt 内的 state_dict 由 Trainer.CheckpointManager 保存
        # 格式：{"step": int, "model_state_dict": dict, "val_loss": float, ...}
        import pickle
        with open(best_path, "rb") as f:
            payload = pickle.load(f)
        sd = payload.get("model_state_dict", payload)
        model.load_state_dict(sd, strict=False)
    else:
        raise FileNotFoundError(
            f"未找到模型文件：{full_model_path} 或 {best_path}，请先训练。"
        )

    # 3. 生成
    if prompts is None:
        prompts = list(_DEFAULT_PROMPTS)

    results = []
    print(f"[evaluate] 开始生成 {len(prompts)} 条 prompt，每条 {max_new_tokens} tokens", flush=True)
    with no_grad():
        model.eval()
        for prompt in prompts:
            ids = _safe_encode(tok, prompt)
            if not ids:
                # 编码失败：用空 prompt 也至少产生一个 token
                ids = []
            if not ids:
                # 极端情况：用 0 作为 BOS
                ids = [0]
            idx_np = np.asarray(ids, dtype=np.int64).reshape(1, -1)
            try:
                generated = model.generate(
                    idx_np,
                    max_new_tokens=int(max_new_tokens),
                    temperature=1.0,
                    top_k=top_k,
                )
                if isinstance(generated, Tensor):
                    gen_ids = generated.data.reshape(-1).tolist()
                else:
                    gen_ids = np.asarray(generated).reshape(-1).tolist()
            except Exception as e:
                print(f"[evaluate] 生成失败 prompt={prompt!r}: {e}", flush=True)
                gen_ids = list(ids)

            full_text = _safe_decode(tok, gen_ids)
            results.append({
                "prompt": prompt,
                "generated": full_text,
                "n_tokens": len(gen_ids),
            })
            print(f"  [prompt] {prompt}", flush=True)
            print(f"  [output] {full_text}", flush=True)
            print(f"  (tokens: {len(gen_ids)})", flush=True)
            print("", flush=True)

    wall_clock = time.time() - start_time
    print(f"[evaluate] 完成 wall_clock={wall_clock:.2f}s", flush=True)
    return {
        "results": results,
        "wall_clock": wall_clock,
        "vocab_size": vocab_size,
    }


__all__ = ["evaluate"]
