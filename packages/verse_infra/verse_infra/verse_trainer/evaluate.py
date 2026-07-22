"""评估入口：加载 checkpoint → 生成示例文本 + 可选打分（Part4K1 Task 6.2）。

从 ``data/demo/train/evaluate.py`` 迁入并升级：
- 与 ``verse_trainer.trainer._build_model`` / ``_load_tokenizer`` 共用构建逻辑
- 保留 5 条预设 prompt + ScoringEvaluator 打分模式
- 兼容 ``CometSparkLM.from_pretrained`` 与 ``best.pt`` state_dict 两种加载方式
"""

from __future__ import annotations

import os
import pickle
import time
from pathlib import Path

import numpy as np

from verse_torch import Tensor, no_grad
from verse_torch.scoring import ScoringEvaluator

from .trainer import _resolve_path, _load_full_config, _build_model, _load_tokenizer


# 5 条预设 prompt（与 data/demo 保持一致）
_DEFAULT_PROMPTS = [
    "床前明月光，",
    "白日依山尽，",
    "你好，",
    "1+1=",
    "春风",
]


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
        try:
            return tok.decode(list(ids), strip_special=True)
        except Exception:
            return tok.decode(list(ids))


def _get_eos_id(tok):
    """从 tokenizer 获取 eos_id（兼容 ByteTokenizer / BPETokenizer / CharTokenizer）。"""
    eos_id = getattr(tok, "eos_id", None)
    if eos_id is not None:
        return int(eos_id)
    vocab = getattr(tok, "vocab", None)
    if isinstance(vocab, dict):
        eos = vocab.get("<eos>")
        if eos is not None:
            return int(eos)
    return None


def _safe_decode_with_prompt(tok, prompt_text, prompt_ids, gen_ids):
    """分别 decode prompt 和生成部分再拼接，避免边界乱码。

    - prompt 部分直接用原始文本（不 decode，避免 prompt 末尾字节与生成
      首字节拼接产生非法 UTF-8 序列）；
    - 生成部分用 ``tokenizer.decode``（已内置字节对齐检查）。
    """
    n_prompt = len(prompt_ids)
    if n_prompt >= len(gen_ids):
        generated_ids = []
    else:
        generated_ids = list(gen_ids[n_prompt:])
    decoded_generated = _safe_decode(tok, generated_ids)
    return prompt_text + decoded_generated


def evaluate(
    config_path: str,
    base_dir: str = ".",
    prompts=None,
    max_new_tokens=None,
    temperature: float = 1.0,
    top_k=None,
    top_p=None,
    seed: int = 42,
    score: bool = False,
    references_file: str = None,
    checkpoint: str = None,
) -> dict:
    """加载 checkpoint 生成示例文本。

    Part4K2 Task 3 升级：``max_new_tokens`` 默认改为 ``None``，即不限制输出
    token 数，让模型按 EOS 自然停止（安全上限 100K 防无限循环）。用户在 CLI
    显式指定 ``--max-tokens`` 时按值限制（兼容旧行为）。

    Args:
        config_path: 配置文件路径
        base_dir: 相对路径基准
        prompts: 自定义 prompt 列表；None 用默认 5 条
        max_new_tokens: 每条 prompt 生成 token 数；``None`` 表示不限
            （生成到 EOS 自然停止，安全上限 100K）。默认 ``None``。
        temperature: 采样温度；1.0 等价 greedy，>1 增加随机性，<1 收敛
        top_k: top-k 采样；None 表示 greedy
        top_p: nucleus sampling 阈值 (0,1)；None 表示不限制
        seed: 随机种子
        score: 是否对生成结果打分（需提供 references_file）
        references_file: 参考答案文件路径，每行一个 reference
        checkpoint: 指定 checkpoint 文件路径；None 时自动查找 best.pt / cometspark.pt
    Returns:
        dict 包含 results（list of {prompt, generated}）/ wall_clock /
        scores（score=True 时含 5 个指标的均值）
    """
    start_time = time.time()
    try:
        from verse_torch import set_seed
        set_seed(seed)
    except Exception:
        import random
        random.seed(seed)
        np.random.seed(seed)

    full_cfg = _load_full_config(config_path)
    tok_cfg = full_cfg.get("tokenizer", {})
    ckpt_cfg = full_cfg.get("checkpoint", {})
    model_cfg = full_cfg.get("model", {})

    save_dir = _resolve_path(base_dir, str(ckpt_cfg.get("save_dir", "checkpoints")))

    # 1. 加载 tokenizer
    tok = _load_tokenizer(tok_cfg, base_dir, save_dir)
    vocab_size = len(tok)

    # 2. 加载模型
    # 优先：用户指定 checkpoint 路径
    # 其次：完整模型 cometspark.pt（含 config）
    # 再次：best.pt（仅 state_dict）
    if checkpoint is not None and os.path.exists(checkpoint):
        ckpt_path = checkpoint
        print(f"[evaluate] 加载指定 checkpoint {ckpt_path}", flush=True)
        model, config = _build_model(model_cfg, vocab_size)
        with open(ckpt_path, "rb") as f:
            payload = pickle.load(f)
        sd = payload.get("model_state_dict", payload)
        if hasattr(model, "load_state_dict"):
            model.load_state_dict(sd, strict=False)
    else:
        full_model_path = os.path.join(save_dir, "cometspark.pt")
        best_path = os.path.join(save_dir, "best.pt")
        if os.path.exists(full_model_path) and hasattr(_import_cometspark_lm(), "from_pretrained"):
            try:
                from model.model import CometSparkLM
                print(f"[evaluate] 加载完整模型 {full_model_path}", flush=True)
                model = CometSparkLM.from_pretrained(full_model_path)
                config = model.config
            except Exception:
                model, config = _build_model(model_cfg, vocab_size)
                if os.path.exists(best_path):
                    with open(best_path, "rb") as f:
                        payload = pickle.load(f)
                    sd = payload.get("model_state_dict", payload)
                    if hasattr(model, "load_state_dict"):
                        model.load_state_dict(sd, strict=False)
        elif os.path.exists(best_path):
            print(f"[evaluate] best.pt 存在，用 config 重建模型", flush=True)
            model, config = _build_model(model_cfg, vocab_size)
            with open(best_path, "rb") as f:
                payload = pickle.load(f)
            sd = payload.get("model_state_dict", payload)
            if hasattr(model, "load_state_dict"):
                model.load_state_dict(sd, strict=False)
        else:
            raise FileNotFoundError(
                f"未找到模型文件：{full_model_path} 或 {best_path}，请先训练。"
            )

    # 3. 生成
    if prompts is None:
        prompts = list(_DEFAULT_PROMPTS)

    results = []
    token_limit_desc = "不限（EOS 自然停止）" if max_new_tokens is None else f"{max_new_tokens} tokens"
    print(
        f"[evaluate] 开始生成 {len(prompts)} 条 prompt，每条 {token_limit_desc} "
        f"(temperature={temperature}, top_k={top_k}, top_p={top_p})",
        flush=True,
    )
    eos_id = _get_eos_id(tok)
    with no_grad():
        if hasattr(model, "eval"):
            model.eval()
        for i, prompt in enumerate(prompts):
            ids = _safe_encode(tok, prompt)
            if not ids:
                ids = [0]
            idx_np = np.asarray(ids, dtype=np.int64).reshape(1, -1)
            try:
                # Part4K2 Task 3：max_new_tokens=None 时让模型 EOS 自然停止
                gen_kwargs = dict(
                    temperature=float(temperature),
                    top_k=top_k,
                    eos_id=eos_id,
                )
                if max_new_tokens is not None:
                    gen_kwargs["max_new_tokens"] = int(max_new_tokens)
                try:
                    generated = model.generate(idx_np, top_p=top_p, **gen_kwargs)
                except TypeError:
                    # 旧模型 generate 不接受 top_p 参数
                    generated = model.generate(idx_np, **gen_kwargs)
                if isinstance(generated, Tensor):
                    gen_ids = generated.data.reshape(-1).tolist()
                else:
                    gen_ids = np.asarray(generated).reshape(-1).tolist()
            except Exception as e:
                print(f"[evaluate] 生成失败 prompt={prompt!r}: {e}", flush=True)
                gen_ids = list(ids)

            full_text = _safe_decode_with_prompt(tok, prompt, ids, gen_ids)
            results.append({
                "prompt": prompt,
                "generated": full_text,
                "n_tokens": len(gen_ids),
            })
            print("-" * 60, flush=True)
            print(f"  [{i+1}/{len(prompts)}] [prompt] {prompt}", flush=True)
            print(f"  [output] {full_text}", flush=True)
            print(f"  (tokens: {len(gen_ids)})", flush=True)
        print("-" * 60, flush=True)

    # 打分模式
    scores = None
    if score:
        scores = _run_scoring(results, prompts, references_file, base_dir)

    wall_clock = time.time() - start_time
    print(f"[evaluate] 完成 wall_clock={wall_clock:.2f}s", flush=True)
    result_dict = {
        "results": results,
        "wall_clock": wall_clock,
        "vocab_size": vocab_size,
    }
    if scores is not None:
        result_dict["scores"] = scores
    return result_dict


def _import_cometspark_lm():
    """延迟导入 CometSparkLM（避免 verse_nex 不可用时整个模块加载失败）。"""
    try:
        import sys as _sys
        demo_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(_sys.modules[__name__].__file__))),
            "data", "demo",
        )
        if demo_dir not in _sys.path:
            _sys.path.insert(0, demo_dir)
        from model.model import CometSparkLM
        return CometSparkLM
    except Exception:
        class _Dummy:
            @staticmethod
            def from_pretrained(*a, **k):
                raise ImportError("CometSparkLM 不可用")
        return _Dummy


def _run_scoring(results, prompts, references_file, base_dir):
    """加载 references 文件并计算 5 个指标。"""
    if references_file is None:
        print("[evaluate] --score 已启用但未提供 --references-file，跳过打分",
              flush=True)
        return None

    ref_path = references_file
    if not os.path.isabs(ref_path):
        ref_path = _resolve_path(base_dir, references_file)
    if not os.path.exists(ref_path):
        print(f"[evaluate] references 文件不存在：{ref_path}，跳过打分", flush=True)
        return None

    references = []
    with open(ref_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n").rstrip("\r")
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            references.append(line)

    if len(references) != len(results):
        print(
            f"[evaluate] references 数 ({len(references)}) 与 prompts 数 "
            f"({len(results)}) 不一致，跳过打分",
            flush=True,
        )
        return None

    predictions = [r["generated"] for r in results]
    evaluator = ScoringEvaluator()
    scores = evaluator.evaluate(predictions, references)
    report = evaluator.report(scores)
    print("[evaluate] 评分报告：", flush=True)
    print(report, flush=True)
    return scores


__all__ = ["evaluate"]
