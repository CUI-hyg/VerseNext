"""评估入口：加载 best.pt → 生成示例文本。

5 条预设 prompt（中英混合 + 数字序列），用 greedy + top-k 生成，打印结果。

Part3K2 Task 1.7: 集成 ScoringEvaluator 支持 ``--score`` 模式：
    - 加载 references 文件（每行一个参考答案，与 prompts 一一对应）
    - 对生成结果与参考答案计算 5 个指标（exact_match / prefix_accuracy /
      char_f1 / bleu / rouge_l）
    - 打印评分报告
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np

from verse_torch import Tensor, no_grad
from verse_torch.scoring import ScoringEvaluator

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


def _get_eos_id(tok):
    """从 tokenizer 获取 eos_id（兼容 ByteTokenizer / BPETokenizer / CharTokenizer）。

    - ByteTokenizer: ``tok.eos_id``
    - BPETokenizer / CharTokenizer: ``tok.vocab.get("<eos>")``
    """
    # ByteTokenizer 直接暴露 eos_id 属性
    eos_id = getattr(tok, "eos_id", None)
    if eos_id is not None:
        return int(eos_id)
    # BPETokenizer / CharTokenizer: 从 vocab 查 "<eos>"
    vocab = getattr(tok, "vocab", None)
    if isinstance(vocab, dict):
        eos = vocab.get("<eos>")
        if eos is not None:
            return int(eos)
    return None


def _safe_decode_with_prompt(tok, prompt_text, prompt_ids, gen_ids):
    """Task 5.4: 分别 decode prompt 和生成部分再拼接，避免边界乱码。

    - prompt 部分直接用原始文本（不 decode，避免 prompt 末尾字节与生成
      首字节拼接产生非法 UTF-8 序列）；
    - 生成部分用 ``tokenizer.decode``（已内置字节对齐检查）。

    Args:
        tok: tokenizer 实例
        prompt_text: 原始 prompt 字符串
        prompt_ids: prompt 编码后的 id 列表
        gen_ids: 完整生成结果（含 prompt + 生成）的 id 列表

    Returns:
        拼接后的完整文本 ``prompt_text + decoded_generated``
    """
    # 只取生成部分（去掉开头的 prompt_ids）
    n_prompt = len(prompt_ids)
    # 防御性：若 gen_ids 长度小于 prompt_ids，全部当作生成部分
    if n_prompt >= len(gen_ids):
        generated_ids = []
    else:
        generated_ids = list(gen_ids[n_prompt:])
    # prompt 用原始文本，生成部分用 tokenizer.decode
    decoded_generated = _safe_decode(tok, generated_ids)
    return prompt_text + decoded_generated


def evaluate(
    config_path: str,
    base_dir: str = ".",
    prompts=None,
    max_new_tokens: int = 32,
    temperature: float = 1.0,
    top_k: int = None,
    top_p: float = None,
    seed: int = 42,
    score: bool = False,
    references_file: str = None,
) -> dict:
    """加载 best.pt 生成示例文本。

    Args:
        config_path: 配置文件路径
        base_dir: 相对路径基准
        prompts: 自定义 prompt 列表；None 用默认 5 条
        max_new_tokens: 每条 prompt 生成 token 数
        temperature: 采样温度；1.0 等价 greedy，>1 增加随机性，<1 收敛
        top_k: top-k 采样；None 表示 greedy
        top_p: nucleus sampling 阈值 (0,1)；None 表示不限制。
            注意：CometSparkLM.generate 当前不支持 top_p，会自动降级为
            temperature+top_k 采样（Trainer.inference 已用 try/except 处理）。
        seed: 随机种子
        score: Part3K2 Task 1.7 是否对生成结果打分（需提供 references_file）
        references_file: 参考答案文件路径，每行一个 reference（与 prompts 一一对应）；
            仅当 score=True 时生效
    Returns:
        dict 包含 results（list of {prompt, generated}）/ wall_clock /
        scores（score=True 时含 5 个指标的均值）
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
    print(f"[evaluate] 开始生成 {len(prompts)} 条 prompt，每条 {max_new_tokens} tokens "
          f"(temperature={temperature}, top_k={top_k}, top_p={top_p})", flush=True)
    # Task 5.3: 获取 eos_id，传给 generate 以确保末尾追加 eos
    eos_id = _get_eos_id(tok)
    with no_grad():
        model.eval()
        for i, prompt in enumerate(prompts):
            ids = _safe_encode(tok, prompt)
            if not ids:
                # 编码失败：用空 prompt 也至少产生一个 token
                ids = []
            if not ids:
                # 极端情况：用 0 作为 BOS
                ids = [0]
            idx_np = np.asarray(ids, dtype=np.int64).reshape(1, -1)
            try:
                # Part3K2 Task 1.7: 兼容 top_p（CometSparkLM.generate 不支持时降级）
                try:
                    generated = model.generate(
                        idx_np,
                        max_new_tokens=int(max_new_tokens),
                        temperature=float(temperature),
                        top_k=top_k,
                        top_p=top_p,
                        eos_id=eos_id,
                    )
                except TypeError:
                    # 模型 generate 不支持 top_p 参数，降级调用
                    generated = model.generate(
                        idx_np,
                        max_new_tokens=int(max_new_tokens),
                        temperature=float(temperature),
                        top_k=top_k,
                        eos_id=eos_id,
                    )
                if isinstance(generated, Tensor):
                    gen_ids = generated.data.reshape(-1).tolist()
                else:
                    gen_ids = np.asarray(generated).reshape(-1).tolist()
            except Exception as e:
                print(f"[evaluate] 生成失败 prompt={prompt!r}: {e}", flush=True)
                gen_ids = list(ids)

            # Task 5.4: 分别 decode prompt 和生成部分再拼接，避免边界乱码
            # prompt 部分用原始文本（不 decode），生成部分用 tokenizer.decode
            full_text = _safe_decode_with_prompt(tok, prompt, ids, gen_ids)
            results.append({
                "prompt": prompt,
                "generated": full_text,
                "n_tokens": len(gen_ids),
            })
            # Task 7.4: 评估输出格式化（每个 prompt 用分隔线隔开）
            print("-" * 60, flush=True)
            print(f"  [{i+1}/{len(prompts)}] [prompt] {prompt}", flush=True)
            print(f"  [output] {full_text}", flush=True)
            print(f"  (tokens: {len(gen_ids)})", flush=True)
        print("-" * 60, flush=True)

    # Part3K2 Task 1.7: 打分模式（--score）
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


def _run_scoring(results, prompts, references_file, base_dir):
    """Part3K2 Task 1.7: 加载 references 文件并计算 5 个指标。

    Args:
        results: evaluate() 内生成的 results 列表，每项含 ``generated`` 字段
        prompts: prompt 列表（用于报错时定位）
        references_file: 参考答案文件路径（相对 base_dir 或绝对路径）
        base_dir: 用于解析相对路径

    Returns:
        dict: 5 个指标的均值 + per_sample + n_samples；失败时返回 None
    """
    if references_file is None:
        print("[evaluate] --score 已启用但未提供 --references-file，跳过打分",
              flush=True)
        return None

    # 解析 references 文件路径
    ref_path = references_file
    if not os.path.isabs(ref_path):
        ref_path = _resolve_path(base_dir, references_file)
    if not os.path.exists(ref_path):
        print(f"[evaluate] references 文件不存在：{ref_path}，跳过打分",
              flush=True)
        return None

    # 读取 references（每行一个，忽略空行与 # 注释）
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

    # 提取 predictions（generated 文本）
    predictions = [r["generated"] for r in results]

    # 用 ScoringEvaluator 计算 5 个指标
    evaluator = ScoringEvaluator()
    scores = evaluator.evaluate(predictions, references)
    report = evaluator.report(scores)
    print("[evaluate] 评分报告：", flush=True)
    print(report, flush=True)
    return scores


__all__ = ["evaluate"]
