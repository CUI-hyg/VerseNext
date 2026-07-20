"""CometSpark-v0.1 一键入口。

流程：
    1. set_seed + ensure_dir
    2. （可选）build_tokenizer
    3. （可选）train
    4. （可选）evaluate
    5. （可选）visualize

用法：
    python run.py                              # 全流程
    python run.py --skip-train                 # 仅 build + eval
    python run.py --skip-train --skip-eval     # 仅 build
    python run.py --config config/my.yml       # 自定义配置
"""

from __future__ import annotations

import argparse
import os
import sys
import time

# 把当前目录加入 sys.path，使 model/ src/ train/ 包可被 import
_DEMO_DIR = os.path.dirname(os.path.abspath(__file__))
if _DEMO_DIR not in sys.path:
    sys.path.insert(0, _DEMO_DIR)

# 限制 BLAS 线程数：4 核 CPU 上设为 4 避免过度并行
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")

from src.utils import set_seed, ensure_dir
from model.config import load_full_config
from model.tokenizer import build_tokenizer, load_tokenizer
from train.trainer import train as train_fn
from train.evaluate import evaluate as evaluate_fn
from train.visualize import visualize as visualize_fn


def _resolve(base_dir: str, path_str: str) -> str:
    p = os.path.join(base_dir, path_str) if not os.path.isabs(path_str) else path_str
    return os.path.abspath(p)


def stage_build_tokenizer(config_path: str, base_dir: str, force: bool = False) -> str:
    """构建并保存 tokenizer。"""
    print("=" * 70, flush=True)
    print("[stage 1/3] 构建 tokenizer", flush=True)
    print("=" * 70, flush=True)
    t0 = time.time()

    full_cfg = load_full_config(config_path)
    tok_cfg = full_cfg.get("tokenizer", {})
    ckpt_cfg = full_cfg.get("checkpoint", {})
    data_cfg = full_cfg.get("data", {})

    tok_kind = str(tok_cfg.get("kind", "byte"))
    vocab_size = int(tok_cfg.get("vocab_size", 259))
    save_dir = _resolve(base_dir, str(ckpt_cfg.get("save_dir", "checkpoints")))
    ensure_dir(save_dir)
    tok_path = os.path.join(save_dir, "tokenizer.json")

    if os.path.exists(tok_path) and not force:
        print(f"[build_tokenizer] 已存在 {tok_path}，跳过（--force 可覆盖）", flush=True)
        return tok_path

    if tok_kind == "byte":
        # ByteTokenizer 不需要 corpus
        print(f"[build_tokenizer] kind=byte, vocab_size=259 (固定)", flush=True)
        build_tokenizer(
            corpus_path="",
            vocab_size=259,
            save_path=tok_path,
            kind="byte",
        )
    else:
        # BPE 需要从训练语料构建
        train_path = _resolve(base_dir, str(data_cfg.get("train_path", "data/train.jsonl")))
        if not os.path.exists(train_path):
            raise FileNotFoundError(f"BPE 训练语料不存在：{train_path}")
        print(f"[build_tokenizer] kind=bpe, vocab_size={vocab_size}, corpus={train_path}", flush=True)
        build_tokenizer(
            corpus_path=train_path,
            vocab_size=vocab_size,
            save_path=tok_path,
            kind="bpe",
        )

    # 验证：加载并打印 vocab_size
    tok = load_tokenizer(tok_path, kind=tok_kind)
    print(f"[build_tokenizer] 完成，vocab_size={len(tok)}，wall_clock={time.time()-t0:.2f}s", flush=True)
    return tok_path


def stage_train(config_path: str, base_dir: str) -> dict:
    """训练阶段。"""
    print("=" * 70, flush=True)
    print("[stage 2/3] 训练", flush=True)
    print("=" * 70, flush=True)
    return train_fn(config_path, base_dir=base_dir)


def stage_evaluate(config_path: str, base_dir: str) -> dict:
    """评估阶段。"""
    print("=" * 70, flush=True)
    print("[stage 3/3] 评估", flush=True)
    print("=" * 70, flush=True)
    return evaluate_fn(config_path, base_dir=base_dir)


def stage_visualize(loss_history_path: str, save_dir: str) -> str:
    """可视化阶段。"""
    print("=" * 70, flush=True)
    print("[extra] 可视化 loss 曲线", flush=True)
    print("=" * 70, flush=True)
    curve_path = os.path.join(save_dir, "loss_curve.png")
    return visualize_fn(loss_history_path, save_path=curve_path)


def main():
    parser = argparse.ArgumentParser(
        description="CometSpark-v0.1 一键训练 + 评估入口"
    )
    parser.add_argument(
        "--config", default="config/config.yml",
        help="配置文件路径（默认 config/config.yml）",
    )
    parser.add_argument(
        "--skip-train", action="store_true",
        help="跳过训练阶段（仅 build + eval）",
    )
    parser.add_argument(
        "--skip-eval", action="store_true",
        help="跳过评估阶段",
    )
    parser.add_argument(
        "--skip-build", action="store_true",
        help="跳过 tokenizer 构建（已有 tokenizer.json 时使用）",
    )
    parser.add_argument(
        "--force-build", action="store_true",
        help="强制重建 tokenizer（覆盖已有文件）",
    )
    args = parser.parse_args()

    base_dir = _DEMO_DIR
    config_path = _resolve(base_dir, args.config)
    if not os.path.exists(config_path):
        print(f"错误：配置文件不存在 {config_path}", file=sys.stderr)
        sys.exit(1)

    print(f"CometSpark-v0.1 端到端流程开始", flush=True)
    print(f"  base_dir    = {base_dir}", flush=True)
    print(f"  config_path = {config_path}", flush=True)
    print(f"  skip_build  = {args.skip_build}", flush=True)
    print(f"  skip_train  = {args.skip_train}", flush=True)
    print(f"  skip_eval   = {args.skip_eval}", flush=True)
    print("", flush=True)

    overall_t0 = time.time()
    set_seed(42)

    # Stage 1: build tokenizer
    if not args.skip_build:
        try:
            tok_path = stage_build_tokenizer(
                config_path, base_dir, force=args.force_build
            )
        except Exception as e:
            print(f"[build_tokenizer] 失败：{e}", file=sys.stderr)
            raise
    else:
        print("[stage 1/3] 跳过 tokenizer 构建", flush=True)

    # Stage 2: train
    train_result = None
    if not args.skip_train:
        try:
            train_result = stage_train(config_path, base_dir)
        except Exception as e:
            print(f"[train] 失败：{e}", file=sys.stderr)
            raise
    else:
        print("[stage 2/3] 跳过训练", flush=True)

    # Stage 3: evaluate
    eval_result = None
    if not args.skip_eval:
        try:
            eval_result = stage_evaluate(config_path, base_dir)
        except Exception as e:
            print(f"[evaluate] 失败：{e}", file=sys.stderr)
            # 评估失败不应让整体流程退出码非 0
            print("[evaluate] 跳过评估（继续后续步骤）", flush=True)

    # Extra: visualize
    if train_result is not None:
        try:
            stage_visualize(
                train_result["loss_history_path"],
                train_result["checkpoint_dir"],
            )
        except Exception as e:
            print(f"[visualize] 失败：{e}", file=sys.stderr)

    # 汇总
    print("", flush=True)
    print("=" * 70, flush=True)
    print("CometSpark-v0.1 流程汇总", flush=True)
    print("=" * 70, flush=True)
    print(f"  总耗时 wall_clock = {time.time() - overall_t0:.2f}s", flush=True)
    if train_result is not None:
        print(f"  训练初始 loss = {train_result['initial_loss']:.4f}", flush=True)
        print(f"  训练最终 loss = {train_result['final_loss']:.4f}", flush=True)
        print(f"  最佳验证 loss = {train_result['best_val_loss']:.4f}", flush=True)
        print(f"  检查点目录    = {train_result['checkpoint_dir']}", flush=True)
    if eval_result is not None:
        print(f"  评估输出 {len(eval_result['results'])} 条样本：", flush=True)
        for r in eval_result["results"]:
            print(f"    [{r['prompt']!r}] -> {r['generated']!r}", flush=True)
    print("完成。", flush=True)


if __name__ == "__main__":
    main()
