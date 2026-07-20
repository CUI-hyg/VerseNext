"""测试：Mamba-2 推理时内存与序列长度无关 (Task 3.7).

验证 Mamba-2 在 recurrent 模式下单步解码的内存与已处理的序列长度无关。
理论上 recurrent 模式维护固定大小的状态 (B, n_heads, d_state, d_head) 与
conv_state (B, d_conv-1, d_inner)，与序列长度 T 无关。

测试方法：
    1. 跑 recurrent 模式处理 T=1000 个 token，测量峰值 RSS
    2. 跑 recurrent 模式处理 T=10000 个 token，测量峰值 RSS
    3. 比较两者的 RSS 差异，应 <= 10%

运行：
    python tests/test_mamba2_memory.py
"""

from __future__ import annotations

import os
import sys
import gc
import time
import resource
import argparse
from pathlib import Path

import numpy as np

# 确保 verse_torch 与 verse_nex 可导入
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "packages" / "verse_torch"))
sys.path.insert(0, str(REPO_ROOT / "packages" / "verse_nex"))

from verse_torch import Tensor, no_grad
from verse_nex import Mamba2Block


def get_rss_kb() -> int:
    """返回当前进程的 RSS（单位 KB，Linux）。"""
    try:
        # ru_maxrss: 最大 RSS，单位 KB (Linux) / bytes (macOS)
        return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    except Exception:
        return 0


def reset_rss_tracking():
    """强制 GC 后再清空缓存（仅尽量减小累积，ru_maxrss 不可重置）。"""
    gc.collect()


def run_recurrent_decode(model: Mamba2Block, n_tokens: int, dim: int, batch: int = 1):
    """跑 recurrent 解码 n_tokens 步，返回每步耗时与峰值 RSS。"""
    state = None
    # 模拟随机输入
    rng = np.random.default_rng(0)
    x_data = rng.standard_normal((batch, 1, dim)).astype(np.float32)

    t0 = time.time()
    for _ in range(n_tokens):
        x = Tensor(x_data)
        out, state = model.forward_recurrent(x, state)
        # 显式释放 out 的引用
        del out
    elapsed = time.time() - t0
    return elapsed, state


def _worker_decode(model_params, n_tokens, dim, batch, q):
    """子进程 worker：构建模型 + 跑 recurrent 解码 + 测量峰值 RSS。

    必须放在模块级别才能被 multiprocessing 序列化。
    """
    try:
        model = Mamba2Block(**model_params)
        run_recurrent_decode(model, n_tokens, dim, batch)
        rss = get_rss_kb()
        q.put(rss)
    except Exception as e:
        q.put(("error", str(e)))


def measure_max_rss_kb_during_decode(model: Mamba2Block, n_tokens: int, dim: int, batch: int = 1) -> int:
    """跑 recurrent 解码并测量峰值 RSS（KB）。

    注意：ru_maxrss 是进程生命周期内的最大值，不可重置。
    为得到 "本次解码" 的内存峰值，我们用子进程隔离测量：
    fork 一个子进程跑解码，父进程读取子进程的 ru_maxrss。
    """
    import multiprocessing as mp

    # 提取模型参数
    model_params = {
        "dim": model.dim,
        "d_state": model.d_state,
        "d_conv": model.d_conv,
        "expand": model.expand,
        "n_heads": model.n_heads,
    }

    q = mp.Queue()
    p = mp.Process(target=_worker_decode, args=(model_params, n_tokens, dim, batch, q))
    p.start()
    p.join()
    result = q.get()
    if isinstance(result, tuple) and result[0] == "error":
        raise RuntimeError(f"Worker failed: {result[1]}")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--short", type=int, default=1000, help="短序列长度（默认 1000）")
    parser.add_argument("--long", type=int, default=10000, help="长序列长度（默认 10000）")
    parser.add_argument("--dim", type=int, default=64, help="模型维度")
    parser.add_argument("--d-state", type=int, default=64, help="SSM 状态维度")
    parser.add_argument("--n-heads", type=int, default=8, help="头数")
    parser.add_argument("--expand", type=int, default=2, help="扩展倍数")
    parser.add_argument("--threshold", type=float, default=0.10, help="内存差异阈值（默认 10%%）")
    args = parser.parse_args()

    print(f"=== Mamba-2 Recurrent Memory Test ===")
    print(f"  Short sequence: {args.short} tokens")
    print(f"  Long sequence:  {args.long} tokens")
    print(f"  Model: dim={args.dim}, d_state={args.d_state}, n_heads={args.n_heads}, expand={args.expand}")
    print(f"  Threshold: {args.threshold * 100:.1f}%")
    print()

    # 创建模型（父进程）
    model = Mamba2Block(
        dim=args.dim,
        d_state=args.d_state,
        n_heads=args.n_heads,
        expand=args.expand,
    )

    # 测量短序列内存
    print(f"[1/2] Running recurrent decode for {args.short} tokens...")
    rss_short = measure_max_rss_kb_during_decode(model, args.short, args.dim)
    print(f"      Peak RSS: {rss_short} KB ({rss_short / 1024:.2f} MB)")

    # 测量长序列内存
    print(f"[2/2] Running recurrent decode for {args.long} tokens...")
    rss_long = measure_max_rss_kb_during_decode(model, args.long, args.dim)
    print(f"      Peak RSS: {rss_long} KB ({rss_long / 1024:.2f} MB)")

    # 比较
    rss_diff = abs(rss_long - rss_short)
    rss_max = max(rss_short, rss_long)
    rel_diff = rss_diff / rss_max if rss_max > 0 else 0.0

    print()
    print(f"=== Result ===")
    print(f"  RSS difference: {rss_diff} KB ({rss_diff / 1024:.2f} MB)")
    print(f"  Relative diff: {rel_diff * 100:.2f}%")
    print(f"  Threshold:     {args.threshold * 100:.2f}%")

    if rel_diff <= args.threshold:
        print(f"  PASS: 内存与序列长度无关（差异在阈值内）")
        return 0
    else:
        print(f"  FAIL: 内存差异超过阈值")
        # 提供诊断信息
        # 检查状态大小是否真的固定
        B = 1
        H = args.n_heads
        N = args.d_state
        d = (args.expand * args.dim) // H
        d_conv = 4  # default
        ssm_state_size = B * H * N * d * 4  # float32 = 4 bytes
        conv_state_size = B * (d_conv - 1) * (args.expand * args.dim) * 4
        total_state = ssm_state_size + conv_state_size
        print(f"  State size: ssm={ssm_state_size} B, conv={conv_state_size} B, total={total_state} B ({total_state / 1024:.2f} KB)")
        print(f"  Note: RSS includes Python interpreter + numpy + libraries overhead (~30-50 MB)")
        return 1


if __name__ == "__main__":
    sys.exit(main())
