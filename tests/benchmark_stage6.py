#!/usr/bin/env python3
"""Verse Framework Stage 6 综合基准测试 (Task 6.3).

包含三类基准：
1. 量化基准：FP32 / INT8 / INT4(cache_fp32=True|False) / ternary(cache_fp32=True|False)
   shapes: [512,512] / [1024,1024] / [2048,2048]，每配置 5 次取中位数
   报告 tokens/s、相对 FP32 加速比、与 FP32 的最大绝对误差

2. 内存基准：Mamba-2 与 RWKV-7 在 seq_len [1k, 4k, 16k, 64k] 下的单步解码
   峰值 RSS (KB)，每个配置在独立子进程中测量，避免内存累积

3. 训练吞吐量基准：Mamba-2 backbone (dim=64, layers=2, batch=4, seq=128)
   测 10 step wall-clock，报告 samples/s + tokens/s

约束：纯 Python + NumPy + 标准库；不依赖 pytest；内存约束 2GB
自包含，可独立运行：
    python3 /workspace/tests/benchmark_stage6.py
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import time

# 把 verse_torch / verse_nex 加入 path（独立运行场景）
_HERE = os.path.dirname(os.path.abspath(__file__))
_WORKSPACE = os.path.dirname(_HERE)
for _pkg in ("verse_torch", "verse_nex"):
    _p = os.path.join(_WORKSPACE, "packages", _pkg)
    if os.path.isdir(_p):
        sys.path.insert(0, _p)

import numpy as np  # noqa: E402

# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------
SEED = 42

# 量化基准配置
QUANT_SHAPES = [(512, 512), (1024, 1024), (2048, 2048)]
QUANT_BATCH = 8
QUANT_SEQ = 128
QUANT_N_ITERS = 5  # 每配置 5 次取中位数
QUANT_WARMUP = 2

# 内存基准配置
MEM_ARCHS = ["mamba2", "rwkv7"]
MEM_SEQ_LENS = [1024, 4096, 16384, 65536]
MEM_DIM = 64
MEM_BATCH = 1

# 训练吞吐量配置
TRAIN_DIM = 64
TRAIN_LAYERS = 2
TRAIN_BATCH = 4
TRAIN_SEQ = 128
TRAIN_STEPS = 10


# ---------------------------------------------------------------------------
# 量化基准
# ---------------------------------------------------------------------------


def model_size_bytes(obj) -> int:
    """估算模型/层的存储大小（bytes）。"""
    from verse_torch import nn
    from verse_torch.quantize import QuantizedLinear

    if isinstance(obj, nn.Linear):
        total = obj.weight.data.nbytes
        if obj.bias is not None:
            total += obj.bias.data.nbytes
        return total
    if isinstance(obj, QuantizedLinear):
        total = obj.packed.nbytes + obj.scale.nbytes
        if obj.bias is not None:
            total += obj.bias.nbytes
        return total
    raise TypeError(f"Unknown model type: {type(obj)}")


def time_forward_median(fn, x, n_iters: int, warmup: int) -> float:
    """计时 n_iters 次 forward，返回中位数时间（秒）。"""
    for _ in range(warmup):
        fn(x)
    times = []
    for _ in range(n_iters):
        start = time.perf_counter()
        fn(x)
        times.append(time.perf_counter() - start)
    times.sort()
    return times[n_iters // 2]


def run_quant_benchmark() -> list:
    from verse_torch import Tensor, nn
    from verse_torch.quantize import QuantizedLinear

    np.random.seed(SEED)
    tokens_per_call = QUANT_BATCH * QUANT_SEQ
    results = []

    print("=" * 110)
    print(f"Quantization Benchmark: batch={QUANT_BATCH}, seq={QUANT_SEQ}, "
          f"iters={QUANT_N_ITERS} (median), warmup={QUANT_WARMUP}")
    print("=" * 110)

    for (in_f, out_f) in QUANT_SHAPES:
        print(f"\n--- Shape: ({in_f}, {out_f}) ---")
        linear = nn.Linear(in_f, out_f, bias=True)
        x_np = np.random.randn(QUANT_BATCH, QUANT_SEQ, in_f).astype(np.float32)
        x_tensor = Tensor(x_np)
        y_fp32 = linear(x_tensor).numpy()

        configs = [
            ("FP32",            lambda lin: lin,                                                 x_tensor),
            ("INT8",            lambda lin: QuantizedLinear(lin, qtype="int8",    cache_fp32=True),  x_tensor),
            ("INT4_fp32",       lambda lin: QuantizedLinear(lin, qtype="int4",    cache_fp32=True),  x_tensor),
            ("INT4_nocache",    lambda lin: QuantizedLinear(lin, qtype="int4",    cache_fp32=False), x_tensor),
            ("Ternary_fp32",    lambda lin: QuantizedLinear(lin, qtype="ternary", cache_fp32=True),  x_tensor),
            ("Ternary_nocache", lambda lin: QuantizedLinear(lin, qtype="ternary", cache_fp32=False), x_tensor),
        ]

        fp32_tps = None
        header = (f"  {'Name':<18} {'Size(KB)':>10} {'Time(ms)':>10} "
                  f"{'Tokens/s':>12} {'Speedup':>10} {'MaxDiff':>12}")
        print(header)
        print("  " + "-" * (len(header) - 2))

        for name, factory, x_in in configs:
            model = factory(linear)
            t = time_forward_median(model, x_in, QUANT_N_ITERS, QUANT_WARMUP)
            tps = tokens_per_call / t
            size_b = model_size_bytes(model)
            if fp32_tps is None:
                fp32_tps = tps
            speedup = tps / fp32_tps

            y_q = model(x_in)
            if isinstance(y_q, Tensor):
                y_q = y_q.numpy()
            max_diff = float(np.abs(y_fp32 - y_q).max())

            results.append({
                "shape": f"{in_f}x{out_f}",
                "name": name,
                "size_bytes": size_b,
                "time_ms": t * 1000.0,
                "tokens_per_sec": tps,
                "speedup": speedup,
                "max_diff": max_diff,
            })
            print(f"  {name:<18} {size_b/1024:>10.1f} {t*1000:>10.3f} "
                  f"{tps:>12.1f} {speedup:>9.2f}x {max_diff:>12.6f}")

    return results


# ---------------------------------------------------------------------------
# 内存基准（子进程测量）
# ---------------------------------------------------------------------------


MEM_CHILD_SCRIPT_TEMPLATE = """
import sys, os, gc, resource
for _pkg in ("verse_torch", "verse_nex"):
    _p = os.path.join({workspace!r}, "packages", _pkg)
    if os.path.isdir(_p):
        sys.path.insert(0, _p)

import numpy as np
from verse_torch import Tensor, no_grad
from verse_nex import Mamba2Block, RWKV7Block

np.random.seed(42)
dim = {dim}
seq_len = {seq_len}
batch = {batch}
arch = {arch!r}

if arch == "mamba2":
    model = Mamba2Block(dim=dim, d_state=64, d_conv=4, expand=2, n_heads=4)
else:
    model = RWKV7Block(dim=dim, n_head=4, head_size=16, hidden=128)
model.eval()

# 跑 seq_len 步 recurrent 预热（填满状态），再跑 1 步解码
with no_grad():
    state = None
    for t in range(seq_len):
        x = Tensor(np.random.randn(batch, 1, dim).astype(np.float32))
        out = model(x, state=state, mode="recurrent")
        state = out._state
    # 单步解码
    x = Tensor(np.random.randn(batch, 1, dim).astype(np.float32))
    _ = model(x, state=state, mode="recurrent")

gc.collect()
rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
print(rss_kb)
"""


def run_memory_benchmark() -> list:
    print("\n" + "=" * 110)
    print(f"Memory Benchmark: dim={MEM_DIM}, batch={MEM_BATCH}, "
          f"recurrent single-step decode (peak RSS in subprocess)")
    print(f"Sequence lengths: {MEM_SEQ_LENS}")
    print("=" * 110)

    header = f"{'Arch':<10} " + " ".join(f"seq={s:<6d}" for s in MEM_SEQ_LENS)
    print(header)
    print("-" * len(header))

    results = []
    for arch in MEM_ARCHS:
        row = []
        for seq_len in MEM_SEQ_LENS:
            script = MEM_CHILD_SCRIPT_TEMPLATE.format(
                workspace=_WORKSPACE, dim=MEM_DIM, seq_len=seq_len,
                batch=MEM_BATCH, arch=arch,
            )
            try:
                # 64k 步可能耗时较长，给 600s 超时
                result = subprocess.run(
                    [sys.executable, "-c", script],
                    capture_output=True, text=True, timeout=900,
                )
                if result.returncode != 0:
                    print(f"  [ERROR] {arch} seq={seq_len}: "
                          f"{result.stderr.strip()[:300]}", file=sys.stderr)
                    row.append(None)
                else:
                    rss_kb = int(result.stdout.strip())
                    row.append(rss_kb)
            except subprocess.TimeoutExpired:
                print(f"  [TIMEOUT] {arch} seq={seq_len}", file=sys.stderr)
                row.append(None)

        for seq_len, rss in zip(MEM_SEQ_LENS, row):
            results.append({
                "arch": arch,
                "seq_len": seq_len,
                "rss_kb": rss,
            })

        cells = []
        for rss in row:
            if rss is None:
                cells.append(f"{'N/A':>10}")
            else:
                cells.append(f"{rss/1024:>7.1f} MB")
        print(f"{arch:<10} " + " ".join(cells))

    return results


# ---------------------------------------------------------------------------
# 训练吞吐量基准
# ---------------------------------------------------------------------------


def run_train_throughput_benchmark() -> dict:
    from verse_torch import Tensor, AdamW
    from verse_torch.losses import cross_entropy
    from verse_nex import HybridLM

    print("\n" + "=" * 110)
    print(f"Train Throughput Benchmark: Mamba-2 backbone, dim={TRAIN_DIM}, "
          f"layers={TRAIN_LAYERS}, batch={TRAIN_BATCH}, seq={TRAIN_SEQ}, "
          f"steps={TRAIN_STEPS}")
    print("=" * 110)

    np.random.seed(SEED)
    vocab = 256
    model = HybridLM(
        vocab_size=vocab, dim=TRAIN_DIM, n_layers=TRAIN_LAYERS,
        sparse_ratio=0.0, ssm_kind="mamba2",
        ssm_kwargs={"d_state": 64, "d_conv": 4, "expand": 2, "n_heads": 4},
    )
    model.train()
    optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)

    input_data = np.random.randint(0, vocab, size=(TRAIN_BATCH, TRAIN_SEQ)).astype(np.int64)
    target_data = np.random.randint(0, vocab, size=(TRAIN_BATCH, TRAIN_SEQ)).astype(np.int64)

    # 预热 2 步
    print("Warming up 2 steps...")
    for _ in range(2):
        x = Tensor(input_data)
        logits = model.forward_parallel(x)
        B, T, V = logits.data.shape
        logits_flat = logits.reshape(B * T, V)
        targets_flat = Tensor(target_data.reshape(B * T))
        loss = cross_entropy(logits_flat, targets_flat)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # 计时
    print(f"Timing {TRAIN_STEPS} steps...")
    start = time.perf_counter()
    final_loss = None
    for _ in range(TRAIN_STEPS):
        x = Tensor(input_data)
        logits = model.forward_parallel(x)
        B, T, V = logits.data.shape
        logits_flat = logits.reshape(B * T, V)
        targets_flat = Tensor(target_data.reshape(B * T))
        loss = cross_entropy(logits_flat, targets_flat)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        final_loss = float(loss.data)
    elapsed = time.perf_counter() - start

    samples_per_sec = (TRAIN_BATCH * TRAIN_STEPS) / elapsed
    tokens_per_sec = (TRAIN_BATCH * TRAIN_SEQ * TRAIN_STEPS) / elapsed

    result = {
        "dim": TRAIN_DIM,
        "layers": TRAIN_LAYERS,
        "batch": TRAIN_BATCH,
        "seq": TRAIN_SEQ,
        "steps": TRAIN_STEPS,
        "wall_clock_s": elapsed,
        "samples_per_sec": samples_per_sec,
        "tokens_per_sec": tokens_per_sec,
        "final_loss": final_loss,
    }
    print(f"  wall-clock:    {elapsed:.3f}s ({elapsed/TRAIN_STEPS:.3f}s/step)")
    print(f"  samples/s:     {samples_per_sec:.3f}")
    print(f"  tokens/s:      {tokens_per_sec:.1f}")
    print(f"  final loss:    {final_loss:.4f}")
    return result


# ---------------------------------------------------------------------------
# 环境信息收集
# ---------------------------------------------------------------------------


def collect_env_info() -> dict:
    env = {
        "python_version": sys.version.split()[0],
        "numpy_version": np.__version__,
        "platform": platform.platform(),
        "processor": platform.processor() or "unknown",
        "cpu_count": os.cpu_count() or "unknown",
    }
    try:
        import numba
        env["numba_version"] = numba.__version__
    except ImportError:
        env["numba_version"] = None
    try:
        blas_info = np.__config__.get_info("blas_opt")
        env["numpy_blas"] = blas_info.get("libraries", ["unknown"]) if blas_info else "unknown"
    except Exception:
        env["numpy_blas"] = "unknown"
    # VerseTorch / VerseNex 版本
    try:
        import verse_torch
        env["verse_torch_version"] = verse_torch.__version__
    except Exception:
        env["verse_torch_version"] = "unknown"
    try:
        import verse_nex
        env["verse_nex_version"] = verse_nex.__version__
    except Exception:
        env["verse_nex_version"] = "unknown"
    return env


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------


def main() -> int:
    print("Verse Framework Stage 6 Benchmark Suite")
    print(f"Workspace: {_WORKSPACE}")
    print(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    env = collect_env_info()
    print(f"Env: {json.dumps(env, indent=2, default=str)}")
    print()

    # 1. 量化基准
    quant_results = run_quant_benchmark()

    # 2. 内存基准
    mem_results = run_memory_benchmark()

    # 3. 训练吞吐量
    train_result = run_train_throughput_benchmark()

    # 输出汇总 JSON
    summary = {
        "env": env,
        "quant": quant_results,
        "memory": mem_results,
        "train": train_result,
        "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
    }
    out_json = os.path.join(_WORKSPACE, "docs", "benchmarks", "benchmark_stage6_data.json")
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nJSON data saved to: {out_json}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
