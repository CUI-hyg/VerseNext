"""Task 5.5: 端到端 CPU 推理示例。

构建一个小型 Mamba-2 LM（< 50M 参数），用 ``verse_inference.StreamingGenerator``
流式生成 100 个 token，并报告：

- 生成时间（秒）
- 吞吐量（tokens/s）
- 峰值 RSS（MB）
- 参数量

约束
----
- 4 核 CPU 上 5 分钟内完成
- 峰值 RSS ≤ 8 GB

运行
----
    cd /workspace
    PYTHONPATH=packages/verse_torch:packages/verse_nex:packages/verse_infra \
        python3 examples/cpu_inference_demo.py

可选参数
--------
    --vocab-size 256 --dim 128 --n-layers 4 --max-new-tokens 100
    --temperature 0.8 --top-k 20 --top-p 0.95
"""

from __future__ import annotations

import argparse
import os
import resource
import sys
import time

import numpy as np


# ---------------------------------------------------------------------------
# 工具：测量峰值 RSS（MB）
# ---------------------------------------------------------------------------


def _peak_rss_mb() -> float:
    """返回当前进程峰值 RSS（MB）。

    Linux: 用 ``resource.getrusage(RUSAGE_SELF).ru_maxrss``
    （单位是 KB on Linux，bytes on macOS，这里只支持 Linux）。
    """
    ru = resource.getrusage(resource.RUSAGE_SELF)
    # Linux: ru_maxrss 单位是 KB
    return ru.ru_maxrss / 1024.0


def _count_parameters(model) -> int:
    """统计模型可训练参数总量。

    注意：``model.parameters()`` 通常只返回 ``requires_grad=True`` 的参数，
    而 ``ModelLoader.load()`` 在返回前会把所有参数的 ``requires_grad`` 关闭
    （推理模式）。因此这里改用 ``state_dict()``，它会返回所有参数（无过滤），
    保证统计到完整参数量。
    """
    n = 0
    for _, v in model.state_dict().items():
        arr = np.asarray(v)
        n += int(arr.size)
    return n


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="VerseInference CPU demo")
    parser.add_argument("--arch", default="mamba2",
                        help="arch: mamba2 / rwkv7 / hybrid")
    parser.add_argument("--vocab-size", type=int, default=256)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--d-state", type=int, default=64,
                        help="SSM 状态维度（仅 mamba2 / hybrid）")
    parser.add_argument("--n-heads", type=int, default=4,
                        help="SSD 头数（仅 mamba2 / hybrid）")
    parser.add_argument("--max-new-tokens", type=int, default=100)
    parser.add_argument("--prompt", default="Hello Mamba",
                        help="prompt 文本（由 CharTokenizer 编码）")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # 设置随机种子
    np.random.seed(args.seed)

    # 延迟导入，便于 --help 快速响应
    from verse_torch import Tensor, no_grad
    from verse_nex import HybridLM
    from verse_infra.verse_tokenizer import CharTokenizer
    from verse_infra.verse_inference import ModelLoader, Sampler, StreamingGenerator

    print("=" * 72)
    print("VerseInference 端到端 CPU 推理示例")
    print("=" * 72)
    print(f"架构:           {args.arch}")
    print(f"vocab_size:     {args.vocab_size}")
    print(f"dim:            {args.dim}")
    print(f"n_layers:       {args.n_layers}")
    print(f"max_new_tokens: {args.max_new_tokens}")
    print(f"temperature:    {args.temperature}")
    print(f"top_k:          {args.top_k}")
    print(f"top_p:          {args.top_p}")
    print(f"CPU 核数:       {os.cpu_count()}")
    print()

    # 1. 构建 LM
    print("[1/4] 构建 Mamba-2 LM ...")
    t0 = time.perf_counter()
    loader = ModelLoader(
        arch=args.arch,
        vocab_size=args.vocab_size,
        dim=args.dim,
        n_layers=args.n_layers,
        ssm_kwargs={"d_state": args.d_state, "d_conv": 4, "expand": 2, "n_heads": args.n_heads},
        sparse_kwargs={"n_heads": args.n_heads, "chunk_size": 16,
                       "n_sliding_chunks": 1, "topk_chunks": 1},
    )
    model = loader.load()
    n_params = _count_parameters(model)
    t_build = time.perf_counter() - t0
    print(f"    参数量:      {n_params:,} ({n_params / 1e6:.2f}M)")
    print(f"    构建时间:    {t_build:.2f}s")
    assert n_params < 50_000_000, f"参数量 {n_params} 超过 50M 上限"

    # 2. 准备 tokenizer
    print("[2/4] 准备 tokenizer ...")
    tokenizer = CharTokenizer()
    # 预填充 CharTokenizer 的 vocab：把所有可能的字节值（0..255）映射为字符，
    # 这样模型生成的任意 id (4..vocab_size-1) 都能被 decode 还原为可见字符。
    # 否则 CharTokenizer 是懒加载的，只包含 prompt 中出现过的字符，
    # 模型生成 prompt 之外的 id 时 decode 会返回空字符串。
    # 注意：special tokens 占用 0..3，所以字符 id 从 4 开始；
    # 我们填充 (vocab_size - 4) 个字符，保证 model 输出 id < vocab_size 都能解码。
    n_prepopulate = max(0, args.vocab_size - 4)
    for b in range(n_prepopulate):
        # 用 latin-1 解码字节为字符（0..255 全部可解码，无异常）
        tokenizer._ensure_char(chr(b))
    print(f"    Tokenizer:   CharTokenizer (vocab={len(tokenizer)}, pre-populated)")
    print(f"    Prompt:      {args.prompt!r}")

    prompt_ids = tokenizer.encode(args.prompt, add_special_tokens=False)
    print(f"    Prompt ids:  {prompt_ids} (len={len(prompt_ids)})")
    # 确保所有 id < vocab_size
    for i in prompt_ids:
        if i >= args.vocab_size:
            print(f"    警告: prompt id {i} >= vocab_size {args.vocab_size}，"
                  f"将截断。请使用更短的 prompt 或更大的 vocab_size。")
            prompt_ids = [i for i in prompt_ids if i < args.vocab_size]
            break

    # 3. 流式生成
    print("[3/4] 流式生成 ...")
    sampler = Sampler(
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        seed=args.seed,
    )
    gen = StreamingGenerator(model, tokenizer=tokenizer, sampler=sampler)

    t_gen_start = time.perf_counter()
    generated_ids = []
    generated_text_parts = []
    print("    ", end="", flush=True)
    for tok_id in gen.generate(prompt_ids, max_new_tokens=args.max_new_tokens):
        generated_ids.append(tok_id)
        piece = tokenizer.decode([tok_id])
        generated_text_parts.append(piece)
        # 打印（避免控制字符破坏终端）
        safe = piece.replace("\n", "\\n").replace("\r", "\\r")
        print(safe, end="", flush=True)
    print()
    t_gen = time.perf_counter() - t_gen_start

    # 4. 报告
    print("[4/4] 报告:")
    full_text = tokenizer.decode(prompt_ids) + "".join(generated_text_parts)
    peak_rss = _peak_rss_mb()
    tokens_per_sec = len(generated_ids) / t_gen if t_gen > 0 else 0.0

    print(f"    生成时间:    {t_gen:.2f}s")
    print(f"    生成 token:  {len(generated_ids)}")
    print(f"    吞吐量:      {tokens_per_sec:.2f} tokens/s")
    print(f"    峰值 RSS:    {peak_rss:.1f} MB")
    print(f"    完整文本:    {full_text!r}")

    # 断言约束
    print()
    print("约束检查:")
    ok_time = t_gen < 300  # 5 分钟
    ok_rss = peak_rss < 8 * 1024  # 8 GB
    ok_params = n_params < 50_000_000
    print(f"    [{'OK' if ok_time else 'FAIL'}] 生成时间 < 300s  (实际 {t_gen:.1f}s)")
    print(f"    [{'OK' if ok_rss else 'FAIL'}] 峰值 RSS < 8192MB (实际 {peak_rss:.1f}MB)")
    print(f"    [{'OK' if ok_params else 'FAIL'}] 参数量 < 50M    (实际 {n_params / 1e6:.2f}M)")

    if not (ok_time and ok_rss and ok_params):
        print("\n部分约束未通过，但 demo 仍完成。")
        sys.exit(1)
    print("\n所有约束通过！")


if __name__ == "__main__":
    main()
