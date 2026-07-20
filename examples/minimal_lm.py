"""示例：字符级 LM 训练 (Task 3.9).

使用 VerseNex Mamba-2 backbone 训练一个微型字符级语言模型。

任务：
    输入一段文本，模型学习预测下一个字符。

数据：
    内置一段英文文本（莎士比亚风格），按字符切分。
    训练时随机采样固定长度的子序列。

模型：
    HybridLM (vocab=charset, dim=64, n_layers=2, sparse_ratio=0.0)
    即纯 Mamba-2 backbone（sparse_ratio=0 表示无 sparse attention 层）

训练：
    - 优化器: AdamW
    - Loss: cross-entropy
    - 采样固定长度子序列训练若干步
    - 打印 loss 曲线
    - 训练后用 recurrent 模式生成一段文本

运行：
    python examples/minimal_lm.py
"""

from __future__ import annotations

import sys
import argparse
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "packages" / "verse_torch"))
sys.path.insert(0, str(REPO_ROOT / "packages" / "verse_nex"))

from verse_torch import Tensor, no_grad
from verse_torch.optim import AdamW
from verse_torch.losses import cross_entropy
from verse_nex import HybridLM


# ---------------------------------------------------------------------------
# 数据：内置文本
# ---------------------------------------------------------------------------


SAMPLE_TEXT = """To be, or not to be, that is the question:
Whether 'tis nobler in the mind to suffer
The slings and arrows of outrageous fortune,
Or to take arms against a sea of troubles
And by opposing end them. To die—to sleep,
No more; and by a sleep to say we end
The heart-ache and the thousand natural shocks
That flesh is heir to: 'tis a consummation
Devoutly to be wish'd. To die, to sleep;
To sleep, perchance to dream—ay, there's the rub:
For in that sleep of death what dreams may come,
When we have shuffled off this mortal coil,
Must give us pause."""


def build_vocab(text: str):
    """构建字符级词表。"""
    chars = sorted(set(text))
    char_to_id = {c: i for i, c in enumerate(chars)}
    id_to_char = {i: c for i, c in enumerate(chars)}
    return chars, char_to_id, id_to_char


def encode(text: str, char_to_id: dict) -> np.ndarray:
    return np.array([char_to_id[c] for c in text], dtype=np.int64)


def decode(ids: np.ndarray, id_to_char: dict) -> str:
    return "".join(id_to_char[int(i)] for i in ids)


# ---------------------------------------------------------------------------
# 训练数据采样
# ---------------------------------------------------------------------------


def sample_batch(
    data: np.ndarray,
    batch_size: int,
    seq_len: int,
    rng: np.random.Generator,
):
    """随机采样 batch_size 个长度为 seq_len+1 的子序列。

    Returns:
        inputs: (B, seq_len) 输入字符 ids
        targets: (B, seq_len) 目标字符 ids（输入右移一位）
    """
    n = len(data)
    starts = rng.integers(0, n - seq_len - 1, size=batch_size)
    inputs = np.stack([data[s:s + seq_len] for s in starts], axis=0)
    targets = np.stack([data[s + 1:s + seq_len + 1] for s in starts], axis=0)
    return inputs, targets


# ---------------------------------------------------------------------------
# 训练
# ---------------------------------------------------------------------------


def train(
    model: HybridLM,
    data: np.ndarray,
    n_steps: int,
    batch_size: int,
    seq_len: int,
    lr: float = 1e-3,
    log_every: int = 20,
    rng: np.random.Generator = None,
):
    if rng is None:
        rng = np.random.default_rng(0)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    model.train()
    losses = []
    for step in range(n_steps):
        inputs, targets = sample_batch(data, batch_size, seq_len, rng)
        input_ids = Tensor(inputs)
        logits = model.forward_parallel(input_ids)  # (B, T, V)
        # flatten for cross_entropy: (B*T, V) and (B*T,)
        B, T, V = logits.shape
        logits_flat = logits.reshape(B * T, V)
        targets_flat = Tensor(targets.reshape(B * T))
        loss = cross_entropy(logits_flat, targets_flat)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(float(loss.data))
        if log_every > 0 and (step + 1) % log_every == 0:
            recent = losses[-log_every:]
            print(f"  Step {step + 1}/{n_steps}: loss = {np.mean(recent):.4f} (min={np.min(recent):.4f})")
    return losses


# ---------------------------------------------------------------------------
# 生成
# ---------------------------------------------------------------------------


def generate(
    model: HybridLM,
    prompt: str,
    char_to_id: dict,
    id_to_char: dict,
    max_new_tokens: int = 100,
    mode: str = "recurrent",
) -> str:
    """从 prompt 开始生成文本。"""
    model.eval()
    # encode prompt
    prompt_ids = np.array([char_to_id[c] for c in prompt if c in char_to_id], dtype=np.int64)
    if len(prompt_ids) == 0:
        # 用第一个字符作为默认 prompt
        prompt_ids = np.array([0], dtype=np.int64)
        prompt = id_to_char[0]

    input_ids = Tensor(prompt_ids[None, :])  # (1, T_prompt)
    with no_grad():
        generated = model.generate(input_ids, max_new_tokens=max_new_tokens, mode=mode)
    generated = generated[0]  # (T_prompt + max_new_tokens,)
    text = decode(generated, id_to_char)
    return text


# ---------------------------------------------------------------------------
# 主程序
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dim", type=int, default=64, help="模型维度")
    parser.add_argument("--n-layers", type=int, default=2, help="层数")
    parser.add_argument("--sparse-ratio", type=float, default=0.0, help="Sparse 层占比（0=纯 SSM）")
    parser.add_argument("--n-heads", type=int, default=4, help="SSM 头数")
    parser.add_argument("--seq-len", type=int, default=32, help="训练序列长度")
    parser.add_argument("--batch-size", type=int, default=4, help="训练 batch size")
    parser.add_argument("--n-steps", type=int, default=100, help="训练步数")
    parser.add_argument("--lr", type=float, default=1e-3, help="学习率")
    parser.add_argument("--gen-len", type=int, default=80, help="生成长度")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    args = parser.parse_args()

    print("=== VerseNex Minimal Character-Level LM ===")
    print(f"  Model: dim={args.dim}, n_layers={args.n_layers}, sparse_ratio={args.sparse_ratio}")
    print(f"  Training: {args.n_steps} steps, batch={args.batch_size}, seq_len={args.seq_len}, lr={args.lr}")
    print()

    # 构建词表
    text = SAMPLE_TEXT
    chars, char_to_id, id_to_char = build_vocab(text)
    vocab_size = len(chars)
    print(f"Vocab size: {vocab_size}")
    print(f"Vocab: {''.join(chars)}")
    print()

    # encode
    data = encode(text, char_to_id)
    print(f"Data length: {len(data)} chars")
    print()

    # 设置随机种子
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)

    # 构建模型
    model = HybridLM(
        vocab_size=vocab_size,
        dim=args.dim,
        n_layers=args.n_layers,
        sparse_ratio=args.sparse_ratio,
        ssm_kind="mamba2",
        ssm_kwargs={
            "n_heads": args.n_heads,
            "d_state": 32,
            "d_conv": 4,
            "expand": 2,
        },
        sparse_kwargs={
            "n_heads": args.n_heads,
            "chunk_size": 16,
            "n_sliding_chunks": 1,
            "topk_chunks": 1,
        },
        sparse_placement="spread",
        tie_weights=True,
    )
    n_params = sum(p.data.size for p in model.parameters())
    print(f"Model parameters: {n_params}")
    print()

    # 训练前生成（baseline）
    print("[1/3] Sample before training:")
    sample_text = generate(model, "To be", char_to_id, id_to_char,
                           max_new_tokens=args.gen_len, mode="recurrent")
    print(f"  {sample_text!r}")
    print()

    # 训练
    print("[2/3] Training...")
    losses = train(
        model, data, args.n_steps, args.batch_size, args.seq_len,
        lr=args.lr, log_every=20, rng=rng,
    )
    print(f"  Loss: initial={losses[0]:.4f}, final={losses[-1]:.4f}, "
          f"min={min(losses):.4f}")
    # 验证 loss 单调下降（大致趋势）
    first_quarter = np.mean(losses[:max(1, len(losses) // 4)])
    last_quarter = np.mean(losses[-max(1, len(losses) // 4):])
    print(f"  Avg loss (first 25%): {first_quarter:.4f}")
    print(f"  Avg loss (last 25%):  {last_quarter:.4f}")
    if last_quarter < first_quarter:
        print(f"  Loss decreased: YES (delta = {first_quarter - last_quarter:.4f})")
    else:
        print(f"  Loss decreased: NO (delta = {last_quarter - first_quarter:.4f})")
    print()

    # 训练后生成
    print("[3/3] Sample after training:")
    sample_text = generate(model, "To be", char_to_id, id_to_char,
                           max_new_tokens=args.gen_len, mode="recurrent")
    print(f"  {sample_text!r}")
    print()

    # 比较 parallel 与 recurrent 生成（一致性验证）
    print("[Bonus] Consistency check: parallel vs recurrent generation:")
    gen_par = generate(model, "To be", char_to_id, id_to_char,
                       max_new_tokens=20, mode="parallel")
    gen_rec = generate(model, "To be", char_to_id, id_to_char,
                       max_new_tokens=20, mode="recurrent")
    print(f"  Parallel: {gen_par!r}")
    print(f"  Recurrent: {gen_rec!r}")
    match = gen_par == gen_rec
    print(f"  Match: {match}")
    print()

    print("=== Done ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
