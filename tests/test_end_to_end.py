"""Task 6.2: Verse 框架端到端测试.

6 个端到端测试，每个有超时保护（signal.alarm）：
1. test_mnist_mlp_smoke: MNIST MLP 1 epoch loss 下降
2. test_char_lm_smoke: 字符级 LM loss 下降
3. test_ijepa_cifar10_smoke: I-JEPA 合成数据预训练 loss 下降
4. test_rssm_moving_mnist_smoke: RSSM 视频预测 MSE ≤ 0.20
5. test_jepa_demo_smoke: JEPA demo 最终 loss < 0.1
6. test_cpu_inference_smoke: CPU 推理生成 token 数 ≥ 10

运行方式：
    python3 tests/test_end_to_end.py
"""

from __future__ import annotations

import os
import signal
import sys
import time
import traceback
from pathlib import Path

import numpy as np

# 让 tests/ 目录能 import 各个包
_REPO_ROOT = Path(__file__).resolve().parent.parent
for pkg in ("verse_torch", "verse_nex", "verse_awm",
            "verse_tokenizer", "verse_inference"):
    p = _REPO_ROOT / "packages" / pkg
    if p.is_dir():
        sys.path.insert(0, str(p))

# 把 examples/ 加入 path 以便复用其中的辅助函数
sys.path.insert(0, str(_REPO_ROOT / "examples"))
# 把 tests/ 自身加入 path 以便复用现有测试的辅助函数
sys.path.insert(0, str(_REPO_ROOT / "tests"))


# ---------------------------------------------------------------------------
# 超时保护
# ---------------------------------------------------------------------------


class TestTimeout(Exception):
    """测试超时异常。"""


def _alarm_handler(signum, frame):
    raise TestTimeout("test timed out")


def run_with_timeout(fn, timeout_sec: int, name: str):
    """运行 fn()，最多 timeout_sec 秒。

    Returns:
        (passed: bool, detail: str)
    """
    old_handler = signal.signal(signal.SIGALRM, _alarm_handler)
    signal.alarm(timeout_sec)
    try:
        result = fn()
        # 期望 fn 返回 (passed, detail)
        if isinstance(result, tuple) and len(result) == 2:
            return result
        return (bool(result), "")
    except TestTimeout:
        return (False, f"timeout after {timeout_sec}s")
    except Exception as e:
        tb = traceback.format_exc()
        return (False, f"exception: {e!r}\n{tb}")
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


# ---------------------------------------------------------------------------
# 统计
# ---------------------------------------------------------------------------

_RESULTS: list[tuple[str, bool, str, float]] = []  # (name, passed, detail, elapsed)


def _run_test(name: str, fn, timeout_sec: int):
    """运行单个测试并记录结果。"""
    print(f"\n--- {name} (timeout={timeout_sec}s) ---")
    t0 = time.time()
    passed, detail = run_with_timeout(fn, timeout_sec, name)
    elapsed = time.time() - t0
    status = "PASS" if passed else "FAIL"
    print(f"  [{status}] {name}  ({elapsed:.1f}s)")
    if detail and not passed:
        # 只打印前 500 字符，避免过长
        print(f"  detail: {detail[:500]}")
    _RESULTS.append((name, passed, detail, elapsed))


# ---------------------------------------------------------------------------
# Test 1: MNIST MLP smoke
# ---------------------------------------------------------------------------


def test_mnist_mlp_smoke():
    """1 epoch MLP 训练，验证 loss 下降。MNIST 不可用则用合成数据。"""
    from verse_torch import Tensor, optim, losses, no_grad
    import mnist_mlp as mnist_example

    np.random.seed(42)
    # 尝试加载 MNIST，失败则用合成数据
    try:
        train_x, train_y, _, _ = mnist_example.load_mnist()
        # 子采样以加速：仅取 2000 样本
        train_x = train_x[:2000]
        train_y = train_y[:2000]
        print(f"  MNIST loaded: {train_x.shape}")
    except Exception as e:
        print(f"  MNIST load failed ({e!r}), using synthetic data")
        # 合成数据：10 类，每类一个高斯团
        n_per = 200
        Xs, Ys = [], []
        for c in range(10):
            center = np.random.randn(784) * 3
            Xc = center + np.random.randn(n_per, 784) * 0.5
            Xs.append(Xc.astype(np.float32))
            Ys.append(np.full(n_per, c))
        train_x = np.concatenate(Xs)
        train_y = np.concatenate(Ys).astype(np.int64)

    model = mnist_example.MLP(in_features=784, hidden=64, num_classes=10)
    opt = optim.Adam(model.parameters(), lr=1e-3)

    model.train()
    losses_log = []
    batch_size = 32
    for xb, yb in mnist_example.iterate_minibatches(train_x, train_y,
                                                     batch_size, shuffle=True):
        x = Tensor(xb, requires_grad=False)
        logits = model(x)
        loss = losses.cross_entropy(logits, yb)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses_log.append(float(loss.item()))
        # 仅训练 ~50 个 batch 以控制时间
        if len(losses_log) >= 50:
            break

    if len(losses_log) < 4:
        return (False, f"too few batches: {len(losses_log)}")

    first_avg = float(np.mean(losses_log[:max(1, len(losses_log) // 4)]))
    last_avg = float(np.mean(losses_log[-max(1, len(losses_log) // 4):]))
    print(f"  loss: first={losses_log[0]:.4f}, last={losses_log[-1]:.4f}, "
          f"first_avg={first_avg:.4f}, last_avg={last_avg:.4f}")
    if last_avg < first_avg:
        return (True, f"loss decreased {first_avg:.4f}->{last_avg:.4f}")
    return (False, f"loss did not decrease: {first_avg:.4f}->{last_avg:.4f}")


# ---------------------------------------------------------------------------
# Test 2: Char LM smoke
# ---------------------------------------------------------------------------


def test_char_lm_smoke():
    """字符级 LM 训练 ~30 步，验证 loss 下降。"""
    from verse_torch import Tensor
    from verse_torch.optim import AdamW
    from verse_torch.losses import cross_entropy
    from verse_nex import HybridLM
    import minimal_lm as lm_example

    np.random.seed(42)
    text = lm_example.SAMPLE_TEXT
    chars, char_to_id, id_to_char = lm_example.build_vocab(text)
    data = lm_example.encode(text, char_to_id)
    vocab_size = len(chars)
    print(f"  vocab={vocab_size}, data_len={len(data)}")

    model = HybridLM(
        vocab_size=vocab_size, dim=64, n_layers=2, sparse_ratio=0.0,
        ssm_kind="mamba2",
        ssm_kwargs={"n_heads": 4, "d_state": 32, "d_conv": 4, "expand": 2},
        sparse_kwargs={},
        sparse_placement="spread", tie_weights=True,
    )
    opt = AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    rng = np.random.default_rng(42)

    model.train()
    losses = []
    n_steps = 30
    batch_size = 4
    seq_len = 32
    for step in range(n_steps):
        inputs, targets = lm_example.sample_batch(data, batch_size, seq_len, rng)
        input_ids = Tensor(inputs)
        logits = model.forward_parallel(input_ids)
        B, T, V = logits.shape
        logits_flat = logits.reshape(B * T, V)
        targets_flat = Tensor(targets.reshape(B * T))
        loss = cross_entropy(logits_flat, targets_flat)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(float(loss.data))

    first_avg = float(np.mean(losses[:5]))
    last_avg = float(np.mean(losses[-5:]))
    print(f"  loss: first={losses[0]:.4f}, last={losses[-1]:.4f}, "
          f"first_avg={first_avg:.4f}, last_avg={last_avg:.4f}")
    if last_avg < first_avg:
        return (True, f"loss decreased {first_avg:.4f}->{last_avg:.4f}")
    return (False, f"loss did not decrease: {first_avg:.4f}->{last_avg:.4f}")


# ---------------------------------------------------------------------------
# Test 3: I-JEPA CIFAR-10 smoke (合成数据)
# ---------------------------------------------------------------------------


def test_ijepa_cifar10_smoke():
    """I-JEPA 合成数据预训练 ~15 步，验证 loss 下降。"""
    from verse_torch import Tensor, optim
    from verse_awm import IJEPA, update_target_encoder, ema_decay_schedule
    import test_ijepa_cifar10 as ijepa_test

    np.random.seed(42)
    # 用较小的合成数据集
    train_x, _, _, _ = ijepa_test.make_synthetic_data(n_train=200, n_test=50,
                                                       img_size=32)
    train_x = ijepa_test.normalize_images(train_x)
    print(f"  synthetic data: {train_x.shape}")

    model = IJEPA(
        img_size=32, patch_size=4, in_channels=3,
        embed_dim=32, depth=2, n_heads=2,
        predictor_depth=2,
    )
    params = (
        list(model.context_encoder.parameters())
        + list(model.predictor.parameters())
        + [model.pos_embed]
        + list(model.patch_embed.parameters())
    )
    opt = optim.Adam(params, lr=1e-3)
    rng = np.random.default_rng(0)
    N = train_x.shape[0]

    losses = []
    n_steps = 15
    batch_size = 8
    for step in range(n_steps):
        idx = rng.integers(0, N, size=batch_size)
        batch = train_x[idx]
        if batch.shape[-1] == 3:
            batch = batch.transpose(0, 3, 1, 2)
        batch_t = Tensor(batch.astype(np.float32), requires_grad=False)
        opt.zero_grad()
        loss, _ = model(batch_t, n_targets=4, rng=rng)
        loss.backward()
        opt.step()
        decay = ema_decay_schedule(step, n_steps)
        update_target_encoder(model.context_encoder, model.target_encoder, decay)
        losses.append(float(loss.data))

    drop = losses[0] - losses[-1]
    print(f"  loss: first={losses[0]:.4f}, last={losses[-1]:.4f}, drop={drop:.4f}")
    if drop > 0.05:
        return (True, f"loss dropped by {drop:.4f}")
    return (False, f"loss drop {drop:.4f} <= 0.05")


# ---------------------------------------------------------------------------
# Test 4: RSSM Moving MNIST smoke
# ---------------------------------------------------------------------------


def test_rssm_moving_mnist_smoke():
    """RSSM 视频预测，验证 MSE ≤ 0.20。使用较小配置以适配 90s 超时。"""
    from verse_torch import Tensor, optim, no_grad
    from verse_awm import VideoRSSM
    import test_rssm_moving_mnist as rssm_test

    np.random.seed(42)
    # 较小配置以控制时间
    n_train_steps = 30
    pred_len = 5
    ctx_len = 5
    frame_size = 16
    seq_len = ctx_len + pred_len
    n_train = 12
    n_test = 3
    batch_size = 2

    print(f"  config: steps={n_train_steps}, frame={frame_size}, "
          f"seq={seq_len}, train={n_train}, test={n_test}")
    train_seqs = rssm_test.generate_dataset(n_samples=n_train, seq_len=seq_len,
                                             frame_size=frame_size, seed=42)
    test_seqs = rssm_test.generate_dataset(n_samples=n_test, seq_len=seq_len,
                                            frame_size=frame_size, seed=123)

    model = VideoRSSM(
        frame_size=(frame_size, frame_size), in_channels=1,
        deter_dim=64, stoch_dim=8, stoch_classes=8,
        hidden_dim=64,
    )
    opt = optim.Adam(list(model.parameters()), lr=1e-3)
    rng = np.random.default_rng(0)
    N = train_seqs.shape[0]

    for step in range(n_train_steps):
        idx = rng.integers(0, N, size=batch_size)
        batch = train_seqs[idx]
        batch_t = Tensor(batch.astype(np.float32), requires_grad=False)
        opt.zero_grad()
        out = model.forward_frames(batch_t)
        loss = out["loss"]
        loss.backward()
        opt.step()

    # 评估
    mse, _ = rssm_test.evaluate_prediction(model, test_seqs,
                                            ctx_len=ctx_len, pred_len=pred_len)
    print(f"  prediction MSE: {mse:.6f} (threshold 0.20)")
    if mse <= 0.20:
        return (True, f"MSE={mse:.6f}")
    return (False, f"MSE={mse:.6f} > 0.20")


# ---------------------------------------------------------------------------
# Test 5: JEPA demo smoke
# ---------------------------------------------------------------------------


def test_jepa_demo_smoke():
    """JEPA demo 训练 ~30 步，验证最终 loss < 0.1。"""
    from verse_torch import Tensor, optim
    from verse_awm import IJEPA, update_target_encoder, ema_decay_schedule
    import jepa_demo as jd

    np.random.seed(42)
    images = jd.make_color_block_dataset(n_samples=200, img_size=8,
                                          patch_size=4, seed=42)
    print(f"  data: {images.shape}")

    model = IJEPA(
        img_size=8, patch_size=4, in_channels=3,
        embed_dim=32, depth=2, n_heads=2,
        predictor_depth=2,
    )
    params = (
        list(model.context_encoder.parameters())
        + list(model.predictor.parameters())
        + [model.pos_embed]
        + list(model.patch_embed.parameters())
    )
    opt = optim.Adam(params, lr=1e-3)
    rng = np.random.default_rng(0)
    N = images.shape[0]

    n_steps = 30
    batch_size = 16
    losses = []
    for step in range(n_steps):
        idx = rng.integers(0, N, size=batch_size)
        batch = images[idx]
        batch_t = Tensor(batch.astype(np.float32), requires_grad=False)
        opt.zero_grad()
        loss, _ = model(batch_t, n_targets=4, rng=rng)
        loss.backward()
        opt.step()
        decay = ema_decay_schedule(step, n_steps)
        update_target_encoder(model.context_encoder, model.target_encoder, decay)
        losses.append(float(loss.data))

    final_loss = losses[-1]
    print(f"  loss: first={losses[0]:.4f}, last={final_loss:.4f}")
    if final_loss < 0.1:
        return (True, f"final_loss={final_loss:.4f} < 0.1")
    return (False, f"final_loss={final_loss:.4f} >= 0.1")


# ---------------------------------------------------------------------------
# Test 6: CPU inference smoke
# ---------------------------------------------------------------------------


def test_cpu_inference_smoke():
    """CPU 推理生成 token，验证 token 数 ≥ 10。"""
    from verse_torch import Tensor, no_grad
    from verse_nex import HybridLM
    from verse_tokenizer import CharTokenizer
    from verse_inference import ModelLoader, Sampler, StreamingGenerator

    np.random.seed(42)
    # vocab_size 需要 ≥ 提示词字符的 ASCII + 4（CharTokenizer 的 special token 占 0..3）
    # 'H'=72, 'e'=101, 'l'=108, 'o'=111 -> 最大 111，需要 vocab_size ≥ 111+4 = 115
    vocab_size = 128
    dim = 64
    n_layers = 2
    n_heads = 4

    loader = ModelLoader(
        arch="mamba2",
        vocab_size=vocab_size,
        dim=dim,
        n_layers=n_layers,
        ssm_kwargs={"d_state": 32, "d_conv": 4, "expand": 2, "n_heads": n_heads},
        sparse_kwargs={"n_heads": n_heads, "chunk_size": 16,
                       "n_sliding_chunks": 1, "topk_chunks": 1},
    )
    model = loader.load()
    n_params = sum(np.asarray(v).size for v in model.state_dict().values())
    print(f"  model params: {n_params}")

    tokenizer = CharTokenizer()
    # 预填充 vocab：把 chr(0)..chr(vocab_size-5) 全部加入 tokenizer，
    # 这样模型输出的任意 id (4..vocab_size-1) 都能 decode。
    n_prepopulate = max(0, vocab_size - 4)
    for b in range(n_prepopulate):
        tokenizer._ensure_char(chr(b))

    prompt = "Hello"
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    print(f"  prompt: {prompt!r}, ids: {prompt_ids}")
    # 验证所有 id < vocab_size
    prompt_ids = [i for i in prompt_ids if i < vocab_size]

    sampler = Sampler(temperature=0.8, top_k=20, top_p=0.95, seed=42)
    gen = StreamingGenerator(model, tokenizer=tokenizer, sampler=sampler)

    max_new_tokens = 15
    generated = list(gen.generate(prompt_ids, max_new_tokens=max_new_tokens))
    print(f"  generated {len(generated)} tokens: {generated}")
    if len(generated) >= 10:
        return (True, f"generated {len(generated)} tokens")
    return (False, f"only generated {len(generated)} tokens (< 10)")


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------


def main() -> int:
    print("=" * 72)
    print("Task 6.2: Verse Framework End-to-End Tests")
    print("=" * 72)

    # 注册所有测试
    tests = [
        ("test_mnist_mlp_smoke", test_mnist_mlp_smoke, 60),
        ("test_char_lm_smoke", test_char_lm_smoke, 90),
        ("test_ijepa_cifar10_smoke", test_ijepa_cifar10_smoke, 90),
        ("test_rssm_moving_mnist_smoke", test_rssm_moving_mnist_smoke, 120),
        ("test_jepa_demo_smoke", test_jepa_demo_smoke, 90),
        ("test_cpu_inference_smoke", test_cpu_inference_smoke, 60),
    ]

    for name, fn, timeout in tests:
        _run_test(name, fn, timeout)

    # 汇总
    print("\n" + "=" * 72)
    print("Summary:")
    n_pass = sum(1 for _, p, _, _ in _RESULTS if p)
    n_fail = sum(1 for _, p, _, _ in _RESULTS if not p)
    for name, passed, _, elapsed in _RESULTS:
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name:<40s} ({elapsed:.1f}s)")
    print(f"\n  Total: PASS={n_pass}  FAIL={n_fail}")
    print("=" * 72)
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
