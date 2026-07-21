"""Part4 P3.3 集成测试：CometSparkLM + VerseNex 架构端到端。

验证：
1. arch="versenex" 可正确构造 CometSparkLM
2. forward(idx) 返回 dict（含 logits）
3. forward(idx, targets=y) 返回 dict（含 total_loss）
4. loss.backward() 不崩溃（验证 tensor.py 中的 None guard 修复）
5. generate(idx, max_new_tokens) 返回 ndarray
6. ParallelTrainer 集成（小规模）
"""

import os
import sys
import time

# 设置 sys.path（packages 已 editable 安装，无需手动加入）
sys.path.insert(0, "/workspace/data/demo")

# 限制线程避免内存爆炸
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["NUMEXPR_NUM_THREADS"] = "2"

import numpy as np
from verse_torch import Tensor, no_grad

from model.config import CometSparkConfig
from model.model import CometSparkLM


def test_versenex_basic():
    """测试 1：基本构造 + forward + targets + backward + generate"""
    print("\n=== 测试 1：CometSparkLM(arch=versenex) 基本功能 ===", flush=True)
    # 极小配置：2 层 + d=64 + 2 parts × 2 experts
    config = CometSparkConfig(
        arch="versenex",
        vocab_size=1000,
        n_layer=2,
        n_head=4,
        n_embd=64,
        d_model=64,
        attn_top_k=16,
        mod_d_ff=128,
        mod_n_parts=2,
        mod_n_experts=2,
        mod_top_k_parts=1,
        mod_top_k_experts=1,
        medusa_n_heads=0,  # 关闭副头避免占用内存
        tie_weights=True,
        seq_len=32,
        max_position_embeddings=128,
    )
    t0 = time.time()
    model = CometSparkLM(config)
    print(f"[OK] 构造模型用时 {time.time()-t0:.2f}s", flush=True)

    # forward(idx) 不带 targets，应返回 dict（含 logits）
    idx = Tensor(np.random.randint(0, 1000, size=(2, 16), dtype=np.int64))
    t0 = time.time()
    out = model(idx)
    print(f"[OK] forward(idx) 用时 {time.time()-t0:.2f}s, 返回类型: {type(out).__name__}", flush=True)
    assert isinstance(out, dict), f"期望 dict，得到 {type(out)}"
    assert "logits" in out, "dict 应含 logits"
    print(f"     logits shape = {out['logits'].data.shape}", flush=True)

    # forward(idx, targets=y) 应返回 dict（含 total_loss）
    y = Tensor(np.random.randint(0, 1000, size=(2, 16), dtype=np.int64))
    t0 = time.time()
    out2 = model(idx, targets=y)
    print(f"[OK] forward(idx, targets=y) 用时 {time.time()-t0:.2f}s", flush=True)
    assert "total_loss" in out2, "dict 应含 total_loss"
    loss = out2["total_loss"]
    print(f"     total_loss = {float(loss.data):.4f}", flush=True)
    print(f"     aux_loss = {float(out2['aux_loss'].data) if out2['aux_loss'] is not None else 'None'}", flush=True)

    # backward：验证稀疏激活下 None guard 是否生效
    t0 = time.time()
    loss.backward()
    print(f"[OK] backward 用时 {time.time()-t0:.2f}s", flush=True)

    # 检查梯度（统计有梯度的参数比例）
    n_with_grad = 0
    n_total = 0
    for p in model.parameters():
        n_total += 1
        if p.grad is not None and np.any(p.grad != 0):
            n_with_grad += 1
    print(f"     梯度统计：{n_with_grad}/{n_total} 参数有非零梯度", flush=True)

    # generate
    t0 = time.time()
    prompt = np.array([[1, 2, 3, 4, 5]], dtype=np.int64)
    out_np = model.generate(prompt, max_new_tokens=4, temperature=1.0, top_k=None)
    print(f"[OK] generate 用时 {time.time()-t0:.2f}s, 输出 shape = {out_np.shape}", flush=True)
    assert out_np.shape == (1, 5 + 4), f"期望 (1, 9), 得到 {out_np.shape}"

    # generate 带 temperature
    out_np2 = model.generate(prompt, max_new_tokens=4, temperature=0.7, top_k=10)
    print(f"[OK] generate(temperature=0.7, top_k=10) shape = {out_np2.shape}", flush=True)

    print("=== 测试 1 通过 ===\n", flush=True)
    return model


def test_trainer_versenex():
    """测试 2：Trainer + VerseNex 集成（小规模训练）"""
    print("\n=== 测试 2：Trainer + VerseNex 集成 ===", flush=True)
    from verse_torch.optim import AdamW
    from verse_torch.training import Trainer
    from src.data_loader import TextDataset, collate_fn, BatchLoader

    # 假 tokenizer：直接 id 映射
    class DummyTok:
        def __init__(self, vocab_size=200):
            self.vocab_size = vocab_size
        def __len__(self):
            return self.vocab_size
        def encode(self, text, add_special_tokens=False):
            # 简单按字符编码到 [0, vocab_size)
            return [ord(c) % (self.vocab_size - 10) for c in text]
        def apply_prompt_template(self, prompt):
            return self.encode(prompt)

    tok = DummyTok(vocab_size=200)

    # 写入临时训练数据
    import tempfile, json
    tmp_dir = tempfile.mkdtemp(prefix="versenex_trainer_test_")
    train_path = os.path.join(tmp_dir, "train.jsonl")
    val_path = os.path.join(tmp_dir, "val.jsonl")
    with open(train_path, "w", encoding="utf-8") as f:
        for i in range(20):
            f.write(json.dumps({"prompt": f"问题 {i}", "completion": f"答案 {i}"}) + "\n")
    with open(val_path, "w", encoding="utf-8") as f:
        for i in range(5):
            f.write(json.dumps({"prompt": f"验证 {i}", "completion": f"验证答案 {i}"}) + "\n")

    seq_len = 16
    train_ds = TextDataset(tok, train_path, seq_len=seq_len)
    val_ds = TextDataset(tok, val_path, seq_len=seq_len)
    print(f"[OK] train_ds={len(train_ds)} val_ds={len(val_ds)}", flush=True)

    train_loader = BatchLoader(train_ds, batch_size=4, shuffle=True, collate_fn=collate_fn, seed=42)
    val_loader = BatchLoader(val_ds, batch_size=4, shuffle=False, collate_fn=collate_fn, seed=42)

    config = CometSparkConfig(
        arch="versenex",
        vocab_size=200,
        n_layer=2,
        n_head=4,
        n_embd=32,
        d_model=32,
        attn_top_k=8,
        mod_d_ff=64,
        mod_n_parts=2,
        mod_n_experts=2,
        mod_top_k_parts=1,
        mod_top_k_experts=1,
        medusa_n_heads=0,
        tie_weights=True,
        seq_len=seq_len,
        max_position_embeddings=64,
    )
    model = CometSparkLM(config)
    optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)

    cfg = {
        "max_steps": 6,
        "eval_interval": 3,
        "patience": 100,
        "save_dir": tmp_dir,
        "grad_accum": 1,
        "log_interval": 2,
        "grad_clip": 1.0,
        "label_smoothing": 0.1,
        "enable_progress_bar": False,
        "realtime_plot": False,
        "eta_window": 5,
    }
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=None,
        cfg=cfg,
    )
    t0 = time.time()
    train_losses, val_losses = trainer.fit()
    print(f"[OK] Trainer.fit 用时 {time.time()-t0:.2f}s", flush=True)
    print(f"     train_losses = {train_losses}", flush=True)
    print(f"     val_losses   = {val_losses}", flush=True)
    assert len(train_losses) == 6, f"期望 6 个 train_loss，得到 {len(train_losses)}"
    print("=== 测试 2 通过 ===\n", flush=True)


def test_parallel_trainer_versenex():
    """测试 3：ParallelTrainer + VerseNex 集成"""
    print("\n=== 测试 3：ParallelTrainer + VerseNex 集成 ===", flush=True)
    from verse_torch.optim import AdamW
    from verse_torch.training import ParallelTrainer, CheckpointManager
    from src.data_loader import TextDataset, collate_fn

    class DummyTok:
        def __init__(self, vocab_size=200):
            self.vocab_size = vocab_size
        def __len__(self):
            return self.vocab_size
        def encode(self, text, add_special_tokens=False):
            return [ord(c) % (self.vocab_size - 10) for c in text]
        def apply_prompt_template(self, prompt):
            return self.encode(prompt)

    tok = DummyTok(vocab_size=200)

    import tempfile, json
    tmp_dir = tempfile.mkdtemp(prefix="versenex_parallel_test_")
    train_path = os.path.join(tmp_dir, "train.jsonl")
    val_path = os.path.join(tmp_dir, "val.jsonl")
    with open(train_path, "w", encoding="utf-8") as f:
        for i in range(20):
            f.write(json.dumps({"prompt": f"问题 {i}", "completion": f"答案 {i}"}) + "\n")
    with open(val_path, "w", encoding="utf-8") as f:
        for i in range(5):
            f.write(json.dumps({"prompt": f"验证 {i}", "completion": f"验证答案 {i}"}) + "\n")

    seq_len = 16
    train_ds = TextDataset(tok, train_path, seq_len=seq_len)
    val_ds = TextDataset(tok, val_path, seq_len=seq_len)

    config = CometSparkConfig(
        arch="versenex",
        vocab_size=200,
        n_layer=2,
        n_head=4,
        n_embd=32,
        d_model=32,
        attn_top_k=8,
        mod_d_ff=64,
        mod_n_parts=2,
        mod_n_experts=2,
        mod_top_k_parts=1,
        mod_top_k_experts=1,
        medusa_n_heads=0,
        tie_weights=True,
        seq_len=seq_len,
        max_position_embeddings=64,
    )
    model = CometSparkLM(config)

    cfg = {
        "parallel_chunks": 2,
        "max_steps": 6,
        "batch_size": 4,
        "lr": 3e-3,
        "warmup": 2,
        "eval_interval": 3,
        "grad_clip": 1.0,
        "label_smoothing": 0.1,
        "seed": 42,
        "patience": 100,
        "save_dir": tmp_dir,
        "log_interval": 1000,
        "enable_progress_bar": False,
        "realtime_plot": False,
        "eta_window": 5,
    }
    ckpt = CheckpointManager(tmp_dir)
    trainer = ParallelTrainer(
        model=model,
        train_dataset=train_ds,
        val_dataset=val_ds,
        optimizer_cls=AdamW,
        optimizer_kwargs={"weight_decay": 0.01},
        cfg=cfg,
        collate_fn=collate_fn,
        checkpoint_mgr=ckpt,
    )
    t0 = time.time()
    history = trainer.fit()
    print(f"[OK] ParallelTrainer.fit 用时 {time.time()-t0:.2f}s", flush=True)
    print(f"     history = {history}", flush=True)
    print("=== 测试 3 通过 ===\n", flush=True)


if __name__ == "__main__":
    test_versenex_basic()
    test_trainer_versenex()
    test_parallel_trainer_versenex()
    print("\n所有 P3.3 集成测试通过 ✓", flush=True)
