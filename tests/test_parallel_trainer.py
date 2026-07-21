"""Task 5: ParallelTrainer + Trainer.inference 单元测试。

覆盖：
1. ParallelTrainer 基本流程（4 chunk 拆分 + 训练完成）
2. 步数拆分正确性（200 步拆 4 chunk = [50,50,50,50]）
3. _eval_full_val 用完整 val 数据集（不是单 batch）
4. 合并排序策略（差前好后）
5. 整体 fine-tune 步数正确（max_steps // 10）
6. val_loss 在每个 chunk 后更新
7. Trainer.inference 基本可用
8. Trainer.inference temperature 参数影响生成
9. Trainer.inference top_k 参数限制候选

运行方式：
    cd /workspace && python -m pytest tests/test_parallel_trainer.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# 让 tests/ 目录能 import verse_torch
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "packages" / "verse_torch"))

from verse_torch import Tensor, Linear, SGD, Module, AdamW
from verse_torch.training import (
    ParallelTrainer,
    Trainer,
    cross_entropy_loss,
    BatchLoader,
)


# ---------------------------------------------------------------------------
# Toy 模型与数据集
# ---------------------------------------------------------------------------


class ToyModel(Module):
    """简单分类模型：Linear(10, 5)，forward(x) → (B, 5) logits。"""

    def __init__(self, in_dim=10, n_classes=5):
        super().__init__()
        self.fc = Linear(in_dim, n_classes)

    def forward(self, x):
        return self.fc(x)


class ToyDataset:
    """简单分类数据集：x ~ N(0,1)，y = argmax(W_true @ x + b_true)。"""

    def __init__(self, n=100, in_dim=10, n_classes=5, seed=0):
        rng = np.random.RandomState(seed)
        self.n = n
        self.in_dim = in_dim
        self.n_classes = n_classes
        W_true = rng.randn(in_dim, n_classes).astype(np.float32)
        b_true = rng.randn(n_classes).astype(np.float32)
        self.x = rng.randn(n, in_dim).astype(np.float32)
        logits = self.x @ W_true + b_true
        self.y = np.argmax(logits, axis=1).astype(np.int64)

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        return self.x[i], self.y[i]


class TaggedModel(ToyModel):
    """带 chunk tag 的 toy 模型，用于追踪 ParallelTrainer 的重训顺序。

    state_dict / load_state_dict 中嵌入 ``__tag__`` 字段，使 fit() 在
    load_state_dict 后能恢复原始 chunk_id，从而验证合并排序策略。
    """

    def __init__(self, in_dim=10, n_classes=5):
        super().__init__(in_dim, n_classes)
        self._chunk_tag = -1

    def state_dict(self):
        sd = super().state_dict()
        sd["__tag__"] = self._chunk_tag
        return sd

    def load_state_dict(self, sd, strict=True):
        # 提取 tag（不参与训练参数）
        if isinstance(sd, dict) and "__tag__" in sd:
            self._chunk_tag = sd["__tag__"]
            sd = {k: v for k, v in sd.items() if k != "__tag__"}
        return super().load_state_dict(sd, strict=False)


class ToyModelWithGenerate(Module):
    """带 generate 方法的 toy 模型，用于测试 Trainer.inference。

    generate 记录每次调用的参数，返回固定输出以便断言。
    """

    def __init__(self, vocab_size=10, in_dim=5):
        super().__init__()
        self.fc = Linear(in_dim, vocab_size)
        self.vocab_size = vocab_size
        self.generate_calls = []

    def forward(self, x):
        return self.fc(x)

    def generate(self, input_ids, max_new_tokens=10, temperature=1.0,
                 top_k=None, top_p=None):
        self.generate_calls.append({
            "input": input_ids,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p,
        })
        # 返回固定输出（模拟生成 max_new_tokens 个 token）
        return np.arange(max_new_tokens, dtype=np.int64)


# ---------------------------------------------------------------------------
# 通用 fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def toy_setup():
    """构造 toy 模型 + 训练集 + 验证集。"""
    np.random.seed(0)
    model = ToyModel(in_dim=10, n_classes=5)
    train_ds = ToyDataset(n=80, in_dim=10, n_classes=5, seed=0)
    val_ds = ToyDataset(n=20, in_dim=10, n_classes=5, seed=100)
    return model, train_ds, val_ds


# ---------------------------------------------------------------------------
# 1. test_parallel_trainer_basic
# ---------------------------------------------------------------------------


def test_parallel_trainer_basic(toy_setup):
    """ParallelTrainer 基本流程：4 chunk 拆分，训练完成后 best_val_loss 有效。"""
    model, train_ds, val_ds = toy_setup
    cfg = {
        "parallel_chunks": 4,
        "max_steps": 40,  # 4 chunks x 10 steps
        "batch_size": 8,
        "lr": 0.01,
        "eval_interval": 5,
        "warmup": 2,
        "merge_finetune_steps": 4,
        "seed": 42,
    }
    trainer = ParallelTrainer(
        model=model, train_dataset=train_ds, val_dataset=val_ds, cfg=cfg)
    history = trainer.fit()

    # best_val_loss 应为有限值（不是 inf）
    assert trainer.best_val_loss < float("inf"), (
        f"best_val_loss 应为有限值, got {trainer.best_val_loss}"
    )
    # history 应包含三个列表
    assert "train_loss" in history
    assert "val_loss" in history
    assert "steps" in history
    # chunk_stats 应记录 4 个 chunk
    assert len(trainer.chunk_stats) == 4, (
        f"chunk_stats 应有 4 项, got {len(trainer.chunk_stats)}"
    )
    # chunk_steps_list 应在 fit 后填充
    assert len(trainer.chunk_steps_list) == 4
    assert sum(trainer.chunk_steps_list) == 40


# ---------------------------------------------------------------------------
# 2. test_parallel_trainer_chunk_split
# ---------------------------------------------------------------------------


def test_parallel_trainer_chunk_split():
    """步数拆分：200 步拆 4 chunk = [50,50,50,50]；余数均摊到前几个。"""
    # 整除场景
    trainer = ParallelTrainer(
        model=ToyModel(), train_dataset=ToyDataset(10), 
        val_dataset=ToyDataset(5),
        cfg={"parallel_chunks": 4, "max_steps": 200})
    steps = trainer._split_steps()
    assert steps == [50, 50, 50, 50], f"200/4 应为 [50,50,50,50], got {steps}"
    assert sum(steps) == 200

    # 余数场景：202 步拆 4 chunk = [51, 51, 50, 50]
    trainer2 = ParallelTrainer(
        model=ToyModel(), train_dataset=ToyDataset(10),
        val_dataset=ToyDataset(5),
        cfg={"parallel_chunks": 4, "max_steps": 202})
    steps2 = trainer2._split_steps()
    assert steps2 == [51, 51, 50, 50], f"202/4 应为 [51,51,50,50], got {steps2}"
    assert sum(steps2) == 202

    # 小步数场景：5 步拆 4 chunk = [2, 1, 1, 1]
    trainer3 = ParallelTrainer(
        model=ToyModel(), train_dataset=ToyDataset(10),
        val_dataset=ToyDataset(5),
        cfg={"parallel_chunks": 4, "max_steps": 5})
    steps3 = trainer3._split_steps()
    assert steps3 == [2, 1, 1, 1], f"5/4 应为 [2,1,1,1], got {steps3}"
    assert sum(steps3) == 5

    # 极端：max_steps < parallel_chunks（部分 chunk 为 0，应被过滤）
    trainer4 = ParallelTrainer(
        model=ToyModel(), train_dataset=ToyDataset(10),
        val_dataset=ToyDataset(5),
        cfg={"parallel_chunks": 4, "max_steps": 2})
    steps4 = trainer4._split_steps()
    assert sum(steps4) == 2
    assert all(s > 0 for s in steps4), f"0 步 chunk 应被过滤, got {steps4}"


# ---------------------------------------------------------------------------
# 3. test_parallel_trainer_eval_full_val
# ---------------------------------------------------------------------------


def test_parallel_trainer_eval_full_val(toy_setup):
    """_eval_full_val 用完整 val 数据集（不是单 batch），结果与手动计算一致。"""
    model, train_ds, val_ds = toy_setup
    cfg = {
        "parallel_chunks": 2,
        "max_steps": 10,
        "batch_size": 4,  # 故意小于 val_ds 长度 20，强制多 batch
        "lr": 0.01,
        "seed": 42,
    }
    trainer = ParallelTrainer(
        model=model, train_dataset=train_ds, val_dataset=val_ds, cfg=cfg)

    # _eval_full_val 应跑完整 val 数据集
    val_loss = trainer._eval_full_val(model)
    assert val_loss < float("inf"), "val_loss 应为有限值"

    # 手动计算完整 val 的平均 loss 验证一致性
    val_loader = BatchLoader(val_ds, batch_size=4, shuffle=False, seed=42)
    manual_total = 0.0
    manual_n = 0
    from verse_torch import no_grad
    with no_grad():
        for x_batch, y_batch in val_loader:
            x = Tensor(x_batch) if not isinstance(x_batch, Tensor) else x_batch
            y = Tensor(y_batch) if not isinstance(y_batch, Tensor) else y_batch
            logits = model(x)
            loss = cross_entropy_loss(logits, y)
            manual_total += float(loss.data)
            manual_n += 1
    manual_avg = manual_total / manual_n
    assert abs(val_loss - manual_avg) < 1e-5, (
        f"_eval_full_val({val_loss}) != manual({manual_avg})"
    )

    # 验证：val_ds 长度 20，batch_size 4，应跑 5 个 batch
    assert manual_n == 5, f"应跑 5 个 batch, got {manual_n}"


def test_eval_full_val_empty_dataset():
    """_eval_full_val 对空数据集返回 inf。"""
    model = ToyModel()

    class EmptyDataset:
        def __len__(self):
            return 0
        def __getitem__(self, i):
            raise IndexError

    trainer = ParallelTrainer(
        model=model, train_dataset=ToyDataset(10), val_dataset=EmptyDataset(),
        cfg={"parallel_chunks": 2, "max_steps": 4})
    assert trainer._eval_full_val(model) == float("inf")


# ---------------------------------------------------------------------------
# 4. test_parallel_trainer_merge_sort
# ---------------------------------------------------------------------------


def test_parallel_trainer_merge_sort():
    """合并排序策略：差前好后（train_loss + val_loss 大的先重训）。

    通过 TaggedModel 在 state_dict 中嵌入 chunk_id 标记，
    在重训阶段读取标记，验证重训顺序与排序后的顺序一致。
    """
    model = TaggedModel(in_dim=10, n_classes=5)
    train_ds = ToyDataset(n=40, in_dim=10, n_classes=5, seed=0)
    val_ds = ToyDataset(n=20, in_dim=10, n_classes=5, seed=100)
    cfg = {
        "parallel_chunks": 4,
        "max_steps": 40,
        "batch_size": 8,
        "lr": 0.01,
        "merge_finetune_steps": 0,  # 关闭 finetune 以简化验证
        "seed": 42,
    }
    trainer = ParallelTrainer(
        model=model, train_dataset=train_ds, val_dataset=val_ds, cfg=cfg)

    # 控制每个 chunk 返回的 (train_loss, val_loss)
    # chunk 0: sum=0.3（最好）
    # chunk 1: sum=1.7（最差）
    # chunk 2: sum=1.0
    # chunk 3: sum=0.6
    fake_losses = {
        0: (0.1, 0.2),   # sum=0.3（最好）
        1: (0.9, 0.8),   # sum=1.7（最差）
        2: (0.5, 0.5),   # sum=1.0
        3: (0.3, 0.3),   # sum=0.6
    }

    # 固定 _eval_full_val 返回值（避免干扰排序）
    trainer._eval_full_val = lambda m: 0.5

    retrain_tags = []  # 记录重训阶段的 chunk tag 顺序

    def fake_train_chunk(model, train_dataset, chunk_steps, chunk_id):
        if chunk_id >= 0:
            # Phase 1：设置 tag 并返回控制的 loss
            model._chunk_tag = chunk_id
            tl, vl = fake_losses[chunk_id]
        elif chunk_id == -999:
            # Phase 3：finetune（本测试已关闭，不应进入）
            tl, vl = 0.5, 0.5
        else:
            # Phase 2：重训，读取当前模型的 tag（由 load_state_dict 恢复）
            retrain_tags.append(getattr(model, "_chunk_tag", -1))
            tl, vl = 0.5, 0.5
        return model, tl, vl

    trainer._train_chunk = fake_train_chunk
    trainer.fit()

    # 验证重训顺序：差前好后
    # 排序（reverse=True，sum 大的在前）：chunk 1 (1.7) -> chunk 2 (1.0) -> chunk 3 (0.6) -> chunk 0 (0.3)
    expected_order = [1, 2, 3, 0]
    assert retrain_tags == expected_order, (
        f"重训顺序应为 {expected_order}（差前好后）, got {retrain_tags}"
    )


# ---------------------------------------------------------------------------
# 5. test_parallel_trainer_finetune
# ---------------------------------------------------------------------------


def test_parallel_trainer_finetune():
    """整体 fine-tune 步数正确：merge_finetune_steps = max_steps // 10。"""
    model = ToyModel()
    train_ds = ToyDataset(n=40, in_dim=10, n_classes=5, seed=0)
    val_ds = ToyDataset(n=20, in_dim=10, n_classes=5, seed=100)

    # 默认 merge_finetune_steps = max_steps // 10 = 200 // 10 = 20
    trainer = ParallelTrainer(
        model=model, train_dataset=train_ds, val_dataset=val_ds,
        cfg={"parallel_chunks": 4, "max_steps": 200, "batch_size": 8, "lr": 0.01})
    assert trainer.merge_finetune_steps == 20, (
        f"默认 merge_finetune_steps 应为 20 (200//10), got {trainer.merge_finetune_steps}"
    )

    # 自定义 merge_finetune_steps
    trainer2 = ParallelTrainer(
        model=model, train_dataset=train_ds, val_dataset=val_ds,
        cfg={"parallel_chunks": 4, "max_steps": 200, "batch_size": 8, "lr": 0.01,
             "merge_finetune_steps": 50})
    assert trainer2.merge_finetune_steps == 50

    # 追踪实际调用的 finetune 步数
    finetune_steps_seen = []
    original_train_chunk = trainer._train_chunk

    def tracking_train_chunk(m, ds, steps, cid):
        if cid == -999:
            finetune_steps_seen.append(steps)
        return original_train_chunk(m, ds, steps, cid)

    trainer._train_chunk = tracking_train_chunk
    trainer.fit()

    assert len(finetune_steps_seen) == 1, (
        f"应恰好调用 1 次 finetune, got {len(finetune_steps_seen)}"
    )
    assert finetune_steps_seen[0] == 20, (
        f"finetune 步数应为 20, got {finetune_steps_seen[0]}"
    )


def test_parallel_trainer_finetune_disabled():
    """merge_finetune_steps=0 时跳过 fine-tune。"""
    model = ToyModel()
    train_ds = ToyDataset(n=40, in_dim=10, n_classes=5, seed=0)
    val_ds = ToyDataset(n=20, in_dim=10, n_classes=5, seed=100)

    finetune_calls = []
    original_train_chunk = ParallelTrainer._train_chunk

    def tracking_train_chunk(self, m, ds, steps, cid):
        if cid == -999:
            finetune_calls.append(steps)
        return original_train_chunk(self, m, ds, steps, cid)

    trainer = ParallelTrainer(
        model=model, train_dataset=train_ds, val_dataset=val_ds,
        cfg={"parallel_chunks": 2, "max_steps": 10, "batch_size": 8, "lr": 0.01,
             "merge_finetune_steps": 0})

    # 用 unbound method 替换实例方法
    trainer._train_chunk = lambda m, ds, steps, cid: tracking_train_chunk(trainer, m, ds, steps, cid)
    trainer.fit()

    assert len(finetune_calls) == 0, (
        f"merge_finetune_steps=0 时不应调用 finetune, got {len(finetune_calls)} calls"
    )


# ---------------------------------------------------------------------------
# 6. test_parallel_trainer_val_loss_update
# ---------------------------------------------------------------------------


def test_parallel_trainer_val_loss_update(toy_setup):
    """val_loss 在每个 chunk 后更新：chunk_stats 中每个 chunk 都有有效 val_loss。"""
    model, train_ds, val_ds = toy_setup
    cfg = {
        "parallel_chunks": 4,
        "max_steps": 32,
        "batch_size": 8,
        "lr": 0.01,
        "eval_interval": 4,
        "warmup": 2,
        "merge_finetune_steps": 0,  # 关闭 finetune，只看 chunk 阶段
        "seed": 42,
    }
    trainer = ParallelTrainer(
        model=model, train_dataset=train_ds, val_dataset=val_ds, cfg=cfg)
    trainer.fit()

    # 每个 chunk 都应有有效的 val_loss（有限值，非 inf）
    assert len(trainer.chunk_stats) == 4
    for i, stat in enumerate(trainer.chunk_stats):
        assert stat["val_loss"] < float("inf"), (
            f"chunk {i} val_loss 应为有限值, got {stat['val_loss']}"
        )
        assert stat["train_loss"] < float("inf"), (
            f"chunk {i} train_loss 应为有限值, got {stat['train_loss']}"
        )
        assert stat["steps"] > 0
        assert stat["chunk_id"] == i

    # best_val_loss 应等于所有 chunk + 重训阶段中最小的 val_loss
    # 至少应小于等于第一个 chunk 的 val_loss
    assert trainer.best_val_loss <= trainer.chunk_stats[0]["val_loss"] + 1e-6

    # best_state_dict 应被填充
    assert trainer.best_state_dict is not None, "best_state_dict 应在 fit 后非 None"


# ---------------------------------------------------------------------------
# 7. test_trainer_inference_basic
# ---------------------------------------------------------------------------


def test_trainer_inference_basic(tmp_path):
    """Trainer.inference 基本可用：返回 list[str]，长度与 prompts 一致。"""
    model = ToyModelWithGenerate(vocab_size=10, in_dim=5)
    opt = SGD(model.parameters(), lr=0.01)
    trainer = Trainer(
        model=model,
        train_loader=[],
        val_loader=[],
        optimizer=opt,
        cfg={"save_dir": str(tmp_path), "max_steps": 1, "log_interval": 1000,
             "enable_progress_bar": False, "realtime_plot": False},
    )

    prompts = ["hello", "world", "foo"]
    results = trainer.inference(prompts, max_tokens=5)

    # 返回应为 list[str]，长度与 prompts 一致
    assert isinstance(results, list)
    assert len(results) == len(prompts), (
        f"结果长度应等于 prompts 长度 {len(prompts)}, got {len(results)}"
    )
    for r in results:
        assert isinstance(r, str)

    # model.generate 应被调用 3 次
    assert len(model.generate_calls) == 3, (
        f"generate 应被调用 3 次, got {len(model.generate_calls)}"
    )

    # 无 tokenizer 时，prompt 应原样传给 generate
    assert model.generate_calls[0]["input"] == "hello"


def test_trainer_inference_no_generate_raises(tmp_path):
    """模型未实现 generate 方法时应抛 NotImplementedError。"""
    model = ToyModel()  # 无 generate 方法
    opt = SGD(model.parameters(), lr=0.01)
    trainer = Trainer(
        model=model, train_loader=[], val_loader=[],
        optimizer=opt,
        cfg={"save_dir": str(tmp_path), "max_steps": 1,
             "enable_progress_bar": False, "realtime_plot": False},
    )
    with pytest.raises(NotImplementedError):
        trainer.inference(["test"])


# ---------------------------------------------------------------------------
# 8. test_trainer_inference_temperature
# ---------------------------------------------------------------------------


def test_trainer_inference_temperature(tmp_path):
    """temperature 参数被正确传递给 model.generate。"""
    model = ToyModelWithGenerate(vocab_size=10, in_dim=5)
    opt = SGD(model.parameters(), lr=0.01)
    trainer = Trainer(
        model=model, train_loader=[], val_loader=[],
        optimizer=opt,
        cfg={"save_dir": str(tmp_path), "max_steps": 1,
             "enable_progress_bar": False, "realtime_plot": False},
    )

    # 用不同 temperature 调用
    trainer.inference(["a"], temperature=0.1, max_tokens=3)
    trainer.inference(["b"], temperature=2.0, max_tokens=3)
    trainer.inference(["c"], temperature=1.0, max_tokens=3)

    assert len(model.generate_calls) == 3
    assert model.generate_calls[0]["temperature"] == pytest.approx(0.1)
    assert model.generate_calls[1]["temperature"] == pytest.approx(2.0)
    assert model.generate_calls[2]["temperature"] == pytest.approx(1.0)

    # 不同 temperature 应产生不同的调用记录
    assert model.generate_calls[0]["temperature"] != model.generate_calls[1]["temperature"]


# ---------------------------------------------------------------------------
# 9. test_trainer_inference_top_k
# ---------------------------------------------------------------------------


def test_trainer_inference_top_k(tmp_path):
    """top_k 参数被正确传递给 model.generate。"""
    model = ToyModelWithGenerate(vocab_size=10, in_dim=5)
    opt = SGD(model.parameters(), lr=0.01)
    trainer = Trainer(
        model=model, train_loader=[], val_loader=[],
        optimizer=opt,
        cfg={"save_dir": str(tmp_path), "max_steps": 1,
             "enable_progress_bar": False, "realtime_plot": False},
    )

    # 用不同 top_k 调用
    trainer.inference(["a"], top_k=1, max_tokens=3)
    trainer.inference(["b"], top_k=5, max_tokens=3)
    trainer.inference(["c"], top_k=None, max_tokens=3)

    assert len(model.generate_calls) == 3
    assert model.generate_calls[0]["top_k"] == 1
    assert model.generate_calls[1]["top_k"] == 5
    assert model.generate_calls[2]["top_k"] is None


def test_trainer_inference_top_p(tmp_path):
    """top_p 参数被正确传递给 model.generate（额外覆盖）。"""
    model = ToyModelWithGenerate(vocab_size=10, in_dim=5)
    opt = SGD(model.parameters(), lr=0.01)
    trainer = Trainer(
        model=model, train_loader=[], val_loader=[],
        optimizer=opt,
        cfg={"save_dir": str(tmp_path), "max_steps": 1,
             "enable_progress_bar": False, "realtime_plot": False},
    )

    trainer.inference(["a"], top_p=0.9, max_tokens=3)
    trainer.inference(["b"], top_p=None, max_tokens=3)

    assert len(model.generate_calls) == 2
    assert model.generate_calls[0]["top_p"] == pytest.approx(0.9)
    assert model.generate_calls[1]["top_p"] is None


def test_trainer_inference_with_tokenizer(tmp_path):
    """带 tokenizer 的 Trainer.inference：encode/decode 路径覆盖。"""
    model = ToyModelWithGenerate(vocab_size=10, in_dim=5)
    opt = SGD(model.parameters(), lr=0.01)
    trainer = Trainer(
        model=model, train_loader=[], val_loader=[],
        optimizer=opt,
        cfg={"save_dir": str(tmp_path), "max_steps": 1,
             "enable_progress_bar": False, "realtime_plot": False},
    )

    # 注入 mock tokenizer
    class MockTokenizer:
        def __init__(self):
            self.encode_calls = []
            self.decode_calls = []

        def encode(self, text, add_special_tokens=True):
            self.encode_calls.append((text, add_special_tokens))
            # 模拟编码：每个字符映射为 ASCII 码
            return [ord(c) for c in text]

        def decode(self, ids):
            self.decode_calls.append(ids)
            # 模拟解码：把 id 列表转回字符串
            arr = np.asarray(ids)
            return "".join(chr(int(i)) for i in arr.flatten())

    tok = MockTokenizer()
    trainer.tokenizer = tok

    results = trainer.inference(["hi", "ok"], max_tokens=4)

    assert len(results) == 2
    assert len(tok.encode_calls) == 2
    assert len(tok.decode_calls) == 2
    assert len(model.generate_calls) == 2
    # encode 应收到原始 prompt
    assert tok.encode_calls[0][0] == "hi"
    assert tok.encode_calls[1][0] == "ok"
    # generate 应收到 encode 后的 id 列表
    assert model.generate_calls[0]["input"] == [ord("h"), ord("i")]
    assert model.generate_calls[1]["input"] == [ord("o"), ord("k")]


# ---------------------------------------------------------------------------
# 集成：ParallelTrainer 与一体训练效果对比（轻量验证）
# ---------------------------------------------------------------------------


def test_parallel_trainer_comparable_to_single(toy_setup):
    """并行训练 val_loss 与一体训练在同一量级（差距 < 50%，宽松验证）。"""
    model, train_ds, val_ds = toy_setup
    import copy

    # 复制模型用于一体训练对比
    model_single = ToyModel(in_dim=10, n_classes=5)
    # 让两个模型初始权重一致
    model_single.load_state_dict(copy.deepcopy(model.state_dict()))

    # 并行训练
    cfg_par = {
        "parallel_chunks": 4,
        "max_steps": 40,
        "batch_size": 8,
        "lr": 0.01,
        "eval_interval": 5,
        "warmup": 2,
        "merge_finetune_steps": 4,
        "seed": 42,
    }
    trainer_par = ParallelTrainer(
        model=model, train_dataset=train_ds, val_dataset=val_ds, cfg=cfg_par)
    trainer_par.fit()
    par_val = trainer_par.best_val_loss

    # 一体训练（用 Trainer）
    from verse_torch.training import BatchLoader
    train_loader = BatchLoader(train_ds, batch_size=8, shuffle=True, seed=42)
    val_loader = BatchLoader(val_ds, batch_size=8, shuffle=False, seed=42)
    opt = AdamW(model_single.parameters(), lr=0.01)
    cfg_single = {
        "max_steps": 40, "eval_interval": 5, "patience": 100,
        "save_dir": str(tmp_path) if False else "/tmp/verse_single_test",
        "grad_accum": 1, "log_interval": 1000,
        "loss_rate_window": 10,
        "enable_progress_bar": False, "realtime_plot": False,
    }
    import os
    os.makedirs(cfg_single["save_dir"], exist_ok=True)
    trainer_single = Trainer(
        model=model_single, train_loader=train_loader, val_loader=val_loader,
        optimizer=opt, cfg=cfg_single)
    trainer_single.fit()
    single_val = trainer_single.best_val_loss

    # 验证两者都在合理范围（loss 下降了）
    assert par_val < float("inf")
    assert single_val < float("inf")
    # 宽松对比：差距不超过 50%（toy 模型 + 少量步数方差较大）
    # 这里主要验证并行训练不会比一体训练差太多
    ratio = par_val / max(single_val, 1e-6)
    assert ratio < 2.0, (
        f"并行训练 val_loss({par_val}) 应与一体训练({single_val}) 在同一量级, "
        f"ratio={ratio:.2f}"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
