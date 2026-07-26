"""tests/test_resume_manager.py

Part5K1.3 Task 6.8: ResumeManager 断点续训测试。

覆盖 SubTask 6.8 要求的测试要点：
1. 断点续训全流程：ResumeManager.save 写 .vn + ResumeManager.load 读取 +
   ResumeManager.apply 应用到 trainer（model + optimizer + step + rng +
   best_val_loss + epoch + patience_count 全字段一致）
2. v1 .vn 向后兼容：手工构造 v1 .vn 文件，ResumeManager.load 缺失字段返回 None
3. optimizer state 恢复：AdamW m/v 矩阵数值一致（float32 吻合 1e-7）
4. rng_state 恢复：numpy RandomState.get_state() 往返一致
5. .pt 向后兼容：旧 ParallelTrainerSafe._save_resume_state 写出的 .pt pickle
   可被新 _load_resume_state 兼容读取
6. ParallelTrainerSafe 集成：_save_resume_state 写 .vn + _load_resume_state 读 .vn
7. .vn 不存在时回退到 .pt：_load_resume_state 优先 .vn，回退 .pt
8. ResumeState namedtuple 字段定义

运行方式：
    cd /workspace && python -m pytest tests/test_resume_manager.py -v
"""

from __future__ import annotations

import io
import json
import os
import pickle
import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import pytest

# 让 tests/ 目录能 import verse_torch / verse_infra
_REPO_ROOT = Path(__file__).resolve().parent.parent
for _pkg in ("verse_torch", "verse_nex", "verse_infra"):
    _p = _REPO_ROOT / "packages" / _pkg
    if _p.is_dir() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from verse_torch import Linear, Module, SGD, AdamW, Tensor  # noqa: E402
from verse_torch.training import (  # noqa: E402
    CheckpointManager,
    ResumeManager,
    ResumeState,
    ParallelTrainer,
)


# ---------------------------------------------------------------------------
# Toy 模型与数据集（对齐 test_parallel_trainer.py 模式）
# ---------------------------------------------------------------------------


class ToyModel(Module):
    """简单分类模型：Linear(10, 5)，forward(x) → (B, 5) logits。"""

    def __init__(self, in_dim=10, n_classes=5):
        super().__init__()
        self.fc = Linear(in_dim, n_classes)

    def forward(self, x):
        return self.fc(x)


class ToyDataset:
    """简单分类数据集。"""

    def __init__(self, n=40, in_dim=10, n_classes=5, seed=0):
        rng = np.random.RandomState(seed)
        self.n = n
        self.x = rng.randn(n, in_dim).astype(np.float32)
        self.y = rng.randint(0, n_classes, size=n).astype(np.int64)

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        return self.x[i], self.y[i]


# ---------------------------------------------------------------------------
# 辅助：手工构造 v1 .vn 文件（模拟 Part5K1 写出的 v1 格式）
# ---------------------------------------------------------------------------


def _state_dict_to_npz_bytes(state_dict: dict) -> bytes:
    """把 state_dict 序列化为 npz 字节流（与 vn_format._npz_to_bytes 一致）。"""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as zf:
        for name, arr in state_dict.items():
            arr = np.ascontiguousarray(arr)
            npy_buf = io.BytesIO()
            np.lib.format.write_array(npy_buf, arr, allow_pickle=False)
            zf.writestr(f"{name}.npy", npy_buf.getvalue())
    return buf.getvalue()


def _write_v1_vn_file(vn_path: str, state_dict: dict):
    """手工构造 v1 .vn 文件（无 vn_format_version 字段，无 training_state 等）。"""
    npz_buf = _state_dict_to_npz_bytes(state_dict)
    meta = {
        "arch": "versenex",
        "weight_format": "npz",
        "compression_info": None,
        "created_at": "2024-01-01T00:00:00",
        "weight_count": len(state_dict),
    }
    with zipfile.ZipFile(vn_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("model.npz", npz_buf)
        zf.writestr("meta.json", json.dumps(meta, ensure_ascii=False, indent=2))
        zf.writestr("config.yml", "arch: versenex\nn_layer: 2\n")


# ---------------------------------------------------------------------------
# 公共 fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_ckpt_dir():
    """临时 checkpoint 目录。"""
    with tempfile.TemporaryDirectory() as d:
        yield d


@pytest.fixture
def toy_model():
    return ToyModel(in_dim=10, n_classes=5)


@pytest.fixture
def toy_optimizer(toy_model):
    """构造一个 AdamW optimizer 并执行若干步，使其 state 非空。

    verse_torch AdamW 的 state 按 ``id(p)`` 键控，值含 ``m`` / ``v`` 矩阵
    （NumPy 实现，非 PyTorch 的 exp_avg / exp_avg_sq）。
    """
    opt = AdamW(toy_model.parameters(), lr=0.01)
    # 执行若干步，填充 m / v 矩阵
    for i in range(3):
        x = Tensor(np.random.randn(4, 10).astype(np.float32))
        y = Tensor(np.array([0, 1, 2, 3], dtype=np.int64))
        logits = toy_model(x)
        # 简单 MSE loss 驱动 backward
        target = np.zeros((4, 5), dtype=np.float32)
        target[np.arange(4), y.data] = 1.0
        loss = ((logits.data - target) ** 2).sum()
        loss_tensor = Tensor(loss, requires_grad=True)
        toy_model.zero_grad()
        loss_tensor.backward()
        opt.step()
    return opt


# ---------------------------------------------------------------------------
# 1. test_resume_state_fields：ResumeState namedtuple 字段定义
# ---------------------------------------------------------------------------


def test_resume_state_fields():
    """ResumeState namedtuple 包含 7 个必需字段。"""
    expected_fields = (
        "model_state_dict",
        "optimizer_state",
        "step",
        "rng_state",
        "best_val_loss",
        "epoch",
        "patience_count",
    )
    assert ResumeState._fields == expected_fields, (
        f"ResumeState 字段应为 {expected_fields}, "
        f"got {ResumeState._fields}"
    )
    # 默认值测试：所有字段为 None
    state = ResumeState(*([None] * 7))
    assert state.model_state_dict is None
    assert state.optimizer_state is None
    assert state.step is None
    assert state.rng_state is None
    assert state.best_val_loss is None
    assert state.epoch is None
    assert state.patience_count is None


# ---------------------------------------------------------------------------
# 2. test_resume_manager_save_load_roundtrip：全字段往返一致
# ---------------------------------------------------------------------------


def test_resume_manager_save_load_roundtrip(tmp_ckpt_dir, toy_model, toy_optimizer):
    """ResumeManager.save + load 全字段往返一致。"""
    vn_path = os.path.join(tmp_ckpt_dir, "resume.vn")
    rng_state = np.random.RandomState(42).get_state()

    ResumeManager.save(
        vn_path,
        model=toy_model,
        optimizer=toy_optimizer,
        step=123,
        best_val_loss=0.456,
        epoch=7,
        patience_count=2,
        rng_state=rng_state,
    )

    # 文件存在
    assert os.path.exists(vn_path), f".vn 文件应存在: {vn_path}"

    state = ResumeManager.load(vn_path)

    # model_state_dict 一致
    assert state.model_state_dict is not None
    original_sd = toy_model.state_dict()
    for key in original_sd:
        assert key in state.model_state_dict, f"model state_dict 缺键 {key}"
        np.testing.assert_array_equal(
            original_sd[key], state.model_state_dict[key],
            err_msg=f"model state_dict key={key} 数值不一致",
        )

    # optimizer_state 一致
    assert state.optimizer_state is not None
    # verse_torch Optimizer：state dict 按 id(p) 键控，含 m/v 矩阵
    # （非 PyTorch 风格的 state_dict + exp_avg/exp_avg_sq）
    assert "state" in state.optimizer_state
    assert "param_groups" in state.optimizer_state
    # 验证 m / v 矩阵数值一致（float32 吻合 1e-7）
    # verse_torch AdamW state: {id(p): {"m": ndarray, "v": ndarray}}
    for idx, s in toy_optimizer.state.items():
        assert idx in state.optimizer_state["state"], (
            f"optimizer state 缺失参数 id={idx}"
        )
        restored = state.optimizer_state["state"][idx]
        if "m" in s:
            np.testing.assert_allclose(
                s["m"], restored["m"], rtol=1e-7, atol=1e-7,
                err_msg=f"optimizer m 数值不一致 (param {idx})",
            )
        if "v" in s:
            np.testing.assert_allclose(
                s["v"], restored["v"], rtol=1e-7, atol=1e-7,
                err_msg=f"optimizer v 数值不一致 (param {idx})",
            )

    # 标量字段一致
    assert state.step == 123
    assert state.best_val_loss == pytest.approx(0.456)
    assert state.epoch == 7
    assert state.patience_count == 2

    # rng_state 一致
    assert state.rng_state is not None
    # rng_state 是 tuple ('MT19937', ndarray, 624, 0, 0.0)
    assert state.rng_state[0] == rng_state[0]  # 算法名
    np.testing.assert_array_equal(state.rng_state[1], rng_state[1])  # 状态数组


# ---------------------------------------------------------------------------
# 3. test_resume_manager_save_minimal：仅 model + step（其他字段为 None）
# ---------------------------------------------------------------------------


def test_resume_manager_save_minimal(tmp_ckpt_dir, toy_model):
    """ResumeManager.save 仅传 model + step，load 时其他字段为 None。"""
    vn_path = os.path.join(tmp_ckpt_dir, "resume.vn")

    ResumeManager.save(vn_path, model=toy_model, step=50)

    state = ResumeManager.load(vn_path)
    assert state.model_state_dict is not None
    assert state.step == 50
    # 未传的字段为 None
    assert state.optimizer_state is None
    assert state.best_val_loss is None
    assert state.epoch is None
    assert state.patience_count is None
    assert state.rng_state is None


# ---------------------------------------------------------------------------
# 4. test_resume_manager_save_to_dir：path 为目录时用默认文件名
# ---------------------------------------------------------------------------


def test_resume_manager_save_to_dir(tmp_ckpt_dir, toy_model):
    """ResumeManager.save 传目录时用默认文件名 resume.vn。"""
    returned_path = ResumeManager.save(
        tmp_ckpt_dir, model=toy_model, step=10,
    )
    expected = os.path.join(tmp_ckpt_dir, ResumeManager.DEFAULT_FILENAME)
    assert os.path.exists(expected), f"默认文件 {expected} 应存在"
    assert os.path.normpath(returned_path) == os.path.normpath(expected)


# ---------------------------------------------------------------------------
# 5. test_resume_manager_load_v1_vn：v1 .vn 向后兼容（缺失字段为 None）
# ---------------------------------------------------------------------------


def test_resume_manager_load_v1_vn(tmp_ckpt_dir, toy_model):
    """v1 .vn 文件（无 training_state / optimizer_state / extra_state）向后兼容。

    ResumeManager.load 返回 model_state_dict（v1 有权重），
    其余字段（optimizer_state / step / rng_state / best_val_loss / epoch /
    patience_count）为 None。
    """
    vn_path = os.path.join(tmp_ckpt_dir, "v1_resume.vn")
    original_sd = toy_model.state_dict()
    _write_v1_vn_file(vn_path, original_sd)

    state = ResumeManager.load(vn_path)

    # model_state_dict 存在（v1 有权重）
    assert state.model_state_dict is not None
    for key in original_sd:
        np.testing.assert_array_equal(
            original_sd[key], state.model_state_dict[key],
            err_msg=f"v1 .vn model state_dict key={key} 数值不一致",
        )

    # 其余字段为 None（v1 不含 training_state / optimizer_state / extra_state）
    assert state.optimizer_state is None
    assert state.step is None
    assert state.rng_state is None
    assert state.best_val_loss is None
    assert state.epoch is None
    assert state.patience_count is None


# ---------------------------------------------------------------------------
# 6. test_resume_manager_load_pt：旧 .pt pickle 向后兼容
# ---------------------------------------------------------------------------


def test_resume_manager_load_pt(tmp_ckpt_dir, toy_model):
    """旧 .pt pickle 文件可被 ResumeManager.load 兼容读取。"""
    pt_path = os.path.join(tmp_ckpt_dir, "resume.pt")
    original_sd = toy_model.state_dict()
    # 模拟旧 ParallelTrainerSafe._save_resume_state 写出的 .pt pickle
    payload = {
        "step": 99,
        "model_state_dict": original_sd,
        "best_state_dict": original_sd,
        "best_val_loss": 0.789,
        "history": {"train_loss": [1.0], "val_loss": [0.9], "steps": [1]},
    }
    with open(pt_path, "wb") as f:
        pickle.dump(payload, f)

    state = ResumeManager.load(pt_path)

    # model_state_dict 存在
    assert state.model_state_dict is not None
    for key in original_sd:
        np.testing.assert_array_equal(
            original_sd[key], state.model_state_dict[key],
        )
    # step / best_val_loss 从 pickle payload 提取
    assert state.step == 99
    assert state.best_val_loss == pytest.approx(0.789)
    # .pt 不含 optimizer_state / rng_state / epoch / patience_count
    assert state.optimizer_state is None
    assert state.rng_state is None
    assert state.epoch is None
    assert state.patience_count is None


# ---------------------------------------------------------------------------
# 7. test_resume_manager_load_not_found：文件不存在抛 FileNotFoundError
# ---------------------------------------------------------------------------


def test_resume_manager_load_not_found(tmp_ckpt_dir):
    """ResumeManager.load 文件不存在时抛 FileNotFoundError。"""
    fake_path = os.path.join(tmp_ckpt_dir, "nonexistent.vn")
    with pytest.raises(FileNotFoundError):
        ResumeManager.load(fake_path)


# ---------------------------------------------------------------------------
# 8. test_resume_manager_apply_full：apply 恢复全部字段到 trainer
# ---------------------------------------------------------------------------


class FakeTrainer:
    """模拟 Trainer / ParallelTrainerSafe 的最小属性集，用于测试 apply。"""

    def __init__(self, model):
        self.model = model
        self.optimizer = None
        self.best_val_loss = float("inf")
        self.step = 0
        self.epoch = 0
        self.early_stopping = None


class FakeEarlyStopping:
    """模拟 EarlyStopping，含 patience_count 属性。"""

    def __init__(self):
        self.patience_count = 0


def test_resume_manager_apply_full(tmp_ckpt_dir, toy_model, toy_optimizer):
    """ResumeManager.apply 恢复 model / optimizer / step / rng / best_val_loss。"""
    vn_path = os.path.join(tmp_ckpt_dir, "resume.vn")
    rng_state = np.random.RandomState(123).get_state()

    ResumeManager.save(
        vn_path,
        model=toy_model,
        optimizer=toy_optimizer,
        step=200,
        best_val_loss=0.321,
        epoch=5,
        patience_count=3,
        rng_state=rng_state,
    )

    # 构造新 trainer（model 权重不同，optimizer 不同）
    new_model = ToyModel(in_dim=10, n_classes=5)
    # 修改 new_model 权重，确保与 toy_model 不同
    for p in new_model.parameters():
        p.data = p.data + 1.0
    new_opt = AdamW(new_model.parameters(), lr=0.01)

    trainer = FakeTrainer(new_model)
    trainer.optimizer = new_opt
    trainer.early_stopping = FakeEarlyStopping()

    # apply
    state = ResumeManager.apply(trainer, vn_path)

    # model 恢复
    original_sd = toy_model.state_dict()
    restored_sd = trainer.model.state_dict()
    for key in original_sd:
        np.testing.assert_array_equal(
            original_sd[key], restored_sd[key],
            err_msg=f"apply 后 model key={key} 应与原值一致",
        )

    # optimizer 恢复（state 字典键集一致）
    # verse_torch Optimizer 没有 state_dict() 方法，直接比较 .state 属性
    assert trainer.optimizer is not None
    # apply 后 trainer.optimizer.state 应等于 toy_optimizer.state
    # （ResumeManager.apply 直接替换 opt.state = saved_state）
    assert set(trainer.optimizer.state.keys()) == set(toy_optimizer.state.keys())
    # 验证 m / v 矩阵数值一致
    for idx, s in toy_optimizer.state.items():
        restored = trainer.optimizer.state[idx]
        if "m" in s:
            np.testing.assert_allclose(
                s["m"], restored["m"], rtol=1e-7, atol=1e-7,
                err_msg=f"optimizer m 数值不一致 (param {idx})",
            )
        if "v" in s:
            np.testing.assert_allclose(
                s["v"], restored["v"], rtol=1e-7, atol=1e-7,
                err_msg=f"optimizer v 数值不一致 (param {idx})",
            )

    # 标量字段恢复
    assert trainer.best_val_loss == pytest.approx(0.321)
    assert trainer.step == 200
    assert trainer.epoch == 5
    assert trainer.early_stopping.patience_count == 3

    # rng_state 恢复：apply 后 np.random 状态与 rng_state 一致
    # 生成一个随机数，再用相同 state 生成，应一致
    val_after_apply = np.random.rand()
    np.random.set_state(rng_state)
    val_expected = np.random.rand()
    assert val_after_apply == pytest.approx(val_expected), (
        "apply 后 np.random 状态应与保存的 rng_state 一致"
    )

    # 返回值是 ResumeState
    assert isinstance(state, ResumeState)
    assert state.step == 200


# ---------------------------------------------------------------------------
# 9. test_resume_manager_apply_missing_fields：None 字段跳过恢复
# ---------------------------------------------------------------------------


def test_resume_manager_apply_missing_fields(tmp_ckpt_dir, toy_model):
    """ResumeManager.apply 对 None 字段跳过恢复（向后兼容）。"""
    vn_path = os.path.join(tmp_ckpt_dir, "resume.vn")
    # 仅保存 model + step（其他字段为 None）
    ResumeManager.save(vn_path, model=toy_model, step=42)

    new_model = ToyModel(in_dim=10, n_classes=5)
    for p in new_model.parameters():
        p.data = p.data + 2.0
    trainer = FakeTrainer(new_model)
    trainer.optimizer = AdamW(new_model.parameters(), lr=0.01)
    trainer.best_val_loss = 0.5  # 预设值，不应被 apply 覆盖（state.best_val_loss 为 None）
    trainer.step = 0
    trainer.early_stopping = FakeEarlyStopping()
    trainer.early_stopping.patience_count = 7  # 预设值，不应被覆盖

    ResumeManager.apply(trainer, vn_path)

    # model 恢复
    original_sd = toy_model.state_dict()
    restored_sd = trainer.model.state_dict()
    for key in original_sd:
        np.testing.assert_array_equal(original_sd[key], restored_sd[key])

    # step 恢复
    assert trainer.step == 42

    # best_val_loss 未恢复（state 为 None，保留预设值）
    assert trainer.best_val_loss == 0.5
    # patience_count 未恢复（state 为 None，保留预设值）
    assert trainer.early_stopping.patience_count == 7


# ---------------------------------------------------------------------------
# 10. test_parallel_trainer_safe_save_load_vn：ParallelTrainerSafe 集成
# ---------------------------------------------------------------------------


def test_parallel_trainer_safe_save_load_vn(tmp_ckpt_dir):
    """ParallelTrainerSafe._save_resume_state 写 .vn + _load_resume_state 读 .vn。

    验证：
    - _save_resume_state 写出 resume.vn（不是 .pt）
    - _load_resume_state 读取 .vn，恢复 model / best_val_loss / best_state_dict
    """
    from verse_infra.verse_trainer.trainer import ParallelTrainerSafe

    model = ToyModel(in_dim=10, n_classes=5)
    train_ds = ToyDataset(n=20, in_dim=10, n_classes=5, seed=0)
    val_ds = ToyDataset(n=10, in_dim=10, n_classes=5, seed=100)

    ckpt_mgr = CheckpointManager(
        save_dir=tmp_ckpt_dir, format="vn", use_vmpc=True,
    )
    trainer = ParallelTrainerSafe(
        model=model,
        train_dataset=train_ds,
        val_dataset=val_ds,
        cfg={"parallel_chunks": 2, "max_steps": 4, "batch_size": 4, "lr": 0.01},
        checkpoint_mgr=ckpt_mgr,
    )

    # 设置已知状态
    trainer.best_val_loss = 0.555
    trainer.best_state_dict = model.state_dict()

    # _save_resume_state 写出 .vn
    resume_path = os.path.join(tmp_ckpt_dir, "resume.vn")
    trainer._save_resume_state(resume_path, step=42)

    # 验证写出的是 .vn 文件
    assert os.path.exists(resume_path), f"resume.vn 应存在: {resume_path}"
    # 不应有 .pt 文件
    pt_path = os.path.join(tmp_ckpt_dir, "resume.pt")
    assert not os.path.exists(pt_path), "不应写出 .pt 文件"

    # 构造新 trainer，从 .vn 恢复
    new_model = ToyModel(in_dim=10, n_classes=5)
    for p in new_model.parameters():
        p.data = p.data + 1.0  # 确保权重不同
    new_trainer = ParallelTrainerSafe(
        model=new_model,
        train_dataset=train_ds,
        val_dataset=val_ds,
        cfg={"parallel_chunks": 2, "max_steps": 4, "batch_size": 4, "lr": 0.01},
        checkpoint_mgr=ckpt_mgr,
    )
    # 初始 best_val_loss 应为 inf
    assert new_trainer.best_val_loss == float("inf")

    # _load_resume_state 从 .vn 恢复
    new_trainer._load_resume_state(resume_path)

    # best_val_loss 恢复
    assert new_trainer.best_val_loss == pytest.approx(0.555)
    # best_state_dict 恢复
    assert new_trainer.best_state_dict is not None
    original_sd = model.state_dict()
    for key in original_sd:
        np.testing.assert_array_equal(
            original_sd[key], new_trainer.best_state_dict[key],
            err_msg=f"best_state_dict key={key} 应恢复",
        )
    # model 也恢复（ResumeManager.apply 加载 model_state_dict）
    restored_sd = new_trainer.model.state_dict()
    for key in original_sd:
        np.testing.assert_array_equal(
            original_sd[key], restored_sd[key],
            err_msg=f"model key={key} 应恢复",
        )


# ---------------------------------------------------------------------------
# 11. test_parallel_trainer_safe_load_pt_backward_compat：旧 .pt 向后兼容
# ---------------------------------------------------------------------------


def test_parallel_trainer_safe_load_pt_backward_compat(tmp_ckpt_dir):
    """旧 ParallelTrainerSafe._save_resume_state 写出的 .pt 可被新 _load_resume_state 兼容读取。

    验证：_load_resume_state 优先尝试 .vn，.vn 不存在时回退到 .pt。
    """
    from verse_infra.verse_trainer.trainer import ParallelTrainerSafe

    model = ToyModel(in_dim=10, n_classes=5)
    train_ds = ToyDataset(n=20, in_dim=10, n_classes=5, seed=0)
    val_ds = ToyDataset(n=10, in_dim=10, n_classes=5, seed=100)

    # 手工写出旧 .pt 格式 resume 文件（模拟旧 ParallelTrainerSafe 输出）
    pt_path = os.path.join(tmp_ckpt_dir, "resume.pt")
    original_sd = model.state_dict()
    payload = {
        "step": 77,
        "model_state_dict": original_sd,
        "best_state_dict": original_sd,
        "best_val_loss": 0.888,
        "history": {"train_loss": [1.0], "val_loss": [0.9], "steps": [1]},
    }
    with open(pt_path, "wb") as f:
        pickle.dump(payload, f)

    # 构造新 trainer，传入 .pt 路径
    new_model = ToyModel(in_dim=10, n_classes=5)
    for p in new_model.parameters():
        p.data = p.data + 1.0
    new_trainer = ParallelTrainerSafe(
        model=new_model,
        train_dataset=train_ds,
        val_dataset=val_ds,
        cfg={"parallel_chunks": 2, "max_steps": 4, "batch_size": 4, "lr": 0.01},
    )

    # _load_resume_state 传入 .pt 路径，应回退到 .pt 读取
    new_trainer._load_resume_state(pt_path)

    # best_val_loss 恢复
    assert new_trainer.best_val_loss == pytest.approx(0.888)
    # best_state_dict 恢复
    assert new_trainer.best_state_dict is not None
    for key in original_sd:
        np.testing.assert_array_equal(
            original_sd[key], new_trainer.best_state_dict[key],
        )
    # model 恢复
    restored_sd = new_trainer.model.state_dict()
    for key in original_sd:
        np.testing.assert_array_equal(original_sd[key], restored_sd[key])


# ---------------------------------------------------------------------------
# 12. test_parallel_trainer_safe_pt_path_falls_back_to_vn：传 .pt 路径但 .vn 存在
# ---------------------------------------------------------------------------


def test_parallel_trainer_safe_pt_path_prefers_vn(tmp_ckpt_dir):
    """传 .pt 路径但 .vn 存在时，_load_resume_state 优先读 .vn。"""
    from verse_infra.verse_trainer.trainer import ParallelTrainerSafe

    model = ToyModel(in_dim=10, n_classes=5)
    train_ds = ToyDataset(n=20, in_dim=10, n_classes=5, seed=0)
    val_ds = ToyDataset(n=10, in_dim=10, n_classes=5, seed=100)

    # 写出 .vn（新格式，best_val_loss=0.111）
    vn_path = os.path.join(tmp_ckpt_dir, "resume.vn")
    ResumeManager.save(
        vn_path, model=model, step=10, best_val_loss=0.111,
    )
    # 同时写出 .pt（旧格式，best_val_loss=0.999）
    pt_path = os.path.join(tmp_ckpt_dir, "resume.pt")
    payload = {
        "step": 99,
        "model_state_dict": model.state_dict(),
        "best_val_loss": 0.999,
    }
    with open(pt_path, "wb") as f:
        pickle.dump(payload, f)

    # 构造新 trainer，传入 .pt 路径
    new_model = ToyModel(in_dim=10, n_classes=5)
    new_trainer = ParallelTrainerSafe(
        model=new_model,
        train_dataset=train_ds,
        val_dataset=val_ds,
        cfg={"parallel_chunks": 2, "max_steps": 4, "batch_size": 4, "lr": 0.01},
    )

    # 传 .pt 路径，但 .vn 存在 → 应优先读 .vn（best_val_loss=0.111）
    new_trainer._load_resume_state(pt_path)
    assert new_trainer.best_val_loss == pytest.approx(0.111), (
        "应优先读 .vn（best_val_loss=0.111），而非 .pt（0.999）"
    )


# ---------------------------------------------------------------------------
# 13. test_parallel_trainer_safe_resume_path_in_init：__init__ 通过 resume_path 恢复
# ---------------------------------------------------------------------------


def test_parallel_trainer_safe_resume_path_in_init(tmp_ckpt_dir):
    """ParallelTrainerSafe(resume_path=...) 在 __init__ 时自动恢复状态。"""
    from verse_infra.verse_trainer.trainer import ParallelTrainerSafe

    model = ToyModel(in_dim=10, n_classes=5)
    train_ds = ToyDataset(n=20, in_dim=10, n_classes=5, seed=0)
    val_ds = ToyDataset(n=10, in_dim=10, n_classes=5, seed=100)

    # 先用一个 trainer 保存 resume.vn
    ckpt_mgr = CheckpointManager(
        save_dir=tmp_ckpt_dir, format="vn", use_vmpc=True,
    )
    trainer1 = ParallelTrainerSafe(
        model=model,
        train_dataset=train_ds,
        val_dataset=val_ds,
        cfg={"parallel_chunks": 2, "max_steps": 4, "batch_size": 4, "lr": 0.01},
        checkpoint_mgr=ckpt_mgr,
    )
    trainer1.best_val_loss = 0.444
    trainer1.best_state_dict = model.state_dict()
    resume_path = os.path.join(tmp_ckpt_dir, "resume.vn")
    trainer1._save_resume_state(resume_path, step=33)

    # 新 trainer 通过 resume_path 恢复
    new_model = ToyModel(in_dim=10, n_classes=5)
    for p in new_model.parameters():
        p.data = p.data + 1.0
    new_trainer = ParallelTrainerSafe(
        model=new_model,
        train_dataset=train_ds,
        val_dataset=val_ds,
        cfg={"parallel_chunks": 2, "max_steps": 4, "batch_size": 4, "lr": 0.01},
        resume_path=resume_path,  # __init__ 时自动恢复
    )

    # best_val_loss 恢复
    assert new_trainer.best_val_loss == pytest.approx(0.444)
    # best_state_dict 恢复
    assert new_trainer.best_state_dict is not None
    original_sd = model.state_dict()
    for key in original_sd:
        np.testing.assert_array_equal(
            original_sd[key], new_trainer.best_state_dict[key],
        )


# ---------------------------------------------------------------------------
# 14. test_resume_manager_save_returns_path：save 返回绝对路径
# ---------------------------------------------------------------------------


def test_resume_manager_save_returns_absolute_path(tmp_ckpt_dir, toy_model):
    """ResumeManager.save 返回 .vn 文件的绝对路径。"""
    vn_path = os.path.join(tmp_ckpt_dir, "resume.vn")
    returned = ResumeManager.save(vn_path, model=toy_model, step=1)
    assert os.path.isabs(returned), f"应返回绝对路径, got {returned}"
    assert os.path.exists(returned)


# ---------------------------------------------------------------------------
# 15. test_resume_manager_apply_returns_state：apply 返回 ResumeState
# ---------------------------------------------------------------------------


def test_resume_manager_apply_returns_state(tmp_ckpt_dir, toy_model):
    """ResumeManager.apply 返回 ResumeState，便于调用方进一步处理。"""
    vn_path = os.path.join(tmp_ckpt_dir, "resume.vn")
    ResumeManager.save(vn_path, model=toy_model, step=55, best_val_loss=0.66)

    trainer = FakeTrainer(ToyModel())
    state = ResumeManager.apply(trainer, vn_path)

    assert isinstance(state, ResumeState)
    assert state.step == 55
    assert state.best_val_loss == pytest.approx(0.66)
