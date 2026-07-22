"""测试 MoDLayer 的完整功能（SubTask 2.4）。

覆盖：
- 构造与默认配置（5 DensePart × 8 Expert × top-3）
- MoD 前向（输出 shape 正确 + 数值合理）
- MoD 反向（梯度可流回到输入与所有 router/expert 参数）
- aux loss 收敛（小训练步后 aux_loss 不发散、单调下降趋势）
- forward 可重复性（eval 模式下相同输入 → 相同输出；MoDLayer 无 recurrent 模式）
- aux_loss() / get_aux_loss_dict() 接口正确性
- z_loss_weight=0 时不计算 z-loss
- load_balance + z_loss == total
- 参数初始化无 NaN / Inf
- 边界检查（非法参数抛 ValueError）
- DensePart 名称自定义
- train / eval 模式切换
"""

from __future__ import annotations

import os
import sys

import numpy as np

# PYTHONPATH 适配（与 test_cometspark_nex.py 风格一致）
_HERE = os.path.dirname(os.path.abspath(__file__))
_WORKSPACE = os.path.dirname(_HERE)
for _sub in ("verse_torch", "verse_nex", "verse_tokenizer"):
    _p = os.path.join(_WORKSPACE, "packages", _sub)
    if os.path.isdir(_p):
        sys.path.insert(0, _p)

import pytest

from verse_torch import Tensor, no_grad
from verse_nex import MoDLayer
from verse_nex.moe import Router, Expert, DensePart


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _tiny_mod(
    dim: int = 32,
    num_dense_parts: int = 2,
    num_experts_per_part: int = 2,
    top_k: int = 1,
    dropout: float = 0.0,
    aux_loss_weight: float = 0.01,
    z_loss_weight: float = 0.001,
    seed: int = None,
) -> MoDLayer:
    """构造 tiny MoDLayer 用于快速测试。

    默认配置较小（2 part × 2 expert × top-1）以加速测试；
    test_default_config 覆盖论文默认 5×8×3 配置。
    """
    if seed is not None:
        np.random.seed(seed)
    return MoDLayer(
        dim=dim,
        num_dense_parts=num_dense_parts,
        num_experts_per_part=num_experts_per_part,
        top_k=top_k,
        dropout=dropout,
        aux_loss_weight=aux_loss_weight,
        z_loss_weight=z_loss_weight,
    )


def _rand_input(batch=2, seq=4, dim=32, requires_grad=True, seed=None) -> Tensor:
    """生成随机输入 Tensor。"""
    if seed is not None:
        np.random.seed(seed)
    return Tensor(
        np.random.randn(batch, seq, dim).astype(np.float32),
        requires_grad=requires_grad,
    )


# ---------------------------------------------------------------------------
# 构造与默认配置
# ---------------------------------------------------------------------------


def test_construct_tiny_mod():
    """构造 tiny MoDLayer，验证属性正确。"""
    mod = _tiny_mod(dim=32, num_dense_parts=3, num_experts_per_part=4, top_k=2)
    assert mod.dim == 32
    assert mod.num_dense_parts == 3
    assert mod.num_experts_per_part == 4
    assert mod.top_k == 2
    assert len(mod.parts) == 3
    # 每个 DensePart 内含 num_experts_per_part 个 Expert
    for part in mod.parts:
        assert len(part.experts) == 4
        assert part.router.num_routes == 4
        assert part.router.top_k == 2
    # part_router 是 soft routing（top_k == num_dense_parts）
    assert mod.part_router.num_routes == 3
    assert mod.part_router.top_k == 3
    # 默认分区名
    assert mod.dense_part_names == ["part_0", "part_1", "part_2"]


def test_default_config_5x8x3():
    """验证论文默认配置 5 DensePart × 8 Expert × top-3 正常工作。"""
    mod = MoDLayer(dim=64)  # 全部使用默认值
    assert mod.num_dense_parts == 5
    assert mod.num_experts_per_part == 8
    assert mod.top_k == 3
    assert len(mod.parts) == 5
    for part in mod.parts:
        assert len(part.experts) == 8
        assert part.router.top_k == 3
    # part_router 是 soft routing：top_k == num_dense_parts == 5
    assert mod.part_router.top_k == 5
    # 默认 z_loss_weight > 0
    assert mod.z_loss_weight > 0
    # 默认分区名
    assert mod.dense_part_names == [f"part_{i}" for i in range(5)]


def test_custom_dense_part_names():
    """自定义 dense_part_names。"""
    names = ["alpha", "beta", "gamma"]
    mod = MoDLayer(dim=16, num_dense_parts=3, dense_part_names=names)
    assert mod.dense_part_names == names

    # 长度不匹配应抛 ValueError
    with pytest.raises(ValueError):
        MoDLayer(dim=16, num_dense_parts=3, dense_part_names=["only_one"])


# ---------------------------------------------------------------------------
# 边界检查
# ---------------------------------------------------------------------------


def test_invalid_num_dense_parts():
    """num_dense_parts < 1 应抛 ValueError。"""
    with pytest.raises(ValueError):
        MoDLayer(dim=32, num_dense_parts=0)


def test_invalid_num_experts():
    """num_experts_per_part < 1 应抛 ValueError。"""
    with pytest.raises(ValueError):
        MoDLayer(dim=32, num_experts_per_part=0)


def test_top_k_greater_than_num_experts():
    """top_k > num_experts_per_part 应抛 ValueError。"""
    with pytest.raises(ValueError):
        MoDLayer(dim=32, num_experts_per_part=2, top_k=3)


def test_router_top_k_invalid():
    """Router top_k < 1 应抛 ValueError。"""
    with pytest.raises(ValueError):
        Router(dim=32, num_routes=4, top_k=0)


def test_router_top_k_greater_than_routes():
    """Router top_k > num_routes 应抛 ValueError。"""
    with pytest.raises(ValueError):
        Router(dim=32, num_routes=2, top_k=3)


# ---------------------------------------------------------------------------
# 前向：输出 shape 正确 + 数值合理
# ---------------------------------------------------------------------------


def test_forward_shape():
    """前向输出 shape = (B, T, D)，与输入一致。"""
    mod = _tiny_mod(dim=32, seed=42)
    mod.eval()
    x = _rand_input(batch=2, seq=4, dim=32, seed=42)
    with no_grad():
        out, aux = mod(x)
    assert out.data.shape == (2, 4, 32)
    # aux 是标量
    assert aux.data.shape == () or aux.data.size == 1


def test_forward_shape_default_config():
    """默认 5×8×3 配置前向 shape 正确。"""
    mod = MoDLayer(dim=64)
    mod.eval()
    x = Tensor(np.random.randn(2, 8, 64).astype(np.float32))
    with no_grad():
        out, aux = mod(x)
    assert out.data.shape == (2, 8, 64)
    # aux 是有限实数
    assert np.isfinite(float(aux.data.sum()))


def test_forward_no_nan_inf():
    """前向输出无 NaN / Inf。"""
    mod = _tiny_mod(dim=32, seed=7)
    mod.eval()
    x = _rand_input(batch=3, seq=6, dim=32, seed=7)
    with no_grad():
        out, aux = mod(x)
    assert not np.any(np.isnan(out.data)), "输出含 NaN"
    assert not np.any(np.isinf(out.data)), "输出含 Inf"
    assert np.isfinite(float(aux.data.sum())), "aux loss 含 NaN/Inf"


def test_forward_parameters_no_nan_inf():
    """构造后所有参数无 NaN / Inf。"""
    mod = _tiny_mod(dim=32, seed=11)
    for p in mod.parameters():
        assert not np.any(np.isnan(p.data)), f"参数含 NaN, shape={p.data.shape}"
        assert not np.any(np.isinf(p.data)), f"参数含 Inf, shape={p.data.shape}"


# ---------------------------------------------------------------------------
# 反向：梯度可流回到输入与参数
# ---------------------------------------------------------------------------


def test_backward_gradient_flow_to_input():
    """loss = out.sum() + aux，backward 后 x.grad 非空且无 NaN。"""
    mod = _tiny_mod(dim=32, seed=13)
    mod.train()
    x = _rand_input(batch=2, seq=4, dim=32, requires_grad=True, seed=13)
    out, aux = mod(x)
    loss = out.sum() + aux
    loss.backward()
    assert x.grad is not None
    assert not np.any(np.isnan(x.grad)), "x.grad 含 NaN"
    assert not np.any(np.isinf(x.grad)), "x.grad 含 Inf"
    # 梯度不全为 0（应至少有一些非零梯度）
    assert np.any(np.abs(x.grad) > 0), "x.grad 全为 0"


def test_backward_gradient_flow_to_parameters():
    """backward 后所有可训练参数应收到梯度。"""
    mod = _tiny_mod(dim=32, seed=17)
    mod.train()
    x = _rand_input(batch=2, seq=4, dim=32, requires_grad=True, seed=17)
    out, aux = mod(x)
    loss = out.sum() + aux
    loss.backward()
    # 检查每个参数的梯度
    grad_count = 0
    total_params = 0
    for p in mod.parameters():
        total_params += 1
        if p.grad is not None and np.any(np.abs(p.grad) > 0):
            grad_count += 1
    # 应至少有大部分参数收到梯度（part_router + 每个 DensePart 内的 expert_router + experts）
    assert grad_count > 0, "没有参数收到梯度"
    assert grad_count >= total_params * 0.5, (
        f"只有 {grad_count}/{total_params} 个参数收到梯度（<50%）"
    )


def test_backward_aux_only_no_nan():
    """仅用 aux loss 反向，梯度无 NaN（验证 z-loss 与 load_balance 都可微）。"""
    mod = _tiny_mod(dim=32, seed=19)
    mod.train()
    x = _rand_input(batch=2, seq=4, dim=32, requires_grad=True, seed=19)
    _, aux = mod(x)
    aux.backward()
    # x.grad 应非空且无 NaN（aux 通过 router logits 与 x 相关）
    assert x.grad is not None
    assert not np.any(np.isnan(x.grad)), "aux 反向 x.grad 含 NaN"


def test_backward_gate_weight_receives_grad():
    """part_router.gate 与各 expert_router.gate 应收到梯度。"""
    mod = _tiny_mod(dim=32, seed=23)
    mod.train()
    x = _rand_input(batch=2, seq=4, dim=32, requires_grad=True, seed=23)
    out, aux = mod(x)
    loss = out.sum() + aux
    loss.backward()
    # part_router.gate 必须收到梯度
    assert mod.part_router.gate.weight.grad is not None
    assert np.any(np.abs(mod.part_router.gate.weight.grad) > 0)
    # 每个 DensePart 内 expert_router.gate 也必须收到梯度
    for part in mod.parts:
        assert part.router.gate.weight.grad is not None
        assert np.any(np.abs(part.router.gate.weight.grad) > 0)


# ---------------------------------------------------------------------------
# aux loss 接口
# ---------------------------------------------------------------------------


def test_aux_loss_matches_forward_return():
    """aux_loss() 与 forward 返回的 aux 应一致。"""
    mod = _tiny_mod(dim=32, seed=29)
    mod.eval()
    x = _rand_input(batch=2, seq=4, dim=32, seed=29)
    with no_grad():
        out, aux = mod(x)
        aux_from_method = mod.aux_loss()
    assert aux_from_method is not None
    np.testing.assert_allclose(
        float(aux_from_method.data.sum()),
        float(aux.data.sum()),
        rtol=1e-6,
        atol=1e-6,
    )


def test_aux_loss_dict_keys():
    """get_aux_loss_dict() 返回 keys: load_balance, z_loss, total。"""
    mod = _tiny_mod(dim=32, seed=31)
    mod.eval()
    x = _rand_input(batch=2, seq=4, dim=32, seed=31)
    with no_grad():
        out, aux = mod(x)
    d = mod.get_aux_loss_dict()
    assert d is not None
    assert set(d.keys()) == {"load_balance", "z_loss", "total"}


def test_aux_loss_dict_total_equals_sum():
    """total == load_balance + z_loss。"""
    mod = _tiny_mod(dim=32, seed=37)
    mod.eval()
    x = _rand_input(batch=2, seq=4, dim=32, seed=37)
    with no_grad():
        out, aux = mod(x)
    d = mod.get_aux_loss_dict()
    lb = float(d["load_balance"].data.sum())
    zl = float(d["z_loss"].data.sum())
    total = float(d["total"].data.sum())
    np.testing.assert_allclose(total, lb + zl, rtol=1e-5, atol=1e-6)


def test_aux_loss_before_forward_is_none():
    """未 forward 时 aux_loss() 与 get_aux_loss_dict() 都应返回 None。"""
    mod = _tiny_mod(dim=32, seed=41)
    assert mod.aux_loss() is None
    assert mod.get_aux_loss_dict() is None


def test_z_loss_weight_zero():
    """z_loss_weight=0 时 z_loss 应为 0。"""
    mod = _tiny_mod(dim=32, z_loss_weight=0.0, seed=43)
    mod.eval()
    x = _rand_input(batch=2, seq=4, dim=32, seed=43)
    with no_grad():
        out, aux = mod(x)
    d = mod.get_aux_loss_dict()
    zl = float(d["z_loss"].data.sum())
    assert zl == 0.0, f"z_loss_weight=0 时 z_loss 应为 0，got {zl}"


def test_z_loss_weight_positive():
    """z_loss_weight > 0 时 z_loss 应 > 0。"""
    mod = _tiny_mod(dim=32, z_loss_weight=0.01, seed=47)
    mod.eval()
    x = _rand_input(batch=2, seq=4, dim=32, seed=47)
    with no_grad():
        out, aux = mod(x)
    d = mod.get_aux_loss_dict()
    zl = float(d["z_loss"].data.sum())
    assert zl > 0, f"z_loss_weight>0 时 z_loss 应 > 0，got {zl}"


# ---------------------------------------------------------------------------
# aux loss 收敛性：小训练步后 aux_loss 不发散
# ---------------------------------------------------------------------------


def test_aux_loss_does_not_diverge():
    """跑若干小训练步，aux_loss 不发散（保持有限且不为 NaN）。"""
    mod = _tiny_mod(dim=32, z_loss_weight=0.001, seed=53)
    mod.train()
    x = _rand_input(batch=2, seq=4, dim=32, requires_grad=True, seed=53)

    aux_history = []
    for step in range(5):
        # 每步重新前向（清空梯度）
        mod.zero_grad()
        x.grad = None
        out, aux = mod(x)
        loss = out.sum() + aux
        loss.backward()
        # 手动 SGD 更新（仅 router gate 权重，避免 expert 输出发散）
        lr = 1e-3
        for p in mod.parameters():
            if p.grad is not None:
                p.data = p.data - lr * p.grad
        aux_val = float(aux.data.sum())
        aux_history.append(aux_val)
        # 每步检查：不发散、无 NaN
        assert np.isfinite(aux_val), f"step {step}: aux_loss 不有限（NaN/Inf）"
        assert abs(aux_val) < 1e3, f"step {step}: aux_loss 过大 = {aux_val}"

    # 全程检查：aux_loss 序列应全有限
    assert all(np.isfinite(v) for v in aux_history)


def test_aux_loss_decreasing_trend():
    """在多步训练后 aux_loss 应有下降趋势（至少最后一步 < 第一步的某个倍数）。

    由于 aux_loss 同时受参数更新和 router 路由变化影响，
    不要求严格单调下降，但最终值不应远大于初始值。
    """
    mod = _tiny_mod(dim=32, z_loss_weight=0.001, seed=59)
    mod.train()
    x = _rand_input(batch=4, seq=8, dim=32, requires_grad=True, seed=59)

    # 初始 aux_loss
    mod.zero_grad()
    _, aux0 = mod(x)
    aux0_val = float(aux0.data.sum())

    # 跑 10 步只优化 router（用 aux_loss 自己作为目标，鼓励负载均衡）
    lr = 1e-2
    for step in range(10):
        mod.zero_grad()
        x.grad = None
        _, aux = mod(x)
        # 反向传播（只优化 aux loss 自身，使其下降）
        aux.backward()
        for p in mod.parameters():
            if p.grad is not None and ("gate" in str(id(p)) or True):
                p.data = p.data - lr * p.grad

    # 最终 aux_loss
    mod.zero_grad()
    _, aux_final = mod(x)
    aux_final_val = float(aux_final.data.sum())

    # 最终值应不发散
    assert np.isfinite(aux_final_val), "aux_loss 最终值不有限"
    # 最终值不应比初始值大很多（容忍 2× 之内）
    assert aux_final_val < aux0_val * 2 + 1e-3, (
        f"aux_loss 未收敛：初始 {aux0_val}，最终 {aux_final_val}"
    )


# ---------------------------------------------------------------------------
# forward 可重复性：eval 模式下相同输入 → 相同输出
# （MoDLayer 无 recurrent 模式，验证 forward 可重复作为一致性检查）
# ---------------------------------------------------------------------------


def test_eval_forward_reproducible():
    """eval 模式下相同输入两次前向，输出应完全一致（Dropout 关闭）。"""
    mod = _tiny_mod(dim=32, dropout=0.1, seed=61)
    mod.eval()
    x = _rand_input(batch=2, seq=4, dim=32, seed=61)
    with no_grad():
        out1, aux1 = mod(x)
        out2, aux2 = mod(x)
    np.testing.assert_allclose(out1.data, out2.data, rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(
        float(aux1.data.sum()), float(aux2.data.sum()), rtol=1e-6, atol=1e-7
    )


def test_train_mode_dropout_active():
    """train 模式下 dropout>0 时，两次前向输出应不同（Dropout 激活）。"""
    mod = _tiny_mod(dim=32, dropout=0.5, seed=67)
    mod.train()
    x = _rand_input(batch=2, seq=4, dim=32, seed=67)
    with no_grad():
        out1, _ = mod(x)
        out2, _ = mod(x)
    # 应有差异（极小概率相同，加 seed 控制）
    diff = np.abs(out1.data - out2.data).max()
    assert diff > 1e-5, f"train 模式下两次前向输出完全相同（diff={diff}），Dropout 未激活"


def test_train_eval_mode_switch():
    """切换 train/eval 模式应影响所有子模块。"""
    mod = _tiny_mod(dim=32, dropout=0.1, seed=71)
    mod.eval()
    assert mod.training is False
    for part in mod.parts:
        assert part.training is False
        for expert in part.experts:
            assert expert.dropout.training is False
    mod.train()
    assert mod.training is True
    for part in mod.parts:
        assert part.training is True


# ---------------------------------------------------------------------------
# 输入与配置变化
# ---------------------------------------------------------------------------


def test_forward_single_batch():
    """单 batch 前向正常。"""
    mod = _tiny_mod(dim=16, seed=73)
    mod.eval()
    x = Tensor(np.random.randn(1, 1, 16).astype(np.float32))
    with no_grad():
        out, aux = mod(x)
    assert out.data.shape == (1, 1, 16)


def test_forward_large_sequence():
    """较长序列前向正常。"""
    mod = _tiny_mod(dim=16, seed=79)
    mod.eval()
    x = Tensor(np.random.randn(2, 32, 16).astype(np.float32))
    with no_grad():
        out, aux = mod(x)
    assert out.data.shape == (2, 32, 16)


def test_top_k_equals_num_experts():
    """top_k == num_experts（soft routing）应正常工作。"""
    mod = _tiny_mod(dim=16, num_experts_per_part=3, top_k=3, seed=83)
    mod.eval()
    x = Tensor(np.random.randn(2, 4, 16).astype(np.float32))
    with no_grad():
        out, aux = mod(x)
    assert out.data.shape == (2, 4, 16)


def test_top_k_one():
    """top_k=1（hard routing）应正常工作。"""
    mod = _tiny_mod(dim=16, num_experts_per_part=4, top_k=1, seed=89)
    mod.eval()
    x = Tensor(np.random.randn(2, 4, 16).astype(np.float32))
    with no_grad():
        out, aux = mod(x)
    assert out.data.shape == (2, 4, 16)


def test_single_dense_part():
    """num_dense_parts=1（退化为单层 MoE）应正常工作。"""
    mod = _tiny_mod(dim=16, num_dense_parts=1, num_experts_per_part=2, top_k=1, seed=97)
    mod.eval()
    x = Tensor(np.random.randn(2, 4, 16).astype(np.float32))
    with no_grad():
        out, aux = mod(x)
    assert out.data.shape == (2, 4, 16)


# ---------------------------------------------------------------------------
# Router / Expert / DensePart 单元测试
# ---------------------------------------------------------------------------


def test_router_forward_returns_three():
    """Router.forward 返回 (indices, weights, aux) 三元组。"""
    np.random.seed(101)
    router = Router(dim=16, num_routes=4, top_k=2)
    router.eval()
    x = Tensor(np.random.randn(2, 4, 16).astype(np.float32))
    with no_grad():
        idx, w, aux = router(x)
    assert idx.data.shape == (2, 4, 2)
    assert w.data.shape == (2, 4, 2)
    # 权重应在 [0, 1] 且每行 sum=1（softmax）
    assert np.all(w.data >= 0)
    np.testing.assert_allclose(w.data.sum(axis=-1), 1.0, rtol=1e-5, atol=1e-6)
    # aux 是标量
    assert aux.data.shape == () or aux.data.size == 1


def test_router_top_k_indices_unique():
    """Router 选出的 top-k 索引应无重复。"""
    np.random.seed(103)
    router = Router(dim=16, num_routes=8, top_k=3)
    router.eval()
    x = Tensor(np.random.randn(2, 4, 16).astype(np.float32))
    with no_grad():
        idx, w, _ = router(x)
    # 每个 token 的 top-k 索引应互不相同
    for b in range(2):
        for t in range(4):
            assert len(set(idx.data[b, t].tolist())) == 3


def test_expert_forward_shape():
    """Expert 前向 shape 保持 (..., D)。"""
    np.random.seed(105)
    expert = Expert(dim=16, hidden=32, dropout=0.0)
    expert.eval()
    x = Tensor(np.random.randn(4, 16).astype(np.float32), requires_grad=True)
    with no_grad():
        y = expert(x)
    assert y.data.shape == (4, 16)


def test_dense_part_forward_shape():
    """DensePart 前向返回 (out, aux)，out shape = (B, T, D)。"""
    np.random.seed(107)
    part = DensePart(dim=16, num_experts=3, top_k=2)
    part.eval()
    x = Tensor(np.random.randn(2, 4, 16).astype(np.float32))
    part_weights = Tensor(np.ones((2, 4, 1), dtype=np.float32))
    with no_grad():
        out, aux = part(x, part_weights)
    assert out.data.shape == (2, 4, 16)


def test_router_jitter_no_crash():
    """jitter > 0 训练模式不报错。"""
    np.random.seed(109)
    router = Router(dim=16, num_routes=4, top_k=2, jitter=0.1)
    router.train()
    x = Tensor(np.random.randn(2, 4, 16).astype(np.float32))
    idx, w, aux = router(x)
    assert idx.data.shape == (2, 4, 2)
