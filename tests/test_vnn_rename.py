"""Part5K1 Task 1：VerseTorch.nn → VerseTorch.vnn 重命名测试。

验证内容：
1. 新导入路径 ``from verse_torch.vnn import ...`` 全部核心符号可用。
2. 旧路径 ``from verse_torch import nn`` 仍可工作（向后兼容，不报错）。
3. transformer 系旧名（``TransformerLM`` / ``TransformerBlock`` / ``GQASelfAttention``）
   从 ``verse_torch.nn`` 导入时抛 ``ImportError``（Part5K1 REMOVED）。
4. ``vnn`` 中的核心类可正常实例化（小尺寸）。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# 让 tests/ 目录能 import verse_torch
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "packages" / "verse_torch"))


# ---------------------------------------------------------------------------
# 1. 新导入路径可用
# ---------------------------------------------------------------------------


def test_vnn_import_core_symbols():
    """新路径 ``from verse_torch.vnn import ...`` 全部核心符号导入成功。"""
    from verse_torch.vnn import (  # noqa: F401
        Module,
        Linear,
        Embedding,
        LayerNorm,
        RMSNorm,
        Dropout,
        Sequential,
        ModuleList,
        SwiGLUMLP,
        kaiming_uniform_,
        xavier_uniform_,
    )


def test_vnn_import_extended_symbols():
    """新路径下扩展符号也可用。"""
    from verse_torch.vnn import (  # noqa: F401
        SlidingWindowAttention,
        ALiBi,
        DeepNorm,
        RotaryEmbedding,
        KVCache,
        StaticCache,
        DynamicCache,
        GroupNorm,
        Conv1d,
        LayerNormFast,
        repeat_kv,
        normal_,
        zeros_,
        ones_,
        uniform_,
    )


def test_vnn_private_implementations_preserved():
    """vnn 保留私有实现（``_`` 前缀），供内部使用。"""
    from verse_torch.vnn import (  # noqa: F401
        _GQASelfAttention,
        _TransformerBlock,
        _TransformerLM,
        _concat,
    )


# ---------------------------------------------------------------------------
# 2. 旧路径 ``from verse_torch import nn`` 仍可工作
# ---------------------------------------------------------------------------


def test_old_nn_alias_still_works():
    """``from verse_torch import nn`` 不报错（nn 指向 vnn 别名）。"""
    from verse_torch import nn
    # nn 应是 vnn 模块（或至少提供相同的核心符号）
    assert hasattr(nn, "Module")
    assert hasattr(nn, "Linear")
    assert hasattr(nn, "Embedding")
    assert hasattr(nn, "LayerNorm")


def test_from_verse_torch_nn_import_module_works():
    """``from verse_torch.nn import Module`` 仍可用（经薄壳 re-export）。"""
    from verse_torch.nn import Module as ShimModule
    from verse_torch.vnn import Module as VnnModule
    assert ShimModule is VnnModule


# ---------------------------------------------------------------------------
# 3. transformer 系旧名抛 ImportError
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("old_name", ["TransformerLM", "TransformerBlock", "GQASelfAttention"])
def test_transformer_old_names_raise_importerror_via_getattr(old_name):
    """``getattr(verse_torch.nn, <old_name>)`` 应抛 ImportError。"""
    import importlib
    nn_mod = importlib.import_module("verse_torch.nn")
    with pytest.raises(ImportError, match=old_name):
        getattr(nn_mod, old_name)


@pytest.mark.parametrize("old_name", ["TransformerLM", "TransformerBlock", "GQASelfAttention"])
def test_transformer_old_names_raise_importerror_via_from_import(old_name):
    """``from verse_torch.nn import <old_name>`` 应抛 ImportError。"""
    import importlib
    nn_mod = importlib.import_module("verse_torch.nn")
    with pytest.raises(ImportError, match=old_name):
        # 等价于 from verse_torch.nn import <old_name>
        getattr(nn_mod, old_name)


def test_transformer_private_names_still_importable_from_vnn():
    """私有实现（``_`` 前缀）仍可从 vnn 正常导入。"""
    from verse_torch.vnn import _TransformerLM, _TransformerBlock, _GQASelfAttention
    # 它们应是类（可被子类化/实例化）
    assert isinstance(_TransformerLM, type)
    assert isinstance(_TransformerBlock, type)
    assert isinstance(_GQASelfAttention, type)


# ---------------------------------------------------------------------------
# 4. vnn 核心类可正常实例化
# ---------------------------------------------------------------------------


def test_linear_instantiation():
    """Linear(4, 2) 可正常实例化与前向。"""
    import numpy as np
    from verse_torch import Tensor
    from verse_torch.vnn import Linear

    np.random.seed(0)
    layer = Linear(4, 2)
    assert layer.weight.shape == (2, 4)
    assert layer.bias.shape == (2,)
    x = Tensor(np.random.randn(3, 4).astype(np.float32), requires_grad=False)
    out = layer(x)
    assert out.shape == (3, 2)


def test_embedding_instantiation():
    """Embedding(10, 4) 可正常实例化与前向。"""
    import numpy as np
    from verse_torch.vnn import Embedding

    np.random.seed(0)
    emb = Embedding(10, 4)
    assert emb.weight.shape == (10, 4)
    out = emb([0, 1, 2])
    assert out.shape == (3, 4)


def test_layernorm_instantiation():
    """LayerNorm(4) 可正常实例化与前向。"""
    import numpy as np
    from verse_torch import Tensor
    from verse_torch.vnn import LayerNorm

    np.random.seed(0)
    ln = LayerNorm(4)
    assert ln.weight.shape == (4,)
    assert ln.bias.shape == (4,)
    x = Tensor(np.random.randn(2, 3, 4).astype(np.float32), requires_grad=False)
    out = ln(x)
    assert out.shape == (2, 3, 4)


def test_rmsnorm_instantiation():
    """RMSNorm(4) 可正常实例化与前向。"""
    import numpy as np
    from verse_torch import Tensor
    from verse_torch.vnn import RMSNorm

    np.random.seed(0)
    rn = RMSNorm(4)
    assert rn.weight.shape == (4,)
    x = Tensor(np.random.randn(2, 3, 4).astype(np.float32), requires_grad=False)
    out = rn(x)
    assert out.shape == (2, 3, 4)


def test_dropout_instantiation():
    """Dropout(0.5) 可正常实例化。"""
    from verse_torch.vnn import Dropout

    d = Dropout(0.5)
    assert d.p == 0.5
    # eval 模式下应原样返回
    d.eval()
    import numpy as np
    from verse_torch import Tensor
    x = Tensor(np.random.randn(2, 3).astype(np.float32), requires_grad=False)
    out = d(x)
    assert out.shape == (2, 3)


def test_sequential_instantiation():
    """Sequential 可正常实例化与调用。"""
    import numpy as np
    from verse_torch import Tensor
    from verse_torch.vnn import Sequential, Linear

    np.random.seed(0)
    seq = Sequential(Linear(4, 8), Linear(8, 2))
    x = Tensor(np.random.randn(3, 4).astype(np.float32), requires_grad=False)
    out = seq(x)
    assert out.shape == (3, 2)


def test_modulelist_instantiation():
    """ModuleList 可正常实例化与索引。"""
    from verse_torch.vnn import ModuleList, Linear

    ml = ModuleList([Linear(4, 2), Linear(4, 3)])
    assert len(ml) == 2
    assert ml[0].out_features == 2
    assert ml[1].out_features == 3


def test_swiglu_mlp_instantiation():
    """SwiGLUMLP 可正常实例化与前向。"""
    import numpy as np
    from verse_torch import Tensor
    from verse_torch.vnn import SwiGLUMLP

    np.random.seed(0)
    mlp = SwiGLUMLP(8, dropout=0.0)
    x = Tensor(np.random.randn(2, 3, 8).astype(np.float32), requires_grad=False)
    out = mlp(x)
    assert out.shape == (2, 3, 8)


def test_init_helpers_run():
    """kaiming_uniform_ / xavier_uniform_ 可对 Tensor 正常调用。"""
    import numpy as np
    from verse_torch import Tensor
    from verse_torch.vnn import kaiming_uniform_, xavier_uniform_

    np.random.seed(0)
    t = Tensor.empty(4, 4, requires_grad=True)
    kaiming_uniform_(t)
    assert t.shape == (4, 4)

    t2 = Tensor.empty(4, 4, requires_grad=True)
    xavier_uniform_(t2)
    assert t2.shape == (4, 4)
