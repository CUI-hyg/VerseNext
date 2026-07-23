"""VerseTorch.nn（Part5K1：已重命名为 vnn，本文件为向后兼容薄壳）。

Part5K1 BREAKING 变更：
- 推荐导入路径已改为 ``verse_torch.vnn``（``from verse_torch.vnn import ...``）。
- 本 ``nn`` 模块仍作为薄壳保留，``from verse_torch.nn import Module`` 等非
  transformer 系符号仍可用（自动从 ``vnn`` re-export）。
- transformer 系旧名 ``TransformerLM`` / ``TransformerBlock`` / ``GQASelfAttention``
  已从公开 API 移除：``from verse_torch.nn import TransformerLM`` 现抛 ``ImportError``。
  请改用 ``from verse_nex import VerseNexLM / VerseNexBlock / VerseNexAttention``，
 或从 ``verse_torch.vnn`` 导入对应私有实现（``_TransformerLM`` 等，``_`` 前缀）。
"""

# 从 vnn re-export 全部公开符号（不含 ``_`` 前缀私有名）
from .vnn import *  # noqa: F401, F403

# 显式 re-export 私有实现（``import *`` 不会捕获 ``_`` 前缀名，但 __init__.py
# 与部分内部代码需要按私有名导入它们）
from .vnn import (  # noqa: F401
    _GQASelfAttention,
    _TransformerBlock,
    _TransformerLM,
    _concat,
)

# Part5K1 REMOVED：transformer 系公开别名从 DeprecationWarning 升级为 ImportError。
# 旧名 → 新品牌名（提示信息用）
_REMOVED_NN_ALIASES = {
    "TransformerLM": "VerseNexLM",
    "TransformerBlock": "VerseNexBlock",
    "GQASelfAttention": "VerseNexAttention",
}


def __getattr__(name):
    """模块级 ``__getattr__`` 钩子：transformer 系旧名访问时抛 ``ImportError``。

    当 ``from verse_torch.nn import TransformerLM`` 时触发（因为这些公开名
    未在本 shim 中定义，Python 回退到模块级 ``__getattr__``）。
    """
    if name in _REMOVED_NN_ALIASES:
        new_name = _REMOVED_NN_ALIASES[name]
        raise ImportError(
            f"`{name}` 已从 verse_torch.nn 移除（Part5K1 BREAKING）。"
            f"请改用 `from verse_nex import {new_name}`，"
            f"或从 verse_torch.vnn 导入私有实现 `_{name}`。"
        )
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
