"""统一的路径引导模块

在项目未做 pip install 时，确保所有包都能正确导入。
只需在入口文件（spark/__init__.py、spark/run.py、CLI 入口）导入一次即可，
子模块不需要重复 sys.path.insert。

使用方式：
    import spark._bootstrap  # 自动设置好所有路径

或显式调用：

    from spark._bootstrap import ensure_paths
    ensure_paths()

设计说明
--------
- 使用 ``pathlib.Path`` 做路径计算，不手动拼字符串，避免跨平台问题。
- 幂等：多次调用 ``ensure_paths()`` 不会重复添加路径。
- 模块导入时自动执行一次 ``ensure_paths()``，后续显式调用是安全的空操作。
- 仅注入项目自身包目录，不污染 ``sys.path`` 与第三方包的命名空间。
"""

from __future__ import annotations

import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# 路径计算（基于 __file__ 推断，不硬编码 /workspace）
# ---------------------------------------------------------------------------
# 本文件位于 <repo_root>/spark/_bootstrap.py
_THIS_FILE = Path(__file__).resolve()
_SPARK_DIR = _THIS_FILE.parent              # <repo_root>/spark
_REPO_ROOT = _SPARK_DIR.parent              # <repo_root>
_PACKAGES_DIR = _REPO_ROOT / "packages"     # <repo_root>/packages
_DATA_DIR = _REPO_ROOT / "data"             # <repo_root>/data

# 需要注入 sys.path 的目录列表（顺序：底层依赖在前）
# - verse_torch：核心 Tensor 引擎（最底层依赖）
# - verse_nex：替代架构库（依赖 verse_torch）
# - verse_infra：聚合总包（依赖 verse_torch / verse_nex）
# - spark：CometSpark 模型（依赖 verse_torch / verse_nex / verse_infra）
# - data：数据工具（DatasetDownloader 等，namespace package）
_PATHS_TO_ADD: list = [
    _PACKAGES_DIR / "verse_torch",
    _PACKAGES_DIR / "verse_nex",
    _PACKAGES_DIR / "verse_infra",
    _SPARK_DIR,
    _DATA_DIR,
]


def ensure_paths() -> None:
    """幂等地将项目所有包目录注入 ``sys.path``。

    多次调用安全：已存在的路径不会重复添加。
    缺失的目录（如未克隆子模块）会被跳过，不抛异常。
    """
    for path in _PATHS_TO_ADD:
        if path.is_dir():
            path_str = str(path)
            if path_str not in sys.path:
                sys.path.insert(0, path_str)


# 模块导入时自动执行一次，确保 ``import spark._bootstrap`` 即完成路径设置
ensure_paths()
