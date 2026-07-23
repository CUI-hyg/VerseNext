"""Checkpoint 目录迁移工具（Part5K1 Task 10.2）。

设计目标
--------
- **自动迁移旧 checkpoint 目录**：检测旧的 ``checkpoints_XXX/`` 目录并
  重命名为新的 ``mf_XXX/``（Part5K1 命名规范化）。
- **保守策略**：目标目录已存在且非空时**不迁移**，避免覆盖用户数据。
- **幂等**：无旧目录或目标已存在时直接返回目标目录，不报错。

使用方式
--------
在 :func:`verse_trainer.trainer.train` 入口（或任意 checkpoint 保存逻辑前）
调用 :func:`migrate_checkpoint_dir`::

    from verse_trainer.checkpoint_utils import migrate_checkpoint_dir
    save_dir = config.get("checkpoint", {}).get("save_dir", "mf_small")
    model_level = config.get("model_level", "small")
    save_dir = migrate_checkpoint_dir(save_dir, model_level)

迁移逻辑
--------
1. 目标目录 ``save_dir``（如 ``mf_small``）已存在且非空 → 直接返回，不动。
2. 检测旧目录 ``checkpoints_{model_level}``（如 ``checkpoints_small``）：
   - 存在且非空 → 发出 ``DeprecationWarning`` + ``os.rename`` 迁移 + 打印日志。
   - 不存在或为空 → 不动，返回目标目录。
3. ``save_dir`` 为空时，默认目标为 ``mf_{model_level}``。

路径解析
--------
- ``save_dir`` 为绝对路径时，旧目录解析为相对于其父目录的同级目录。
- ``save_dir`` 为相对路径时，旧目录也用相对路径（相对于 cwd）。
"""
from __future__ import annotations

import os
import warnings
from typing import Optional


def migrate_checkpoint_dir(
    save_dir: Optional[str],
    model_level: str = "small",
) -> str:
    """检测旧 ``checkpoints_XXX/`` 目录并迁移为 ``mf_XXX/``。

    保守策略：目标目录已存在且非空时**不迁移**，直接返回目标。

    Args:
        save_dir: 目标 checkpoint 目录（如 ``"mf_small"``）。
            传入 ``None`` / 空字符串时，默认使用 ``f"mf_{model_level}"``。
        model_level: 模型级别（``"small"`` / ``"mate"``），用于推断旧目录名
            ``checkpoints_{model_level}``。

    Returns:
        实际使用的 save_dir（迁移后的路径）。如果 ``save_dir`` 传入为空，
        返回 ``f"mf_{model_level}"``。

    Note:
        - 幂等：无旧目录或目标已存在时不报错。
        - 迁移时发出 :class:`DeprecationWarning`（Part5K1 重命名提示）。
    """
    old_name = f"checkpoints_{model_level}"
    # save_dir 为空时默认目标为 mf_{model_level}
    new_dir = save_dir if save_dir else f"mf_{model_level}"

    # 1. 目标目录已存在且非空 → 直接返回，不动（保守策略）
    if os.path.exists(new_dir) and os.path.isdir(new_dir) and os.listdir(new_dir):
        return new_dir

    # 2. 解析旧目录路径
    #    - save_dir 为绝对路径时，旧目录解析为相对于其父目录的同级目录
    #    - save_dir 为相对路径时，旧目录也用相对路径（相对于 cwd）
    if os.path.isabs(new_dir):
        old_dir = os.path.join(os.path.dirname(new_dir), old_name)
    else:
        old_dir = old_name

    # 3. 检测旧目录：存在且非空才迁移
    if os.path.exists(old_dir) and os.path.isdir(old_dir) and os.listdir(old_dir):
        warnings.warn(
            f"检测到旧 checkpoint 目录 {old_dir}/，自动迁移为 {new_dir}/"
            f"（Part5K1 重命名）",
            DeprecationWarning,
            stacklevel=2,
        )
        os.rename(old_dir, new_dir)
        print(f"[checkpoint] 迁移 {old_dir}/ → {new_dir}/", flush=True)

    return new_dir


__all__ = [
    "migrate_checkpoint_dir",
]
