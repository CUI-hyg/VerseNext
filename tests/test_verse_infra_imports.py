"""Task 7.9: VerseInfra 总包聚合导入测试。

覆盖三类场景：
1. 子模块导入：``from verse_infra.verse_xxx import ...``
2. 便捷重导出：``from verse_infra import BPETokenizer, ModelLoader, train, RLTrainer``
3. Part1：旧路径顶层 shim（``verse_tokenizer`` / ``verse_compat`` /
   ``verse_inference`` / ``verse_trainer``）已彻底删除，导入应失败

运行方式：
    cd /workspace && python -m pytest tests/test_verse_infra_imports.py -v
"""

from __future__ import annotations

import os
import subprocess
import sys
import warnings
from pathlib import Path

import pytest

# 让 tests/ 目录能 import verse_infra 及其子模块
_REPO_ROOT = Path(__file__).resolve().parent.parent
for _pkg in ("verse_infra", "verse_torch", "verse_nex"):
    _p = _REPO_ROOT / "packages" / _pkg
    if _p.is_dir() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


# ---------------------------------------------------------------------------
# 1. 子模块导入测试
# ---------------------------------------------------------------------------


class TestSubmoduleImports:
    """验证 ``from verse_infra.verse_xxx import ...`` 全部可用。"""

    def test_verse_tokenizer_submodule(self):
        """子模块 verse_infra.verse_tokenizer 公共 API 可导入。"""
        from verse_infra.verse_tokenizer import (
            BPETokenizer,
            ByteTokenizer,
            CharTokenizer,
            BaseTokenizer,
            load_tokenizer,
        )
        assert BPETokenizer is not None
        assert ByteTokenizer is not None
        assert CharTokenizer is not None
        assert BaseTokenizer is not None
        assert callable(load_tokenizer)

    def test_verse_compat_submodule(self):
        """子模块 verse_infra.verse_compat 公共 API 可导入。"""
        from verse_infra.verse_compat import (
            load_hf_state_dict,
            Tensor,
            nn,
            optim,
            losses,
        )
        assert callable(load_hf_state_dict)
        assert Tensor is not None
        assert nn is not None
        assert optim is not None
        assert losses is not None

    def test_verse_inference_submodule(self):
        """子模块 verse_infra.verse_inference 公共 API 可导入。"""
        from verse_infra.verse_inference import (
            ModelLoader,
            StateCache,
            Sampler,
            GreedySampler,
            StreamingGenerator,
        )
        assert ModelLoader is not None
        assert StateCache is not None
        assert Sampler is not None
        assert GreedySampler is not None
        assert StreamingGenerator is not None

    def test_verse_trainer_submodule(self):
        """子模块 verse_infra.verse_trainer 公共 API 可导入。"""
        from verse_infra.verse_trainer import (
            CachedDataset,
            TextDataset,
            BatchLoader,
            collate_fn,
            load_jsonl,
            train,
            ParallelTrainerSafe,
            ChunkOOMError,
            evaluate,
            visualize,
            LossOptimizer,
            RLTrainer,
        )
        assert CachedDataset is not None
        assert TextDataset is not None
        assert BatchLoader is not None
        assert callable(collate_fn)
        assert callable(load_jsonl)
        assert callable(train)
        assert ParallelTrainerSafe is not None
        assert ChunkOOMError is not None
        assert callable(evaluate)
        assert callable(visualize)
        assert LossOptimizer is not None
        assert RLTrainer is not None

    def test_verse_tokenizer_submodule_lazy(self):
        """子模块延迟访问：``from verse_infra.verse_tokenizer.bpe import ...`` 可用。"""
        from verse_infra.verse_tokenizer.bpe import BPETokenizer as BPE2
        from verse_infra.verse_tokenizer.verse import _import_transformers
        assert BPE2 is not None
        assert callable(_import_transformers)

    def test_verse_trainer_cli_submodule(self):
        """子模块延迟访问：``from verse_infra.verse_trainer.cli import ...`` 可用。"""
        from verse_infra.verse_trainer.cli import train_main
        assert callable(train_main)


# ---------------------------------------------------------------------------
# 2. 便捷重导出测试
# ---------------------------------------------------------------------------


class TestConvenienceReexport:
    """验证 ``from verse_infra import ...`` 便捷重导出可用。"""

    def test_reexport_tokenizer_apis(self):
        """``from verse_infra import BPETokenizer, ByteTokenizer`` 可用。"""
        from verse_infra import BPETokenizer, ByteTokenizer, CharTokenizer
        assert BPETokenizer is not None
        assert ByteTokenizer is not None
        assert CharTokenizer is not None

    def test_reexport_inference_apis(self):
        """``from verse_infra import ModelLoader, StreamingGenerator`` 可用。"""
        from verse_infra import ModelLoader, Sampler, StreamingGenerator
        assert ModelLoader is not None
        assert Sampler is not None
        assert StreamingGenerator is not None

    def test_reexport_trainer_apis(self):
        """``from verse_infra import train, RLTrainer, CachedDataset`` 可用。"""
        from verse_infra import train, RLTrainer, CachedDataset, LossOptimizer
        assert callable(train)
        assert RLTrainer is not None
        assert CachedDataset is not None
        assert LossOptimizer is not None

    def test_reexport_compat_apis(self):
        """``from verse_infra import load_hf_state_dict, Tensor`` 可用。"""
        from verse_infra import load_hf_state_dict, Tensor
        assert callable(load_hf_state_dict)
        assert Tensor is not None

    def test_reexport_combined(self):
        """``from verse_infra import BPETokenizer, ModelLoader, train, RLTrainer``
        一次性从 4 个子模块取值可用。"""
        from verse_infra import BPETokenizer, ModelLoader, train, RLTrainer
        assert BPETokenizer is not None
        assert ModelLoader is not None
        assert callable(train)
        assert RLTrainer is not None

    def test_submodule_attribute_access(self):
        """``import verse_infra; verse_infra.verse_tokenizer`` 子模块属性访问可用。"""
        import verse_infra
        # 通过 __getattr__ 延迟加载子模块
        sub = verse_infra.verse_tokenizer
        assert sub is not None
        assert hasattr(sub, "BPETokenizer")
        # 第二次访问应从 globals() 直接取（已缓存）
        sub2 = verse_infra.verse_tokenizer
        assert sub is sub2

    def test_invalid_attribute_raises(self):
        """访问不存在的属性应抛 AttributeError。"""
        import verse_infra
        with pytest.raises(AttributeError):
            _ = verse_infra.nonexistent_module_name


# ---------------------------------------------------------------------------
# 3. Part1：旧路径顶层 shim 已删除，导入应失败
# ---------------------------------------------------------------------------


def _run_subprocess(code: str, timeout: int = 30) -> subprocess.CompletedProcess:
    """在子进程中执行 Python 代码，返回 CompletedProcess。

    用子进程隔离，避免主进程 sys.modules 缓存干扰。
    """
    return subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True, text=True, timeout=timeout,
    )


class TestShimRemoved:
    """验证旧路径顶层 shim（Part1 已彻底删除）导入应抛 ImportError。"""

    @pytest.mark.parametrize("shim_name", [
        "verse_tokenizer", "verse_compat", "verse_inference", "verse_trainer",
    ])
    def test_old_top_level_shim_import_fails(self, shim_name):
        """``import verse_tokenizer`` 等旧路径应抛 ImportError（shim 已删除）。"""
        code = (
            "import sys\n"
            f"sys.path.insert(0, '{_REPO_ROOT / 'packages'}')\n"
            "try:\n"
            f"    import {shim_name}\n"
            "except ImportError:\n"
            "    print('ImportError as expected')\n"
            "else:\n"
            "    raise SystemExit(f'{shim_name} should have been removed')\n"
        )
        result = _run_subprocess(code)
        assert result.returncode == 0, (
            f"shim 删除验证失败：\nstdout={result.stdout}\nstderr={result.stderr}"
        )
        assert "ImportError as expected" in result.stdout

    @pytest.mark.parametrize("shim_name", [
        "verse_tokenizer", "verse_compat", "verse_inference", "verse_trainer",
    ])
    def test_shim_dir_absent(self, shim_name):
        """shim 包目录应不存在于 packages/。"""
        assert not (_REPO_ROOT / "packages" / shim_name).exists(), (
            f"packages/{shim_name} 应已删除"
        )


# ---------------------------------------------------------------------------
# 4. verse_torch / verse_nex 保持独立未并入 VerseInfra
# ---------------------------------------------------------------------------


def test_verse_torch_still_independent():
    """verse_torch 保持独立，未并入 verse_infra。"""
    import verse_torch
    # verse_torch 应该有自己的 __init__.py，不是 verse_infra 的子模块
    assert "verse_torch" in sys.modules
    assert not verse_torch.__name__.startswith("verse_infra.")


def test_verse_nex_still_independent():
    """verse_nex 保持独立，未并入 verse_infra。"""
    import verse_nex
    assert "verse_nex" in sys.modules
    assert not verse_nex.__name__.startswith("verse_infra.")
