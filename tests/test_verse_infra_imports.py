"""Task 7.9: VerseInfra 总包聚合导入测试。

覆盖三类场景：
1. 子模块导入：``from verse_infra.verse_xxx import ...``
2. 便捷重导出：``from verse_infra import BPETokenizer, ModelLoader, train, RLTrainer``
3. 旧路径 shim DeprecationWarning：``from verse_tokenizer import ...`` 应发出
   ``DeprecationWarning`` 但仍可正常使用

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
# 3. 旧路径 shim DeprecationWarning 测试（子进程隔离，避免模块缓存污染）
# ---------------------------------------------------------------------------


def _run_subprocess(code: str, timeout: int = 30) -> subprocess.CompletedProcess:
    """在子进程中执行 Python 代码，返回 CompletedProcess。

    用子进程隔离每个 shim 测试，避免主进程 sys.modules 缓存干扰
    DeprecationWarning 触发次数。
    """
    return subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True, text=True, timeout=timeout,
    )


class TestShimDeprecationWarning:
    """验证旧路径 ``from verse_xxx import ...`` 经 shim 转发 + DeprecationWarning。"""

    def test_verse_tokenizer_shim(self):
        """``from verse_tokenizer import BPETokenizer`` 应发 DeprecationWarning 且可用。"""
        code = (
            "import sys, warnings\n"
            f"sys.path.insert(0, '{_REPO_ROOT / 'packages' / 'verse_tokenizer'}')\n"
            "with warnings.catch_warnings(record=True) as w:\n"
            "    warnings.simplefilter('always')\n"
            "    from verse_tokenizer import BPETokenizer\n"
            "    assert BPETokenizer is not None, 'BPETokenizer missing'\n"
            "    assert any(issubclass(x.category, DeprecationWarning) for x in w), \\\n"
            "        f'expected DeprecationWarning, got: {[(x.category, str(x.message)) for x in w]}'\n"
            "    print('shim OK: BPETokenizer =', BPETokenizer)\n"
        )
        result = _run_subprocess(code)
        assert result.returncode == 0, (
            f"shim 测试失败：\nstdout={result.stdout}\nstderr={result.stderr}"
        )
        assert "shim OK" in result.stdout

    def test_verse_compat_shim(self):
        """``from verse_compat import load_hf_state_dict`` 应发 DeprecationWarning 且可用。"""
        code = (
            "import sys, warnings\n"
            f"sys.path.insert(0, '{_REPO_ROOT / 'packages' / 'verse_compat'}')\n"
            "with warnings.catch_warnings(record=True) as w:\n"
            "    warnings.simplefilter('always')\n"
            "    from verse_compat import load_hf_state_dict\n"
            "    assert callable(load_hf_state_dict), 'load_hf_state_dict missing'\n"
            "    assert any(issubclass(x.category, DeprecationWarning) for x in w), \\\n"
            "        f'expected DeprecationWarning, got: {[(x.category, str(x.message)) for x in w]}'\n"
            "    print('shim OK: load_hf_state_dict =', load_hf_state_dict)\n"
        )
        result = _run_subprocess(code)
        assert result.returncode == 0, (
            f"shim 测试失败：\nstdout={result.stdout}\nstderr={result.stderr}"
        )
        assert "shim OK" in result.stdout

    def test_verse_inference_shim(self):
        """``from verse_inference import ModelLoader`` 应发 DeprecationWarning 且可用。"""
        code = (
            "import sys, warnings\n"
            f"sys.path.insert(0, '{_REPO_ROOT / 'packages' / 'verse_inference'}')\n"
            "with warnings.catch_warnings(record=True) as w:\n"
            "    warnings.simplefilter('always')\n"
            "    from verse_inference import ModelLoader\n"
            "    assert ModelLoader is not None, 'ModelLoader missing'\n"
            "    assert any(issubclass(x.category, DeprecationWarning) for x in w), \\\n"
            "        f'expected DeprecationWarning, got: {[(x.category, str(x.message)) for x in w]}'\n"
            "    print('shim OK: ModelLoader =', ModelLoader)\n"
        )
        result = _run_subprocess(code)
        assert result.returncode == 0, (
            f"shim 测试失败：\nstdout={result.stdout}\nstderr={result.stderr}"
        )
        assert "shim OK" in result.stdout

    def test_verse_trainer_shim(self):
        """``from verse_trainer import train, RLTrainer`` 应发 DeprecationWarning 且可用。"""
        code = (
            "import sys, warnings\n"
            f"sys.path.insert(0, '{_REPO_ROOT / 'packages' / 'verse_trainer'}')\n"
            "with warnings.catch_warnings(record=True) as w:\n"
            "    warnings.simplefilter('always')\n"
            "    from verse_trainer import train, RLTrainer\n"
            "    assert callable(train), 'train missing'\n"
            "    assert RLTrainer is not None, 'RLTrainer missing'\n"
            "    assert any(issubclass(x.category, DeprecationWarning) for x in w), \\\n"
            "        f'expected DeprecationWarning, got: {[(x.category, str(x.message)) for x in w]}'\n"
            "    print('shim OK: train =', train)\n"
        )
        result = _run_subprocess(code)
        assert result.returncode == 0, (
            f"shim 测试失败：\nstdout={result.stdout}\nstderr={result.stderr}"
        )
        assert "shim OK" in result.stdout

    def test_verse_tokenizer_shim_submodule_lazy(self):
        """``verse_tokenizer.verse`` 子模块属性访问应经 shim __getattr__ 转发可用。

        注意：PEP 562 的 ``__getattr__`` 仅支持属性访问（``import verse_tokenizer;
        verse_tokenizer.verse``），不支持 ``from verse_tokenizer.verse import ...``
        形式（后者需要物理文件）。这里用属性访问验证延迟转发。
        """
        code = (
            "import sys, warnings\n"
            f"sys.path.insert(0, '{_REPO_ROOT / 'packages' / 'verse_tokenizer'}')\n"
            "with warnings.catch_warnings(record=True) as w:\n"
            "    warnings.simplefilter('always')\n"
            "    import verse_tokenizer\n"
            "    # 通过 __getattr__ 延迟访问子模块 verse\n"
            "    verse_mod = verse_tokenizer.verse\n"
            "    assert hasattr(verse_mod, '_import_transformers'), \\\n"
            "        f'verse module missing _import_transformers: {dir(verse_mod)}'\n"
            "    assert callable(verse_mod._import_transformers), \\\n"
            "        '_import_transformers not callable'\n"
            "    assert any(issubclass(x.category, DeprecationWarning) for x in w), \\\n"
            "        f'expected DeprecationWarning, got: {[(x.category, str(x.message)) for x in w]}'\n"
            "    print('shim submodule OK')\n"
        )
        result = _run_subprocess(code)
        assert result.returncode == 0, (
            f"shim 子模块测试失败：\nstdout={result.stdout}\nstderr={result.stderr}"
        )
        assert "shim submodule OK" in result.stdout


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
