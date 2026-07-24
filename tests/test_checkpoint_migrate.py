"""Part5K1 Task 10：checkpoint 重命名 + .vn 默认输出测试。

覆盖 SubTask 10.4 的全部测试用例：
1. save(format="vn") 生成 .vn 文件 / save(format="pt") 生成 .pt 文件
2. save(path) 不传 format，默认生成 .vn
3. save(path, format="invalid") 抛 ValueError
4. 目录迁移：checkpoints_small/ → mf_small/，文件保留
5. 目标已存在不迁移
6. 无旧目录不迁移，返回目标目录
7. mate 级别迁移：checkpoints_mate/ → mf_mate/
8. save_pretrained(path) 默认生成 .vn

运行方式：
    cd /workspace && python -m pytest tests/test_checkpoint_migrate.py -x -q
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# sys.path 注入（与 test_dual_model.py 一致）
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parent.parent
for _pkg in ("verse_torch", "verse_nex", "verse_infra", "verse_trainer"):
    _p = _REPO_ROOT / "packages" / _pkg
    if _p.is_dir() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


# 主导入路径：verse_infra.verse_trainer（推荐路径，无 DeprecationWarning）
# verse_trainer 顶层包是 shim，会发 DeprecationWarning，仅用于包导出测试。
_IMPORT_PATH = "verse_infra.verse_trainer.checkpoint_utils"


# ---------------------------------------------------------------------------
# SubTask 10.1: save 方法支持 format 参数
# ---------------------------------------------------------------------------


class TestSaveFormatParameter:
    """save(format=...) 参数验证（用 Small 模型，小尺寸配置）。"""

    def test_save_vn_generates_vn_file(self, tmp_path):
        """save(path, format='vn') 生成 .vn 文件。"""
        from spark.small.model import CometSparkSmall

        model = CometSparkSmall()
        out = str(tmp_path / "model_vn")  # 不带扩展名
        model.save(out, format="vn")

        vn_path = tmp_path / "model_vn.vn"
        assert vn_path.exists(), f".vn 文件未生成：{vn_path}"
        # .vn 是 ZIP 容器，大小应 > 0
        assert vn_path.stat().st_size > 0

    def test_save_pt_generates_pt_file(self, tmp_path):
        """save(path, format='pt') 在 legacy 模式（use_vmpc=False）下生成 .pt 文件。

        Part5K1.1：``use_vmpc=True``（默认）时强制 .vn，禁止 .pt；此处显式关闭
        VMPC 以验证 legacy .pt 保存路径仍然可用。
        """
        from spark.small.model import CometSparkSmall

        model = CometSparkSmall()
        # Part5K1.1：legacy 模式才允许 .pt
        model.config.use_vmpc = False
        out = str(tmp_path / "model_pt")  # 不带扩展名
        model.save(out, format="pt")

        pt_path = tmp_path / "model_pt.pt"
        assert pt_path.exists(), f".pt 文件未生成：{pt_path}"
        assert pt_path.stat().st_size > 0

    def test_save_pt_blocked_when_vmpc_enabled(self, tmp_path):
        """Part5K1.1：use_vmpc=True 时 save(format='pt') 抛 ValueError（强制 .vn）。"""
        from spark.small.model import CometSparkSmall

        model = CometSparkSmall()
        # 默认 use_vmpc=True，保存 .pt 应被拦截
        assert model.config.use_vmpc is True
        out = str(tmp_path / "model_blocked")
        with pytest.raises(ValueError, match="强制使用 .vn 格式"):
            model.save(out, format="pt")

    def test_save_default_format_is_vn(self, tmp_path):
        """save(path) 不传 format，默认生成 .vn（检查文件扩展名）。"""
        from spark.small.model import CometSparkSmall

        model = CometSparkSmall()
        out = str(tmp_path / "model_default")  # 不带扩展名
        model.save(out)  # 不传 format

        vn_path = tmp_path / "model_default.vn"
        pt_path = tmp_path / "model_default.pt"
        assert vn_path.exists(), f"默认应生成 .vn 文件：{vn_path}"
        assert not pt_path.exists(), "默认 format 是 vn，不应生成 .pt"

    def test_save_with_extension_already_present(self, tmp_path):
        """save(path.v.v.v, format='vn') 时 path 已含 .vn 后缀，不重复添加。"""
        from spark.small.model import CometSparkSmall

        model = CometSparkSmall()
        out = str(tmp_path / "model_explicit.vn")  # 已含 .vn
        model.save(out, format="vn")

        vn_path = tmp_path / "model_explicit.vn"
        assert vn_path.exists()
        # 不应生成 .vn.vn
        assert not (tmp_path / "model_explicit.vn.vn").exists()

    def test_save_invalid_format_raises_value_error(self, tmp_path):
        """save(path, format='invalid') 抛 ValueError。"""
        from spark.small.model import CometSparkSmall

        model = CometSparkSmall()
        out = str(tmp_path / "model_err")
        with pytest.raises(ValueError, match="未知 format"):
            model.save(out, format="invalid")

    def test_save_invalid_format_raises_value_error_mate(self, tmp_path):
        """Mate 模型 save(format='invalid') 同样抛 ValueError。"""
        from spark.mate.model import CometSparkMate

        # 小尺寸配置避免 OOM
        model = CometSparkMate(
            vocab_size=256, n_embd=64, n_layer=2, n_head=4, n_kv_head=2,
            seq_len=64, max_position_embeddings=256,
            window_size=32, num_global_tokens=4,
            use_alibi=True, use_rope=False,
        )
        out = str(tmp_path / "mate_err")
        with pytest.raises(ValueError, match="未知 format"):
            model.save(out, format="bad")


# ---------------------------------------------------------------------------
# SubTask 10.1（续）: save_pretrained 默认 vn
# ---------------------------------------------------------------------------


class TestSavePretrainedFormat:
    """save_pretrained(format=...) 参数验证。"""

    def test_save_pretrained_default_generates_vn(self, tmp_path):
        """save_pretrained(dir) 不传 format，默认生成 model.vn + config.yml。"""
        from spark.small.model import CometSparkSmall

        model = CometSparkSmall()
        out_dir = str(tmp_path / "pretrained_default")
        model.save_pretrained(out_dir)

        assert os.path.isdir(out_dir)
        # 默认应生成 model.vn
        assert (tmp_path / "pretrained_default" / "model.vn").exists(), (
            "save_pretrained 默认应生成 model.vn"
        )
        # 不应生成 model.pt
        assert not (tmp_path / "pretrained_default" / "model.pt").exists(), (
            "save_pretrained 默认 format 是 vn，不应生成 model.pt"
        )
        # config.yml 应存在
        assert (tmp_path / "pretrained_default" / "config.yml").exists(), (
            "save_pretrained 应生成 config.yml"
        )

    def test_save_pretrained_pt_format(self, tmp_path):
        """save_pretrained(dir, format='pt') 在 legacy 模式下生成 model.pt + config.yml。

        Part5K1.1：``use_vmpc=True``（默认）时强制 .vn，禁止 .pt；此处显式关闭
        VMPC 以验证 legacy .pt 保存路径仍然可用。
        """
        from spark.small.model import CometSparkSmall

        model = CometSparkSmall()
        # Part5K1.1：legacy 模式才允许 .pt
        model.config.use_vmpc = False
        out_dir = str(tmp_path / "pretrained_pt")
        model.save_pretrained(out_dir, format="pt")

        assert (tmp_path / "pretrained_pt" / "model.pt").exists()
        assert (tmp_path / "pretrained_pt" / "config.yml").exists()
        assert not (tmp_path / "pretrained_pt" / "model.vn").exists()


# ---------------------------------------------------------------------------
# SubTask 10.2: migrate_checkpoint_dir 工具函数
# ---------------------------------------------------------------------------


class TestMigrateCheckpointDir:
    """migrate_checkpoint_dir 测试。"""

    def test_migrate_small_renames_old_dir(self, tmp_path, monkeypatch):
        """目录迁移：checkpoints_small/ → mf_small/，文件保留。"""
        from verse_infra.verse_trainer.checkpoint_utils import migrate_checkpoint_dir

        # chdir 到 tmp_path，让相对路径生效
        monkeypatch.chdir(tmp_path)

        # 准备旧目录 + 文件
        old_dir = tmp_path / "checkpoints_small"
        old_dir.mkdir()
        (old_dir / "model.pt").write_bytes(b"fake model data")
        (old_dir / "config.yml").write_text("arch: versenex\n")

        # 调用迁移
        result = migrate_checkpoint_dir("mf_small", "small")

        assert result == "mf_small"
        # 旧目录应被重命名
        assert not old_dir.exists(), "旧目录 checkpoints_small/ 应被重命名"
        # 新目录应存在，且文件保留
        new_dir = tmp_path / "mf_small"
        assert new_dir.exists()
        assert (new_dir / "model.pt").read_bytes() == b"fake model data"
        assert (new_dir / "config.yml").read_text() == "arch: versenex\n"

    def test_migrate_target_exists_no_migrate(self, tmp_path, monkeypatch):
        """目标已存在（非空）不迁移。"""
        from verse_infra.verse_trainer.checkpoint_utils import migrate_checkpoint_dir

        monkeypatch.chdir(tmp_path)

        # 目标目录已存在且非空
        new_dir = tmp_path / "mf_small"
        new_dir.mkdir()
        (new_dir / "existing.pt").write_bytes(b"existing")

        # 旧目录也存在
        old_dir = tmp_path / "checkpoints_small"
        old_dir.mkdir()
        (old_dir / "old.pt").write_bytes(b"old data")

        result = migrate_checkpoint_dir("mf_small", "small")

        assert result == "mf_small"
        # 旧目录应保留（未迁移）
        assert old_dir.exists(), "目标已存在，旧目录不应被迁移"
        assert (old_dir / "old.pt").exists()
        # 目标目录内容不变
        assert (new_dir / "existing.pt").read_bytes() == b"existing"
        # 旧目录的文件不应出现在目标目录
        assert not (new_dir / "old.pt").exists()

    def test_migrate_no_old_dir_returns_target(self, tmp_path, monkeypatch):
        """无旧目录不迁移，返回目标目录，不报错。"""
        from verse_infra.verse_trainer.checkpoint_utils import migrate_checkpoint_dir

        monkeypatch.chdir(tmp_path)

        # 无旧目录，也无目标目录
        result = migrate_checkpoint_dir("mf_small", "small")

        assert result == "mf_small"
        # 不应创建任何目录（迁移函数只做 rename，不 mkdir）
        assert not (tmp_path / "mf_small").exists()
        assert not (tmp_path / "checkpoints_small").exists()

    def test_migrate_mate_level(self, tmp_path, monkeypatch):
        """mate 级别迁移：checkpoints_mate/ → mf_mate/。"""
        from verse_infra.verse_trainer.checkpoint_utils import migrate_checkpoint_dir

        monkeypatch.chdir(tmp_path)

        old_dir = tmp_path / "checkpoints_mate"
        old_dir.mkdir()
        (old_dir / "mate_model.pt").write_bytes(b"mate model")

        result = migrate_checkpoint_dir("mf_mate", "mate")

        assert result == "mf_mate"
        assert not old_dir.exists()
        new_dir = tmp_path / "mf_mate"
        assert new_dir.exists()
        assert (new_dir / "mate_model.pt").read_bytes() == b"mate model"

    def test_migrate_empty_save_dir_defaults_to_mf_level(
        self, tmp_path, monkeypatch
    ):
        """save_dir 为空时，默认目标为 mf_{model_level}。"""
        from verse_infra.verse_trainer.checkpoint_utils import migrate_checkpoint_dir

        monkeypatch.chdir(tmp_path)

        # 准备旧目录
        old_dir = tmp_path / "checkpoints_small"
        old_dir.mkdir()
        (old_dir / "model.pt").write_bytes(b"data")

        # save_dir 传空字符串
        result = migrate_checkpoint_dir("", "small")

        assert result == "mf_small"
        assert not old_dir.exists()
        assert (tmp_path / "mf_small" / "model.pt").exists()

    def test_migrate_none_save_dir_defaults_to_mf_level(
        self, tmp_path, monkeypatch
    ):
        """save_dir 为 None 时，默认目标为 mf_{model_level}。"""
        from verse_infra.verse_trainer.checkpoint_utils import migrate_checkpoint_dir

        monkeypatch.chdir(tmp_path)

        old_dir = tmp_path / "checkpoints_mate"
        old_dir.mkdir()
        (old_dir / "model.pt").write_bytes(b"mate")

        result = migrate_checkpoint_dir(None, "mate")

        assert result == "mf_mate"
        assert not old_dir.exists()
        assert (tmp_path / "mf_mate" / "model.pt").exists()

    def test_migrate_emits_deprecation_warning(
        self, tmp_path, monkeypatch
    ):
        """迁移时发出 DeprecationWarning（Part5K1 重命名提示）。"""
        from verse_infra.verse_trainer.checkpoint_utils import migrate_checkpoint_dir

        monkeypatch.chdir(tmp_path)

        old_dir = tmp_path / "checkpoints_small"
        old_dir.mkdir()
        (old_dir / "model.pt").write_bytes(b"data")

        with pytest.warns(DeprecationWarning, match="Part5K1 重命名"):
            migrate_checkpoint_dir("mf_small", "small")

    def test_migrate_absolute_path(self, tmp_path):
        """save_dir 为绝对路径时，旧目录解析为同级目录。"""
        from verse_infra.verse_trainer.checkpoint_utils import migrate_checkpoint_dir

        # 用绝对路径，无需 chdir
        old_dir = tmp_path / "checkpoints_small"
        old_dir.mkdir()
        (old_dir / "model.pt").write_bytes(b"abs data")

        new_dir_abs = str(tmp_path / "mf_small")
        result = migrate_checkpoint_dir(new_dir_abs, "small")

        assert result == new_dir_abs
        assert not old_dir.exists()
        assert (tmp_path / "mf_small" / "model.pt").read_bytes() == b"abs data"

    def test_migrate_absolute_path_target_exists_no_migrate(self, tmp_path):
        """绝对路径：目标已存在不迁移。"""
        from verse_infra.verse_trainer.checkpoint_utils import migrate_checkpoint_dir

        # 目标已存在
        new_dir = tmp_path / "mf_small"
        new_dir.mkdir()
        (new_dir / "existing.pt").write_bytes(b"existing")

        # 旧目录也存在
        old_dir = tmp_path / "checkpoints_small"
        old_dir.mkdir()
        (old_dir / "old.pt").write_bytes(b"old")

        new_dir_abs = str(tmp_path / "mf_small")
        result = migrate_checkpoint_dir(new_dir_abs, "small")

        assert result == new_dir_abs
        assert old_dir.exists(), "目标已存在，旧目录不应被迁移"
        assert (new_dir / "existing.pt").exists()
        assert not (new_dir / "old.pt").exists()

    def test_migrate_empty_old_dir_no_migrate(self, tmp_path, monkeypatch):
        """旧目录存在但为空时，不迁移（避免空目录噪声）。"""
        from verse_infra.verse_trainer.checkpoint_utils import migrate_checkpoint_dir

        monkeypatch.chdir(tmp_path)

        # 旧目录存在但为空
        old_dir = tmp_path / "checkpoints_small"
        old_dir.mkdir()
        assert os.listdir(old_dir) == []  # 确认空

        result = migrate_checkpoint_dir("mf_small", "small")

        assert result == "mf_small"
        # 空旧目录不迁移（保留原状）
        assert old_dir.exists(), "空旧目录不应触发迁移"
        assert not (tmp_path / "mf_small").exists()


# ---------------------------------------------------------------------------
# 集成：migrate_checkpoint_dir 在 verse_infra.verse_trainer 包可导入
# ---------------------------------------------------------------------------


class TestPackageExports:
    """验证 migrate_checkpoint_dir 在包层级可导入（推荐路径）。"""

    def test_import_from_verse_infra_top_level(self):
        """from verse_infra.verse_trainer import migrate_checkpoint_dir 可用。"""
        from verse_infra.verse_trainer import migrate_checkpoint_dir
        assert callable(migrate_checkpoint_dir)

    def test_import_from_trainer_module(self):
        """from verse_infra.verse_trainer.trainer import migrate_checkpoint_dir 可用。"""
        from verse_infra.verse_trainer.trainer import migrate_checkpoint_dir
        assert callable(migrate_checkpoint_dir)

    def test_import_from_checkpoint_utils_module(self):
        """from verse_infra.verse_trainer.checkpoint_utils import migrate_checkpoint_dir 可用。"""
        from verse_infra.verse_trainer.checkpoint_utils import migrate_checkpoint_dir
        assert callable(migrate_checkpoint_dir)

    def test_shim_path_import_with_deprecation_warning(self):
        """shim 路径 from verse_trainer import migrate_checkpoint_dir 仍可用（带 DeprecationWarning）。"""
        import warnings

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            # 清理 shim 缓存，确保重新走 shim（不影响 verse_infra.verse_trainer）
            sys.modules.pop("verse_trainer", None)
            from verse_trainer import migrate_checkpoint_dir  # noqa: E402
            assert callable(migrate_checkpoint_dir)
            # shim 应发 DeprecationWarning
            assert any(issubclass(x.category, DeprecationWarning) for x in w), (
                f"shim 应发 DeprecationWarning，得到："
                f"{[(x.category, str(x.message)) for x in w]}"
            )


# ---------------------------------------------------------------------------
# 端到端集成：train() 入口实际触发迁移
# ---------------------------------------------------------------------------


class TestTrainIntegration:
    """端到端集成测试：验证 train() 入口实际调用 migrate_checkpoint_dir。

    用真实的小配置（cometspark_small.yml）+ max_steps=1 + 跳过评估，
    验证旧 checkpoints_small/ 目录在 train() 启动时被迁移为 mf_small/。
    """

    def test_train_triggers_migration_small(self, tmp_path, monkeypatch):
        """train() 入口自动迁移 checkpoints_small/ → mf_small/。"""
        yml_src = _REPO_ROOT / "spark" / "small" / "config" / "cometspark_small.yml"
        assert yml_src.exists(), f"测试配置文件不存在：{yml_src}"

        # chdir 到 tmp_path，让相对路径迁移生效（模拟用户在 base_dir 下运行）
        monkeypatch.chdir(tmp_path)

        # 在 tmp_path 下创建旧 checkpoint 目录 + marker 文件
        old_dir = tmp_path / "checkpoints_small"
        old_dir.mkdir()
        marker_content = b"old checkpoint marker"
        (old_dir / "marker.txt").write_bytes(marker_content)

        # 调用 train()：最小步数 + 跳过评估 + 静默
        from verse_infra.verse_trainer import train

        result = train(
            config_path=str(yml_src),
            base_dir=str(tmp_path),
            max_steps_override=1,
            quiet=True,
            eval_after=False,
        )

        # 旧目录应被迁移（重命名）
        assert not old_dir.exists(), (
            "train() 应迁移旧 checkpoints_small/ 目录"
        )
        # 新目录应存在，且 marker 文件保留（证明是迁移来的，不是 train 新建的）
        new_dir = tmp_path / "mf_small"
        assert new_dir.exists()
        assert (new_dir / "marker.txt").read_bytes() == marker_content, (
            "迁移后 marker 文件应保留在新目录 mf_small/"
        )
        # train() 返回的 checkpoint_dir 应指向 mf_small
        assert result["checkpoint_dir"].endswith("mf_small"), (
            f"checkpoint_dir 应指向 mf_small，实际：{result['checkpoint_dir']}"
        )

    def test_train_no_old_dir_works(self, tmp_path, monkeypatch):
        """无旧目录时 train() 正常运行（不报错）。"""
        yml_src = _REPO_ROOT / "spark" / "small" / "config" / "cometspark_small.yml"
        assert yml_src.exists()

        monkeypatch.chdir(tmp_path)
        # 不创建旧目录

        from verse_infra.verse_trainer import train

        result = train(
            config_path=str(yml_src),
            base_dir=str(tmp_path),
            max_steps_override=1,
            quiet=True,
            eval_after=False,
        )

        # 无旧目录时，train() 正常创建 mf_small
        assert (tmp_path / "mf_small").exists()
        assert result["checkpoint_dir"].endswith("mf_small")
        # 旧目录不应被创建
        assert not (tmp_path / "checkpoints_small").exists()
