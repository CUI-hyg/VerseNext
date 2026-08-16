"""检查点管理（CheckpointManager / ResumeManager）。

Part1 Task2 拆分自 training.py：与训练循环解耦的持久化逻辑集中在此模块，
training.py 顶部 re-export 以保持 ``from verse_torch.training import ...`` 兼容。
"""

from __future__ import annotations

import json
import os
import pickle
import subprocess
import sys
from collections import namedtuple
from pathlib import Path
from typing import Any, Optional

import numpy as np

from .tensor import Tensor
from .vn_format import VNFileWriter, VNFileReader

__all__ = ["CheckpointManager", "ResumeManager", "ResumeState"]


# ---------------------------------------------------------------------------
# Task 2.4: CheckpointManager
# ---------------------------------------------------------------------------


def _to_serializable(obj: Any) -> Any:
    """递归把 Tensor / np.ndarray 等转为 pickle 友好的形式。"""
    if isinstance(obj, Tensor):
        return {"__tensor__": True, "data": obj.data, "requires_grad": obj.requires_grad}
    if isinstance(obj, np.ndarray):
        return obj  # pickle 原生支持 ndarray
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, dict):
        return {k: _to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_serializable(v) for v in obj]
    return obj


def _from_serializable(obj: Any) -> Any:
    """递归把序列化形式还原（必要时把 dict 还原成 Tensor）。"""
    if isinstance(obj, dict):
        if obj.get("__tensor__") is True:
            return Tensor(obj["data"], requires_grad=bool(obj.get("requires_grad", False)))
        return {k: _from_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_from_serializable(v) for v in obj]
    return obj


def _pt_to_vn_worker(pt_path: str, vn_path: str) -> None:
    """Part5K1.5：subprocess worker —— 将 ``best.pt`` 转为 ``best.vn``。

    由 :meth:`CheckpointManager._async_save_vn` 通过 ``subprocess.Popen``
    在独立进程中执行，避免阻塞主训练进程。

    流程：
    1. 读取 ``pt_path`` 的 pickle payload（经 ``_from_serializable`` 还原）
    2. 调用 :class:`CheckpointManager` 的 ``_save_vn`` 写 ``vn_path``（原子写）
    3. 失败时：仅打印错误到 stderr，不抛异常（``best.pt`` 已作为备份保留）

    Args:
        pt_path: 源 ``.pt`` 文件路径（pickle，由 ``_atomic_save`` 写入）
        vn_path: 目标 ``.vn`` 文件路径（VNFileWriter 原生格式）
    """
    import traceback

    try:
        # 1. 读取 .pt payload
        with open(pt_path, "rb") as f:
            raw = pickle.load(f)
        state = _from_serializable(raw)

        # 2. 提取子状态（与 save_best 的 format="vn" 路径一致）
        #    _atomic_save 时 _to_serializable 已把 Tensor 转 dict，
        #    state 可能含 model_state_dict / training_state / optimizer_state / extra_state
        #    以及顶层标准字段（step / val_loss / best_val_loss 等）
        #    _save_vn 内部会从 state 中自动提取 model_state_dict 和标准字段，
        #    所以这里直接传整个 state，让 _save_vn 做提取。
        training_state = None
        optimizer_state = None
        extra_state = None
        if isinstance(state, dict):
            training_state = state.get("training_state")
            optimizer_state = state.get("optimizer_state")
            extra_state = state.get("extra_state")

        # 3. 用 CheckpointManager._save_vn 原子写 .vn
        #    构造一个临时 manager 仅复用 _save_vn 方法（避免重复实现）
        #    use_vmpc=True, format="vn" 保证 _resolve_path 正确
        mgr = CheckpointManager(
            save_dir=os.path.dirname(os.path.abspath(vn_path)) or ".",
            best_path=vn_path,
            last_path=vn_path + ".last",  # 占位，不会用到
            format="vn",
            use_vmpc=True,
            async_vn=False,
        )
        # 直接传整个 state：_save_vn 会从中提取 model_state_dict 和标准字段
        mgr._save_vn(
            Path(vn_path),
            state,
            training_state=training_state,
            optimizer_state=optimizer_state,
            extra_state=extra_state,
        )
        print(
            f"[_pt_to_vn_worker] OK: {pt_path} → {vn_path}",
            file=sys.stderr,
            flush=True,
        )
    except Exception:
        # 失败时 best.pt 已保留作为备份，仅打印错误
        print(
            f"[_pt_to_vn_worker] FAIL: 转换 {pt_path} → {vn_path} 失败：\n"
            f"{traceback.format_exc()}",
            file=sys.stderr,
            flush=True,
        )
        # 不抛异常：subprocess 退出码非 0 也不影响主训练（best.pt 已备份）


class CheckpointManager:
    """检查点管理器：保存/加载 best 与 last 模型状态。

    Args:
        save_dir: 保存目录
        best_path: 自定义 best 文件路径（默认 save_dir/best.pt 或 save_dir/best.vn）
        last_path: 自定义 last 文件路径（默认 save_dir/last.pt 或 save_dir/last.vn）
        format: 检查点格式，"auto" | "vn" | "pt"（默认 "auto"）
            - "auto": 根据 ``use_vmpc`` 自动选择（True → "vn"，否则 "pt"）
            - "vn": 写 .vn 文件（Part5K1.3 Task 4 原生 VNFileWriter 调用，
              支持 model + training_state + optimizer_state + extra_state）
            - "pt": 写 .pt 文件（pickle，向后兼容）
        use_vmpc: 是否使用 VMPC（决定 "auto" 时的最终格式）；True 时强制 "vn"，
            与 ``CometSparkSmallLM._enforce_vn_format`` 一致

    用法:
        >>> ckpt = CheckpointManager("./checkpoints")
        >>> ckpt.save_best({"model": model.state_dict(), "val_loss": 0.5})
        >>> state = ckpt.load_best()
        >>> # 完整字段（含 training_state / optimizer_state 等）
        >>> full = ckpt.load_best_full()
    """

    def __init__(
        self,
        save_dir,
        best_path: Optional[os.PathLike] = None,
        last_path: Optional[os.PathLike] = None,
        format: str = "auto",
        use_vmpc: bool = False,
        async_vn: bool = False,
    ):
        """Part5K1.5：async_vn 模式。

        ``async_vn=True`` 时（需 ``use_vmpc=True``）：
        - ``last`` 始终用 ``.pt``（快速缓存，不阻塞训练）
        - ``best`` 先快速保存 ``best.pt``（备份），再通过 subprocess 异步保存 ``best.vn``
        - subprocess 失败时 ``best.pt`` 保留作为备份
        """
        # Part5K1.5：async_vn 模式校验
        if async_vn and not use_vmpc:
            raise ValueError("async_vn=True 需要 use_vmpc=True")
        self.async_vn = async_vn

        # 校验 format 取值
        if format not in ("auto", "vn", "pt"):
            raise ValueError(
                f"format 必须是 'auto' / 'vn' / 'pt'，得到 {format!r}"
            )
        # 解析 "auto" → 根据 use_vmpc 选择最终格式
        if format == "auto":
            resolved_format = "vn" if use_vmpc else "pt"
        else:
            resolved_format = format
        # use_vmpc=True 时强制 .vn（format="pt" 视为冲突）
        if use_vmpc and resolved_format == "pt":
            raise ValueError(
                "use_vmpc=True 时强制 .vn 格式（format='vn' 或 'auto'），"
                "请改为 format='vn' 或 format='auto'"
            )

        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.format = resolved_format
        self.use_vmpc = use_vmpc

        # 默认路径根据 format 选择扩展名（SubTask 4.3: _resolve_path 自动加扩展名）
        self.best_path = (
            Path(best_path) if best_path is not None
            else self._resolve_path("best")
        )
        self.last_path = (
            Path(last_path) if last_path is not None
            else self._resolve_path("last")
        )

    # ------------------------------------------------------------------
    # 路径解析（SubTask 4.3）
    # ------------------------------------------------------------------

    def _resolve_path(self, name: str) -> Path:
        """根据 format 返回 ``save_dir/{name}.{ext}`` 路径。

        format="vn" → ``save_dir/{name}.vn``；format="pt" → ``save_dir/{name}.pt``。

        Args:
            name: 文件名主体（如 ``"best"`` / ``"last"``）
        """
        ext = ".vn" if self.format == "vn" else ".pt"
        return self.save_dir / f"{name}{ext}"

    # ------------------------------------------------------------------
    # 原子写（SubTask 2.1 + Task 4: format="vn" 走 VNFileWriter）
    # ------------------------------------------------------------------

    def _atomic_save(self, final_path: Path, state: dict) -> None:
        """原子写 .pt：先写 ``.tmp`` 临时文件（pickle），再 ``os.replace`` 重命名。

        写入失败时清理 ``.tmp`` 文件，避免残留半截文件影响下次启动。
        仅用于 format="pt" 路径；format="vn" 路径走 :meth:`_save_vn`。
        """
        tmp_path = final_path.parent / (final_path.name + ".tmp")
        try:
            payload = _to_serializable(state)
            with open(tmp_path, "wb") as f:
                pickle.dump(payload, f)
            os.replace(tmp_path, final_path)
        except Exception:
            # 清理 .tmp（若存在），避免残留半截文件；原 final_path 未被触碰
            try:
                tmp_path.unlink()
            except FileNotFoundError:
                pass
            raise

    def _save_vn(
        self,
        final_path: Path,
        state: dict,
        training_state: Optional[dict] = None,
        optimizer_state: Optional[dict] = None,
        extra_state: Optional[Any] = None,
    ) -> None:
        """原子写 .vn：调用 ``VNFileWriter`` 写入完整状态（Part5K1.3 Task 4.1）。

        写入流程：
        1. 从 ``state`` 中提取模型权重（``state["model_state_dict"]`` 或 ``state``
           本身若全为 ndarray）；非权重字段（step / val_loss 等）合并到
           ``training_state``。
        2. 用 ``VNFileWriter`` 写到 ``{final_path}.vn.tmp`` 临时文件。
        3. ``os.replace`` 原子重命名为目标 ``.vn`` 文件。
        4. 失败时清理 ``.tmp``，不影响已有 checkpoint。

        Args:
            final_path: 目标 ``.vn`` 文件路径
            state: 模型 state_dict 或含 ``model_state_dict`` 的状态字典
            training_state: 训练状态（step / epoch / best_val_loss 等）；
                若为 None，则从 ``state`` 中提取标准字段
            optimizer_state: optimizer 状态（AdamW m/v 等）
            extra_state: 额外任意状态（EMA / grad scaler 等）
        """
        # 1. 提取模型权重
        if isinstance(state, dict) and "model_state_dict" in state:
            model_weights = state.get("model_state_dict") or {}
        elif isinstance(state, dict) and state and all(
            isinstance(v, (np.ndarray, np.generic)) for v in state.values()
        ):
            # 纯 state_dict {name: ndarray}
            model_weights = state
        else:
            # state 既不含 model_state_dict 也不是纯 state_dict：
            # 视为无权重 checkpoint（仅 training/optimizer/extra_state）
            model_weights = {}

        # 2. 构建 training_state：显式参数优先，缺失字段从 state 提取
        ts: dict = dict(training_state) if training_state else {}
        if isinstance(state, dict):
            for key in ("step", "epoch", "best_val_loss", "val_loss", "train_loss"):
                if key in state and key not in ts:
                    try:
                        ts[key] = float(state[key]) if key in ("best_val_loss", "val_loss", "train_loss") else state[key]
                    except (TypeError, ValueError):
                        ts[key] = state[key]

        # 3. 原子写：先写 .vn.tmp，再 os.replace
        tmp_path = final_path.parent / (final_path.name + ".tmp")
        try:
            with VNFileWriter(
                str(tmp_path), arch="checkpoint", config={},
            ) as w:
                # write_weights 必须调用（VNFileReader.read_weights 期望权重条目存在）
                w.write_weights(model_weights)
                if ts:
                    w.write_training_state(ts)
                if optimizer_state is not None:
                    w.write_optimizer_state(optimizer_state)
                if extra_state is not None:
                    w.write_extra_state(extra_state)
            os.replace(tmp_path, final_path)
        except Exception:
            # 清理 .tmp（若存在），避免残留半截文件；原 final_path 未被触碰
            try:
                if tmp_path.exists():
                    tmp_path.unlink()
            except FileNotFoundError:
                pass
            raise

    def _load_vn(self, path: Path) -> dict:
        """读取 .vn 文件并返回统一 dict（Part5K1.3 Task 4.2）。

        调用 ``VNFileReader`` 读取 model weights + training_state +
        optimizer_state + extra_state，组装为统一 dict。
        v1 文件的缺失字段返回 None（向后兼容）。

        返回字段:
            - ``model_state_dict``: Optional[dict] —— 模型权重（v1/v2 均有）
            - ``training_state``: Optional[dict]
            - ``optimizer_state``: Optional[dict]
            - ``extra_state``: Optional[Any]
            - ``step``: Optional[int] —— 从 training_state 提取
            - ``best_val_loss``: Optional[float] —— 从 training_state 提取
              （兼容 ``val_loss`` 字段）
        """
        with VNFileReader(str(path)) as r:
            try:
                model_state_dict = r.read_weights()
            except (ValueError, KeyError):
                # .vn 文件缺少权重条目（极端罕见：仅写 training/optimizer state）
                model_state_dict = None
            training_state = r.read_training_state()
            optimizer_state = r.read_optimizer_state()
            extra_state = r.read_extra_state()

        # 从 training_state 提取标准字段（step / best_val_loss）
        step: Optional[int] = None
        best_val_loss: Optional[float] = None
        if training_state:
            step_val = training_state.get("step")
            if step_val is not None:
                try:
                    step = int(step_val)
                except (TypeError, ValueError):
                    step = None
            bvl = training_state.get("best_val_loss")
            if bvl is None:
                # 兼容旧文件可能用 val_loss 字段
                bvl = training_state.get("val_loss")
            if bvl is not None:
                try:
                    best_val_loss = float(bvl)
                except (TypeError, ValueError):
                    best_val_loss = None

        return {
            "model_state_dict": model_state_dict,
            "training_state": training_state,
            "optimizer_state": optimizer_state,
            "extra_state": extra_state,
            "step": step,
            "best_val_loss": best_val_loss,
        }

    # ------------------------------------------------------------------
    # save_best / save_last（SubTask 2.3：扩展签名 + 原子写）
    # ------------------------------------------------------------------

    def save_best(
        self,
        state: dict,
        training_state: Optional[dict] = None,
        optimizer_state: Optional[dict] = None,
        extra_state: Optional[Any] = None,
        step: Optional[int] = None,
    ) -> None:
        """保存最佳模型状态到 best.pt（或 best.vn），原子写。

        Part5K1.5：``async_vn=True`` 时：
        1. 先快速保存 ``best.pt``（pickle，作为备份/缓存）
        2. 通过 subprocess 异步保存 ``best.vn``（不阻塞训练）
        3. subprocess 失败时 ``best.pt`` 保留作为备份

        Args:
            state: 模型 state_dict 或任意可 pickle 的状态字典
            training_state: 训练状态（step / epoch / best_val_loss 等）；
                format="pt" 时忽略（保持原行为），format="vn" 时写入
                ``training_state.json``
            optimizer_state: optimizer 状态（AdamW m/v 等）；
                format="pt" 时忽略，format="vn" 时写入 ``optimizer_state.pkl``
            extra_state: 额外任意状态（EMA / grad scaler 等）；
                format="pt" 时忽略，format="vn" 时写入 ``extra_state.pkl``
            step: 当前训练步数（便于 load_best_full 提取）

        Note:
            - format="pt" 路径仅 pickle ``state`` 参数（保持原行为），忽略额外参数；
            - format="vn" 路径调用 ``VNFileWriter`` 写入完整状态（Part5K1.3 Task 4.1），
              从 ``state`` 中提取 ``model_state_dict``（或纯 state_dict）作为权重，
              其余标准字段（step / val_loss 等）合并到 training_state。
        """
        if self.async_vn:
            # Part5K1.5：async_vn 模式
            # 1. 先快速保存 best.pt（pickle，作为备份）
            best_pt_path = self.save_dir / "best.pt"
            self._atomic_save(best_pt_path, state)
            # 2. subprocess 异步保存 best.vn（不阻塞训练）
            self._async_save_vn(best_pt_path, self.save_dir / "best.vn")
        elif self.format == "vn":
            self._save_vn(
                self.best_path, state, training_state,
                optimizer_state, extra_state,
            )
        else:
            self._atomic_save(self.best_path, state)

    def save_last(
        self,
        state: dict,
        training_state: Optional[dict] = None,
        optimizer_state: Optional[dict] = None,
        extra_state: Optional[Any] = None,
        step: Optional[int] = None,
    ) -> None:
        """保存最近一次检查点到 last.pt（或 last.vn），原子写。

        Part5K1.5：``async_vn=True`` 时始终用 ``.pt``（快速缓存，不阻塞训练）。

        参数语义同 :meth:`save_best`。
        """
        if self.async_vn:
            # Part5K1.5：async_vn 模式 → last 始终用 .pt
            last_pt_path = self.save_dir / "last.pt"
            self._atomic_save(last_pt_path, state)
        elif self.format == "vn":
            self._save_vn(
                self.last_path, state, training_state,
                optimizer_state, extra_state,
            )
        else:
            self._atomic_save(self.last_path, state)

    def _async_save_vn(self, pt_path: Path, vn_path: Path) -> None:
        """Part5K1.5：通过 subprocess 异步将 best.pt 转为 best.vn。

        - 不阻塞主训练进程（subprocess.Popen 后立即返回）
        - 失败时 best.pt 保留作为备份（不抛异常，仅打印警告）
        - 使用 ``__main__`` 级别的 ``_pt_to_vn_worker`` 作为子进程入口
        """
        import subprocess
        import sys

        # 构造子进程命令：调用模块级 worker 函数
        script = (
            "import sys, pickle;\n"
            "sys.path.insert(0, {paths!r});\n"
            "from verse_torch.checkpoint import _pt_to_vn_worker;\n"
            "_pt_to_vn_worker({pt!r}, {vn!r});\n"
        ).format(
            paths=[str(p) for p in sys.path if p],
            pt=str(pt_path),
            vn=str(vn_path),
        )

        try:
            # 启动子进程（不等待完成，异步执行）
            proc = subprocess.Popen(
                [sys.executable, "-c", script],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            # 不等待 proc 完成（异步）。进程结束后 OS 自动回收。
            # 记录 PID 便于调试（可选）
            if not hasattr(self, "_async_procs"):
                self._async_procs = []
            self._async_procs.append(proc)
        except Exception as e:
            # subprocess 启动失败：best.pt 已保存，作为备份
            import warnings
            warnings.warn(
                f"Part5K1.5: async best.vn 保存失败（best.pt 已作为备份）：{e}",
                RuntimeWarning,
                stacklevel=2,
            )

    # ------------------------------------------------------------------
    # load_best / load_last（向后兼容：返回保存的 state dict）
    # ------------------------------------------------------------------

    def load_best(self) -> dict:
        """从 best.pt（或 best.vn）读取并返回状态字典（向后兼容）。

        - format="pt"：返回 ``save_best(state)`` 时传入的 ``state`` 字典
          （经 pickle 序列化往返还原），旧代码 ``ckpt.load_best()["step"]`` 等
          访问保持不变。
        - format="vn"：调用 ``VNFileReader`` 读取，返回包含 ``model_state_dict``
          + ``training_state`` + ``optimizer_state`` + ``extra_state`` 的统一 dict，
          并把 ``training_state`` 中的标准字段（step / val_loss 等）展开到顶层
          （便于 ``load_best()["step"]`` 这样的旧式访问）。
        - Part5K1.5 ``async_vn=True``：优先尝试 ``best.vn``，不存在时回退到
          ``best.pt``（异步保存未完成或失败时的备份）。
        """
        if self.async_vn:
            # Part5K1.5：async_vn 模式 — 优先 best.vn，回退 best.pt
            if self.best_path.exists():
                return self._load_vn_state(self.best_path)
            best_pt = self.save_dir / "best.pt"
            if best_pt.exists():
                with open(best_pt, "rb") as f:
                    payload = pickle.load(f)
                return _from_serializable(payload)
            raise FileNotFoundError(
                f"未找到 best checkpoint：{self.best_path} 和 {best_pt} 均不存在"
            )
        if self.format == "vn":
            return self._load_vn_state(self.best_path)
        with open(self.best_path, "rb") as f:
            payload = pickle.load(f)
        return _from_serializable(payload)

    def load_last(self) -> dict:
        """从 last.pt（或 last.vn）读取并返回状态字典（向后兼容）。

        Part5K1.5 ``async_vn=True``：始终从 ``last.pt`` 读取（async_vn 模式下
        last 始终用 .pt 快速缓存）。
        """
        if self.async_vn:
            # Part5K1.5：async_vn 模式 — last 始终是 .pt
            last_pt = self.save_dir / "last.pt"
            with open(last_pt, "rb") as f:
                payload = pickle.load(f)
            return _from_serializable(payload)
        if self.format == "vn":
            return self._load_vn_state(self.last_path)
        with open(self.last_path, "rb") as f:
            payload = pickle.load(f)
        return _from_serializable(payload)

    # ------------------------------------------------------------------
    # load_best_full / load_last_full（SubTask 2.4：返回统一 dict）
    # ------------------------------------------------------------------

    def load_best_full(self) -> dict:
        """返回统一 dict，包含标准字段（缺失字段为 None）。

        返回字段:
            - model_state_dict: Optional[dict]  —— 模型权重（缺失为 None）
            - training_state: Optional[dict]   —— 训练状态（缺失为 None）
            - optimizer_state: Optional[dict]  —— optimizer 状态（缺失为 None）
            - extra_state: Optional[Any]       —— 额外状态（缺失为 None）
            - step: Optional[int]              —— 训练步数（缺失为 None）
            - best_val_loss: Optional[float]   —— 最佳验证 loss（缺失为 None，
              兼容旧文件的 ``val_loss`` 字段）

        format="vn" 路径调用 ``VNFileReader``；format="pt" 路径从 pickle 中提取。

        Part5K1.5 ``async_vn=True``：优先 ``best.vn``，不存在时回退 ``best.pt``。
        """
        if self.async_vn:
            # Part5K1.5：async_vn 模式 — 优先 best.vn，回退 best.pt
            if self.best_path.exists():
                return self._load_full(self.best_path)
            best_pt = self.save_dir / "best.pt"
            if best_pt.exists():
                return self._load_full(best_pt)
            raise FileNotFoundError(
                f"未找到 best checkpoint：{self.best_path} 和 {best_pt} 均不存在"
            )
        return self._load_full(self.best_path)

    def load_last_full(self) -> dict:
        """同 :meth:`load_best_full` 但读 last 文件。

        Part5K1.5 ``async_vn=True``：始终读 ``last.pt``（async_vn 模式下
        last 始终用 .pt 快速缓存）。
        """
        if self.async_vn:
            # Part5K1.5：async_vn 模式 — last 始终是 .pt
            last_pt = self.save_dir / "last.pt"
            return self._load_full(last_pt)
        return self._load_full(self.last_path)

    def _load_vn_state(self, path: Path) -> dict:
        """读取 .vn 文件并返回向后兼容的 state dict（Part5K1.3 Task 4.2）。

        与 :meth:`_load_vn` 不同，本方法把 ``training_state`` 中的标准字段
        （step / val_loss / best_val_loss 等）展开到顶层，便于 ``load_best()["step"]``
        这样的旧式访问。
        """
        full = self._load_vn(path)
        # 展开标准字段到顶层
        result: dict = {
            "model_state_dict": full["model_state_dict"],
            "training_state": full["training_state"],
            "optimizer_state": full["optimizer_state"],
            "extra_state": full["extra_state"],
            "step": full["step"],
            "best_val_loss": full["best_val_loss"],
        }
        # 兼容 val_loss 别名（旧代码可能访问 load_best()["val_loss"]）
        if full["best_val_loss"] is not None:
            result["val_loss"] = full["best_val_loss"]
        return result

    def _load_full(self, path: Path) -> dict:
        """读取 checkpoint 并返回统一 dict（缺失字段为 None）。

        - ``.vn`` 文件：调用 :meth:`_load_vn`（VNFileReader 读取）
        - ``.pt`` 文件：从 pickle payload 中提取标准字段

        Part5K1.5：根据**文件扩展名**判断格式（而非 ``self.format``），
        以支持 ``async_vn`` 模式下回退到 ``best.pt`` 的场景。
        """
        # Part5K1.5：根据文件扩展名判断格式（支持 async_vn 回退到 .pt）
        if str(path).endswith(".vn"):
            return self._load_vn(path)

        with open(path, "rb") as f:
            payload = pickle.load(f)
        state = _from_serializable(payload)
        if isinstance(state, dict):
            model_state_dict = state.get("model_state_dict", None)
            training_state = state.get("training_state", None)
            optimizer_state = state.get("optimizer_state", None)
            extra_state = state.get("extra_state", None)
            step = state.get("step", None)
            best_val_loss = state.get("best_val_loss", None)
            if best_val_loss is None:
                # 兼容旧文件可能用 val_loss 字段
                val_loss = state.get("val_loss", None)
                if val_loss is not None:
                    try:
                        best_val_loss = float(val_loss)
                    except (TypeError, ValueError):
                        best_val_loss = None
        else:
            # 非 dict 载荷（极罕见）：全部字段为 None
            model_state_dict = None
            training_state = None
            optimizer_state = None
            extra_state = None
            step = None
            best_val_loss = None
        return {
            "model_state_dict": model_state_dict,
            "training_state": training_state,
            "optimizer_state": optimizer_state,
            "extra_state": extra_state,
            "step": step,
            "best_val_loss": best_val_loss,
        }


# ---------------------------------------------------------------------------
# Part5K1.3 Task 6: ResumeState + ResumeManager 断点续训
# ---------------------------------------------------------------------------


ResumeState = namedtuple(
    "ResumeState",
    [
        "model_state_dict",    # Optional[dict]  —— 模型权重
        "optimizer_state",     # Optional[dict]  —— optimizer 状态（AdamW m/v 等）
        "step",                # Optional[int]   —— 当前训练步数
        "rng_state",           # Optional[Any]   —— numpy RandomState.get_state() 返回
        "best_val_loss",       # Optional[float] —— 最佳验证 loss
        "epoch",               # Optional[int]   —— 当前 epoch
        "patience_count",      # Optional[int]   —— EarlyStopping 已等待步数
    ],
)


class ResumeManager:
    """断点续训管理器（Part5K1.3 Task 6）。

    统一管理断点续训状态（model + optimizer + step + rng + best_val_loss +
    epoch + patience_count），通过 :class:`CheckpointManager` 写入/读取
    ``.vn`` checkpoint，不重复实现序列化逻辑。

    设计要点
    --------
    - **复用 CheckpointManager**：``save`` 调用 :meth:`CheckpointManager.save_best`
      写 ``.vn``（含 model + training_state + optimizer_state + extra_state）；
      ``load`` 调用 :meth:`CheckpointManager.load_best_full` 读取。
    - **向后兼容**：v1 ``.vn`` 文件 / 旧 ``.pt`` pickle 文件的缺失字段返回 None
      + 警告日志（``apply`` 时跳过 None 字段）。
    - **rng_state 存 extra_state**：numpy ``RandomState.get_state()`` 返回 tuple，
      非 JSON-able，故存到 ``extra_state.pkl``（pickle），不存 ``training_state.json``。
    - **apply 支持 Trainer / ParallelTrainerSafe**：仅恢复存在的字段，None 跳过。
    """

    # 默认 resume 文件名（与 ParallelTrainerSafe 旧 .pt resume 区分）
    DEFAULT_FILENAME = "resume.vn"
    # ParallelTrainerSafe 旧 resume 文件名（向后兼容回退）
    LEGACY_FILENAME = "resume.pt"

    # ------------------------------------------------------------------
    # save（SubTask 6.3）
    # ------------------------------------------------------------------

    @staticmethod
    def save(
        path: str,
        model,
        optimizer=None,
        step: Optional[int] = None,
        *,
        best_val_loss: Optional[float] = None,
        epoch: Optional[int] = None,
        patience_count: Optional[int] = None,
        rng_state: Optional[Any] = None,
        extra_state: Optional[Any] = None,
    ) -> str:
        """保存断点续训状态到 ``.vn`` checkpoint。

        调用 :meth:`CheckpointManager.save_best` 写 ``.vn`` 文件，含 model +
        optimizer + step + rng + best_val_loss + epoch + patience_count。

        Args:
            path: 目标 ``.vn`` 文件路径（或目录，目录下用默认文件名
                ``resume.vn``）。
            model: 模型对象（需有 ``state_dict()`` 方法）；None 不保存权重。
            optimizer: optimizer 对象（需有 ``state_dict()`` 方法）；None 不保存。
            step: 当前训练步数。
            best_val_loss: 最佳验证 loss。
            epoch: 当前 epoch。
            patience_count: EarlyStopping 已等待步数。
            rng_state: numpy ``RandomState.get_state()`` 返回值；None 不保存。
            extra_state: 额外任意状态（用户自定义，如 best_state_dict / history）。

        Returns:
            实际写入的 ``.vn`` 文件绝对路径。
        """
        # path 既可以是目录（用默认文件名）也可以是完整文件路径
        path_obj = Path(path)
        if path_obj.is_dir():
            save_dir = path_obj
            best_path = save_dir / ResumeManager.DEFAULT_FILENAME
        else:
            save_dir = path_obj.parent
            best_path = path_obj
        save_dir.mkdir(parents=True, exist_ok=True)

        # 提取 model state_dict
        model_sd = None
        if model is not None and hasattr(model, "state_dict"):
            try:
                model_sd = model.state_dict()
            except Exception as e:
                print(
                    f"[ResumeManager] 警告：获取 model state_dict 失败：{e}",
                    flush=True,
                )
                model_sd = None

        # 提取 optimizer state_dict
        # 支持 PyTorch 风格（state_dict()/load_state_dict()）与 verse_torch 风格
        # （self.state dict + self.param_groups）两种 optimizer
        opt_state = None
        if optimizer is not None:
            if hasattr(optimizer, "state_dict") and callable(optimizer.state_dict):
                try:
                    opt_state = optimizer.state_dict()
                except Exception as e:
                    print(
                        f"[ResumeManager] 警告：获取 optimizer state_dict 失败：{e}",
                        flush=True,
                    )
                    opt_state = None
            elif hasattr(optimizer, "state") and hasattr(optimizer, "param_groups"):
                # verse_torch Optimizer：直接序列化 state dict + param_groups 超参。
                # 注意：param_groups["params"] 是 Tensor 对象引用，Tensor 内部含
                # ``_backward`` lambda 不可 pickle，故剥离 params 引用，
                # 仅保留超参（lr / betas / eps / weight_decay 等）+ 参数数量占位。
                # state dict 按 ``id(p)`` 键控，仅同 session 内有效（跨 session 键
                # 不匹配新 params，apply 时仅恢复超参）。
                try:
                    safe_param_groups = []
                    for g in optimizer.param_groups:
                        g_copy = {k: v for k, v in g.items() if k != "params"}
                        # params 用占位索引列表，保持结构但不引用 Tensor
                        g_copy["params"] = list(range(len(g.get("params", []))))
                        safe_param_groups.append(g_copy)
                    opt_state = {
                        "state": optimizer.state,
                        "param_groups": safe_param_groups,
                    }
                except Exception as e:
                    print(
                        f"[ResumeManager] 警告：获取 optimizer state 失败：{e}",
                        flush=True,
                    )
                    opt_state = None

        # 构建 training_state（写入 .vn 的 training_state.json，仅 JSON-able 字段）
        training_state: dict = {}
        if step is not None:
            try:
                training_state["step"] = int(step)
            except (TypeError, ValueError):
                pass
        if best_val_loss is not None:
            try:
                training_state["best_val_loss"] = float(best_val_loss)
            except (TypeError, ValueError):
                pass
        if epoch is not None:
            try:
                training_state["epoch"] = int(epoch)
            except (TypeError, ValueError):
                pass
        if patience_count is not None:
            try:
                training_state["patience_count"] = int(patience_count)
            except (TypeError, ValueError):
                pass

        # rng_state 是 tuple（非 JSON-able），合并到 extra_state 用 pickle 序列化
        merged_extra: dict = dict(extra_state) if extra_state else {}
        if rng_state is not None:
            merged_extra["rng_state"] = rng_state

        # state 参数：CheckpointManager._save_vn 期望含 model_state_dict 或纯 state_dict
        state = {"model_state_dict": model_sd} if model_sd is not None else {}

        ckpt = CheckpointManager(
            save_dir=save_dir,
            best_path=best_path,
            format="vn",
            use_vmpc=True,  # 强制 .vn 格式
        )
        ckpt.save_best(
            state=state,
            training_state=training_state if training_state else None,
            optimizer_state=opt_state,
            extra_state=merged_extra if merged_extra else None,
            step=step,
        )
        return str(best_path.resolve())

    # ------------------------------------------------------------------
    # load（SubTask 6.4）
    # ------------------------------------------------------------------

    @staticmethod
    def load(path: str) -> ResumeState:
        """从 ``.vn`` / ``.pt`` checkpoint 读取断点续训状态。

        调用 :meth:`CheckpointManager.load_best_full` 读取，v1 ``.vn`` / 旧
        ``.pt`` 缺失字段返回 None + 警告日志。

        Args:
            path: ``.vn`` 文件路径（或兼容的 ``.pt`` pickle 文件路径）。

        Returns:
            :class:`ResumeState` namedtuple（缺失字段为 None）。

        Raises:
            FileNotFoundError: path 不存在。
        """
        path_obj = Path(path)
        if not path_obj.exists():
            raise FileNotFoundError(f"resume checkpoint 不存在：{path}")

        # 根据扩展名决定 format（.vn → vn，.pt → pt，其他默认 vn）
        ext = path_obj.suffix.lower()
        if ext == ".vn":
            fmt = "vn"
            use_vmpc = True
        elif ext == ".pt":
            fmt = "pt"
            use_vmpc = False
        else:
            fmt = "vn"
            use_vmpc = True

        ckpt = CheckpointManager(
            save_dir=path_obj.parent if str(path_obj.parent) else Path("."),
            best_path=path_obj,
            format=fmt,
            use_vmpc=use_vmpc,
        )
        full = ckpt.load_best_full()

        # 从 training_state 提取 epoch / patience_count
        training_state = full.get("training_state") or {}
        epoch = training_state.get("epoch")
        patience_count = training_state.get("patience_count")

        # rng_state 从 extra_state 中提取（apply 时调用 np.random.set_state）
        extra_state = full.get("extra_state")
        rng_state = None
        if isinstance(extra_state, dict) and "rng_state" in extra_state:
            rng_state = extra_state["rng_state"]

        # 缺失字段警告日志（旧文件向后兼容）
        missing = []
        if full.get("model_state_dict") is None:
            missing.append("model_state_dict")
        if full.get("optimizer_state") is None:
            missing.append("optimizer_state")
        if full.get("step") is None:
            missing.append("step")
        if rng_state is None:
            missing.append("rng_state")
        if full.get("best_val_loss") is None:
            missing.append("best_val_loss")
        if epoch is None:
            missing.append("epoch")
        if patience_count is None:
            missing.append("patience_count")
        if missing:
            print(
                f"[ResumeManager] 警告：从 {path} 读取的 resume state 缺失字段："
                f"{', '.join(missing)}（旧文件向后兼容，缺失字段跳过恢复）",
                flush=True,
            )

        return ResumeState(
            model_state_dict=full.get("model_state_dict"),
            optimizer_state=full.get("optimizer_state"),
            step=full.get("step"),
            rng_state=rng_state,
            best_val_loss=full.get("best_val_loss"),
            epoch=epoch,
            patience_count=patience_count,
        )

    # ------------------------------------------------------------------
    # apply（SubTask 6.5）
    # ------------------------------------------------------------------

    @staticmethod
    def apply(trainer, path: str) -> ResumeState:
        """把 :class:`ResumeState` 应用到 ``Trainer`` / ``ParallelTrainerSafe`` 实例。

        恢复 model / optimizer / step / rng / best_val_loss / epoch /
        patience_count；None 字段跳过恢复（向后兼容）。

        Args:
            trainer: :class:`Trainer` 或 :class:`ParallelTrainerSafe` 实例。
            path: ``.vn`` 文件路径（或兼容的 ``.pt`` 文件）。

        Returns:
            加载的 :class:`ResumeState`（便于调用方进一步处理，如提取
            extra_state 中的 best_state_dict）。
        """
        state = ResumeManager.load(path)

        # 恢复 model
        if state.model_state_dict is not None and hasattr(trainer, "model"):
            model = getattr(trainer, "model")
            if model is not None and hasattr(model, "load_state_dict"):
                try:
                    # pickle 往返避免外部状态污染（与 ParallelTrainerSafe 旧实现一致）
                    sd = pickle.loads(
                        pickle.dumps(state.model_state_dict, protocol=4)
                    )
                    model.load_state_dict(sd)
                except Exception as e:
                    print(
                        f"[ResumeManager] 警告：恢复 model state_dict 失败：{e}",
                        flush=True,
                    )

        # 恢复 optimizer（仅当 trainer 有 optimizer 属性且非 None）
        # 支持 PyTorch 风格（load_state_dict）与 verse_torch 风格（self.state dict）
        if state.optimizer_state is not None and hasattr(trainer, "optimizer"):
            opt = getattr(trainer, "optimizer")
            if opt is not None:
                if hasattr(opt, "load_state_dict") and callable(opt.load_state_dict):
                    try:
                        opt_sd = pickle.loads(
                            pickle.dumps(state.optimizer_state, protocol=4)
                        )
                        opt.load_state_dict(opt_sd)
                    except Exception as e:
                        print(
                            f"[ResumeManager] 警告：恢复 optimizer state 失败：{e}",
                            flush=True,
                        )
                elif hasattr(opt, "state") and isinstance(state.optimizer_state, dict):
                    # verse_torch Optimizer：恢复 state dict + param_groups 超参。
                    # state dict 按 ``id(p)`` 键控，跨 session 键不匹配新 params
                    # （仅同 session 内有效）；param_groups 仅恢复超参，保留新
                    # optimizer 的 ``params`` 引用（不覆盖为占位索引列表）。
                    try:
                        opt_sd = pickle.loads(
                            pickle.dumps(state.optimizer_state, protocol=4)
                        )
                        if "state" in opt_sd:
                            opt.state = opt_sd["state"]
                        if "param_groups" in opt_sd:
                            # 按位置对齐 param_groups，仅恢复超参（不覆盖 params）
                            for new_g, saved_g in zip(
                                opt.param_groups, opt_sd["param_groups"]
                            ):
                                for k, v in saved_g.items():
                                    if k != "params":
                                        new_g[k] = v
                    except Exception as e:
                        print(
                            f"[ResumeManager] 警告：恢复 optimizer state 失败：{e}",
                            flush=True,
                        )

        # 恢复 best_val_loss
        if state.best_val_loss is not None:
            try:
                trainer.best_val_loss = float(state.best_val_loss)
            except Exception:
                pass

        # 恢复 step（Trainer 可能在 self.step / self.global_step）
        if state.step is not None:
            try:
                trainer.step = int(state.step)
            except Exception:
                pass
            if hasattr(trainer, "global_step"):
                try:
                    trainer.global_step = int(state.step)
                except Exception:
                    pass

        # 恢复 epoch
        if state.epoch is not None and hasattr(trainer, "epoch"):
            try:
                trainer.epoch = int(state.epoch)
            except Exception:
                pass

        # 恢复 patience_count（EarlyStopping）
        if state.patience_count is not None:
            es = getattr(trainer, "early_stopping", None)
            if es is not None and hasattr(es, "patience_count"):
                try:
                    es.patience_count = int(state.patience_count)
                except Exception:
                    pass
            elif hasattr(trainer, "patience_count"):
                try:
                    trainer.patience_count = int(state.patience_count)
                except Exception:
                    pass

        # 恢复 rng_state（numpy RandomState）
        if state.rng_state is not None:
            try:
                np.random.set_state(state.rng_state)
            except Exception as e:
                print(
                    f"[ResumeManager] 警告：恢复 rng_state 失败：{e}",
                    flush=True,
                )

        return state


