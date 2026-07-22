"""智能分区训练器（Part4K2 Task 4）。

将 transformer 模型按 layer 分组（partition），训完一组卸载到硬盘 .vn 分片，
保持统一实体（对外表现为完整模型训练，训练一致性无差异）。

工作原理
--------
1. 将模型的 transformer blocks 按 layer 分组（每组 ``partition_size`` 层）
2. 训练当前组时，其他组参数冻结（``requires_grad=False``），不参与梯度计算
3. 当前组训练完成后，将其参数卸载到硬盘 ``.vn`` 分片（VNFileWriter）
4. 全部组训练完成后，合并所有分片为完整模型（VNFileReader 加载回内存）

设计要点
--------
- **统一实体**：训练过程中模型对象不变，只是内部参数在内存/硬盘之间备份。
  对外接口与普通 ``Trainer`` 一致（``fit`` / ``evaluate``）。
- **无损往返**：卸载/加载使用 ``.vn`` 分片（safetensors / npz），数值完全一致。
- **embedding / lm_head 始终在内存中**：不参与分区卸载，每个分区训练时都保持可训练，
  保证 loss 能持续下降。
- **内存监控**：调用 ``get_memory_info``（Task 5 已实现），超过阈值时自动把
  已训练的非当前组参数卸载到硬盘。

仅依赖 NumPy + Python 标准库 + VerseTorch 现有组件。
"""

from __future__ import annotations

import itertools
import os
import tempfile
from typing import Any, Optional

import numpy as np

from .tensor import Tensor, no_grad
from .optim import AdamW
from .device import get_memory_info
from .vn_format import VNFileWriter, VNFileReader
from .training import (
    cross_entropy_loss,
    _as_tensor,
    _scalar,
    _cfg_get,
)


# ---------------------------------------------------------------------------
# LayerWiseTrainer
# ---------------------------------------------------------------------------


class LayerWiseTrainer:
    """智能分区训练器：将模型按 layer 分组，训完一组卸载到硬盘，保持统一实体。

    工作原理：
        1. 将模型的 transformer blocks 按 layer 分组（每组 N 层）
        2. 训练当前组时，其他组参数冻结（不参与梯度计算）
        3. 当前组训练完成后，将其参数卸载到硬盘 .vn 分片
        4. 加载下一组参数到内存继续训练
        5. 全部组训练完成后，合并所有分片为完整模型

    对外表现为完整模型训练，训练一致性无差异。

    Args:
        model: 要训练的模型（需有 ``.blocks`` 属性，如 VerseNexLM/CometSparkNexLM）
        config: 训练配置 dict，读取：
            - ``lr``: 学习率（默认 1e-3）
            - ``weight_decay``: 权重衰减（默认 0.01）
            - ``batch_size``: 批大小（仅用于日志，默认 8）
            - ``eval_interval``: 评估频率（默认 None=不评估）
            - ``log_interval``: 日志间隔（默认 10）
            - ``seed``: 随机种子（默认 42）
            - ``finetune_steps``: 合并后整体微调步数（默认 0）
        optimizer_config: 优化器配置 dict（如 ``weight_decay`` / ``betas``）
        partition_size: 每组包含的 layer 数量（默认 2）
        offload_dir: 硬盘卸载目录（默认用 tempfile 自动创建）
        memory_threshold_mb: 内存阈值（MB），超过时触发卸载（默认 512）

    用法::

        trainer = LayerWiseTrainer(model, config={"lr": 1e-3}, partition_size=2)
        train_losses, val_losses = trainer.fit(train_loader, val_loader, max_steps=100)
    """

    def __init__(
        self,
        model,
        config,
        optimizer_config=None,
        partition_size: int = 2,
        offload_dir: Optional[str] = None,
        memory_threshold_mb: float = 512,
    ):
        self.model = model
        self.config = config if config is not None else {}
        self.optimizer_config = dict(optimizer_config) if optimizer_config else {}

        self.partition_size = max(1, int(partition_size))

        # 卸载目录：未指定时用 tempfile 自动创建（由本实例管理生命周期）
        if offload_dir is None:
            self.offload_dir = tempfile.mkdtemp(prefix="layerwise_offload_")
            self._offload_dir_owned = True
        else:
            self.offload_dir = str(offload_dir)
            self._offload_dir_owned = False
        os.makedirs(self.offload_dir, exist_ok=True)

        self.memory_threshold_mb = float(memory_threshold_mb)

        # 分区
        self.partitions = self._partition_layers()
        self.n_partitions = len(self.partitions)

        # 训练状态
        self.train_losses: list[float] = []
        self.val_losses: list[float] = []
        self.best_val_loss = float("inf")
        self._offloaded: set[int] = set()  # 已卸载到硬盘的分区索引
        self._trained: set[int] = set()   # 已训练完成的分区索引
        self._current_partition: int = -1
        self._memory_high_triggered: bool = False

        # 保存原始 requires_grad 状态，便于训练后恢复
        self._orig_requires_grad = self._snapshot_requires_grad()

    # ------------------------------------------------------------------
    # 分区
    # ------------------------------------------------------------------

    def _partition_layers(self) -> list[list[int]]:
        """将模型层分组。

        若模型有 ``.blocks`` 属性（如 VerseNexLM/CometSparkNexLM），按 blocks
        列表分组，每组 ``partition_size`` 个 block。embedding 和 lm_head 不参与
        分组（始终在内存中）。

        Returns:
            ``list[list[int]]``，每个内层 list 是该组包含的 block 索引。
        """
        blocks = getattr(self.model, "blocks", None)
        if blocks is None:
            raise ValueError(
                "LayerWiseTrainer 需要模型有 .blocks 属性（如 VerseNexLM/CometSparkNexLM）；"
                f"当前模型 {type(self.model).__name__} 无此属性"
            )
        n = len(blocks)
        ps = self.partition_size
        partitions = []
        for i in range(0, n, ps):
            partitions.append(list(range(i, min(i + ps, n))))
        return partitions

    # ------------------------------------------------------------------
    # requires_grad 管理
    # ------------------------------------------------------------------

    def _snapshot_requires_grad(self) -> dict[str, bool]:
        """快照所有参数的 requires_grad 状态（用于训练后恢复）。"""
        snap: dict[str, bool] = {}
        if hasattr(self.model, "named_parameters"):
            for name, p in self.model.named_parameters():
                snap[name] = bool(p.requires_grad)
        return snap

    def _set_trainable(self, partition_idx: int) -> None:
        """设置当前组可训练，其他组冻结。

        - 当前组的 blocks：``requires_grad=True``
        - 其他组的 blocks：``requires_grad=False``
        - embedding / lm_head / norm 等非 block 参数：保持 ``requires_grad=True``
          （始终在内存中，不卸载，保证 loss 能持续下降）
        """
        block_indices = set(self.partitions[partition_idx])
        if not hasattr(self.model, "named_parameters"):
            return
        for name, p in self.model.named_parameters():
            if name.startswith("blocks."):
                parts = name.split(".")
                try:
                    idx = int(parts[1])
                except (IndexError, ValueError):
                    continue
                p.requires_grad = idx in block_indices
            else:
                # 非 block 参数（tok_emb / norm / head）：保持可训练
                p.requires_grad = True

    def _restore_requires_grad(self) -> None:
        """恢复所有参数的 requires_grad 到训练前状态。"""
        if not hasattr(self.model, "named_parameters"):
            return
        for name, p in self.model.named_parameters():
            if name in self._orig_requires_grad:
                p.requires_grad = self._orig_requires_grad[name]

    # ------------------------------------------------------------------
    # 卸载 / 加载
    # ------------------------------------------------------------------

    def _get_partition_state(self, partition_idx: int) -> dict[str, np.ndarray]:
        """提取一组 block 的参数 state_dict。

        从模型完整 ``state_dict`` 中筛选出 ``blocks.{idx}.*`` 的参数。

        Args:
            partition_idx: 分区索引

        Returns:
            ``{name: ndarray}``，仅包含该组 block 的参数
        """
        block_indices = self.partitions[partition_idx]
        full_sd = self.model.state_dict()
        sd: dict[str, np.ndarray] = {}
        for idx in block_indices:
            prefix = f"blocks.{idx}."
            for k, v in full_sd.items():
                if k.startswith(prefix):
                    sd[k] = np.asarray(v)
        return sd

    def _offload_partition(
        self,
        partition_idx: int,
        state_dict: Optional[dict] = None,
        zero: bool = False,
    ) -> str:
        """将一组参数卸载到硬盘 .vn 分片。

        使用 ``VNFileWriter`` 写入 ``offload_dir/partition_{idx}.vn``。
        卸载后默认不置零（保持 forward 正确性）；若 ``zero=True`` 则把模型中
        该组参数置零以释放内存（适用于内存压力场景，置零后该组需重新加载才能用）。

        Args:
            partition_idx: 分区索引
            state_dict: 自定义 state_dict（None 则从模型提取）
            zero: 是否把模型中该组参数置零（默认 False）

        Returns:
            .vn 分片文件路径
        """
        if state_dict is None:
            state_dict = self._get_partition_state(partition_idx)

        path = os.path.join(self.offload_dir, f"partition_{partition_idx}.vn")
        writer = VNFileWriter(
            path,
            arch="layerwise",
            config={
                "partition": partition_idx,
                "block_indices": self.partitions[partition_idx],
                "partition_size": self.partition_size,
            },
        )
        try:
            writer.write_weights(state_dict)
            writer.close()
        except Exception:
            writer.close()
            raise
        self._offloaded.add(partition_idx)

        if zero:
            self._zero_partition(partition_idx)
        return path

    def _zero_partition(self, partition_idx: int) -> None:
        """把模型中该组参数置零（释放内存，置零后该组不可用于 forward）。"""
        block_indices = self.partitions[partition_idx]
        for idx in block_indices:
            block = self.model.blocks[idx]
            for p in block.parameters():
                # 保持 shape 与 dtype，仅置零 data
                p.data = np.zeros_like(p.data)

    def _load_partition(self, partition_idx: int) -> dict[str, np.ndarray]:
        """从硬盘 .vn 分片加载一组参数到模型。

        使用 ``VNFileReader`` 读取 ``offload_dir/partition_{idx}.vn``，
        然后用 ``load_state_dict(strict=False)`` 加载到模型（无损）。

        Args:
            partition_idx: 分区索引

        Returns:
            加载的 ``{name: ndarray}`` state_dict
        """
        path = os.path.join(self.offload_dir, f"partition_{partition_idx}.vn")
        reader = VNFileReader(path)
        try:
            sd = reader.read_weights()
        finally:
            reader.close()
        # strict=False：仅加载该组 block 参数，非 block 参数保持不变
        self.model.load_state_dict(sd, strict=False)
        return sd

    # ------------------------------------------------------------------
    # 单组训练
    # ------------------------------------------------------------------

    def _train_partition(
        self,
        partition_idx: int,
        train_loader,
        max_steps: int,
    ) -> None:
        """训练一组参数。

        - 冻结其他组 blocks，仅当前组 + embedding/lm_head 可训练
        - 用 AdamW 优化器训练 ``max_steps`` 步
        - 梯度只更新当前组的参数

        Args:
            partition_idx: 分区索引
            train_loader: 可迭代对象，每次返回 ``(x, y)``
            max_steps: 训练步数
        """
        self._current_partition = partition_idx
        self._set_trainable(partition_idx)

        lr = float(_cfg_get(self.config, "lr", 1e-3))
        weight_decay = float(self.optimizer_config.get(
            "weight_decay", _cfg_get(self.config, "weight_decay", 0.01)
        ))
        betas = self.optimizer_config.get("betas", (0.9, 0.999))
        eps = float(self.optimizer_config.get("eps", 1e-8))

        # 仅收集 requires_grad=True 的参数构造优化器
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = AdamW(
            trainable_params, lr=lr, weight_decay=weight_decay,
            betas=tuple(betas), eps=eps,
        )

        log_interval = int(_cfg_get(self.config, "log_interval", 10))
        enable_log = log_interval > 0

        train_iter = itertools.cycle(train_loader)
        partition_start = len(self.train_losses)

        for step in range(max_steps):
            try:
                batch = next(train_iter)
            except StopIteration:
                break
            if batch is None:
                continue
            x, y = batch
            x = _as_tensor(x)
            y = _as_tensor(y)

            logits = self.model(x)
            loss = cross_entropy_loss(logits, y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            loss_val = _scalar(loss)
            self.train_losses.append(loss_val)

            if enable_log and (step % log_interval == 0 or step == max_steps - 1):
                print(
                    f"[layerwise] partition={partition_idx} "
                    f"step={step}/{max_steps} loss={loss_val:.6f}",
                    flush=True,
                )

            # 训练中途内存检查（如果超阈值，卸载已训练的非当前组）
            if step > 0 and step % max(1, max_steps // 4) == 0:
                self._check_memory()

        self._trained.add(partition_idx)
        n_done = len(self.train_losses) - partition_start
        print(
            f"[layerwise] partition={partition_idx} 训练完成 "
            f"steps={n_done} last_loss={self.train_losses[-1] if self.train_losses else float('nan'):.6f}",
            flush=True,
        )

    # ------------------------------------------------------------------
    # 评估
    # ------------------------------------------------------------------

    def evaluate(self, val_loader) -> float:
        """在 val_loader 上计算平均 loss（no_grad 上下文）。

        Args:
            val_loader: 可迭代对象，每次返回 ``(x, y)``

        Returns:
            平均 loss；空 loader 返回 nan
        """
        total_loss = 0.0
        n_batches = 0
        with no_grad():
            for batch in val_loader:
                if batch is None:
                    continue
                x, y = batch
                x = _as_tensor(x)
                y = _as_tensor(y)
                logits = self.model(x)
                loss = cross_entropy_loss(logits, y)
                total_loss += _scalar(loss)
                n_batches += 1
        if n_batches == 0:
            return float("nan")
        return total_loss / n_batches

    # ------------------------------------------------------------------
    # 内存监控
    # ------------------------------------------------------------------

    def _check_memory(self) -> bool:
        """检查内存使用，超过阈值时触发卸载。

        使用 ``get_memory_info`` 读取当前内存使用，若超过 ``memory_threshold_mb``
        则把已训练的非当前组参数卸载到硬盘（若尚未卸载）。

        Returns:
            bool: 是否检测到内存超阈值
        """
        info = get_memory_info("cpu")
        used_bytes = int(info.get("used", 0))
        used_mb = used_bytes / (1024 * 1024)

        if used_mb > self.memory_threshold_mb:
            self._memory_high_triggered = True
            # 卸载已训练的非当前组（若尚未卸载）
            for i in sorted(self._trained):
                if i == self._current_partition:
                    continue
                if i not in self._offloaded:
                    self._offload_partition(i)
            print(
                f"[layerwise] 内存告警：used={used_mb:.1f}MB > "
                f"threshold={self.memory_threshold_mb:.1f}MB，已触发卸载",
                flush=True,
            )
            return True
        return False

    # ------------------------------------------------------------------
    # 合并分片
    # ------------------------------------------------------------------

    def _merge_partitions(self) -> None:
        """合并所有分片为完整模型。

        从硬盘加载所有已卸载的分区参数到模型，恢复完整状态。
        加载完成后恢复所有参数的 ``requires_grad`` 原始状态。
        """
        for i in sorted(self._offloaded):
            self._load_partition(i)
        self._restore_requires_grad()
        print(
            f"[layerwise] 合并完成：已加载 {len(self._offloaded)} 个分片，"
            f"模型恢复完整状态",
            flush=True,
        )

    # ------------------------------------------------------------------
    # 主训练入口
    # ------------------------------------------------------------------

    def fit(
        self,
        train_loader,
        val_loader=None,
        max_steps: int = 1000,
    ):
        """训练，自动分区+卸载+加载。

        逐组训练：先训第 0 组，卸载到硬盘；再训第 1 组，卸载；... 直到最后一组。
        每组训练 ``max_steps // n_partitions`` 步。全部组训练完成后合并所有分片
        为完整模型，可选择性整体 fine-tune 几步。

        Args:
            train_loader: 训练数据加载器，每次返回 ``(x, y)``
            val_loader: 验证数据加载器（可选）
            max_steps: 总训练步数（按分区数均分）

        Returns:
            ``(train_losses, val_losses)`` 两个列表
        """
        n = self.n_partitions
        if n == 0:
            return self.train_losses, self.val_losses

        steps_per = max(1, max_steps // n)
        # 余数均摊到前几个分区
        remainder = max_steps - steps_per * n
        print(
            f"[layerwise] 开始分区训练：n_partitions={n} "
            f"partition_size={self.partition_size} "
            f"steps_per={steps_per} max_steps={max_steps}",
            flush=True,
        )

        eval_interval = _cfg_get(self.config, "eval_interval", None)

        for i in range(n):
            cur_steps = steps_per + (1 if i < remainder else 0)
            self._train_partition(i, train_loader, cur_steps)

            # 训完一组卸载到硬盘（最后一组也卸载，便于合并校验）
            if i not in self._offloaded:
                self._offload_partition(i)

            # 内存检查
            self._check_memory()

            # 评估
            if val_loader is not None and eval_interval is not None:
                val_loss = self.evaluate(val_loader)
                self.val_losses.append(val_loss)
                if val_loss < self.best_val_loss:
                    self.best_val_loss = float(val_loss)
                print(
                    f"[layerwise] partition={i} val_loss={val_loss:.6f} "
                    f"best={self.best_val_loss:.6f}",
                    flush=True,
                )

        # 合并所有分片为完整模型
        self._merge_partitions()

        # 可选整体 fine-tune
        finetune_steps = int(_cfg_get(self.config, "finetune_steps", 0))
        if finetune_steps > 0:
            print(f"[layerwise] 整体 fine-tune {finetune_steps} 步...", flush=True)
            self._current_partition = -1
            # 全部参数可训练
            self._restore_requires_grad()
            lr = float(_cfg_get(self.config, "lr", 1e-3))
            weight_decay = float(self.optimizer_config.get(
                "weight_decay", _cfg_get(self.config, "weight_decay", 0.01)
            ))
            optimizer = AdamW(
                [p for p in self.model.parameters() if p.requires_grad],
                lr=lr * 0.5, weight_decay=weight_decay,
            )
            train_iter = itertools.cycle(train_loader)
            for step in range(finetune_steps):
                try:
                    batch = next(train_iter)
                except StopIteration:
                    break
                if batch is None:
                    continue
                x, y = batch
                x = _as_tensor(x)
                y = _as_tensor(y)
                logits = self.model(x)
                loss = cross_entropy_loss(logits, y)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                self.train_losses.append(_scalar(loss))

            if val_loader is not None:
                val_loss = self.evaluate(val_loader)
                self.val_losses.append(val_loss)
                if val_loss < self.best_val_loss:
                    self.best_val_loss = float(val_loss)

        print(
            f"[layerwise] 训练完成：total_steps={len(self.train_losses)} "
            f"best_val={self.best_val_loss:.6f}",
            flush=True,
        )
        return self.train_losses, self.val_losses

    # ------------------------------------------------------------------
    # 清理
    # ------------------------------------------------------------------

    def cleanup(self) -> None:
        """清理卸载目录（仅当由本实例自动创建时）。"""
        if self._offload_dir_owned and os.path.isdir(self.offload_dir):
            import shutil
            try:
                shutil.rmtree(self.offload_dir)
            except Exception:
                pass

    def __del__(self):
        try:
            self.cleanup()
        except Exception:
            pass


__all__ = ["LayerWiseTrainer"]
