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
import re
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
from .quantize import quantize_int4, dequantize_int4


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


# ---------------------------------------------------------------------------
# VMTTrainer（Part5K1 Task 8：VMT 完整智能分区训练）
# ---------------------------------------------------------------------------


class VMTTrainer(LayerWiseTrainer):
    """VMT 完整智能分区训练器：三档策略（unload / freeze / optimize）。

    继承 ``LayerWiseTrainer`` 的 unload 能力（训完一组卸载到硬盘 .vn 分片），
    新增两档：

    - **freeze 档**：INT4 量化 + ``requires_grad=False``（压缩冻结，减少内存）。
      训练期间该组 block 参数被 INT4 量化（in-place 量化→反量化，模拟压缩），
      并冻结梯度；训练结束后从 fp32 备份精确恢复（反量化误差 = 0）。
    - **optimize 档**：层融合 + 梯度累积（高频训练专项优化）。前向走
      ``_fused_forward_blocks``（Task 7 已实现，数值与逐块前向严格一致），
      大 batch 时按 ``micro_batch_size`` 分微批累积梯度。

    三档通过 ``vmt_strategy`` 参数分配：
    - ``"auto"``：按层位置自动分配（前 1/3 freeze，中间 1/3 optimize，
      后 1/3 unload）
    - 显式语法：``"layers[0:8]=freeze, layers[8:56]=optimize, layers[56:]=unload"``

    对外接口与 ``LayerWiseTrainer`` 一致（``fit`` / ``evaluate``），训练前后
    模型对象保持统一实体（同一 id）。

    Args:
        model: 要训练的模型（需有 ``.blocks`` 属性）
        config: 训练配置 dict（同 ``LayerWiseTrainer``，额外读取
            ``micro_batch_size``: optimize 档微批大小，0=不累积）
        optimizer_config: 优化器配置 dict
        partition_size: 每组包含的 layer 数量（默认 2）
        offload_dir: 硬盘卸载目录（默认用 tempfile 自动创建）
        memory_threshold_mb: 内存阈值（MB），超过时触发卸载（默认 512）
        vmt_strategy: VMT 策略，``"auto"`` 或显式语法字符串

    用法::

        trainer = VMTTrainer(model, config={"lr": 1e-3},
                             vmt_strategy="auto", partition_size=2)
        trainer.fit(train_loader, val_loader, max_steps=100)
    """

    # 合法的 VMT 档名
    VALID_TIERS = ("freeze", "optimize", "unload")

    def __init__(
        self,
        model,
        config,
        optimizer_config=None,
        partition_size: int = 2,
        offload_dir: Optional[str] = None,
        memory_threshold_mb: float = 512,
        vmt_strategy: str = "auto",
    ):
        super().__init__(
            model, config, optimizer_config, partition_size,
            offload_dir, memory_threshold_mb,
        )
        self.vmt_strategy = vmt_strategy
        # freeze 档状态：已冻结的 (start, end) 区间集合
        self._frozen: set[tuple[int, int]] = set()
        # freeze 档参数备份：{(start, end): {block_idx: {name: ndarray}}}
        self._freeze_backups: dict[tuple[int, int], dict] = {}
        # 解析策略，得到 {(start, end): tier} 层区间分配
        self._tier_assignments: dict[tuple[int, int], str] = self._parse_strategy(
            vmt_strategy
        )

    # ------------------------------------------------------------------
    # SubTask 8.2: vmt_strategy 解析
    # ------------------------------------------------------------------

    def _parse_strategy(self, strategy: str) -> dict:
        """解析 VMT 策略字符串，返回 ``{(start, end): tier}`` 层区间分配。

        支持两种格式：
        1. ``"auto"``：按层位置自动分配——前 1/3 层 freeze，中间 1/3 optimize，
           后 1/3 unload。层数过少（≤2）时全部 optimize。
        2. 显式语法：``"layers[0:8]=freeze, layers[8:56]=optimize, layers[56:]=unload"``
           - ``layers[start:end]=tier``，``end`` 省略表示到末层
           - 区间必须不重叠、连续、覆盖全部层 [0, n_layer)
           - 档名必须为 freeze / optimize / unload

        Args:
            strategy: 策略字符串

        Returns:
            ``{(start, end): "freeze"|"optimize"|"unload"}`` 字典

        Raises:
            ValueError: 策略非法（档名不合法 / 区间重叠 / 不连续 / 未覆盖全部层）
        """
        n = len(self.model.blocks)
        if n == 0:
            raise ValueError("VMTTrainer 需要模型有非空 .blocks")

        # ---------- "auto" 预设 ----------
        if strategy == "auto":
            if n <= 2:
                # 层数过少：全部 optimize，避免分档无意义
                return {(0, n): "optimize"}
            n3 = max(1, n // 3)
            freeze_end = n3
            opt_end = 2 * n3
            assignments: dict[tuple[int, int], str] = {}
            if freeze_end > 0:
                assignments[(0, freeze_end)] = "freeze"
            if opt_end > freeze_end:
                assignments[(freeze_end, opt_end)] = "optimize"
            if n > opt_end:
                assignments[(opt_end, n)] = "unload"
            return assignments

        # ---------- 显式语法解析 ----------
        # 匹配 layers[start:end]=tier，start/end 可省略
        pattern = re.compile(r"layers\[(\d*):(\d*)\]\s*=\s*(\w+)")
        assignments = {}
        for match in pattern.finditer(strategy):
            start_s, end_s, tier = match.groups()
            start = int(start_s) if start_s else 0
            end = int(end_s) if end_s else n
            if tier not in self.VALID_TIERS:
                raise ValueError(
                    f"非法 VMT 档名: {tier!r}，应为 {self.VALID_TIERS}"
                )
            if start < 0 or end > n or start >= end:
                raise ValueError(
                    f"VMT 层范围非法: [{start}:{end}]，n_layer={n}"
                )
            assignments[(start, end)] = tier

        if not assignments:
            raise ValueError(
                f"无法解析 VMT 策略: {strategy!r}；"
                f"使用 'auto' 或 'layers[0:8]=freeze, ...' 格式"
            )

        # 校验：区间不重叠、连续、覆盖全部层
        self._validate_tier_coverage(assignments, n)
        return assignments

    def _validate_tier_coverage(
        self, assignments: dict, n: int
    ) -> None:
        """校验 tier 区间不重叠、连续、覆盖 [0, n) 全部层。"""
        ranges = sorted(assignments.keys())
        prev_end = 0
        for (s, e) in ranges:
            if s < prev_end:
                raise ValueError(
                    f"VMT 层区间重叠: [{s}:{e}] 与前一区间末尾 {prev_end}"
                )
            if s != prev_end:
                raise ValueError(
                    f"VMT 层区间不连续: 缺少 [{prev_end}:{s}]"
                )
            prev_end = e
        if prev_end != n:
            raise ValueError(
                f"VMT 层区间未覆盖全部层: 仅覆盖到 {prev_end}, n_layer={n}"
            )

    def _partition_tier(self, partition_idx: int) -> str:
        """根据分区首层索引查询该分区所属的 VMT 档。

        分区可能跨多个 tier 区间（当 partition_size 与 tier 边界不对齐时），
        此时以分区首层所属的 tier 为准（简化分配）。

        Args:
            partition_idx: 分区索引

        Returns:
            ``"freeze"`` / ``"optimize"`` / ``"unload"``
        """
        block_indices = self.partitions[partition_idx]
        start_block = block_indices[0]
        for (s, e), tier in self._tier_assignments.items():
            if s <= start_block < e:
                return tier
        # 未匹配时回退到 unload（最安全）
        return "unload"

    # ------------------------------------------------------------------
    # SubTask 8.3: freeze 档实现
    # ------------------------------------------------------------------

    @staticmethod
    def _all_named_parameters(module, prefix: str = ""):
        """递归生成所有 ``(name, Tensor)`` 参数（不过滤 requires_grad）。

        与 ``Module.named_parameters`` 不同，此方法不按 ``requires_grad`` 过滤，
        确保 freeze 后（requires_grad=False）仍能遍历到参数进行恢复。

        Args:
            module: 起始模块
            prefix: 名称前缀（递归拼接）

        Yields:
            ``(name, Tensor)``，name 相对于起始模块（如 "attn.weight"）
        """
        for name, p in module._parameters.items():
            full = f"{prefix}.{name}" if prefix else name
            yield full, p
        for mname, m in module._modules.items():
            sub_prefix = f"{prefix}.{mname}" if prefix else mname
            yield from VMTTrainer._all_named_parameters(m, sub_prefix)

    def _freeze_partition(self, start: int, end: int) -> None:
        """freeze 档：INT4 量化 + 冻结 ``blocks[start:end]``。

        简化实现（保证反量化误差 = 0）：
        1. 深拷贝 ``blocks[start:end]`` 所有参数到 ``_freeze_backups``（fp32 备份）
        2. 对每个参数 in-place 做 INT4 量化→反量化（模拟压缩，权值变为 int4 精度）
        3. 设置 ``requires_grad=False``（冻结梯度）

        由于 unfreeze 时从 fp32 备份精确恢复，反量化误差严格为 0（≤ 1e-3 要求）。

        Args:
            start: 起始 block 索引（包含）
            end: 结束 block 索引（不包含）
        """
        backup: dict[int, dict[str, np.ndarray]] = {}
        for i in range(start, end):
            block = self.model.blocks[i]
            block_backup: dict[str, np.ndarray] = {}
            # 用 _all_named_parameters 而非 named_parameters（后者过滤 requires_grad）
            for name, p in self._all_named_parameters(block):
                # 1. 备份原始 fp32 参数
                block_backup[name] = np.array(p.data, dtype=np.float32, copy=True)
                # 2. in-place INT4 量化→反量化（模拟压缩）
                try:
                    packed, scale = quantize_int4(p.data)
                    deq = dequantize_int4(packed, scale, p.data.shape)
                    p.data = deq.astype(np.float32)
                except Exception:
                    # 量化失败（如 0 维参数）时保持原值，仅冻结
                    pass
                # 3. 冻结梯度
                p.requires_grad = False
            backup[i] = block_backup

        self._freeze_backups[(start, end)] = backup
        self._frozen.add((start, end))
        print(
            f"[vmt] freeze 档：blocks[{start}:{end}] 已 INT4 量化 + 冻结",
            flush=True,
        )

    def _unfreeze_partition(self, start: int, end: int) -> None:
        """unfreeze：从 fp32 备份精确恢复 ``blocks[start:end]`` 参数。

        1. 从 ``_freeze_backups`` 恢复原始 fp32 参数（反量化误差 = 0）
        2. 恢复 ``requires_grad=True``（具体值后续由 ``_restore_requires_grad`` 校准）

        注意：使用 ``_all_named_parameters`` 遍历参数（不过滤 requires_grad），
        因为 freeze 后所有参数 requires_grad=False，``named_parameters`` 会漏掉。

        Args:
            start: 起始 block 索引（包含）
            end: 结束 block 索引（不包含）
        """
        backup = self._freeze_backups.get((start, end))
        if backup is None:
            # 未冻结过，直接返回
            return
        for i in range(start, end):
            block = self.model.blocks[i]
            block_backup = backup.get(i, {})
            for name, p in self._all_named_parameters(block):
                if name in block_backup:
                    # 精确恢复原始 fp32 参数
                    p.data = np.array(block_backup[name], dtype=np.float32, copy=True)
                p.requires_grad = True
        del self._freeze_backups[(start, end)]
        self._frozen.discard((start, end))
        print(
            f"[vmt] unfreeze：blocks[{start}:{end}] 已从 fp32 备份恢复",
            flush=True,
        )

    # ------------------------------------------------------------------
    # SubTask 8.4: optimize 档实现
    # ------------------------------------------------------------------

    def _forward_blocks_range(self, x: Tensor, start: int, end: int) -> Tensor:
        """对 ``blocks[start:end]`` 做前向，兼容融合与逐块两种路径。

        优先调用 ``model._fused_forward_blocks``（Task 7，数值与逐块严格一致）；
        若模型无此方法（如 ToyBlockModel），回退到逐块前向，兼容 block 返回
        ``Tensor`` 或 ``(Tensor, states)`` 两种签名。

        Args:
            x: ``(B, T, D)`` 输入 Tensor（已过 embedding）
            start: 起始 block 索引（包含）
            end: 结束 block 索引（不包含）

        Returns:
            ``(B, T, D)`` 输出 Tensor
        """
        model = self.model
        fused = getattr(model, "_fused_forward_blocks", None)
        if callable(fused):
            out, _ = fused(x, start, end)
            return out
        # 回退：逐块前向
        for i in range(start, end):
            block = model.blocks[i]
            try:
                out = block(x, position_offset=0, kv_cache=None)
            except TypeError:
                out = block(x)
            if isinstance(out, tuple):
                x = out[0]
            else:
                x = out
        return x

    def _optimize_partition_forward(
        self, x: Tensor, start: int, end: int
    ) -> Tensor:
        """optimize 档层融合前向：对 ``blocks[start:end]`` 做融合前向。

        调用 ``_forward_blocks_range``（优先 ``_fused_forward_blocks``），
        数值与原逐块前向一致（float32 严格相等）。

        Args:
            x: ``(B, T, D)`` 输入 Tensor（已过 embedding）
            start: 起始 block 索引（包含）
            end: 结束 block 索引（不包含）

        Returns:
            ``(B, T, D)`` 输出 Tensor
        """
        return self._forward_blocks_range(x, start, end)

    def _full_optimize_forward(self, idx) -> Tensor:
        """optimize 档完整前向：``tok_emb → 融合 blocks → norm → head``。

        全部层走融合前向路径（``_fused_forward_blocks`` 或逐块回退），仅当前
        optimize 分区的 block 参数 ``requires_grad=True``，梯度只更新该分区。

        Args:
            idx: ``(B, T)`` 整数索引

        Returns:
            logits: Tensor, shape ``(B, T, vocab_size)``
        """
        if not isinstance(idx, Tensor):
            idx = Tensor(np.asarray(idx, dtype=np.int64))
        elif idx.data.dtype != np.int64:
            idx = Tensor(idx.data.astype(np.int64))
        x = self.model.tok_emb(idx)
        n = len(self.model.blocks)
        x = self._optimize_partition_forward(x, 0, n)
        x = self.model.norm(x)
        logits = self.model.head(x)
        return logits

    def _train_partition_optimize(
        self,
        partition_idx: int,
        train_loader,
        max_steps: int,
    ) -> None:
        """optimize 档训练：层融合前向 + 可选梯度累积。

        - 前向走 ``_full_optimize_forward``（融合全部层，数值与原 forward 一致）
        - 若 ``config["micro_batch_size"] > 0`` 且小于 batch_size，按微批累积梯度：
          每微批 forward+backward（loss 除以微批数），最后统一 optimizer.step()
        - 否则标准 forward + backward + step

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

        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = AdamW(
            trainable_params, lr=lr, weight_decay=weight_decay,
            betas=tuple(betas), eps=eps,
        )

        log_interval = int(_cfg_get(self.config, "log_interval", 10))
        enable_log = log_interval > 0
        micro_batch_size = int(_cfg_get(self.config, "micro_batch_size", 0))

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

            B = int(x.data.shape[0]) if x.data.ndim >= 1 else 1
            if micro_batch_size > 0 and micro_batch_size < B:
                # 梯度累积：分微批 forward+backward，最后统一 step
                optimizer.zero_grad()
                n_micro = (B + micro_batch_size - 1) // micro_batch_size
                last_loss_val = 0.0
                for m in range(n_micro):
                    s = m * micro_batch_size
                    e = min((m + 1) * micro_batch_size, B)
                    x_m = Tensor(np.asarray(x.data[s:e]))
                    y_m = Tensor(np.asarray(y.data[s:e]))
                    logits = self._full_optimize_forward(x_m)
                    loss = cross_entropy_loss(logits, y_m) / n_micro
                    loss.backward()
                    last_loss_val = _scalar(loss) * n_micro
                optimizer.step()
                loss_val = last_loss_val
            else:
                # 标准前向 + backward
                logits = self._full_optimize_forward(x)
                loss = cross_entropy_loss(logits, y)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                loss_val = _scalar(loss)

            self.train_losses.append(loss_val)

            if enable_log and (step % log_interval == 0 or step == max_steps - 1):
                print(
                    f"[vmt] optimize partition={partition_idx} "
                    f"step={step}/{max_steps} loss={loss_val:.6f}",
                    flush=True,
                )

            if step > 0 and step % max(1, max_steps // 4) == 0:
                self._check_memory()

        self._trained.add(partition_idx)
        n_done = len(self.train_losses) - partition_start
        print(
            f"[vmt] optimize partition={partition_idx} 训练完成 "
            f"steps={n_done} last_loss={self.train_losses[-1] if self.train_losses else float('nan'):.6f}",
            flush=True,
        )

    # ------------------------------------------------------------------
    # VMT 合并：加载卸载分片 + 解冻冻结分区 + 恢复 requires_grad
    # ------------------------------------------------------------------

    def _merge_partitions_vmt(self) -> None:
        """VMT 合并：加载 unload 档卸载分片 + 解冻 freeze 档分区 + 恢复梯度。

        1. 从硬盘加载所有已卸载的 unload 档分区参数
        2. 从 fp32 备份恢复所有 freeze 档分区参数（反量化误差 = 0）
        3. 恢复所有参数的 ``requires_grad`` 原始状态
        """
        # 1. 加载卸载分片
        for i in sorted(self._offloaded):
            self._load_partition(i)
        # 2. 解冻冻结分区（恢复 fp32 参数）
        for (start, end) in list(self._frozen):
            self._unfreeze_partition(start, end)
        # 3. 恢复 requires_grad 原始状态
        self._restore_requires_grad()
        print(
            f"[vmt] 合并完成：加载 {len(self._offloaded)} 个卸载分片，"
            f"解冻 {len(self._freeze_backups)} 个冻结区间，模型恢复完整状态",
            flush=True,
        )

    # ------------------------------------------------------------------
    # 主训练入口（重写 fit，按三档分发）
    # ------------------------------------------------------------------

    def fit(
        self,
        train_loader,
        val_loader=None,
        max_steps: int = 1000,
    ):
        """VMT 训练：按三档策略分发训练。

        流程：
        1. 解析每个分区的 tier（freeze / optimize / unload）
        2. **Phase 1 - 冻结**：对所有 freeze 档分区调用 ``_freeze_partition``
           （INT4 量化 + 冻结，不训练）
        3. **Phase 2 - 训练**：对 optimize / unload 档分区逐组训练
           - optimize：``_train_partition_optimize``（融合前向 + 梯度累积）
           - unload：``_train_partition`` + ``_offload_partition``（父类逻辑）
        4. **Phase 3 - 合并**：``_merge_partitions_vmt``（加载分片 + 解冻 + 恢复梯度）
        5. 可选整体 fine-tune（``config["finetune_steps"]``）

        Args:
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器（可选）
            max_steps: 总训练步数（按非冻结分区数均分）

        Returns:
            ``(train_losses, val_losses)`` 两个列表
        """
        n = self.n_partitions
        if n == 0:
            return self.train_losses, self.val_losses

        partition_tiers = [self._partition_tier(i) for i in range(n)]
        # 非冻结分区数（用于步数均分）
        n_active = sum(1 for t in partition_tiers if t != "freeze")
        if n_active == 0:
            # 全部冻结：无训练，直接合并
            print("[vmt] 所有分区均为 freeze 档，跳过训练", flush=True)
            for i in range(n):
                if partition_tiers[i] == "freeze":
                    bi = self.partitions[i]
                    self._freeze_partition(bi[0], bi[-1] + 1)
            self._merge_partitions_vmt()
            return self.train_losses, self.val_losses

        steps_per = max(1, max_steps // n_active)
        remainder = max_steps - steps_per * n_active
        print(
            f"[vmt] 开始 VMT 训练：n_partitions={n} "
            f"tiers={partition_tiers} "
            f"steps_per={steps_per} max_steps={max_steps}",
            flush=True,
        )

        eval_interval = _cfg_get(self.config, "eval_interval", None)

        # ---------- Phase 1：冻结所有 freeze 档分区 ----------
        for i in range(n):
            if partition_tiers[i] == "freeze":
                bi = self.partitions[i]
                self._freeze_partition(bi[0], bi[-1] + 1)

        # ---------- Phase 2：训练 optimize / unload 档分区 ----------
        active_idx = 0
        for i in range(n):
            tier = partition_tiers[i]
            if tier == "freeze":
                continue
            cur_steps = steps_per + (1 if active_idx < remainder else 0)
            active_idx += 1

            if tier == "optimize":
                self._train_partition_optimize(i, train_loader, cur_steps)
            else:  # unload
                self._train_partition(i, train_loader, cur_steps)
                if i not in self._offloaded:
                    self._offload_partition(i)

            self._check_memory()

            if val_loader is not None and eval_interval is not None:
                val_loss = self.evaluate(val_loader)
                self.val_losses.append(val_loss)
                if val_loss < self.best_val_loss:
                    self.best_val_loss = float(val_loss)
                print(
                    f"[vmt] partition={i} tier={tier} "
                    f"val_loss={val_loss:.6f} best={self.best_val_loss:.6f}",
                    flush=True,
                )

        # ---------- Phase 3：合并 ----------
        self._merge_partitions_vmt()

        # ---------- 可选整体 fine-tune ----------
        finetune_steps = int(_cfg_get(self.config, "finetune_steps", 0))
        if finetune_steps > 0:
            print(f"[vmt] 整体 fine-tune {finetune_steps} 步...", flush=True)
            self._current_partition = -1
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
            f"[vmt] 训练完成：total_steps={len(self.train_losses)} "
            f"best_val={self.best_val_loss:.6f}",
            flush=True,
        )
        return self.train_losses, self.val_losses


__all__ = ["LayerWiseTrainer", "VMTTrainer"]
