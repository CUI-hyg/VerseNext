"""verse_torch.compress: 模型压缩 PoC 模块（阶段 5，Task 5.2-5.6）.

提供：
- ``OutlierSafePruner``: 结构化剪枝（mask + 冻结策略，PoC 简化版）
- ``LoRALinear``: 低秩适配器（base frozen + A/B trainable）
- ``KnowledgeDistiller``: 知识蒸馏（Hinton KL + CE 联合损失）
- ``compress_pipeline``: 端到端压缩流程（prune → quantize → lora_wrap，可配置）
- 单技术函数: ``prune_only`` / ``quantize_only`` / ``lora_only`` /
  ``distill_only`` / ``ternary_only``
- ``QLinear``: 把 ``QuantizedLinear`` 包装为 ``nn.Module`` 子类，便于嵌入模型树

设计约束：
- 仅使用 NumPy + 标准库，不依赖 torch/tensorflow/jax。
- 剪枝采用 mask 策略（保留原结构、将被剪参数置零），便于后续量化/LoRA 包装。
- 压缩比按 bit-level 精确计算（INT4=4bit, INT8=8bit, ternary=2bit, fp32=32bit）。

参考论文见 ``docs/papers/compression_references.md``。
"""

from __future__ import annotations

import numpy as np

from .tensor import Tensor, no_grad
from . import nn
from .losses import cross_entropy, kl_div_loss, mse_loss
from .quantize import QuantizedLinear


# ---------------------------------------------------------------------------
# 内部辅助：参数量 / bit 数统计
# ---------------------------------------------------------------------------


def _iter_all_tensors(model):
    """递归生成所有 Tensor 参数（包括 requires_grad=False）。"""
    for p in model._parameters.values():
        yield p
    for m in model._modules.values():
        yield from _iter_all_tensors(m)


def count_parameters(model) -> int:
    """统计模型总参数量（包括 frozen 与未参与计算图的 Tensor）。"""
    total = 0
    for p in _iter_all_tensors(model):
        total += int(p.data.size)
    return total


def count_nonzero_params(model) -> int:
    """统计非零参数量（用于稀疏度评估）。"""
    total = 0
    for p in _iter_all_tensors(model):
        total += int(np.count_nonzero(p.data))
    return total


# ---------------------------------------------------------------------------
# Task 5.2: OutlierSafePruner
# ---------------------------------------------------------------------------


class OutlierSafePruner:
    """按 head/channel 维度的 |weight|_mean 进行结构化剪枝（mask + 冻结策略）。

    策略（推荐 mask 方式：保留原模型结构，将剪掉的参数置零）：
    - GQASelfAttention: 按 n_head 维度计算 |wq_weight|_mean，
      剪掉 bottom ``sparsity`` 比例的 head（同时 mask wq 行与 proj 列）
    - SwiGLUMLP: 按 hidden 维度计算 |w_gate|+|w_up| 的 mean，
      剪掉 bottom ``sparsity`` 比例（同时 mask w_gate/w_up 行与 w_down 列）
    - Linear（非上述子模块内）: 按 output channel 剪
    - Embedding / head (tie_weights): 跳过（避免破坏词表语义）

    Args:
        model: nn.Module 模型（如 TransformerLM）
        sparsity: 剪枝比例（0-1），如 0.3 表示剪掉 30%
    """

    SKIP_NAME_PATTERNS = ("tok_emb", "head")  # 跳过 embedding 层

    def __init__(self, model, sparsity: float = 0.3):
        self.model = model
        self.sparsity = float(sparsity)

    def apply(self):
        """执行剪枝，返回 (model, report)。"""
        report = {}
        processed_ids = set()
        for name, m in self.model.named_modules():
            if id(m) in processed_ids or name == "":
                continue
            if self._should_skip(name):
                continue
            if isinstance(m, nn.GQASelfAttention):
                self._prune_attention(name, m, report)
                for _, sub_m in m.named_modules():
                    processed_ids.add(id(sub_m))
            elif isinstance(m, nn.SwiGLUMLP):
                self._prune_mlp(name, m, report)
                for _, sub_m in m.named_modules():
                    processed_ids.add(id(sub_m))
            elif isinstance(m, nn.Linear):
                self._prune_linear(name, m, report)
        return self.model, report

    def _should_skip(self, name: str) -> bool:
        return any(p in name for p in self.SKIP_NAME_PATTERNS)

    def _prune_attention(self, name: str, m, report: dict):
        n_head = m.n_head
        n_prune = int(np.floor(n_head * self.sparsity))
        if n_prune == 0:
            return
        head_dim = m.head_dim
        wq = m.wq.weight.data  # (n_head * head_dim, d)
        # 每个 head 的 score = mean(|wq[head_segment]|)
        head_scores = np.mean(np.abs(wq.reshape(n_head, head_dim, -1)), axis=(1, 2))
        prune_heads = np.argsort(head_scores)[:n_prune]
        # mask wq 对应行
        for h in prune_heads:
            wq[h * head_dim:(h + 1) * head_dim] = 0.0
        # mask proj 对应列（proj weight shape: (d, n_head * head_dim)）
        proj_w = m.proj.weight.data
        for h in prune_heads:
            proj_w[:, h * head_dim:(h + 1) * head_dim] = 0.0
        orig = int(wq.size + proj_w.size)
        pruned = int(n_prune * head_dim * wq.shape[1]
                     + n_prune * head_dim * proj_w.shape[0])
        report[name] = {
            "type": "GQASelfAttention",
            "n_head": int(n_head),
            "n_pruned_head": int(n_prune),
            "original_params": orig,
            "kept_params": orig - pruned,
            "prune_ratio": float(pruned / orig) if orig > 0 else 0.0,
        }

    def _prune_mlp(self, name: str, m, report: dict):
        hidden = m.hidden
        n_prune = int(np.floor(hidden * self.sparsity))
        if n_prune == 0:
            return
        wg = m.w_gate.weight.data  # (hidden, d)
        wu = m.w_up.weight.data    # (hidden, d)
        wd = m.w_down.weight.data  # (d, hidden)
        scores = np.mean(np.abs(wg), axis=1) + np.mean(np.abs(wu), axis=1)
        prune_indices = np.argsort(scores)[:n_prune]
        wg[prune_indices] = 0.0
        wu[prune_indices] = 0.0
        wd[:, prune_indices] = 0.0
        orig = int(wg.size + wu.size + wd.size)
        pruned = int(n_prune * (wg.shape[1] + wu.shape[1] + wd.shape[0]))
        report[name] = {
            "type": "SwiGLUMLP",
            "hidden": int(hidden),
            "n_pruned": int(n_prune),
            "original_params": orig,
            "kept_params": orig - pruned,
            "prune_ratio": float(pruned / orig) if orig > 0 else 0.0,
        }

    def _prune_linear(self, name: str, m, report: dict):
        w = m.weight.data  # (out, in)
        out_dim = w.shape[0]
        n_prune = int(np.floor(out_dim * self.sparsity))
        if n_prune == 0:
            return
        scores = np.mean(np.abs(w), axis=1)
        prune_indices = np.argsort(scores)[:n_prune]
        w[prune_indices] = 0.0
        bias_size = 0
        bias_pruned = 0
        if m.bias is not None:
            m.bias.data[prune_indices] = 0.0
            bias_size = int(m.bias.data.size)
            bias_pruned = int(n_prune)
        orig = int(w.size + bias_size)
        pruned = int(n_prune * w.shape[1] + bias_pruned)
        report[name] = {
            "type": "Linear",
            "out_features": int(out_dim),
            "n_pruned": int(n_prune),
            "original_params": orig,
            "kept_params": orig - pruned,
            "prune_ratio": float(pruned / orig) if orig > 0 else 0.0,
        }


# ---------------------------------------------------------------------------
# Task 5.3: LoRALinear
# ---------------------------------------------------------------------------


class LoRALinear(nn.Module):
    """LoRA 包装层：frozen base + 低秩增量 A @ B。

    forward: ``y = base(x) + (x @ A) @ B * (alpha / r)``

    - ``base``: 冻结的 Linear / QLinear（requires_grad=False）
    - ``A``: Tensor (d_in, r)，高斯初始化（std=0.01）
    - ``B``: Tensor (r, d_out)，零初始化（保证训练初始 ΔW = 0）

    Args:
        d_in: 输入维度
        d_out: 输出维度
        r: LoRA 秩（默认 8）
        alpha: LoRA 缩放因子（默认 16），实际 scale = alpha / r
        base: 可选，传入已有的 Linear/QLinear 作为 frozen base（不传入则新建 Linear）
    """

    def __init__(self, d_in: int, d_out: int, r: int = 8,
                 alpha: float = 16.0, base=None):
        super().__init__()
        if base is not None:
            # 复用已有 base，从 base 推断维度
            d_in = base.in_features
            d_out = base.out_features
        self.d_in = int(d_in)
        self.d_out = int(d_out)
        self.r = int(r)
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.r
        # 初始化 base
        if base is None:
            base = nn.Linear(self.d_in, self.d_out, bias=True)
        # 冻结 base 的所有参数
        for p in base.parameters():
            p.requires_grad = False
        # base 是 nn.Module 子类（Linear 或 QLinear），setattr 会注册到 _modules
        self.base = base
        # A: (d_in, r) 高斯初始化（std=0.01，避免初始 lora 输出过大）
        a_data = (np.random.randn(self.d_in, self.r) * 0.01).astype(np.float32)
        self.A = Tensor(a_data, requires_grad=True)
        # B: (r, d_out) 零初始化（保证训练初始 ΔW = A @ B = 0）
        b_data = np.zeros((self.r, self.d_out), dtype=np.float32)
        self.B = Tensor(b_data, requires_grad=True)

    def forward(self, x: Tensor) -> Tensor:
        # base(x): frozen，不构建梯度图（base 内部参数 requires_grad=False）
        base_out = self.base(x)
        if not isinstance(base_out, Tensor):
            base_out = Tensor(base_out, requires_grad=False)
        # lora: (x @ A) @ B * scaling
        # x: (..., d_in), A: (d_in, r), B: (r, d_out)
        lora_out = (x @ self.A) @ self.B
        lora_out = lora_out * self.scaling
        return base_out + lora_out

    def merge(self) -> "nn.Linear":
        """将 A @ B 加到 base.weight 上，返回合并后的新 Linear（无 LoRA 开销）。

        注意：仅当 base 是 nn.Linear 时支持完整合并；
        QLinear base 不支持（量化权重无法直接相加，会抛出 NotImplementedError）。
        """
        if not isinstance(self.base, nn.Linear):
            raise NotImplementedError(
                "merge() only supports nn.Linear base; QLinear base not supported."
            )
        # ΔW = A @ B * scaling，shape (d_in, r) @ (r, d_out) = (d_in, d_out)
        # base.weight shape (d_out, d_in)，所以 ΔW.T = (d_out, d_in)
        # 注意：merge 后的权重应等价于 forward 中的 base + lora_out，
        # 因此必须乘 scaling（与 forward 中的 lora_out = (x@A@B) * scaling 一致）
        delta_w = (self.A.data @ self.B.data).T * self.scaling  # (d_out, d_in)
        new_w = self.base.weight.data + delta_w
        # 构造新 Linear（复用 base 的 bias）
        new_linear = nn.Linear(self.d_in, self.d_out,
                               bias=self.base.bias is not None)
        new_linear.weight.data = new_w.astype(np.float32)
        if self.base.bias is not None:
            new_linear.bias.data = self.base.bias.data.copy()
        return new_linear

    def extra_repr(self) -> str:
        return (f"d_in={self.d_in}, d_out={self.d_out}, "
                f"r={self.r}, alpha={self.alpha}, scaling={self.scaling:.4f}")


# ---------------------------------------------------------------------------
# Task 5.4: KnowledgeDistiller
# ---------------------------------------------------------------------------


class KnowledgeDistiller:
    """知识蒸馏器 V1.3：大模型 → 小模型能力转移（以小博大）。

    V1.3 损失 = alpha * T^2 * KL(teacher/T || student/T)     # 软标签蒸馏
              + (1 - alpha) * CE(student, labels)            # 硬标签蒸馏
              + feature_loss_weight * MSE(student_feat, teacher_feat)  # 中间层特征匹配

    V1.3 增强点：
    - **中间层特征蒸馏**（feature-level distillation）：匹配 teacher / student 的
      中间层输出，使小模型学到等效能力（``distill_layers`` + ``feature_loss_weight``）。
    - **自适应温度调度**（temperature annealing）：训练过程中温度从高到低渐变，
      前期放大软标签信息、后期收敛到尖锐分布。
    - **三重损失**：软标签 + 硬标签 + 特征匹配。

    Args:
        teacher: frozen 教师模型（自动 eval + requires_grad=False）
        student: 可训练学生模型
        temperature: 蒸馏温度（默认 4.0），soft target 平滑度
        alpha: soft loss 权重（默认 0.7），(1-alpha) 为 hard loss 权重
        distill_layers: 指定蒸馏的中间层名称列表（feature-level distillation）；
            目前仅作元数据记录，真正的特征提取通过 ``feature_extractor`` 回调完成。
        feature_loss_weight: 中间层特征匹配损失权重（默认 0.3）
        T: ``temperature`` 的旧参数别名（向后兼容 V1.0）；优先级低于 ``temperature``
    """

    def __init__(self, teacher, student, temperature: float = 4.0,
                 alpha: float = 0.7, distill_layers=None,
                 feature_loss_weight: float = 0.3, T: float = None):
        self.teacher = teacher
        self.student = student
        # T 是旧参数别名：仅当显式传入（非 None）时覆盖 temperature
        if T is not None:
            temperature = T
        self.temperature = float(temperature)
        self.alpha = float(alpha)
        self.distill_layers = list(distill_layers) if distill_layers else None
        self.feature_loss_weight = float(feature_loss_weight)
        # 温度退火调度：从初始温度线性退火到 T_min
        self._T_init = self.temperature
        self._T_min = max(1.0, self.temperature * 0.25)
        # 冻结 teacher：eval 模式 + 所有参数 requires_grad=False
        self.teacher.eval()
        for p in _iter_all_params_static(teacher):
            p.requires_grad = False

    # ------------------------------------------------------------------
    # T 属性：向后兼容 V1.0（self.T 读写映射到 self.temperature）
    # ------------------------------------------------------------------
    @property
    def T(self) -> float:
        return self.temperature

    @T.setter
    def T(self, value: float):
        self.temperature = float(value)
        # 重新初始化退火调度基准（仅在用户显式设 T 时更新）
        self._T_init = self.temperature
        self._T_min = max(1.0, self.temperature * 0.25)

    def compute_loss(self, student_logits: Tensor, teacher_logits: Tensor,
                     student_features=None, teacher_features=None,
                     labels=None) -> Tensor:
        """V1.3 联合损失计算。

        Loss = alpha * T^2 * KL(teacher/T || student/T)
             + (1 - alpha) * CE(student, labels)
             + feature_loss_weight * MSE(student_features, teacher_features)

        Args:
            student_logits: 学生模型 logits（可微）
            teacher_logits: 教师模型 logits（内部 detach，不回传梯度）
            student_features: 学生中间层特征，``Tensor`` / ``list[Tensor]`` / ``None``
            teacher_features: 教师中间层特征，``Tensor`` / ``list[Tensor]`` / ``None``
            labels: 硬标签（``None`` 时跳过 CE 项）

        Returns:
            标量 Tensor，支持 backward
        """
        T = self.temperature
        # soft loss: KL(softmax(teacher/T) || log_softmax(student/T)) * T^2
        # teacher_logits.detach() 切断梯度，不回传到 teacher
        teacher_probs = (teacher_logits.detach() / T).softmax(dim=-1)
        student_log_probs = (student_logits / T).log_softmax(dim=-1)
        soft_loss = kl_div_loss(student_log_probs, teacher_probs) * (T * T)
        # hard loss: CE(student, labels)
        if labels is not None:
            hard_loss = cross_entropy(student_logits, labels)
            total = self.alpha * soft_loss + (1.0 - self.alpha) * hard_loss
        else:
            # 无硬标签时，soft loss 全权
            total = self.alpha * soft_loss
        # feature matching loss（V1.3 新增）
        feat_loss = self._feature_loss(student_features, teacher_features)
        if feat_loss is not None:
            total = total + self.feature_loss_weight * feat_loss
        return total

    def _feature_loss(self, student_features, teacher_features):
        """中间层特征匹配损失（MSE），返回标量 Tensor 或 None。"""
        if (student_features is None or teacher_features is None
                or self.feature_loss_weight == 0.0):
            return None
        s_list = (student_features if isinstance(student_features, (list, tuple))
                  else [student_features])
        t_list = (teacher_features if isinstance(teacher_features, (list, tuple))
                  else [teacher_features])
        n = min(len(s_list), len(t_list))
        if n == 0:
            return None
        total = None
        for i in range(n):
            s, t = self._align_features(s_list[i], t_list[i])
            # teacher 特征 detach，仅 student 侧回传梯度
            l = mse_loss(s, t.detach())
            total = l if total is None else total + l
        return total / float(n)

    @staticmethod
    def _align_features(student_feat: Tensor, teacher_feat: Tensor):
        """对齐 student / teacher 特征的最后一维（不同维时截断到较小者）。"""
        sd = int(student_feat.data.shape[-1])
        td = int(teacher_feat.data.shape[-1])
        if sd == td:
            return student_feat, teacher_feat
        target = min(sd, td)
        if sd != target:
            student_feat = student_feat[..., :target]
        if td != target:
            teacher_feat = teacher_feat[..., :target]
        return student_feat, teacher_feat

    def forward(self, student_logits: Tensor, teacher_logits: Tensor,
                hard_targets) -> Tensor:
        """V1.0 兼容前向：等价于 ``compute_loss(..., labels=hard_targets)``。"""
        return self.compute_loss(student_logits, teacher_logits,
                                 labels=hard_targets)

    def __call__(self, student_logits, teacher_logits, hard_targets=None,
                 student_features=None, teacher_features=None):
        return self.compute_loss(student_logits, teacher_logits,
                                 student_features=student_features,
                                 teacher_features=teacher_features,
                                 labels=hard_targets)

    def distill(self, train_loader, epochs=3, lr=1e-3, optimizer=None,
                max_steps=None, eval_fn=None, eval_every: int = 0,
                feature_extractor=None, anneal_temperature: bool = True):
        """端到端蒸馏训练（V1.3）。

        Args:
            train_loader: 可迭代对象，每次返回 ``(x, y)`` 或 ``(x, y, *rest)``
            epochs: 训练轮数（V1.3 默认 3）
            lr: 学习率（``optimizer=None`` 时内部创建 AdamW）
            optimizer: 可选优化器（向后兼容 V1.0：旧调用 ``distill(loader, optimizer,
                max_steps=...)`` 会把 optimizer 作为第 2 个位置参数传入）
            max_steps: 最大训练步数上限（``None`` 表示不限，按 epochs 遍历）
            eval_fn: 可选回调 ``(step, student) -> None``
            eval_every: 每隔多少步调用 eval_fn
            feature_extractor: 可选 ``model -> (logits, features)`` 回调；
                传入时启用中间层特征蒸馏，否则仅做 logit 级蒸馏
            anneal_temperature: 是否启用自适应温度调度（从 ``temperature`` 线性
                退火到 ``_T_min``）

        Returns:
            训练损失历史 ``list[float]``
        """
        # 向后兼容：旧 API distill(train_loader, optimizer, max_steps=, ...)
        # 第 2 个位置参数（epochs）被当成 optimizer 传入的情况
        if not isinstance(epochs, (int, float)):
            optimizer = epochs
            epochs = 3
        epochs = int(epochs)
        if optimizer is None:
            from .optim import AdamW
            optimizer = AdamW(self.student.parameters(), lr=lr)
        if max_steps is not None:
            max_steps = int(max_steps)

        losses_hist = []
        total_steps = 0
        self.student.train()
        for epoch in range(epochs):
            # 自适应温度调度：线性退火
            if anneal_temperature and epochs > 1:
                frac = epoch / (epochs - 1)
                self.temperature = self._T_init + (self._T_min - self._T_init) * frac
            for batch in train_loader:
                if max_steps is not None and total_steps >= max_steps:
                    break
                x, y = batch[0], batch[1]
                optimizer.zero_grad()
                # teacher forward（no_grad，不构建计算图）
                with no_grad():
                    if feature_extractor is not None:
                        teacher_logits, teacher_feats = feature_extractor(
                            self.teacher, x)
                    else:
                        teacher_logits = self.teacher(x)
                        teacher_feats = None
                # student forward（构建计算图）
                if feature_extractor is not None:
                    student_logits, student_feats = feature_extractor(
                        self.student, x)
                else:
                    student_logits = self.student(x)
                    student_feats = None
                loss = self.compute_loss(
                    student_logits, teacher_logits,
                    student_features=student_feats,
                    teacher_features=teacher_feats, labels=y)
                loss.backward()
                optimizer.step()
                losses_hist.append(float(loss.data))
                total_steps += 1
                if (eval_fn is not None and eval_every > 0
                        and total_steps % eval_every == 0):
                    eval_fn(total_steps, self.student)
            if max_steps is not None and total_steps >= max_steps:
                break
        return losses_hist


# 静态辅助：递归生成所有 Tensor 参数（与 _iter_all_tensors 同义，避免前向引用）
def _iter_all_params_static(model):
    for p in model._parameters.values():
        yield p
    for m in model._modules.values():
        yield from _iter_all_params_static(m)


# ---------------------------------------------------------------------------
# Task 5.6: QLinear 包装类 + 单技术函数
# ---------------------------------------------------------------------------


class QLinear(nn.Module):
    """量化 Linear 包装器（可作为 nn.Module 嵌入模型树）。

    内部持有 ``QuantizedLinear``（推理专用，无可训练参数）。
    forward 时调用 QuantizedLinear.forward，保证返回 Tensor。
    """

    def __init__(self, linear: "nn.Linear", qtype: str = "int4",
                 cache_fp32: bool = True):
        super().__init__()
        self.in_features = int(linear.in_features)
        self.out_features = int(linear.out_features)
        self.qtype = qtype
        # QuantizedLinear 不是 nn.Module，存到 __dict__（Module.__setattr__ 走 else 分支）
        self._qlin = QuantizedLinear(linear, qtype=qtype, cache_fp32=cache_fp32)

    def forward(self, x: Tensor) -> Tensor:
        out = self._qlin(x)
        if not isinstance(out, Tensor):
            out = Tensor(out, requires_grad=False)
        return out

    def extra_repr(self) -> str:
        return (f"in_features={self.in_features}, "
                f"out_features={self.out_features}, qtype={self.qtype}")


def _qlinear_bits(qlinear_module: "QLinear") -> int:
    """计算 QLinear 的存储 bit 数（packed + scale + bias）。"""
    ql = qlinear_module._qlin
    bits = int(ql.packed.nbytes * 8)  # packed 权重
    bits += int(ql.scale.nbytes * 8)  # per-channel scale
    if ql.bias is not None:
        bits += int(ql.bias.nbytes * 8)
    return bits


def compute_compressed_bits(model) -> int:
    """递归计算压缩后的总存储 bit 数。

    规则：
    - LoRALinear: base 按其类型计算 + A/B 按 fp32 (32 bit)
    - QLinear: 按 packed + scale + bias 的实际字节数
    - 其他 Module（Linear/Embedding/Norm）: 按 fp32 (32 bit) 计
    """
    bits = 0
    # model 自身的参数（fp32）
    for p in model._parameters.values():
        bits += int(p.data.size * 32)
    # 子模块
    for m in model._modules.values():
        if isinstance(m, LoRALinear):
            # base 按 QLinear 或 Linear 计算
            if isinstance(m.base, QLinear):
                bits += _qlinear_bits(m.base)
            elif isinstance(m.base, nn.Linear):
                bits += int(m.base.weight.data.size * 32)
                if m.base.bias is not None:
                    bits += int(m.base.bias.data.size * 32)
            # A, B 按 fp32
            bits += int(m.A.data.size * 32)
            bits += int(m.B.data.size * 32)
        elif isinstance(m, QLinear):
            bits += _qlinear_bits(m)
        else:
            # 递归普通 Module
            bits += compute_compressed_bits(m)
    return bits


def _quantize_module(model, qtype: str = "int4"):
    """递归把所有 nn.Linear 替换成 QLinear（原地修改）。"""
    for name, child in list(model._modules.items()):
        if isinstance(child, nn.Linear):
            qlin = QLinear(child, qtype=qtype, cache_fp32=True)
            setattr(model, name, qlin)
        elif isinstance(child, nn.Module):
            _quantize_module(child, qtype=qtype)


def _lora_wrap_module(model, r: int = 8, alpha: float = 16.0):
    """递归把所有 QLinear / Linear 包装成 LoRALinear（原地修改）。"""
    for name, child in list(model._modules.items()):
        if isinstance(child, (QLinear, nn.Linear)):
            lora = LoRALinear(child.in_features, child.out_features,
                              r=r, alpha=alpha, base=child)
            setattr(model, name, lora)
        elif isinstance(child, nn.Module):
            _lora_wrap_module(child, r=r, alpha=alpha)


def prune_only(model, sparsity: float = 0.3):
    """仅剪枝：返回 (pruned_model, report)。

    Args:
        model: nn.Module 模型
        sparsity: 剪枝比例（0-1）

    Returns:
        (model, report) —— model 是原地修改后的模型，report 是 dict
    """
    pruner = OutlierSafePruner(model, sparsity=sparsity)
    return pruner.apply()


def quantize_only(model, dtype: str = "int4"):
    """仅量化：把模型中所有 nn.Linear 替换为 QLinear。

    Args:
        model: nn.Module 模型
        dtype: 量化类型，"int4" / "int8" / "ternary"

    Returns:
        修改后的 model（原地替换 Linear → QLinear）
    """
    if dtype not in ("int4", "int8", "ternary"):
        raise ValueError(f"Unknown dtype: {dtype!r}, expected int4/int8/ternary")
    _quantize_module(model, qtype=dtype)
    return model


def lora_only(model, r: int = 8, alpha: float = 16.0):
    """仅 LoRA 包装：把模型中所有 Linear/QLinear 包装为 LoRALinear。

    Args:
        model: nn.Module 模型
        r: LoRA 秩
        alpha: LoRA 缩放因子

    Returns:
        修改后的 model（原地包装）
    """
    _lora_wrap_module(model, r=r, alpha=alpha)
    return model


def ternary_only(model):
    """仅 ternary 量化（BitNet b1.58 风格，2 bit/value）。"""
    return quantize_only(model, dtype="ternary")


def distill_only(teacher, student, train_loader, max_steps: int = 100,
                 T: float = 2.0, alpha: float = 0.5, lr: float = 1e-3,
                 eval_fn=None, eval_every: int = 0):
    """仅蒸馏：用 teacher 蒸馏 student，返回训练后的 student。

    Args:
        teacher: frozen 教师模型
        student: 可训练学生模型
        train_loader: 可迭代对象，每次返回 (x, y)
        max_steps: 最大训练步数
        T: 温度
        alpha: soft loss 权重
        lr: 学习率（默认 1e-3）
        eval_fn: 可选回调
        eval_every: 每隔多少步调用 eval_fn

    Returns:
        训练后的 student
    """
    from .optim import AdamW
    optimizer = AdamW(student.parameters(), lr=lr)
    distiller = KnowledgeDistiller(teacher, student, T=T, alpha=alpha)
    distiller.distill(train_loader, optimizer, max_steps=max_steps,
                      eval_fn=eval_fn, eval_every=eval_every)
    return student


# ---------------------------------------------------------------------------
# Task 5.5: compress_pipeline
# ---------------------------------------------------------------------------


def _parse_version_tuple(v) -> tuple:
    """把版本字符串解析为可比较的整数元组，如 ``"1.3.0"`` → ``(1, 3, 0)``。

    用于替代字符串直接比较，避免 ``"1.30"`` / ``"1.3.0"`` 等等价版本
    因字符串不等而被误判。
    """
    parts = []
    for p in str(v).split("."):
        try:
            parts.append(int(p))
        except ValueError:
            parts.append(0)
    return tuple(parts) if parts else (0,)


def compress_pipeline(model, config=None, return_stats: bool = False,
                      version: str = "1.3",
                      # 旧 API 向后兼容参数（任一非 None / 非 dict 时走旧 API）
                      target_ratio: float = 0.1, eval_fn=None,
                      sparsity: float = 0.3, qtype: str = "int4",
                      lora_r: int = 8, lora_alpha: float = 16.0,
                      use_lora: bool = False):
    """一键压缩管线，支持任意组合 prune/quantize/lora/ternary/distill。

    支持两种调用方式：

    **新 API（推荐）**::

        compressed_model = compress_pipeline(model, config_dict)
        compressed_model, stats = compress_pipeline(model, config_dict, return_stats=True)

    其中 ``config_dict`` 的 key 是压缩方法，value 是参数::

        {
            "prune":     {"sparsity": 0.5, "method": "outlier_safe"},
            "quantize":  {"bits": 4, "schema": "symmetric"},
            "lora":      {"rank": 8, "alpha": 16},
            "ternary":   {},
            "distill":   {"teacher": teacher_model, "epochs": 10, "lr": 1e-4}
        }

    V1.3 新增：``config`` 顶层可放 ``"teacher_model"`` / ``"teacher"`` 作为蒸馏
    教师的便捷入口（等价于 ``config["distill"]["teacher"]``），并可选 ``"train_loader"``。
    当 ``teacher_model`` 存在但无 ``train_loader`` 时，仅冻结 teacher、为学生做好准备
    （不实际训练）；有 ``train_loader`` 时执行端到端蒸馏。

    新 API **不修改原模型**：内部深拷贝 model 后再应用管线。

    **旧 API（向后兼容）**::

        stats = compress_pipeline(model, target_ratio=0.1, qtype="int4",
                                   sparsity=0.3, use_lora=False)

    旧 API **原地修改** model，返回统计 dict（与 v1 行为一致）。

    Args:
        model: 待压缩模型（需是 Module 子类）
        config: dict，新 API 的压缩配置；若为 None 则走旧 API
        return_stats: 新 API 下是否返回 (compressed_model, stats) 元组
            （旧 API 始终返回 stats dict，此参数被忽略）
        version: 压缩管线版本。``"1.3"``（默认，以小博大：prune → quantize →
            distill → lora，含压缩报告）；``"1.0"`` / ``"1.2"`` 走旧 v2 流程
            （prune → quantize → lora → ternary → distill）。仅对 dict 配置生效。
        target_ratio / eval_fn / sparsity / qtype / lora_r / lora_alpha / use_lora:
            旧 API 参数，仅当 ``config`` 不是 dict 时生效

    Returns:
        - 新 API + ``return_stats=False``: 返回压缩后的新 model
        - 新 API + ``return_stats=True``: 返回 ``(new_model, stats_dict)``
        - 旧 API（``config`` 非 dict）: 返回 stats_dict（原地修改 model）
    """
    # ------------------------------------------------------------------
    # 分派：新 API vs 旧 API
    # ------------------------------------------------------------------
    if isinstance(config, dict):
        # Part4K2.5 Task 5：用版本号元组比较替代字符串比较，
        # 确保 "1.3" / "1.30" / "1.3.0" 等等价版本都能正确走 v13 分支
        if _parse_version_tuple(version) >= (1, 3):
            return _compress_pipeline_v13(model, config, return_stats=return_stats)
        return _compress_pipeline_v2(model, config, return_stats=return_stats)
    return _compress_pipeline_v1(
        model,
        target_ratio=target_ratio,
        eval_fn=eval_fn,
        sparsity=sparsity,
        qtype=qtype,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        use_lora=use_lora,
    )


def _compress_pipeline_v1(model, target_ratio: float = 0.1, eval_fn=None,
                          sparsity: float = 0.3, qtype: str = "int4",
                          lora_r: int = 8, lora_alpha: float = 16.0,
                          use_lora: bool = False):
    """旧版 compress_pipeline（v1，原地修改 + 返回 stats dict）。

    保留此函数确保向后兼容：现有测试与下游代码（test_compression_poc.py /
    verse_inference 等）依赖此 API。
    """
    original_params = count_parameters(model)
    original_bits = int(original_params * 32)

    # 压缩前 eval
    original_loss = None
    if eval_fn is not None:
        with no_grad():
            original_loss = float(eval_fn(model))

    steps = []

    # 1. prune（mask + 冻结策略，原模型结构不变）
    pruner = OutlierSafePruner(model, sparsity=sparsity)
    _, prune_report = pruner.apply()
    steps.append({
        "step": "prune",
        "sparsity": float(sparsity),
        "report": prune_report,
    })

    # 2. quantize（把所有 nn.Linear 替换为 QLinear）
    quantize_only(model, dtype=qtype)
    steps.append({
        "step": "quantize",
        "qtype": qtype,
    })

    # 3. lora wrap（可选，QLoRA 风格）
    if use_lora:
        lora_only(model, r=lora_r, alpha=lora_alpha)
        steps.append({
            "step": "lora_wrap",
            "r": int(lora_r),
            "alpha": float(lora_alpha),
        })

    # 计算压缩后 bit 数（精确版）
    compressed_bits = compute_compressed_bits(model)
    # 等效 fp32 参数量（用于报告展示）
    compressed_params = compressed_bits / 32
    compression_ratio = (original_bits / compressed_bits
                        if compressed_bits > 0 else float("inf"))

    # 压缩后 eval
    compressed_loss = None
    loss_diff_pct = None
    if eval_fn is not None:
        with no_grad():
            compressed_loss = float(eval_fn(model))
        if original_loss is not None and original_loss > 0:
            loss_diff_pct = (abs(compressed_loss - original_loss)
                            / original_loss * 100)

    return {
        "original_params": int(original_params),
        "compressed_params": float(compressed_params),
        "compressed_bits": int(compressed_bits),
        "original_bits": int(original_bits),
        "compression_ratio": float(compression_ratio),
        "original_loss": original_loss,
        "compressed_loss": compressed_loss,
        "loss_diff_pct": loss_diff_pct,
        "steps": steps,
    }


def _deep_copy_model(model):
    """深拷贝模型（用于 v2 pipeline 的非破坏式压缩）。

    优先尝试 ``copy.deepcopy``；若失败（罕见，如某些自定义属性不可 pickle），
    降级到 state_dict 复制路径：基于 ``model.config`` 重建同型模型并加载权重。
    """
    import copy as _copy
    try:
        return _copy.deepcopy(model)
    except Exception:
        # 降级：假设 model 有 .config 属性（如 CometSparkLM）
        cfg = getattr(model, "config", None)
        if cfg is None:
            raise
        cls = type(model)
        new_model = cls(cfg)
        sd = model.state_dict() if hasattr(model, "state_dict") else {}
        if hasattr(new_model, "load_state_dict"):
            new_model.load_state_dict(
                {k: v.copy() for k, v in sd.items()}, strict=False
            )
        return new_model


def _compress_pipeline_v2(model, config: dict, return_stats: bool = False):
    """新版 compress_pipeline（v2）：dict 配置 + 不修改原模型。

    按 prune → quantize → lora → ternary → distill 顺序应用，
    每步可选；返回压缩后的新模型（深拷贝），可选附加统计 dict。
    """
    import copy as _copy

    # 记录原始参数量与 bit 数（压缩前）
    original_params = count_parameters(model)
    original_bits = int(original_params * 32)

    # 深拷贝模型，确保不修改原模型
    new_model = _deep_copy_model(model)

    steps = []
    sparsity_applied = 0.0
    # 实际稀疏度：在 prune 之后、quantize 之前测量（quantize 后 Linear 权重
    # 被打包为 uint8，count_nonzero_params 无法正确统计稀疏度）
    actual_sparsity = 0.0
    qtype_applied = None
    has_quantize = False

    # ------------------------------------------------------------------
    # 1. prune（可选）
    # ------------------------------------------------------------------
    if "prune" in config and config["prune"] is not None:
        prune_cfg = dict(config["prune"])  # 浅拷贝避免污染用户输入
        sparsity = float(prune_cfg.pop("sparsity", 0.3))
        # method 字段当前仅支持 "outlier_safe"（OutlierSafePruner）
        method = prune_cfg.pop("method", "outlier_safe")
        if method not in ("outlier_safe", None):
            raise ValueError(
                f"prune.method 仅支持 'outlier_safe'，得到 {method!r}"
            )
        pruner = OutlierSafePruner(new_model, sparsity=sparsity)
        _, prune_report = pruner.apply()
        sparsity_applied = sparsity
        # 在 quantize 前测量实际稀疏度（此时 Linear 仍是 fp32 Tensor）
        nonzero_after_prune = count_nonzero_params(new_model)
        actual_sparsity = (1.0 - nonzero_after_prune / original_params
                           if original_params > 0 else 0.0)
        steps.append({
            "step": "prune",
            "sparsity": sparsity,
            "method": method,
            "report": prune_report,
            "actual_sparsity": float(actual_sparsity),
        })

    # ------------------------------------------------------------------
    # 2. quantize（可选）：bits=4 → int4, bits=8 → int8
    # ------------------------------------------------------------------
    if "quantize" in config and config["quantize"] is not None:
        q_cfg = dict(config["quantize"])
        bits = int(q_cfg.pop("bits", 4))
        # schema 字段当前仅支持 "symmetric"（量化模块实现）
        schema = q_cfg.pop("schema", "symmetric")
        if schema not in ("symmetric", None):
            raise ValueError(
                f"quantize.schema 仅支持 'symmetric'，得到 {schema!r}"
            )
        if bits == 4:
            qtype_applied = "int4"
        elif bits == 8:
            qtype_applied = "int8"
        else:
            raise ValueError(
                f"quantize.bits 仅支持 4 或 8，得到 {bits}"
            )
        quantize_only(new_model, dtype=qtype_applied)
        has_quantize = True
        steps.append({
            "step": "quantize",
            "bits": bits,
            "qtype": qtype_applied,
            "schema": schema,
        })

    # ------------------------------------------------------------------
    # 3. lora（可选）：在 quantize 之后包装
    # ------------------------------------------------------------------
    if "lora" in config and config["lora"] is not None:
        lora_cfg = dict(config["lora"])
        rank = int(lora_cfg.pop("rank", 8))
        alpha = float(lora_cfg.pop("alpha", 16.0))
        lora_only(new_model, r=rank, alpha=alpha)
        steps.append({
            "step": "lora",
            "rank": rank,
            "alpha": alpha,
        })

    # ------------------------------------------------------------------
    # 4. ternary（可选）：等价于 quantize_only(dtype="ternary")
    # 若已显式 quantize，ternary 会覆盖（最后生效）
    # ------------------------------------------------------------------
    if "ternary" in config and config["ternary"] is not None:
        # ternary 与 quantize 互斥：ternary 优先
        ternary_only(new_model)
        qtype_applied = "ternary"
        has_quantize = True
        steps.append({
            "step": "ternary",
            "qtype": "ternary",
        })

    # ------------------------------------------------------------------
    # 5. distill（可选）：知识蒸馏，需要 teacher
    # ------------------------------------------------------------------
    if "distill" in config and config["distill"] is not None:
        d_cfg = dict(config["distill"])
        teacher = d_cfg.pop("teacher", None)
        if teacher is None:
            raise ValueError("distill 配置必须提供 'teacher' 字段")
        epochs = int(d_cfg.pop("epochs", 10))
        lr = float(d_cfg.pop("lr", 1e-4))
        T = float(d_cfg.pop("T", 2.0))
        alpha = float(d_cfg.pop("alpha", 0.5))
        # 构造一个简单的 toy loader（distill_only 需要 train_loader）
        # 若用户提供 train_loader，则优先使用
        train_loader = d_cfg.pop("train_loader", None)
        if train_loader is None:
            # 无 train_loader 时跳过实际训练，仅做 teacher 冻结
            # （下游可自行调用 KnowledgeDistiller.distill）
            KnowledgeDistiller(teacher, new_model, T=T, alpha=alpha)
            steps.append({
                "step": "distill",
                "epochs": 0,
                "lr": lr,
                "note": "no train_loader; teacher frozen, student ready",
            })
        else:
            max_steps = int(d_cfg.pop("max_steps", epochs))
            distill_only(
                teacher, new_model, train_loader,
                max_steps=max_steps, T=T, alpha=alpha, lr=lr,
            )
            steps.append({
                "step": "distill",
                "epochs": epochs,
                "lr": lr,
                "max_steps": max_steps,
            })

    # ------------------------------------------------------------------
    # 计算压缩后统计
    # ------------------------------------------------------------------
    compressed_bits = compute_compressed_bits(new_model)
    compressed_params = compressed_bits / 32
    compression_ratio = (original_bits / compressed_bits
                        if compressed_bits > 0 else float("inf"))
    # 平均 bit / param
    avg_bits = (compressed_bits / original_params
                if original_params > 0 else 32.0)

    stats = {
        "original_params": int(original_params),
        "compressed_params": float(compressed_params),
        "compressed_bits": int(compressed_bits),
        "original_bits": int(original_bits),
        "compression_ratio": float(compression_ratio),
        # sparsity 用 prune 后测得的实际稀疏度（若未 prune 则为 0）
        "sparsity": float(actual_sparsity),
        "bits": float(avg_bits),
        "qtype": qtype_applied if has_quantize else None,
        "steps": steps,
    }

    if return_stats:
        return new_model, stats
    return new_model


# ---------------------------------------------------------------------------
# Task 6 (V1.3): compress_pipeline v1.3 —— 以小博大
# ---------------------------------------------------------------------------


def _resolve_teacher(config: dict):
    """从 config 中解析 teacher_model（支持顶层便捷字段与 distill 子配置）。"""
    # 顶层便捷字段
    teacher = config.get("teacher_model", config.get("teacher"))
    train_loader = config.get("train_loader")
    # distill 子配置优先级更高
    d_cfg = config.get("distill")
    if isinstance(d_cfg, dict):
        teacher = d_cfg.get("teacher", teacher)
        train_loader = d_cfg.get("train_loader", train_loader)
    return teacher, train_loader, d_cfg


def _compress_pipeline_v13(model, config: dict, return_stats: bool = False):
    """V1.3 压缩流水线：prune → quantize → distill → lora（以小博大）。

    相对 v2 的变化：
    - **流程重排**：蒸馏在量化之后、LoRA 包装之前进行（先量化减存储、再蒸馏
      转移能力、最后 LoRA 包装为微调准备）。
    - **teacher_model 便捷入口**：``config`` 顶层可直接放 ``teacher_model`` /
      ``teacher`` / ``train_loader``。
    - **吞吐率优化**：量化后 QLinear 内部使用 fused matmul（``matmul_int4``）。
    - **压缩报告**：stats 内嵌 ``compression_report`` 字段。

    输出 stats 与 v2 完全兼容（同名同义键），并追加 V1.3 专属字段。
    """
    # 记录原始参数量与 bit 数（压缩前）
    original_params = count_parameters(model)
    original_bits = int(original_params * 32)

    # 深拷贝模型，确保不修改原模型
    new_model = _deep_copy_model(model)

    steps = []
    actual_sparsity = 0.0
    qtype_applied = None
    has_quantize = False

    # ------------------------------------------------------------------
    # 1. prune（可选）：结构化剪枝
    # ------------------------------------------------------------------
    if config.get("prune") is not None:
        prune_cfg = dict(config["prune"])
        sparsity = float(prune_cfg.pop("sparsity", 0.3))
        method = prune_cfg.pop("method", "outlier_safe")
        if method not in ("outlier_safe", None):
            raise ValueError(
                f"prune.method 仅支持 'outlier_safe'，得到 {method!r}"
            )
        pruner = OutlierSafePruner(new_model, sparsity=sparsity)
        _, prune_report = pruner.apply()
        nonzero_after_prune = count_nonzero_params(new_model)
        actual_sparsity = (1.0 - nonzero_after_prune / original_params
                           if original_params > 0 else 0.0)
        steps.append({
            "step": "prune",
            "sparsity": sparsity,
            "method": method,
            "report": prune_report,
            "actual_sparsity": float(actual_sparsity),
        })

    # ------------------------------------------------------------------
    # 2. quantize（可选）：INT4 / INT8，fused matmul 加速
    # ------------------------------------------------------------------
    if config.get("quantize") is not None:
        q_cfg = dict(config["quantize"])
        bits = int(q_cfg.pop("bits", 4))
        schema = q_cfg.pop("schema", "symmetric")
        if schema not in ("symmetric", None):
            raise ValueError(
                f"quantize.schema 仅支持 'symmetric'，得到 {schema!r}"
            )
        if bits == 4:
            qtype_applied = "int4"
        elif bits == 8:
            qtype_applied = "int8"
        else:
            raise ValueError(f"quantize.bits 仅支持 4 或 8，得到 {bits}")
        quantize_only(new_model, dtype=qtype_applied)
        has_quantize = True
        steps.append({
            "step": "quantize",
            "bits": bits,
            "qtype": qtype_applied,
            "schema": schema,
            "fused_matmul": True,  # V1.3：QuantizedLinear 内部走 fused 路径
        })

    # ------------------------------------------------------------------
    # 2b. ternary（可选）：覆盖 quantize（与 v2 行为一致，最后生效）
    # ------------------------------------------------------------------
    if config.get("ternary") is not None:
        ternary_only(new_model)
        qtype_applied = "ternary"
        has_quantize = True
        steps.append({
            "step": "ternary",
            "qtype": "ternary",
            "fused_matmul": True,
        })

    # ------------------------------------------------------------------
    # 3. distill（可选，V1.3 核心）：teacher → student 能力转移
    #    若提供 train_loader 则端到端蒸馏；否则仅冻结 teacher 做准备
    # ------------------------------------------------------------------
    teacher, train_loader, d_cfg = _resolve_teacher(config)
    if teacher is not None:
        d_cfg = dict(d_cfg) if isinstance(d_cfg, dict) else {}
        epochs = int(d_cfg.get("epochs", 3))
        lr = float(d_cfg.get("lr", 1e-3))
        T = float(d_cfg.get("T", d_cfg.get("temperature", 4.0)))
        alpha = float(d_cfg.get("alpha", 0.7))
        feature_loss_weight = float(d_cfg.get("feature_loss_weight", 0.3))
        distill_layers = d_cfg.get("distill_layers")
        max_steps = d_cfg.get("max_steps")
        feature_extractor = d_cfg.get("feature_extractor")
        distiller = KnowledgeDistiller(
            teacher, new_model, temperature=T, alpha=alpha,
            distill_layers=distill_layers,
            feature_loss_weight=feature_loss_weight,
        )
        if train_loader is not None:
            losses = distiller.distill(
                train_loader, epochs=epochs, lr=lr, max_steps=max_steps,
                feature_extractor=feature_extractor,
            )
            steps.append({
                "step": "distill",
                "epochs": epochs,
                "lr": lr,
                "max_steps": max_steps,
                "feature_level": feature_extractor is not None,
                "final_loss": float(losses[-1]) if losses else None,
                "loss_history_len": len(losses),
            })
        else:
            # 无 train_loader：teacher 已冻结，student 就绪
            steps.append({
                "step": "distill",
                "epochs": 0,
                "lr": lr,
                "note": "no train_loader; teacher frozen, student ready",
            })

    # ------------------------------------------------------------------
    # 4. lora（可选）：为微调准备（QLoRA 风格）
    # ------------------------------------------------------------------
    if config.get("lora") is not None:
        lora_cfg = dict(config["lora"])
        rank = int(lora_cfg.pop("rank", 8))
        alpha = float(lora_cfg.pop("alpha", 16.0))
        lora_only(new_model, r=rank, alpha=alpha)
        steps.append({
            "step": "lora",
            "rank": rank,
            "alpha": alpha,
        })

    # ------------------------------------------------------------------
    # 计算压缩后统计（与 v2 同构 + V1.3 压缩报告）
    # ------------------------------------------------------------------
    compressed_bits = compute_compressed_bits(new_model)
    compressed_params = compressed_bits / 32
    compression_ratio = (original_bits / compressed_bits
                        if compressed_bits > 0 else float("inf"))
    avg_bits = (compressed_bits / original_params
                if original_params > 0 else 32.0)
    nonzero_after = count_nonzero_params(new_model)
    sparsity_final = (1.0 - nonzero_after / original_params
                      if original_params > 0 else 0.0)

    # 估算吞吐率提升：INT4 ≈ 4×、INT8 ≈ 2×、ternary ≈ 8×（相对 fp32 的权重访存）
    throughput_factor = {None: 1.0, "int4": 4.0, "int8": 2.0, "ternary": 8.0}
    est_throughput = throughput_factor.get(qtype_applied, 1.0)

    report = {
        "original_params": int(original_params),
        "compressed_params": float(compressed_params),
        "compression_ratio": float(compression_ratio),
        "sparsity": float(sparsity_final),
        "bits_per_param": float(avg_bits),
        "estimated_throughput_improvement": float(est_throughput),
        "version": "1.3",
    }

    stats = {
        "original_params": int(original_params),
        "compressed_params": float(compressed_params),
        "compressed_bits": int(compressed_bits),
        "original_bits": int(original_bits),
        "compression_ratio": float(compression_ratio),
        "sparsity": float(actual_sparsity),
        "bits": float(avg_bits),
        "qtype": qtype_applied if has_quantize else None,
        "steps": steps,
        # V1.3 专属
        "version": "1.3",
        "estimated_throughput_improvement": float(est_throughput),
        "compression_report": report,
    }

    if return_stats:
        return new_model, stats
    return new_model


def compression_report(model, compressed_model) -> dict:
    """生成压缩报告（V1.3）。

    Args:
        model: 原始模型（Module）
        compressed_model: 压缩后模型（Module，可能含 QLinear / LoRALinear）

    Returns:
        报告 dict::

            {
              "original_params": int,
              "compressed_params": float,   # 等效 fp32 参数量
              "compression_ratio": float,   # original_bits / compressed_bits
              "sparsity": float,            # 1 - nonzero / total
              "bits_per_param": float,      # 平均 bit/param
              "estimated_throughput_improvement": float,
              "version": "1.3",
            }
    """
    orig_params = count_parameters(model)
    comp_params_count = count_parameters(compressed_model)
    orig_bits = int(orig_params * 32)
    comp_bits = compute_compressed_bits(compressed_model)
    ratio = (orig_bits / comp_bits) if comp_bits > 0 else float("inf")
    avg_bits = (comp_bits / orig_params) if orig_params > 0 else 32.0
    nonzero = count_nonzero_params(compressed_model)
    sparsity = (1.0 - nonzero / comp_params_count
                if comp_params_count > 0 else 0.0)
    # 从压缩模型推断 qtype（首个 QLinear）
    qtype = None
    for _, m in compressed_model.named_modules():
        ql = getattr(m, "_qlin", None)
        if ql is not None and hasattr(ql, "qtype"):
            qtype = ql.qtype
            break
        if isinstance(m, QLinear):
            qtype = m.qtype
            break
    throughput_factor = {None: 1.0, "int4": 4.0, "int8": 2.0, "ternary": 8.0}
    return {
        "original_params": int(orig_params),
        "compressed_params": float(comp_bits / 32),
        "compression_ratio": float(ratio),
        "sparsity": float(sparsity),
        "bits_per_param": float(avg_bits),
        "estimated_throughput_improvement": float(
            throughput_factor.get(qtype, 1.0)),
        "version": "1.3",
    }


# ---------------------------------------------------------------------------
# Part4 P10: MoD Expert 结构化剪枝（MoD-Aware 压缩）
# ---------------------------------------------------------------------------


def _iter_module_tensors(module):
    """递归生成 module 内所有 Tensor 参数（含 requires_grad=False 的）。"""
    for p in module._parameters.values():
        yield p
    for m in module._modules.values():
        yield from _iter_module_tensors(m)


def compress_mod_experts(model, keep_ratio: float = 0.5,
                         min_experts_per_part: int = 1,
                         return_stats: bool = False):
    """MoD Expert 结构化剪枝。

    对模型中所有 :class:`verse_nex.moe.MoDLayer` 实例：

    1. 收集每个 Expert 的参数 L2 范数（``sqrt(sum(p.data**2 for p in
       expert.parameters()))``）；
    2. 按 ``keep_ratio`` 在每个 ``DensePart`` 内保留范数最高的 Expert
       （保留 ``max(min_experts_per_part, int(num_experts * keep_ratio))`` 个）；
    3. 原地替换 ``DensePart.experts`` ModuleList（仅保留 kept Expert）；
    4. 修改 ``DensePart.router.gate`` 的权重矩阵（删除被裁 Expert 对应的行）；
    5. 修改 ``DensePart.router.num_routes`` 与 ``DensePart.top_k``，
       后者 ``top_k = min(top_k, remaining_experts)``。

    剪枝后 MoD 的前向路径（``MoDLayer.forward``）仍可正常工作：
    ``DensePart.router`` 的输出维度变为新 Expert 数，``_dispatch_and_combine``
    按 ``len(experts)`` 遍历，自动适配。

    Args:
        model: 含 :class:`MoDLayer` 的模型（任意 ``nn.Module``）
        keep_ratio: 保留比例（0.5 = 保留一半 Experts）
        min_experts_per_part: 每个 ``DensePart`` 最少保留 Expert 数（默认 1）
        return_stats: 是否返回压缩统计

    Returns:
        - ``return_stats=False``: 返回剪枝后的 ``model``（原地修改）
        - ``return_stats=True``: 返回 ``(model, stats)``，其中 stats::

              {
                "original_experts": int,   # 剪枝前 Expert 总数
                "kept_experts": int,       # 剪枝后 Expert 总数
                "compression_ratio": float,  # 1 - kept / original
              }
    """
    # 延迟导入 MoDLayer（避免顶层 import 时 verse_nex 不可用）
    try:
        from verse_nex.moe import MoDLayer
    except ImportError:  # pragma: no cover - 环境无 verse_nex 时无可剪枝对象
        MoDLayer = None

    total_before = 0
    total_after = 0

    if MoDLayer is not None:
        for m in model.modules():
            if not isinstance(m, MoDLayer):
                continue
            # 遍历每个 DensePart（MoDLayer.parts 是 ModuleList）
            for part in m.parts:
                n_experts = part.num_experts
                if n_experts <= 1:
                    # 已是最小，跳过（不剪）
                    total_before += n_experts
                    total_after += n_experts
                    continue

                # --- 1. 计算每个 Expert 的参数 L2 范数 ---
                # Expert 内部参数都在子 Linear 中（w_gate / w_up / w_down），
                # 因此用递归遍历 _parameters + _modules。
                norms = []
                for expert in part.experts:
                    sum_sq = 0.0
                    for p in _iter_module_tensors(expert):
                        sum_sq += float(np.sum(p.data ** 2))
                    norms.append(float(np.sqrt(sum_sq)))

                # --- 2. 计算保留数量并选取 kept 索引（范数最大的）---
                keep_n = max(int(min_experts_per_part),
                             int(n_experts * float(keep_ratio)))
                keep_n = min(keep_n, n_experts)  # 不能超过原数
                # argsort 降序取前 keep_n，再按原顺序排序（保证可复现性）
                sorted_idx = np.argsort(np.asarray(norms))[::-1][:keep_n]
                kept_indices = sorted(sorted_idx.tolist())

                # --- 3. 替换 experts ModuleList（仅保留 kept Expert） ---
                new_experts = nn.ModuleList(
                    [part.experts[i] for i in kept_indices]
                )
                setattr(part, "experts", new_experts)

                # --- 4. 修改 expert router 的 gate 权重矩阵行 ---
                # Router.gate 是 nn.Linear，weight shape (num_routes, dim)
                gate = part.router.gate
                kept_arr = np.asarray(kept_indices, dtype=np.int64)
                new_w = gate.weight.data[kept_arr]
                gate.weight.data = new_w
                gate.out_features = int(len(kept_indices))

                # --- 5. 更新 router 元数据 + top_k 调整 ---
                new_num_routes = int(len(kept_indices))
                part.router.num_routes = new_num_routes
                new_top_k = min(int(part.router.top_k), new_num_routes)
                if new_top_k < 1:
                    new_top_k = 1
                part.router.top_k = new_top_k

                # --- 6. 同步 DensePart 元数据 ---
                part.num_experts = new_num_routes
                part.top_k = new_top_k

                total_before += n_experts
                total_after += new_num_routes

    if return_stats:
        ratio = (1.0 - total_after / total_before
                 if total_before > 0 else 0.0)
        stats = {
            "original_experts": int(total_before),
            "kept_experts": int(total_after),
            "compression_ratio": float(ratio),
        }
        return model, stats
    return model


__all__ = [
    # 类
    "OutlierSafePruner",
    "LoRALinear",
    "KnowledgeDistiller",
    "QLinear",
    # 函数
    "compress_pipeline",
    "compress_mod_experts",
    "compression_report",
    "prune_only",
    "quantize_only",
    "lora_only",
    "ternary_only",
    "distill_only",
    "count_parameters",
    "count_nonzero_params",
    "compute_compressed_bits",
]