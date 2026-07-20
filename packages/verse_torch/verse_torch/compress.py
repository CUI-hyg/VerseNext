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
from .losses import cross_entropy, kl_div_loss
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
    """知识蒸馏器：teacher (frozen) + student (trainable)。

    Loss = alpha * T^2 * KL(softmax(teacher/T) || log_softmax(student/T))
           + (1 - alpha) * CE(student, hard_targets)

    Args:
        teacher: frozen 教师模型（自动 eval + requires_grad=False）
        student: 可训练学生模型
        T: 温度（默认 2.0），soft target 平滑度
        alpha: soft loss 权重（默认 0.5），(1-alpha) 为 hard loss 权重
    """

    def __init__(self, teacher, student, T: float = 2.0, alpha: float = 0.5):
        self.teacher = teacher
        self.student = student
        self.T = float(T)
        self.alpha = float(alpha)
        # 冻结 teacher：eval 模式 + 所有参数 requires_grad=False
        self.teacher.eval()
        for p in _iter_all_params_static(teacher):
            p.requires_grad = False

    def forward(self, student_logits: Tensor, teacher_logits: Tensor,
                hard_targets) -> Tensor:
        T = self.T
        # soft loss: KL(softmax(teacher/T) || log_softmax(student/T)) * T^2
        # kl_div_loss(log_probs, target_probs) = sum(target * (log(target) - log_probs)).mean()
        # teacher_logits.detach() 切断梯度，不回传到 teacher
        teacher_probs = (teacher_logits.detach() / T).softmax(dim=-1)
        student_log_probs = (student_logits / T).log_softmax(dim=-1)
        soft_loss = kl_div_loss(student_log_probs, teacher_probs) * (T * T)
        # hard loss: CE(student, hard_targets)
        hard_loss = cross_entropy(student_logits, hard_targets)
        # 联合损失
        total = self.alpha * soft_loss + (1.0 - self.alpha) * hard_loss
        return total

    def __call__(self, student_logits, teacher_logits, hard_targets):
        return self.forward(student_logits, teacher_logits, hard_targets)

    def distill(self, train_loader, optimizer, max_steps: int = 100,
                eval_fn=None, eval_every: int = 0):
        """蒸馏训练循环（可选）。

        Args:
            train_loader: 可迭代对象，每次返回 (x, y)
            optimizer: 优化器（如 AdamW）
            max_steps: 最大训练步数
            eval_fn: 可选回调 (step, student) -> None
            eval_every: 每隔多少步调用 eval_fn

        Returns:
            训练损失历史 list
        """
        losses_hist = []
        step = 0
        self.student.train()
        while step < max_steps:
            for batch in train_loader:
                if step >= max_steps:
                    break
                x, y = batch
                optimizer.zero_grad()
                # teacher forward (no_grad)
                with no_grad():
                    teacher_logits = self.teacher(x)
                # student forward
                student_logits = self.student(x)
                loss = self.forward(student_logits, teacher_logits, y)
                loss.backward()
                optimizer.step()
                losses_hist.append(float(loss.data))
                step += 1
                if eval_fn is not None and eval_every > 0 and step % eval_every == 0:
                    eval_fn(step, self.student)
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


def compress_pipeline(model, target_ratio: float = 0.1, eval_fn=None,
                      sparsity: float = 0.3, qtype: str = "int4",
                      lora_r: int = 8, lora_alpha: float = 16.0,
                      use_lora: bool = False):
    """端到端压缩 pipeline：prune → quantize → (可选) lora_wrap。

    默认 ``use_lora=False``（PoC 简化：LoRA 是训练阶段技术，纯推理 pipeline 不需要）。
    若需 QLoRA 风格（量化基座 + LoRA 增量微调），设 ``use_lora=True``。

    压缩比计算（bit-level 精确版）：
        compression_ratio = original_bits / compressed_bits
        - fp32: 32 bit/param
        - INT4: 4 bit/param（packed 后实际字节 * 8）
        - INT8: 8 bit/param
        - ternary: 2 bit/param

    Args:
        model: 待压缩模型（nn.Module）
        target_ratio: 目标压缩比（默认 0.1，即压缩到 1/10 大小）
        eval_fn: 可选，接收 model 返回 loss 的函数
        sparsity: 剪枝稀疏度（默认 0.3）
        qtype: 量化类型（默认 "int4"；为达到 10× 压缩比可改 "ternary"）
        lora_r: LoRA 秩（默认 8）
        lora_alpha: LoRA 缩放（默认 16）
        use_lora: 是否在最后挂 LoRA 适配器（默认 False）

    Returns:
        dict: {original_params, compressed_params, compression_ratio,
               original_loss, compressed_loss, loss_diff_pct, steps}
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


__all__ = [
    # 类
    "OutlierSafePruner",
    "LoRALinear",
    "KnowledgeDistiller",
    "QLinear",
    # 函数
    "compress_pipeline",
    "prune_only",
    "quantize_only",
    "lora_only",
    "ternary_only",
    "distill_only",
    "count_parameters",
    "count_nonzero_params",
    "compute_compressed_bits",
]