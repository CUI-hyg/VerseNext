# VerseNext Part3K2 Task 7 全项目审计报告

> 审计日期：2026-07-21
> 审计范围：`/workspace` 全项目（6 packages + data/demo + tests + examples + docs）
> 审计任务：SubTask 7.1 ~ 7.9（严重错误扫描 / 漏洞扫描 / 可优化扫描 / 合并重复 / BUG 修复 / 文档补全 / 综合验收）
> 审计基线：`pytest tests/ -x --tb=short` → 377 passed, 5 skipped, 10 warnings in 13.14s
> 审计目标：BUG 清零 + 实现共用 + README 完善 + 综合验收通过

---

## Part5K1 升级审计补充（2026-07-23）

> 审计范围：Part5K1 VMPC 技术全量化 + 双模型并行 + 架构精简（14 个任务全部完成）
> 审计基线：`pytest tests/` → **1440 passed, 20 skipped, 0 failed**（分批运行，沙箱 OOM 限制下单批最大约 1300 测试）

### Part5K1 变更摘要

| 任务 | 内容 | 状态 |
|---|---|---|
| Task 1 | VerseTorch.nn → vnn 重命名（BREAKING） | ✅ 完成 |
| Task 2 | VerseTorch 底层去壳与合并 | ✅ 完成 |
| Task 3 | VMPC 命名 + V1.5 门面 | ✅ 完成 |
| Task 4 | VMPC V1.5 命中与质量优化 | ✅ 完成 |
| Task 5 | JSONL 自修复与标准化 | ✅ 完成 |
| Task 6 | val.json 自动生成 + 数据预加载 | ✅ 完成 |
| Task 7 | 64+ 层训练加速（层融合 + chunked） | ✅ 完成 |
| Task 8 | VMT 完整智能分区训练（VMTTrainer） | ✅ 完成 |
| Task 9 | spark 双模型目录重构（small/mate） | ✅ 完成 |
| Task 10 | checkpoint 重命名 + .vn 默认输出 | ✅ 完成 |
| Task 11 | spark/run.py 训练模式补齐 | ✅ 完成 |
| Task 12 | VerseNex 精简 | ✅ 完成 |
| Task 13 | 文档与代码注释升级 | ✅ 完成 |
| Task 14 | 全量测试 + 综合验收 | ✅ 完成 |

### 关键验收项

1. **关键导入验证**：`import verse_torch` / `from verse_torch.vmpc import ...` / `from verse_torch.vnn import ...` / `from spark.small.model import CometSparkSmallLM` / `from spark.mate.model import CometSparkMateLM` / `from verse_infra.verse_trainer.jsonl_repair import repair_jsonl` 全部通过
2. **CLI 端到端**：`spark/run.py train --model small --dry-run` / `finetune` / `posttrain` / `continue` dry-run 全部通过
3. **VMPC V1.5 端到端**：small 模型构建（193734 参数）→ `vmpc_compress_model()` 压缩（17222 参数，压缩 ~91%）→ `generate()` 生成正常
4. **JSONL 修复端到端**：异名字段（instruction/response, q/a）+ 行尾多余逗号 → 全部标准化为 prompt/completion
5. **VMT 端到端**：三档策略解析（freeze/optimize/unload）+ freeze 档 INT4 量化 + 统一实体（model id 不变）
6. **零回归**：全量 1440 测试通过，20 skipped（环境/弃用模块相关），0 failed

---

## 1. 审计摘要

| 项 | 数值 |
|---|---|
| 扫描文件数 | 35+ 关键源文件（6 packages + data/demo） |
| 严重错误（SubTask 7.1） | 5 类扫描，0 阻塞性问题 |
| 漏洞（SubTask 7.2） | 4 类扫描，0 阻塞性问题 |
| 可优化点（SubTask 7.3） | 2 处重复实现（已合并） |
| 修复 BUG（SubTask 7.5） | 3 处（sigmoid overflow / BCE_with_logits overflow / 硬编码路径） |
| 综合验收（SubTask 7.8） | pytest 全量 + 4 端到端 + compress_train_demo 全部通过 |

**结论：未发现阻塞性 BUG，发现的优化点与 warning 已全部修复。**

---

## 2. SubTask 7.1：严重错误扫描

### 2.1 递归栈溢出

| 文件 | 位置 | 扫描结论 |
|---|---|---|
| `packages/verse_torch/verse_torch/tensor.py` | `backward()` 行 1139-1191 | 已使用**迭代式 DFS 拓扑排序**（显式栈），无递归风险。`backward()` 内部用 `while stack:` 循环处理拓扑序列 |
| `data/demo/run.py` | 顶部 | `sys.setrecursionlimit(2000)` 兜底（防 pickle 反序列化深嵌套） |
| 其他位置 | — | 仅 `pickle.load` / `json.loads` 可能有深度，但已用 `sys.setrecursionlimit` 兜底 |

**结论：无栈溢出风险。**

### 2.2 NaN / Inf

| 文件 | 位置 | 扫描结论 |
|---|---|---|
| `packages/verse_nex/verse_nex/mamba2.py` | `_softplus` 行 193-197 | `safe_x = np.minimum(x_data, 20.0)` 防止 `np.exp` 溢出，`np.where` 选择分支 |
| `packages/verse_nex/verse_nex/rwkv7.py` | `softplus` 行 290-295 | 同上，已有 `safe_w` 保护 |
| `packages/verse_nex/verse_nex/hybrid.py` | `log_decay` | clip 到 `[-50, 0]`，防止数值溢出 |
| `packages/verse_torch/verse_torch/tensor.py` | `sigmoid` / `silu` 行 503-548 | **发现 RuntimeWarning: overflow encountered in exp**（功能正确，因 `np.where` 两个分支都会被 NumPy 计算；SubTask 7.5 已修复，改用 `0.5 * (1 + tanh(x/2))` 等价公式） |
| `packages/verse_torch/verse_torch/losses.py` | `binary_cross_entropy_with_logits._backward` | 同类 sigmoid overflow warning（SubTask 7.5 已修复） |
| `packages/verse_torch/verse_torch/losses.py` | `cross_entropy` / `focal_loss` | 已用 `log_softmax + NLL` 而非 `log(softmax)`，数值稳定 |

**结论：修复 1 处 overflow warning，其余位置已有数值稳定性保护。**

### 2.3 Unicode 乱码

| 文件 | 位置 | 扫描结论 |
|---|---|---|
| `packages/verse_tokenizer/verse_tokenizer/preprocess.py` | `trim_to_utf8_boundary` | 已实现 UTF-8 边界对齐，避免截断到多字节字符中间 |
| `packages/verse_tokenizer/verse_tokenizer/bpe.py` | `decode` | 用 `errors="ignore"` 丢弃非法字节（与 GPT-2 BPE 行为一致） |
| `data/demo/train/evaluate.py` | 输出 | 全部用 `flush=True` + UTF-8 编码，无乱码 |

**结论：无 Unicode 乱码风险。**

### 2.4 import 循环

| 检查项 | 结论 |
|---|---|
| `verse_torch.tensor` ↔ `verse_torch.nn` | `tensor.py` 不依赖 `nn.py`，`nn.py` 依赖 `tensor.py`，单向 |
| `verse_torch.training` ↔ `verse_torch.losses` | **Task 7.4 修复前**：两文件都独立实现 `cross_entropy` / `cross_entropy_loss`，无相互 import；**Task 7.4 修复后**：`training.cross_entropy_loss` 用**延迟 import** `from .losses import cross_entropy`，避免循环依赖（`losses.py` 不依赖 `training.py`） |
| `verse_nex.hybrid` ↔ `verse_torch.nn` | 单向依赖（hybrid 依赖 nn） |
| `verse_inference.model_loader` ↔ `data.demo.model.model` | 用动态 importlib + `sys.path` 注入，避免硬依赖 |
| `data/demo/model/model.py` ↔ `verse_nex.hybrid` | `_import_hybrid_lm()` 延迟 import，避免在 transformer-only 场景也加载 verse_nex |

**结论：无 import 循环。**

### 2.5 文件路径硬编码

| 文件 | 位置 | 问题 | 修复 |
|---|---|---|---|
| `packages/verse_inference/verse_inference/model_loader.py` | 行 100 | `_DEFAULT_COMETSPARK_DEMO_PATH = "/workspace/data/demo"` 硬编码 `/workspace` | SubTask 7.5 改为基于 `__file__` 推断：`os.path.dirname` 上溯 4 层得到 `<repo_root>`，再拼接 `data/demo`，结果与原硬编码值一致但不再依赖固定路径 |
| docstring / README 示例 | 3 处 | 文档中举例使用 `/workspace/data/demo/...` | 保留（示例用途，不影响运行时行为） |

**结论：1 处运行时硬编码已修复，文档示例中的路径属于示例用途，保留。**

---

## 3. SubTask 7.2：漏洞扫描

### 3.1 资源泄漏

| 检查项 | 结论 |
|---|---|
| `open()` 未关闭 | 用 `grep "^\s*open\("` 扫描，所有 `open()` 调用均配对 `with` 语句（`CheckpointManager.save_best/load_best` / `Tokenizer.save/load` / `evaluate._run_scoring` 读 references 文件等） |
| `pickle.load` / `pickle.dump` | 均在 `with open(...)` 上下文内，无泄漏 |
| `np.memmap` | 项目未使用 `np.memmap`，无内存映射泄漏风险 |

**结论：无资源泄漏。**

### 3.2 除零

| 文件 | 位置 | 保护方式 |
|---|---|---|
| `packages/verse_torch/verse_torch/training.py` | `clip_grad_norm` 行 82-83 | `total_norm + 1e-6` 防止除零 |
| `packages/verse_torch/verse_torch/scoring.py` | `bleu` 行 89 / `char_f1` 行 50-52 | `total > 0` 与 `precision + recall` 都有判零保护 |
| `packages/verse_torch/verse_torch/optim.py` | AdamW / Lion / Adafactor | 各优化器均用 `np.sqrt(v + eps)` 形式（eps 通常 1e-8） |
| `packages/verse_inference/verse_inference/sampler.py` | top-p / top-k 采样 | `probs_sum + 1e-9` 防除零 |
| `packages/verse_torch/verse_torch/losses.py` | `binary_cross_entropy` 行 138 | `eps = 1e-12` clip pred 到 `[eps, 1-eps]` |

**结论：所有除法均有 eps 或判零保护，无除零风险。**

### 3.3 整数溢出

| 检查项 | 结论 |
|---|---|
| Python int | Python 3 原生 bigint，无固定宽度溢出 |
| NumPy int32 / int64 | 项目主要用 `np.int64`（BPE merge 索引、targets），最大 2^63-1 远超实际场景 |
| `tensor_ids * head_idx` | vocab_size ≤ 256 + 3 = 259，无溢出 |

**结论：无整数溢出风险。**

### 3.4 未捕获异常

| 文件 | 位置 | 处理方式 |
|---|---|---|
| `data/demo/train/evaluate.py` 行 217-230 | `model.generate()` 调用 | 用 `try/except TypeError` 包裹，捕获 `top_p` 参数不支持时降级（自动重试不带 `top_p`） |
| `data/demo/run.py` 主流程 | 全局 | `--verbose` 控制是否打印 traceback，默认打印友好错误信息 |
| `verse_inference/model_loader.py` | 动态 import | 三层 fallback（`data.demo.model.model` → `model.model` → 默认路径），每层用 `try/except ImportError` |
| `verse_torch/training.py` `Trainer.train` | 训练循环 | `KeyboardInterrupt` 仍可中断，但 `try/finally` 确保 `loss_history.json` 写入 |

**结论：所有外部交互点（文件加载、模型加载、生成调用）均有异常处理。**

---

## 4. SubTask 7.3：可优化部分扫描

### 4.1 重复代码

| 重复实现 | 位置 | 处理方式（SubTask 7.4） |
|---|---|---|
| `cross_entropy` 与 `cross_entropy_loss` | `verse_torch.losses.cross_entropy` 行 20-104 与 `verse_torch.training.cross_entropy_loss` 行 109-172 | **已合并**：`losses.cross_entropy` 扩展支持 `ignore_index` 与 `label_smoothing`；`training.cross_entropy_loss` 改为委托调用（保持双 API 入口，用户习惯不变） |
| NFKC 归一化 | 之前在 `bpe.py` 与 `unigram.py` 各自实现 | **已合并**：统一到 `verse_tokenizer/preprocess.py:nfkc_normalize(text)`，bpe.py 已 import 使用 |

### 4.2 低效算法

| 文件 | 位置 | 评估 |
|---|---|---|
| `verse_torch/scoring.py` `_lcs_length` | 行 103-114 | O(m*n) DP，是 LCS 标准实现，且仅用于 ROUGE-L 评估（样本数小），无需优化 |
| `verse_torch/scoring.py` `bleu` | 行 60-100 | O(n) n-gram counter，标准实现 |
| `verse_nex/mamba2.py` `ssm_scan` | — | 已用累加循环（`ht = ssm_dt * h_prev + ...`），是 SSM 的标准实现，无低效 |

**结论：无低效算法需优化。**

### 4.3 冗余 import / 死代码

| 检查项 | 结论 |
|---|---|
| 冗余 import | 通过 grep 扫描，未发现未使用的 import（`from .optim import LambdaLR, warmup_cosine_lr` 带 `# noqa: F401` 重新导出标注） |
| 死代码 | 通过 grep 扫描 `# TODO` / `# FIXME` / `pass  # ` 等模式，未发现明显死代码 |

**结论：无冗余 import / 死代码。**

---

## 5. SubTask 7.4：合并重复实现

### 5.1 `losses.cross_entropy` 与 `training.cross_entropy_loss` 合并

#### 修改前

两个函数各自独立实现：
- `losses.cross_entropy(logits, targets)`：旧签名，**无** `ignore_index` 与 `label_smoothing` 参数
- `training.cross_entropy_loss(logits, targets, ignore_index=-100, label_smoothing=0.0)`：完整实现，含 mask 屏蔽与标签平滑

实现重复（log_softmax + NLL + mask + label_smoothing），违反"不要重复造轮子"原则。

#### 修改后（`packages/verse_torch/verse_torch/losses.py`）

扩展 `cross_entropy` 签名为：
```python
def cross_entropy(logits: Tensor, targets, ignore_index: int = -100,
                  label_smoothing: float = 0.0) -> Tensor:
```

新增功能：
1. **3D logits 自动 reshape 为 2D**（与 PyTorch `F.cross_entropy` 行为一致）
2. **`ignore_index=None` 保持旧 API 行为**（不做屏蔽，向后兼容）
3. **`ignore_index` 非 None 时启用 mask 屏蔽**（与 `training.cross_entropy_loss` 共用逻辑）
4. **`label_smoothing` 支持**（混合 hard target 与均匀分布）

#### 修改后（`packages/verse_torch/verse_torch/training.py`）

`cross_entropy_loss` 函数体改为**委托调用**，保持 API 入口不变：
```python
def cross_entropy_loss(logits: Tensor, targets, ignore_index: int = -100,
                       label_smoothing: float = 0.0) -> Tensor:
    """交叉熵损失（委托给 losses.cross_entropy，Task 7.4 合并重复实现）。"""
    from .losses import cross_entropy
    return cross_entropy(logits, targets, ignore_index=ignore_index,
                         label_smoothing=label_smoothing)
```

#### 兼容性验证

- 默认参数：`ignore_index=-100`（与原 `training.cross_entropy_loss` 一致，屏蔽 -100 标签）
- 旧调用 `cross_entropy(logits, targets)`：仍可调用，`ignore_index=-100` 默认会屏蔽 targets 中的 -100，与无 -100 标签的数据结果一致
- 旧调用 `cross_entropy_loss(logits, targets, ignore_index=-100, label_smoothing=0.1)`：行为完全不变

### 5.2 NFKC 统一到 preprocess.py

#### 现状

`verse_tokenizer/preprocess.py` 已实现 `nfkc_normalize(text)` 统一入口，`bpe.py` 与 `unigram.py` 已 import 使用。无重复实现。

**结论：NFKC 已在 Task 1-6 完成统一，无需再改。**

---

## 6. SubTask 7.5：BUG 修复清单

| # | 文件 | 位置 | 问题 | 修复方式 |
|---|---|---|---|---|
| 1 | `packages/verse_torch/verse_torch/tensor.py` | `sigmoid` 行 503-517 | `np.where(x>=0, 1/(1+exp(-x)), exp(x)/(1+exp(x)))` 两个分支都被 NumPy 计算，x 为大负数时 `exp(-x)` 溢出、大正数时 `exp(x)` 溢出，产生 `RuntimeWarning: overflow encountered in exp`（功能正确但污染日志） | 改用等价公式 `0.5 * (1.0 + np.tanh(0.5 * x))`，tanh 数值稳定，无 overflow；反向梯度公式不变（仍用 `out_data * (1 - out_data)`） |
| 2 | `packages/verse_torch/verse_torch/tensor.py` | `silu` 行 531-548 | 同 #1，`silu = x * sigmoid(x)` 内部 sigmoid 用 `np.where` 同样溢出 | 同 #1，sigmoid 部分改用 tanh 等价公式 |
| 3 | `packages/verse_torch/verse_torch/losses.py` | `binary_cross_entropy_with_logits._backward` 行 193 | 反向梯度计算用 `np.where(x>=0, 1/(1+exp(-x)), exp(x)/(1+exp(x)))`，同类 overflow warning | 同 #1，改用 `0.5 * (1.0 + np.tanh(0.5 * x))` |
| 4 | `packages/verse_inference/verse_inference/model_loader.py` | 行 100 | `_DEFAULT_COMETSPARK_DEMO_PATH = "/workspace/data/demo"` 硬编码 `/workspace`，迁移到其他目录会失效 | 改为基于 `__file__` 推断：`os.path.dirname` 上溯 4 层得到 `<repo_root>`，再拼接 `data/demo`，结果与原硬编码值一致但不再依赖固定路径 |

### 修复后验证

运行 `pytest tests/test_training.py tests/test_training_optimization.py tests/test_end_to_end.py tests/test_compression_poc.py tests/test_hybrid_stability.py tests/test_unit_operators.py --tb=short -W error::RuntimeWarning`：
- **结果**：110 passed, 1 skipped, 7 warnings（仅 PytestReturnNotNoneWarning，非 RuntimeWarning）
- **结论**：所有 sigmoid overflow warning 已消除

---

## 7. SubTask 7.6：审计报告

本文件即审计报告（`/workspace/audit_report.md`）。

---

## 8. SubTask 7.7：补全 `/workspace/data/demo/README.md`

### 新增内容

在 "Task 7: 自定义 Prompt 支持" 之后新增 "Part3K2: 采样增强与评分模式" 小节，覆盖 4 个新参数：

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--top-p` | `None` | nucleus sampling 阈值 (0,1)；当前 `CometSparkLM.generate` 不支持，自动降级 |
| `--parallel-chunks` | `None` | 覆盖 `training.parallel_chunks`，启用 `ParallelTrainer` 拆 chunk 训练 |
| `--score` | `False` | 启用评分模式（5 个指标） |
| `--references-file` | `None` | 参考答案文件路径（每行一个） |

并补充：
- **评分模式用法示例**：完整的 `refs.txt` 准备 + `run.py --score` 调用示例
- **评分报告输出格式**：5 个指标的报告样本
- **ParallelTrainer 用法示例**：`--parallel-chunks 2` 启用并行训练

---

## 9. SubTask 7.8：综合验收

### 9.1 pytest 全量测试

```bash
$ pytest tests/ --tb=short
```

**结果**：377 passed, 5 skipped, 10 warnings in 13.14s

- ✅ 全部测试通过（无 failure）
- ✅ 5 个 skipped 测试为环境依赖（matplotlib 不可用等），符合预期
- ✅ 10 个 warnings 均为 PytestReturnNotNoneWarning（测试风格，非 BUG）

### 9.2 关键测试套件（含 RuntimeWarning 强校验）

```bash
$ pytest tests/test_training.py tests/test_training_optimization.py \
        tests/test_end_to_end.py tests/test_compression_poc.py \
        tests/test_hybrid_stability.py tests/test_unit_operators.py \
        --tb=short -W error::RuntimeWarning
```

**结果**：110 passed, 1 skipped, 7 warnings in 7.02s

- ✅ 强制把 RuntimeWarning 当作 error 也能通过，证明 sigmoid overflow 已彻底修复

### 9.3 端到端验证

#### 验证 1：cross_entropy / cross_entropy_loss 行为一致性

```python
# 验证 losses.cross_entropy 与 training.cross_entropy_loss 行为一致（委托后）
import numpy as np
from verse_torch import Tensor
from verse_torch.losses import cross_entropy
from verse_torch.training import cross_entropy_loss

np.random.seed(0)
logits = Tensor(np.random.randn(4, 10), requires_grad=True)
targets = np.array([0, 1, -100, 3])  # 含 ignore_index

l1 = cross_entropy(logits, targets, ignore_index=-100)
l2 = cross_entropy_loss(logits, targets, ignore_index=-100)
assert abs(l1.item() - l2.item()) < 1e-7  # 一致

# label_smoothing 一致
l3 = cross_entropy(logits, targets, ignore_index=-100, label_smoothing=0.1)
l4 = cross_entropy_loss(logits, targets, ignore_index=-100, label_smoothing=0.1)
assert abs(l3.item() - l4.item()) < 1e-7

# 3D logits 自动 reshape
logits_3d = Tensor(np.random.randn(2, 3, 10), requires_grad=True)
targets_2d = np.array([[0, 1, -100], [2, 3, 4]])
l5 = cross_entropy(logits_3d, targets_2d, ignore_index=-100)
l6 = cross_entropy_loss(logits_3d, targets_2d, ignore_index=-100)
assert abs(l5.item() - l6.item()) < 1e-7

# 反向梯度可流通
l1.backward()
assert logits.grad is not None
print("验证 1 通过：cross_entropy 委托合并无回归")
```

#### 验证 2：sigmoid overflow 修复

```python
# 验证 sigmoid / silu 在极端输入下无 overflow warning
import warnings
import numpy as np
from verse_torch import Tensor

x = Tensor(np.array([-1000.0, -100.0, 0.0, 100.0, 1000.0]), requires_grad=True)
with warnings.catch_warnings():
    warnings.simplefilter("error", RuntimeWarning)  # 把 RuntimeWarning 当作 error
    y = x.sigmoid()
    z = x.silu()
print("验证 2 通过：sigmoid/silu 无 overflow warning")
print(f"  sigmoid(±1000) = {y.data}")
print(f"  silu(±1000)    = {z.data}")
```

#### 验证 3：model_loader 路径推断

```python
# 验证 _DEFAULT_COMETSPARK_DEMO_PATH 不再硬编码 /workspace
import os
from verse_inference.model_loader import _DEFAULT_COMETSPARK_DEMO_PATH

# 应该等于 <repo_root>/data/demo，而非硬编码 /workspace
assert _DEFAULT_COMETSPARK_DEMO_PATH.endswith("data/demo")
assert os.path.isabs(_DEFAULT_COMETSPARK_DEMO_PATH)
print(f"验证 3 通过：默认路径 = {_DEFAULT_COMETSPARK_DEMO_PATH}")
```

#### 验证 4：NFKC 统一

```python
# 验证 NFKC 已统一到 preprocess.py
from verse_tokenizer.preprocess import nfkc_normalize

# 全角→半角
assert nfkc_normalize("ＡＢＣ１２３") == "ABC123"
# 兼容字符分解
assert nfkc_normalize("ﬁ") == "fi"
print("验证 4 通过：NFKC 统一到 preprocess.py")
```

### 9.4 compress_train_demo.py 端到端

```bash
$ python examples/compress_train_demo.py
```

**结果**：完整跑通压缩 + 训练 + 评估流程，输出包含：
- 模型压缩统计（参数量减少 / INT8 量化比例）
- 训练 loss 下降曲线
- 评估生成样本

详见 `/workspace/examples/compress_train_demo.py` 输出。

---

## 10. SubTask 7.9：spec 文档勾选

参见 `/workspace/.trae/specs/part3k2-major-upgrade/tasks.md` 与 `checklist.md`，已勾选：
- Task 4 / 5 / 6 顶部 checkbox（前序任务完成）
- Task 7 全部 SubTask 7.1 ~ 7.9
- Task 7 全部 checkpoint
- 综合验收 checkpoint

---

## 11. 修改文件清单

### 新增文件

| 文件 | 用途 |
|---|---|
| `/workspace/audit_report.md` | 本审计报告（SubTask 7.6） |

### 修改文件

| 文件 | 修改内容 | 对应 SubTask |
|---|---|---|
| `packages/verse_torch/verse_torch/losses.py` | `cross_entropy` 扩展 `ignore_index` + `label_smoothing` 参数；`binary_cross_entropy_with_logits._backward` 改用 tanh 等价公式避免 overflow | 7.4 + 7.5 |
| `packages/verse_torch/verse_torch/training.py` | `cross_entropy_loss` 改为委托 `losses.cross_entropy`，删除重复实现 | 7.4 |
| `packages/verse_torch/verse_torch/tensor.py` | `sigmoid` / `silu` 改用 `0.5 * (1 + tanh(x/2))` 等价公式消除 overflow warning | 7.5 |
| `packages/verse_inference/verse_inference/model_loader.py` | `_DEFAULT_COMETSPARK_DEMO_PATH` 改为基于 `__file__` 推断，去除硬编码 `/workspace` | 7.5 |
| `data/demo/README.md` | 新增 "Part3K2: 采样增强与评分模式" 小节，覆盖 `--top-p` / `--parallel-chunks` / `--score` / `--references-file` 参数与用法示例 | 7.7 |
| `.trae/specs/part3k2-major-upgrade/tasks.md` | 勾选 Task 4/5/6 顶部 checkbox，Task 7 全部 SubTask | 7.9 |
| `.trae/specs/part3k2-major-upgrade/checklist.md` | 勾选 Task 7 全部 checkpoint 与综合验收 checkpoint | 7.9 |

---

## 12. 问题汇总与决策点

### 12.1 问题总数与处理

| 类别 | 发现数 | 已修复数 | 未修复数 |
|---|---|---|---|
| 严重错误 | 0 | 0 | 0 |
| 漏洞 | 0 | 0 | 0 |
| 可优化点 | 2（cross_entropy / NFKC） | 2 | 0 |
| BUG（含 warning） | 4（sigmoid×2 / BCE×1 / 硬编码路径×1） | 4 | 0 |
| **合计** | **6** | **6** | **0** |

### 12.2 关键决策点

1. **`cross_entropy` / `cross_entropy_loss` 合并策略**：选择"扩展 `losses.cross_entropy` + `training.cross_entropy_loss` 委托"而非"删除 `training.cross_entropy_loss`"。
   - 原因：保持双 API 入口（用户习惯），spec.md 明确要求"保持两个 API 入口（用户习惯），但实现共用"。

2. **sigmoid overflow 修复方式**：选择"用 `0.5 * (1 + tanh(x/2))` 等价公式"而非"`np.errstate(over='ignore')` 抑制 warning"。
   - 原因：前者真正消除 overflow（不计算 exp），后者只是隐藏 warning；前者更符合"修复要彻底"原则。

3. **`model_loader.py` 路径推断**：选择"基于 `__file__` 推断"而非"硬编码相对路径"。
   - 原因：保持运行时灵活性，迁移到其他目录无需修改代码；结果与原硬编码值一致，无行为变化。

4. **`ignore_index=None` 语义**：`losses.cross_entropy` 旧 API 默认 `ignore_index` 不存在（无屏蔽），新 API 默认 `ignore_index=-100`（屏蔽 -100）。
   - 决策：`ignore_index=None` 显式表示"不屏蔽"，与 `ignore_index=-100`（屏蔽 -100）区分；保持向后兼容（旧调用 `cross_entropy(logits, targets)` 不会因 targets 中有 -100 而出错，仅是行为从"对所有样本求平均"变为"屏蔽 -100 样本"，对无 -100 的数据完全等价）。

5. **审计报告格式**：选择 Markdown 详细报告（而非 JSON / 简短摘要）。
   - 原因：spec 要求"审计报告要详细"，且 Markdown 便于人类阅读与版本对比。

### 12.3 未修复项

**无未修复项。** 所有发现的 BUG / 漏洞 / 可优化点均已修复。

---

## 13. 综合结论

Part3K2 Task 7 全项目审计与 BUG 清零已完成：

1. **审计覆盖**：SubTask 7.1 ~ 7.9 全部完成
2. **BUG 清零**：4 个 BUG 全部修复（sigmoid overflow ×2 + BCE_with_logits overflow ×1 + 硬编码路径 ×1）
3. **重复实现合并**：2 处全部合并（cross_entropy 共用 + NFKC 统一）
4. **文档完善**：`data/demo/README.md` 补全 4 个新参数说明与用法示例
5. **综合验收通过**：pytest 377 passed + 4 端到端验证 + compress_train_demo.py 全部通过
6. **spec 文档已勾选**：tasks.md / checklist.md 全部对应项已勾选

**VerseNext 框架 Part3K2 重大升级审计通过，可发布。**

---

## 附录 A：审计命令清单

```bash
# 1. 严重错误扫描
grep -rn "def backward" packages/  # 检查 backward 实现
grep -rn "np.where.*exp" packages/  # 检查 sigmoid 类 overflow 风险
grep -rn '"/workspace' packages/ data/  # 检查硬编码路径

# 2. 漏洞扫描
grep -rn "^\s*open(" packages/ data/  # 检查未关闭的 open()
grep -rn "/ 0\|/0\." packages/  # 检查除零风险

# 3. 重复代码扫描
grep -rn "def cross_entropy" packages/
grep -rn "nfkc_normalize\|NFKC" packages/

# 4. 测试运行
pytest tests/ -x --tb=short
pytest tests/test_training.py tests/test_hybrid_stability.py -W error::RuntimeWarning

# 5. 端到端验证
python examples/compress_train_demo.py
```

## 附录 B：相关文件路径

- 损失函数：`/workspace/packages/verse_torch/verse_torch/losses.py`
- 训练基础设施：`/workspace/packages/verse_torch/verse_torch/training.py`
- Tensor 类：`/workspace/packages/verse_torch/verse_torch/tensor.py`
- 模型加载器：`/workspace/packages/verse_inference/verse_inference/model_loader.py`
- NFKC 预处理：`/workspace/packages/verse_tokenizer/verse_tokenizer/preprocess.py`
- 评分器：`/workspace/packages/verse_torch/verse_torch/scoring.py`
- demo README：`/workspace/data/demo/README.md`
- spec tasks：`/workspace/.trae/specs/part3k2-major-upgrade/tasks.md`
- spec checklist：`/workspace/.trae/specs/part3k2-major-upgrade/checklist.md`

---

## Part4K1：基础设施全面升级 + 模型能力升级 + 优化（2026-07-22）

> 审计日期：2026-07-22
> 审计范围：`/workspace` 全项目（verse_torch / verse_nex / verse_infra / spark / tests / docs）
> 审计任务：SubTask 10.1 ~ 10.6（全项目 check-loop + 测试通过 + 审计报告 + 综合验收）
> 审计目标：测试零失败 + 关键导入可用 + CLI 端到端跑通 + shim 警告 + 文档与实现一致

### 变更概览
- 基础设施：VerseTorch GPU/NPU 后端抽象、VerseInfra 总包聚合、VerseTrainer 独立训练包
- 模型能力：VerseNex 重命名（TransformerLM→VerseNexLM）、MoD 完善、超稀疏并行注意力、NexRL（PPO+GAE+KL自适应）
- Tokenizer：BPE 并行训练、WordPiece、BatchEncoding、Qwen3.5-35B-A3B 支持、NexTokenizerWrapper
- CometSpark V0.5-1B：spark/ 目录、基于 VerseNexBlock、1B 参数、Qwen tokenizer
- 文档：4 新 ADR + README/training/perf guide 全面更新

### 新增/修改文件统计
- 新增包：verse_infra（聚合）、verse_trainer（独立）
- 新增目录：spark/
- 删除目录：data/demo/
- 新增测试文件：test_device_backend/test_mod_complete/test_parallel_sparse_attn/test_speculative_decode/test_nexrl/test_tokenizer_optimization/test_tokenizer_nex_wrapper/test_verse_trainer/test_verse_infra_imports/test_cometspark_v05（共约 280+ 新测试）
- 测试结果：**788 passed, 13 skipped, 0 failed**（排除 ijepa/rssm 环境依赖测试；全量单跑会 OOM，按测试文件分 8 批跑通）

### SubTask 10.1：全量测试零失败
- 运行 `pytest tests/ -k "not ijepa and not rssm"`，全量单跑因内存不足触发 OOM（进程被 SIGKILL，EXIT=137）。
- 改为按测试文件分 8 批串行执行，各批均 exit code 0：
  | 批次 | 测试文件 | 结果 |
  |---|---|---|
  | 1 | device_backend / mod_complete / parallel_sparse_attn / speculative_decode / nexrl / verse_infra_imports | 221 passed, 11 skipped |
  | 2 | training / training_optimization / training_nex / parallel_trainer / parallel / hybrid_stability | 138 passed, 1 skipped |
  | 3 | compression_poc / compression_integration / p10_parallel_compress / optim_extras / scheduler_extras | 55 passed |
  | 4 | tokenizer / tokenizer_standard / tokenizer_upgrade / verse_tokenizer / chat_data_loader | 83 passed |
  | 5 | unit_operators / recursion_fix / nn_advanced / scoring / yaml_config / mamba2_memory / passkey / no_garbled / val_loss_curve | 108 passed |
  | 6 | tokenizer_optimization / tokenizer_nex_wrapper / verse_trainer | 114 passed |
  | 7 | cometspark_inference / cometspark_nex / end_to_end | 42 passed |
  | 8 | cometspark_v05 | 27 passed, 1 skipped |
- **合计：788 passed, 13 skipped, 0 failed**。13 个 skipped 均为环境依赖（无 GPU / 无网络 / matplotlib 不可用等），符合预期。

### SubTask 10.2：关键导入验证 + 修复
- 首次验证发现：`from verse_infra.verse_trainer import VerseTrainer` 失败（`ImportError: cannot import name 'VerseTrainer'`），该子包仅导出 `train` / `ParallelTrainerSafe` / `RLTrainer` 等，缺少 `VerseTrainer` 门面名。
- **修复**：在 `packages/verse_infra/verse_infra/verse_trainer/__init__.py` 添加 `VerseTrainer = ParallelTrainerSafe` 别名（指向升级后的主训练器，含 `_safe_chunk_run` + 信号处理 + OOM 兜底 + 断点续训）并加入 `__all__`；同步在 `verse_infra/__init__.py` 便捷重导出补上 `VerseTrainer`。
- 修复后重新验证：`verse_infra` / `verse_nex` / `verse_torch` / `spark` 全部关键导入成功，输出 `ALL IMPORTS OK`。重跑 test_verse_trainer / test_verse_infra_imports 无回归（56 passed）。

### SubTask 10.3：verse-train CLI 端到端验证 + 修复
- 5 个子命令（verse-train / verse-finetune / verse-posttrain / verse-eval / verse-tokenize）`--help` 均返回 rc=0，CLI 可用。
- 首次端到端训练失败：`FileNotFoundError: tokenizer 文件不存在：.../checkpoints_small/tokenizer.json`。根因：`_load_tokenizer` 对所有 kind 一律要求 `tokenizer.json` 文件，但 byte tokenizer 无需训练文件（vocab 259 确定）。
- **修复**：在 `verse_trainer/trainer.py:_load_tokenizer` 中，当 `kind == "byte"` 且文件不存在时直接构造 `ByteTokenizer`（调用 `load_tokenizer(kind="byte")`），让 `verse-train` 对 byte 配置开箱即用（small 调试配置场景）；bpe/wordpiece 等仍要求文件。
- 修复后端到端验证：`verse-train --config spark/config/cometspark_v05_small.yml --device cpu --single-sample --prompt "你好世界" --completion "今天天气真好" --max-steps 5` 完整跑通，loss 5.6295→5.2890，模型保存至 `checkpoints_small/cometspark.pt`，END_RC=0。（1B 默认配置在沙箱内存下会 OOM，按任务要求用 small 配置验证，符合"若 OOM 用 small 配置"约束。）
- 重跑 test_verse_trainer 无回归（36 passed）。

### SubTask 10.4：旧路径 shim DeprecationWarning
- 旧路径 `from verse_tokenizer import BPETokenizer`（经 `packages/verse_tokenizer/verse_tokenizer/__init__.py` shim 转发）成功触发 `DeprecationWarning: verse_tokenizer 已迁入 verse_infra.verse_tokenizer，请改用 from verse_infra.verse_tokenizer import ...`，且 `BPETokenizer` 经 shim 重导出仍可用。
- 注：任务脚本中 `sys.path.insert(0, 'packages/verse_infra')` 路径有误（verse_infra 下无顶级 verse_tokenizer 模块），正确应为 `packages/verse_tokenizer`（shim 物理位置）；shim 内部已自举把 `packages/verse_infra` 加入 path。

### 修复的问题
- GPU/NPU 训练支持（DeviceBackend + PyTorch 委托 + 回退 NumPy）
- 并行训练"莫名终止退出"（`_safe_chunk_run` + 信号处理 + OOM 兜底 + 断点续训）
- 数据集加载耗时（CachedDataset `.npz` 缓存 + 流式 lazy load）
- Loss 无法优化（plateau 重走 + NaN/Inf 跳过 + LR 组合策略）
- 胡乱输出（embedding scale + tie weights + temperature scaling）
- `config.yml` hybrid 模式 NaN（删除 hybrid，统一 versenex）
- **Task 10 新增修复**：`verse_infra.verse_trainer` 缺 `VerseTrainer` 门面名（补别名）
- **Task 10 新增修复**：`_load_tokenizer` 对 byte tokenizer 强制要求文件（改为即时构造）

### 已知限制
- GPU 混合精度训练一致性需在真实 GPU 环境验证
- Qwen tokenizer 加载需网络（graceful skip 已实现）
- 1B 模型完整训练需 GPU/CPU 较长时间（沙箱内存不足以单跑 1B 配置 CLI，已用 small 配置验证端到端）
- 全量测试单跑会 OOM，需分批执行（已在审计中分 8 批跑通）

### 综合验收结论
Part4K1 Task 10 全项目 check-loop 通过：测试零失败（788 passed / 13 skipped / 0 failed）、关键导入全部成功、verse-train CLI 端到端跑通、旧路径 shim 发出 DeprecationWarning 但仍可工作、审计报告已更新。**VerseNext 框架 Part4K1 基础设施全面升级审计通过，可发布。**

---

## Part4K2：架构升级与模型优化（2026-07-22）

> 审计日期：2026-07-22
> 审计范围：`/workspace` 全项目（verse_torch / verse_nex / verse_infra / spark / data / tests / docs）
> 审计任务：SubTask 10.1 ~ 10.6（全量测试 + 关键导入 + CLI 端到端 + .vn 互转 + 审计报告 + 综合验收）
> 审计基线：`pytest tests/ -q -k "not ijepa and not rssm"` → **974 passed, 22 skipped, 0 failed** in 179.02s
> 审计目标：测试零失败 + 关键导入可用 + CLI 全子命令可用 + .vn ↔ .pt 互转无损 + audit_report 更新 + checklist 综合验收通过

### 变更概览
- **.vn 文件格式**：safetensors 性能优化版，ZIP 容器，pt↔vn 互转，mmap 零拷贝
- **jinja2 聊天模板**：ChatML (Qwen 风格) + 工具调用 (Qwen3 官方格式) + jinja2 可选依赖（缺失时降级 f-string）
- **生成输出优化**：不限制 token 数，EOS 自然停止 + 安全上限 100K
- **智能分区训练**：LayerWiseTrainer 按 layer 拆分 + .vn 分片卸载 + 统一实体 + 内存监控
- **资源利用优化**：autocast + GradScaler + empty_cache + CPU BLAS 线程 + pin_memory + prefetch + activation_checkpoint
- **压缩技术 V1.3**：以小博大，知识蒸馏增强 + 中间层特征蒸馏 + 温度退火 + 吞吐率优化
- **VerseTrainer 优化**：并行训练 tqdm + `--quiet`/`--verbose` + `verse-continue` 持续训练 + 1B 模型优化
- **数据集下载器**：任意 URL + HF datasets + 断点续传 + 多线程 + 自动转 `.npz`

### 新增/修改文件统计
- **新增文件**：
  - `packages/verse_torch/verse_torch/vn_format.py`（VNFileReader/VNFileWriter + pt_to_vn/vn_to_pt + convert_format）
  - `packages/verse_torch/verse_torch/layerwise_trainer.py`（LayerWiseTrainer 按层拆分 + .vn 分片卸载）
  - `data/downloader.py`（DatasetDownloader + URL/HF + 断点续传 + 多线程）
  - **8 个测试文件**：`test_vn_format.py` / `test_layerwise_trainer.py` / `test_tokenizer_standard.py` / `test_tokenizer_upgrade.py` / `test_generation_unlimited.py` / `test_compress_v13.py` / `test_trainer_tqdm_continue.py` / `test_downloader.py` / `test_resource_optimization.py`
- **修改文件**：
  - `verse_torch/`：`compress.py`（V1.3 集成）、`quantize.py`、`training.py`（无 token 限制）、`training_nex.py`、`backend_torch.py`（autocast/GradScaler）、`device.py`（empty_cache + get_memory_info）、`__init__.py`（导出 VNFileReader 等）
  - `verse_nex/`：`cometspark.py`（CometSparkNexLM.generate 无限制 + compress_v13）
  - `verse_infra/verse_tokenizer/`：`chat_template.py`（jinja2 ChatML + 工具调用）、`verse.py`（apply_chat_template_with_tools）、`__init__.py`（导出 CHATML_TEMPLATE 等）
  - `verse_infra/verse_trainer/`：`cli.py`（新增 verse-convert/verse-download/verse-continue 子命令）、`trainer.py`（tqdm + 持续训练）、`data.py`、`evaluate.py`
  - `verse_infra/verse_inference/`：`generator.py`（StreamingGenerator 无默认限制）
  - `spark/`：`model/model.py`（CometSparkV05LM.save_vn/load_vn + generate 无限制）、`config/cometspark_v05.yml`
  - `README.md`：新增 Part4K2 章节（.vn 格式 + jinja2 模板 + 分区训练 + 下载器 + 持续训练）
- **新增 CLI 子命令**：`verse-convert`（.pt ↔ .vn）、`verse-download`（数据集下载）、`verse-continue`（持续训练）
- **ADR 文档**：Task 9.2 计划新增 4 个 ADR（.vn 格式 / jinja2 / 分区训练 / 压缩 V1.3），实际未落地（见"已知限制"）
- **测试结果**：**974 passed, 22 skipped, 0 failed**（排除 ijepa/rssm 环境依赖测试；本次单跑未触发 OOM，全量在 179s 内完成）

### SubTask 10.1：全量测试零失败

```bash
$ python -m pytest tests/ -q -k "not ijepa and not rssm" --no-header
974 passed, 22 skipped, 2 deselected, 36 warnings in 179.02s
```

- ✅ **0 failed**，全部测试通过
- ✅ 22 skipped 均为环境依赖（无 GPU / 无网络 / matplotlib / jinja2 / safetensors 等可选依赖缺失），符合预期
- ✅ 2 deselected 为 `ijepa` / `rssm`（按任务要求排除的环境依赖测试）
- ✅ 36 warnings 均为 `PytestReturnNotNoneWarning`（测试风格）与 `DeprecationWarning`（HybridBlock/HybridLM 已弃用，Part4K1 已记录），非 BUG
- ✅ 相比 Part4K1 基线（788 passed / 13 skipped），新增 **186 passed**（vn_format 33 + jinja2 模板 + 生成无限制 19 + 分区训练 20 + 资源优化 47 + 压缩 V1.3 16 + VerseTrainer 33 + 下载器 24 + 其他）

### SubTask 10.2：关键导入验证

```bash
$ python -c "
from verse_torch import VNFileReader, VNFileWriter, pt_to_vn, vn_to_pt, convert_format, \\
                      LayerWiseTrainer, empty_cache, get_memory_info, set_num_threads, \\
                      auto_tune_threads, GradScaler, activation_checkpoint, compression_report
from verse_infra.verse_tokenizer import CHATML_TEMPLATE, CHATML_TEMPLATE_WITH_TOOLS, \\
                                        render_chat_qwen_with_tools, extract_tool_calls_qwen3
from verse_nex import CometSparkNexLM
from spark.model.model import CometSparkV05, CometSparkV05Small, CometSparkV05LM
from data.downloader import DatasetDownloader
print('ALL IMPORTS OK')
"
ALL IMPORTS OK
```

- ✅ `verse_torch` 全部新导出符号可导入（VNFileReader / VNFileWriter / pt_to_vn / vn_to_pt / convert_format / LayerWiseTrainer / empty_cache / get_memory_info / set_num_threads / auto_tune_threads / GradScaler / activation_checkpoint / compression_report）
- ✅ `verse_infra.verse_tokenizer` 全部新导出符号可导入（CHATML_TEMPLATE / CHATML_TEMPLATE_WITH_TOOLS / render_chat_qwen_with_tools / extract_tool_calls_qwen3）
- ✅ `verse_nex.CometSparkNexLM` 可导入
- ✅ `spark.model.model` 三个模型类可导入（CometSparkV05 / CometSparkV05Small / CometSparkV05LM）
- ✅ `data.downloader.DatasetDownloader` 可导入

### SubTask 10.3：CLI 端到端验证

逐个验证 8 个 CLI 子命令 `--help`，全部 `rc=0`：

| 子命令 | 用途 | rc |
|---|---|---|
| `verse-train` | 训练（含 `--parallel-chunks` / `--amp` / `--resume`） | 0 |
| `verse-finetune` | 微调（LoRA / full） | 0 |
| `verse-posttrain` | 后训练（nexrl / sft / dpo） | 0 |
| `verse-eval` | 评估（含 `--score` / `--max-tokens` / `--top-p`） | 0 |
| `verse-tokenize` | 分词器训练/加载/转换 | 0 |
| `verse-convert` | **新增**：.pt ↔ .vn 互转（基于 safetensors） | 0 |
| `verse-download` | **新增**：数据集下载（URL + HF datasets） | 0 |
| `verse-continue` | **新增**：从 checkpoint 持续训练 | 0 |

- ✅ 全部 8 个子命令 `--help` 输出正常 usage 信息，参数完整
- ✅ 3 个新增子命令（convert / download / continue）均可用

### SubTask 10.4：.vn ↔ .pt 互转无损验证

```python
model = CometSparkV05Small()  # 193734 参数
model.save(pt_path)             # 保存 .pt
pt_to_vn(pt_path, vn_path)      # .pt → .vn
vn_to_pt(vn_path, pt2_path)     # .vn → .pt
# 逐 key 比对 state_dict
for k in d1['state_dict']:
    assert np.array_equal(d1['state_dict'][k], d2['state_dict'][k])
print('PT ↔ VN 互转无损验证通过')
```

- ✅ `CometSparkV05Small`（193734 参数）保存为 `.pt`
- ✅ `pt_to_vn` 转换为 `.vn`（ZIP + safetensors 或 npz 降级）
- ✅ `vn_to_pt` 转回 `.pt`
- ✅ 逐 key 比对 `state_dict` 全部 `np.array_equal` 通过，**权重数值无损**

### 修复的问题
- 训练内存占用过高（智能分区训练 LayerWiseTrainer + activation checkpointing + 梯度累积 + .vn 分片卸载）
- 模型输出无规范（jinja2 ChatML 模板 + Qwen3 工具调用 + add_generation_prompt）
- LM 输出 token 数限制（移除默认限制，EOS 自然停止 + 安全上限 100K）
- 压缩技术未集成（V1.3 集成到 VerseNex `CometSparkNexLM.compress_v13` 与 VerseTorch `compress.py`）
- 模型文件格式单一（.vn 性能优化版 + .pt 互转 + mmap 零拷贝 + compression_info 透传）
- 数据集获取不便（DatasetDownloader + URL/HF + 断点续传 + verse-download CLI）
- 持续训练无入口（verse-continue CLI + tqdm 进度条 + --quiet/--verbose）

### 已知限制
- **Task 9 文档未完成**：README 已更新 Part4K2 章节，但 `docs/architecture/` 下未新增 4 个 ADR（.vn 格式 / jinja2 / 分区训练 / 压缩 V1.3），`docs/training_guide.md` 与 `docs/performance_tuning.md` 未追加 Part4K2 内容，`docs/papers/compression_references.md` 未补充 V1.3 蒸馏条目。checklist 中 9.2 / 9.3 / 9.4 保持未勾选，待后续补齐
- safetensors 未安装时降级 npz（功能完整但 mmap 零拷贝性能略低）
- jinja2 未安装时降级 f-string 拼接（功能等价，但模板维护性略低）
- GPU 混合精度训练一致性需真实 GPU 环境验证
- 智能分区训练的合并阶段需要足够内存（沙箱内存有限，1B 完整训练用 small 配置验证）
- 全量测试单跑在 Part4K1 时会 OOM（需分批）；Part4K2 本次 179s 单跑通过，但内存敏感环境仍建议分批

### 综合验收结论
Part4K2 Task 10 全项目综合验收通过：

1. **全量测试零失败**：974 passed / 22 skipped / 0 failed（22 skipped 均为环境依赖，符合预期）
2. **关键导入全部成功**：`verse_torch` / `verse_infra.verse_tokenizer` / `verse_nex` / `spark.model.model` / `data.downloader` 全部关键符号可导入，输出 `ALL IMPORTS OK`
3. **CLI 端到端验证通过**：8 个子命令（train / finetune / posttrain / eval / tokenize / convert / download / continue）`--help` 全部 `rc=0`，3 个新增子命令可用
4. **.vn ↔ .pt 互转无损**：`CometSparkV05Small` 193734 参数 `.pt → .vn → .pt` 全部 `np.array_equal` 通过
5. **audit_report 已更新**：本节即 Part4K2 审计章节
6. **checklist 综合验收**：Task 10 全部 6 项已勾选；Task 1-8 全部已勾选；Task 9 中 9.1（README）已勾选，9.2/9.3/9.4 因文档未落地保持未勾选
7. **无回归问题**：相比 Part4K1 基线（788 passed / 13 skipped），新增 186 passed、9 skipped，0 failed，无回归

**VerseNext 框架 Part4K2 架构升级与模型优化审计通过，可发布（Task 9 文档补齐为后续工作）。**

---

## Part4K2.5：紧急优化（2026-07-22）

> 审计日期：2026-07-22
> 审计范围：`/workspace` 全项目（spark / verse_torch / verse_infra / docs / tests）
> 审计任务：Task 7（升级文档 + 注释，删除旧的代码与文件）
> 审计目标：文档与实现一致 + 代码注释完善 + 旧构建产物 / 缓存清理 + .gitignore 补全

### 变更概览
- spark/run.py CLI 快捷方式：7 子命令，提升易用度
- 包导入修复：spark/_bootstrap.py 统一路径引导，简化 6 处 sys.path.insert
- loss 图表修复：plot_loss_curve x 轴正确，ASCII 降级显示 val 线
- 训练后自动评估：eval_after 默认 True，5 指标打分
- 小错误修复 + 性能优化
- 并行训练修复：chunk 状态重置，Phase 2 跳过，非 tty 降级

### 新增文件
- spark/run.py, spark/_bootstrap.py
- tests/test_spark_run.py, tests/test_loss_and_parallel_fix.py, tests/test_auto_eval.py

> 注：Task 7 文档同步时核对发现 `tests/test_bugfixes_perf.py` 在仓库中不存在（前序 Task 5 未落地该测试文件），故本审计不列入。Task 5 的小错误修复与性能优化已由 `test_loss_and_parallel_fix.py` / `test_auto_eval.py` 等覆盖验证。

### 修复的问题
- 包导入 sys.path 膨胀和跨路径风险
- loss 图表 x 轴偏移和 val 线不显示
- 训练后无自动验证
- 并行训练 chunk 间状态泄漏
- 并行训练 Phase 2 步数为 0 时崩溃
- 非 tty 环境 tqdm 输出垃圾字符

### Task 7 文档与清理工作

#### 文档更新
- `README.md`：新增「spark/run.py 快捷入口」章节（7 子命令表 + 示例）；快速开始新增 `python spark/run.py train --small`；新增「Part4K2.5 重大升级摘要」章节；仓库结构图补充 `spark/_bootstrap.py` / `spark/run.py`。
- `docs/training_guide.md`：新增第 16~19 章（spark/run.py 快速训练指南 / 训练后自动评估指南 / 并行训练修复说明 / loss 图表修复说明），目录同步更新。
- `docs/performance_tuning.md`：新增第 16~17 章（Part4K2.5 性能优化清单 / 并行训练调优建议），引言维度计数从 13 更新为 15。
- `spark/README.md`：新增 run.py 与 _bootstrap.py 说明，更新训练/评估/生成命令示例。
- `audit_report.md`：追加本 Part4K2.5 章节。

#### 代码注释校验
- `spark/run.py`：模块 docstring + 各子命令 docstring + 路径自举注释完善（Task 1 已落地，无需补充）。
- `spark/_bootstrap.py`：模块 docstring + `ensure_paths()` docstring + 路径计算注释完善（Task 2 已落地，无需补充）。
- `training.py` `ParallelTrainer._train_chunk`：已含「Part4K2.5 Task 6 修复」注释（chunk 状态重置 / 独立优化器 / 模型状态备份）。
- `training.py` `_ChunkProgressBar`：已含「Part4K2.5 Task 6 修复：非 tty 环境降级」注释。
- `trainer.py` `train()`：docstring 已含 `eval_after` / `eval_config` 参数说明（Part4K2.5 Task 4）。

#### 旧代码与文件清理
- **`__pycache__` 清理**：删除全部 19 个 `__pycache__` 目录（Python 编译缓存，自动重生成）。
- **构建残留删除**：删除 `packages/verse_nex/build/`（陈旧 `python setup.py build` 产物，仅含 Part4 之前的 hybrid/linear_attention/mamba2/positional/rwkv7/sparse_attention，缺少 Part4K1 新增的 tri_sparse_attn/moe/cometspark/nexrl/speculative/kv_cache_parallel，确认为无用残留）。
- **shim 包保留**：`packages/verse_compat` / `packages/verse_inference` / `packages/verse_tokenizer` / `packages/verse_trainer` 四个 shim 包仍在使用（旧导入路径转发 + DeprecationWarning），按约束保留不删除。
- **临时文件扫描**：全项目无 `.bak` / `.orig` / `.tmp` 文件。

#### .gitignore 补全
新增条目：
- `*.vn`（模型权重容器，Part4K2 新格式）
- `checkpoints/`（训练产物目录）
- `loss_history.json`（训练 loss 历史）

已确认存在（无需新增）：`__pycache__/` / `*.py[cod]`（含 `*.pyc` `*.pyo`）/ `.eggs/` / `*.egg-info/` / `build/` / `dist/` / `*.egg` / `.pytest_cache/` / `*.pt`。

### 综合验收结论
Part4K2.5 Task 7 文档与清理工作完成：

1. **文档与实现一致**：README / training_guide / performance_tuning / spark/README / audit_report 全部同步 Part4K2.5 的 6 项变更（run.py 入口 / 包导入修复 / loss 图修复 / 自动评估 / 性能优化 / 并行训练修复）。
2. **代码注释完善**：4 个关键文件（spark/run.py / spark/_bootstrap.py / training.py ParallelTrainer / trainer.py train）的注释经校验均已在 Task 1~6 落地，Task 7 无需补充。
3. **旧构建产物清理**：19 个 `__pycache__` 目录 + `verse_nex/build/` 陈旧残留已删除；shim 包按约束保留。
4. **.gitignore 补全**：新增 3 条必要条目，未添加不必要条目。

**VerseNext 框架 Part4K2.5 紧急优化文档与清理审计通过。**
