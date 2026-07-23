# ADR-016: verse_torch.nn → verse_torch.vnn 重命名

- **状态**：Accepted（BREAKING）
- **日期**：2026-07-23
- **决策者**：Verse 框架作者（CometFuture / CUI-hyg）
- **相关规范**：Part5K1 升级任务集
- **前置 ADR**：[ADR-001 CPU 优先](adr-001-cpu-first.md)（VerseTorch 是核心引擎）、Part4K1 VerseNex 品牌落地（TransformerLM → VerseNexLM 等命名统一）
- **相关 ADR**：[ADR-006 VerseInfra 总包聚合](adr-006-verse-infra-aggregation.md)（shim 兼容模式参考）

## 上下文

Part4K1 完成 VerseNex 品牌落地（`TransformerLM` → `VerseNexLM`、`GQASelfAttention` → `VerseNexAttention`，旧名作为 `DeprecationWarning` 别名）。但 `verse_torch` 包内的核心神经网络模块仍叫 `verse_torch.nn`，与 VerseNext 品牌体系不一致：

1. **命名不统一**：VerseTorch 的核心类（`Module` / `Linear` / `Embedding` / `LayerNorm` / `Conv1d` / `TransformerLM` 等）放在 `verse_torch/nn.py`，导入路径为 `from verse_torch.nn import Module`。`nn` 是 PyTorch 的命名习惯，Verse 作为独立框架应有自己的命名（`vnn` = verse_torch.nn）。
2. **与 VerseNex 品牌脱节**：VerseNex 已完成品牌落地（`VerseNexLM` / `VerseNexAttention` / `VerseNexBlock`），但 VerseTorch 的 `nn` 模块仍是 PyTorch 风格命名，品牌体系不连贯。
3. **旧 transformer 类名仍可导入**：Part4K1 将 `TransformerLM` / `TransformerBlock` / `GQASelfAttention` 作为 `DeprecationWarning` 别名保留，但 Part5K1 已进入品牌统一阶段，这些旧名应升级为抛 `ImportError`（明确引导用户迁移）。
4. **`from verse_torch import nn` 仍需兼容**：PyTorch 用户习惯 `from torch import nn`，Verse 需保留 `from verse_torch import nn` 的兼容路径（作为 `vnn` 的别名），降低迁移成本。

## 决策

**将 `verse_torch.nn` 重命名为 `verse_torch.vnn`（BREAKING）；`nn.py` 降级为 thin shim（`from .vnn import *`）；transformer 系旧名（TransformerLM / TransformerBlock / GQASelfAttention）从 DeprecationWarning 升级为抛 ImportError；`__init__.py` 中 `nn = vnn` 别名（向后兼容 `from verse_torch import nn`）。**

具体含义：

1. **`verse_torch.nn` → `verse_torch.vnn`（BREAKING）**：
   - 原 `verse_torch/nn.py` 内容迁移到 `verse_torch/vnn.py`（核心类 `Module` / `Linear` / `Embedding` / `LayerNorm` / `Conv1d` / `GroupNorm` / `RotaryEmbedding` / `KVCache` / `VerseNexLM` / `VerseNexBlock` / `VerseNexAttention` 等）。
   - 导入路径：`from verse_torch.nn import Module` → `from verse_torch.vnn import Module`。

2. **`nn.py` 降级为 thin shim**：
   - `verse_torch/nn.py` 保留，但内容仅为 `from .vnn import *`（thin shim 转发）。
   - 旧代码 `from verse_torch.nn import Module` 仍可工作（经 shim 转发到 `vnn`），但视为已废弃路径。
   - shim 模式借鉴自 VerseInfra 总包聚合（[ADR-006](adr-006-verse-infra-aggregation.md)）的旧路径兼容策略。

3. **transformer 系旧名升级为抛 ImportError**：
   - Part4K1 的 `DeprecationWarning` 别名（`TransformerLM` / `TransformerBlock` / `GQASelfAttention`）升级为抛 `ImportError`，明确引导用户迁移到 VerseNex 命名（`VerseNexLM` / `VerseNexBlock` / `VerseNexAttention`）。
   - 迁移期结束，旧名不再可用（BREAKING）。

4. **`__init__.py` 中 `nn = vnn` 别名**：
   - `verse_torch/__init__.py` 中保留 `from . import nn`（指向 `vnn` 模块），向后兼容 `from verse_torch import nn`。
   - `from verse_torch import nn` 等价于 `from verse_torch import vnn`，两者是同一对象。
   - PyTorch 用户的 `from verse_torch import nn` 习惯不破坏。

## 后果

### 优点

- **品牌统一**：`vnn`（verse_torch.nn）与 `VerseNex` / `VMPC` / `VMT` 品牌体系一致，Verse 作为独立框架的命名连贯性提升。
- **引导迁移**：旧 transformer 类名抛 `ImportError`，明确引导用户迁移到 VerseNex 命名，避免新旧名共存的技术债。
- **向后兼容**：`nn.py` thin shim + `__init__.py` 别名保证 `from verse_torch.nn import` 与 `from verse_torch import nn` 仍可工作，降低迁移成本。
- **清理技术债**：Part4K1 的 `DeprecationWarning` 别名在 Part5K1 正式移除，避免长期维护两套命名。

### 缺点

- **BREAKING 变更**：`from verse_torch.nn import TransformerLM` 会抛 `ImportError`，依赖旧名的代码需迁移。
- **双路径共存**：`vnn` 与 `nn`（shim）并存，短期内仍有两条导入路径，需文档引导用户优先用 `vnn`。
- **第三方代码迁移成本**：使用 `from verse_torch.nn import` 的第三方代码需改为 `from verse_torch.vnn import`（虽然 shim 仍可工作，但官方推荐迁移）。

### 风险与缓解

| 风险 | 缓解策略 |
|---|---|
| 旧 transformer 类名抛 ImportError 破坏现有代码 | Part4K1 已发 DeprecationWarning 预告；迁移指南明确（TransformerLM → VerseNexLM 等） |
| `nn.py` shim 长期存在导致用户不迁移 | 文档明确推荐 `vnn`；shim 视为已废弃；未来版本可能移除 shim |
| `from verse_torch import nn` 与 `from verse_torch import vnn` 行为不一致 | 两者是同一对象（`nn = vnn` 别名），行为完全一致 |
| 第三方教程 / 示例仍用旧路径 | 官方文档与示例统一改为 `vnn`；README 迁移指南明确 |

## 迁移指南

### 导入路径迁移

```python
# 旧（仍可工作，但已废弃）
from verse_torch.nn import Module, Linear, Embedding

# 新（推荐）
from verse_torch.vnn import Module, Linear, Embedding

# 兼容路径（仍可工作，不破坏）
from verse_torch import nn  # nn 是 vnn 的别名
nn.Module, nn.Linear, nn.Embedding
```

### transformer 系旧名迁移（BREAKING）

```python
# 旧（Part5K1 起抛 ImportError）
from verse_torch.nn import TransformerLM       # ❌ ImportError
from verse_torch.nn import TransformerBlock    # ❌ ImportError
from verse_torch.nn import GQASelfAttention    # ❌ ImportError

# 新（VerseNex 命名）
from verse_torch.vnn import VerseNexLM         # ✅
from verse_torch.vnn import VerseNexBlock      # ✅
from verse_torch.vnn import VerseNexAttention  # ✅
```

### 行为等价性

`vnn` 与 `nn`（shim）是同一模块对象，所有类与函数完全等价：

```python
from verse_torch import vnn, nn
assert vnn.Module is nn.Module                 # True
assert vnn.Linear is nn.Linear                 # True
assert vnn.VerseNexLM is nn.VerseNexLM         # True
```

## 替代方案（已否决）

### 方案 A：保留 nn 不改名

**描述**：`verse_torch.nn` 保持不变，不引入 `vnn`。

**否决理由**：命名不统一，`nn` 是 PyTorch 习惯，Verse 作为独立框架应有自己的命名；与 VerseNex / VMPC / VMT 品牌体系脱节；旧 transformer 类名的 DeprecationWarning 长期不清理会积累技术债。

### 方案 B：直接删除 nn.py（无 shim）

**描述**：`nn.py` 直接删除，所有导入强制改为 `vnn`。

**否决理由**：破坏所有 `from verse_torch.nn import` 与 `from verse_torch import nn` 的现有代码，迁移成本过高；违背"渐进式迁移"原则；shim 是行业标准的兼容策略（参考 VerseInfra [ADR-006](adr-006-verse-infra-aggregation.md)）。

### 方案 C：保留 transformer 旧名为 DeprecationWarning

**描述**：`TransformerLM` 等旧名继续作为 `DeprecationWarning` 别名保留，不升级为 ImportError。

**否决理由**：Part4K1 已发 DeprecationWarning，Part5K1 是品牌统一阶段，旧名应正式移除；长期保留两套命名增加维护成本与用户困惑。

## 备注

- 本 ADR 是 Part5K1 "verse_torch.nn → vnn 重命名"的核心决策（BREAKING）。
- `vnn` 命名含义：verse_torch.nn（v = verse_torch，nn = neural network），与 VerseNex / VMPC / VMT 品牌体系一致。
- `nn.py` thin shim 模式借鉴自 VerseInfra 总包聚合（[ADR-006](adr-006-verse-infra-aggregation.md)）的旧路径兼容策略。
- transformer 系旧名（TransformerLM / TransformerBlock / GQASelfAttention）的 DeprecationWarning 在 Part4K1 发出，Part5K1 正式升级为 ImportError。
- `__init__.py` 的 `nn = vnn` 别名保证 `from verse_torch import nn` 习惯不破坏（PyTorch 用户友好）。
- 相关测试：`tests/test_vnn_rename.py` 覆盖 vnn 导入 / nn shim 转发 / 旧名 ImportError / nn=vnn 别名等价性。
- 相关文档：[主 README - vnn 重命名](../../README.md)、[Verse 训练指南 - 推荐导入路径](../training_guide.md)
