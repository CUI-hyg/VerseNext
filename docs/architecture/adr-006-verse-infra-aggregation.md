# ADR-006: VerseInfra 总包聚合

- **状态**：Accepted
- **日期**：2026-07-22
- **决策者**：Verse 框架作者（CometFuture / CUI-hyg）
- **相关规范**：[`/workspace/.trae/specs/part4k1-infra-model-upgrade/spec.md`](../../../.trae/specs/part4k1-infra-model-upgrade/spec.md)
- **前置 ADR**：[ADR-001 CPU 优先](adr-001-cpu-first.md)
- **相关 ADR**：[ADR-005 GPU/NPU 后端](adr-005-gpu-npu-backend.md)（VerseTrainer 通过 DeviceBackend 支持 GPU/NPU）

## 上下文

Part4K1 之前，Verse 仓库的辅助包布局如下：

```
packages/
├── verse_torch/        # 张量 + autograd
├── verse_nex/          # 线性复杂度架构
├── verse_awm/          # 世界模型
├── verse_tokenizer/    # 分词器（独立顶层包）
├── verse_compat/       # HF/PyTorch 兼容层（独立顶层包）
└── verse_inference/    # 推理引擎（独立顶层包）
```

`verse_tokenizer` / `verse_compat` / `verse_inference` 是 3 个独立的顶层包，存在以下问题：

1. **安装繁琐**：用户需要 `pip install -e` 多次才能装齐辅助包
2. **版本漂移**：4 个包各自 `pyproject.toml`，版本号与依赖声明容易不一致
3. **训练包缺失**：Part4K1 新增的 `verse_trainer`（从 `data/demo/` 迁移的训练栈）需要决定放哪
4. **导入路径割裂**：`from verse_tokenizer import X` / `from verse_inference import Y` / `from verse_compat import Z` 风格不统一
5. **shim 维护成本**：若要重命名顶层包，需要为每个旧路径维护 shim

同时，必须保持向后兼容：现有用户的 `from verse_tokenizer import BPETokenizer` 代码不能突然失效。

## 决策

**创建 `verse_infra` 总包，将 `verse_tokenizer` / `verse_compat` / `verse_inference` / `verse_trainer` 聚合为子模块；旧顶层包位置保留 thin shim 转发 + `DeprecationWarning`。**

具体含义：

1. **总包结构**：
   ```
   packages/verse_infra/
   ├── pyproject.toml                  # 总包元数据 + 依赖声明
   └── verse_infra/
       ├── __init__.py                 # 便捷重导出 + __getattr__ 延迟导入
       ├── verse_tokenizer/            # 子模块（原 packages/verse_tokenizer/ 源码）
       ├── verse_compat/               # 子模块（原 packages/verse_compat/ 源码）
       ├── verse_inference/            # 子模块（原 packages/verse_inference/ 源码）
       └── verse_trainer/              # 子模块（Part4K1 新增训练包）
   ```

2. **`verse_torch` / `verse_nex` 保持独立**：它们是底层后端（autograd / 架构），依赖关系与辅助包不同，不并入 VerseInfra

3. **便捷重导出**（`verse_infra/__init__.py`）：
   - `__all__` 列出所有子模块的常用 API（`BPETokenizer` / `ModelLoader` / `train` / `RLTrainer` 等）
   - `__getattr__` 延迟导入：`import verse_infra` 不强制加载子包，首次访问某个 API 时才加载对应子模块
   - 这避免"导入 `verse_infra` 就拖入训练栈"的副作用

4. **shim 兼容**（原顶层包位置）：
   - `packages/verse_tokenizer/` / `packages/verse_compat/` / `packages/verse_inference/` 保留 thin shim
   - shim 内容：`from verse_infra.verse_xxx import *` + `warnings.warn(..., DeprecationWarning)`
   - shim 只保留一个版本，下次 major release 删除

5. **导入路径迁移**：
   - 旧：`from verse_tokenizer import BPETokenizer`
   - 新（推荐）：`from verse_infra.verse_tokenizer import BPETokenizer`
   - 便捷：`from verse_infra import BPETokenizer`
   - 全项目（`tests/` / `examples/` / `packages/verse_nex/` / `packages/verse_torch/` / `spark/` / `docs/`）已统一迁移到新路径

6. **根 `pyproject.toml`**：声明 `verse_infra` 为 workspace 成员，删除旧 `verse_tokenizer` / `verse_compat` / `verse_inference` 顶层包声明

## 后果

### 优点

- **单包安装**：`pip install -e packages/verse_infra` 一次装齐四个子模块
- **版本对齐**：所有子模块共享 `verse_infra.__version__`，依赖声明统一
- **导入路径统一**：`from verse_infra.verse_xxx import` 风格一致，便于 IDE 自动补全
- **便捷重导出**：常用 API 可直接 `from verse_infra import BPETokenizer, train`，减少导入行数
- **延迟加载**：`__getattr__` 避免"导入 `verse_infra` 就拖入训练栈"，启动快
- **向后兼容**：旧路径仍工作（shim 转发 + 警告），用户代码不会突然失效
- **训练包有家**：`verse_trainer` 作为子模块自然归属，不再悬空

### 缺点

- **shim 维护成本**：shim 保留一个版本，需要在本周期内推动用户迁移到新路径
- **导入路径变长**：`from verse_infra.verse_tokenizer import BPETokenizer` 比旧的 `from verse_tokenizer import BPETokenizer` 长——可通过便捷重导出 `from verse_infra import BPETokenizer` 缓解
- **`__getattr__` 调试困难**：延迟导入意味着 `dir(verse_infra)` 不一定列出所有 API（已通过 `__dir__()` 缓解，tab 补全可用）
- **子模块 README 路径变深**：`packages/verse_infra/verse_infra/verse_tokenizer/README.md` 比 `packages/verse_tokenizer/README.md` 长

### 风险与缓解

| 风险 | 缓解策略 |
|---|---|
| 用户忽略 `DeprecationWarning` 导致下次 major release 代码失效 | 警告信息明确指引迁移路径；`tests/test_verse_infra_imports.py` 验证 shim + 警告；文档（`packages/verse_infra/README.md`）详述迁移指南 |
| `__getattr__` 延迟导入导致首次访问慢 | 缓存机制：首次访问后 `globals()[name] = _val`，后续直接从 `__dict__` 取值 |
| 子模块循环依赖 | `verse_trainer` 依赖 `verse_torch` / `verse_nex`，但它们不在 `verse_infra` 内，通过 `sys.path` 自举解决（见 `__init__.py` 路径自举代码） |
| shim 与真包子模块名冲突 | shim 只在原顶层包位置（`packages/verse_tokenizer/`），真包在 `packages/verse_infra/verse_infra/verse_tokenizer/`，Python import 机制通过 `sys.path` 优先级区分 |

## 替代方案（已否决）

### 方案 A：保持 4 个独立顶层包 + 新增 `verse_trainer` 为第 5 个

**描述**：不聚合，`verse_trainer` 作为新的独立顶层包。

**否决理由**：
- 安装更繁琐（5 个包）
- 版本/依赖漂移问题不解决
- 导入路径仍割裂
- 与 Part4K1 "基础设施全面升级"目标不符

### 方案 B：所有包（含 verse_torch / verse_nex）聚合为单一 `verse` 包

**描述**：创建 `verse` 总包，把所有 6 个包都聚合为子模块。

**否决理由**：
- `verse_torch` / `verse_nex` 是底层后端，依赖关系与辅助包不同（`verse_nex` 依赖 `verse_torch`，`verse_infra` 依赖两者）
- 聚合后循环依赖风险高
- 破坏 `pip install verse-torch` 单独安装的能力
- 现有用户 `from verse_torch import Tensor` 代码全部失效

### 方案 C：用 namespace package（PEP 420）替代总包

**描述**：用 `verse` 作为 namespace package，各子包独立安装但共享 `verse.` 前缀。

**否决理由**：
- namespace package 调试困难（`__path__` 不直观）
- IDE 支持不如显式总包好
- 无法提供便捷重导出（namespace package 没有 `__init__.py`）
- 现有 `verse_torch` / `verse_nex` 命名不带 `verse.` 前缀，迁移成本高

### 方案 D：不保留 shim，直接删除旧顶层包

**描述**：物理删除 `packages/verse_tokenizer/` 等，强制用户立即迁移。

**否决理由**：
- 破坏所有现有用户代码（`from verse_tokenizer import` 全部失效）
- 违反"向后兼容"承诺
- 风险过高，不符合渐进式迁移原则

## 备注

- 本 ADR 是 Part4K1 "基础设施全面升级"的核心决策之一
- `verse_torch` / `verse_nex` 保持独立的理由：它们是底层后端，被 `verse_infra` 依赖（依赖方向不能反转）
- shim 保留一个版本（Part4K1 周期），下次 major release 删除
- 相关测试：`tests/test_verse_infra_imports.py` 覆盖子模块导入 / 便捷重导出 / 旧路径 shim DeprecationWarning
- 相关文档：[`packages/verse_infra/README.md`](../../packages/verse_infra/README.md)

## 演进更新（Part4K2）

本 ADR 的 VerseInfra 总包聚合结构保持不变。Part4K2 在此基础上扩展了 `verse_infra` 的导出能力与 CLI 子命令：

- **DatasetDownloader 顶层导出**：`data/downloader.py` 的 `DatasetDownloader` 通过 `verse_infra/__init__.py` 的 `__getattr__` 懒加载导出，用户可 `from verse_infra import DatasetDownloader` 直接使用（无需知道子模块路径）。
- **CLI 新增 3 个子命令**：`verse-convert`（模型格式互转）/ `verse-download`（数据集下载）/ `verse-continue`（持续训练，通过 `python -m verse_infra.verse_trainer.cli verse-continue` 统一分发入口调用，未注册为独立 console_script）。
- **可选依赖新增**：`pyproject.toml` 新增 `chatml = ["jinja2>=3.0"]` 可选依赖（ChatML 模板渲染）；`datasets>=2.18` 作为 `verse-download --hf` 的可选依赖。

`verse_trainer` 子模块的 CLI 从 5 个扩展为 8 个入口，但总包结构（`verse_tokenizer` / `verse_compat` / `verse_inference` / `verse_trainer` 四子模块）不变。
