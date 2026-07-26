# ADR-018: Gigatoken 集成（默认 Tokenizer + Lazy Import + 降级策略）

- **状态**：Accepted
- **日期**：2026-07-25
- **决策者**：Verse 框架作者（CometFuture / CUI-hyg）
- **相关规范**：[`/workspace/.trae/specs/part5k1.3-bugfix-stability-upgrade/spec.md`](../../../.trae/specs/part5k1.3-bugfix-stability-upgrade/spec.md)
- **前置 ADR**：[ADR-001 CPU 优先](adr-001-cpu-first.md)（零重型依赖原则）、[ADR-006 VerseInfra 总包聚合](adr-006-verse-infra-aggregation.md)（`verse_tokenizer` 子模块）
- **相关 ADR**：[ADR-010 jinja2 聊天模板](adr-010-jinja2-chat-template.md)（gigatoken 兼容模式保留 HF `apply_chat_template` 接口）、[ADR-014 双模型并行 small/mate](adr-014-dual-model-small-mate.md)（双模型默认 tokenizer 切换到 gigatoken）

## 上下文

`verse_infra.verse_tokenizer` 在 Part3K2 / Part4K1 / Part4K2 已落地完整的 tokenizer 体系：`ByteTokenizer` / `BPETokenizer` / `SentencePieceUnigramTokenizer` / `WordPieceTokenizer` / `NexTokenizerWrapper` + ChatML jinja2 模板。其中生产路径主力是 `BPETokenizer.from_pretrained("Qwen/Qwen3.5-35B-A3B")`（vocab 248320），内部 lazy import HuggingFace `transformers.AutoTokenizer` 加载。

但 Part5K1.3 在落地 mate 旗舰（≈1.12B 参数）大规模预训练时，暴露出 3 个性能与生态短板：

1. **`VerseTokenizer` 大语料 encode 慢**：`BPETokenizer.from_pretrained` 内部委托 HuggingFace `AutoTokenizer`，HF tokenizers 库虽是 Rust 实现，但 Python ↔ Rust 跨语言开销 + HF 的 `BatchEncoding` 封装层在大规模 encode（百万条以上文本）下吞吐量受限；mate 模型预训练 tokenize 阶段耗时占比可达 30%+。
2. **社区已有更优解**：`gigatoken`（Rust 实现，~1000× 快于 HF tokenizers，drop-in 兼容 HF tokenizer）已在社区成熟，**直接 `gt.Tokenizer(hf_tokenizer).as_hf()` 即可作为 HF tokenizer 的 drop-in replacement**，且 `encode_batch` 性能远超 HF。项目未集成 gigatoken，违背"优先复用已有优秀库，不自研"的宗旨。
3. **VerseTokenizer 与 gigatoken 接口差异**：gigatoken 原生 API 与 `BaseTokenizer` 抽象（`encode` / `decode` / `encode_batch` / `decode_batch` / `save` / `load` / `__len__` / `apply_chat_template`）不完全一致，需要 wrapper 适配；同时不能强制用户安装 gigatoken（破坏 ADR-001 的"零重型依赖"原则）。

同时必须保持向后兼容：现有 `load_tokenizer(kind="byte"|"bpe"|"hf")` 调用行为不变；`VerseTokenizer` 不删除（用户既有代码与 checkpoint 不失效）；不强制安装 gigatoken（CPU 端侧 / 树莓派场景仍可纯标准库运行）。

## 决策

**新增 `GigaTokenizerWrapper(BaseTokenizer)` 适配器（lazy import gigatoken），通过 `load_tokenizer(kind="giga")` 路径暴露；`spark/small` / `spark/mate` 配置默认 `tokenizer.kind="giga"`；gigatoken 不可用时自动降级到 `VerseTokenizer`（打印警告，不报错）；`pyproject.toml` 的 `verse-tokenizer` extras 新增 `[giga]` 可选依赖。**

具体含义：

1. **`GigaTokenizerWrapper` 适配器**（`verse_infra/verse_tokenizer/giga.py`）：

   - 继承 `BaseTokenizer`，实现 `encode` / `decode` / `encode_batch` / `decode_batch` / `save` / `load` / `__len__` / `apply_chat_template` 接口（对齐 `VerseTokenizer`）。
   - **lazy import gigatoken**：模块 import 不触发 `gigatoken` 加载，仅构造 `GigaTokenizerWrapper` 时 `import gigatoken as gt`；不可用时抛 `ImportError` 含安装提示（`pip install gigatoken`）。
   - **兼容模式（默认）**：内部用 `gt.Tokenizer(hf_tokenizer).as_hf()` 把 HF tokenizer 包装为 gigatoken 兼容模式（drop-in replacement），保证与 `VerseTokenizer` 输出**逐 token 一致**（同一 `model_id` 下 `encode("hello")` 结果完全相同）。
   - **原生模式（可选）**：`native=True` 走 `gt.Tokenizer(model_id)` 原生 API（更快，但需要单独训练/加载，与 HF tokenizer 行为可能有差异，仅适合高级用户）。
   - **缓存**：构造时一次解析 `bos_id` / `eos_id` / `pad_id` / `vocab_size`（避免每次 `__len__` 都查底层）；`vocab` 属性懒加载（首次访问时构建 `{token: id}` 反向表）。
   - **`apply_chat_template` 委托**：委托底层 HF tokenizer 的 `apply_chat_template`（gigatoken 兼容模式下保留 HF 接口，与 ADR-010 jinja2 模板系统对齐）。

2. **`load_tokenizer(kind="giga")` 分支 + 自动降级**：

   - 优先尝试 `GigaTokenizerWrapper(model_id=...)`
   - gigatoken 不可用时（`ImportError`）**自动降级**到 `VerseTokenizer`（即 `BPETokenizer.from_pretrained`），打印警告日志：
     ```
     [load_tokenizer] gigatoken 未安装，降级到 VerseTokenizer（性能较慢）。
     安装 gigatoken 可获得 ≥10× 批量 encode 加速：pip install gigatoken
     ```
   - 不抛异常，不阻塞训练（与 ADR-001 "无重型依赖下 CPU 路径完全不变"一致）。

3. **默认配置切换**：

   - `spark/small/config/cometspark_small.yml`：`tokenizer.kind` 默认从 `"verse"` 改为 `"giga"`
   - `spark/mate/config/cometspark_mate.yml`：同步修改
   - `tokenizer.repo` 保留（仍为 `Qwen/Qwen3.5-35B-A3B`，gigatoken 兼容模式下复用 HF tokenizer 加载）

4. **可选依赖声明**：

   `packages/verse_infra/pyproject.toml` 的 `verse-tokenizer` extras 新增 `[giga]`：

   ```toml
   [project.optional-dependencies]
   giga = ["gigatoken >= 0.1.0"]
   ```

   不强制安装（`pip install verse-infra` 不会拉取 gigatoken）；用户按需 `pip install "verse-infra[giga]"`。

5. **`verse_tokenizer/__init__.py` 导出**：

   ```python
   from .giga import GigaTokenizerWrapper
   __all__ = [..., "GigaTokenizerWrapper"]
   ```

   用户可直接 `from verse_infra.verse_tokenizer import GigaTokenizerWrapper`。

6. **不复刻原则（核心约束）**：

   - **不修改 gigatoken 源码**：仅做 `BaseTokenizer` 接口适配 wrapper
   - **不重新实现 BPE/Unigram**：gigatoken 兼容模式下复用其 Rust 实现
   - **不删除 `VerseTokenizer` / `BPETokenizer` / `ByteTokenizer`**：保留向后兼容，仅在默认路径上让位给 `GigaTokenizerWrapper`
   - **不强制依赖 gigatoken**：CPU 端侧 / 树莓派 / CI 环境仍可纯标准库运行（自动降级）

## 后果

### 优点

- **批量 encode 加速 ≥10×**：gigatoken Rust 实现在百万条文本 `encode_batch` 下吞吐量远超 HF tokenizers，mate 模型预训练 tokenize 阶段从 30% 占比降到 < 5%。
- **零行为差异（兼容模式）**：`gt.Tokenizer(hf_tokenizer).as_hf()` 是 HF tokenizer 的 drop-in replacement，`encode("hello")` 与 `VerseTokenizer` 逐 token 一致，既有 checkpoint 与训练数据缓存（`CachedDataset` 的 `.npz` 缓存）无需重新生成。
- **零重型依赖**：gigatoken 为可选依赖，`pip install verse-infra` 不拉取；CPU 端侧 / 树莓派 / CI 环境自动降级到 `VerseTokenizer`，行为完全一致（仅性能较慢）。
- **默认路径获得加速**：`spark/small` / `spark/mate` 配置默认 `kind="giga"`，新用户开箱即用获得 gigatoken 加速；老用户配置 `kind="verse"` 仍可回退。
- **`VerseTokenizer` 保留向后兼容**：既有代码 `load_tokenizer(kind="bpe")` / `kind="hf"` 行为不变，`VerseTokenizer` 不删除，仅默认路径让位。
- **接口统一**：`GigaTokenizerWrapper` 实现 `BaseTokenizer` 全部接口，与 `VerseTokenizer` / `BPETokenizer` 可互换；`apply_chat_template` 委托 HF 接口，与 ADR-010 jinja2 模板系统对齐。

### 缺点

- **gigatoken 为可选依赖**：用户需显式 `pip install gigatoken`（或 `pip install "verse-infra[giga]"`）才能获得加速；未安装时自动降级，但性能较慢（与 `VerseTokenizer` 一致）。
- **`GigaTokenizerWrapper` 引入间接层**：相比直接用 `gigatoken.Tokenizer`，wrapper 多一层 Python 调用开销（单次 encode 微秒级，可忽略；批量 encode 时 gigatoken Rust 路径主导，开销不变）。
- **`vocab` 属性懒加载**：首次访问 `wrapper.vocab` 时构建反向表（`{token: id}`），O(vocab_size) 内存与时间开销；不访问则不构建（与 `VerseTokenizer` 行为一致）。
- **`native=True` 路径与 HF 行为可能差异**：原生 API 走 `gt.Tokenizer(model_id)` 独立加载，可能与 HF tokenizer 在特殊 token / 添加 token / 编码边缘 case 上有差异；仅适合高级用户，默认 `native=False` 兼容模式。
- **配置默认值变更**：`spark/small` / `spark/mate` 的 `tokenizer.kind` 从 `"verse"` 改为 `"giga"`，老用户若显式覆盖配置则行为不变，未覆盖则切换到 gigatoken 路径（输出一致，性能提升）。

### 风险与缓解

| 风险 | 缓解策略 |
|---|---|
| gigatoken 版本与 HF tokenizer 不兼容（输出 token id 不一致） | 兼容模式 `gt.Tokenizer(hf_tokenizer).as_hf()` 直接复用 HF tokenizer 的词表与 merge 表，输出严格一致；`tests/test_giga_tokenizer.py` 断言 `encode` / `decode` 与 `VerseTokenizer` 逐 token 一致 |
| gigatoken 未安装时降级路径性能差 | 降级时打印明确警告 + 安装提示；用户知晓后可一键 `pip install gigatoken` 升级；默认配置在 CI / 端侧环境自动走降级路径，不阻塞训练 |
| `apply_chat_template` 在 gigatoken 兼容模式下行为不一致 | 委托底层 HF tokenizer 的 `apply_chat_template`（gigatoken 兼容模式保留 HF 接口），与 ADR-010 jinja2 模板系统完全对齐；测试覆盖工具调用渲染 |
| gigatoken 升级破坏 wrapper 兼容性 | wrapper 仅用 gigatoken 稳定 API（`gt.Tokenizer` / `.as_hf()` / `.encode` / `.decode` / `.encode_batch` / `.decode_batch`）；CI 矩阵覆盖 `gigatoken>=0.1.0` 多个版本 |
| 用户误用 `native=True` 导致输出不一致 | `native=True` 在 docstring 明确警告"可能与 HF tokenizer 行为有差异，仅适合高级用户"；默认 `native=False` 兼容模式 |
| `spark/small` / `spark/mate` 配置默认值变更影响既有训练 | 既有 `CachedDataset` 的 `.npz` 缓存基于 token id 序列，gigatoken 兼容模式输出与 `VerseTokenizer` 严格一致，缓存无需重新生成；测试 `test_tokenizer_upgrade.py` 覆盖默认路径切换 |

## 替代方案（已否决）

### 方案 A：自研 BPE Rust 实现

**描述**：用 Rust + PyO3 自研一个 BPE tokenizer 库，性能对标 gigatoken，完全可控。

**否决理由**：
- 维护成本极高（Rust + PyO3 跨语言绑定 + BPE 算法 + 词表加载 + 特殊 token 处理 + 与 HF tokenizer 互操作）
- 性能未必及 gigatoken（gigatoken 已经过社区大规模验证与优化）
- 违反 ADR-001 "不重新发明底层工具"原则
- 与"优先复用已有优秀库，不自研"宗旨冲突

### 方案 B：强制依赖 gigatoken（删除 `VerseTokenizer`）

**描述**：把 gigatoken 设为硬依赖，删除 `VerseTokenizer` / `BPETokenizer` / `ByteTokenizer`，统一走 gigatoken。

**否决理由**：
- 违反 ADR-001 "零重型依赖"原则（CPU 端侧 / 树莓派 / CI 环境无法安装 gigatoken）
- 破坏向后兼容（既有代码 `load_tokenizer(kind="bpe")` 失效）
- `ByteTokenizer` 是零依赖教学级 tokenizer，删除会破坏 ADR-001 的"开箱即用"承诺
- gigatoken 自身依赖 HF tokenizer（兼容模式），删除 `VerseTokenizer` 后无降级路径

### 方案 C：删除 `VerseTokenizer`，仅保留 `GigaTokenizerWrapper`

**描述**：`VerseTokenizer` 不删除但不再维护，新代码一律用 `GigaTokenizerWrapper`。

**否决理由**：
- `VerseTokenizer` 是 `BPETokenizer.from_pretrained` 的核心路径，删除会破坏 `load_tokenizer(kind="bpe"|"hf")` 既有调用
- gigatoken 不可用时无降级路径（`GigaTokenizerWrapper` 构造抛 `ImportError`）
- 与"自动降级"设计冲突（降级目标就是 `VerseTokenizer`）

### 方案 D：在 `BPETokenizer` 内部 lazy import gigatoken（不引入 wrapper）

**描述**：`BPETokenizer.from_pretrained` 内部检测 gigatoken 可用，可用则走 gigatoken 路径，不可用走 HF 路径。

**否决理由**：
- `BPETokenizer` 职责膨胀（既是 BPE 训练器又是 HF 加载器又是 gigatoken 包装器），违反单一职责
- 用户无法显式选择 gigatoken vs HF 路径（`kind="bpe"` 行为不稳定，依赖环境）
- `GigaTokenizerWrapper` 独立类便于扩展（如 `native=True` 原生模式）
- 测试隔离困难（gigatoken 可用 / 不可用两条路径在同一类内，测试需 mock）

### 方案 E：用 HuggingFace `tokenizers` 库直接加速（不走 gigatoken）

**描述**：直接用 HF `tokenizers.Tokenizer.from_pretrained` 而非 `AutoTokenizer`，绕过 `BatchEncoding` 封装层。

**否决理由**：
- 性能提升有限（HF `tokenizers` 已是 Rust 实现，主要开销在 Python ↔ Rust 跨语言调用，gigatoken 通过更激进的批处理 + zero-copy 优化）
- 失去 `AutoTokenizer` 的 `apply_chat_template` / 特殊 token 处理 / `add_bos` / `add_eos` 等高级接口
- 与社区生态脱节（gigatoken 是事实标准的加速方案）

## 备注

- 本 ADR 是 ADR-001 "零重型依赖"原则在 tokenizer 加速场景的具体落地：gigatoken 为可选依赖，自动降级保证 CPU 端侧可用
- `GigaTokenizerWrapper` **仅做接口适配**，不修改 gigatoken 源码，不重新实现 BPE/Unigram（与 ADR-009 "不重新实现 pickle 序列化"约束同源）
- 兼容模式（`native=False`）下 `encode` / `decode` / `encode_batch` / `decode_batch` / `apply_chat_template` 与 `VerseTokenizer` 输出**严格一致**，既有 `CachedDataset` 的 `.npz` 缓存无需重新生成
- 相关测试：`tests/test_giga_tokenizer.py` 覆盖 encode/decode 与 `VerseTokenizer` 一致 + 批量 encode 加速（≥10×）+ 自动降级 + `apply_chat_template`；现有 `tests/test_verse_tokenizer.py` / `test_tokenizer_nex_wrapper.py` / `test_tokenizer_upgrade.py` 零回归
- 相关代码：
  - [`verse_infra/verse_tokenizer/giga.py`](../../packages/verse_infra/verse_infra/verse_tokenizer/giga.py) —— `GigaTokenizerWrapper`（lazy import gigatoken + 兼容模式 + 原生模式）
  - [`verse_infra/verse_tokenizer/__init__.py`](../../packages/verse_infra/verse_infra/verse_tokenizer/__init__.py) —— 导出 `GigaTokenizerWrapper`
  - [`verse_infra/verse_tokenizer/bpe.py`](../../packages/verse_infra/verse_infra/verse_tokenizer/bpe.py) —— `load_tokenizer(kind="giga")` 分支 + 自动降级
  - [`spark/small/config/cometspark_small.yml`](../../spark/small/config/cometspark_small.yml) —— `tokenizer.kind` 默认 `"giga"`
  - [`spark/mate/config/cometspark_mate.yml`](../../spark/mate/config/cometspark_mate.yml) —— 同步
  - [`packages/verse_infra/pyproject.toml`](../../packages/verse_infra/pyproject.toml) —— `[giga]` extras 声明

## 演进路线

- **`native=True` 路径增强**：未来可针对 `Qwen3.5-35B-A3B` 等 Verse 主力 tokenizer 预训练 gigatoken 原生模型，进一步加速（性能预期再提升 2-3×）
- **`GigaTokenizerWrapper` 扩展接口**：若 gigatoken 后续暴露 `train` / `merge` 等高级 API，wrapper 可透传
- **多 tokenizer 加速方案对比**：未来可对比 `gigatoken` / `hf tokenizers` / `sentencepiece` 在 Verse 主力语料上的性能，动态选择最优路径
