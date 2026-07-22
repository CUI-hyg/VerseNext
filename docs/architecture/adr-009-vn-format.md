# ADR-009: .vn 文件格式

- **状态**：Accepted
- **日期**：2026-07-22
- **决策者**：Verse 框架作者（CometFuture / CUI-hyg）
- **相关规范**：[`/workspace/.trae/specs/part4k2-arch-model-upgrade/spec.md`](../../../.trae/specs/part4k2-arch-model-upgrade/spec.md)
- **前置 ADR**：[ADR-001 CPU 优先](adr-001-cpu-first.md)、[ADR-005 GPU/NPU 后端](adr-005-gpu-npu-backend.md)
- **相关 ADR**：[ADR-011 智能分区训练](adr-011-layerwise-training.md)（.vn 分片卸载复用本格式）、[ADR-010 jinja2 聊天模板](adr-010-jinja2-chat-template.md)（chat_template.jinja 内嵌于 .vn）

## 上下文

Part4K2 之前，Verse 仓库的模型权重交付格式为基于 `pickle` 的 `.pt` 文件（`CometSparkV05LM.save` 写出 `{"arch", "config", "state_dict"}` payload）。这种格式存在以下问题：

1. **安全性风险**：`pickle.load` 可执行任意代码，加载不可信 `.pt` 文件存在 RCE（远程代码执行）风险。
2. **加载性能差**：`pickle` 反序列化需要一次性把整个 `state_dict` 读入内存，无 mmap 零拷贝能力；大模型加载耗时长、峰值内存高。
3. **非自描述**：`.pt` payload 的 `config` 字段结构因模型而异，`arch` 字段可选；缺乏标准化的元数据（格式版本、权重格式、压缩信息、创建时间）。
4. **缺乏配套资产**：模型交付时聊天模板（`chat_template.jinja`）、tokenizer（`tokenizer.json`）需要单独分发，与权重文件分离易导致版本错配。
5. **与生态对接困难**：HuggingFace 生态以 `safetensors` 为事实标准（pickle-free + mmap），Verse 的 `.pt` 格式难以被 `transformers` / `ggml` 等工具直接读取。

同时必须保持向后兼容：现有用户的 `.pt` checkpoint 不能突然失效，`save` / `load` / `from_pretrained` 等接口签名需平滑过渡。

## 决策

**定义 `.vn` 文件格式：基于 `zipfile` 的 ZIP 容器，内含 `model.safetensors`（或降级 `model.npz`）+ `config.yml` + `chat_template.jinja`（可选）+ `tokenizer.json`（可选）+ `meta.json`；提供 `VNFileReader` / `VNFileWriter` / `pt_to_vn` / `vn_to_pt` / `convert_format` API，以及 CLI `verse-convert`。**

具体含义：

1. **ZIP 容器结构**：
   ```
   model.vn (ZIP)
   ├── model.safetensors   # 权重（safetensors 可用时）
   ├── model.npz           # 权重（safetensors 不可用时降级）
   ├── config.yml          # 模型配置（YAML，优先 PyYAML，否则 JSON 兼容子集）
   ├── chat_template.jinja # 聊天模板（可选）
   ├── tokenizer.json      # tokenizer（可选）
   └── meta.json           # 元数据
   ```

2. **`meta.json` 自描述**：
   ```json
   {
       "vn_format_version": 1,
       "arch": "versenex",
       "weight_format": "safetensors" | "npz",
       "compression_info": {...} | null,
       "created_at": "ISO8601 时间戳",
       "weight_count": 12
   }
   ```
   读取时校验 `vn_format_version`，仅支持版本 1。

3. **safetensors 优先 + npz 降级**：
   - safetensors 可用时存为 `model.safetensors`，读取通过 `safe_open` 的 mmap 零拷贝（`f.get_tensor(key)` 按需映射）。
   - safetensors 不可用时降级为 `model.npz`（手工构造 ZIP(npz) 以支持 `blocks.0.attn.q.weight` 这类带点号的参数名，`np.savez` 仅接受合法标识符无法承载）。
   - npz 路径强制 `allow_pickle=False`，杜绝 pickle 反序列化攻击。

4. **`VNFileWriter` / `VNFileReader` API**：
   - `VNFileWriter(path, arch, config, compression_info=None)`：上下文管理器，`write_weights` / `write_chat_template` / `write_tokenizer` / `close`。
   - `VNFileReader(path)`：上下文管理器，`read_meta` / `read_config` / `read_weights(mmap=True)` / `read_chat_template` / `read_tokenizer` / `close`。
   - 写入时若 `state_dict` 数组携带 `quant_info` 属性，自动收集到 `meta.json` 的 `compression_info` 字段（支持量化模型交付）。

5. **无损互转**：
   - `pt_to_vn(pt_path, vn_path, arch=None, config=None, chat_template=None, tokenizer=None)`：`.pt` → `.vn`，权重数值完全一致。
   - `vn_to_pt(vn_path, pt_path)`：`.vn` → `.pt`，输出 payload 结构与 `CometSparkV05LM.save` 一致。
   - `convert_format(src_path, dst_path)`：自动检测后缀互转。

6. **CLI `verse-convert`**：`verse-convert --input model.pt --output model.vn --chat-template chat_template.jinja --tokenizer tokenizer.json --arch versenex`。

7. **向后兼容**：`.pt` 格式继续保留（`save` / `load` 不变）；`.vn` 作为推荐交付格式，新旧格式可通过 `verse-convert` 无损互转。

## 后果

### 优点

- **安全性**：safetensors 本身 pickle-free；npz 路径 `allow_pickle=False`；加载不可信 `.vn` 文件无 RCE 风险。
- **加载性能**：safetensors 路径 mmap 零拷贝，大模型加载时间从"全量反序列化"降为"按需映射"，峰值内存显著下降。
- **自描述**：`meta.json` 记录格式版本 / 架构 / 权重格式 / 压缩信息 / 创建时间，工具链可据此路由加载逻辑。
- **配套资产内嵌**：`chat_template.jinja` + `tokenizer.json` 与权重同包分发，避免版本错配。
- **生态对接**：safetensors 是 HuggingFace 事实标准，`.vn` 可被 `transformers` 工具链间接读取（解包后取 `model.safetensors`）。
- **优雅降级**：无 safetensors 时自动降级 npz，纯标准库 + numpy 即可工作，符合 ADR-001 的"零重型依赖"原则。
- **无损往返**：`.pt ↔ .vn` 互转权重数值完全一致，迁移零风险。

### 缺点

- **新增依赖**：safetensors / PyYAML 为可选依赖（已在 `pyproject.toml` 声明）；npz / JSON 路径无需额外依赖。
- **文件体积**：ZIP 容器相比裸 `.pt` 多一层封装，但 `ZIP_DEFLATED` 对 config/meta 有压缩，权重本身已是二进制影响极小。
- **临时文件**：safetensors 的 `safe_open` 需要文件路径，读取时需把 `model.safetensors` 从 ZIP 解包到临时文件（`tempfile.mkstemp`），`VNFileReader.close` 时清理。

### 风险与缓解

| 风险 | 缓解策略 |
|---|---|
| 用户仍用 `.pt` 不迁移 | `.pt` 继续工作；文档推荐 `.vn`；`verse-convert` 一键迁移 |
| safetensors 版本不兼容 | `try/except` 探测，不可用时自动降级 npz |
| 临时文件泄漏 | `VNFileReader.close` / `__del__` / `__exit__` 三重清理；上下文管理器推荐用法 |
| `meta.json` 格式演进 | `vn_format_version` 字段版本化，读取时校验；未来 v2 可向后兼容 v1 |

## 替代方案（已否决）

### 方案 A：直接采用 HuggingFace `safetensors` 单文件格式

**描述**：不设计容器，直接用 `model.safetensors` + 旁路的 `config.json` / `tokenizer.json`。

**否决理由**：多文件分发易丢失；缺乏统一元数据；与 Verse 的 `config.yml`（YAML）风格不一致；无法内嵌 `chat_template.jinja`。

### 方案 B：继续使用 `.pt`（pickle）+ 旁路安全清单

**描述**：保留 pickle，通过 `pickle.Unpickler.find_class` 白名单限制可加载类。

**否决理由**：白名单维护成本高；mmap 零拷贝无法实现；性能问题不解决；与生态对接仍困难。

### 方案 C：自定义二进制格式（类似 ggml）

**描述**：自研二进制格式（magic number + header + tensor table）。

**否决理由**：开发维护成本高；失去 ZIP 容器的工具链生态（`unzip` / 文件管理器可直接查看）；与 ADR-001"不重新发明底层工具"原则不符。

## 备注

- 本 ADR 是 Part4K2 "模型交付格式标准化"的核心决策。
- `VN_FORMAT_VERSION = 1`，未来格式演进时递增版本号并保持向后兼容。
- 相关测试：`tests/test_vn_format.py` 覆盖写入/读取/互转/降级/量化元数据。
- 相关文档：[Verse 训练指南 - .vn 格式使用指南](../training_guide.md)
