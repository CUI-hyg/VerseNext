# VerseInference

> 中文定位：模型加载、状态缓存、流式生成；支持 Mamba / RWKV / Transformer / CometSpark 多种架构，可选 OpenAI 兼容 HTTP server。

[返回主 README](../../README.md)

## 特性

- 多架构加载：`mamba2` / `rwkv7` / `hybrid` / `transformer` / `cometspark`。
- 状态缓存：`StateCache` 自动管理 Mamba / RWKV 递归状态。
- 采样器：`GreedySampler` / `Sampler`（temperature / top_k / top_p）。
- 流式生成：`StreamingGenerator` 逐步产生 token。
- CometSpark 兼容：`register_cometspark_path` + `arch="cometspark"` 动态加载。
- 可选 HTTP server：OpenAI 兼容（FastAPI，按需安装）。

## 安装

```bash
pip install -e packages/verse_inference
```

## 快速开始

### 加载 CometSpark 模型并生成

```python
from verse_inference import ModelLoader, GreedySampler, StreamingGenerator, StateCache
from verse_inference.model_loader import register_cometspark_path

# 1. 注册 CometSpark demo 路径（动态导入 CometSparkLM 类）
register_cometspark_path("/workspace/data/demo")

# 2. 加载已训练的 CometSpark checkpoint（.pt pickle）
loader = ModelLoader(arch="cometspark")
model = loader.load("/workspace/data/demo/checkpoints/cometspark.pt")

# 3. 流式生成
sampler = GreedySampler()
state_cache = StateCache()
gen = StreamingGenerator(model, sampler=sampler, state_cache=state_cache)

prompt = [256, 233, 150, 129]   # <bos> + "你" 的字节 id
for token in gen.stream(prompt, max_new_tokens=64):
    print("token:", token)
```

### 加载 Mamba-2 模型并流式生成

```python
from verse_inference import ModelLoader, GreedySampler, StreamingGenerator, StateCache

# 1. 自构建 mamba2 LM（CPU 友好，dim=128，n_layers=4）
loader = ModelLoader(arch="mamba2", vocab_size=259, dim=128, n_layers=4)
model = loader.load()                          # 随机初始化，eval + no_grad

# 2. 流式生成
gen = StreamingGenerator(model, sampler=GreedySampler(), state_cache=StateCache())
tokens = gen.generate(prompt=[10, 20, 30], max_new_tokens=32)
print("generated tokens:", tokens)
```

## API 详解

### ModelLoader

```python
ModelLoader(model_path=None, arch="mamba2", config=None, ...)
```

| `arch` | 行为 |
| --- | --- |
| `"mamba2"` | 构建纯 Mamba-2 LM（`HybridLM` with `ssm_kind="mamba2"`，`sparse_ratio=0`） |
| `"rwkv7"` | 构建纯 RWKV-7 LM |
| `"hybrid"` | 构建 Mamba-2 + Sparse Attention 混合 LM（`sparse_ratio=0.25`） |
| `"transformer"` | Transformer 风格 LM（由 `verse_nex` 提供） |
| `"cometspark"` | 从 `.pt` pickle 加载完整 `CometSparkLM`（含 config + state_dict） |

主要方法：

- `load(repo_or_path=None, strict=False)` → 返回 LM 实例（已 `eval` + `requires_grad=False`）。
  - `arch="cometspark"` 时 `repo_or_path` 必须指向 `.pt` 文件。
- `register_cometspark_path(demo_path)` 模块级函数：把 `data/demo` 路径注入 `sys.path`，便于动态导入 `CometSparkLM` 类。也可通过环境变量 `COMETSPARK_DEMO_PATH` 设置。

### StateCache

- 管理 Mamba / RWKV 的递归状态。
- 在 `StreamingGenerator.stream` 中按 session 自动缓存与更新。

### Sampler / GreedySampler

- `GreedySampler()`：贪心采样，每步 argmax。
- `Sampler(temperature=1.0, top_k=0, top_p=1.0)`：温度 + top-k + top-p 采样。

### StreamingGenerator

```python
StreamingGenerator(model, sampler, state_cache=None)
```

- `generate(prompt, max_new_tokens)` → `List[int]`：一次性生成。
- `stream(prompt, max_new_tokens=None)` → 迭代器：逐步 yield token，O(1) 内存。
- 自动调用 `model.forward_recurrent` 维护递归状态。

## 测试

- `tests/test_cometspark_inference.py`：16 项测试，覆盖 CometSpark 加载与生成（100 tokens ≤ 5s）。
- `tests/test_end_to_end.py`：端到端流程测试。

## 相关文档

- [CometSpark 训练仓库](../../data/demo/README.md)
- [CPU 推理 demo](../../examples/README_cpu_inference_demo.md)
