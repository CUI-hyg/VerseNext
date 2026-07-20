# 示例：纯 CPU 流式推理 demo（verse_inference）

> 对应脚本：[`cpu_inference_demo.py`](file:///workspace/examples/cpu_inference_demo.py)

## 目标

在纯 CPU 上构建一个小型 Mamba-2 语言模型，用 `verse_inference` 的 `ModelLoader + StreamingGenerator + Sampler` 流式生成 100 个 token，并报告：

- 模型参数量；
- 生成时间与吞吐量（tokens/s）；
- 峰值 RSS（MB）。

约束：
- 4 核 CPU 上 5 分钟内完成；
- 峰值 RSS ≤ 8 GB；
- 模型参数量 < 50M。

## 模型架构

通过 `verse_inference.ModelLoader` 构建一个 `arch="mamba2"` 的 `HybridLM`：

```
CharTokenizer vocab (256)
   │
   ▼
Embedding(256, 128)
   │
   ▼
4 × Mamba2Block(dim=128, n_heads=4, d_state=64, d_conv=4, expand=2)
   │
   ▼
LayerNorm
   │
   ▼
LM Head (Linear, no tie)
   │
   ▼
logits (B, 1, 256)
```

参数量约 0.6M（见 [构建 LM 段](file:///workspace/examples/cpu_inference_demo.py#L119-L135)）。架构默认配置见 [`_DEFAULT_ARCH_CONFIGS`](file:///workspace/packages/verse_inference/verse_inference/model_loader.py#L47-L72)。

## 推理流程

```
prompt 文本
   │
   ▼
CharTokenizer.encode → prompt_ids
   │
   ▼
StreamingGenerator.generate(prompt_ids, max_new_tokens=100)
   │  ┌─────────────────────────────────────────┐
   │  │ 1. Prefill: 逐 token 走 forward_recurrent│
   │  │    维护每层 SSM 状态 + conv_state         │
   │  │ 2. Decode: 每步前向得到 logits            │
   │  │    → Sampler(temperature, top_k, top_p)   │
   │  │    → yield next token id                   │
   │  └─────────────────────────────────────────┘
   ▼
逐 token 流式输出文本
```

关键点：
- **prefill + decode 都用 recurrent 模式**：每步 O(1) 内存，不保存 KV cache；
- **StreamingGenerator 内部维护 states 列表**：每层一个 state，跨 token 持续更新；
- **Sampler 支持 greedy / top-k / top-p / temperature**：默认 temperature=0.8, top_k=20, top_p=0.95。

实现入口见 [main 函数中流式生成段](file:///workspace/examples/cpu_inference_demo.py#L164-L184)。

## 运行方式

```bash
cd /workspace
PYTHONPATH=packages/verse_torch:packages/verse_nex:packages/verse_compat:packages/verse_tokenizer:packages/verse_inference \
    python3 examples/cpu_inference_demo.py
```

可选参数：

```bash
python3 examples/cpu_inference_demo.py \
    --arch mamba2 \
    --vocab-size 256 \
    --dim 128 \
    --n-layers 4 \
    --d-state 64 \
    --n-heads 4 \
    --max-new-tokens 100 \
    --prompt "Hello Mamba" \
    --temperature 0.8 \
    --top-k 20 \
    --top-p 0.95 \
    --seed 42
```

完整参数定义见 [argparse 段](file:///workspace/examples/cpu_inference_demo.py#L76-L93)。

## 预期结果

实测输出：

```
========================================================================
VerseInference 端到端 CPU 推理示例
========================================================================
架构:           mamba2
vocab_size:     256
dim:            128
n_layers:       4
max_new_tokens: 100
temperature:    0.8
top_k:          20
top_p:          0.95
CPU 核数:       4

[1/4] 构建 Mamba-2 LM ...
    参数量:      594,944 (0.59M)
    构建时间:    0.05s

[2/4] 准备 tokenizer ...
    Tokenizer:   CharTokenizer (vocab=256, pre-populated)
    Prompt:      'Hello Mamba'

[3/4] 流式生成 ...
    Hello Mamba<生成内容...>

[4/4] 报告:
    生成时间:    0.14s
    生成 token:  100
    吞吐量:      715.20 tokens/s
    峰值 RSS:    44.5 MB
    完整文本:    'Hello Mamba...'

约束检查:
    [OK] 生成时间 < 300s  (实际 0.1s)
    [OK] 峰值 RSS < 8192MB (实际 44.5MB)
    [OK] 参数量 < 50M    (实际 0.59M)

所有约束通过！
```

关键指标：
- **0.6M 参数量**，远低于 50M 上限；
- **0.14 秒生成 100 tokens**，吞吐量 **715 tokens/s**；
- **峰值 RSS 44.5 MB**，远低于 8 GB 上限；
- 所有约束通过。

## 关键代码引用

### 1. ModelLoader 构建模型

```python
# 见 cpu_inference_demo.py 第 121-130 行
loader = ModelLoader(
    arch=args.arch,
    vocab_size=args.vocab_size,
    dim=args.dim,
    n_layers=args.n_layers,
    ssm_kwargs={"d_state": args.d_state, "d_conv": 4, "expand": 2, "n_heads": args.n_heads},
    sparse_kwargs={"n_heads": args.n_heads, "chunk_size": 16, ...},
)
model = loader.load()
```

`ModelLoader` 内部用 `verse_nex.HybridLM` 构建模型，可选用 `verse_compat.load_hf_state_dict` 加载预训练权重覆盖（见 [`model_loader.py`](file:///workspace/packages/verse_inference/verse_inference/model_loader.py)）。

### 2. Sampler 配置采样策略

```python
# 见 cpu_inference_demo.py 第 165-170 行
sampler = Sampler(
    temperature=args.temperature,
    top_k=args.top_k,
    top_p=args.top_p,
    seed=args.seed,
)
```

支持 greedy / temperature / top-k / top-p 任意组合，与 OpenAI API 语义一致。

### 3. StreamingGenerator 流式生成

```python
# 见 cpu_inference_demo.py 第 171-184 行
gen = StreamingGenerator(model, tokenizer=tokenizer, sampler=sampler)
for tok_id in gen.generate(prompt_ids, max_new_tokens=args.max_new_tokens):
    generated_ids.append(tok_id)
    piece = tokenizer.decode([tok_id])
    print(piece, end="", flush=True)
```

`generate()` 是一个 Python generator，每步 yield 一个 token id，便于上层打印或推送给客户端（如 HTTP 流式响应）。

### 4. 峰值 RSS 测量

```python
# 见 cpu_inference_demo.py 第 44-52 行
def _peak_rss_mb() -> float:
    ru = resource.getrusage(resource.RUSAGE_SELF)
    return ru.ru_maxrss / 1024.0  # Linux: KB → MB
```

用 `resource.getrusage` 读取进程峰值 RSS，Linux 上单位是 KB（macOS 是 bytes，本 demo 仅支持 Linux）。

## 注意事项

1. **CharTokenizer 预填充**：CharTokenizer 默认是懒加载的（只对 prompt 中出现的字符建词表），但模型可能生成 prompt 之外的字符 id。所以本 demo 预填充 0..255 全部字节值到词表，确保任意 id 都能 decode（见 [tokenizer 预填充段](file:///workspace/examples/cpu_inference_demo.py#L139-L150)）。
2. **未加载预训练权重**：默认构建一个随机初始化的 LM，生成的文本是随机字符。要生成有意义的文本需要：
   - 用 `verse_compat.load_hf_state_dict` 从 HF Hub 加载预训练权重；
   - 或自己训练一个 LM（参考 `examples/minimal_lm.py`）后保存 state_dict 再加载。
3. **recurrent 模式 vs parallel 模式**：
   - recurrent：每步 O(1) 内存，适合长序列生成；
   - parallel：每步重算整个序列，O(T) 内存，但便于复用 Transformer 风格的代码。
   - 本 demo 用 recurrent（默认），符合 spec 要求的"流式生成"。
4. **CPU 核数检测**：脚本用 `os.cpu_count()` 报告，但 NumPy 内部并行度由环境变量 `OMP_NUM_THREADS` / `MKL_NUM_THREADS` 决定，未设置时默认使用所有核。
5. **吞吐量受 dim/d_state 影响**：dim=128, d_state=64 是 CPU 友好的小模型；如需更高吞吐量，可减小 dim 到 64 或减少层数到 2；如需更大模型，参考 `--dim 256 --n-layers 8`，但 RSS 与延迟会相应上升。
6. **prompt 长度不影响吞吐量**：因为 prefill 也是用 recurrent 模式逐 token 处理，所以 1 token prompt 与 1000 token prompt 的 decode 阶段吞吐量基本一致（prefill 阶段除外）。这是 Mamba-2 相对 Transformer 的核心优势——Transformer 的 KV cache 会随 prompt 长度线性增长，而 SSM 的状态保持恒定。
7. **不支持 batch 推理**：当前 `StreamingGenerator` 仅支持 batch=1；多请求并发需要在外部起多个实例或扩展为 batch 推理（后续 spec）。
