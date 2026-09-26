# 后训练与微调指南（SFT / LoRA / 对话生成）

VerseNext 提供完整的后训练链路：**分词器训练 → 对话 SFT（LoRA 或全参）→ 对话生成**。

## 1. 训练分词器

byte-level BPE，基础词表 256 字节（任意文本无 OOV），训练后自动支持
`<|...|>` 形式的特殊 token。

```bash
# 用自己的语料（多个 txt 文件）
verse train-tokenizer --input corpus1.txt corpus2.txt --vocab-size 4096 --out my_tokenizer/
```

产物为 HF 风格目录：`tokenizer.json` + `chat_template.jinja`（若提供 `--template`）。

## 2. chat_template.jinja

模板遵循 HF transformers 约定，用 jinja2 渲染：

```jinja
{%- for message in messages -%}
{{ '<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>\n' }}
{%- endfor -%}
{%- if add_generation_prompt -%}
{{ '<|im_start|>assistant\n' }}
{%- endif -%}
```

- 未提供模板文件时自动回退到内置 ChatML 模板。
- 模板中使用的 `<|...|>` 标记会在编码时自动注册为特殊 token，开箱即用。

Python API：

```python
from verse_tokenizer import BPETokenizer, ChatTemplate

tok = BPETokenizer.from_pretrained("my_tokenizer/")
tpl = ChatTemplate.from_pretrained("my_tokenizer/")   # 或 ChatTemplate.default()

ids, labels = tpl.encode_chat(
    [{"role": "user", "content": "你好"},
     {"role": "assistant", "content": "你好！有什么可以帮你？"}],
    tok, train_on_last_turns=1,
)
# labels 只在最后一轮 assistant 回答（含 <|im_end|>）处非 -100
```

掩码定位：将目标回答替换为唯一哨兵串重新渲染，得到精确字节区间，再与
token 偏移求交集——不依赖具体模板写法。

## 3. SFT 微调

数据格式（JSON/JSONL，也支持 CSV/TSV/Parquet）：

```json
[
  {"messages": [
    {"role": "system", "content": "You are helpful."},
    {"role": "user", "content": "hello"},
    {"role": "assistant", "content": "how can I help you today"}
  ]}
]
```

`SFTDataset.from_file(path, ...)` 按后缀自动识别 jsonl/json/csv/tsv/parquet，
并兼容 `messages` / `conversations` / `system+user+assistant` / JSON 字符串等
多种对话结构（见 `verse_trainer.sft.load_conversations`）。

命令行（LoRA 微调，`--lora-r 0` 切换全参微调）：

```bash
verse sft --data chats.json --tokenizer my_tokenizer/ \
    --d-model 256 --num-layers 4 --steps 500 \
    --lora-r 8 --save adapter.pt
```

Python API：

```python
from verse_nn import LoRAConfig, VerseTransformerConfig
from verse_trainer import SFTConfig, SFTDataset, Trainer, TrainerConfig
from verse_tokenizer import BPETokenizer, ChatTemplate

tok = BPETokenizer.from_pretrained("my_tokenizer/")
tpl = ChatTemplate.from_pretrained("my_tokenizer/")
ds = SFTDataset.from_json("chats.json", tok, tpl, SFTConfig(batch_size=4))

config = TrainerConfig(
    model=VerseTransformerConfig(vocab_size=tok.vocab_size, d_model=256, num_layers=4),
    lora=LoRAConfig(r=8, lora_alpha=16),   # None 即全参微调
    steps=500,
)
trainer = Trainer(config)
trainer.train(ds)                  # SFTDataset 直接作为数据源
trainer.save_adapter("adapter.pt") # 只存适配器（<1% 主干体积）
```

### LoRA 细节

- 默认作用于 `q_proj / kv_proj / o_proj`，可通过 `target_modules` 扩展到 FFN。
- `B` 初始化为 0：微调起点等价于原模型。
- `merge_lora(model)` 可把增量合并进主干后导出完整模型。
- SFT 只对 assistant 回答计算 loss（指令前缀掩码为 -100）。

## 4. 断点续训（full resume）

`Trainer` 的 checkpoint 保存**完整训练状态**（权重、优化器/缩放器、LR 调度进度、
RNG、监控与早停进度、累计 token 数、loss 历史、数据指纹、resolved 配置），
`load_checkpoint()` 可一键恢复——续训与不间断训练**逐位一致**，不丢失、不折损。

```bash
# 预训练中途崩溃后，从同阶段 final 自动续训（沿用原配置剩余步数）
python run.py train --resume

# 从指定 checkpoint 续训并新增 500 步
python run.py train --resume checkpoints/CometSpark-Exp-0.3_pretrain_final.pt --steps 500

# 换数据继续训练（需显式放行；优化器动量完整保留，不会二次 warmup）
python run.py train --resume --data data/new_corpus.jsonl --allow-data-change
```

```python
from verse_trainer import Trainer, TrainerConfig

trainer = Trainer(TrainerConfig.load("config.yaml"))
trainer.load_checkpoint("checkpoints/xxx_final.pt")   # 恢复全部续训状态
trainer.train(tokens, eval_tokens)                    # 从 global_step 继续
```

语义约定：

| 项 | 行为 |
|---|---|
| `--steps N` | 在断点基础上**新增 N 步**；不传则沿用原配置剩余步数 |
| LR 调度 | 延续原 warmup-cosine，**不重开 warmup**（避免 LR 回弹/二次升温造成质量折损） |
| `--horizon-steps` | 显式重设 cosine 衰减地平线（默认取 checkpoint 原值） |
| 数据/形状校验 | 数据内容、`seq_len`、`batch_size` 与 checkpoint 不一致时**报错**；`--allow-data-change` 放行 |
| 崩溃兜底 | `save_last: true` 时每个 `ckpt_every` 覆盖保存 `*_last.pt` |
| final 续训 | `final_keep_optimizer: true`（默认）时 final 保留优化器状态，可直接作为续训起点 |

> 注意：`save_best_only: true` 时周期产物只在 loss 创新低时保存；若需「随时可续训」
> 请同时开启 `save_last: true`。

## 5. 生成

```bash
# 对话生成（CPU int8 量化）
verse generate --model adapter.pt --tokenizer my_tokenizer/ \
    --chat --system "You are helpful." --prompt "hello" \
    --quantize --greedy
```

```python
from verse_trainer import GenerateConfig, Generator

gen = Generator.from_pretrained("adapter.pt", "my_tokenizer/",
                                GenerateConfig(quantize=True, temperature=0.7))
print(gen.chat([{"role": "user", "content": "hello"}]))
print(gen.complete("good morning"))
```

### 抑制乱码 / 复读

`GenerateConfig` 默认启用一组采样控制（可通过命令行或配置覆盖）：

| 参数 | 默认 | 作用 |
|---|---|---|
| `temperature` | 0.7 | 采样温度 |
| `top_k` / `top_p` | 50 / 0.9 | top-k 与核采样 |
| `repetition_penalty` | 1.1 | CTRL 式重复惩罚（>1 抑制复读） |
| `no_repeat_ngram` | 3 | 禁止重复出现的 n-gram |
| `greedy` | False | 贪心解码（忽略 temperature/top_k/top_p） |

此外 `generate()` 会把输入/输出 id 一律 clamp 到 `[0, vocab)`（越界 id 会让
embedding 取到错误行，是乱码的常见来源），并在候选被全部屏蔽时回退均匀分布
避免 NaN。

## 6. CPU 性能优化清单

| 优化 | 说明 | 收益 |
|---|---|---|
| KV 缓存解码 | prefill 一次，每步仅前向 1 token | 实测 ~2.3x（128 ctx + 32 生成） |
| int8 动态量化 | `Generator(quantize=True)`，Linear 权重 int8 | 大模型上 1.5-2x + 体积 1/4（小模型上开销抵消收益） |
| 线程数控制 | `num_threads`（默认按 affinity 核数 × `resources.cpu_fraction`，50%） | 训练/推理通用 |
| 资源上限 | `ResourceConfig`（CPU/GPU/内存默认各 50%） | 容器内避免抢占、稳定吞吐 |
| 分块 CE | `loss_chunk_size>0`，按序列维分块 lm_head+CE | 避免 `(B,S,vocab)` logits 峰值 |
| flush denormal | 关闭亚正规数慢路径 | 矩阵运算稳定提速 |
| torch.inference_mode | 生成全程关闭 autograd | 默认已启用 |
| torch.compile | `--compile`（长训推荐） | 视模型而定 |
