# CometSpark PoC 训练数据

## 数据格式（新格式 v2）

每行一个 JSON 对象，支持 **两种格式混用**：

### 1. chat 数组格式

```json
[{"role":"user","content":"你好"},{"role":"assistant","content":"你好，很高兴见到你。"}]
```

- 渲染为 `<|user|>你好<|assistant|>你好，很高兴见到你。<|eos|>`
- **loss mask**：仅 assistant content + `<|eos|>` 参与 loss，user 部分屏蔽
- 支持多轮对话：`[{"role":"user",...},{"role":"assistant",...},{"role":"user",...},{"role":"assistant",...}]`

### 2. prompt-completion 格式

```json
{"prompt":"床前明月光，","completion":"疑是地上霜。举头望明月，低头思故乡。"}
```

- 渲染为 `<|user|>床前明月光，<|assistant|>疑是地上霜。举头望明月，低头思故乡。<|eos|>`
- **loss mask**：仅 completion + `<|eos|>` 参与 loss，prompt 部分屏蔽

### 3. 旧格式（已废弃）

```json
{"text": "床前明月光，疑是地上霜。"}
```

- **已废弃**：`TextDataset` 加载时会抛 `ValueError`
- 如需迁移，把 `text` 拆分为 `prompt` + `completion`，或转为 chat 数组

## 文件清单

| 文件 | 行数 | 用途 |
|------|------|------|
| `train.jsonl` | 127 | 训练集 |
| `val.jsonl`   | 32  | 验证集 |

## 数据来源

合成数据，3 类混合：

1. **唐诗（5-7 言绝句）**：prompt-completion 格式，prompt 为首句，completion 为后续诗句
2. **简单问答**：chat 数组格式，含你好/再见/自我介绍/知识问答等
3. **数字序列**：prompt-completion 格式，prompt 为前几项，completion 为后续等差数列

设计目标：让 LM 学到 chat 模板结构 + 语言模式（诗句续写、问答对应、数字递增）。

## 替换为真实数据

直接覆盖 `train.jsonl` / `val.jsonl`，保持每行为 chat 数组或 prompt-completion 格式即可。
建议训练集 ≥ 100 条、验证集 ≥ 20 条。
