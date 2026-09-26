# VerseNext

高性能神经网络架构框架 + 下游实验模型，含底层算子与设备专项优化。

## 目录

| 路径 | 说明 |
|---|---|
| [`VerseNext/`](VerseNext/) | 框架本体（uv workspace，多包在 `src/`）：模型架构、分词、训练、RSI 自进化、CLI |
| [`CometSpark/`](CometSpark/) | 下游实验模型项目，复用 `verse_nn.transformer.VerseTransformer`（0.1 / 0.2 / 0.3，0.3 ≈ 0.6B） |
| [`HANDOFF.md`](HANDOFF.md) | 工作区交接文档：当前主线、已完成 / 待办、环境约束 |
| [`VerseNext/docs/kernels.md`](VerseNext/docs/kernels.md) | 算子与设备分层、构建、正确性契约、已知限制 |

## 底层算子与设备优化

- **CPU**：融合算子、线程池并行、int8 动态量化、bf16/fp16 混合精度推理。
- **CUDA / ROCm**：6 个自研前向 kernel（融合 RMSNorm / SwiGLU / RoPE /
  分块在线 softmax FlashAttention / 分块 CE / KDA chunkwise），ROCm 由 hipify 复用。
- **CANN NPU**：CANN 原生融合算子 + 自研 AscendC + ACLNN 适配层。
- **三级回退派发**（自研 → torch 内建 → 参考），绝不因算子问题中断训练。
- **资源锁**：CPU / CUDA / ROCm / NPU 默认 50% 占用上限。

## 快速开始

```bash
cd VerseNext
export PYTHONPATH=$(ls -d src/*/ | tr '\n' ':')   # 包未 pip 安装时
python -m pytest tests/ -q

# 编译自研算子（需对应硬件工具链）
python -m verse_nn.kernels.build --backend cuda   # 或 rocm / npu
python -m verse_nn.kernels.build --info
```

详见 [`HANDOFF.md`](HANDOFF.md) 与 [`VerseNext/README.md`](VerseNext/README.md)。

## License

[Apache-2.0](LICENSE)
