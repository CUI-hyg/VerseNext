# 模型压缩技术参考

> 本文档汇总 Verse 框架 `verse_torch.compress` 模块所参考/集成的核心压缩技术。
> 覆盖：1.58-bit ternary 量化、4-bit + LoRA、Outlier-Safe 训练、知识蒸馏、结构化剪枝、LoRA、SparseGPT、GPTQ、AWQ、SmoothQuant。
> 每篇条目包含：标题、作者、年份、关键思想、与本框架的关系。

---

## 1. BitNet b1.58（1.58-bit Ternary 量化）

- **标题**: The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits
- **作者**: Shenzhi Wang, Beichen Zhang, Yue Liao, Qiansong Wei, Junlin Fang, Wenhao Chai, Yuxuan Lou, Ningxin Zheng, Wei Li, Jiashuo Liu, Hongyu Yang, Jianqiao Wang, Yujun Shen, Bei Yu, Lei Chen, Ping Luo, Chun Yuan, Qifeng Chen, Deli Zhao
- **年份**: 2024（arXiv:2402.01689）
- **关键思想**:
  - 把所有权重约束为三元 `{−1, 0, +1}`，每权重 ≈ log₂(3) ≈ 1.58 bit。
  - scale = mean(|w|) / 0.5，量化 `w_q = round(w / scale).clip(-1, 1)`。
  - 推理时矩阵乘法退化为加减法，CPU/边缘端可显著加速。
- **与本框架关系**:
  - `verse_torch.quantize.quantize_ternary` 实现了 BitNet b1.58 风格打包（2 bit/value，4 values/byte）。
  - `compress.ternary_only` 提供「全模型 ternary 量化」入口。

---

## 2. QLoRA（4-bit + LoRA）

- **标题**: QLoRA: Efficient Finetuning of Quantized LLMs
- **作者**: Tim Dettmers, Artidoro Pagnoni, Ari Holtzman, Luke Zettlemoyer
- **年份**: 2023（arXiv:2305.14314, NeurIPS 2023）
- **关键思想**:
  - 4-bit NormalFloat（NF4）量化 + 双重量化（double quantization）+ paged attention。
  - 在 4-bit 冻结基座上挂 LoRA 适配器（bf16）做参数高效微调。
  - 65B 模型可在单卡 48GB GPU 上微调。
- **与本框架关系**:
  - `compress.LoRALinear` 与 `compress.quantize_only(dtype='int4')` 的组合是 QLoRA 的 PoC 简化版。
  - `compress_pipeline` 的最后一步 `lora_wrap` 即复刻 QLoRA 的「量化基座 + LoRA 增量」思路。

---

## 3. OSP（Outlier-Safe Pre-Training）

- **标题**: Outlier-Safe Pre-Training for Large Language Models
- **作者**: Seungwoo Choi, Changhun Lee, Eungae Kim, Wonpyo Park, Hyegang Ju, Yeon Su Jung, Soojin Yoon, Woncheol Shin, Gunho Park, Seungjun Moon, Saurav Prakash, Jaeyeol Lee, Guseung Kim, Hanseok Oh, Kyuam Kim, Hyungjun Lee, Jung-Woo Ha, Donghoon Lee
- **年份**: 2025（arXiv:2504.07418）
- **关键思想**:
  - 大模型激活分布存在显著 outlier channel，是 INT4/INT8 量化误差的主要来源。
  - OSP 通过 Pre-Norm + 适当初始化 + 训练时正则抑制 outlier 出现，从训练阶段就让模型对量化友好。
  - 训练完成后再做 INT4 量化时，相比标准训练可降低 5-10× 量化误差。
- **与本框架关系**:
  - `OutlierSafePruner` 类名致敬该工作；本框架 PoC 中以「按 |weight|_mean 剪掉 bottom 30%」作为 Outlier-Safe 的简化实现（mask + 冻结）。
  - 长期路线：在训练侧引入 outlier-aware 正则，使压缩前权重分布更紧凑。

---

## 4. 知识蒸馏（Hinton 2015）

- **标题**: Distilling the Knowledge in a Neural Network
- **作者**: Geoffrey Hinton, Oriol Vinyals, Jeff Dean
- **年份**: 2015（arXiv:1503.02531）
- **关键思想**:
  - Teacher 模型 logits 经温度 T softmax 得到 soft target，Student 模型学习 soft target 的 KL 散度。
  - `L = α · T² · KL(softmax(t_s/T) || softmax(t_t/T)) + (1−α) · CE(t_s, y_hard)`。
  - T² 系数补偿 T 缩放导致的梯度幅度衰减。
- **与本框架关系**:
  - `compress.KnowledgeDistiller(teacher, student, T=2.0, alpha=0.5)` 实现该 loss 公式。
  - `compress.distill_only` 提供蒸馏训练循环入口。

---

## 5. 结构化剪枝（Han 2015）

- **标题**: Learning both Weights and Connections for Efficient Neural Networks
- **作者**: Song Han, Jeff Pool, John Tran, William J. Dally
- **年份**: 2015（arXiv:1506.02626, NeurIPS 2015）
- **关键思想**:
  - 学权重的同时学连接：用 L1 magnitude 对权重剪枝，再 retrain 恢复精度。
  - 迭代「prune → retrain → prune」收敛到稀疏结构。
  - 配合 Huffman 编码 + 量化可达 35× 压缩比。
- **与本框架关系**:
  - `OutlierSafePruner` 借鉴按 magnitude 剪枝的思路，但按 head/channel 整体打分（结构化），不破坏稠密存储。
  - 剪掉的参数通过 mask 置零并冻结，PoC 不做 retrain 以简化。

---

## 6. LoRA（Hu 2021）

- **标题**: LoRA: Low-Rank Adaptation of Large Language Models
- **作者**: Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen
- **年份**: 2021（arXiv:2106.09685, ICLR 2022）
- **关键思想**:
  - 冻结预训练权重 W，挂上低秩增量 ΔW = B @ A，其中 A ∈ ℝ^{r×d_in}, B ∈ ℝ^{d_out×r}，r ≪ min(d_in, d_out)。
  - A 高斯初始化，B 零初始化，保证训练初始 ΔW = 0。
  - 推理时可将 BA 合并到 W，无额外开销。
- **与本框架关系**:
  - `compress.LoRALinear(d_in, d_out, r=8, alpha=16)` 是 LoRA 的 PoC 实现。
  - `merge()` 方法可把 A @ B 加回 base.weight，得到无开销 Linear。

---

## 7. SparseGPT

- **标题**: SparseGPT: Massive Language Models Can be Accurately Pruned in One-Shot
- **作者**: Elias Frantar, Dan Alistarh
- **年份**: 2023（arXiv:2301.00774, ICML 2023）
- **关键思想**:
  - 一次性把 175B 模型剪到 50% 稀疏度（非结构化），几乎无精度损失。
  - 基于层内最优回归：求解最小化 ||Wx − W'x||² 的稀疏 W'。
  - 不需要 retrain，远快于 magnitude pruning + retraining。
- **与本框架关系**:
  - 当前 PoC 未直接实现 SparseGPT 算法，但作为后续路线参考。
  - `OutlierSafePruner` 的 mask + 冻结策略与 SparseGPT 「one-shot」理念相通。

---

## 8. GPTQ

- **标题**: GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers
- **作者**: Elias Frantar, Saleh Ashkboos, Torsten Hoefler, Dan Alistarh
- **年份**: 2022（arXiv:2210.17323, ICLR 2023）
- **关键思想**:
  - 基于 OBQ（Optimal Brain Quantization）的逐列量化：每次量化一行后立即更新剩余权重以补偿误差。
  - 配合 Cholesky 分解稳定求解 Hessian 逆。
  - 支持 INT4 量化 + 组内 group-wise scale。
- **与本框架关系**:
  - `verse_torch.quantize.quantize_int4` 提供对称 per-channel INT4 量化（PoC 简化版，未实现误差补偿）。
  - `compress.quantize_only(dtype='int4')` 是 GPTQ 思想的「快速 PoC 版本」。

---

## 9. AWQ（Activation-aware Weight Quantization）

- **标题**: AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration
- **作者**: Lin Ji, Tang Jiaming, Tang Haotian, Yang Shang, Dang Xingyu, Gan Chuang, Li Song
- **年份**: 2023（arXiv:2306.00978, MLSys 2024）
- **关键思想**:
  - 观察到：激活 magnitude 大的 channel 对应权重对量化误差更敏感。
  - 对这部分「显著 channel」乘以 scale s，对应输入乘以 1/s，保持等价但量化误差更小。
  - 通过 grid search 找最优 s（搜索空间 ≤ 1% 参数）。
- **与本框架关系**:
  - 当前 PoC 未实现 per-channel scaling search，但 `OutlierSafePruner` 的「按 |weight|_mean 识别重要通道」是类似动机的简化。
  - 后续可把 AWQ 的 scale 搜索插入 `compress.quantize_only`。

---

## 10. SmoothQuant

- **标题**: SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models
- **作者**: Guangxuan Xiao, Ji Lin, Mickael Seznec, Hao Wu, Julien Demouth, Song Han
- **年份**: 2022（arXiv:2211.10438, ICML 2023）
- **关键思想**:
  - 将「难量化的 weight」与「易量化的 activation」对调：把 activation 的 outlier 平滑迁移到 weight。
  - 公式：`x' = x / s; W' = W * s`，s 由 (max|x|)^α / (max|W|)^(1−α) 决定。
  - 让 INT8 量化在 LLM 上接近 FP16 精度。
- **与本框架关系**:
  - 当前 PoC 未实现 SmoothQuant 算子，但作为量化误差控制的后备方案。
  - 若 `compress_pipeline` 的 INT4 量化误差过大（>5%），可降级到 INT8（即 SmoothQuant 推荐精度）。

---

## 11. Outlier Weights 控制（补充：Bondarenko et al.）

- **标题**: Quantizable Transformers: Removing Outliers by Helping Attention Heads Do Nothing
- **作者**: Yelysei Bondarenko, Markus Nagel, Tijmen Blankevoort
- **年份**: 2023（arXiv:2306.12929, NeurIPS 2023）
- **关键思想**:
  - 在 Attention 上引入「ability to do nothing」机制（LayerNorm + gated residual），让某些 head 输出恒为 0。
  - 这些 head 自然进入低 magnitude 区域，量化误差显著下降。
  - 与 OSP 互补：OSP 在训练阶段抑制 outlier，本工作在网络结构层面给「outlier-friendly」的退路。
- **与本框架关系**:
  - `OutlierSafePruner` 对 GQASelfAttention 按 head 维度剪枝（mask 整个 head）正是受此启发。
  - 当 head |weight|_mean 过低时直接 mask 掉，等价于让该 head 输出恒为 0。

---

## 12. AQLM（Additive Quantization for LMs，补充）

- **标题**: AQLM: Extreme Compression of Large Language Models via Quantization with Codebook Learning
- **作者**: Vage Egiazarian, Ella Prazdnichnykh, Alexander Kuznedelev, Michael Diskin, Anton Babenko, Denis Kuznedelev, Dan Alistarh
- **年份**: 2024（arXiv:2401.06118, ICML 2024）
- **关键思想**:
  - 把每行权重表示为多本 codebook 中 code 的累加（additive quantization）。
  - 配合端到端蒸馏微调，可在 2-bit 等级上保持接近 FP16 精度。
- **与本框架关系**:
  - 当前 PoC 未实现 AQLM；作为「ternary 量化精度不足」时的进阶路线。

---

## 13. 综述参考（Han et al. 2015 综述）

- **标题**: Deep Compression: Compressing Deep Neural Networks with Pruning, Trained Quantization and Huffman Coding
- **作者**: Song Han, Huizi Mao, William J. Dally
- **年份**: 2015（arXiv:1510.00149, ICLR 2016 best paper）
- **关键思想**:
  - 三段式压缩 pipeline：剪枝 → 量化 → Huffman 编码，总体可达 35-49× 压缩比。
  - 证明「prune → quantize → encode」是组合最优的。
- **与本框架关系**:
  - `compress_pipeline` 的三段流程（prune → quantize → lora_wrap）与此经典 pipeline 在结构上对齐。
  - Huffman 编码留给后续 PoC（量化打包已包含紧凑 bit-level 存储需求）。

---

## 致谢与版权说明

以上参考文献均为公开 arXiv 论文，本框架仅引用其方法名作为 PoC 命名/思路参考。完整实现细节请参考原文。

| 序号 | 技术 | 在本框架中的对应 |
|------|------|------------------|
| 1 | BitNet b1.58 | `quantize_ternary` + `ternary_only` |
| 2 | QLoRA | `quantize_only(int4)` + `lora_wrap`（在 `compress_pipeline` 中组合）|
| 3 | OSP | `OutlierSafePruner`（mask + 冻结，PoC 简化版）|
| 4 | 蒸馏 | `KnowledgeDistiller` + `distill_only` |
| 5 | 结构化剪枝 | `OutlierSafePruner`（按 head/channel）|
| 6 | LoRA | `LoRALinear` |
| 7 | SparseGPT | 路线参考（PoC 未实现）|
| 8 | GPTQ | `quantize_int4`（简化版，未做误差补偿）|
| 9 | AWQ | 路线参考（PoC 未实现 per-channel scaling search）|
| 10 | SmoothQuant | INT4 失败时降级到 INT8 的理论依据 |
| 11 | Outlier-friendly 结构 | GQASelfAttention head 剪枝 |
| 12 | AQLM | 进阶路线（PoC 未实现）|
| 13 | Deep Compression pipeline | `compress_pipeline` 三段式结构 |
