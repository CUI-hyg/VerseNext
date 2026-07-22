# ADR-007: NexRL 强化学习设计

- **状态**：Accepted
- **日期**：2026-07-22
- **决策者**：Verse 框架作者（CometFuture / CUI-hyg）
- **相关规范**：[`/workspace/.trae/specs/part4k1-infra-model-upgrade/spec.md`](../../../.trae/specs/part4k1-infra-model-upgrade/spec.md)
- **前置 ADR**：[ADR-002 线性复杂度架构](adr-002-linear-complexity.md)（VerseNexLM 作为策略网络）
- **相关 ADR**：[ADR-005 GPU/NPU 后端](adr-005-gpu-npu-backend.md)（并行 rollout 受益于 GPU 批量前向）

## 上下文

Part4K1 之前，Verse 框架的训练能力局限于**监督学习**（SFT / DPO / LoRA 微调），缺少强化学习后训练路径。随着 CometSpark V0.5-1B 推出，需要 RLHF 风格的后训练来：

1. **对齐人类偏好**：SFT 只学"复述标注"，无法优化"生成质量"的多维目标（correctness + fluency + safety + length_penalty）
2. **避免 reward hacking**：单纯最大化 reward 会导致模型钻空子（如重复高分 token），需要 KL 约束 + 多维 reward
3. **CPU 友好**：Verse 是 CPU-first 框架，RL 训练也必须能在 CPU 上跑（rollout 是主要瓶颈）
4. **与 VerseNexLM 集成**：策略网络必须是 VerseNexLM（线性复杂度 + KV cache），不能强制要求 PyTorch nn.Module
5. **CLI 集成**：用户应能通过 `verse-posttrain --rl nexrl` 一键启动

同时，业界 RLHF 工具链（TRL / OpenRLHF / verl）普遍依赖 PyTorch + Transformers，与 Verse 的"零重型依赖"原则冲突。需要一个自研的轻量 RL 抽象。

## 决策

**实现 `verse_nex.nexrl` 子包，采用"五要素抽象"（NexAgent / NexEnv / NexState / NexAction / NexReward）+ ParallelRolloutCollector + NexTrainer（PPO clipped surrogate + GAE + KL 自适应 + value function）。**

具体含义：

### 1. 五要素抽象

| 要素 | 类 | 职责 |
|---|---|---|
| `NexAgent` | `nexrl/agent.py` | 策略网络（VerseNexLM）+ 参考网络（冻结副本，KL 约束）+ KL 散度计算 |
| `NexEnv` | `nexrl/env.py` | 任务环境（抽象基类 + `ChatEnv` / `MathEnv` / `CodeEnv`），提供 observation + reward |
| `NexState` | `nexrl/state.py` | RL 状态数据类：prompt + prompt_tokens + generated_tokens + kv_cache + logprobs + step + done |
| `NexAction` | `nexrl/action.py` | 动作采样：ε-greedy / softmax / nucleus + ExplorationSchedule（探索衰减）+ repeat_penalty（重复惩罚） |
| `NexReward` | `nexrl/reward.py` | 多维奖励：correctness + fluency + safety + length_penalty + RewardNormalizer（running mean/std）+ RewardShaper（potential-based） |

### 2. ParallelRolloutCollector

- 多 prompt / 多 rollout 并行采样（batched forward）
- GPU 批量前向（通过 DeviceBackend 委托 PyTorch）
- 返回 `Rollout` 数据类：state 序列 + action 序列 + reward 序列 + logprob 序列 + value 序列

### 3. NexTrainer（PPO 风格）

- **PPO clipped surrogate loss**：`L_clip = mean(min(r_t * A_t, clip(r_t, 1-ε, 1+ε) * A_t))`
- **GAE（Generalized Advantage Estimation）**：`A_t = Σ (γλ)^l * δ_{t+l}`，其中 `δ_t = r_t + γV(s_{t+1}) - V(s_t)`
- **KL 自适应**：监控 policy 与 ref_model 的 KL 散度，超阈值时自动增加 KL 惩罚权重（避免策略崩溃）
- **Value function**：critic 网络预测 V(s)，与 policy 共享 backbone 或独立
- **纯策略梯度 fallback**：若 value function 不可用，退化为 REINFORCE（无 GAE，仅用 return）

### 4. NexTokenizerWrapper 集成

`verse_tokenizer.nex_wrapper.NexTokenizerWrapper` 在 token 边界注入 RL 信号（reward-weighted token preference），高频高奖励子串优先成 token——这是 RL 与 tokenizer 训练的桥梁。

### 5. CLI 集成

```bash
verse-posttrain --config spark/config/cometspark_v05.yml --rl nexrl
```

`verse-posttrain` 调用 `verse_infra.verse_trainer.RLTrainer`，后者实例化 `NexTrainer` + `ParallelRolloutCollector` + 五要素。

## 后果

### 优点

- **自研轻量 RL**：不依赖 TRL / OpenRLDF / verl，与 Verse 零重型依赖原则一致
- **五要素抽象清晰**：Agent / Env / State / Action / Reward 职责分离，易于扩展新任务（只需实现新的 `NexEnv` 子类）
- **PPO + GAE + KL 自适应**：业界验证的稳定 RL 算法栈，避免 reward hacking + 策略崩溃
- **并行 rollout**：`ParallelRolloutCollector` 批量前向，GPU 下吞吐量显著提升
- **CPU 友好**：rollout 用 KV cache + recurrent 模式，常数内存；CPU 上也能跑（虽然慢）
- **VerseNexLM 原生集成**：策略网络就是 VerseNexLM，不需要适配层
- **多维 reward**：correctness + fluency + safety + length_penalty 可加权组合，避免单一 reward 偏差
- **Reward shaping**：potential-based shaping 保证最优策略不变，但加速收敛

### 缺点

- **CPU 上 RL 训练慢**：rollout 需要生成完整序列，CPU 上比 GPU 慢 1-2 个数量级——但这是已知限制，GPU 通过 DeviceBackend 加速
- **critic 网络额外内存**：value function 需要独立 critic（或共享 backbone），增加模型体积
- **KL 自适应超参敏感**：KL 阈值与惩罚权重的初始值需要调参，不同任务差异大
- **无 reward model 训练**：当前 `NexReward` 是规则驱动（correctness 用 exact match、fluency 用 perplexity 等），未实现 learnable reward model——留待后续

### 风险与缓解

| 风险 | 缓解策略 |
|---|---|
| 策略崩溃（KL 爆炸） | KL 自适应：超阈值时自动增加 KL 惩罚权重；`NexAgent` 计算 KL 后回传给 trainer，trainer 监控并调整 |
| Reward hacking（重复高分 token） | `NexAction.repeat_penalty` 惩罚重复动作；多维 reward（length_penalty 抑制过短/过长输出） |
| Value function 不准导致 GAE 偏差 | 提供 REINFORCE fallback（无 value function）；value loss 加入训练目标 |
| Rollout 内存爆炸（长序列） | `NexState.kv_cache` 复用 KV cache，recurrent 模式常数内存；`ParallelRolloutCollector` 限制 max_rollout_length |
| CPU 环境 RL 训练不可行 | 文档明确建议 RL 训练用 GPU（`--device cuda`）；CPU 仅用于小规模验证 |

## 替代方案（已否决）

### 方案 A：依赖 TRL（HuggingFace Transformers Reinforcement Learning）

**描述**：直接用 TRL 的 PPOTrainer / DPOTrainer，策略网络用 HuggingFace Transformers 模型。

**否决理由**：
- TRL 硬依赖 PyTorch + Transformers，违反 Verse 零重型依赖原则
- 策略网络必须是 HuggingFace `PreTrainedModel`，无法用 VerseNexLM
- TRL 的 RLHF 流程与 Verse 的 CPU-first 路径冲突
- 安装体积庞大（Transformers + Tokenizers + PyTorch）

### 方案 B：纯 REINFORCE（无 PPO / GAE）

**描述**：用最简单的 REINFORCE（policy gradient），不引入 PPO clip / GAE / value function。

**否决理由**：
- REINFORCE 方差大，训练不稳定
- 无 KL 约束容易策略崩溃
- 无 advantage estimation 收敛慢
- 业界已普遍用 PPO 替代 REINFORCE

### 方案 C：DPO 替代 RLHF

**描述**：只用 DPO（Direct Preference Optimization），不做 PPO 风格 RL。

**否决理由**：
- DPO 需要偏好数据（chosen / rejected 对），数据成本高
- DPO 无法优化多维 reward（correctness / fluency / safety）
- DPO 是离线方法，无法在线探索
- Verse 已有 `DPOTrainer`（监督学习路径），NexRL 是补充而非替代

### 方案 D：用 Stable-Baselines3 或 OpenRLHF

**描述**：集成现有 RL 库（SB3 / OpenRLHF / verl）。

**否决理由**：
- SB3 面向游戏 RL（Atari / MuJoCo），不擅长 LLM 的 token 级动作空间
- OpenRLHF / verl 硬依赖 PyTorch + Transformers
- 都不符合 Verse 的零依赖 + CPU-first 原则

## 备注

- 本 ADR 是 Part4K1 "后训练路线"的核心决策
- NexRL 与现有 `SFTTrainer` / `DPOTrainer` / `LoRATrainer` 互补，用户可按需选择
- `verse-posttrain --rl nexrl|sft|dpo` CLI 统一入口
- 五要素抽象灵感来源于 OpenAI Gym（Env / Action / Observation）+ RLHF 文献（Agent / Reward）
- 相关测试：`tests/test_nexrl.py` 覆盖多维奖励 / 并行 rollout / KL 防崩溃 / 动作采样策略
- 相关代码：[`verse_nex/nexrl/`](../../packages/verse_nex/verse_nex/nexrl/)
