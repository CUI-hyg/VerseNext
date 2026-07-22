"""NexRL: VerseNex 强化学习算法包（Part4K1 Task 4）。

定义 RL 五要素抽象：
- NexAgent: 策略网络（VerseNexLM）+ 参考网络（KL 约束）
- NexEnv: 任务环境抽象（observation + reward）
- NexState: RL 状态（prompt + tokens + KV cache）
- NexAction: RL 动作（token + 采样策略）
- NexReward: 多维奖励（correctness + fluency + safety + length_penalty）

训练组件：
- ParallelRolloutCollector: 并行 rollout 采集器
- NexTrainer: PPO 风格训练器（clip + GAE + KL + value function）

用法：
    from verse_nex.nexrl import NexAgent, NexTrainer, NexReward
    agent = NexAgent(policy=model)
    trainer = NexTrainer(agent=agent, cfg={"clip_ratio": 0.2})
    trainer.fit(prompts=["1+1=", "2+2="], n_epochs=10)
"""

from .state import NexState, batch_states
from .action import (
    NexAction,
    ActionSampler,
    ExplorationSchedule,
    repeat_penalty,
)
from .reward import (
    NexReward,
    RewardNormalizer,
    RewardShaper,
)
from .env import (
    NexEnv,
    ChatEnv,
    MathEnv,
    CodeEnv,
)
from .agent import NexAgent
from .collector import (
    Rollout,
    ParallelRolloutCollector,
)
from .trainer import NexTrainer

__all__ = [
    # 五要素
    "NexAgent",
    "NexEnv",
    "NexState",
    "NexAction",
    "NexReward",
    # 环境
    "ChatEnv",
    "MathEnv",
    "CodeEnv",
    # 动作
    "ActionSampler",
    "ExplorationSchedule",
    "repeat_penalty",
    # 奖励
    "RewardNormalizer",
    "RewardShaper",
    # 采集与训练
    "Rollout",
    "ParallelRolloutCollector",
    "NexTrainer",
    # 工具
    "batch_states",
]

__version__ = "0.1.0"
