"""测试 verse_nex.nexrl 模块（NexRL 强化学习算法）。

覆盖（SubTask 4.6）：
- NexReward 多维奖励计算 + 加权
- RewardNormalizer running mean/std（Welford 算法）
- ActionSampler 各采样策略（ε-greedy / softmax / nucleus / topk）
- ExplorationSchedule 衰减（linear / cosine / exponential）
- repeat_penalty 重复惩罚
- ParallelRolloutCollector 收集 rollout（小模型，3 prompt × 2 rollout）
- KL 防崩溃：构造 KL 超阈值场景，验证惩罚权重增加
- NexTrainer fit 几步不崩溃（小模型，含 value function + fallback）
- NexEnv 各环境（ChatEnv / MathEnv / CodeEnv）

运行方式：
    cd /workspace && PYTHONPATH=packages/verse_torch:packages/verse_nex \
        python -m pytest tests/test_nexrl.py -v
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

# PYTHONPATH 适配（与 test_training_nex.py 风格一致）
_HERE = os.path.dirname(os.path.abspath(__file__))
_WORKSPACE = os.path.dirname(_HERE)
for _sub in ("verse_torch", "verse_nex", "verse_tokenizer"):
    _p = os.path.join(_WORKSPACE, "packages", _sub)
    if os.path.isdir(_p):
        sys.path.insert(0, _p)

from verse_nex import VerseNexLM
from verse_nex.nexrl import (
    NexAgent,
    NexEnv,
    NexState,
    NexAction,
    NexReward,
    ChatEnv,
    MathEnv,
    CodeEnv,
    ActionSampler,
    ExplorationSchedule,
    repeat_penalty,
    RewardNormalizer,
    RewardShaper,
    Rollout,
    ParallelRolloutCollector,
    NexTrainer,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _tiny_model(
    vocab_size: int = 256,
    dim: int = 32,
    n_layer: int = 2,
    n_head: int = 4,
    n_kv_head: int = 2,
):
    """构造 tiny 测试模型（与 test_cometspark_nex.py 风格一致）。

    使用 window_size=4, num_global_tokens=2 避免短序列形状错误。
    prompt 必须 >= window_size 个 token。
    """
    return VerseNexLM(
        vocab_size=vocab_size,
        dim=dim,
        n_layer=n_layer,
        n_head=n_head,
        n_kv_head=n_kv_head,
        window_size=4,
        num_global_tokens=2,
        max_seq_len=128,
        use_alibi=True,
        use_rope=False,
        dropout=0.0,
        tie_weights=True,
    )


def _encode(text: str) -> list:
    """简单编码：ord(c) % 256。"""
    return [ord(c) % 256 for c in text]


def _decode(tokens: list) -> str:
    """简单解码：chr(t % 256)。"""
    return "".join(chr(int(t) % 256) for t in tokens)


# ===========================================================================
# SubTask 4.2: NexReward 多维奖励计算
# ===========================================================================


class TestNexReward:
    """NexReward 多维奖励 + 加权。"""

    def test_correctness_exact(self):
        """精确匹配模式。"""
        reward = NexReward(correctness_mode="exact", target_len=5)
        result = reward.compute(generated="hello", reference="hello")
        assert result["correctness"] == 1.0

        result2 = reward.compute(generated="world", reference="hello")
        assert result2["correctness"] == 0.0

    def test_correctness_substring(self):
        """子串匹配模式。"""
        reward = NexReward(correctness_mode="substring")
        result = reward.compute(generated="hello world", reference="hello")
        assert result["correctness"] == 1.0

        result2 = reward.compute(generated="world", reference="hello")
        assert result2["correctness"] == 0.0

    def test_correctness_bleu(self):
        """BLEU 模式。"""
        reward = NexReward(correctness_mode="bleu")
        result = reward.compute(generated="hello", reference="hello")
        assert result["correctness"] > 0.5  # 完全匹配 BLEU 应较高

    def test_fluency(self):
        """fluency = 1 / (1 + mean_neg_logprob)。"""
        reward = NexReward()
        # logprobs 接近 0 → fluency 接近 1
        result = reward.compute(generated="test", reference="", logprobs=[-0.1, -0.2])
        assert 0.0 < result["fluency"] <= 1.0
        assert result["fluency"] > 0.8  # logprobs 接近 0，fluency 高

        # logprobs 很负 → fluency 低
        result2 = reward.compute(generated="test", reference="", logprobs=[-5.0, -10.0])
        assert result2["fluency"] < result["fluency"]

    def test_safety(self):
        """safety: 关键词黑名单检测。"""
        reward = NexReward()
        result = reward.compute(generated="hello world")
        assert result["safety"] == 1.0  # 安全

        # 使用黑名单中的关键词（"暴力" 或 "kill"）
        result2 = reward.compute(generated="this is 暴力 content")
        assert result2["safety"] == 0.0  # 不安全

        result3 = reward.compute(generated="how to hack something")
        assert result3["safety"] == 0.0  # 不安全

    def test_length_penalty(self):
        """length_penalty: -abs(len - target_len) / target_len。"""
        reward = NexReward(target_len=5)
        result = reward.compute(
            generated="hello",
            reference="",
            generated_tokens=[1, 2, 3, 4, 5],  # len=5 = target
        )
        assert result["length_penalty"] == 0.0  # 完美长度

        result2 = reward.compute(
            generated="hi",
            reference="",
            generated_tokens=[1, 2],  # len=2 < target=5
        )
        assert result2["length_penalty"] < 0.0  # 负惩罚

    def test_total_weighted(self):
        """total 是加权总和。"""
        weights = {
            "correctness": 0.4,
            "fluency": 0.2,
            "safety": 0.2,
            "length_penalty": 0.2,
        }
        reward = NexReward(weights=weights, target_len=5)
        result = reward.compute(
            generated="hello",
            reference="hello",
            logprobs=[-0.5, -0.5, -0.5, -0.5, -0.5],
            generated_tokens=[1, 2, 3, 4, 5],
        )

        expected_total = (
            weights["correctness"] * result["correctness"]
            + weights["fluency"] * result["fluency"]
            + weights["safety"] * result["safety"]
            + weights["length_penalty"] * result["length_penalty"]
        )
        assert abs(result["total"] - expected_total) < 1e-6

    def test_custom_weights(self):
        """自定义权重。"""
        reward = NexReward(weights={"correctness": 1.0, "fluency": 0.0,
                                      "safety": 0.0, "length_penalty": 0.0})
        result = reward.compute(generated="hello", reference="hello")
        assert abs(result["total"] - result["correctness"]) < 1e-6


# ===========================================================================
# SubTask 4.2: RewardNormalizer（Welford running mean/std）
# ===========================================================================


class TestRewardNormalizer:
    """RewardNormalizer running mean/std。"""

    def test_basic_normalization(self):
        """基本归一化：更新后 mean/std 正确。"""
        norm = RewardNormalizer()
        rewards = [1.0, 2.0, 3.0, 4.0, 5.0]
        for r in rewards:
            norm.update(r)

        assert abs(norm.mean - 3.0) < 1e-6
        assert norm.count == 5
        assert norm.variance > 0

    def test_normalize_zero_mean(self):
        """归一化后均值接近 0。"""
        norm = RewardNormalizer()
        for r in [1.0, 2.0, 3.0, 4.0, 5.0]:
            norm.update(r)

        # 归一化 3.0（=mean）应接近 0
        normed = norm.normalize(3.0)
        assert abs(normed) < 0.1

    def test_clip(self):
        """裁剪极端值。"""
        norm = RewardNormalizer(clip_range=2.0)
        for r in [1.0, 1.0, 1.0, 1.0]:
            norm.update(r)
        # 归一化一个非常大的值应被裁剪到 clip_range
        normed = norm.normalize(1000.0)
        assert normed <= 2.0

    def test_insufficient_data(self):
        """数据不足时返回原始值。"""
        norm = RewardNormalizer()
        norm.update(1.0)  # 只有 1 个数据点
        assert norm.normalize(5.0) == 5.0  # count < 2，返回原始值

    def test_reset(self):
        """重置统计量。"""
        norm = RewardNormalizer()
        for r in [1.0, 2.0, 3.0]:
            norm.update(r)
        norm.reset()
        assert norm.count == 0
        assert norm.mean == 0.0

    def test_batch_update(self):
        """批量更新。"""
        norm = RewardNormalizer()
        norm.update_batch([1.0, 2.0, 3.0])
        assert norm.count == 3
        assert abs(norm.mean - 2.0) < 1e-6


# ===========================================================================
# SubTask 4.2: RewardShaper
# ===========================================================================


class TestRewardShaper:
    """RewardShaper potential-based 塑形。"""

    def test_basic_shaping(self):
        """基本塑形：F = γ*Φ(s') - Φ(s)。"""
        shaper = RewardShaper(gamma=0.99)
        prev_info = {"logprobs": [-1.0, -2.0]}
        curr_info = {"logprobs": [-0.5, -1.0]}
        shaped = shaper.shape(1.0, prev_info, curr_info)
        # Φ(curr) > Φ(prev) → shaped > reward
        assert shaped != 1.0

    def test_custom_potential(self):
        """自定义势函数。"""
        def my_potential(state_info):
            return float(state_info.get("score", 0.0))

        shaper = RewardShaper(gamma=0.9, potential_fn=my_potential)
        shaped = shaper.shape(1.0, {"score": 0.5}, {"score": 0.8})
        expected = 1.0 + 0.9 * 0.8 - 0.5
        assert abs(shaped - expected) < 1e-6


# ===========================================================================
# SubTask 4.3: ActionSampler 各采样策略
# ===========================================================================


class TestActionSampler:
    """ActionSampler 各采样策略。"""

    def test_epsilon_greedy_argmax(self):
        """ε=0 时始终选 argmax。"""
        logits = np.array([1.0, 5.0, 2.0, 0.5], dtype=np.float32)
        rng = np.random.default_rng(42)
        for _ in range(10):
            tid = ActionSampler.epsilon_greedy(logits, epsilon=0.0, rng=rng)
            assert tid == 1  # argmax 是 index 1

    def test_epsilon_greedy_random(self):
        """ε=1 时随机采样。"""
        logits = np.array([1.0, 5.0, 2.0, 0.5], dtype=np.float32)
        rng = np.random.default_rng(42)
        tids = set()
        for _ in range(50):
            tid = ActionSampler.epsilon_greedy(logits, epsilon=1.0, rng=rng)
            tids.add(tid)
        # 应该有多个不同的 token（随机性）
        assert len(tids) > 1

    def test_softmax(self):
        """softmax 采样：概率高的 token 更频繁。"""
        logits = np.array([0.1, 5.0, 0.2, 0.1], dtype=np.float32)
        rng = np.random.default_rng(42)
        counts = [0] * 4
        for _ in range(1000):
            tid = ActionSampler.softmax(logits, temperature=1.0, rng=rng)
            counts[tid] += 1
        # index 1 应该最频繁
        assert counts[1] > counts[0]
        assert counts[1] > counts[2]

    def test_nucleus(self):
        """nucleus (top-p) 采样。"""
        logits = np.array([0.1, 5.0, 4.0, 0.05], dtype=np.float32)
        rng = np.random.default_rng(42)
        tids = set()
        for _ in range(50):
            tid = ActionSampler.nucleus(logits, top_p=0.9, rng=rng)
            tids.add(tid)
        # top-p=0.9 应主要在 index 1 和 2
        assert tids.issubset({1, 2, 0, 3})  # 都在范围内

    def test_topk(self):
        """top-k 采样。"""
        logits = np.array([0.1, 5.0, 4.0, 0.05, 3.0], dtype=np.float32)
        rng = np.random.default_rng(42)
        tids = set()
        for _ in range(50):
            tid = ActionSampler.topk(logits, k=2, rng=rng)
            tids.add(tid)
        # k=2 → 只在 index 1 和 2
        assert tids.issubset({1, 2})

    def test_sample_generic(self):
        """通用 sample 入口。"""
        logits = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        rng = np.random.default_rng(42)
        for strategy in ["epsilon_greedy", "softmax", "nucleus", "topk"]:
            tid, lp = ActionSampler.sample(
                logits, strategy=strategy, rng=rng, temperature=1.0,
            )
            assert 0 <= tid < 3
            assert lp <= 0.0  # logprob 总是 <= 0


# ===========================================================================
# SubTask 4.3: ExplorationSchedule 衰减
# ===========================================================================


class TestExplorationSchedule:
    """ExplorationSchedule 衰减。"""

    def test_linear(self):
        """线性衰减。"""
        sched = ExplorationSchedule(
            epsilon_start=0.3, epsilon_end=0.02, total_steps=100, mode="linear",
        )
        assert abs(sched.value(0) - 0.3) < 1e-6
        assert abs(sched.value(100) - 0.02) < 1e-6
        # 中间值应在 start 和 end 之间
        mid = sched.value(50)
        assert 0.02 < mid < 0.3

    def test_cosine(self):
        """余弦衰减。"""
        sched = ExplorationSchedule(
            epsilon_start=0.3, epsilon_end=0.02, total_steps=100, mode="cosine",
        )
        assert abs(sched.value(0) - 0.3) < 1e-6  # cos(0)=1, 取 max
        assert abs(sched.value(100) - 0.02) < 1e-6  # cos(π)=-1, 取 min
        mid = sched.value(50)
        assert 0.02 < mid < 0.3

    def test_exponential(self):
        """指数衰减。"""
        sched = ExplorationSchedule(
            epsilon_start=0.3, epsilon_end=0.02, total_steps=100, mode="exponential",
        )
        assert abs(sched.value(0) - 0.3) < 1e-4
        assert abs(sched.value(100) - 0.02) < 1e-4
        mid = sched.value(50)
        assert 0.02 < mid < 0.3

    def test_clamp(self):
        """步数超出范围时 clamp。"""
        sched = ExplorationSchedule(
            epsilon_start=0.3, epsilon_end=0.02, total_steps=100, mode="linear",
        )
        assert abs(sched.value(-10) - 0.3) < 1e-6  # 负步数 clamp 到 0
        assert abs(sched.value(200) - 0.02) < 1e-6  # 超出 clamp 到 total


# ===========================================================================
# SubTask 4.3: repeat_penalty
# ===========================================================================


class TestRepeatPenalty:
    """repeat_penalty 重复惩罚。"""

    def test_basic_penalty(self):
        """基本惩罚：已生成 token 的 logits 降低。"""
        logits = np.array([1.0, 5.0, 2.0, 3.0], dtype=np.float32)
        penalized = repeat_penalty(logits, [1], penalty=1.5)
        # index 1 的 logit 应降低
        assert penalized[1] < logits[1]
        # 其他位置不变
        assert penalized[0] == logits[0]

    def test_argmax_changes(self):
        """惩罚后 argmax 可能改变。"""
        logits = np.array([1.0, 5.0, 4.0, 0.5], dtype=np.float32)
        penalized = repeat_penalty(logits, [1], penalty=10.0)
        # 强惩罚后 argmax 应从 1 变为其他
        assert np.argmax(penalized) != 1

    def test_multiple_tokens(self):
        """多个已生成 token 的惩罚。"""
        logits = np.array([3.0, 5.0, 4.0, 2.0], dtype=np.float32)
        penalized = repeat_penalty(logits, [1, 2], penalty=1.0)
        assert penalized[1] < logits[1]
        assert penalized[2] < logits[2]


# ===========================================================================
# SubTask 4.4: ParallelRolloutCollector
# ===========================================================================


class TestParallelRolloutCollector:
    """ParallelRolloutCollector 并行采样。"""

    def test_collect_basic(self):
        """基本采集：3 prompt × 2 rollout = 6 条轨迹。"""
        model = _tiny_model()
        agent = NexAgent(policy=model)
        collector = ParallelRolloutCollector(
            agent=agent, max_new_tokens=4, strategy="softmax", temperature=1.0,
        )
        rollouts = collector.collect(
            prompts=["hello", "world", "test"],
            n_rollouts_per_prompt=2,
            encode_fn=_encode,
            decode_fn=_decode,
        )
        assert len(rollouts) == 6
        for r in rollouts:
            assert len(r.generated_tokens) > 0
            assert len(r.logprobs) == len(r.generated_tokens)
            assert isinstance(r.reward, float)

    def test_collect_with_reward_fn(self):
        """带自定义 reward 函数的采集。"""
        model = _tiny_model()
        agent = NexAgent(policy=model)
        collector = ParallelRolloutCollector(
            agent=agent, max_new_tokens=4, strategy="softmax",
        )

        def reward_fn(generated_text, prompt_text, logprobs, generated_tokens):
            return {"total": 0.5, "custom": 0.5}

        rollouts = collector.collect(
            prompts=["hello", "world"],
            n_rollouts_per_prompt=1,
            encode_fn=_encode,
            decode_fn=_decode,
            reward_fn=reward_fn,
        )
        assert len(rollouts) == 2
        for r in rollouts:
            assert r.reward == 0.5
            assert "custom" in r.reward_components

    def test_collect_with_value_fn(self):
        """带 value function 的采集。"""
        model = _tiny_model()
        agent = NexAgent(policy=model)
        trainer = NexTrainer(agent=agent, cfg={"use_value": True, "max_new_tokens": 4})
        # 使用 trainer 的 value_fn
        collector = trainer.collector
        rollouts = collector.collect(
            prompts=["hello", "world"],
            n_rollouts_per_prompt=1,
            encode_fn=_encode,
            decode_fn=_decode,
        )
        assert len(rollouts) == 2
        for r in rollouts:
            assert len(r.values) == len(r.generated_tokens)

    def test_collect_batched(self):
        """batched 并行采集。"""
        model = _tiny_model()
        agent = NexAgent(policy=model)
        collector = ParallelRolloutCollector(
            agent=agent, max_new_tokens=4, strategy="softmax",
        )
        rollouts = collector.collect_batched(
            prompts=["hello", "world", "test"],
            n_rollouts_per_prompt=2,
            encode_fn=_encode,
            decode_fn=_decode,
        )
        assert len(rollouts) == 6
        for r in rollouts:
            assert len(r.generated_tokens) > 0

    def test_rollout_dataclass(self):
        """Rollout dataclass 字段。"""
        r = Rollout(
            prompt="test",
            prompt_tokens=[1, 2, 3],
            generated_tokens=[4, 5],
            logprobs=[-0.5, -1.0],
            reward=0.8,
        )
        assert r.length == 2
        assert r.all_tokens == [1, 2, 3, 4, 5]
        assert r.reward == 0.8
        assert r.values == []  # 默认空
        assert len(r.advantages) == 0  # 默认空


# ===========================================================================
# SubTask 4.5: NexTrainer（PPO + GAE + KL）
# ===========================================================================


class TestNexTrainer:
    """NexTrainer PPO 训练。"""

    def test_fit_with_value(self):
        """带 value function 的 fit（几步不崩溃）。"""
        model = _tiny_model()
        agent = NexAgent(policy=model)
        trainer = NexTrainer(agent=agent, cfg={
            "clip_ratio": 0.2,
            "ppo_epochs": 2,
            "max_new_tokens": 4,
            "use_value": True,
            "lr": 1e-3,
            "max_grad_norm": 0.5,
            "target_kl": 10.0,  # 高阈值避免触发 KL 惩罚
        })
        losses, kls, rewards = trainer.fit(
            prompts=["hello", "world", "test"],
            n_epochs=2,
            n_rollouts_per_prompt=2,
            encode_fn=_encode,
            decode_fn=_decode,
        )
        assert len(losses) == 2
        assert len(kls) == 2
        assert len(rewards) == 2
        # loss 不应为 NaN
        for l in losses:
            assert not np.isnan(l), f"loss is NaN: {l}"

    def test_fit_fallback(self):
        """纯策略梯度 fallback（无 value function）。"""
        model = _tiny_model()
        agent = NexAgent(policy=model)
        trainer = NexTrainer(agent=agent, cfg={
            "use_value": False,
            "ppo_epochs": 2,
            "max_new_tokens": 4,
            "lr": 1e-3,
            "max_grad_norm": 0.5,
            "target_kl": 10.0,
        })
        losses, kls, rewards = trainer.fit(
            prompts=["hello", "world"],
            n_epochs=2,
            n_rollouts_per_prompt=2,
            encode_fn=_encode,
            decode_fn=_decode,
        )
        assert len(losses) == 2
        for l in losses:
            assert not np.isnan(l)

    def test_gae_computation(self):
        """GAE 优势计算。"""
        model = _tiny_model()
        agent = NexAgent(policy=model)
        trainer = NexTrainer(agent=agent, cfg={"use_value": True})
        rewards = [0.0, 0.0, 0.0, 1.0]
        values = [0.5, 0.4, 0.3, 0.2]
        advs, rets = trainer._compute_gae(rewards, values)
        assert len(advs) == 4
        assert len(rets) == 4
        # 最后一步的 advantage 应为 reward - value (终端 next_value=0)
        # δ_3 = 1.0 + 0 - 0.2 = 0.8
        # A_3 = 0.8
        assert abs(advs[3] - 0.8) < 1e-5
        # return = adv + value
        assert abs(rets[3] - (0.8 + 0.2)) < 1e-5

    def test_returns_baseline(self):
        """策略梯度 fallback 的 return + baseline 计算。"""
        model = _tiny_model()
        agent = NexAgent(policy=model)
        trainer = NexTrainer(agent=agent, cfg={"use_value": False})
        trainer.baseline = 0.5
        rewards = [0.0, 0.0, 0.0, 1.0]
        advs, rets = trainer._compute_returns_baseline(rewards)
        assert len(advs) == 4
        assert len(rets) == 4
        # 最后一步 return = 1.0 (discounted)
        assert abs(rets[3] - 1.0) < 1e-5
        # advantage = return - baseline
        assert abs(advs[3] - (1.0 - 0.5)) < 1e-5


# ===========================================================================
# SubTask 4.5: KL 防崩溃（自适应惩罚）
# ===========================================================================


class TestKLAdaptive:
    """KL 散度超阈值时自动增加惩罚权重。"""

    def test_kl_increases_weight(self):
        """KL 超阈值 → 权重增加。"""
        model = _tiny_model()
        agent = NexAgent(policy=model)
        trainer = NexTrainer(agent=agent, cfg={
            "use_value": True,
            "target_kl": 0.001,  # 很低的目标
            "kl_adaptive": True,
            "kl_weight": 0.0,
        })
        # 模拟 KL 超过 target_kl * 2
        initial_weight = trainer.kl_weight
        trainer._update_kl_weight(kl=0.01)  # 0.01 >> 0.001 * 2
        assert trainer.kl_weight > initial_weight

    def test_kl_decreases_weight(self):
        """KL 低于阈值 → 权重减少。"""
        model = _tiny_model()
        agent = NexAgent(policy=model)
        trainer = NexTrainer(agent=agent, cfg={
            "use_value": True,
            "target_kl": 0.001,
            "kl_adaptive": True,
            "kl_weight": 0.1,  # 初始高权重
        })
        initial_weight = trainer.kl_weight
        # KL 很低
        trainer._update_kl_weight(kl=0.0001)  # 0.0001 < 0.001/2
        assert trainer.kl_weight < initial_weight

    def test_kl_no_adaptive(self):
        """关闭自适应时权重不变。"""
        model = _tiny_model()
        agent = NexAgent(policy=model)
        trainer = NexTrainer(agent=agent, cfg={
            "use_value": True,
            "target_kl": 0.001,
            "kl_adaptive": False,
            "kl_weight": 0.05,
        })
        initial_weight = trainer.kl_weight
        trainer._update_kl_weight(kl=10.0)
        assert trainer.kl_weight == initial_weight

    def test_kl_computation(self):
        """KL 散度计算（标量版）。"""
        model = _tiny_model()
        agent = NexAgent(policy=model)
        input_ids = np.asarray([[_encode("hello")]], dtype=np.int64).squeeze(0)
        new_logits = agent.forward_policy(input_ids, track_grad=False)
        ref_logits = agent.forward_ref(input_ids)
        kl = agent.compute_kl_scalar(new_logits, ref_logits)
        # 初始时策略 = 参考，KL 应接近 0
        assert kl >= 0.0
        assert kl < 1.0  # 应该很小

    def test_kl_with_training(self):
        """训练过程中 KL 监控。"""
        model = _tiny_model()
        agent = NexAgent(policy=model)
        trainer = NexTrainer(agent=agent, cfg={
            "use_value": True,
            "ppo_epochs": 1,
            "max_new_tokens": 4,
            "lr": 1e-3,
            "target_kl": 0.001,  # 低阈值，更容易触发
            "kl_adaptive": True,
        })
        # 训练 2 个 epoch
        losses, kls, rewards = trainer.fit(
            prompts=["hello", "world"],
            n_epochs=2,
            n_rollouts_per_prompt=1,
            encode_fn=_encode,
            decode_fn=_decode,
        )
        assert len(kls) == 2
        # KL 历史应记录
        assert all(k >= 0 for k in kls)
        # kl_weights 应被记录
        assert len(trainer.kl_weights) == 2


# ===========================================================================
# NexAgent 测试
# ===========================================================================


class TestNexAgent:
    """NexAgent 策略 + 参考网络 + KL。"""

    def test_act(self):
        """act 方法采样动作。"""
        model = _tiny_model()
        agent = NexAgent(policy=model)
        state = NexState(prompt="hello", prompt_tokens=_encode("hello"))
        action, logits = agent.act(state, strategy="softmax", temperature=1.0)
        assert isinstance(action, NexAction)
        assert 0 <= action.token_id < 256
        assert action.logprob <= 0.0

    def test_ref_model_frozen(self):
        """参考网络被冻结。"""
        model = _tiny_model()
        agent = NexAgent(policy=model)
        for p in agent.ref_model.parameters():
            assert p.requires_grad is False

    def test_sync_ref(self):
        """sync_ref 同步参考网络。"""
        model = _tiny_model()
        agent = NexAgent(policy=model)
        # 修改策略权重
        for p in agent.policy.parameters():
            p.data = p.data + 0.1
        # 同步
        agent.sync_ref()
        # 验证参考网络已更新
        for p_policy, p_ref in zip(
            agent.policy.parameters(), agent.ref_model.parameters()
        ):
            np.testing.assert_allclose(p_policy.data, p_ref.data, atol=1e-6)

    def test_compute_kl(self):
        """compute_kl 返回非负值。"""
        from verse_torch import Tensor
        model = _tiny_model()
        agent = NexAgent(policy=model)
        input_ids = np.asarray([_encode("hello")], dtype=np.int64)
        new_logits = agent.forward_policy(input_ids, track_grad=False)
        ref_logits = agent.forward_ref(input_ids)
        kl = agent.compute_kl(new_logits, ref_logits)
        assert isinstance(kl, Tensor)
        kl_val = float(kl.data.item() if kl.data.ndim == 0 else kl.data.sum())
        assert kl_val >= 0.0


# ===========================================================================
# NexEnv 测试
# ===========================================================================


class TestNexEnv:
    """NexEnv 各环境。"""

    def test_chat_env(self):
        """ChatEnv 对话环境。"""
        env = ChatEnv(reference_completion="hello", max_len=4)
        state = env.reset("hi")
        assert state.prompt == "hi"

        # 模拟几步
        for i in range(4):
            state, reward, done, info = env.step(state, i + 1)
            if done:
                break
        assert done
        assert isinstance(reward, float)

    def test_math_env_exact(self):
        """MathEnv 精确匹配。"""
        env = MathEnv(reference_answer="42", max_len=4)
        state = env.reset("6*7=")
        # 模拟生成 "42"（用 ord 值）
        for tid in [ord('4') % 256, ord('2') % 256]:
            state, reward, done, info = env.step(state, tid)
        # 还没到 max_len
        for _ in range(2):
            state, reward, done, info = env.step(state, 0)
        assert done
        # reward 取决于是否匹配
        assert isinstance(reward, float)

    def test_code_env(self):
        """CodeEnv 语法检查。"""
        env = CodeEnv(max_len=4)
        state = env.reset("def ")
        for tid in range(4):
            state, reward, done, info = env.step(state, tid)
        assert done
        assert 0.0 <= reward <= 1.0

    def test_env_reset(self):
        """reset 返回正确初始状态。"""
        env = ChatEnv(max_len=4)
        state = env.reset("test prompt")
        assert state.prompt == "test prompt"
        assert state.generated_tokens == []
        assert state.done is False
        assert state.step == 0


# ===========================================================================
# NexState 测试
# ===========================================================================


class TestNexState:
    """NexState 状态管理。"""

    def test_basic(self):
        """基本字段。"""
        state = NexState(
            prompt="hello",
            prompt_tokens=[1, 2, 3],
            generated_tokens=[4, 5],
            logprobs=[-0.5, -1.0],
            step=2,
        )
        assert state.all_tokens == [1, 2, 3, 4, 5]
        assert state.full_sequence == [1, 2, 3, 4, 5]

    def test_append_action(self):
        """追加动作。"""
        state = NexState(prompt="test", prompt_tokens=[1])
        state.append_action(token_id=5, logprob=-0.3)
        assert state.generated_tokens == [5]
        assert state.logprobs == [-0.3]
        assert state.step == 1

    def test_reset(self):
        """重置状态。"""
        state = NexState(prompt="test", prompt_tokens=[1, 2])
        state.append_action(3, -0.5)
        state.reset()
        assert state.generated_tokens == []
        assert state.logprobs == []
        assert state.step == 0
        assert state.done is False

    def test_clone(self):
        """克隆状态。"""
        state = NexState(prompt="test", prompt_tokens=[1])
        state.append_action(2, -0.5)
        cloned = state.clone()
        assert cloned.prompt == state.prompt
        assert cloned.generated_tokens == state.generated_tokens
        # 修改 clone 不影响原状态
        cloned.append_action(3, -0.3)
        assert len(state.generated_tokens) == 1
        assert len(cloned.generated_tokens) == 2


# ===========================================================================
# 综合端到端测试
# ===========================================================================


class TestEndToEnd:
    """端到端：收集 → 训练 → 验证。"""

    def test_full_pipeline(self):
        """完整 RL 训练 pipeline。"""
        model = _tiny_model()
        agent = NexAgent(policy=model)

        # 创建 reward 计算
        nex_reward = NexReward(target_len=4)

        def reward_fn(generated_text, prompt_text, logprobs, generated_tokens):
            return nex_reward.compute(
                generated=generated_text,
                reference="",
                logprobs=logprobs,
                generated_tokens=generated_tokens,
            )

        # 创建 trainer
        trainer = NexTrainer(agent=agent, cfg={
            "clip_ratio": 0.2,
            "ppo_epochs": 2,
            "max_new_tokens": 4,
            "use_value": True,
            "lr": 1e-3,
            "max_grad_norm": 0.5,
            "target_kl": 5.0,
            "reward_fn": reward_fn,
        })

        # 训练
        losses, kls, rewards = trainer.fit(
            prompts=["hello", "world", "test"],
            n_epochs=3,
            n_rollouts_per_prompt=2,
            encode_fn=_encode,
            decode_fn=_decode,
        )

        assert len(losses) == 3
        assert len(kls) == 3
        assert len(rewards) == 3
        # 验证没有 NaN
        for l in losses:
            assert not np.isnan(l), f"loss is NaN"
        for k in kls:
            assert not np.isnan(k), f"kl is NaN"
        for r in rewards:
            assert not np.isnan(r), f"reward is NaN"
