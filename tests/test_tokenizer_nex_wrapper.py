"""Task 5.4: NexTokenizerWrapper 测试。

覆盖：
1. NexTokenizerWrapper 包装 BPETokenizer 基本功能
2. reward-weighted 训练：高 reward 子串优先成 token
3. token 命中率提升（高 reward 样本的子串更可能成为 token）
4. encode 时记录 reward + get_token_rewards 统计
5. nexrl_integration=False 退化为标准 BPE
6. dict 形式 rewards 支持
7. NexReward 集成（可选）

运行方式：
    cd /workspace && python -m pytest tests/test_tokenizer_nex_wrapper.py -v
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import pytest

# 让 tests/ 目录能 import verse_infra.verse_tokenizer
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "packages" / "verse_infra"))

from verse_infra.verse_tokenizer import BPETokenizer, NexTokenizerWrapper
from verse_infra.verse_tokenizer.nex_wrapper import NexTokenizerWrapper as _NW


# ---------------------------------------------------------------------------
# 测试语料：构造频率不同且 reward 不同的样本
# ---------------------------------------------------------------------------
# "ab ab ab" (reward=0.0)：('a','b') pair freq=3
# "xy xy"    (reward=1.0)：('x','y') pair freq=2
# 标准 BPE 选 freq 最高的 ('a','b')；
# reward-weighted 时 score('x','y')=2*(1+1*1)=4 > score('a','b')=3*(1+1*0)=3，
# 选 ('x','y')（高 reward 加权后胜出）。
_CORPUS = ["ab ab ab", "xy xy"]
_REWARDS = [0.0, 1.0]


# ===========================================================================
# 1. 基本包装功能
# ===========================================================================


class TestNexWrapperBasic:
    """NexTokenizerWrapper 基本包装功能测试。"""

    def test_construct_with_base_tokenizer(self):
        """传入底层 BPETokenizer 构造包装器。"""
        base = BPETokenizer.train(["hello world test"], vocab_size=200)
        wrapper = NexTokenizerWrapper(base_tokenizer=base)
        # 透传属性
        assert wrapper.vocab is base.vocab
        assert wrapper.merge_ranks is base.merge_ranks
        assert wrapper.special_tokens is base.special_tokens
        assert len(wrapper) == len(base)

    def test_construct_default_base_tokenizer(self):
        """未传入底层 tokenizer 时构造空 BPETokenizer。"""
        wrapper = NexTokenizerWrapper()
        assert isinstance(wrapper.base, BPETokenizer)
        # 空 BPETokenizer 的 vocab 为空（train 后才填充 256 基础字节字符）
        assert len(wrapper) == 0
        assert wrapper.vocab == {}

    def test_encode_decode_roundtrip(self):
        """encode/decode 往返（委托底层）。"""
        wrapper = NexTokenizerWrapper()
        wrapper.train_with_rewards(_CORPUS, _REWARDS, vocab_size=300)
        for text in ["xy", "ab", "hello"]:
            ids = wrapper.encode(text, add_special_tokens=False)
            decoded = wrapper.decode(ids)
            assert isinstance(ids, list)
            assert all(isinstance(i, int) for i in ids)
            # decode 不产生乱码
            assert "\ufffd" not in decoded, (
                f"decode 产生 U+FFFD：{text!r} → {decoded!r}"
            )

    def test_encode_with_special_tokens(self):
        """encode 默认加 bos/eos。"""
        wrapper = NexTokenizerWrapper()
        wrapper.train_with_rewards(_CORPUS, _REWARDS, vocab_size=300)
        ids_with = wrapper.encode("xy", add_special_tokens=True)
        ids_without = wrapper.encode("xy", add_special_tokens=False)
        assert len(ids_with) >= len(ids_without)

    def test_nexrl_integration_param(self):
        """nexrl_integration 构造参数可读。"""
        w_on = NexTokenizerWrapper(nexrl_integration=True)
        w_off = NexTokenizerWrapper(nexrl_integration=False)
        assert w_on.nexrl_integration is True
        assert w_off.nexrl_integration is False

    def test_alpha_param(self):
        """alpha 构造参数可读。"""
        w = NexTokenizerWrapper(alpha=2.5)
        assert w.alpha == 2.5

    def test_class_imported_from_both_paths(self):
        """NexTokenizerWrapper 可从 verse_tokenizer 和子模块导入。"""
        from verse_infra.verse_tokenizer import NexTokenizerWrapper as Nw1
        from verse_infra.verse_tokenizer.nex_wrapper import NexTokenizerWrapper as Nw2
        assert Nw1 is Nw2
        assert _NW is Nw2


# ===========================================================================
# 2. reward-weighted 训练：高 reward 子串优先成 token
# ===========================================================================


class TestRewardWeightedTraining:
    """reward 加权影响 BPE merge 选择顺序测试。"""

    def test_high_reward_pair_merged_first(self):
        """高 reward 的 pair 应优先被合并（即使频率更低）。

        语料 "ab ab ab"(reward=0.0) 和 "xy xy"(reward=1.0)：
        - ('a','b') freq=3，('x','y') freq=2；
        - 标准 BPE 选 freq 最高的 ('a','b')；
        - reward-weighted 时 score('x','y')=4 > score('a','b')=3，选 ('x','y')。
        """
        wrapper = NexTokenizerWrapper(nexrl_integration=True, alpha=1.0)
        wrapper.train_with_rewards(_CORPUS, _REWARDS, vocab_size=300, min_frequency=1)

        # 第一个 merge（rank=0）应是 ('x', 'y')
        first_pair = min(wrapper.merge_ranks.items(), key=lambda x: x[1])[0]
        assert first_pair == ("x", "y"), (
            f"reward-weighted 训练后第一个 merge 应是 ('x','y')，"
            f"实际是 {first_pair}"
        )

    def test_standard_bpe_picks_highest_freq_pair(self):
        """标准 BPE（nexrl_integration=False）选频率最高的 pair。

        作为对照：freq('a','b')=3 > freq('x','y')=2，标准 BPE 选 ('a','b')。
        """
        wrapper = NexTokenizerWrapper(nexrl_integration=False)
        wrapper.train_with_rewards(_CORPUS, _REWARDS, vocab_size=300, min_frequency=1)

        first_pair = min(wrapper.merge_ranks.items(), key=lambda x: x[1])[0]
        # 标准 BPE 选 freq 最高的 ('a','b')
        assert first_pair == ("a", "b"), (
            f"标准 BPE 第一个 merge 应是 ('a','b')，实际是 {first_pair}"
        )

    def test_reward_changes_merge_order(self):
        """reward 加权确实改变 merge 顺序。"""
        w_on = NexTokenizerWrapper(nexrl_integration=True, alpha=1.0)
        w_on.train_with_rewards(_CORPUS, _REWARDS, vocab_size=300, min_frequency=1)

        w_off = NexTokenizerWrapper(nexrl_integration=False)
        w_off.train_with_rewards(_CORPUS, _REWARDS, vocab_size=300, min_frequency=1)

        # 第一个 merge 应不同
        first_on = min(w_on.merge_ranks.items(), key=lambda x: x[1])[0]
        first_off = min(w_off.merge_ranks.items(), key=lambda x: x[1])[0]
        assert first_on != first_off, (
            f"reward 加权应改变 merge 顺序，但都是 {first_on}"
        )

    def test_alpha_zero_degrades_to_standard_bpe(self):
        """alpha=0 时退化为标准 BPE（reward 不影响选择）。"""
        w_zero = NexTokenizerWrapper(nexrl_integration=True, alpha=0.0)
        w_zero.train_with_rewards(_CORPUS, _REWARDS, vocab_size=300, min_frequency=1)

        w_std = NexTokenizerWrapper(nexrl_integration=False)
        w_std.train_with_rewards(_CORPUS, _REWARDS, vocab_size=300, min_frequency=1)

        # alpha=0 应与标准 BPE 一致
        assert w_zero.merge_ranks == w_std.merge_ranks, (
            "alpha=0 应退化为标准 BPE，但 merge_ranks 不一致"
        )

    def test_dict_rewards_supported(self):
        """支持 dict 形式（键为样本字符串）。"""
        rewards_dict = {
            "ab ab ab": 0.0,
            "xy xy": 1.0,
        }
        wrapper = NexTokenizerWrapper(nexrl_integration=True, alpha=1.0)
        wrapper.train_with_rewards(_CORPUS, rewards_dict, vocab_size=300, min_frequency=1)

        # dict 形式应与 list 形式产生相同的 merge 顺序
        first_pair = min(wrapper.merge_ranks.items(), key=lambda x: x[1])[0]
        assert first_pair == ("x", "y")

    def test_rewards_length_mismatch_raises(self):
        """rewards 长度与 corpus 不一致时报错。"""
        wrapper = NexTokenizerWrapper()
        with pytest.raises(ValueError, match="不一致"):
            wrapper.train_with_rewards(
                _CORPUS, [1.0, 0.0, 0.5], vocab_size=300
            )


# ===========================================================================
# 3. token 命中率提升
# ===========================================================================


class TestTokenHitRate:
    """高 reward 子串更可能成为独立 token，提升 encode 命中率。"""

    def test_high_reward_substring_fewer_tokens(self):
        """限制 vocab_size 时，高 reward 子串用更少 token 编码。

        vocab_size=268 → target_merges=1，只合并 1 个 pair：
        - reward-weighted：合并 ('x','y')，"xy" → 1 token；
        - 标准 BPE：合并 freq 最高的 ('a','b')，"xy" → 2 token。
        """
        w_on = NexTokenizerWrapper(nexrl_integration=True, alpha=1.0)
        w_on.train_with_rewards(_CORPUS, _REWARDS, vocab_size=268, min_frequency=1)

        w_off = NexTokenizerWrapper(nexrl_integration=False)
        w_off.train_with_rewards(_CORPUS, _REWARDS, vocab_size=268, min_frequency=1)

        # 高 reward 的 "xy" 在 reward-weighted 下应编码为更少 token
        ids_on = w_on.encode("xy", add_special_tokens=False)
        ids_off = w_off.encode("xy", add_special_tokens=False)
        assert len(ids_on) < len(ids_off), (
            f"高 reward 子串 'xy' 在 reward-weighted 下应编码为更少 token，"
            f"但 reward-weighted={len(ids_on)}，标准={len(ids_off)}"
        )

    def test_low_reward_substring_not_prioritized(self):
        """低 reward 子串 'ab' 在 reward-weighted 下不被优先合并。"""
        w_on = NexTokenizerWrapper(nexrl_integration=True, alpha=1.0)
        w_on.train_with_rewards(_CORPUS, _REWARDS, vocab_size=268, min_frequency=1)

        # vocab_size=268 只合并 1 个 pair，且是 ('x','y')
        # 所以 "ab" 仍是 2 token（'a', 'b'）
        ids_ab = w_on.encode("ab", add_special_tokens=False)
        assert len(ids_ab) == 2, (
            f"低 reward 子串 'ab' 应未被合并（2 token），实际 {len(ids_ab)} token"
        )

    def test_more_vocab_merges_both(self):
        """vocab_size 足够大时两种子串都会被合并（但顺序不同）。"""
        w_on = NexTokenizerWrapper(nexrl_integration=True, alpha=1.0)
        w_on.train_with_rewards(_CORPUS, _REWARDS, vocab_size=300, min_frequency=1)

        # vocab 足够大，两个 pair 都会被合并
        ids_xy = w_on.encode("xy", add_special_tokens=False)
        ids_ab = w_on.encode("ab", add_special_tokens=False)
        # 都应编码为单 token
        assert len(ids_xy) == 1, f"'xy' 应编码为 1 token，实际 {len(ids_xy)}"
        assert len(ids_ab) == 1, f"'ab' 应编码为 1 token，实际 {len(ids_ab)}"


# ===========================================================================
# 4. encode 时记录 reward + get_token_rewards 统计
# ===========================================================================


class TestTokenRewardTracking:
    """encode 时记录 reward + get_token_rewards 统计测试。"""

    def test_encode_without_reward_no_stats(self):
        """encode 不传 reward 时不记录统计。"""
        wrapper = NexTokenizerWrapper()
        wrapper.train_with_rewards(_CORPUS, _REWARDS, vocab_size=300)
        wrapper.encode("xy xy xy", add_special_tokens=False)
        rewards = wrapper.get_token_rewards()
        assert rewards == {}, "未传 reward 时统计应为空"

    def test_encode_with_reward_records_stats(self):
        """encode 传 reward 时记录到 token 统计。"""
        wrapper = NexTokenizerWrapper()
        wrapper.train_with_rewards(_CORPUS, _REWARDS, vocab_size=300)
        ids = wrapper.encode("xy", reward=0.5, add_special_tokens=False)

        rewards = wrapper.get_token_rewards()
        assert len(rewards) == len(set(ids)), "每个 unique token id 都应有统计"
        for tid in set(ids):
            assert tid in rewards
            stat = rewards[tid]
            # "xy" 编码为单 token，count=1
            assert stat["total_reward"] == 0.5
            assert stat["count"] >= 1
            assert stat["avg_reward"] == 0.5

    def test_multiple_encodes_accumulate_reward(self):
        """多次 encode 累加 reward。"""
        wrapper = NexTokenizerWrapper()
        wrapper.train_with_rewards(_CORPUS, _REWARDS, vocab_size=300)
        # "xy" 编码为单 token，连续 encode 3 次
        wrapper.encode("xy", reward=1.0, add_special_tokens=False)
        wrapper.encode("xy", reward=1.0, add_special_tokens=False)
        wrapper.encode("xy", reward=2.0, add_special_tokens=False)

        rewards = wrapper.get_token_rewards()
        # "xy" 对应的 token id 应有累计 reward = 4.0
        total_rewards = sum(s["total_reward"] for s in rewards.values())
        assert total_rewards == 4.0, f"累计 reward 应为 4.0，实际 {total_rewards}"

    def test_reset_token_rewards(self):
        """reset_token_rewards 清空统计。"""
        wrapper = NexTokenizerWrapper()
        wrapper.train_with_rewards(_CORPUS, _REWARDS, vocab_size=300)
        wrapper.encode("xy", reward=0.5, add_special_tokens=False)
        assert wrapper.get_token_rewards() != {}

        wrapper.reset_token_rewards()
        assert wrapper.get_token_rewards() == {}

    def test_get_token_rewards_returns_dict_with_required_keys(self):
        """get_token_rewards 返回的 dict 含 total_reward/count/avg_reward。"""
        wrapper = NexTokenizerWrapper()
        wrapper.train_with_rewards(_CORPUS, _REWARDS, vocab_size=300)
        wrapper.encode("xy", reward=0.5, add_special_tokens=False)
        rewards = wrapper.get_token_rewards()
        for tid, stat in rewards.items():
            assert "total_reward" in stat
            assert "count" in stat
            assert "avg_reward" in stat
            assert isinstance(stat["total_reward"], float)
            assert isinstance(stat["count"], int)
            assert isinstance(stat["avg_reward"], float)


# ===========================================================================
# 5. nexrl_integration=False 退化为标准 BPE
# ===========================================================================


class TestNexrlIntegrationOff:
    """nexrl_integration=False 退化为标准 BPE 测试。"""

    def test_off_matches_standard_bpe(self):
        """nexrl_integration=False 的训练结果与标准 BPETokenizer.train 一致。"""
        wrapper = NexTokenizerWrapper(nexrl_integration=False)
        wrapper.train_with_rewards(_CORPUS, _REWARDS, vocab_size=300, min_frequency=1)

        std_tok = BPETokenizer.train(_CORPUS, vocab_size=300, min_frequency=1)

        # merge_ranks 应一致（reward 不影响选择）
        assert wrapper.merge_ranks == std_tok.merge_ranks, (
            "nexrl_integration=False 应与标准 BPE 一致"
        )

    def test_off_encode_decode_consistent(self):
        """nexrl_integration=False 时 encode/decode 正常。"""
        wrapper = NexTokenizerWrapper(nexrl_integration=False)
        wrapper.train_with_rewards(_CORPUS, _REWARDS, vocab_size=300, min_frequency=1)
        for text in ["xy xy xy", "ab ab ab", "hello"]:
            ids = wrapper.encode(text, add_special_tokens=False)
            decoded = wrapper.decode(ids)
            assert "\ufffd" not in decoded


# ===========================================================================
# 6. save / load
# ===========================================================================


class TestSaveLoad:
    """save/load 持久化测试。"""

    def test_save_load_roundtrip(self):
        """save/load 往返（底层 BPETokenizer 序列化）。"""
        wrapper = NexTokenizerWrapper(nexrl_integration=True, alpha=1.0)
        wrapper.train_with_rewards(_CORPUS, _REWARDS, vocab_size=300, min_frequency=1)

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            wrapper.save(path)
            loaded = NexTokenizerWrapper.load(path)
            assert len(loaded) == len(wrapper), "reload 后 vocab 大小不一致"
            assert loaded.merge_ranks == wrapper.merge_ranks, "reload 后 merge_ranks 不一致"
            # encode 一致
            ids1 = wrapper.encode("xy", add_special_tokens=False)
            ids2 = loaded.encode("xy", add_special_tokens=False)
            assert ids1 == ids2, "reload 后 encode 结果不一致"
        finally:
            os.unlink(path)


# ===========================================================================
# 7. NexReward 集成（可选）
# ===========================================================================


class TestNexRewardIntegration:
    """NexReward 实例集成测试（可选，不依赖 nexrl 包也能通过）。"""

    def test_nex_reward_param_accepted(self):
        """nex_reward 构造参数被接受（不强制依赖 nexrl 包）。"""
        # 不传入 nex_reward，仅验证参数被接受
        wrapper = NexTokenizerWrapper(nexrl_integration=True, alpha=1.0)
        assert wrapper.nex_reward is None

    def test_nex_reward_instance_integration(self):
        """传入 NexReward 实例时正常工作。

        即使 nexrl 包不可用，本测试也能通过（nex_reward 仅用于 reward 归一化辅助）。
        """
        # 添加 verse_torch / verse_nex 到 sys.path（nexrl 依赖 verse_torch）
        for _sub in ("verse_torch", "verse_nex"):
            _p = str(_REPO_ROOT / "packages" / _sub)
            if _p not in sys.path and os.path.isdir(_p):
                sys.path.insert(0, _p)
        try:
            from verse_nex.nexrl import NexReward
        except ImportError:
            pytest.skip("verse_nex.nexrl 不可用，跳过 NexReward 集成测试")

        nex_reward = NexReward()
        wrapper = NexTokenizerWrapper(
            nexrl_integration=True, alpha=1.0, nex_reward=nex_reward
        )
        wrapper.train_with_rewards(_CORPUS, _REWARDS, vocab_size=300, min_frequency=1)
        # 传入 NexReward 后仍应优先合并高 reward pair
        first_pair = min(wrapper.merge_ranks.items(), key=lambda x: x[1])[0]
        assert first_pair == ("x", "y")


# ===========================================================================
# 8. 与现有测试兼容性（不破坏 BPE 训练）
# ===========================================================================


class TestBackwardCompatibility:
    """确保不破坏现有 BPETokenizer 行为。"""

    def test_wrapper_does_not_mutate_global_state(self):
        """包装器训练不影响全局 BPETokenizer 类。"""
        # 训练前先记录标准 BPE 的 merge_ranks
        std_before = BPETokenizer.train(_CORPUS, vocab_size=300, min_frequency=1)

        # 训练包装器
        wrapper = NexTokenizerWrapper(nexrl_integration=True, alpha=1.0)
        wrapper.train_with_rewards(_CORPUS, _REWARDS, vocab_size=300, min_frequency=1)

        # 训练后标准 BPE 应不变
        std_after = BPETokenizer.train(_CORPUS, vocab_size=300, min_frequency=1)
        assert std_before.merge_ranks == std_after.merge_ranks, (
            "包装器训练不应影响标准 BPETokenizer.train 的结果"
        )

    def test_wrapper_base_is_bpe_tokenizer(self):
        """包装器的 base 属性是 BPETokenizer 实例。"""
        wrapper = NexTokenizerWrapper()
        wrapper.train_with_rewards(_CORPUS, _REWARDS, vocab_size=300)
        assert isinstance(wrapper.base, BPETokenizer)


# ===========================================================================
# 入口
# ===========================================================================


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
