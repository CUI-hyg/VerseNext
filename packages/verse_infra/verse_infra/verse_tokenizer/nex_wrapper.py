"""Task 5.4: NexTokenizerWrapper —— 把 NexRL 集成到 VerseTokenizer。

在 token 边界注入 RL 信号（reward-weighted token preference），高频高奖励子串优先成 token。

核心思想
--------
标准 BPE 在每次合并时选择频率最高的 pair；本包装器把 reward 信号注入到合并
选择中：

    pair_score = frequency × (1 + α × avg_reward_of_pair)

其中 avg_reward_of_pair 是该 pair 所在样本的平均 reward。这样：
- 出现在高 reward 样本中的 pair 会被加权，优先合并为 token；
- 后续 encode 时，这些子串更可能成为独立 token（提升 token 命中率）；
- 当 nexrl_integration=False 或 α=0 时，退化为标准 BPE。

主要 API
--------
- :meth:`NexTokenizerWrapper.train_with_rewards`：用 reward 标注的语料训练
- :meth:`NexTokenizerWrapper.encode`：普通 encode，可选记录 reward
- :meth:`NexTokenizerWrapper.decode`：委托底层
- :meth:`NexTokenizerWrapper.get_token_rewards`：返回每个 token 的累计 reward 统计

依赖
----
- :class:`verse_tokenizer.bpe.BPETokenizer`（底层 tokenizer）
- :mod:`verse_nex.nexrl`（可选；``nexrl_integration=True`` 时仅用 NexReward 做
  reward 归一化辅助，未安装也不影响主流程）
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple, Union

from .bpe import (
    BPETokenizer,
    DEFAULT_SPECIAL_TOKENS,
    _BYTE_DECODER,
    _BYTE_ENCODER,
    _byte_encode,
    _count_pair_freq_parallel,
    _gpt4_pre_tokenize,
)


class NexTokenizerWrapper:
    """NexRL 集成到 VerseTokenizer 的包装器。

    在 token 边界注入 RL 信号：高频高奖励子串优先成 token。

    Args:
        base_tokenizer: 底层 tokenizer 实例。``None`` 时构造空 BPETokenizer
            （仅含 256 基础字节字符，无 merges）。
        nexrl_integration: 是否开启 NexRL 集成。``True`` 时
            :meth:`train_with_rewards` 用 reward 加权 pair_score；
            ``False`` 时退化为标准 BPE（pair_score = frequency）。
        alpha: reward 加权系数。pair_score = freq × (1 + α × avg_reward)。
            ``α=0`` 时等价于关闭加权。
        nex_reward: 可选的 ``NexReward`` 实例（来自 ``verse_nex.nexrl``），
            用于 reward 归一化辅助。未传入时跳过归一化。

    Examples:
        >>> from verse_tokenizer import BPETokenizer
        >>> from verse_tokenizer.nex_wrapper import NexTokenizerWrapper
        >>> wrapper = NexTokenizerWrapper(nexrl_integration=True, alpha=1.0)
        >>> corpus = ["ab ab ab", "cd cd cd"]
        >>> rewards = [1.0, 0.0]  # "ab" 样本高 reward
        >>> wrapper.train_with_rewards(corpus, rewards, vocab_size=300)
        >>> # 高 reward 的 "ab" 应优先合并为 token
        >>> ids = wrapper.encode("ab", add_special_tokens=False)
    """

    def __init__(
        self,
        base_tokenizer: Optional[BPETokenizer] = None,
        nexrl_integration: bool = True,
        alpha: float = 1.0,
        nex_reward=None,  # Optional[NexReward]；为避免硬依赖不在签名标注类型
    ):
        # 底层 tokenizer：未提供时构造空 BPETokenizer
        self.base: BPETokenizer = (
            base_tokenizer
            if base_tokenizer is not None
            else BPETokenizer({}, [], byte_level=True)
        )
        self.nexrl_integration: bool = bool(nexrl_integration)
        self.alpha: float = float(alpha)
        self.nex_reward = nex_reward
        # token id → 累计 reward（用于 get_token_rewards）
        self._token_rewards: Dict[int, float] = defaultdict(float)
        # token id → 出现次数
        self._token_counts: Dict[int, int] = defaultdict(int)

    # ------------------------------------------------------------------
    # reward-weighted 训练
    # ------------------------------------------------------------------

    def train_with_rewards(
        self,
        corpus: Union[str, List[str]],
        rewards: Union[List[float], Dict[str, float]],
        vocab_size: int = 300,
        min_frequency: int = 2,
        max_token_length: Optional[int] = None,
        workers: int = 1,
    ) -> "NexTokenizerWrapper":
        """用 reward 标注的语料训练 BPE，高 reward 子串优先成 token。

        算法步骤（基于 :meth:`BPETokenizer.train`，加入 reward 加权）：
            1. pre-tokenize + byte-level 编码，每个 word 关联其所属样本的 reward；
            2. 初始化 vocab 为 256 个基础字节字符；
            3. 每轮迭代统计相邻 pair 的频率与 reward 之和；
            4. 计算 ``pair_score = freq × (1 + α × avg_reward_of_pair)``；
               - ``nexrl_integration=False`` 时 ``pair_score = freq``（标准 BPE）；
            5. 选 score 最高的 pair 合并（同分按 pair 字典序保证可复现）；
            6. 字节边界检查 + max_token_length 检查（与 BPETokenizer.train 一致）；
            7. 训练完成后注册 DEFAULT_SPECIAL_TOKENS。

        Args:
            corpus: 训练语料，``str`` 或 ``List[str]``。
            rewards: 每个样本的 reward。
                - ``List[float]``：长度需与 ``corpus`` 一致；
                - ``Dict[str, float]``：键为样本字符串，值为 reward（未命中默认 0）。
            vocab_size: 目标 vocab 大小（含 256 基础字节 + special tokens）。
            min_frequency: pair 最小出现次数，低于此值不合并。
            max_token_length: token 最大长度（按合并后 unicode 字符数计）。
                ``None`` 表示不限。
            workers: 并行统计 pair 频率的线程数（默认 1=串行）。

        Returns:
            ``self``（支持链式调用）。

        Raises:
            ValueError: ``rewards`` 为 list 且长度与 ``corpus`` 不一致时。
        """
        # 1. 统一 corpus 为 list
        if isinstance(corpus, str):
            corpus_list = [corpus]
        else:
            corpus_list = [str(c) for c in corpus]

        # 2. 统一 rewards 为 list[float]，与 corpus_list 对齐
        if isinstance(rewards, dict):
            # dict: 按样本字符串查找 reward，未命中默认 0
            rewards_list = [float(rewards.get(c, 0.0)) for c in corpus_list]
        else:
            rewards_list = [float(r) for r in rewards]
            if len(rewards_list) != len(corpus_list):
                raise ValueError(
                    f"rewards 长度 ({len(rewards_list)}) 与 corpus 长度 "
                    f"({len(corpus_list)}) 不一致"
                )

        # 3. 可选：reward 归一化到 [0, 1] 区间
        #    避免负 reward 导致 pair_score 为负（反而被排斥）。
        #    nexrl_integration=False 时不需要归一化（pair_score = freq）。
        if self.nexrl_integration and rewards_list:
            min_r = min(rewards_list)
            max_r = max(rewards_list)
            range_r = max_r - min_r
            if range_r > 1e-8:
                rewards_list = [(r - min_r) / range_r for r in rewards_list]
            else:
                # 所有 reward 相同：归一化后全为 0（不影响相对排序）
                rewards_list = [0.0] * len(rewards_list)

        # 4. pre-tokenize + byte-level 编码
        #    每个 word 关联其所属样本的 reward（用于后续 pair reward 统计）
        word_list: List[Tuple[str, ...]] = []
        word_rewards: List[float] = []
        for text, reward in zip(corpus_list, rewards_list):
            pieces = _gpt4_pre_tokenize(text)
            for p in pieces:
                if not p:
                    continue
                chars = tuple(_byte_encode(p))
                if chars:
                    word_list.append(chars)
                    word_rewards.append(reward)

        # 5. 初始化 vocab：256 个基础字节字符
        byte_chars = sorted(set(_BYTE_ENCODER.values()))
        vocab: Dict[str, int] = {ch: i for i, ch in enumerate(byte_chars)}
        merges: List[Tuple[str, str]] = []

        # 训练目标 merges 数 = vocab_size - 256 - len(DEFAULT_SPECIAL_TOKENS)
        target_merges = max(0, vocab_size - 256 - len(DEFAULT_SPECIAL_TOKENS))

        if workers < 1:
            workers = 1

        # 6. 重复合并直到达到目标或无 pair 可合并
        skipped_pairs: set = set()
        while len(merges) < target_merges:
            # 6.1 统计相邻 pair 频率（支持并行分片）
            pair_counts: Counter = _count_pair_freq_parallel(word_list, workers)

            # 6.2 统计每个 pair 的 reward 之和
            #     pair_reward_sum[pair] = sum(reward_of_word for word containing pair)
            #     avg_reward_of_pair = pair_reward_sum[pair] / pair_counts[pair]
            pair_reward_sum: Dict[Tuple[str, str], float] = defaultdict(float)
            for word, reward in zip(word_list, word_rewards):
                for i in range(len(word) - 1):
                    pair_reward_sum[(word[i], word[i + 1])] += reward

            # 移除已跳过的不合法 pair
            for sp in skipped_pairs:
                pair_counts.pop(sp, None)
                pair_reward_sum.pop(sp, None)

            # 6.3 过滤低于 min_frequency 的 pair
            if min_frequency > 1:
                pair_counts = Counter(
                    {p: c for p, c in pair_counts.items() if c >= min_frequency}
                )
            if not pair_counts:
                break

            # 6.4 计算 reward-weighted pair_score
            #     nexrl_integration=True:  pair_score = freq × (1 + α × avg_reward)
            #     nexrl_integration=False: pair_score = freq（标准 BPE）
            if self.nexrl_integration and self.alpha != 0.0:
                pair_scores: Dict[Tuple[str, str], float] = {}
                for pair, freq in pair_counts.items():
                    avg_reward = pair_reward_sum.get(pair, 0.0) / max(freq, 1)
                    pair_scores[pair] = freq * (1.0 + self.alpha * avg_reward)
            else:
                pair_scores = {pair: float(freq) for pair, freq in pair_counts.items()}

            # 6.5 选择 score 最高的 pair；同分时按 pair 字典序保证可复现
            best_pair = max(pair_scores.items(), key=lambda x: (x[1], x[0]))[0]
            best_freq = pair_counts.get(best_pair, 0)
            if best_freq < min(min_frequency, 1):
                break

            # 6.6 字节边界检查——只接受合并后字节序列为合法 UTF-8 的 merge
            #     （与 BPETokenizer.train 一致，避免 decode 产生 U+FFFD 乱码）
            combined_bytes = [
                _BYTE_DECODER[ch]
                for ch in (best_pair[0] + best_pair[1])
                if ch in _BYTE_DECODER
            ]
            try:
                bytes(combined_bytes).decode("utf-8", errors="strict")
            except UnicodeDecodeError:
                skipped_pairs.add(best_pair)
                continue

            # 6.7 max_token_length 检查
            new_token = best_pair[0] + best_pair[1]
            if max_token_length is not None and len(new_token) > max_token_length:
                skipped_pairs.add(best_pair)
                continue

            # 6.8 合并 best_pair 产生新 token
            merges.append(best_pair)
            if new_token not in vocab:
                vocab[new_token] = len(vocab)

            # 6.9 更新 word_list 与 word_rewards（同步合并位置）
            new_word_list: List[Tuple[str, ...]] = []
            new_word_rewards: List[float] = []
            a, b = best_pair
            for word, reward in zip(word_list, word_rewards):
                new_word: List[str] = []
                i = 0
                while i < len(word):
                    if i < len(word) - 1 and word[i] == a and word[i + 1] == b:
                        new_word.append(new_token)
                        i += 2
                    else:
                        new_word.append(word[i])
                        i += 1
                new_word_list.append(tuple(new_word))
                new_word_rewards.append(reward)
            word_list = new_word_list
            word_rewards = new_word_rewards

        # 7. 创建底层 BPETokenizer 实例并注册 special tokens
        merges_str = [f"{a} {b}" for a, b in merges]
        instance = BPETokenizer(vocab, merges_str, special_tokens=None, byte_level=True)
        instance.add_special_tokens(DEFAULT_SPECIAL_TOKENS)
        self.base = instance
        return self

    # ------------------------------------------------------------------
    # encode / decode
    # ------------------------------------------------------------------

    def encode(
        self,
        text: str,
        reward: Optional[float] = None,
        add_special_tokens: Optional[bool] = None,
        **kwargs,
    ) -> List[int]:
        """编码文本为 token id 列表。可选记录 reward 用于后续统计。

        Args:
            text: 输入文本。
            reward: 该样本的 reward（可选）。若提供则累加到对应 token 的
                reward 统计（用于 :meth:`get_token_rewards`）。
            add_special_tokens: 是否加 bos/eos。``None`` 时用底层默认。
            **kwargs: 透传给底层 ``BPETokenizer.encode``。

        Returns:
            token id 列表。
        """
        if add_special_tokens is not None:
            ids = self.base.encode(text, add_special_tokens=add_special_tokens, **kwargs)
        else:
            try:
                ids = self.base.encode(text, **kwargs)
            except TypeError:
                ids = self.base.encode(text)

        # 可选：累加 reward 到 token 统计
        if reward is not None:
            r = float(reward)
            for tid in ids:
                self._token_rewards[int(tid)] += r
                self._token_counts[int(tid)] += 1
        return ids

    def decode(self, ids: List[int]) -> str:
        """解码 token id 列表为字符串（委托底层）。"""
        return self.base.decode(ids)

    # ------------------------------------------------------------------
    # token reward 统计
    # ------------------------------------------------------------------

    def get_token_rewards(self) -> Dict[int, Dict[str, float]]:
        """返回每个 token 的累计 reward 统计。

        Returns:
            ``{token_id: {"total_reward": float, "count": int, "avg_reward": float}}``。
            其中 ``avg_reward = total_reward / count``（count=0 时为 0）。

        Note:
            只有调用 :meth:`encode` 时传入 ``reward`` 参数的 token 才会被统计。
        """
        result: Dict[int, Dict[str, float]] = {}
        for tid, total in self._token_rewards.items():
            cnt = self._token_counts.get(tid, 0)
            avg = total / cnt if cnt > 0 else 0.0
            result[tid] = {
                "total_reward": float(total),
                "count": int(cnt),
                "avg_reward": float(avg),
            }
        return result

    def reset_token_rewards(self) -> None:
        """重置 token reward 统计。"""
        self._token_rewards.clear()
        self._token_counts.clear()

    # ------------------------------------------------------------------
    # 透传底层 tokenizer 属性与方法
    # ------------------------------------------------------------------

    @property
    def vocab(self) -> Dict[str, int]:
        """底层 vocab。"""
        return self.base.vocab

    @property
    def id_to_token(self) -> Dict[int, str]:
        """底层 id→token 反查表。"""
        return self.base.id_to_token

    @property
    def merge_ranks(self) -> Dict[Tuple[str, str], int]:
        """底层 merge_ranks。"""
        return self.base.merge_ranks

    @property
    def special_tokens(self) -> Dict[str, int]:
        """底层 special_tokens。"""
        return self.base.special_tokens

    @property
    def byte_level(self) -> bool:
        """底层是否使用 byte-level 编码。"""
        return self.base.byte_level

    def __len__(self) -> int:
        return len(self.base)

    def save(self, path: str) -> None:
        """保存底层 tokenizer 到 JSON 文件（委托）。"""
        self.base.save(path)

    @classmethod
    def load(
        cls,
        path: str,
        nexrl_integration: bool = True,
        alpha: float = 1.0,
    ) -> "NexTokenizerWrapper":
        """从 JSON 文件加载底层 tokenizer 并包装为 NexTokenizerWrapper。

        Args:
            path: JSON 文件路径（BPETokenizer.save 输出格式）。
            nexrl_integration: 是否开启 NexRL 集成。
            alpha: reward 加权系数。

        Returns:
            加载好的 :class:`NexTokenizerWrapper` 实例。
        """
        base = BPETokenizer.load(path)
        return cls(
            base_tokenizer=base,
            nexrl_integration=nexrl_integration,
            alpha=alpha,
        )


__all__ = ["NexTokenizerWrapper"]
