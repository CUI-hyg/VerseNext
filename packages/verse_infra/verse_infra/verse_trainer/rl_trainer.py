"""RLTrainer: 包装 :class:`verse_nex.nexrl.NexTrainer`（Part4K1 Task 6.6）。

对接 ``verse-posttrain --rl nexrl``。本类只做"模型 + tokenizer + RL 训练
编排"的胶水工作，底层 PPO + GAE + KL 自适应全部复用
``verse_nex.nexrl.NexTrainer``，不重写算法。

设计目标
========
1. **零侵入**：不修改 ``NexTrainer``，只构造 ``NexAgent`` + 适配
   encode/decode 函数后调用 ``NexTrainer.fit``。
2. **可独立使用**：``RLTrainer`` 不依赖任何 CLI / config 文件，
   只需传入模型 + tokenizer + prompts 即可训练。
3. **CPU-first**：与 VerseTorch 保持一致，无 GPU 依赖。
"""

from __future__ import annotations

import os
import warnings
from typing import Any, Callable, List, Optional, Tuple

import numpy as np


class RLTrainer:
    """RL 后训练器：包装 :class:`verse_nex.nexrl.NexTrainer`。

    工作流程：
    1. 把策略模型包装为 :class:`verse_nex.nexrl.NexAgent`（含冻结的参考网络）
    2. 构造 :class:`verse_nex.nexrl.NexTrainer`（PPO + GAE + KL 自适应）
    3. 用 tokenizer 的 encode/decode 适配 ``NexTrainer.fit`` 的
       ``encode_fn`` / ``decode_fn`` 参数
    4. 可选：训练后保存模型到 checkpoint

    Args:
        model: 策略模型（通常为 :class:`verse_nex.cometspark.CometSparkNexLM`
               或 :class:`verse_nex.VerseNexLM`）
        tokenizer: tokenizer 对象（需有 ``encode`` / ``decode``）
        cfg: RL 训练配置 dict，传递给 ``NexTrainer``。常用字段：
            - clip_ratio: PPO clip 比率（默认 0.2）
            - gamma: 折扣因子（默认 0.99）
            - gae_lambda: GAE lambda（默认 0.95）
            - ppo_epochs: PPO 更新轮数（默认 4）
            - lr: 学习率（默认 1e-4）
            - max_new_tokens: rollout 最大生成 token 数（默认 16）
            - use_value: 是否启用 value function（默认 True）
            - target_kl: KL 目标阈值（默认 0.02）
            - kl_adaptive: 是否自适应 KL（默认 True）
        reward_fn: 自定义 reward 函数；None 用 NexTrainer 默认
        save_dir: checkpoint 保存目录（None 不保存）

    用法:
        >>> from verse_nex import VerseNexLM
        >>> from verse_tokenizer import ByteTokenizer
        >>> model = VerseNexLM(vocab_size=259, dim=32, n_layer=2)
        >>> tok = ByteTokenizer()
        >>> trainer = RLTrainer(model, tok, cfg={"ppo_epochs": 2})
        >>> trainer.fit(prompts=["1+1=", "你好"], n_epochs=2)
    """

    def __init__(
        self,
        model,
        tokenizer=None,
        cfg: Optional[dict] = None,
        reward_fn: Optional[Callable] = None,
        save_dir: Optional[str] = None,
    ):
        # 延迟导入 NexTrainer / NexAgent，避免 verse_nex 不可用时整个包无法 import
        from verse_nex.nexrl import NexAgent, NexTrainer

        self.model = model
        self.tokenizer = tokenizer
        self.cfg = dict(cfg) if cfg is not None else {}
        if reward_fn is not None:
            self.cfg["reward_fn"] = reward_fn
        self.save_dir = save_dir

        # 1. 构造 NexAgent（内部会深拷贝 policy 作为冻结的 ref_model）
        self.agent = NexAgent(policy=model)

        # 2. 构造 NexTrainer
        self.nex_trainer = NexTrainer(agent=self.agent, cfg=self.cfg)

        # 训练历史（fit 后填充）
        self.train_losses: List[float] = []
        self.kl_history: List[float] = []
        self.reward_history: List[float] = []

    # ------------------------------------------------------------------
    # encode / decode 适配
    # ------------------------------------------------------------------

    def _make_encode_fn(self) -> Callable[[str], List[int]]:
        """构造 encode_fn：text -> list[int]，适配 NexTrainer.fit。"""
        tok = self.tokenizer
        if tok is None:
            # 无 tokenizer：用 ord(c) % 256 兜底
            def _encode(text: str) -> List[int]:
                return [ord(c) % 256 for c in text]
            return _encode

        def _encode(text: str) -> List[int]:
            try:
                return list(tok.encode(text, add_special_tokens=False))
            except TypeError:
                try:
                    return list(tok.encode(text))
                except Exception:
                    return [ord(c) % 256 for c in text]
        return _encode

    def _make_decode_fn(self) -> Callable[[List[int]], str]:
        """构造 decode_fn：list[int] -> text，适配 NexTrainer.fit。"""
        tok = self.tokenizer
        if tok is None:
            def _decode(tokens: List[int]) -> str:
                return "".join(chr(int(t) % 256) for t in tokens)
            return _decode

        def _decode(tokens: List[int]) -> str:
            try:
                return tok.decode(list(tokens))
            except TypeError:
                try:
                    return tok.decode(list(tokens), strip_special=True)
                except Exception:
                    return "".join(chr(int(t) % 256) for t in tokens)
        return _decode

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------

    def fit(
        self,
        prompts: List[str],
        n_epochs: int = 10,
        n_rollouts_per_prompt: int = 2,
    ) -> Tuple[List[float], List[float], List[float]]:
        """RL 训练主入口。

        Args:
            prompts: prompt 文本列表
            n_epochs: 训练 epoch 数（默认 10）
            n_rollouts_per_prompt: 每个 prompt 的 rollout 数（默认 2）
        Returns:
            (train_losses, kl_history, reward_history) 三元组
        """
        if not prompts:
            warnings.warn("RLTrainer.fit 收到空 prompts 列表，跳过训练")
            return self.train_losses, self.kl_history, self.reward_history

        encode_fn = self._make_encode_fn()
        decode_fn = self._make_decode_fn()

        print(
            f"[RLTrainer] 开始 RL 训练：n_epochs={n_epochs} "
            f"n_prompts={len(prompts)} n_rollouts={n_rollouts_per_prompt}",
            flush=True,
        )

        losses, kls, rewards = self.nex_trainer.fit(
            prompts=prompts,
            n_epochs=n_epochs,
            n_rollouts_per_prompt=n_rollouts_per_prompt,
            encode_fn=encode_fn,
            decode_fn=decode_fn,
        )
        self.train_losses = list(losses)
        self.kl_history = list(kls)
        self.reward_history = list(rewards)

        # 可选保存：保存策略模型 state_dict
        if self.save_dir is not None:
            try:
                os.makedirs(self.save_dir, exist_ok=True)
                save_path = os.path.join(self.save_dir, "rl_policy.pt")
                if hasattr(self.model, "state_dict"):
                    import pickle
                    payload = {
                        "model_state_dict": self.model.state_dict(),
                        "train_losses": self.train_losses,
                        "kl_history": self.kl_history,
                        "reward_history": self.reward_history,
                    }
                    with open(save_path, "wb") as f:
                        pickle.dump(payload, f)
                    print(f"[RLTrainer] 策略模型已保存到 {save_path}", flush=True)
            except Exception as e:
                print(f"[RLTrainer] 警告：保存模型失败：{e}", flush=True)

        return self.train_losses, self.kl_history, self.reward_history

    # ------------------------------------------------------------------
    # 便捷访问
    # ------------------------------------------------------------------

    @property
    def kl_weight(self) -> float:
        """当前 KL 惩罚权重（来自 NexTrainer）。"""
        return float(self.nex_trainer.kl_weight)

    @property
    def baseline(self) -> float:
        """策略梯度 fallback 的 baseline（来自 NexTrainer）。"""
        return float(self.nex_trainer.baseline)


__all__ = ["RLTrainer"]
