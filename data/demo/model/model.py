"""CometSparkLM: CometSpark-v0.1 语言模型。

支持两种架构：
- arch="hybrid": 基于 ``verse_nex.hybrid.HybridLM``（SSM + Sparse Attention 混合）
- arch="transformer": 基于 ``verse_torch.nn.TransformerLM``（GQA + RoPE Transformer）

对外统一接口：
- ``forward(idx)`` → logits (B, T, vocab)
- ``generate(idx, max_new_tokens, temperature, top_k)`` → idx ndarray
- ``save(path)`` / ``load(path)`` / ``from_pretrained(path)``
"""

from __future__ import annotations

import os
import pickle
from typing import Optional, List

import numpy as np

from verse_torch import Tensor, no_grad
from verse_torch.nn import TransformerLM, Module

from .config import CometSparkConfig


# 延迟导入 HybridLM，避免在不使用 hybrid 时也加载 verse_nex
def _import_hybrid_lm():
    from verse_nex.hybrid import HybridLM
    return HybridLM


class CometSparkLM(Module):
    """CometSpark-v0.1 语言模型（hybrid / transformer 二选一）。

    Args:
        config: :class:`CometSparkConfig` 实例
    """

    def __init__(self, config: CometSparkConfig):
        super().__init__()
        self.config = config
        # 校验 arch
        if config.arch not in ("hybrid", "transformer"):
            raise ValueError(
                f"arch 必须为 'hybrid' 或 'transformer'，得到 {config.arch!r}"
            )

        if config.arch == "hybrid":
            HybridLM = _import_hybrid_lm()
            # HybridLM 实际签名：
            #   HybridLM(vocab_size, dim, n_layers=4, sparse_ratio=0.1,
            #            ssm_kind="mamba2", sparse_kwargs=None,
            #            sparse_placement="spread"|"last", tie_weights=False)
            # 注意：sparse_placement 只支持 "spread" / "last"，无 "interleave"
            self.net = HybridLM(
                vocab_size=config.vocab_size,
                dim=config.n_embd,
                n_layers=config.n_layer,
                sparse_ratio=config.sparse_ratio,
                ssm_kind=config.ssm_kind,
                sparse_placement="spread",
                tie_weights=config.tie_weights,
            )
        else:
            # arch == "transformer"
            self.net = TransformerLM(
                vocab_size=config.vocab_size,
                n_layer=config.n_layer,
                n_head=config.n_head,
                n_embd=config.n_embd,
                seq_len=config.seq_len,
                dropout=config.dropout,
                n_kv_head=config.n_kv_head,
                tie_weights=config.tie_weights,
            )

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------

    def forward(self, idx) -> Tensor:
        """前向计算。

        Args:
            idx: 形状 (B, T) 的整数索引，Tensor / ndarray / list 均可

        Returns:
            logits: Tensor, shape (B, T, vocab_size)
        """
        if not isinstance(idx, Tensor):
            idx = Tensor(np.asarray(idx, dtype=np.int64))
        elif idx.data.dtype != np.int64:
            # 用 .astype 拷贝避免破坏原 Tensor
            idx = Tensor(idx.data.astype(np.int64))
        return self.net(idx)

    # ------------------------------------------------------------------
    # forward_recurrent（推理接口，兼容 StreamingGenerator）
    # ------------------------------------------------------------------

    def forward_recurrent(self, input_ids, states=None):
        """单步递推推理接口，与 ``HybridLM.forward_recurrent`` 兼容。

        使 ``CometSparkLM`` 可直接传给 ``verse_inference.StreamingGenerator``。

        - ``arch="hybrid"``: 委托给内部 ``HybridLM.forward_recurrent``，维护 SSM 状态；
        - ``arch="transformer"``: TransformerLM 无递归状态，直接调用 ``forward``，
          ``new_states`` 始终为 ``None``（每步独立计算，无 KV cache 复用）。

        Args:
            input_ids: ``(B, 1)`` 整数索引，Tensor / ndarray
            states: list of per-layer state，或 None

        Returns:
            logits: Tensor, shape ``(B, 1, vocab_size)``
            new_states: list of per-layer state（transformer arch 返回 None）
        """
        # hybrid arch: 内部 net 是 HybridLM，原生支持 forward_recurrent
        if hasattr(self.net, "forward_recurrent"):
            return self.net.forward_recurrent(input_ids, states)
        # transformer arch: 无递归状态，直接 forward
        logits = self.forward(input_ids)  # (B, 1, vocab_size)
        return logits, None

    # ------------------------------------------------------------------
    # generate
    # ------------------------------------------------------------------

    def generate(
        self,
        idx,
        max_new_tokens: int = 32,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
    ) -> np.ndarray:
        """自回归生成。

        默认走 ``self.net.generate``（HybridLM 内置 greedy 实现，使用 recurrent 状态）。
        若 net 没有 generate 方法或希望支持 temperature / top_k，则使用本类的
        ``_generate_with_logits`` 实现。

        Args:
            idx: prompt 序列，shape (B, T_prompt) 或 (T_prompt,)
            max_new_tokens: 最大生成 token 数
            temperature: 采样温度；1.0 表示 greedy 等价（与 argmax 一致），
                        > 1 增加随机性，< 1 收敛
            top_k: 仅在 top-k 中采样；None 表示无限制
        Returns:
            generated: ndarray, shape (B, T_prompt + max_new_tokens)
        """
        # 统一 idx 为 2D ndarray
        if isinstance(idx, Tensor):
            idx_np = idx.data
        else:
            idx_np = np.asarray(idx)
        if idx_np.ndim == 1:
            idx_np = idx_np[None, :]  # (1, T)
        idx_np = idx_np.astype(np.int64)

        # 若 net 有 generate 且不需要采样，直接走 net.generate（更快，使用 recurrent 状态）
        if (
            temperature == 1.0
            and top_k is None
            and hasattr(self.net, "generate")
            and callable(getattr(self.net, "generate"))
        ):
            with no_grad():
                self.eval()
                out = self.net.generate(
                    Tensor(idx_np), max_new_tokens=max_new_tokens, mode="recurrent"
                )
            if isinstance(out, Tensor):
                return out.data
            return np.asarray(out)

        # 否则用 forward 循环 + 采样
        return self._generate_with_logits(
            idx_np, max_new_tokens, temperature, top_k
        )

    def _generate_with_logits(
        self,
        idx_np: np.ndarray,
        max_new_tokens: int,
        temperature: float,
        top_k: Optional[int],
    ) -> np.ndarray:
        """基于 forward 的循环生成（支持 temperature / top_k 采样）。"""
        rng = np.random.default_rng()
        with no_grad():
            self.eval()
            cur = idx_np.copy()
            for _ in range(max_new_tokens):
                # 只取最后 seq_len 个 token 防止上下文过长
                T = cur.shape[1]
                context_len = self.config.seq_len
                if T > context_len:
                    inp = cur[:, -context_len:]
                else:
                    inp = cur
                logits = self.net(Tensor(inp))  # (B, T_in, vocab)
                logits_np = logits.data[:, -1, :]  # (B, vocab)
                if temperature <= 0:
                    # 纯 greedy
                    next_tok = logits_np.argmax(axis=-1)
                else:
                    scaled = logits_np / max(temperature, 1e-8)
                    if top_k is not None and top_k > 0:
                        k = min(top_k, scaled.shape[-1])
                        # 取每行 top_k 的索引
                        top_idx = np.argpartition(-scaled, kth=k - 1, axis=-1)[:, :k]
                        # 在 top_k 中按 softmax 采样
                        for b in range(scaled.shape[0]):
                            vals = scaled[b, top_idx[b]]
                            probs = _softmax(vals)
                            choice = rng.choice(len(top_idx[b]), p=probs)
                            # 保证下面统一赋值
                            if b == 0:
                                next_tok = np.zeros(scaled.shape[0], dtype=np.int64)
                            next_tok[b] = top_idx[b, choice]
                    else:
                        probs = _softmax(scaled)
                        next_tok = np.array(
                            [rng.choice(scaled.shape[-1], p=probs[b])
                             for b in range(scaled.shape[0])],
                            dtype=np.int64,
                        )
                cur = np.concatenate([cur, next_tok[:, None]], axis=1)
        return cur

    # ------------------------------------------------------------------
    # save / load
    # ------------------------------------------------------------------

    def state_dict(self) -> dict:
        """返回 {name: ndarray} 参数字典（递归）。"""
        return self.net.state_dict()

    def load_state_dict(self, sd: dict, strict: bool = True):
        """加载参数到 net。"""
        return self.net.load_state_dict(sd, strict=strict)

    def save(self, path: str) -> None:
        """把 config + state_dict 序列化到 path（pickle 格式）。

        保存内容：
            {
                "config": config.to_dict(),
                "state_dict": {name: ndarray},
                "arch": config.arch,
            }
        """
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        payload = {
            "config": self.config.to_dict(),
            "state_dict": {k: np.asarray(v) for k, v in self.state_dict().items()},
            "arch": self.config.arch,
        }
        with open(path, "wb") as f:
            pickle.dump(payload, f)

    def load(self, path: str) -> "CometSparkLM":
        """从 path 加载 state_dict 到当前模型（config 不变）。"""
        with open(path, "rb") as f:
            payload = pickle.load(f)
        sd = payload["state_dict"] if "state_dict" in payload else payload
        self.load_state_dict(sd, strict=False)
        return self

    @classmethod
    def from_pretrained(cls, path: str) -> "CometSparkLM":
        """从 path 加载完整模型（含 config + 权重）。

        Args:
            path: .pt 文件路径
        Returns:
            新构造的 :class:`CometSparkLM` 实例，已加载权重
        """
        with open(path, "rb") as f:
            payload = pickle.load(f)
        config = CometSparkConfig.from_dict(payload["config"])
        model = cls(config)
        sd = payload["state_dict"] if "state_dict" in payload else payload
        model.load_state_dict(sd, strict=False)
        return model


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------


def _softmax(x: np.ndarray) -> np.ndarray:
    """数值稳定的 softmax，沿最后一维。"""
    x = x - np.max(x, axis=-1, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=-1, keepdims=True)


__all__ = ["CometSparkLM"]
