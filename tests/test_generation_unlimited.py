"""Part4K2 Task 3：生成输出优化（不限制 token 数）测试。

测试覆盖：
1. CometSparkNexLM.generate：
   - max_new_tokens=None 时生成到 EOS 自然停止
   - max_safe_limit 安全上限生效（无 EOS 输出时在 100K 停止）
   - max_new_tokens 指定时按值生成（兼容旧调用）
   - greedy + recurrent 路径支持 EOS 提前停止
   - sampling 路径支持 EOS 提前停止
2. CometSparkV05LM.generate：
   - 默认不限制（max_new_tokens=None）
3. StreamingGenerator.generate：
   - 默认不限制（max_new_tokens=None）
   - max_safe_limit 安全上限生效
4. generate_with_template 方法可用（CometSparkV05LM）
5. 旧调用方式兼容（max_new_tokens=10 仍工作）

运行：
    cd /workspace && python -m pytest tests/test_generation_unlimited.py -x -q
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# sys.path 注入
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parent.parent
for _pkg in ("verse_torch", "verse_nex", "verse_infra"):
    _p = _REPO_ROOT / "packages" / _pkg
    if _p.is_dir() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


# ---------------------------------------------------------------------------
# Mock 模型：可控制生成 EOS 的时机
# ---------------------------------------------------------------------------


class _MockRecurrentModel:
    """可控 mock 模型：forward_recurrent 在第 N 步生成 EOS。

    用于测试 CometSparkNexLM.generate / StreamingGenerator.generate 的
    EOS 提前停止 + max_safe_limit 安全上限逻辑。

    Args:
        vocab_size: 词表大小
        eos_id: EOS token id
        eos_after: 第几次调用 forward_recurrent 时返回 logits.argmax == eos_id
            （None 表示永不输出 EOS，用于测试 max_safe_limit）
    """

    def __init__(
        self,
        vocab_size: int = 16,
        eos_id: int = 0,
        eos_after: int = 5,
    ):
        self.vocab_size = vocab_size
        self.eos_id = eos_id
        self.eos_after = eos_after
        self._call_count = 0
        # 占位属性，供 StreamingGenerator._vocab_size 推断
        self.lm_head = None
        self.embed = None

    def eval(self):
        return self

    def train(self, mode: bool = True):
        return self

    def forward_recurrent(self, input_ids, states=None):
        """单步递推：在第 eos_after 次后输出 eos_id。

        Returns:
            (logits_tensor, new_states)
        """
        # 延迟导入 Tensor（与 verse_nex 内部实现一致）
        from verse_torch import Tensor

        self._call_count += 1
        B = 1
        V = self.vocab_size
        # 构造 logits：让 argmax 落在期望的 token id
        if self.eos_after is not None and self._call_count >= self.eos_after:
            # 输出 eos_id
            next_tok = self.eos_id
        else:
            # 输出非 eos_id（用 (eos_id + 1) % vocab_size 避免与 eos_id 冲突）
            next_tok = (self.eos_id + 1) % self.vocab_size
            if next_tok == self.eos_id:
                next_tok = (self.eos_id + 2) % self.vocab_size

        logits_np = np.zeros((B, 1, V), dtype=np.float32)
        logits_np[0, 0, next_tok] = 100.0  # 让 argmax 必中
        return Tensor(logits_np, requires_grad=False), states


class _MockForwardModel(_MockRecurrentModel):
    """可控 mock 模型：forward（整序列）也支持，用于采样路径测试。"""

    def forward(self, idx):
        from verse_torch import Tensor

        self._call_count += 1
        if isinstance(idx, Tensor):
            arr = idx.data
        else:
            arr = np.asarray(idx)
        B, T = arr.shape[0], arr.shape[1]
        V = self.vocab_size

        if self.eos_after is not None and self._call_count >= self.eos_after:
            next_tok = self.eos_id
        else:
            next_tok = (self.eos_id + 1) % self.vocab_size
            if next_tok == self.eos_id:
                next_tok = (self.eos_id + 2) % self.vocab_size

        logits_np = np.zeros((B, T, V), dtype=np.float32)
        # 仅最后一个位置的 logits 决定下一步 token
        logits_np[:, -1, next_tok] = 100.0
        return Tensor(logits_np, requires_grad=False)


# ---------------------------------------------------------------------------
# 1. CometSparkNexLM.generate：max_new_tokens=None + EOS 提前停止
# ---------------------------------------------------------------------------


def test_nex_generate_unlimited_stops_at_eos():
    """CometSparkNexLM.generate max_new_tokens=None 时生成到 EOS 自然停止。

    使用 _MockRecurrentModel 替换内部 net，让模型在第 5 步生成 EOS。
    验证：
    - 生成的 token 数 < max_safe_limit（说明提前停止了）
    - 末尾含 EOS token
    """
    from verse_nex.cometspark import CometSparkNexLM

    # 构造 tiny 模型，替换为 mock 行为
    # 直接构造一个最小 CometSparkNexLM 实例并替换 forward_recurrent
    try:
        model = CometSparkNexLM(
            vocab_size=16, dim=8, n_layer=1, n_head=2,
            layer_pattern=["trisparse"],
            max_seq_len=32,
            num_dense_parts=2, num_experts_per_part=2, top_k=1,
        )
    except Exception as e:
        pytest.skip(f"构造 tiny CometSparkNexLM 失败：{e}")

    mock = _MockRecurrentModel(vocab_size=16, eos_id=0, eos_after=5)
    # 把 forward_recurrent 替换为 mock 版本（保持 self 绑定）
    model.forward_recurrent = mock.forward_recurrent

    idx = np.array([[1, 2, 3]], dtype=np.int64)
    # max_new_tokens=None：依赖 EOS 提前停止
    out = model.generate(idx, max_new_tokens=None, eos_id=0, max_safe_limit=100_000)

    # 验证：生成在 EOS 处停止，不达到 max_safe_limit
    # mock 在第 5 次 forward_recurrent 时返回 eos_id
    # 前 3 次 forward_recurrent 用于 prompt 预热（_generate_recurrent）
    # 第 4 次开始生成第一个 token，第 5 次生成第二个 token = eos_id
    n_generated = out.shape[1] - 3  # 减去 prompt 长度
    assert n_generated < 100_000, "生成未在 EOS 处停止"
    # 末尾（去掉强制追加的 eos）应该是 eos
    # generate 强制追加 eos，所以末尾必为 eos_id
    assert out[0, -1] == 0, f"末尾应为 eos_id=0，实际 {out[0, -1]}"


def test_nex_generate_unlimited_max_safe_limit():
    """max_safe_limit 生效：mock 永不输出 EOS，验证在 max_safe_limit 停止。"""
    from verse_nex.cometspark import CometSparkNexLM

    try:
        model = CometSparkNexLM(
            vocab_size=16, dim=8, n_layer=1, n_head=2,
            layer_pattern=["trisparse"],
            max_seq_len=32,
            num_dense_parts=2, num_experts_per_part=2, top_k=1,
        )
    except Exception as e:
        pytest.skip(f"构造 tiny CometSparkNexLM 失败：{e}")

    # eos_after=None：永不输出 EOS
    mock = _MockRecurrentModel(vocab_size=16, eos_id=0, eos_after=None)
    model.forward_recurrent = mock.forward_recurrent

    idx = np.array([[1, 2, 3]], dtype=np.int64)
    # 用较小的 max_safe_limit（如 20）加速测试
    out = model.generate(
        idx, max_new_tokens=None, eos_id=0, max_safe_limit=20
    )
    # 生成的 token 数应 ≤ max_safe_limit（20）+ 强制追加 eos（1）
    # 注：generate 末尾会强制追加 eos_id 以确保完整 UTF-8 边界
    n_generated = out.shape[1] - 3
    assert n_generated <= 21, (
        f"max_safe_limit=20 未生效，实际生成 {n_generated} 个 token"
    )


def test_nex_generate_max_new_tokens_compat():
    """max_new_tokens=10 仍按值生成（兼容旧调用）。"""
    from verse_nex.cometspark import CometSparkNexLM

    try:
        model = CometSparkNexLM(
            vocab_size=16, dim=8, n_layer=1, n_head=2,
            layer_pattern=["trisparse"],
            max_seq_len=32,
            num_dense_parts=2, num_experts_per_part=2, top_k=1,
        )
    except Exception as e:
        pytest.skip(f"构造 tiny CometSparkNexLM 失败：{e}")

    # 永不输出 EOS，强制按 max_new_tokens 限制
    mock = _MockRecurrentModel(vocab_size=16, eos_id=0, eos_after=None)
    model.forward_recurrent = mock.forward_recurrent

    idx = np.array([[1, 2, 3]], dtype=np.int64)
    out = model.generate(idx, max_new_tokens=10, eos_id=0)
    # 应正好生成 10 个新 token（不含强制追加的 eos）
    # 注：generate 会在末尾追加 eos_id，所以总长度 = 3 + 10 + 1
    assert out.shape == (1, 3 + 10 + 1), (
        f"max_new_tokens=10 兼容性失败，shape={out.shape}"
    )


def test_nex_generate_sampling_path_eos_stop():
    """采样路径（temperature != 1.0）也支持 EOS 提前停止。"""
    from verse_nex.cometspark import CometSparkNexLM

    try:
        model = CometSparkNexLM(
            vocab_size=16, dim=8, n_layer=1, n_head=2,
            layer_pattern=["trisparse"],
            max_seq_len=32,
            num_dense_parts=2, num_experts_per_part=2, top_k=1,
        )
    except Exception as e:
        pytest.skip(f"构造 tiny CometSparkNexLM 失败：{e}")

    # 替换 forward 为 mock 版本
    mock = _MockForwardModel(vocab_size=16, eos_id=0, eos_after=3)
    model.forward = mock.forward

    idx = np.array([[1, 2, 3]], dtype=np.int64)
    # temperature=0.5 走采样路径
    out = model.generate(
        idx, max_new_tokens=None, temperature=0.5, top_k=None,
        eos_id=0, max_safe_limit=20,
    )
    n_generated = out.shape[1] - 3
    # mock 在第 3 次 forward 调用时输出 eos_id
    # 由于无 prompt 预热（_generate_with_logits 直接 forward 整序列），
    # 第一次 forward 已包含 prompt + 第一个生成 token
    # 期望生成 ≤ 3 个 token 后停止
    assert n_generated <= 5, (
        f"采样路径 EOS 提前停止失败，实际生成 {n_generated} 个 token"
    )


# ---------------------------------------------------------------------------
# 2. CometSparkV05LM.generate：默认不限制
# ---------------------------------------------------------------------------


def test_v05_generate_default_unlimited_signature():
    """CometSparkV05LM.generate 默认 max_new_tokens=None + max_safe_limit 参数。"""
    from spark.src.base_model import CometSparkV05LM, CometSparkV05Small
    import inspect

    sig = inspect.signature(CometSparkV05LM.generate)
    params = sig.parameters

    # max_new_tokens 默认值应为 None
    assert params["max_new_tokens"].default is None, (
        f"CometSparkV05LM.generate max_new_tokens 默认值应为 None，"
        f"实际 {params['max_new_tokens'].default!r}"
    )
    # 应有 max_safe_limit 参数，默认 100_000
    assert "max_safe_limit" in params, "CometSparkV05LM.generate 缺少 max_safe_limit 参数"
    assert params["max_safe_limit"].default == 100_000, (
        f"max_safe_limit 默认值应为 100_000，"
        f"实际 {params['max_safe_limit'].default!r}"
    )


def test_v05_generate_unlimited_via_mock():
    """CometSparkV05LM.generate max_new_tokens=None 时通过 EOS 提前停止。

    替换内部 net.generate 为可控 mock，验证 EOS 提前停止行为。
    """
    from spark.src.base_model import CometSparkV05Small

    model = CometSparkV05Small()
    # 用 mock 替换 net.generate
    mock_calls = {"count": 0}

    def mock_generate(idx, max_new_tokens=None, temperature=1.0, top_k=None,
                      eos_id=None, max_safe_limit=100_000):
        # 模拟 CometSparkNexLM.generate 的行为
        from verse_torch import Tensor
        if isinstance(idx, Tensor):
            idx_np = idx.data
        else:
            idx_np = np.asarray(idx)
        if idx_np.ndim == 1:
            idx_np = idx_np[None, :]
        idx_np = idx_np.astype(np.int64)

        cur = idx_np.copy()
        # 模拟生成 3 个 token 后输出 eos
        if max_new_tokens is None:
            limit = max_safe_limit
        else:
            limit = int(max_new_tokens)

        for i in range(limit):
            if i == 3 and eos_id is not None:
                next_tok = np.array([eos_id], dtype=np.int64)
            else:
                next_tok = np.array([1], dtype=np.int64)
            cur = np.concatenate([cur, next_tok[:, None]], axis=1)
            if eos_id is not None and next_tok[0] == eos_id:
                break
        return cur

    model.net.generate = mock_generate

    idx = np.array([[1, 2, 3]], dtype=np.int64)
    out = model.generate(idx, max_new_tokens=None, eos_id=0, max_safe_limit=100_000)
    n_generated = out.shape[1] - 3
    assert n_generated <= 4, (
        f"CometSparkV05LM EOS 提前停止失败，实际生成 {n_generated} 个 token"
    )


def test_v05_generate_max_new_tokens_compat():
    """CometSparkV05LM.generate max_new_tokens=10 仍按值生成（兼容旧调用）。"""
    from spark.src.base_model import CometSparkV05Small

    model = CometSparkV05Small()
    idx = np.array([[1, 2, 3]], dtype=np.int64)
    out = model.generate(idx, max_new_tokens=10, temperature=1.0, top_k=None)
    # 旧调用：max_new_tokens=10，生成 ≤ 10 个 token（可能提前停止）
    n_generated = out.shape[1] - 3
    assert n_generated <= 10, (
        f"max_new_tokens=10 兼容性失败，实际生成 {n_generated} 个 token"
    )


# ---------------------------------------------------------------------------
# 3. StreamingGenerator：默认不限制
# ---------------------------------------------------------------------------


def test_streaming_generator_default_unlimited_signature():
    """StreamingGenerator.generate 默认 max_new_tokens=None。"""
    from verse_infra.verse_inference.generator import StreamingGenerator
    import inspect

    sig = inspect.signature(StreamingGenerator.generate)
    params = sig.parameters
    assert params["max_new_tokens"].default is None, (
        f"StreamingGenerator.generate max_new_tokens 默认值应为 None，"
        f"实际 {params['max_new_tokens'].default!r}"
    )
    assert "max_safe_limit" in params, (
        "StreamingGenerator.generate 缺少 max_safe_limit 参数"
    )
    assert params["max_safe_limit"].default == 100_000


def test_streaming_generator_unlimited_stops_at_eos():
    """StreamingGenerator max_new_tokens=None 时通过 EOS 提前停止。"""
    from verse_infra.verse_inference.generator import StreamingGenerator

    # mock 模型：在第 5 次 forward_recurrent 后输出 eos
    mock_model = _MockRecurrentModel(vocab_size=16, eos_id=0, eos_after=5)
    gen = StreamingGenerator(mock_model)

    prompt_ids = [1, 2, 3]
    # max_new_tokens=None：依赖 EOS 提前停止
    tokens = list(gen.generate(
        prompt_ids, max_new_tokens=None, eos_token_id=0, max_safe_limit=100_000
    ))
    # 生成应在 EOS 处停止，token 数远小于 100_000
    assert len(tokens) < 100_000, (
        f"StreamingGenerator EOS 提前停止失败，生成了 {len(tokens)} 个 token"
    )
    # 末尾应为 eos_token_id
    assert tokens[-1] == 0, f"末尾应为 eos_token_id=0，实际 {tokens[-1]}"


def test_streaming_generator_max_safe_limit():
    """StreamingGenerator max_safe_limit 生效：无 EOS 时在 max_safe_limit 停止。"""
    from verse_infra.verse_inference.generator import StreamingGenerator

    # mock 永不输出 EOS
    mock_model = _MockRecurrentModel(vocab_size=16, eos_id=0, eos_after=None)
    gen = StreamingGenerator(mock_model)

    prompt_ids = [1, 2, 3]
    # max_safe_limit=10 加速测试
    tokens = list(gen.generate(
        prompt_ids, max_new_tokens=None, eos_token_id=0, max_safe_limit=10
    ))
    assert len(tokens) <= 10, (
        f"max_safe_limit=10 未生效，实际生成 {len(tokens)} 个 token"
    )


def test_streaming_generator_max_new_tokens_compat():
    """StreamingGenerator max_new_tokens=5 仍按值生成（兼容旧调用）。"""
    from verse_infra.verse_inference.generator import StreamingGenerator

    mock_model = _MockRecurrentModel(vocab_size=16, eos_id=0, eos_after=None)
    gen = StreamingGenerator(mock_model)

    prompt_ids = [1, 2, 3]
    tokens = list(gen.generate(prompt_ids, max_new_tokens=5, eos_token_id=0))
    assert len(tokens) == 5, (
        f"max_new_tokens=5 兼容性失败，实际生成 {len(tokens)} 个 token"
    )


# ---------------------------------------------------------------------------
# 4. generate_with_template 方法
# ---------------------------------------------------------------------------


class _MockTokenizer:
    """Mock tokenizer：用于 generate_with_template 测试。

    - apply_chat_template(messages, tools=None, add_generation_prompt=True)
      返回渲染后的字符串
    - encode(text, add_special_tokens=False) 返回 id 列表
    - decode(ids, strip_special=True) 返回字符串
    - eos_id 属性
    """

    def __init__(self, eos_id: int = 0):
        self.eos_id = eos_id
        self.vocab = {"<|im_end|>": eos_id}

    def apply_chat_template(self, messages, tools=None, add_generation_prompt=False):
        # 简单 ChatML 渲染
        parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            parts.append(f"<|im_start|>{role}\n{content}<|im_end|>\n")
        if add_generation_prompt:
            parts.append("<|im_start|>assistant\n")
        return "".join(parts)

    def encode(self, text, add_special_tokens=False):
        # 简单按字符 id 映射
        return [ord(c) % 256 for c in text]

    def decode(self, ids, strip_special=True):
        # 反向映射
        return "".join(chr(i % 256) for i in ids)


def test_generate_with_template_method_exists():
    """CometSparkV05LM 有 generate_with_template 方法。"""
    from spark.src.base_model import CometSparkV05LM
    assert hasattr(CometSparkV05LM, "generate_with_template"), (
        "CometSparkV05LM 缺少 generate_with_template 方法"
    )


def test_generate_with_template_returns_string():
    """generate_with_template 返回字符串（用 mock tokenizer + mock net）。"""
    from spark.src.base_model import CometSparkV05Small

    model = CometSparkV05Small()

    # mock net.generate 返回固定序列
    def mock_generate(idx, max_new_tokens=None, temperature=1.0, top_k=None,
                      eos_id=None, max_safe_limit=100_000):
        from verse_torch import Tensor
        if isinstance(idx, Tensor):
            idx_np = idx.data
        else:
            idx_np = np.asarray(idx)
        if idx_np.ndim == 1:
            idx_np = idx_np[None, :]
        idx_np = idx_np.astype(np.int64)
        # 生成 3 个 token：[1, 2, eos_id]
        new_tokens = np.array([[1, 2, int(eos_id) if eos_id is not None else 3]],
                              dtype=np.int64)
        return np.concatenate([idx_np, new_tokens], axis=1)

    model.net.generate = mock_generate

    tokenizer = _MockTokenizer(eos_id=0)
    messages = [{"role": "user", "content": "hello"}]

    out = model.generate_with_template(messages, tokenizer, max_new_tokens=None)
    # 应返回字符串
    assert isinstance(out, str), f"返回类型应为 str，实际 {type(out)}"
    # 应包含解码后的内容（非空）
    # mock net.generate 返回 [1, 2, 0]（eos_id=0），decode 后是 \x01\x02\x00
    # 用 mock 的 decode 规则：chr(i % 256)
    assert len(out) > 0, "返回字符串应为非空"


def test_generate_with_template_compat_with_old_tokenizer():
    """generate_with_template 兼容旧 tokenizer API（apply_chat_template 不接受 add_generation_prompt）。"""
    from spark.src.base_model import CometSparkV05Small


    class _OldTokenizer:
        """旧 tokenizer：apply_chat_template 只接受 messages 参数。"""
        eos_id = 0
        vocab = {"<|eos|>": 0}

        def apply_chat_template(self, messages):
            # 旧 API
            return "".join(
                f"<|{m.get('role', 'user')}|>{m.get('content', '')}"
                for m in messages
            )

        def encode(self, text, add_special_tokens=False):
            return [ord(c) % 256 for c in text]

        def decode(self, ids, strip_special=True):
            return "".join(chr(i % 256) for i in ids)

    model = CometSparkV05Small()

    # mock net.generate
    def mock_generate(idx, max_new_tokens=None, temperature=1.0, top_k=None,
                      eos_id=None, max_safe_limit=100_000):
        from verse_torch import Tensor
        if isinstance(idx, Tensor):
            idx_np = idx.data
        else:
            idx_np = np.asarray(idx)
        if idx_np.ndim == 1:
            idx_np = idx_np[None, :]
        idx_np = idx_np.astype(np.int64)
        new_tokens = np.array([[1]], dtype=np.int64)
        return np.concatenate([idx_np, new_tokens], axis=1)

    model.net.generate = mock_generate

    tokenizer = _OldTokenizer()
    messages = [{"role": "user", "content": "hi"}]

    # 不应抛 TypeError
    out = model.generate_with_template(messages, tokenizer, max_new_tokens=5)
    assert isinstance(out, str)


# ---------------------------------------------------------------------------
# 5. 旧调用方式兼容性
# ---------------------------------------------------------------------------


def test_legacy_max_new_tokens_eq_10_works_nex():
    """CometSparkNexLM.generate(max_new_tokens=10) 仍工作（兼容旧调用）。"""
    from verse_nex.cometspark import CometSparkNexLM

    try:
        model = CometSparkNexLM(
            vocab_size=16, dim=8, n_layer=1, n_head=2,
            layer_pattern=["trisparse"],
            max_seq_len=32,
            num_dense_parts=2, num_experts_per_part=2, top_k=1,
        )
    except Exception as e:
        pytest.skip(f"构造 tiny CometSparkNexLM 失败：{e}")

    idx = np.array([[1, 2, 3]], dtype=np.int64)
    out = model.generate(idx, max_new_tokens=10, temperature=1.0, top_k=None)
    n_generated = out.shape[1] - 3
    # 应生成 ≤ 10 个 token（不强制 == 10，因为可能内部有其他停止条件）
    assert n_generated <= 10, (
        f"max_new_tokens=10 兼容性失败，实际生成 {n_generated} 个 token"
    )


def test_legacy_max_new_tokens_eq_10_works_v05():
    """CometSparkV05LM.generate(max_new_tokens=10) 仍工作（兼容旧调用）。"""
    from spark.src.base_model import CometSparkV05Small

    model = CometSparkV05Small()
    idx = np.array([[1, 2, 3]], dtype=np.int64)
    out = model.generate(idx, max_new_tokens=10, temperature=1.0, top_k=None)
    n_generated = out.shape[1] - 3
    assert n_generated <= 10, (
        f"max_new_tokens=10 兼容性失败，实际生成 {n_generated} 个 token"
    )


def test_legacy_streaming_max_new_tokens_eq_5_works():
    """StreamingGenerator.generate(max_new_tokens=5) 仍工作（兼容旧调用）。"""
    from verse_infra.verse_inference.generator import StreamingGenerator

    # 用真实 tiny 模型
    try:
        from verse_nex import CometSparkNexLM
        mock_model = CometSparkNexLM(
            vocab_size=16, dim=8, n_layer=1, n_head=2,
            layer_pattern=["trisparse"],
            max_seq_len=32,
            num_dense_parts=2, num_experts_per_part=2, top_k=1,
        )
    except Exception:
        # 降级用 mock
        mock_model = _MockRecurrentModel(vocab_size=16, eos_id=0, eos_after=None)

    gen = StreamingGenerator(mock_model)
    prompt_ids = [1, 2, 3]
    tokens = list(gen.generate(prompt_ids, max_new_tokens=5))
    assert len(tokens) == 5, (
        f"max_new_tokens=5 兼容性失败，实际生成 {len(tokens)} 个 token"
    )


# ---------------------------------------------------------------------------
# 6. verse-eval CLI 默认 max_tokens=None
# ---------------------------------------------------------------------------


def test_cli_eval_max_tokens_default_none():
    """verse-eval CLI --max-tokens 默认值为 None。"""
    from verse_infra.verse_trainer.cli import _build_eval_parser

    parser = _build_eval_parser()
    # 用空 args 解析（仅 --config 必填）
    args = parser.parse_args(["--config", "dummy.yml"])
    assert args.max_tokens is None, (
        f"--max-tokens 默认值应为 None，实际 {args.max_tokens!r}"
    )


def test_cli_eval_max_tokens_explicit_value():
    """verse-eval CLI --max-tokens 显式指定值时正常解析。"""
    from verse_infra.verse_trainer.cli import _build_eval_parser

    parser = _build_eval_parser()
    args = parser.parse_args(["--config", "dummy.yml", "--max-tokens", "50"])
    assert args.max_tokens == 50
