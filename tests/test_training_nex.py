"""测试 verse_torch.training_nex 模块（VerseNex 训练工具链）。

覆盖：
- VerseNexTrainer: aux_loss-aware 训练路径（含 forward_with_aux 检测与退化路径）
- LoRATrainer: LoRA 包装 / 仅训练 A/B / merge_lora 合并回 base
- SFTTrainer + 辅助函数: chat 数据格式 / ignore_index 屏蔽 / SFTDataset
- DPOTrainer + 辅助函数: 偏好对训练 / reference model 冻结 / DPODataset
- 辅助函数: _messages_to_tokens / _build_sft_sample /
            _log_probs_from_logits / _sum_log_probs_for_response / _dpo_loss

Part4K1 Task 8.9: 模型从 data/demo 迁移到 spark/model，使用 CometSparkV05Small。

运行方式：
    cd /workspace && PYTHONPATH=packages/verse_torch:packages/verse_nex:packages/verse_infra \
        python -m pytest tests/test_training_nex.py -v
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pytest

# PYTHONPATH 适配（Part4K1 Task 8.9: 从 spark/ 加载模型）
_HERE = os.path.dirname(os.path.abspath(__file__))
_WORKSPACE = os.path.dirname(_HERE)
for _sub in ("verse_torch", "verse_nex", "verse_infra"):
    _p = os.path.join(_WORKSPACE, "packages", _sub)
    if os.path.isdir(_p):
        sys.path.insert(0, _p)
# 把 /workspace 加入 sys.path 让 spark 包可导入
if _WORKSPACE not in sys.path:
    sys.path.insert(0, _WORKSPACE)

from verse_torch import (
    Tensor,
    AdamW,
    VerseNexTrainer,
    LoRATrainer,
    SFTTrainer,
    DPOTrainer,
    SFTDataset,
    DPODataset,
    LoRALinear,
)
from verse_torch import nn as vt_nn
from verse_torch.training import BatchLoader
from verse_torch.training_nex import (
    _messages_to_tokens,
    _build_sft_sample,
    _log_probs_from_logits,
    _sum_log_probs_for_response,
    _dpo_loss,
    _sft_collate,
    _dpo_collate,
)
# Part4K1 Task 8.9: 从 spark/model 导入（替代 data/demo/model）
from spark.model.model import CometSparkV05Small as CometSparkV02Small
from spark.model.model import CometSparkV05Small as CometSparkSmall


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class MockTokenizer:
    """简单 mock tokenizer：每个字符映射到 ord(c) % vocab_size。

    提供 SFTDataset / DPODataset 所需的 encode / decode / __len__ 接口。
    """

    def __init__(self, vocab_size: int = 256):
        self.vocab_size = int(vocab_size)

    def encode(self, text: str):
        return [ord(c) % self.vocab_size for c in text]

    def decode(self, ids):
        return "".join(chr(int(i)) for i in ids)

    def __len__(self):
        return self.vocab_size


def _make_lm_batches(vocab_size=64, seq_len=16, n_samples=8,
                     batch_size=2, seed=0):
    """构造 LM (x, y) batch 列表，x/y 均为 (B, T) int64 ndarray。

    y 设为 x 右移一位（next-token 预测语义），仅用于喂入 cross_entropy。
    """
    rng = np.random.RandomState(seed)
    batches = []
    for i in range(0, n_samples, batch_size):
        end = min(i + batch_size, n_samples)
        x = rng.randint(0, vocab_size, size=(end - i, seq_len)).astype(np.int64)
        y = np.concatenate(
            [x[:, 1:],
             rng.randint(0, vocab_size, size=(end - i, 1))],
            axis=1,
        ).astype(np.int64)
        batches.append((x, y))
    return batches


def _write_jsonl(path, items):
    """把 list 写入 jsonl 文件，每行一个 JSON。"""
    with open(path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def _count_module_type(model, cls):
    """统计 model.modules() 中指定类型的数量。"""
    return sum(1 for m in model.modules() if isinstance(m, cls))


def _all_params(model):
    """递归 yield 所有 Tensor 参数（不论 requires_grad）。

    verse_torch.nn.Module.parameters() 仅返回 requires_grad=True 的参数，
    因此需要直接遍历 _parameters 字典以覆盖被冻结的参数。
    """
    for m in model.modules():
        for p in m._parameters.values():
            yield p


# ---------------------------------------------------------------------------
# VerseNexTrainer
# ---------------------------------------------------------------------------


def test_verse_nex_trainer_with_aux(tmp_path):
    """aux 路径：CometSparkV02Small 训练 5 步，验证 train/val_losses 非空 + 历史落盘。"""
    model = CometSparkV02Small(vocab_size=64, seq_len=32)
    train_loader = _make_lm_batches(vocab_size=64, seq_len=16,
                                     n_samples=8, batch_size=2, seed=0)
    val_loader = _make_lm_batches(vocab_size=64, seq_len=16,
                                   n_samples=4, batch_size=2, seed=1)
    opt = AdamW(model.parameters(), lr=1e-3)
    cfg = {
        "max_steps": 5,
        "eval_interval": 2,
        "patience": 10,
        "save_dir": str(tmp_path),
        "grad_accum": 1,
        "log_interval": 100,
        "enable_progress_bar": False,
        "realtime_plot": False,
    }
    trainer = VerseNexTrainer(model, train_loader, val_loader, opt, cfg=cfg)
    # CometSparkV02Small.net 是 CometSparkNexLM，提供 forward_with_aux
    assert trainer.use_aux is True, "CometSparkV02Small 应启用 aux 路径"
    train_losses, val_losses = trainer.fit()
    assert len(train_losses) == 5
    assert len(val_losses) > 0
    assert (tmp_path / "loss_history.json").exists()


def test_verse_nex_trainer_without_aux(tmp_path):
    """退化路径：无 MoD 层模型的 aux loss 为 0。

    Part4K1 Task 8.9: 用显式 ``layer_pattern=["trisparse", "trisparse"]`` 构造
    真正无 MoD 层的模型（``mod_every=99`` 仍会在第 0 层创建 MoD，因为
    0 % 99 == 0）。虽然 forward_with_aux 存在，但无 MoD 层时 aux loss 为 0，
    等效于标准 CE 路径。
    """
    # 用显式 layer_pattern 构造真正无 MoD 层的模型
    from spark.model.config import CometSparkV05Config
    from spark.model.model import CometSparkV05LM
    config = CometSparkV05Config(
        arch="versenex",
        vocab_size=256,
        n_layer=2,
        n_head=4,
        n_embd=64,
        n_kv_head=2,
        seq_len=64,
        dropout=0.0,
        tie_weights=True,
        mod_every=99,  # 无效，被 layer_pattern 覆盖
        layer_pattern=["trisparse", "trisparse"],  # 显式全 trisparse
        num_dense_parts=2,
        num_experts_per_part=2,
        top_k=1,
        window_size=32,
        num_global_tokens=4,
        use_alibi=True,
        use_rope=False,
        max_position_embeddings=256,
    )
    model = CometSparkV05LM(config)
    # 验证确实无 MoD 层
    from verse_nex.moe import MoDLayer
    n_mod = sum(1 for m in model.net.modules() if isinstance(m, MoDLayer))
    assert n_mod == 0, f"应无 MoD 层，实际 {n_mod}"

    train_loader = _make_lm_batches(vocab_size=256, seq_len=16,
                                     n_samples=8, batch_size=2, seed=0)
    val_loader = _make_lm_batches(vocab_size=256, seq_len=16,
                                   n_samples=4, batch_size=2, seed=1)
    opt = AdamW(model.parameters(), lr=1e-3)
    cfg = {
        "max_steps": 5,
        "eval_interval": 2,
        "patience": 10,
        "save_dir": str(tmp_path),
        "grad_accum": 1,
        "log_interval": 100,
        "enable_progress_bar": False,
        "realtime_plot": False,
    }
    trainer = VerseNexTrainer(model, train_loader, val_loader, opt, cfg=cfg)
    # Part4K1: versenex 模型有 forward_with_aux，use_aux=True
    # 但无 MoD 层时 aux loss 为 0，等效标准 CE
    train_losses, val_losses = trainer.fit()
    assert len(train_losses) == 5
    # aux_losses 全为 0（无 MoD 层，aux loss 为 0）
    assert all(float(v) == 0.0 for v in trainer.aux_losses), (
        "无 MoD 层时 aux_losses 应全为 0"
    )


def test_verse_nex_trainer_aux_loss_weight_from_config(tmp_path):
    """aux_loss_weight 从 model.config.aux_loss_weight 读取（默认 0.01）。"""
    model = CometSparkV02Small(vocab_size=64, seq_len=32)
    # CometSparkConfig.aux_loss_weight 默认 0.01
    assert model.config.aux_loss_weight == 0.01
    train_loader = _make_lm_batches(vocab_size=64, seq_len=16,
                                     n_samples=4, batch_size=2)
    opt = AdamW(model.parameters(), lr=1e-3)
    cfg = {
        "max_steps": 1,
        "save_dir": str(tmp_path),
        "enable_progress_bar": False,
        "realtime_plot": False,
    }
    trainer = VerseNexTrainer(model, train_loader, train_loader, opt, cfg=cfg)
    assert trainer.aux_loss_weight == model.config.aux_loss_weight
    assert trainer.aux_loss_weight == 0.01


def test_verse_nex_trainer_aux_loss_weight_override(tmp_path):
    """cfg 显式传 aux_loss_weight=0.5 覆盖模型默认值。"""
    model = CometSparkV02Small(vocab_size=64, seq_len=32)
    train_loader = _make_lm_batches(vocab_size=64, seq_len=16,
                                     n_samples=4, batch_size=2)
    opt = AdamW(model.parameters(), lr=1e-3)
    cfg = {
        "max_steps": 1,
        "save_dir": str(tmp_path),
        "aux_loss_weight": 0.5,
        "enable_progress_bar": False,
        "realtime_plot": False,
    }
    trainer = VerseNexTrainer(model, train_loader, train_loader, opt, cfg=cfg)
    assert trainer.aux_loss_weight == 0.5
    assert trainer.aux_loss_weight != model.config.aux_loss_weight


def test_verse_nex_trainer_save_history(tmp_path):
    """_save_history 生成 loss_history.json + 三个 .txt + loss_curve.png/.txt。"""
    model = CometSparkV02Small(vocab_size=64, seq_len=32)
    train_loader = _make_lm_batches(vocab_size=64, seq_len=16,
                                     n_samples=4, batch_size=2)
    opt = AdamW(model.parameters(), lr=1e-3)
    cfg = {
        "max_steps": 3,
        "eval_interval": 1,
        "patience": 10,
        "save_dir": str(tmp_path),
        "grad_accum": 1,
        "log_interval": 100,
        "enable_progress_bar": False,
        "realtime_plot": False,
    }
    trainer = VerseNexTrainer(model, train_loader, train_loader, opt, cfg=cfg)
    trainer.fit()
    # JSON 历史文件
    assert (tmp_path / "loss_history.json").exists()
    # 三个纯文本文件
    assert (tmp_path / "train_losses.txt").exists()
    assert (tmp_path / "val_losses.txt").exists()
    assert (tmp_path / "aux_losses.txt").exists()
    # loss 曲线（matplotlib 不可用时降级为 .txt）
    assert (tmp_path / "loss_curve.png").exists() or (
        tmp_path / "loss_curve.txt").exists()
    # 校验 loss_history.json 内容
    with open(tmp_path / "loss_history.json", "r", encoding="utf-8") as f:
        history = json.load(f)
    assert "train_losses" in history
    assert "val_losses" in history
    assert "aux_losses" in history
    assert history["max_steps"] == 3


# ---------------------------------------------------------------------------
# LoRATrainer
# ---------------------------------------------------------------------------


def test_lora_trainer_wraps_model(tmp_path):
    """构造后所有 Linear 都被包装为 LoRALinear。"""
    model = CometSparkV02Small(vocab_size=64, seq_len=32)
    train_loader = _make_lm_batches(vocab_size=64, seq_len=16,
                                     n_samples=4, batch_size=2)
    cfg = {
        "max_steps": 1,
        "save_dir": str(tmp_path),
        "enable_progress_bar": False,
        "realtime_plot": False,
    }
    # Part4K1 Task 8.9: CometSparkV05LM 不是 Module 子类，
    # LoRATrainer 需要 Module（lora_only 遍历 _modules），
    # 因此传 model.net（CometSparkNexLM = Module 子类）
    trainer = LoRATrainer(model.net, train_loader, train_loader,
                           cfg=cfg, lora_r=4, lora_alpha=8.0)
    # LoRALinear 数量 > 0
    n_lora = _count_module_type(trainer.model, LoRALinear)
    assert n_lora > 0, "应至少有一个 LoRALinear"
    # 每个 Linear 都是某个 LoRALinear 的 base
    # → nn.Linear 数量 == LoRALinear 数量
    n_linear = _count_module_type(trainer.model, vt_nn.Linear)
    assert n_linear == n_lora, (
        f"每个 Linear 应是某 LoRALinear 的 base: "
        f"linear={n_linear} != lora={n_lora}"
    )


def test_lora_trainer_only_lora_params_trainable(tmp_path):
    """base 参数 requires_grad=False，仅 A/B requires_grad=True。"""
    model = CometSparkV02Small(vocab_size=64, seq_len=32)
    train_loader = _make_lm_batches(vocab_size=64, seq_len=16,
                                     n_samples=4, batch_size=2)
    cfg = {
        "max_steps": 1,
        "save_dir": str(tmp_path),
        "enable_progress_bar": False,
        "realtime_plot": False,
    }
    # Part4K1 Task 8.9: 传 model.net（Module 子类）
    trainer = LoRATrainer(model.net, train_loader, train_loader,
                           cfg=cfg, lora_r=4, lora_alpha=8.0)
    n_lora = 0
    for m in trainer.model.modules():
        if isinstance(m, LoRALinear):
            n_lora += 1
            assert m.A.requires_grad is True, "LoRA A 应可训练"
            assert m.B.requires_grad is True, "LoRA B 应可训练"
            assert m.base.weight.requires_grad is False, "base weight 应冻结"
            if m.base.bias is not None:
                assert m.base.bias.requires_grad is False, "base bias 应冻结"
    assert n_lora > 0


def test_lora_trainer_fit_5_steps(tmp_path):
    """LoRATrainer 训练 5 步不报错，train_losses 非空。"""
    model = CometSparkV02Small(vocab_size=64, seq_len=32)
    train_loader = _make_lm_batches(vocab_size=64, seq_len=16,
                                     n_samples=8, batch_size=2, seed=0)
    val_loader = _make_lm_batches(vocab_size=64, seq_len=16,
                                   n_samples=4, batch_size=2, seed=1)
    cfg = {
        "max_steps": 5,
        "eval_interval": 2,
        "patience": 10,
        "save_dir": str(tmp_path),
        "grad_accum": 1,
        "log_interval": 100,
        "lr": 1e-3,
        "enable_progress_bar": False,
        "realtime_plot": False,
    }
    # Part4K1 Task 8.9: 传 model.net（Module 子类）
    trainer = LoRATrainer(model.net, train_loader, val_loader,
                           cfg=cfg, lora_r=4, lora_alpha=8.0)
    train_losses, val_losses = trainer.fit()
    assert len(train_losses) == 5


def test_lora_trainer_merge_lora(tmp_path):
    """merge_lora 后模型中无 LoRALinear，所有层恢复为 Linear。"""
    model = CometSparkV02Small(vocab_size=64, seq_len=32)
    train_loader = _make_lm_batches(vocab_size=64, seq_len=16,
                                     n_samples=4, batch_size=2)
    cfg = {
        "max_steps": 1,
        "save_dir": str(tmp_path),
        "enable_progress_bar": False,
        "realtime_plot": False,
    }
    # Part4K1 Task 8.9: 传 model.net（Module 子类）
    trainer = LoRATrainer(model.net, train_loader, train_loader,
                           cfg=cfg, lora_r=4, lora_alpha=8.0)
    # merge 前：有 LoRALinear
    n_lora_before = _count_module_type(trainer.model, LoRALinear)
    assert n_lora_before > 0
    trainer.merge_lora()
    # merge 后：无 LoRALinear
    n_lora_after = _count_module_type(trainer.model, LoRALinear)
    assert n_lora_after == 0, "merge 后不应残留 LoRALinear"
    # 仍有 Linear（合并后的）
    n_linear = _count_module_type(trainer.model, vt_nn.Linear)
    assert n_linear > 0, "merge 后应有合并后的 Linear"


# ---------------------------------------------------------------------------
# SFTTrainer + 辅助函数
# ---------------------------------------------------------------------------


def test_messages_to_tokens_basic():
    """messages 转 tokens，仅 assistant 内容 + eos 的 mask=1。"""
    tok = MockTokenizer(vocab_size=256)
    messages = [
        {"role": "system", "content": "S"},
        {"role": "user", "content": "U"},
        {"role": "assistant", "content": "A"},
    ]
    token_ids, loss_mask = _messages_to_tokens(messages, tok)
    assert len(token_ids) == len(loss_mask)
    # mask=1 的 token 数应等于 assistant_prefix 后的 "A" + eos_token
    asst_prefix_ids = tok.encode("<|assistant|>")
    assistant_content_ids = tok.encode("A")
    eos_ids = tok.encode("<|endoftext|>")
    expected_mask_sum = len(assistant_content_ids) + len(eos_ids)
    assert sum(loss_mask) == expected_mask_sum, (
        f"assistant mask 总数 {sum(loss_mask)} != 预期 {expected_mask_sum}"
    )
    # 定位 assistant 段起始位置
    asst_start = -1
    for i in range(len(token_ids) - len(asst_prefix_ids) + 1):
        if token_ids[i:i + len(asst_prefix_ids)] == asst_prefix_ids:
            asst_start = i
            break
    assert asst_start >= 0, "未找到 assistant_prefix"
    # assistant 段之前 mask 全为 0
    assert all(m == 0 for m in loss_mask[:asst_start])
    # assistant_prefix 自身 mask=0
    prefix_end = asst_start + len(asst_prefix_ids)
    assert all(m == 0 for m in loss_mask[asst_start:prefix_end])
    # assistant 内容 + eos 部分 mask=1
    assert all(m == 1 for m in loss_mask[prefix_end:])


def test_build_sft_sample_padding():
    """token 总数 < seq_len 时左 pad 0。"""
    tok = MockTokenizer(vocab_size=256)
    messages = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "yo"},
    ]
    seq_len = 64
    input_ids, labels = _build_sft_sample(messages, tok,
                                            seq_len=seq_len, ignore_index=-100)
    assert input_ids.shape == (seq_len,)
    assert labels.shape == (seq_len,)
    # 左 pad：开头应有 0（pad token）
    # 找到第一个非零位置
    nonzero_idx = np.nonzero(input_ids)[0]
    assert len(nonzero_idx) > 0, "应存在非零 token"
    first_nonzero = nonzero_idx[0]
    # 前面应全为 0（pad）
    assert all(input_ids[i] == 0 for i in range(first_nonzero))
    # 对应 labels 应为 ignore_index
    assert all(labels[i] == -100 for i in range(first_nonzero))


def test_build_sft_sample_truncation():
    """token 总数 > seq_len 时右侧截断。"""
    tok = MockTokenizer(vocab_size=256)
    long_content = "A" * 200  # 远超 seq_len
    messages = [
        {"role": "user", "content": "Q"},
        {"role": "assistant", "content": long_content},
    ]
    seq_len = 32
    input_ids, labels = _build_sft_sample(messages, tok,
                                            seq_len=seq_len, ignore_index=-100)
    assert input_ids.shape == (seq_len,)
    assert labels.shape == (seq_len,)


def test_sft_dataset_loads_jsonl(tmp_path):
    """SFTDataset 加载 chat 格式 jsonl。"""
    # 注意：_messages_to_tokens 为每条 message 加角色前缀（如 <|user|> 8 字符、
    # <|assistant|> 12 字符）+ eos_token(<|endoftext|> 13 字符)。为保证
    # assistant 内容能落在 seq_len 窗口内，messages 内容应尽量短或 seq_len 足够大。
    items = [
        {"messages": [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Yo"},
        ]},
        {"messages": [
            {"role": "user", "content": "Bye"},
            {"role": "assistant", "content": "Ok"},
        ]},
    ]
    path = tmp_path / "sft.jsonl"
    _write_jsonl(path, items)
    tok = MockTokenizer(vocab_size=256)
    # seq_len=32 可容纳 <|user|>(8) + "Hi"(2) + <|assistant|>(12) + "Yo"(2) = 24
    # 以及 eos_token 前 8 个 token，使 assistant 段落在窗口内
    ds = SFTDataset(tok, str(path), seq_len=32, ignore_index=-100)
    assert len(ds) == 2
    x, y = ds[0]
    assert x.shape == (32,)
    assert y.shape == (32,)
    # 应同时存在 ignore_index=-100 与有效 label
    assert (y == -100).any(), "system/user 部分 label 应为 -100"
    assert (y != -100).any(), "assistant 部分 label 应为有效 token id"


def test_sft_trainer_fit_5_steps(tmp_path):
    """SFTTrainer 训练 5 步，仅 assistant token 参与 loss（labels 含 -100）。"""
    # messages 内容短到能让 assistant 段落在 seq_len 窗口内：
    # <|user|>(8) + "Hi"(2) + <|assistant|>(12) + "Yo"(2) + eos_prefix(13) = 37
    # 取 seq_len=40 保证 assistant 段完整可见
    items = [
        {"messages": [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Yo"},
        ]}
        for _ in range(8)
    ]
    path = tmp_path / "sft.jsonl"
    _write_jsonl(path, items)
    tok = MockTokenizer(vocab_size=64)
    train_ds = SFTDataset(tok, str(path), seq_len=40, ignore_index=-100)
    val_ds = SFTDataset(tok, str(path), seq_len=40, ignore_index=-100)
    train_loader = BatchLoader(train_ds, batch_size=2, shuffle=False,
                                collate_fn=_sft_collate)
    val_loader = BatchLoader(val_ds, batch_size=2, shuffle=False,
                              collate_fn=_sft_collate)
    # 模型 vocab_size 必须覆盖 MockTokenizer 的 token id 范围；
    # seq_len=64 覆盖数据 seq_len=40
    model = CometSparkV02Small(vocab_size=64, seq_len=64)
    opt = AdamW(model.parameters(), lr=1e-3)
    cfg = {
        "max_steps": 5,
        "eval_interval": 2,
        "patience": 10,
        "save_dir": str(tmp_path / "ckpt"),
        "grad_accum": 1,
        "log_interval": 100,
        "enable_progress_bar": False,
        "realtime_plot": False,
    }
    trainer = SFTTrainer(model, train_loader, val_loader, opt,
                          cfg=cfg, ignore_index=-100)
    train_losses, val_losses = trainer.fit()
    assert len(train_losses) == 5
    # 验证 batch labels 中确实有 -100（assistant 屏蔽其他）
    for x, y in train_loader:
        assert (y == -100).any(), "labels 应含 -100（非 assistant 部分）"
        assert (y != -100).any(), "labels 应含有效 token id（assistant 部分）"
        break


# ---------------------------------------------------------------------------
# DPOTrainer + 辅助函数
# ---------------------------------------------------------------------------


def test_dpo_dataset_loads_jsonl(tmp_path):
    """DPODataset 加载偏好对 jsonl。"""
    items = [
        {"prompt": "1+1=", "chosen": "2", "rejected": "3"},
        {"prompt": "2+2=", "chosen": "4", "rejected": "5"},
    ]
    path = tmp_path / "dpo.jsonl"
    _write_jsonl(path, items)
    tok = MockTokenizer(vocab_size=256)
    ds = DPODataset(tok, str(path), seq_len=32,
                     eos_token="<|endoftext|>")
    assert len(ds) == 2
    sample = ds[0]
    assert "chosen_input_ids" in sample
    assert "rejected_input_ids" in sample
    assert "chosen_mask" in sample
    assert "rejected_mask" in sample
    assert "chosen_labels" in sample
    assert "rejected_labels" in sample
    assert sample["chosen_input_ids"].shape == (32,)
    assert sample["chosen_mask"].shape == (32,)
    # mask 应同时存在 0（prompt 部分）与 1（response 部分）
    assert (sample["chosen_mask"] == 0).any()
    assert (sample["chosen_mask"] == 1).any()


def test_dpo_loss_basic():
    """_dpo_loss 对 toy 输入返回正数 loss。"""
    # policy_c, policy_r, ref_c, ref_r: shape (B,) log_probs
    # policy 比 ref 更偏好 chosen（policy_c - ref_c > policy_r - ref_r）
    policy_c = Tensor(np.array([-1.0, -2.0], dtype=np.float32),
                       requires_grad=True)
    policy_r = Tensor(np.array([-3.0, -4.0], dtype=np.float32),
                       requires_grad=True)
    ref_c = Tensor(np.array([-1.5, -2.5], dtype=np.float32))
    ref_r = Tensor(np.array([-2.5, -3.5], dtype=np.float32))
    loss = _dpo_loss(policy_c, policy_r, ref_c, ref_r, beta=0.1)
    # loss 应为正数（softplus 总是非负）
    assert loss.data.size == 1 or loss.data.shape == ()
    assert float(loss.data) > 0


def test_log_probs_from_logits_shape():
    """_log_probs_from_logits 返回 (B, T) 形状。"""
    B, T, V = 2, 4, 8
    np.random.seed(42)
    logits_np = np.random.randn(B, T, V).astype(np.float32)
    labels_np = np.random.randint(0, V, size=(B, T)).astype(np.int64)
    logits = Tensor(logits_np, requires_grad=True)
    labels = Tensor(labels_np)
    log_probs = _log_probs_from_logits(logits, labels)
    assert log_probs.data.shape == (B, T)


def test_dpo_trainer_fit_5_steps(tmp_path):
    """DPOTrainer 训练 5 步，train_losses 与 val_accuracies 非空 + 历史落盘。"""
    items = [
        {"prompt": "1+1=", "chosen": "2", "rejected": "3"},
        {"prompt": "2+2=", "chosen": "4", "rejected": "5"},
        {"prompt": "3+3=", "chosen": "6", "rejected": "7"},
    ]
    path = tmp_path / "dpo.jsonl"
    _write_jsonl(path, items)
    tok = MockTokenizer(vocab_size=64)
    train_ds = DPODataset(tok, str(path), seq_len=16,
                           eos_token="<|endoftext|>")
    val_ds = DPODataset(tok, str(path), seq_len=16,
                         eos_token="<|endoftext|>")
    train_loader = BatchLoader(train_ds, batch_size=2, shuffle=False,
                                collate_fn=_dpo_collate)
    val_loader = BatchLoader(val_ds, batch_size=2, shuffle=False,
                              collate_fn=_dpo_collate)
    model = CometSparkV02Small(vocab_size=64, seq_len=32)
    cfg = {
        "max_steps": 5,
        "eval_interval": 1,
        "patience": 10,
        "save_dir": str(tmp_path / "dpo_ckpt"),
        "grad_accum": 1,
        "log_interval": 100,
        "lr": 1e-3,
        "beta": 0.1,
        "enable_progress_bar": False,
    }
    trainer = DPOTrainer(model, ref_model=None,
                          train_loader=train_loader,
                          val_loader=val_loader, cfg=cfg)
    train_losses, val_losses, val_accuracies = trainer.fit()
    assert len(train_losses) == 5
    assert len(val_accuracies) > 0, "val_accuracies 应非空"
    assert (tmp_path / "dpo_ckpt" / "dpo_history.json").exists()


def test_dpo_trainer_ref_model_frozen(tmp_path):
    """reference model 所有参数 requires_grad=False。"""
    model = CometSparkV02Small(vocab_size=64, seq_len=32)
    cfg = {
        "max_steps": 1,
        "save_dir": str(tmp_path),
        "enable_progress_bar": False,
    }
    trainer = DPOTrainer(model, ref_model=None,
                          train_loader=None, val_loader=None, cfg=cfg)
    # 遍历 ref_model 的所有参数（包括非 trainable）
    all_params = list(_all_params(trainer.ref_model))
    assert len(all_params) > 0, "ref_model 应有参数"
    n_frozen = sum(1 for p in all_params if not p.requires_grad)
    assert n_frozen == len(all_params), (
        f"所有 ref_model 参数应冻结: frozen={n_frozen}/total={len(all_params)}"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
