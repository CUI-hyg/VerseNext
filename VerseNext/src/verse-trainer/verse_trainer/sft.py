"""SFT（监督微调）：对话数据集与 loss 掩码。

数据格式（JSON / JSONL / CSV / TSV / Parquet）::

    [{"messages": [{"role": "system", "content": "..."},
                   {"role": "user", "content": "..."},
                   {"role": "assistant", "content": "..."}]},
     ...]

- JSON：整体为对话列表（或含 ``messages`` 的列表）。
- JSONL：每行一条对话。
- CSV/TSV/Parquet：``messages`` 列（JSON 字符串）或 ``system/user/assistant`` 列。

labels 只在 assistant 回答（含结束标记）处有效，其余为 -100——模型学习
「生成回答」而非「复述问题」。
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import torch

from verse_core import get_logger
from verse_core.config import BaseConfig
from verse_tokenizer import BPETokenizer, ChatTemplate

logger = get_logger("sft")


@dataclass
class SFTConfig(BaseConfig):
    """SFT 数据处理配置。"""

    max_seq_len: int = 512
    """超长对话将被跳过并告警。"""
    train_on_last_turns: int = 1
    """对最后 N 轮 assistant 回答计算 loss；0 表示对所有 assistant 轮。"""
    batch_size: int = 4
    seed: int = 42

    def validate(self) -> None:
        if self.batch_size <= 0 or self.max_seq_len <= 0:
            raise ValueError("batch_size / max_seq_len 必须为正数")


class SFTDataset:
    """对话微调数据集，实现 ``next_batch()`` 以复用 Trainer 训练循环。

    用法::

        dataset = SFTDataset.from_json("chats.json", tokenizer, template, config)
        trainer = Trainer(trainer_config)   # trainer_config.lora 可选
        trainer.train(dataset, eval_dataset)
    """

    def __init__(
        self,
        conversations: list[list[dict]],
        tokenizer: BPETokenizer,
        template: ChatTemplate,
        config: SFTConfig | None = None,
        encode_kwargs: dict | None = None,
    ) -> None:
        self.config = config or SFTConfig()
        self.tokenizer = tokenizer
        self.template = template
        self.encode_kwargs = encode_kwargs or {}
        self.device = torch.device("cpu")
        self._generator = torch.Generator().manual_seed(self.config.seed)
        self._pad_id = self._resolve_pad_id(tokenizer)

        self.examples: list[tuple[list[int], list[int]]] = []
        skipped = 0
        for conv in conversations:
            ids, labels = template.encode_chat(
                conv, tokenizer,
                train_on_last_turns=self.config.train_on_last_turns,
                **self.encode_kwargs,
            )
            if len(ids) > self.config.max_seq_len:
                skipped += 1
                continue
            self.examples.append((ids, labels))
        if skipped:
            logger.warning("跳过 %d 条超长对话（> %d tokens）", skipped, self.config.max_seq_len)
        if not self.examples:
            raise ValueError("没有可用对话样本")
        logger.info("SFT 数据集: %d 条对话（已编码）", len(self.examples))

    @classmethod
    def from_json(
        cls,
        path: str | Path,
        tokenizer: BPETokenizer,
        template: ChatTemplate | None = None,
        config: SFTConfig | None = None,
        encode_kwargs: dict | None = None,
    ) -> "SFTDataset":
        """兼容旧接口：从 JSON 加载（等价于 :meth:`from_file`）。"""
        return cls.from_file(path, tokenizer, template, config, encode_kwargs)

    @classmethod
    def from_file(
        cls,
        path: str | Path,
        tokenizer: BPETokenizer,
        template: ChatTemplate | None = None,
        config: SFTConfig | None = None,
        encode_kwargs: dict | None = None,
    ) -> "SFTDataset":
        """从 JSON/JSONL/CSV/TSV/Parquet 加载对话数据。

        支持结构：``[{"messages": [...]}]``、``[[...]]``、每行一条（jsonl）、
        表格 ``messages`` 列（JSON 字符串）或 ``system/user/assistant`` 列。
        """
        conversations = load_conversations(path)
        template = template or ChatTemplate.from_pretrained(
            getattr(tokenizer, "_tokenizer_dir", ".")
        )
        return cls(conversations, tokenizer, template, config, encode_kwargs)

    def __len__(self) -> int:
        return len(self.examples)

    @staticmethod
    def _resolve_pad_id(tokenizer: BPETokenizer) -> int:
        """选择 padding 用的 token id。

        优先真实 pad token；其次复用 eos（GPT 系常见做法，语义上「到此结束」，
        比字节 0 更合理）；都没有时才退回 0 并告警——字节 0 是合法 token，
        直接用它做 padding 会让模型把填充当成真实输入。
        """
        pad_id = tokenizer.pad_token_id
        if pad_id is not None:
            return pad_id
        eos_id = tokenizer.eos_token_id
        if eos_id is not None:
            return eos_id
        logger.warning("tokenizer 缺少 pad/eos 特殊 token，padding 回退到 id 0（可能影响质量）")
        return 0

    def _collate(self, indices: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
        """变长对话 padding 成批；targets 由 labels 右移一位构造。"""
        pad_id = self._pad_id
        batch = [self.examples[i] for i in indices]
        max_len = max(len(ids) for ids, _ in batch) - 1  # 输入长度 = L-1

        inputs = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
        targets = torch.full((len(batch), max_len), -100, dtype=torch.long)
        for row, (ids, labels) in enumerate(batch):
            # inputs = ids[:-1]；targets[i] 预测 ids[i+1]，掩码取 labels[i+1]
            n = len(ids) - 1
            inputs[row, :n] = torch.tensor(ids[:-1], dtype=torch.long)
            tgt = torch.tensor(ids[1:], dtype=torch.long)
            mask = torch.tensor(labels[1:], dtype=torch.long) != -100
            tgt[~mask] = -100
            targets[row, :n] = tgt
        return inputs.to(self.device), targets.to(self.device)

    def next_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        idx = torch.randint(0, len(self.examples), (self.config.batch_size,), generator=self._generator)
        return self._collate(idx.tolist())

    def batches(self, num_batches: int):
        for _ in range(num_batches):
            yield self.next_batch()


# --------------------------------------------------------------- 多格式加载

def _json_items(data) -> list:
    """JSON 顶层结构归一为条目列表。"""
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ("data", "records", "items", "conversations"):
            if isinstance(data.get(key), list):
                return data[key]
        return [data]
    return [data]


def _to_conversation(item) -> list[dict] | None:
    """把一条记录归一为 ``[{"role":..., "content":...}, ...]``。"""
    if isinstance(item, str):
        try:
            item = json.loads(item)
        except json.JSONDecodeError:
            return None
    if isinstance(item, list):
        return item
    if not isinstance(item, dict):
        return None
    for key in ("messages", "conversations"):
        if key in item and item[key]:
            value = item[key]
            if isinstance(value, str):
                value = json.loads(value)
            if isinstance(value, list):
                return value
    # 表格列：system / user / assistant
    messages: list[dict] = []
    if item.get("system"):
        messages.append({"role": "system", "content": str(item["system"])})
    if item.get("user") is not None:
        messages.append({"role": "user", "content": str(item["user"])})
    if item.get("assistant") is not None:
        messages.append({"role": "assistant", "content": str(item["assistant"])})
    return messages if len(messages) >= 2 else None


def _parquet_rows(path: Path) -> list[dict]:
    try:
        import pyarrow.parquet as pq

        return pq.read_table(path).to_pylist()
    except ImportError:
        try:
            import pandas as pd

            return pd.read_parquet(path).to_dict(orient="records")
        except ImportError as exc:  # pragma: no cover - 可选依赖
            raise ImportError(
                "读取 parquet 需要 pyarrow 或 pandas：pip install pyarrow"
            ) from exc


def load_conversations(path: str | Path) -> list[list[dict]]:
    """从 JSON/JSONL/CSV/TSV/Parquet 加载对话列表。"""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"SFT 数据不存在: {path}")
    suffix = path.suffix.lower()
    if suffix == ".json":
        items = _json_items(json.loads(path.read_text(encoding="utf-8")))
    elif suffix == ".jsonl":
        items = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    elif suffix in (".csv", ".tsv"):
        delimiter = "\t" if suffix == ".tsv" else ","
        with open(path, encoding="utf-8", newline="") as f:
            items = list(csv.DictReader(f, delimiter=delimiter))
    elif suffix == ".parquet":
        items = _parquet_rows(path)
    else:
        raise ValueError(
            f"不支持的 SFT 数据格式: {suffix}（支持 .json/.jsonl/.csv/.tsv/.parquet）"
        )

    conversations = []
    for item in items:
        conversation = _to_conversation(item)
        if conversation:
            conversations.append(conversation)
    if not conversations:
        raise ValueError(f"{path} 中没有可用对话样本")
    return conversations
