"""CometSpark src 模块：通用工具与数据加载。"""

from .utils import set_seed, num_threads, ensure_dir, get_device
from .data_loader import load_jsonl, TextDataset, collate_fn

__all__ = ["set_seed", "num_threads", "ensure_dir", "get_device",
           "load_jsonl", "TextDataset", "collate_fn"]
