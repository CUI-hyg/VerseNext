"""数据集下载器（Part4K2 Task 8）。

支持任意 HTTP/HTTPS URL 下载与 HuggingFace datasets 下载，
并自动转换为 .npz 缓存格式（与 :class:`verse_infra.verse_trainer.CachedDataset` 对齐）。

特性
====
- **断点续传**：基于本地已下载字节数 + ``Range`` header，从断点继续下载。
- **多线程分块下载**：大文件（>= 10MB）按 ``num_workers`` 分块并行下载，
  小文件直接单线程。
- **格式转换**：``.json`` / ``.jsonl`` / ``.csv`` / ``.txt`` / ``.parquet``
  → ``.npz``（包含 ``ids`` / ``mask`` / ``seq_len`` / ``n_blocks`` / ``source``）。
- **HF datasets**：可选依赖 ``datasets`` 库，缺失时优雅降级并提示安装。
- **pyarrow**：可选依赖（parquet 支持），缺失时跳过。

使用示例
========
::

    from data.downloader import DatasetDownloader

    dl = DatasetDownloader(cache_dir="data/datasets", num_workers=4)

    # 1. 任意 URL 下载
    path = dl.download_url("https://example.com/data.json")

    # 2. HuggingFace datasets 下载
    dir_ = dl.download_hf("wikitext", subset="wikitext-2-raw-v1", split="train")

    # 3. 转 .npz 缓存
    npz = dl.to_npz("data/datasets/data.json", text_key="text")

    # 4. 一站式：下载 + 转 .npz
    npz = dl.download_and_cache("https://example.com/data.json",
                                output_path="data/datasets/cached.npz")
"""

from __future__ import annotations

import csv
import json
import os
import shutil
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional

import numpy as np

# 多线程下载触发阈值：< 此值用单线程
_MULTITHREAD_THRESHOLD = 10 * 1024 * 1024  # 10MB


def _is_url(s: str) -> bool:
    """判断字符串是否为 HTTP/HTTPS URL。"""
    return s.startswith("http://") or s.startswith("https://")


def _url_filename(url: str) -> str:
    """从 URL 提取文件名；失败返回 ``'downloaded.bin'``。"""
    parsed = urllib.parse.urlparse(url)
    name = os.path.basename(parsed.path)
    return name if name else "downloaded.bin"


class DatasetDownloader:
    """数据集下载器，支持任意 URL 和 HuggingFace datasets。

    Args:
        cache_dir: 下载缓存目录（不存在会自动创建）
        num_workers: 多线程下载线程数
        chunk_size: 分块下载大小（字节，最小 1024）
    """

    def __init__(self, cache_dir: str = "data/datasets",
                 num_workers: int = 4, chunk_size: int = 8192):
        self.cache_dir = str(cache_dir)
        self.num_workers = max(1, int(num_workers))
        self.chunk_size = max(1024, int(chunk_size))
        os.makedirs(self.cache_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # download_url
    # ------------------------------------------------------------------

    def download_url(self, url: str, output_path: Optional[str] = None,
                     resume: bool = True) -> str:
        """从任意 HTTP/HTTPS URL 下载文件。

        支持断点续传：如果文件部分下载，从断点继续。
        大文件（>= 10MB）自动启用多线程分块下载。

        Args:
            url: 下载 URL（http/https）
            output_path: 输出路径；None 则用 ``cache_dir + URL 文件名``
            resume: 是否启用断点续传

        Returns:
            下载文件的绝对路径
        """
        if output_path is None:
            output_path = os.path.join(self.cache_dir, _url_filename(url))
        output_path = os.path.abspath(output_path)
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        # 先尝试 HEAD 获取 Content-Length
        total_size = self._head_content_length(url)
        if total_size is None:
            # HEAD 失败：直接单线程下载（带断点续传）
            return self._download_single(url, output_path, resume)

        # 小文件 / 单线程：直接下载
        if total_size < _MULTITHREAD_THRESHOLD or self.num_workers <= 1:
            return self._download_single(url, output_path, resume, total_size)

        # 大文件：多线程分块下载
        return self._download_multithread(url, output_path, total_size, resume)

    def _head_content_length(self, url: str) -> Optional[int]:
        """发送 HEAD 请求获取 Content-Length；失败返回 None。"""
        req = urllib.request.Request(url, method="HEAD")
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                cl = resp.headers.get("Content-Length")
                return int(cl) if cl else None
        except Exception:
            return None

    def _supports_range(self, url: str) -> bool:
        """探测服务器是否支持 Range 请求（发送 ``Range: bytes=0-0``）。"""
        req = urllib.request.Request(url, headers={"Range": "bytes=0-0"})
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                return resp.status == 206
        except Exception:
            return False

    def _download_single(self, url: str, output_path: str,
                         resume: bool,
                         total_size: Optional[int] = None) -> str:
        """单线程下载（带断点续传）。

        Args:
            url: 下载 URL
            output_path: 输出路径
            resume: 是否启用断点续传
            total_size: 已知文件总大小（None 表示未知）
        """
        # 已下载完整则直接返回
        if resume and os.path.exists(output_path) and total_size is not None:
            if os.path.getsize(output_path) == total_size:
                return output_path

        mode = "wb"
        existing = 0
        headers = {}
        if resume and os.path.exists(output_path):
            existing = os.path.getsize(output_path)
            if total_size is not None and existing >= total_size:
                # 已下载完整（或超出，可能是脏数据；保险起见返回）
                return output_path
            if existing > 0:
                headers["Range"] = f"bytes={existing}-"
                mode = "ab"

        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=60) as resp:
            # 206=续传成功；200=服务器忽略 Range（需重头下载）
            if existing > 0 and resp.status == 200:
                mode = "wb"
                existing = 0
            with open(output_path, mode) as f:
                while True:
                    chunk = resp.read(self.chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
        return output_path

    def _download_multithread(self, url: str, output_path: str,
                              total_size: int, resume: bool) -> str:
        """多线程分块下载。

        Args:
            url: 下载 URL（服务器需支持 Range）
            output_path: 最终输出路径
            total_size: 文件总大小
            resume: 是否启用断点续传（对分片生效）
        """
        # 已下载完整则直接返回
        if resume and os.path.exists(output_path) and \
                os.path.getsize(output_path) == total_size:
            return output_path

        # 服务器不支持 Range：回退单线程
        if not self._supports_range(url):
            return self._download_single(url, output_path, resume, total_size)

        # 计算分块：每个 worker 一块
        n_workers = max(1, self.num_workers)
        chunk_size = total_size // n_workers
        ranges: List[tuple] = []
        for i in range(n_workers):
            start = i * chunk_size
            end = (i + 1) * chunk_size - 1 if i < n_workers - 1 \
                else total_size - 1
            ranges.append((i, start, end))

        tmp_dir = output_path + ".parts"
        os.makedirs(tmp_dir, exist_ok=True)

        def _worker(idx_start_end: tuple) -> tuple:
            """单个分片下载任务。"""
            idx, start, end = idx_start_end
            part_path = os.path.join(tmp_dir, f"part_{idx}")
            # 断点续传：检查已下载字节
            existing = 0
            if resume and os.path.exists(part_path):
                existing = os.path.getsize(part_path)
                # 防御：若分片超出应有大小，重头下
                expected = end - start + 1
                if existing > expected:
                    existing = 0
            real_start = start + existing
            if real_start > end:
                return idx, part_path  # 已完成
            headers = {"Range": f"bytes={real_start}-{end}"}
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=60) as resp:
                mode = "ab" if existing > 0 and resp.status == 206 else "wb"
                if mode == "wb":
                    existing = 0
                with open(part_path, mode) as f:
                    while True:
                        chunk = resp.read(self.chunk_size)
                        if not chunk:
                            break
                        f.write(chunk)
            return idx, part_path

        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = [executor.submit(_worker, r) for r in ranges]
            for fut in as_completed(futures):
                fut.result()  # 触发异常向上抛

        # 合并分片（按 idx 顺序）
        with open(output_path, "wb") as out:
            for i in range(n_workers):
                part_path = os.path.join(tmp_dir, f"part_{i}")
                with open(part_path, "rb") as f:
                    shutil.copyfileobj(f, out, self.chunk_size)

        # 清理分片目录
        try:
            shutil.rmtree(tmp_dir)
        except Exception:
            pass

        # 验证 Content-Length
        actual_size = os.path.getsize(output_path)
        if actual_size != total_size:
            raise IOError(
                f"下载大小不匹配：期望 {total_size} 字节，实际 {actual_size} 字节"
            )
        return output_path

    # ------------------------------------------------------------------
    # download_hf
    # ------------------------------------------------------------------

    def download_hf(self, repo_id: str, subset: Optional[str] = None,
                    split: str = "train",
                    output_dir: Optional[str] = None) -> str:
        """从 HuggingFace datasets 下载。

        尝试用 ``datasets`` 库（可选依赖），不可用时提示安装。
        下载结果保存为 ``<output_dir>/<split>.jsonl``。

        Args:
            repo_id: HF dataset repo ID（如 ``wikitext``）
            subset: 子集名（如 ``wikitext-2-raw-v1``），可选
            split: split 名（默认 ``train``）
            output_dir: 输出目录；None 则用 ``cache_dir/repo_id``

        Returns:
            下载目录路径

        Raises:
            RuntimeError: ``datasets`` 库未安装
        """
        if output_dir is None:
            safe_repo = repo_id.replace("/", "_")
            output_dir = os.path.join(self.cache_dir, safe_repo)
        output_dir = os.path.abspath(output_dir)
        os.makedirs(output_dir, exist_ok=True)

        try:
            import datasets  # type: ignore
        except ImportError:
            raise RuntimeError(
                "下载 HuggingFace datasets 需要安装 `datasets` 库："
                "请执行 `pip install datasets`。"
            )

        ds = datasets.load_dataset(repo_id, name=subset, split=split)
        output_path = os.path.join(output_dir, f"{split}.jsonl")
        with open(output_path, "w", encoding="utf-8") as f:
            for item in ds:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"[downloader] HF dataset {repo_id} 已保存到 {output_path}",
              flush=True)
        return output_dir

    # ------------------------------------------------------------------
    # to_npz
    # ------------------------------------------------------------------

    def to_npz(self, input_path: str, output_path: Optional[str] = None,
               text_key: str = "text", tokenizer=None) -> str:
        """将下载的数据转换为 .npz 缓存格式。

        支持 ``.json`` / ``.jsonl`` / ``.csv`` / ``.txt`` / ``.parquet`` 输入。
        输出 ``.npz`` 包含 ``ids`` / ``mask`` / ``seq_len`` / ``n_blocks`` /
        ``source``，与 :class:`verse_infra.verse_trainer.CachedDataset`
        格式一致（``seq_len=1``，可由调用方按需 reshape）。

        Args:
            input_path: 输入文件路径
            output_path: 输出 .npz 路径；None 则用 ``input_path + ".npz"``
            text_key: JSON/CSV 中文本字段名（默认 ``text``）
            tokenizer: tokenizer 对象（需有 ``encode`` 方法）；
                None 则用 ``ByteTokenizer``

        Returns:
            .npz 文件路径
        """
        if output_path is None:
            output_path = input_path + ".npz"
        output_path = os.path.abspath(output_path)

        texts = self._extract_texts(input_path, text_key)

        if tokenizer is None:
            from verse_infra.verse_tokenizer import ByteTokenizer
            tokenizer = ByteTokenizer()

        ids_list: List[int] = []
        mask_list: List[int] = []

        # 优先复用 verse_trainer.data 的编码逻辑（与 CachedDataset 完全对齐）
        try:
            from verse_infra.verse_trainer.data import (
                _encode_text, _safe_encode,
            )
            use_verse_trainer = True
        except ImportError:
            use_verse_trainer = False

        if use_verse_trainer:
            for text in texts:
                ids, mask = _encode_text(tokenizer, text)
                ids_list.extend(ids)
                mask_list.extend(mask)
                # 样本之间用换行分隔（不参与 loss），与 CachedDataset 一致
                nl_ids = _safe_encode(tokenizer, "\n")
                ids_list.extend(nl_ids)
                mask_list.extend([0] * len(nl_ids))
        else:
            # 降级实现：每文本编码 + 换行分隔
            for text in texts:
                ids = self._safe_encode_fallback(tokenizer, text)
                ids_list.extend(ids)
                mask_list.extend([1] * len(ids))
                nl_ids = self._safe_encode_fallback(tokenizer, "\n")
                ids_list.extend(nl_ids)
                mask_list.extend([0] * len(nl_ids))

        if not ids_list:
            raise ValueError(f"无法从 {input_path} 提取任何文本")

        ids_np = np.asarray(ids_list, dtype=np.int64)
        mask_np = np.asarray(mask_list, dtype=np.int64)

        # seq_len=1：n_blocks = total_tokens，调用方可按需 reshape
        seq_len = 1
        n_blocks = len(ids_np) // seq_len
        ids_np = ids_np[: n_blocks * seq_len]
        mask_np = mask_np[: n_blocks * seq_len]

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        np.savez(
            output_path,
            ids=ids_np,
            mask=mask_np,
            seq_len=np.int64(seq_len),
            n_blocks=np.int64(n_blocks),
            source=np.asarray(input_path),
        )
        print(f"[downloader] 已转换 {input_path} → {output_path} "
              f"(tokens={len(ids_np)})", flush=True)
        return output_path

    @staticmethod
    def _safe_encode_fallback(tok, text: str) -> List[int]:
        """tokenizer.encode 兼容封装（降级路径）。"""
        if not text:
            return []
        try:
            return list(tok.encode(text, add_special_tokens=False))
        except TypeError:
            try:
                return list(tok.encode(text))
            except Exception:
                return []

    def _extract_texts(self, input_path: str, text_key: str) -> List[str]:
        """从输入文件提取文本列表（按扩展名分派解析器）。"""
        ext = os.path.splitext(input_path)[1].lower()
        if ext == ".json":
            with open(input_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                return [str(item.get(text_key, "")) if isinstance(item, dict)
                        else str(item) for item in data]
            if isinstance(data, dict):
                # 整体当作单文档（提取 text_key 或整体 str）
                if text_key in data:
                    return [str(data[text_key])]
                return [json.dumps(data, ensure_ascii=False)]
            return [str(data)]
        if ext == ".jsonl":
            texts: List[str] = []
            with open(input_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        # 非 JSON 行直接当作文本
                        texts.append(line)
                        continue
                    if isinstance(obj, dict):
                        texts.append(str(obj.get(text_key, "")))
                    elif isinstance(obj, list):
                        # chat 数组等：拼成文本
                        texts.append(json.dumps(obj, ensure_ascii=False))
                    else:
                        texts.append(str(obj))
            return texts
        if ext == ".csv":
            texts = []
            with open(input_path, "r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    texts.append(str(row.get(text_key, "")))
            return texts
        if ext == ".txt":
            with open(input_path, "r", encoding="utf-8") as f:
                return [line.rstrip("\n").rstrip("\r")
                        for line in f if line.strip()]
        if ext == ".parquet":
            try:
                import pyarrow.parquet as pq  # type: ignore
            except ImportError:
                raise RuntimeError(
                    "读取 .parquet 需要安装 pyarrow：`pip install pyarrow`"
                )
            table = pq.read_table(input_path)
            cols = table.column_names
            if text_key in cols:
                col = table.column(text_key).to_pylist()
            else:
                col = table.column(0).to_pylist()
            return [str(x) for x in col]
        # 未知扩展名：当作纯文本读取
        with open(input_path, "r", encoding="utf-8") as f:
            return [f.read()]

    # ------------------------------------------------------------------
    # download_and_cache
    # ------------------------------------------------------------------

    def download_and_cache(self, url_or_repo: str,
                           output_path: Optional[str] = None,
                           text_key: str = "text") -> str:
        """一站式：下载 + 转 .npz 缓存。

        自动判断 URL 还是 HF repo ID（含 ``http://``/``https://`` 前缀为 URL，
        否则当作 HF repo ID）。

        Args:
            url_or_repo: URL 或 HF repo ID
            output_path: 输出 .npz 路径；None 则自动生成
            text_key: 文本字段名（默认 ``text``）

        Returns:
            .npz 文件路径
        """
        if _is_url(url_or_repo):
            # 下载原始文件到 cache_dir（output_path 留给 .npz）
            raw_path = self.download_url(url_or_repo)
        else:
            # 当作 HF repo
            raw_dir = self.download_hf(url_or_repo)
            candidates = [f for f in os.listdir(raw_dir)
                          if f.endswith(".jsonl")]
            if not candidates:
                raise RuntimeError(
                    f"HF 下载目录 {raw_dir} 中无 .jsonl 文件"
                )
            raw_path = os.path.join(raw_dir, candidates[0])

        if output_path is None:
            output_path = raw_path + ".npz"
        return self.to_npz(raw_path, output_path, text_key=text_key)


__all__ = ["DatasetDownloader"]
