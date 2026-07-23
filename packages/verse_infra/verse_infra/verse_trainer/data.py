"""数据加载：支持 chat 数组 + prompt-completion + text 三种格式。

Part4K1 Task 6.3 升级：
- 保留原 :class:`TextDataset`（向后兼容 ``data/demo`` 旧入口）。
- 新增 :class:`CachedDataset`：首次扫描数据集后缓存为 ``.npz``
  （token ids + loss mask 数组），后续启动毫秒级加载。
  支持流式 lazy load（大数据集不全量载入内存，按 batch 读取）。
- 保留原 :func:`collate_fn` / :class:`BatchLoader` 兼容。

支持的样本格式：
    1. chat 数组：``[{"role":"user","content":"..."},{"role":"assistant","content":"..."}]``
       → 逐消息编码为 ``<|role|>content``，assistant content 参与 loss
    2. prompt-completion：``{"prompt":"...","completion":"..."}``
       → ``<|user|>prompt<|assistant|>completion<|eos|>``，completion 参与 loss
       （允许单样本：只存在 prompt 或只存在 completion 时，存在字段当作纯文本，全部参与 loss）
    3. text：``{"text":"..."}``
       → 纯文本，所有 token 参与 loss（适用于预训练 / 续训）
"""

from __future__ import annotations

import json
import os
import threading
from typing import List, Optional

import numpy as np

# 尝试导入 chat_template 常量（verse_tokenizer 已升级，含此模块）
try:
    from verse_tokenizer.chat_template import EOS_TOKEN as _EOS_STR
except Exception:  # pragma: no cover - verse_tokenizer 不可用时的降级
    _EOS_STR = "<|eos|>"

# Part5K1 Task 5：JSONL 自修复与字段标准化
from .jsonl_repair import repair_jsonl, JSONLRepairError


# ---------------------------------------------------------------------------
# load_jsonl
# ---------------------------------------------------------------------------


def load_jsonl(path: str, repair: bool = True) -> List[dict]:
    """读取 JSONL 文件，每行一个 JSON 对象（或 JSON 数组）。

    Part5K1 Task 5 升级：新增 ``repair`` 参数。

    - ``repair=True``（默认）：调用 :func:`repair_jsonl` 进行字段标准化
      + 损坏行保守修复（缺逗号 / 未闭合 / BOM / 行尾多余逗号等），
      异名字段（``instruction``/``response`` 等）自动映射为标准
      ``prompt``/``completion`` 或 ``text``。
    - ``repair=False``：走原严格解析逻辑（向后兼容），遇到任何解析失败
      抛 :class:`ValueError`。

    Args:
        path: JSONL 文件路径
        repair: 是否启用自修复 + 字段标准化（默认 True）
    Returns:
        List：每行解析为的 dict 或 list
    Raises:
        JSONLRepairError: ``repair=True`` 时存在无法修复的行
        ValueError: ``repair=False`` 时任意行解析失败
    """
    if repair:
        # 自修复 + 字段标准化模式
        return repair_jsonl(path, write_back=False, repair=True)

    # 严格解析模式（原逻辑，向后兼容）
    items: List = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"JSONL 解析失败：{path} 第 {line_no} 行 - {e}"
                ) from e
    return items


# ---------------------------------------------------------------------------
# ensure_val_split（Part5K1 Task 6.1）
# ---------------------------------------------------------------------------


def count_lines(path: str) -> int:
    """统计文件中非空行数（空行不计入）。

    用于 :func:`ensure_val_split` 判断 val_path 是否已存在有效样本，
    以及计算切分后的样本计数。
    """
    if not os.path.exists(path):
        return 0
    n = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                n += 1
    return n


def ensure_val_split(
    train_path: str,
    val_path: str,
    val_ratio: float = 0.05,
    write_back: bool = True,
):
    """确保 val 集存在：val_path 不存在或为空时，从 train 末尾切分 val_ratio 比例。

    Part5K1 Task 6.1：自动 val.json 生成。

    逻辑：
        1. 若 ``val_path`` 存在且非空（非空行数 > 0），**保守不动**，
           直接返回 ``(count_lines(train_path), count_lines(val_path))``。
        2. 否则读取 ``train_path``（调用 :func:`load_jsonl` + 自修复），
           从末尾切分 ``val_ratio`` 比例作为 val。
        3. ``write_back=True`` 时把 val 写到 ``val_path``，**不覆盖** train_path
           （保守起见，train 文件原样保留；如果用户想要 train 也更新，可
           后续扩展 ``update_train`` 参数，默认不更新）。
        4. 返回 ``(n_train, n_val)``。

    切分规则：
        - ``n_val = max(1, int(len(data) * val_ratio))``：保证至少 1 条 val
          （``val_ratio=0.0`` 也会切出 1 条）
        - ``n_val`` 不超过 ``len(data)``（``val_ratio=1.0`` 时全部作为 val）
        - ``n_train = len(data) - n_val``
        - 从末尾切：``val_data = data[n_train:]``（保证 val 是最新数据）

    Args:
        train_path: 训练集 JSONL 路径
        val_path: 验证集 JSONL 路径（不存在或空时自动生成）
        val_ratio: val 占 train 的比例（默认 0.05 = 5%）
        write_back: 是否把切分结果写回 val_path（默认 True）

    Returns:
        (n_train, n_val): 切分后的训练/验证样本数。
        若 val_path 已存在且非空，n_train/n_val 反映当前实际计数
        （此时不会重新切分）。

    Raises:
        FileNotFoundError: train_path 不存在
        ValueError: train_path 为空（无可读样本）或 val_ratio 越界
    """
    # 边界校验：val_ratio 必须在 [0, 1]
    if val_ratio < 0 or val_ratio > 1:
        raise ValueError(f"val_ratio 必须在 [0, 1] 范围内，收到 {val_ratio}")

    # 1. 检查 val_path：存在且非空（大小 > 0 且非空行 > 0）则直接返回计数
    if os.path.exists(val_path) and os.path.getsize(val_path) > 0:
        n_existing_val = count_lines(val_path)
        if n_existing_val > 0:
            # val 已存在有效样本，保守不动
            return (count_lines(train_path), n_existing_val)

    # 2. 读取训练样本（启用 JSONL 自修复）
    data = load_jsonl(train_path, repair=True)
    if not data:
        raise ValueError(f"训练集为空：{train_path}")

    # 3. 计算切分点：val_ratio=0.0 时至少 1 条 val，val_ratio=1.0 时全部作为 val
    n_val = max(1, int(len(data) * val_ratio))
    if n_val > len(data):
        n_val = len(data)
    n_train = len(data) - n_val

    # 4. 切分（从末尾切，保证 val 是最新数据）
    val_data = data[len(data) - n_val:]

    # 5. 写回 val_path（不覆盖 train_path）
    if write_back:
        # 确保目录存在
        val_dir = os.path.dirname(os.path.abspath(val_path))
        os.makedirs(val_dir, exist_ok=True)
        with open(val_path, "w", encoding="utf-8") as f:
            for item in val_data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(
            f"[ensure_val_split] 从 train 切分 val: n_train={n_train}, "
            f"n_val={n_val}, val_ratio={val_ratio}",
            flush=True,
        )
    else:
        print(
            f"[ensure_val_split] (write_back=False) 切分 val: "
            f"n_train={n_train}, n_val={n_val}, val_ratio={val_ratio}",
            flush=True,
        )

    return (n_train, n_val)


# ---------------------------------------------------------------------------
# 格式检测
# ---------------------------------------------------------------------------


def _detect_format(item) -> str:
    """检测样本格式。

    Returns:
        ``"chat"`` | ``"prompt_completion"`` | ``"prompt_only"`` |
        ``"completion_only"`` | ``"text"`` | ``"unknown"``
    """
    if isinstance(item, list):
        # chat 数组：每个元素是 {"role":..., "content":...}
        if all(isinstance(x, dict) and "role" in x and "content" in x for x in item):
            return "chat"
        return "unknown"
    if isinstance(item, dict):
        if "prompt" in item and "completion" in item:
            return "prompt_completion"
        # 单样本：只存在 prompt 或只存在 completion
        if "prompt" in item:
            return "prompt_only"
        if "completion" in item:
            return "completion_only"
        if "text" in item:
            return "text"
    return "unknown"


# ---------------------------------------------------------------------------
# 编码辅助（TextDataset 与 CachedDataset 共用）
# ---------------------------------------------------------------------------


def _safe_encode(tok, text: str) -> List[int]:
    """对 tokenizer.encode 兼容不同签名。"""
    if not text:
        return []
    try:
        return list(tok.encode(text, add_special_tokens=False))
    except TypeError:
        try:
            return list(tok.encode(text))
        except Exception:
            return []


def _encode_chat(tok, messages) -> tuple:
    """编码 chat 数组为 (ids, loss_mask)。

    逐消息编码：``<|role|>`` 标记 + content。
    assistant content 参与 loss（mask=1），其他不参与（mask=0）。
    末尾追加 ``<|eos|>``（参与 loss）。
    """
    ids: List[int] = []
    mask: List[int] = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        marker = f"<|{role}|>"
        marker_ids = _safe_encode(tok, marker)
        ids.extend(marker_ids)
        mask.extend([0] * len(marker_ids))
        content_ids = _safe_encode(tok, content)
        ids.extend(content_ids)
        if role == "assistant":
            mask.extend([1] * len(content_ids))
        else:
            mask.extend([0] * len(content_ids))
    eos_ids = _safe_encode(tok, _EOS_STR)
    ids.extend(eos_ids)
    mask.extend([1] * len(eos_ids))
    return ids, mask


def _encode_prompt_completion(tok, item: dict) -> tuple:
    """编码 prompt-completion 为 (ids, loss_mask)。

    拼接为 ``<|user|>prompt<|assistant|>completion<|eos|>``。
    loss mask: completion + <|eos|> 部分为 1，prompt 部分为 0。
    """
    prompt = item.get("prompt", "")
    completion = item.get("completion", "")

    prompt_ids: List[int] = []
    if hasattr(tok, "apply_prompt_template"):
        try:
            prompt_ids = list(tok.apply_prompt_template(prompt))
        except Exception:
            prompt_ids = _safe_encode(tok, prompt)
    else:
        prompt_ids = _safe_encode(tok, prompt)

    completion_ids = _safe_encode(tok, completion)
    eos_ids = _safe_encode(tok, _EOS_STR)

    ids = list(prompt_ids) + completion_ids + eos_ids
    mask = (
        [0] * len(prompt_ids)
        + [1] * len(completion_ids)
        + [1] * len(eos_ids)
    )
    return ids, mask


def _encode_text(tok, text: str) -> tuple:
    """编码纯文本为 (ids, loss_mask)，所有 token 参与 loss。

    末尾追加 ``<|eos|>``（参与 loss），作为样本边界标记。
    """
    ids = _safe_encode(tok, text)
    eos_ids = _safe_encode(tok, _EOS_STR)
    ids = list(ids) + eos_ids
    mask = [1] * len(ids)
    return ids, mask


def _encode_item(tok, item) -> Optional[tuple]:
    """编码单个样本为 (ids, mask)，未知格式返回 None。"""
    fmt = _detect_format(item)
    if fmt == "unknown":
        return None
    if fmt == "chat":
        return _encode_chat(tok, item)
    if fmt == "prompt_completion":
        return _encode_prompt_completion(tok, item)
    # text / prompt_only / completion_only：存在的字段当作纯文本，全部参与 loss
    text = item.get("text") or item.get("prompt") or item.get("completion") or ""
    return _encode_text(tok, text)


# ---------------------------------------------------------------------------
# TextDataset（原 data/demo/src/data_loader.py，保持向后兼容）
# ---------------------------------------------------------------------------


class TextDataset:
    """基于 jsonl + tokenizer 的 next-token prediction 数据集。

    支持三种样本格式：
        1. chat 数组：``[{"role":"user","content":"..."},{"role":"assistant","content":"..."}]``
           → 逐消息编码为 ``<|role|>content``，assistant content 参与 loss
        2. prompt-completion：``{"prompt":"...","completion":"..."}``
           → ``<|user|>prompt<|assistant|>completion<|eos|>``，completion 参与 loss
           （允许单样本：只存在 prompt 或只存在 completion 时，存在字段当作纯文本，全部参与 loss）
        3. text：``{"text":"..."}``
           → 纯文本，所有 token 参与 loss（适用于预训练 / 续训）

    Args:
        tok: tokenizer 对象（需有 ``encode`` 方法；若有
            ``apply_prompt_template`` 则优先使用）
        jsonl_path: jsonl 文件路径
        seq_len: 单个样本长度
        min_tokens: 最少需要的 token 数（保留接口，当前未使用）
    """

    def __init__(
        self,
        tok,
        jsonl_path: str,
        seq_len: int,
        min_tokens: int = 0,
    ):
        self.tok = tok
        self.seq_len = int(seq_len)
        self.jsonl_path = jsonl_path
        self.min_tokens = int(min_tokens)

        # 1. 加载 jsonl
        items = load_jsonl(jsonl_path)

        # 2. 逐样本编码为 (ids, loss_mask)
        all_ids: List[int] = []
        all_mask: List[int] = []

        for item in items:
            encoded = _encode_item(tok, item)
            if encoded is None:
                continue
            ids, mask = encoded
            all_ids.extend(ids)
            all_mask.extend(mask)
            # 样本之间用换行分隔（不参与 loss）
            nl_ids = _safe_encode(tok, "\n")
            all_ids.extend(nl_ids)
            all_mask.extend([0] * len(nl_ids))

        if not all_ids:
            raise ValueError(f"jsonl 中没有有效样本: {jsonl_path}")

        ids_np = np.asarray(all_ids, dtype=np.int64)
        mask_np = np.asarray(all_mask, dtype=np.int64)

        # 3. 截断为 seq_len 的整数倍
        n_blocks = len(ids_np) // self.seq_len
        if n_blocks == 0:
            raise ValueError(
                f"token 数 {len(ids_np)} 不足 seq_len={self.seq_len}，"
                f"无法构造任何样本。请增加数据量或减小 seq_len。"
            )
        self.ids = ids_np[: n_blocks * self.seq_len]
        self.mask = mask_np[: n_blocks * self.seq_len]
        self.n_blocks = n_blocks

    def __len__(self) -> int:
        return self.n_blocks

    def __getitem__(self, i: int):
        """返回 (x, y)，shape (seq_len,)。

        x = ids[i*seq_len : (i+1)*seq_len]
        y = ids[i*seq_len+1 : (i+1)*seq_len+1]（next-token target）
        若某位置 mask=0，y 设为 -100（ignore_index，cross_entropy_loss 自动忽略）。
        """
        if i < 0 or i >= self.n_blocks:
            raise IndexError(f"index {i} 超出范围 [0, {self.n_blocks})")
        start = i * self.seq_len
        end = start + self.seq_len
        x = self.ids[start:end]
        y_start = start + 1
        y_end = end + 1
        if y_end <= len(self.ids):
            y = self.ids[y_start:y_end]
            y_mask = self.mask[y_start:y_end]
        else:
            y_avail = self.ids[y_start: len(self.ids)]
            y_mask_avail = self.mask[y_start: len(self.ids)]
            pad_len = self.seq_len - len(y_avail)
            y = np.concatenate([y_avail, np.zeros(pad_len, dtype=np.int64)])
            y_mask = np.concatenate(
                [y_mask_avail, np.zeros(pad_len, dtype=np.int64)]
            )
        y = np.where(y_mask == 1, y, -100)
        return x, y


# ---------------------------------------------------------------------------
# CachedDataset：.npz 缓存 + 流式 lazy load
# ---------------------------------------------------------------------------


class CachedDataset:
    """带 ``.npz`` 缓存的数据集（Part4K1 Task 6.3）。

    在 :class:`TextDataset` 之上新增：
    1. **首次扫描缓存**：编码完成后把 ``ids`` / ``mask`` 数组以 ``.npz``
       保存到 ``cache_path``（默认 ``<data_path>.cache.npz``）。
       后续启动直接 ``np.load`` 缓存，跳过 jsonl 解析 + tokenizer 编码，
       毫秒级加载。
    2. **流式 lazy load**：``lazy=True`` 时不在 ``__init__`` 全量载入
       内存，而是通过 ``np.load(..., mmap_mode='r')`` 内存映射 .npz 解包
       后的临时文件，按 batch 读取。仅当 ``__getitem__`` 被调用时才
       触发对应区间的磁盘读取。
    3. **后台预加载**（Part5K1 Task 6.2）：``preload=True`` 时，
       ``__init__`` 启动后台线程执行编码 + 缓存写入，主线程可并行构建
       模型；首次 ``__getitem__`` / ``__len__`` 调用时 ``join`` 等待完成。
       适用于大模型 + 大数据集场景，编码耗时与模型构建耗时重叠。

    缓存文件包含：
    - ``ids`` : int64 数组（全部 token id 顺序拼接）
    - ``mask`` : int64 数组（与 ids 等长，loss mask）
    - ``seq_len`` : int 标量
    - ``n_blocks`` : int 标量
    - ``source`` : 字符串（原始 jsonl 路径，便于校验）

    若 ``data_path`` 的 mtime 比缓存文件新，则缓存自动失效重建。

    Args:
        tok: tokenizer 对象
        data_path: jsonl 文件路径
        seq_len: 单个样本长度
        cache_path: 缓存文件路径；None 则默认 ``<data_path>.cache.npz``
        lazy: 是否启用流式 lazy load（默认 False，全量载入内存）
        min_tokens: 最少 token 数（保留接口）
        preload: 是否启用后台预加载（默认 False，同步构建）。
            ``preload=True`` 时 ``__init__`` 立即返回，后台线程异步编码；
            首次 ``__getitem__`` / ``__len__`` 调用时阻塞等待完成。
            若缓存命中（``_try_load_cache`` 成功），preload 不生效，
            直接同步加载。
    """

    def __init__(
        self,
        tok,
        data_path: str,
        seq_len: int,
        cache_path: Optional[str] = None,
        lazy: bool = False,
        min_tokens: int = 0,
        preload: bool = False,
    ):
        self.tok = tok
        self.data_path = data_path
        self.seq_len = int(seq_len)
        self.min_tokens = int(min_tokens)
        self.lazy = bool(lazy)
        self.preload = bool(preload)

        if cache_path is None:
            cache_path = f"{data_path}.cache.npz"
        self.cache_path = cache_path

        # Part5K1 Task 6.2: 后台预加载状态
        # 预先初始化属性，避免 preload 模式下 __getitem__ 早调用时报 AttributeError
        self._preload_thread: Optional[threading.Thread] = None
        self._preload_error: Optional[BaseException] = None
        self._preload_done = threading.Event()
        # 缓存未就位时，self.ids / self.mask / self.n_blocks 用占位值
        self.ids = None
        self.mask = None
        self.n_blocks = 0
        self._archive = None
        self._cached_source = ""

        # 尝试加载缓存；若缓存不存在 / 过期 / 损坏，则重建
        loaded = self._try_load_cache()
        if loaded:
            # 缓存命中：preload 不生效（无编码可做）
            return

        # 缓存未命中
        if self.preload:
            # preload=True：启动后台线程异步编码 + 缓存写入
            # 主线程可继续构建模型，构造函数立即返回
            self._preload_thread = threading.Thread(
                target=self._preload_worker,
                name="CachedDataset-preload",
                daemon=True,
            )
            self._preload_thread.start()
        else:
            # preload=False：同步构建（原逻辑，向后兼容）
            self._build_and_save_cache()
            # 重建后再加载一次（确保 ids/mask 已就位）
            if not self._try_load_cache():
                raise RuntimeError(
                    f"CachedDataset 缓存重建后仍无法加载：{self.cache_path}"
                )

    # ------------------------------------------------------------------
    # 后台预加载（Part5K1 Task 6.2）
    # ------------------------------------------------------------------

    def _preload_worker(self) -> None:
        """后台线程执行编码 + 缓存写入 + 加载。

        任何异常都捕获并记录到 ``self._preload_error``，由主线程在
        ``_wait_for_preload`` 中重新抛出（避免异常被 daemon 线程吞掉）。
        """
        try:
            self._build_and_save_cache()
            loaded = self._try_load_cache()
            if not loaded:
                self._preload_error = RuntimeError(
                    f"CachedDataset 缓存重建后仍无法加载：{self.cache_path}"
                )
        except BaseException as e:  # noqa: BLE001
            # 捕获所有异常（含 SystemExit / KeyboardInterrupt 子类），
            # 让主线程能看到失败原因
            self._preload_error = e
        finally:
            self._preload_done.set()

    def _wait_for_preload(self) -> None:
        """等待后台预加载完成（首次 ``__getitem__`` / ``__len__`` 时调用）。

        若预加载线程未启动（preload=False 或缓存命中），立即返回。
        否则阻塞等待 ``_preload_done`` 事件，完成后检查是否有异常。
        """
        if self._preload_thread is None:
            return
        if not self._preload_done.is_set():
            print(
                "[CachedDataset] 等待后台预加载完成...",
                flush=True,
            )
            self._preload_done.wait()
        # 清理线程引用（join 用于回收资源，事件已保证完成）
        if self._preload_thread is not None and self._preload_thread.is_alive():
            # 极端情况下事件已 set 但线程未退出（如 finally 中阻塞），
            # 给 5 秒超时避免死锁
            self._preload_thread.join(timeout=5.0)
        self._preload_thread = None
        # 检查后台线程是否抛了异常
        if self._preload_error is not None:
            err = self._preload_error
            self._preload_error = None
            raise err

    # ------------------------------------------------------------------
    # 缓存加载 / 重建
    # ------------------------------------------------------------------

    def _try_load_cache(self) -> bool:
        """尝试从 cache_path 加载 ids/mask。

        Returns:
            True 表示成功加载（self.ids / self.mask / self.n_blocks 已就位）；
            False 表示缓存不存在 / 过期 / 损坏，需要重建。
        """
        if not os.path.exists(self.cache_path):
            return False
        # 缓存过期检测：源文件比缓存新则失效
        try:
            src_mtime = os.path.getmtime(self.data_path)
            cache_mtime = os.path.getmtime(self.cache_path)
            if src_mtime > cache_mtime:
                return False
        except OSError:
            return False

        try:
            if self.lazy:
                # mmap_mode='r' 内存映射：不占内存，按需读盘
                archive = np.load(self.cache_path, allow_pickle=False, mmap_mode='r')
                self.ids = archive["ids"]
                self.mask = archive["mask"]
                self._archive = archive  # 持有引用避免文件关闭
            else:
                archive = np.load(self.cache_path, allow_pickle=False)
                self.ids = np.asarray(archive["ids"], dtype=np.int64)
                self.mask = np.asarray(archive["mask"], dtype=np.int64)
                self._archive = None
            self.n_blocks = int(archive["n_blocks"])
            cached_seq_len = int(archive["seq_len"])
            if cached_seq_len != self.seq_len:
                # seq_len 变了，缓存失效
                self._close_archive()
                return False
            self._cached_source = str(archive["source"]) if "source" in archive.files else ""
            return True
        except Exception as e:
            print(f"[CachedDataset] 缓存加载失败，将重建：{e}", flush=True)
            self._close_archive()
            return False

    def _close_archive(self) -> None:
        """关闭 mmap archive（若有）。"""
        arc = getattr(self, "_archive", None)
        if arc is not None:
            try:
                arc.close()
            except Exception:
                pass
            self._archive = None

    def _build_and_save_cache(self) -> None:
        """扫描 jsonl + 编码 → 保存到 cache_path。"""
        print(f"[CachedDataset] 首次扫描数据集 {self.data_path}，构建缓存...",
              flush=True)
        items = load_jsonl(self.data_path)

        all_ids: List[int] = []
        all_mask: List[int] = []
        for item in items:
            encoded = _encode_item(self.tok, item)
            if encoded is None:
                continue
            ids, mask = encoded
            all_ids.extend(ids)
            all_mask.extend(mask)
            nl_ids = _safe_encode(self.tok, "\n")
            all_ids.extend(nl_ids)
            all_mask.extend([0] * len(nl_ids))

        if not all_ids:
            raise ValueError(f"jsonl 中没有有效样本: {self.data_path}")

        ids_np = np.asarray(all_ids, dtype=np.int64)
        mask_np = np.asarray(all_mask, dtype=np.int64)
        n_blocks = len(ids_np) // self.seq_len
        if n_blocks == 0:
            raise ValueError(
                f"token 数 {len(ids_np)} 不足 seq_len={self.seq_len}，"
                f"无法构造任何样本。请增加数据量或减小 seq_len。"
            )
        ids_np = ids_np[: n_blocks * self.seq_len]
        mask_np = mask_np[: n_blocks * self.seq_len]

        # 保存缓存（覆盖旧文件）
        cache_dir = os.path.dirname(os.path.abspath(self.cache_path))
        os.makedirs(cache_dir, exist_ok=True)
        np.savez(
            self.cache_path,
            ids=ids_np,
            mask=mask_np,
            seq_len=np.int64(self.seq_len),
            n_blocks=np.int64(n_blocks),
            source=np.asarray(self.data_path),
        )
        print(f"[CachedDataset] 缓存已保存到 {self.cache_path} "
              f"(n_blocks={n_blocks}, tokens={len(ids_np)})", flush=True)

    # ------------------------------------------------------------------
    # __len__ / __getitem__（与 TextDataset 接口一致）
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        # Part5K1 Task 6.2: preload 模式下首次调用需等待后台编码完成
        self._wait_for_preload()
        return self.n_blocks

    def __getitem__(self, i: int):
        """返回 (x, y)，shape (seq_len,)。

        与 :class:`TextDataset.__getitem__` 完全一致；mask=0 的位置 y=-100。
        lazy 模式下 self.ids 是 mmap 数组，切片会触发磁盘读取并返回
        普通 ndarray（拷贝），不持有 mmap 引用。

        Part5K1 Task 6.2: preload 模式下首次调用 ``_wait_for_preload``
        阻塞等待后台编码完成。
        """
        # Part5K1 Task 6.2: preload 模式下首次调用需等待后台编码完成
        self._wait_for_preload()
        if i < 0 or i >= self.n_blocks:
            raise IndexError(f"index {i} 超出范围 [0, {self.n_blocks})")
        start = i * self.seq_len
        end = start + self.seq_len
        # 切片：mmap 数组切片自动转普通 ndarray（拷贝到内存）
        x = np.array(self.ids[start:end], dtype=np.int64)
        y_start = start + 1
        y_end = end + 1
        if y_end <= len(self.ids):
            y = np.array(self.ids[y_start:y_end], dtype=np.int64)
            y_mask = np.array(self.mask[y_start:y_end], dtype=np.int64)
        else:
            y_avail = np.array(self.ids[y_start: len(self.ids)], dtype=np.int64)
            y_mask_avail = np.array(self.mask[y_start: len(self.ids)], dtype=np.int64)
            pad_len = self.seq_len - len(y_avail)
            y = np.concatenate([y_avail, np.zeros(pad_len, dtype=np.int64)])
            y_mask = np.concatenate(
                [y_mask_avail, np.zeros(pad_len, dtype=np.int64)]
            )
        y = np.where(y_mask == 1, y, -100)
        return x, y


# ---------------------------------------------------------------------------
# collate_fn
# ---------------------------------------------------------------------------


def collate_fn(batch, pad_id: int = 0):
    """把 list of (x, y) 堆叠为 batched ndarray。

    假设所有 (x, y) 长度一致（TextDataset / CachedDataset 保证），直接 stack。

    Args:
        batch: List[Tuple[ndarray, ndarray]]
        pad_id: padding id（PoC 数据集已保证等长，此参数保留兼容）
    Returns:
        (x_batch, y_batch)：均为 np.ndarray，shape (B, seq_len)
    """
    xs = [np.asarray(b[0], dtype=np.int64) for b in batch]
    ys = [np.asarray(b[1], dtype=np.int64) for b in batch]
    x_batch = np.stack(xs, axis=0)
    y_batch = np.stack(ys, axis=0)
    return x_batch, y_batch


# ---------------------------------------------------------------------------
# BatchLoader：把 dataset + collate_fn 组装成可迭代的 batch 生成器
# ---------------------------------------------------------------------------


class BatchLoader:
    """简易 batch loader：可迭代，每次返回一个 batch。

    与 PyTorch DataLoader 接口对齐（仅 PoC 所需的最小集）。
    ``num_workers`` / ``persistent_workers`` 为占位参数（CPU-only 实现忽略）。

    Part4K2 Task 5.4: ``pin_memory`` 实装 + 数据预取（prefetch）：
    - ``pin_memory=True`` 且 PyTorch 可用时，把每个 batch 的 ndarray 通过
      ``torch.tensor(...).pin_memory()`` 预拷贝到锁页内存，便于后续
      ``non_blocking=True`` 异步传输到 GPU。
    - ``prefetch=True``（默认与 pin_memory 同步开启）时，当前 batch 训练的
      同时在后台线程预取下一个 batch，掩盖 IO 延迟。

    Part5K1 Task 6.2 升级：``prefetch`` 不再依赖 torch，纯 threading 预取
    在无 torch 环境也能工作（``pin_memory`` 仍依赖 torch，无 torch 时自动降级）。

    Args:
        dataset: TextDataset / CachedDataset 或任何实现 __len__ / __getitem__ 的对象
        batch_size: batch 大小
        shuffle: 是否每轮打乱顺序
        collate_fn: 批处理函数
        drop_last: 是否丢弃最后不足 batch_size 的样本
        seed: 随机种子（仅 shuffle=True 时生效）
        num_workers: 占位参数（CPU-only，默认 0，忽略）
        pin_memory: 是否启用锁页内存预拷贝（默认 False，无 torch 时强制 False）
        persistent_workers: 占位参数（默认 False，忽略）
        prefetch: 是否启用预取（默认与 pin_memory 同步开启；不依赖 torch）
    """

    def __init__(
        self,
        dataset,
        batch_size: int = 16,
        shuffle: bool = True,
        collate_fn=None,
        drop_last: bool = False,
        seed: int = 0,
        num_workers: int = 0,
        pin_memory: bool = False,
        persistent_workers: bool = False,
        prefetch: Optional[bool] = None,
    ):
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.collate_fn = collate_fn or (lambda b: tuple(np.stack(x) for x in zip(*b)))
        self.drop_last = bool(drop_last)
        self.rng = np.random.RandomState(seed)
        self._epoch = 0
        # 占位参数（CPU-only 实现忽略）
        self.num_workers = num_workers
        self.pin_memory = bool(pin_memory)
        self.persistent_workers = persistent_workers
        # Part4K2 Task 5.4: prefetch 默认跟随 pin_memory
        self.prefetch = bool(prefetch) if prefetch is not None else self.pin_memory
        # 检测 torch 可用性（pin_memory 依赖 torch；prefetch 不依赖 torch）
        try:
            import torch as _torch  # type: ignore
            self._torch = _torch
        except Exception:
            self._torch = None
            # 无 torch 时仅 pin_memory 降级（pin_memory 需要 torch.Tensor.pin_memory）
            # prefetch 保持用户设置不变：Part5K1 Task 6.2 升级为纯 threading 预取，
            # 不再依赖 torch，可在无 torch 环境正常工作
            self.pin_memory = False

    def __len__(self) -> int:
        n = len(self.dataset)
        if self.drop_last:
            return n // self.batch_size
        return (n + self.batch_size - 1) // self.batch_size

    def _pin_batch(self, batch):
        """把 batch 中的 ndarray 转成 pinned torch.Tensor（GPU 异步传输用）。

        batch 通常是 ``(x_batch, y_batch)`` 元组，每个元素是 ndarray。
        转换为 ``torch.Tensor`` 并调用 ``.pin_memory()``。
        若 batch 结构复杂或转换失败，原样返回。
        """
        if not self.pin_memory or self._torch is None:
            return batch
        try:
            if isinstance(batch, (tuple, list)):
                pinned = []
                for x in batch:
                    if isinstance(x, np.ndarray):
                        t = self._torch.from_numpy(x)
                        try:
                            t = t.pin_memory()
                        except Exception:
                            pass
                        pinned.append(t)
                    elif self._torch is not None and isinstance(x, self._torch.Tensor):
                        try:
                            if not x.is_pinned():
                                x = x.pin_memory()
                        except Exception:
                            pass
                        pinned.append(x)
                    else:
                        pinned.append(x)
                return tuple(pinned) if isinstance(batch, tuple) else pinned
            if isinstance(batch, np.ndarray):
                t = self._torch.from_numpy(batch)
                try:
                    t = t.pin_memory()
                except Exception:
                    pass
                return t
        except Exception:
            pass
        return batch

    def __iter__(self):
        """迭代一个 epoch 的所有 batch。

        Part4K2 Task 5.4: prefetch=True 时，下一个 batch 在后台线程预取，
        当前 batch yield 后立即可用，掩盖 collate + pin_memory 耗时。

        Part5K1 Task 6.2 升级：prefetch 不再依赖 torch，纯 threading 预取
        在无 torch 环境也能工作（pin_memory 仍依赖 torch，无 torch 时自动降级）。
        """
        n = len(self.dataset)
        indices = np.arange(n)
        if self.shuffle:
            self.rng.shuffle(indices)

        # 把 batch 索引切片预先收集
        batch_slices = []
        for s in range(0, n, self.batch_size):
            batch_idx = indices[s: s + self.batch_size]
            if len(batch_idx) == 0:
                continue
            if self.drop_last and len(batch_idx) < self.batch_size:
                break
            batch_slices.append(batch_idx)

        def _materialize(b_idx):
            batch = [self.dataset[int(i)] for i in b_idx]
            collated = self.collate_fn(batch)
            # pin_memory 预拷贝到锁页内存（无 torch 时 _pin_batch 原样返回）
            return self._pin_batch(collated)

        if not self.prefetch or len(batch_slices) <= 1:
            # 不预取：同步迭代
            for b_idx in batch_slices:
                yield _materialize(b_idx)
            return

        # prefetch=True：用一个后台线程预取下一个 batch（纯 threading，不依赖 torch）
        import queue as _queue

        # 队列长度 1（只预取一个 batch，避免内存翻倍）
        q: "_queue.Queue" = _queue.Queue(maxsize=1)
        sentinel = object()
        stop_flag = threading.Event()

        def _producer():
            for b_idx in batch_slices:
                if stop_flag.is_set():
                    return
                try:
                    q.put(_materialize(b_idx))
                except Exception as e:
                    q.put(e)
                    return
            q.put(sentinel)

        producer = threading.Thread(target=_producer, daemon=True)
        producer.start()
        try:
            while True:
                item = q.get()
                if item is sentinel:
                    return
                if isinstance(item, Exception):
                    # 预取过程中出错：向上抛
                    raise item
                yield item
        finally:
            # 主线程退出（含异常）时通知 producer 停止
            stop_flag.set()
            # 排空队列避免 producer 阻塞在 put
            try:
                while True:
                    q.get_nowait()
            except _queue.Empty:
                pass
            producer.join(timeout=1.0)


# ---------------------------------------------------------------------------
# 单样本构造（Task 6.5：--single-sample 支持）
# ---------------------------------------------------------------------------


class SingleSampleDataset:
    """单样本数据集：把一条 prompt/completion 或单文件文本包装为 dataset。

    用于 ``--single-sample --prompt "..." --completion "..."`` 的快速训练/推理调试。
    内部复用 :func:`_encode_item` 编码逻辑，重复填充到至少 1 个 seq_len 块。

    Args:
        tok: tokenizer 对象
        prompt: prompt 文本（可空）
        completion: completion 文本（可空）
        text: 纯文本（与 prompt/completion 二选一）
        seq_len: 单个样本长度
        n_repeat: 把单样本重复多少次拼接到一起（保证至少 1 个 seq_len 块）
    """

    def __init__(
        self,
        tok,
        prompt: str = "",
        completion: str = "",
        text: str = "",
        seq_len: int = 32,
        n_repeat: int = 8,
    ):
        self.tok = tok
        self.seq_len = int(seq_len)

        if text:
            item = {"text": text}
        elif prompt and completion:
            item = {"prompt": prompt, "completion": completion}
        elif prompt:
            item = {"prompt": prompt}
        elif completion:
            item = {"completion": completion}
        else:
            item = {"text": ""}

        encoded = _encode_item(tok, item)
        if encoded is None or not encoded[0]:
            # 编码失败：用 0 token 兜底
            ids = [0]
            mask = [1]
        else:
            ids, mask = encoded

        # 重复 n_repeat 次拼接到一起，保证至少 1 个 seq_len 块
        all_ids = list(ids) * max(1, n_repeat)
        all_mask = list(mask) * max(1, n_repeat)

        ids_np = np.asarray(all_ids, dtype=np.int64)
        mask_np = np.asarray(all_mask, dtype=np.int64)
        n_blocks = max(1, len(ids_np) // self.seq_len)
        self.ids = ids_np[: n_blocks * self.seq_len]
        self.mask = mask_np[: n_blocks * self.seq_len]
        self.n_blocks = n_blocks

    def __len__(self) -> int:
        return self.n_blocks

    def __getitem__(self, i: int):
        if i < 0 or i >= self.n_blocks:
            raise IndexError(f"index {i} 超出范围 [0, {self.n_blocks})")
        start = i * self.seq_len
        end = start + self.seq_len
        x = self.ids[start:end]
        y_start = start + 1
        y_end = end + 1
        if y_end <= len(self.ids):
            y = self.ids[y_start:y_end]
            y_mask = self.mask[y_start:y_end]
        else:
            y_avail = self.ids[y_start: len(self.ids)]
            y_mask_avail = self.mask[y_start: len(self.ids)]
            pad_len = self.seq_len - len(y_avail)
            y = np.concatenate([y_avail, np.zeros(pad_len, dtype=np.int64)])
            y_mask = np.concatenate(
                [y_mask_avail, np.zeros(pad_len, dtype=np.int64)]
            )
        y = np.where(y_mask == 1, y, -100)
        return x, y


__all__ = [
    "load_jsonl",
    "count_lines",
    "ensure_val_split",
    "TextDataset",
    "CachedDataset",
    "SingleSampleDataset",
    "collate_fn",
    "BatchLoader",
]
