"""CometSpark tokenizer 工厂：build_tokenizer / load_tokenizer。

封装 verse_tokenizer 包，统一 PoC 阶段的 tokenizer 构建与加载流程。

Part4 升级：
    - 新增 ``kind="qwen"`` 分支，支持直接复用 Qwen2.5-32B-Instruct 的优质
      BPE tokenizer（HF ``tokenizer.json`` 格式）。
    - ``kind="qwen"`` 时 ``build_tokenizer`` 会把预下载的 Qwen tokenizer.json
      原样复制到 save_path（保留 HF 格式，避免 save 自有格式时丢失 Split
      正则与 NFC normalizer），训练 / 推理阶段用 :meth:`BPETokenizer.from_file`
      加载即可保留完整预处理能力。
"""

from __future__ import annotations

import os
import shutil

from verse_tokenizer import BPETokenizer, ByteTokenizer, load_tokenizer as _vload


# 预下载的 Qwen2.5-32B-Instruct tokenizer.json 路径（相对 demo 根目录）。
# 若不存在，build_tokenizer(kind="qwen") 会尝试从 hf-mirror.com 镜像下载。
_DEFAULT_QWEN_TOKENIZER_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "qwen_tokenizer",
    "tokenizer.json",
)


def _download_qwen_tokenizer(save_path: str) -> str:
    """从 hf-mirror.com 镜像下载 Qwen2.5-32B-Instruct tokenizer.json。

    HuggingFace 官方 SSL 在部分沙箱环境不稳定，使用 hf-mirror.com 镜像更可靠。
    """
    os.makedirs(os.path.dirname(os.path.abspath(save_path)) or ".", exist_ok=True)
    url = (
        "https://hf-mirror.com/Qwen/Qwen2.5-32B-Instruct/"
        "resolve/main/tokenizer.json"
    )
    print(f"[build_tokenizer] 从 {url} 下载 Qwen tokenizer...", flush=True)
    import ssl
    import urllib.request

    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    req = urllib.request.Request(url, headers={"User-Agent": "verse-tokenizer/0.2"})
    with urllib.request.urlopen(req, context=ctx, timeout=60) as resp:
        data = resp.read()
    with open(save_path, "wb") as f:
        f.write(data)
    print(
        f"[build_tokenizer] 下载完成，size={len(data)} bytes → {save_path}",
        flush=True,
    )
    return save_path


def build_tokenizer(
    corpus_path: str,
    vocab_size: int,
    save_path: str,
    kind: str = "bpe",
    source_path: str | None = None,
) -> str:
    """从语料构建 tokenizer 并保存到 save_path。

    Args:
        corpus_path: 训练语料文件路径（纯文本，UTF-8）。仅 kind="bpe" 时使用。
        vocab_size: 目标词表大小（仅 kind="bpe" 时生效）。
        save_path: 保存路径。
        kind:
            - ``"bpe"``：调 :meth:`BPETokenizer.train` 从语料训练 BPE。
            - ``"byte"``：直接构造 ByteTokenizer（vocab_size=259）。
            - ``"qwen"``：Part4 新增。复制预下载的 Qwen2.5-32B-Instruct
              ``tokenizer.json`` 到 save_path（原样保留 HF 格式，含 Split
              正则 + NFC normalizer）。``source_path`` 可指定自定义来源，
              默认用 ``_DEFAULT_QWEN_TOKENIZER_PATH``，若不存在则自动下载。
        source_path: 仅 kind="qwen" 时生效，指定源 tokenizer.json 路径。
            None 时用默认 Qwen 路径。

    Returns:
        实际保存路径
    """
    os.makedirs(os.path.dirname(os.path.abspath(save_path)) or ".", exist_ok=True)

    if kind == "bpe":
        with open(corpus_path, "r", encoding="utf-8") as f:
            corpus = f.read()
        tok = BPETokenizer.train(corpus, vocab_size=int(vocab_size))
        tok.save(save_path)
        return save_path

    if kind == "byte":
        # ByteTokenizer 固定 vocab_size=259，vocab_size 参数忽略
        tok = ByteTokenizer()
        tok.save(save_path)
        return save_path

    if kind == "qwen":
        # Part4: 复制预下载的 Qwen tokenizer.json（原样保留 HF 格式，
        # 避免 save 自有格式时丢失 Split 正则 + NFC normalizer）。
        src = source_path or _DEFAULT_QWEN_TOKENIZER_PATH
        if not os.path.exists(src):
            # 自动下载到默认位置
            _download_qwen_tokenizer(src)
        if not os.path.exists(src):
            raise FileNotFoundError(
                f"kind='qwen' 源 tokenizer.json 不存在：{src}，"
                f"且自动下载失败。请手动放置后重试。"
            )
        # 原样复制（不是 from_file + save，避免丢失 _split_regex/_normalizer）
        shutil.copyfile(src, save_path)
        # 校验：能正常加载并打印 vocab_size
        tok = BPETokenizer.from_file(save_path)
        print(
            f"[build_tokenizer] kind=qwen, vocab_size={len(tok)}, "
            f"source={src}, save={save_path}",
            flush=True,
        )
        return save_path

    raise ValueError(
        f"Unknown tokenizer kind: {kind!r} (expected 'bpe'/'byte'/'qwen')"
    )


def load_tokenizer(path: str, kind: str = "bpe"):
    """加载已保存的 tokenizer。

    Args:
        path: tokenizer 文件路径。
        kind:
            - ``"bpe"``：调 :meth:`BPETokenizer.load`（自动识别 HF / 自有格式）。
            - ``"byte"``：调 :meth:`ByteTokenizer.load`。
            - ``"qwen"``：Part4 新增。强制用 :meth:`BPETokenizer.from_file`
              解析 HF ``tokenizer.json``，完整保留 Split 正则 + NFC normalizer。

    Returns:
        tokenizer 对象，统一接口：
            - ``encode(text)`` → ``List[int]``
            - ``decode(ids)`` → ``str``
            - ``__len__()`` → vocab_size
    """
    return _vload(kind=kind, path=path)


__all__ = ["build_tokenizer", "load_tokenizer"]
