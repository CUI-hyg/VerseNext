#!/usr/bin/env python3
"""下载并构建 Qwen3-32B tokenizer（CometSpark-V0.2 使用）。

设计目标
--------
用户指定 CometSpark-V0.2 使用 Qwen3-32B 优质 tokenizer（vocab_size≈151936）。
本脚本负责：

1. 用 ``huggingface_hub.snapshot_download`` 下载 ``Qwen/Qwen3-32B`` 的 tokenizer 文件
   （``allow_patterns=["tokenizer*"]``，仅下载 tokenizer 相关文件，跳过权重）
2. 失败时降级到 ``Qwen/Qwen2.5-32B-Instruct`` 或 ``Qwen/Qwen2.5-14B-Instruct``
3. 复制 ``tokenizer.json`` + ``tokenizer_config.json`` + 相关文件到目标目录
4. 打印 vocab_size 并验证 encode/decode 往返一致性

输出目标：``data/demo/checkpoints_verse_nex/tokenizer.json``
（与 ``config_verse_nex.yml`` 的 ``checkpoint.save_dir`` 对齐）

使用方式
--------
    cd /workspace/data/demo/scripts
    python build_tokenizer.py                                    # 默认输出
    python build_tokenizer.py --output-dir /path/to/dir         # 自定义输出目录
    python build_tokenizer.py --repo-id Qwen/Qwen3-32B          # 指定模型 repo
    python build_tokenizer.py --skip-verify                     # 跳过 encode/decode 验证

依赖（可选，未安装时自动降级）：
    pip install huggingface_hub transformers
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from typing import Optional


# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

# 默认目标目录：与 config_verse_nex.yml 的 checkpoint.save_dir 对齐
DEFAULT_OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..",
    "checkpoints_verse_nex",
)

# 主选 + 备选 repo（按优先级回退）
CANDIDATE_REPOS: tuple[str, ...] = (
    "Qwen/Qwen3-32B",
    "Qwen/Qwen2.5-32B-Instruct",
    "Qwen/Qwen2.5-14B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",
)

# 需要复制的 tokenizer 相关文件名（用 allow_patterns 通配）
TOKENIZER_FILE_PATTERNS: tuple[str, ...] = (
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.json",
    "merges.txt",
    "added_tokens.json",
    "generation_config.json",
    "chat_template.jinja",
    "tokenizer.model",  # SentencePiece 模型（部分 Qwen 模型保留）
)


# ---------------------------------------------------------------------------
# 日志工具
# ---------------------------------------------------------------------------


def _log(msg: str, level: str = "INFO") -> None:
    """统一日志格式。"""
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] [{level}] {msg}", flush=True)


def _ensure_dir(path: str) -> None:
    """确保目录存在。"""
    os.makedirs(path, exist_ok=True)


# ---------------------------------------------------------------------------
# 方法 1: huggingface_hub.snapshot_download（推荐）
# ---------------------------------------------------------------------------


def _try_snapshot_download(
    repo_id: str,
    target_dir: str,
) -> Optional[str]:
    """用 huggingface_hub.snapshot_download 下载 tokenizer 文件。

    Args:
        repo_id: HuggingFace 模型 ID（如 ``Qwen/Qwen3-32B``）
        target_dir: 下载到的本地目录

    Returns:
        下载后的本地目录路径；失败返回 None
    """
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        _log(f"huggingface_hub 未安装，无法用 snapshot_download 下载 {repo_id}", "WARN")
        return None
    except Exception as e:  # noqa: BLE001
        _log(f"huggingface_hub 导入失败：{e}", "WARN")
        return None

    _log(f"snapshot_download: repo_id={repo_id} -> {target_dir}")
    try:
        local_dir = snapshot_download(
            repo_id=repo_id,
            allow_patterns=list(TOKENIZER_FILE_PATTERNS),
            local_dir=target_dir,
        )
        if local_dir and os.path.isdir(local_dir):
            return local_dir
    except Exception as e:  # noqa: BLE001
        _log(f"snapshot_download {repo_id} 失败：{e}", "ERROR")
        return None
    return None


# ---------------------------------------------------------------------------
# 方法 2: transformers.AutoTokenizer（兜底）
# ---------------------------------------------------------------------------


def _try_transformers_save(
    repo_id: str,
    target_dir: str,
) -> Optional[str]:
    """用 transformers.AutoTokenizer.from_pretrained + save_pretrained 兜底。

    该方法会触发 transformers 内部的下载逻辑，并保存完整 tokenizer 目录。
    """
    try:
        from transformers import AutoTokenizer
    except ImportError:
        _log(f"transformers 未安装，无法用 AutoTokenizer 下载 {repo_id}", "WARN")
        return None
    except Exception as e:  # noqa: BLE001
        _log(f"transformers 导入失败：{e}", "WARN")
        return None

    _log(f"AutoTokenizer.from_pretrained: {repo_id}")
    try:
        tok = AutoTokenizer.from_pretrained(repo_id, trust_remote_code=True)
        _ensure_dir(target_dir)
        tok.save_pretrained(target_dir)
        _log(f"save_pretrained 成功 -> {target_dir}")
        return target_dir
    except Exception as e:  # noqa: BLE001
        _log(f"AutoTokenizer 下载 {repo_id} 失败：{e}", "ERROR")
        return None


# ---------------------------------------------------------------------------
# 主入口：download_tokenizer
# ---------------------------------------------------------------------------


def download_tokenizer(
    repo_id: Optional[str],
    output_dir: str,
    cache_root: str,
) -> Optional[str]:
    """下载 tokenizer，返回本地目录路径，失败返回 None。

    Args:
        repo_id: 指定的模型 ID；None 表示按 CANDIDATE_REPOS 顺序尝试
        output_dir: 最终输出目录
        cache_root: snapshot_download 缓存根目录

    Returns:
        下载后的本地目录路径；全部失败返回 None
    """
    repos_to_try: tuple[str, ...]
    if repo_id:
        repos_to_try = (repo_id,)
    else:
        repos_to_try = CANDIDATE_REPOS

    _ensure_dir(cache_root)
    _ensure_dir(output_dir)

    for idx, rid in enumerate(repos_to_try, start=1):
        _log(f"=== 尝试 repo [{idx}/{len(repos_to_try)}]: {rid} ===")
        # 每个 repo 用独立的缓存子目录，避免互相覆盖
        cache_dir = os.path.join(cache_root, rid.replace("/", "__"))
        _ensure_dir(cache_dir)

        # 方法 1: snapshot_download
        local_dir = _try_snapshot_download(rid, cache_dir)
        if not local_dir:
            # 方法 2: transformers 兜底
            local_dir = _try_transformers_save(rid, cache_dir)

        if not local_dir:
            _log(f"[{rid}] 所有下载方式均失败", "WARN")
            continue

        # 把 tokenizer 相关文件复制到 output_dir
        copied = _copy_tokenizer_files(local_dir, output_dir)
        if copied == 0:
            _log(f"[{rid}] 缓存目录下未找到 tokenizer 文件：{local_dir}", "WARN")
            continue

        _log(f"[{rid}] 成功下载并复制 {copied} 个 tokenizer 文件 -> {output_dir}")
        return output_dir

    _log(f"所有候选 repo 均失败：{repos_to_try}", "ERROR")
    return None


def _copy_tokenizer_files(src_dir: str, dst_dir: str) -> int:
    """把 src_dir 下的 tokenizer 相关文件复制到 dst_dir。

    Returns:
        成功复制的文件数
    """
    _ensure_dir(dst_dir)
    n = 0
    for fname in TOKENIZER_FILE_PATTERNS:
        # 精确文件名匹配
        src = os.path.join(src_dir, fname)
        if os.path.isfile(src):
            dst = os.path.join(dst_dir, fname)
            try:
                shutil.copy2(src, dst)
                n += 1
                _log(f"  复制 {fname} -> {dst}")
            except Exception as e:  # noqa: BLE001
                _log(f"  复制 {fname} 失败：{e}", "WARN")

    # 兜底：把 src_dir 下所有 tokenizer 开头的文件也复制一份
    if os.path.isdir(src_dir):
        for fname in os.listdir(src_dir):
            fl = fname.lower()
            if fl.startswith("tokenizer") and not os.path.isfile(os.path.join(dst_dir, fname)):
                src = os.path.join(src_dir, fname)
                if os.path.isfile(src):
                    try:
                        shutil.copy2(src, os.path.join(dst_dir, fname))
                        n += 1
                        _log(f"  复制 {fname} -> {os.path.join(dst_dir, fname)}")
                    except Exception as e:  # noqa: BLE001
                        _log(f"  复制 {fname} 失败：{e}", "WARN")

    return n


# ---------------------------------------------------------------------------
# 验证：encode / decode 往返
# ---------------------------------------------------------------------------


def verify_tokenizer(tokenizer_dir: str) -> bool:
    """验证 tokenizer 可加载且 encode/decode 往返一致。

    Args:
        tokenizer_dir: 含 tokenizer.json 的目录

    Returns:
        True 验证通过，False 验证失败
    """
    tokenizer_json = os.path.join(tokenizer_dir, "tokenizer.json")
    if not os.path.isfile(tokenizer_json):
        _log(f"未找到 {tokenizer_json}，跳过验证", "WARN")
        return False

    # 读取 vocab_size（从 tokenizer.json 中解析 model.vocab 字段长度）
    vocab_size = _read_vocab_size_from_json(tokenizer_json)
    _log(f"tokenizer.json vocab_size = {vocab_size}")

    # 优先用 transformers 验证 encode/decode
    try:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(tokenizer_dir, trust_remote_code=True)
        _log(f"AutoTokenizer 加载成功：vocab_size={tok.vocab_size}")

        test_cases = [
            "你好，世界！",
            "Hello, world!",
            "床前明月光，疑是地上霜。",
            "The quick brown fox jumps over the lazy dog.",
            "1+1=2，2+3=5。",
        ]

        all_pass = True
        for text in test_cases:
            try:
                ids = tok.encode(text, add_special_tokens=False)
                decoded = tok.decode(ids, skip_special_tokens=True)
                if decoded != text:
                    _log(f"  往返不一致：{text!r} -> {decoded!r}", "WARN")
                    all_pass = False
                else:
                    _log(f"  OK ({len(ids)} tokens): {text!r}")
            except Exception as e:  # noqa: BLE001
                _log(f"  encode/decode 失败：{text!r} - {e}", "ERROR")
                all_pass = False

        if all_pass:
            _log("✓ transformers 验证全部通过")
        else:
            _log("⚠ 部分用例往返不一致（可能是 normalizer 行为，不一定是 bug）", "WARN")
        return all_pass

    except ImportError:
        _log("transformers 未安装，跳过 encode/decode 验证", "WARN")
        return True  # 不视为失败
    except Exception as e:  # noqa: BLE001
        _log(f"transformers 验证异常：{e}", "ERROR")
        return False


def _read_vocab_size_from_json(tokenizer_json_path: str) -> int:
    """从 HuggingFace tokenizer.json 解析 vocab_size。

    tokenizer.json 结构（BPE）::

        {
          "model": {
            "vocab": {"foo": 0, "bar": 1, ...},
            "merges": [...]
          },
          "added_tokens": [...]
        }

    Returns:
        vocab 大小（vocab + added_tokens 数）；解析失败返回 0
    """
    try:
        with open(tokenizer_json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:  # noqa: BLE001
        _log(f"读取 tokenizer.json 失败：{e}", "WARN")
        return 0

    size = 0
    model = data.get("model")
    if isinstance(model, dict):
        vocab = model.get("vocab")
        if isinstance(vocab, dict):
            size = len(vocab)
    added = data.get("added_tokens")
    if isinstance(added, list):
        size += len(added)
    return size


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description="下载并构建 Qwen3-32B tokenizer（CometSpark-V0.2）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help=f"输出目录（默认：{DEFAULT_OUTPUT_DIR}）",
    )
    parser.add_argument(
        "--cache-root",
        default=os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "raw",
            "_hf_cache",
        ),
        help="HuggingFace 缓存根目录（默认：./raw/_hf_cache）",
    )
    parser.add_argument(
        "--repo-id",
        default=None,
        help="指定模型 ID（默认按候选顺序尝试 Qwen3-32B → Qwen2.5-32B → Qwen2.5-14B）",
    )
    parser.add_argument(
        "--skip-verify",
        action="store_true",
        help="跳过 encode/decode 验证",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="强制重新下载（即使目标目录已有 tokenizer.json）",
    )
    args = parser.parse_args()

    output_dir = os.path.abspath(args.output_dir)
    cache_root = os.path.abspath(args.cache_root)

    _log(f"输出目录：{output_dir}")
    _log(f"缓存根目录：{cache_root}")

    # 若已有 tokenizer.json 且未指定 --force，跳过下载
    existing = os.path.join(output_dir, "tokenizer.json")
    if os.path.isfile(existing) and not args.force:
        _log(f"已存在 {existing}，跳过下载（--force 可强制重下）")
        if not args.skip_verify:
            verify_tokenizer(output_dir)
        _log(f"\n下一步：将 {output_dir} 作为 tokenizer 目录用于训练")
        return 0

    # 下载
    result = download_tokenizer(
        repo_id=args.repo_id,
        output_dir=output_dir,
        cache_root=cache_root,
    )

    if not result:
        _log("tokenizer 下载失败，请检查网络或手动下载", "ERROR")
        _log(f"手动方式：从 https://huggingface.co/Qwen/Qwen3-32B 下载 tokenizer.json 等文件到 {output_dir}", "ERROR")
        return 1

    # 验证
    if not args.skip_verify:
        ok = verify_tokenizer(output_dir)
        if not ok:
            _log("tokenizer 验证未通过，但文件已下载", "WARN")

    _log(f"\n完成：tokenizer 文件位于 {output_dir}")
    _log(f"下一步：在 config_verse_nex.yml 中确认 tokenizer.kind=qwen 并指向此目录")
    return 0


if __name__ == "__main__":
    sys.exit(main())
