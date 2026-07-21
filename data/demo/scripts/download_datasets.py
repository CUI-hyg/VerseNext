#!/usr/bin/env python3
"""下载公开数据集脚本（CometSpark-V0.2 训练数据准备）。

设计目标
--------
- 用户在脚本顶部 ``ENABLED_SOURCES`` 字典中开关数据源
- 每个数据源独立子目录 ``raw/<source_name>/``
- 支持 3 种下载方式（按优先级回退）：
    1. ``huggingface_hub.hf_hub_download`` —— 推荐
    2. ``datasets.load_dataset`` —— 兜底
    3. ``urllib.request`` 直接下载 —— 最后兜底
- 下载失败的源跳过，不影响其他源

可使用的数据源（均为 HuggingFace 公开数据集）：
    - ``BelleGroup/train_3.5M_CN``     —— 中文 SFT 对话
    - ``YeungNLP/firefly-train-1.1M``  —— 中文 SFT 多任务
    - ``sahil2801/code-alpaca``        —— 代码指令
    - ``cmrc2018``                     —— 中文阅读理解（SFT 改写）
    - ``wikiann`` (zh)                 —— 中文 NER（可作 LM 训练）
    - ``tatsu-lab/alpaca``             —— 英文 SFT（默认关闭）

使用方式
--------
    cd /workspace/data/demo/scripts
    python download_datasets.py                      # 下载 ENABLED_SOURCES 中所有 enabled=True 的源
    python download_datasets.py --raw-dir ./raw      # 自定义 raw 目录
    python download_datasets.py --only belle_chat    # 仅下载指定源
    python download_datasets.py --list               # 仅列出可用源

依赖（可选，未安装时自动降级）：
    pip install huggingface_hub datasets
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.request
import urllib.error
from typing import Optional


# ---------------------------------------------------------------------------
# 数据源配置：用户在此处开关
# ---------------------------------------------------------------------------
# 字段说明：
#   enabled:    是否启用
#   repo_id:    HuggingFace 数据集 ID（hf_hub_download / load_dataset 用）
#   url:        直接下载 URL（urllib 兜底用）
#   format:     输出格式（jsonl / parquet / json / txt）
#   lang:       语言（zh / en / code）
#   task_type:  任务类型（sft / lm / qa / ner / code）
#   subset:     load_dataset 的 subset 参数（可选）
#   split:      load_dataset 的 split 参数（默认 train）
#   filename:   hf_hub_download 指定文件名（可选，未指定时尝试常见文件名）

ENABLED_SOURCES: dict[str, dict] = {
    # ------------------------------------------------------------------
    # 中文通用语料
    # ------------------------------------------------------------------
    "wiki_zh": {
        "enabled": True,
        "repo_id": "wikiann",
        "subset": "zh",
        "url": "https://huggingface.co/datasets/wikiann/resolve/main/data/zh/train.parquet",
        "format": "parquet",
        "lang": "zh",
        "task_type": "ner",
        "split": "train",
        "description": "wikiann 中文部分（NER 数据，可作 LM 训练）",
    },

    # ------------------------------------------------------------------
    # 中文 SFT 对话
    # ------------------------------------------------------------------
    "belle_chat": {
        "enabled": True,
        "repo_id": "BelleGroup/train_3.5M_CN",
        "url": "https://huggingface.co/datasets/BelleGroup/train_3.5M_CN/resolve/main/Belle_train_3.5M_CN.jsonl",
        "format": "jsonl",
        "lang": "zh",
        "task_type": "sft",
        "split": "train",
        "description": "BelleGroup/train_3.5M_CN（中文 SFT 大规模对话）",
    },
    "firefly_zh": {
        "enabled": True,
        "repo_id": "YeungNLP/firefly-train-1.1M",
        "url": "https://huggingface.co/datasets/YeungNLP/firefly-train-1.1M/resolve/main/firefly-train-1.1M.jsonl",
        "format": "jsonl",
        "lang": "zh",
        "task_type": "sft",
        "split": "train",
        "description": "YeungNLP/firefly-train-1.1M（中文 SFT 多任务指令）",
    },

    # ------------------------------------------------------------------
    # 代码
    # ------------------------------------------------------------------
    "code_alpaca": {
        "enabled": True,
        "repo_id": "sahil2801/code-alpaca",
        "url": "https://huggingface.co/datasets/sahil2801/code-alpaca/resolve/main/data/train-00000-of-00001-a09b74b3ef9c3b56.parquet",
        "format": "parquet",
        "lang": "code",
        "task_type": "code",
        "split": "train",
        "description": "sahil2801/code-alpaca（代码指令微调）",
    },

    # ------------------------------------------------------------------
    # 数学
    # ------------------------------------------------------------------
    "math_qa_zh": {
        "enabled": True,
        "repo_id": "BelleGroup/train_2M_CN",
        "url": "https://huggingface.co/datasets/BelleGroup/train_2M_CN/resolve/main/Belle_train_2M_CN.jsonl",
        "format": "jsonl",
        "lang": "zh",
        "task_type": "sft",
        "split": "train",
        "description": "BelleGroup/train_2M_CN（含数学问答的中文 SFT，作 math_qa_zh 备用源）",
    },

    # ------------------------------------------------------------------
    # 中文阅读理解（SFT 改写）
    # ------------------------------------------------------------------
    "cmrc2018": {
        "enabled": True,
        "repo_id": "cmrc2018",
        "url": "https://huggingface.co/datasets/cmrc2018/resolve/main/cmrc2018/train.json",
        "format": "json",
        "lang": "zh",
        "task_type": "qa",
        "split": "train",
        "description": "cmrc2018（中文阅读理解，可作 SFT 改写）",
    },

    # ------------------------------------------------------------------
    # 英文通用（默认关闭）
    # ------------------------------------------------------------------
    "alpaca_en": {
        "enabled": False,
        "repo_id": "tatsu-lab/alpaca",
        "url": "https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/main/alpaca_data.json",
        "format": "json",
        "lang": "en",
        "task_type": "sft",
        "split": "train",
        "description": "tatsu-lab/alpaca（英文 SFT 指令数据）",
    },
}


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------


def _log(msg: str, level: str = "INFO") -> None:
    """统一日志格式。"""
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] [{level}] {msg}", flush=True)


def _ensure_dir(path: str) -> None:
    """确保目录存在。"""
    os.makedirs(path, exist_ok=True)


def _write_jsonl(records: list, path: str) -> int:
    """把 records 列表写入 jsonl 文件，返回写入条数。"""
    _ensure_dir(os.path.dirname(os.path.abspath(path)) or ".")
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            try:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n += 1
            except (TypeError, ValueError):
                continue
    return n


def _is_nonempty_file(path: str) -> bool:
    """判断文件存在且非空。"""
    return os.path.isfile(path) and os.path.getsize(path) > 0


# ---------------------------------------------------------------------------
# 方法 1: huggingface_hub.hf_hub_download（推荐）
# ---------------------------------------------------------------------------


def _try_hf_hub(name: str, cfg: dict, target_dir: str) -> Optional[str]:
    """用 huggingface_hub 下载，返回 jsonl 文件路径；失败返回 None。"""
    try:
        from huggingface_hub import hf_hub_download, list_repo_files
    except ImportError:
        _log(f"[{name}] huggingface_hub 未安装，跳过此方法", "WARN")
        return None
    except Exception as e:  # noqa: BLE001
        _log(f"[{name}] huggingface_hub 导入失败：{e}", "WARN")
        return None

    repo_id = cfg["repo_id"]
    cfg_format = cfg.get("format", "jsonl")
    _log(f"[{name}] 尝试 hf_hub_download: repo_id={repo_id}")

    try:
        # 列出仓库文件，挑出可能的训练数据文件
        files = list_repo_files(repo_id, repo_type="dataset")
    except Exception as e:  # noqa: BLE001
        _log(f"[{name}] list_repo_files 失败：{e}，尝试用 filename 直接下载", "WARN")
        files = []

    # 候选文件优先级：jsonl > parquet > json > txt
    candidates: list[str] = []
    if files:
        for f in files:
            fl = f.lower()
            if "test" in fl or "val" in fl or "dev" in fl:
                continue  # 跳过测试/验证集
            if fl.endswith(".jsonl"):
                candidates.append((0, f))
            elif fl.endswith(".parquet"):
                candidates.append((1, f))
            elif fl.endswith(".json") and "train" in fl:
                candidates.append((2, f))
        candidates.sort()
        candidates = [f for _, f in candidates]

    # 如果 cfg 提供 filename，优先使用
    if "filename" in cfg and cfg["filename"]:
        candidates.insert(0, cfg["filename"])

    if not candidates and cfg_format == "jsonl":
        # 仓库里没有显式 jsonl 文件，但 URL 提供了
        return None

    # 尝试下载候选文件
    for fname in candidates:
        try:
            local_path = hf_hub_download(
                repo_id=repo_id,
                filename=fname,
                repo_type="dataset",
                local_dir=target_dir,
            )
            if _is_nonempty_file(local_path):
                _log(f"[{name}] hf_hub 下载成功：{fname} -> {local_path}")
                return _normalize_to_jsonl(local_path, target_dir, name)
        except Exception as e:  # noqa: BLE001
            _log(f"[{name}] hf_hub_download {fname} 失败：{e}", "WARN")
            continue

    return None


# ---------------------------------------------------------------------------
# 方法 2: datasets.load_dataset（兜底）
# ---------------------------------------------------------------------------


def _try_datasets(name: str, cfg: dict, target_dir: str) -> Optional[str]:
    """用 datasets.load_dataset 下载并转为 jsonl。"""
    try:
        from datasets import load_dataset
    except ImportError:
        _log(f"[{name}] datasets 库未安装，跳过此方法", "WARN")
        return None
    except Exception as e:  # noqa: BLE001
        _log(f"[{name}] datasets 导入失败：{e}", "WARN")
        return None

    repo_id = cfg["repo_id"]
    subset = cfg.get("subset")
    split = cfg.get("split", "train")
    _log(f"[{name}] 尝试 load_dataset: repo_id={repo_id} subset={subset} split={split}")

    try:
        if subset:
            ds = load_dataset(repo_id, subset, split=split)
        else:
            ds = load_dataset(repo_id, split=split)
    except Exception as e:  # noqa: BLE001
        # 重试：不指定 split（让 datasets 返回 DatasetDict）
        _log(f"[{name}] load_dataset split={split} 失败：{e}，尝试不指定 split", "WARN")
        try:
            if subset:
                ds_dict = load_dataset(repo_id, subset)
            else:
                ds_dict = load_dataset(repo_id)
            # 优先 train，否则取第一个
            if "train" in ds_dict:
                ds = ds_dict["train"]
            else:
                ds = next(iter(ds_dict.values()))
        except Exception as e2:  # noqa: BLE001
            _log(f"[{name}] load_dataset 最终失败：{e2}", "ERROR")
            return None

    # 转为 jsonl
    out_path = os.path.join(target_dir, "data.jsonl")
    n = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for item in ds:
            try:
                # datasets.Dataset 的 item 是 dict-like
                record = {k: _coerce_jsonable(v) for k, v in dict(item).items()}
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                n += 1
            except Exception:  # noqa: BLE001
                continue
    if n == 0:
        _log(f"[{name}] load_dataset 转换为 jsonl 后 0 条记录", "WARN")
        return None
    _log(f"[{name}] load_dataset 成功：{n} 条 -> {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# 方法 3: urllib 直接下载（最后兜底，仅适用于直接 URL）
# ---------------------------------------------------------------------------


def _try_urllib(name: str, cfg: dict, target_dir: str) -> Optional[str]:
    """用 urllib.request 直接下载 URL（仅适用于直接 URL）。"""
    url = cfg.get("url")
    if not url:
        _log(f"[{name}] 未配置 url，urllib 兜底失败", "WARN")
        return None

    cfg_format = cfg.get("format", "jsonl")
    # 根据 URL 后缀决定本地文件名
    if url.endswith(".jsonl"):
        local_name = "data.jsonl"
    elif url.endswith(".parquet"):
        local_name = "data.parquet"
    elif url.endswith(".json"):
        local_name = "data.json"
    elif url.endswith(".txt"):
        local_name = "data.txt"
    else:
        local_name = f"data.{cfg_format}"

    local_path = os.path.join(target_dir, local_name)
    _log(f"[{name}] 尝试 urllib 下载：{url} -> {local_path}")

    # 设置 User-Agent 避免某些服务器拒绝
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 (compatible; CometSparkDataScript/1.0)"},
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = resp.read()
    except urllib.error.HTTPError as e:
        _log(f"[{name}] urllib HTTP 错误：{e.code} {e.reason}", "ERROR")
        return None
    except urllib.error.URLError as e:
        _log(f"[{name}] urllib URL 错误：{e.reason}", "ERROR")
        return None
    except Exception as e:  # noqa: BLE001
        _log(f"[{name}] urllib 未知错误：{e}", "ERROR")
        return None

    if not data:
        _log(f"[{name}] urllib 下载内容为空", "WARN")
        return None

    _ensure_dir(target_dir)
    with open(local_path, "wb") as f:
        f.write(data)
    _log(f"[{name}] urllib 下载成功：{len(data)} bytes -> {local_path}")

    return _normalize_to_jsonl(local_path, target_dir, name)


# ---------------------------------------------------------------------------
# 文件格式归一化：把 parquet / json 转为 jsonl
# ---------------------------------------------------------------------------


def _normalize_to_jsonl(local_path: str, target_dir: str, name: str) -> Optional[str]:
    """把下载的文件归一化为 jsonl 格式。

    - .jsonl 文件：直接返回
    - .parquet 文件：用 pyarrow / pandas 转 jsonl
    - .json 文件：解析后逐条写出
    - .txt 文件：每行作为 {"text": line} 写出
    """
    if not _is_nonempty_file(local_path):
        return None

    lower = local_path.lower()
    if lower.endswith(".jsonl"):
        return local_path

    out_path = os.path.join(target_dir, "data.jsonl")

    # parquet 处理
    if lower.endswith(".parquet"):
        try:
            import pyarrow.parquet as pq

            table = pq.read_table(local_path)
            records = table.to_pylist()
            n = _write_jsonl(records, out_path)
            _log(f"[{name}] parquet -> jsonl: {n} 条 -> {out_path}")
            return out_path if n > 0 else None
        except ImportError:
            _log(f"[{name}] pyarrow 未安装，尝试 pandas 读取 parquet", "WARN")
        except Exception as e:  # noqa: BLE001
            _log(f"[{name}] pyarrow 读取 parquet 失败：{e}", "WARN")

        try:
            import pandas as pd

            df = pd.read_parquet(local_path)
            records = df.to_dict(orient="records")
            n = _write_jsonl(records, out_path)
            _log(f"[{name}] pandas parquet -> jsonl: {n} 条 -> {out_path}")
            return out_path if n > 0 else None
        except ImportError:
            _log(f"[{name}] pandas 也未安装，无法转换 parquet", "ERROR")
            return None
        except Exception as e:  # noqa: BLE001
            _log(f"[{name}] pandas 读取 parquet 失败：{e}", "ERROR")
            return None

    # json 处理（可能为 list 或 dict）
    if lower.endswith(".json"):
        try:
            with open(local_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:  # noqa: BLE001
            _log(f"[{name}] JSON 解析失败：{e}", "ERROR")
            return None

        records: list = []
        if isinstance(data, list):
            records = data
        elif isinstance(data, dict):
            # 常见结构：{"data": [...]} / {"train": [...]} / {"examples": [...]}
            for key in ("data", "train", "examples", "items", "rows"):
                if key in data and isinstance(data[key], list):
                    records = data[key]
                    break
            if not records:
                # 直接把整个 dict 作为单条记录
                records = [data]
        n = _write_jsonl(records, out_path)
        _log(f"[{name}] json -> jsonl: {n} 条 -> {out_path}")
        return out_path if n > 0 else None

    # txt 处理：每行作为 {"text": line}
    if lower.endswith(".txt"):
        records: list = []
        try:
            with open(local_path, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    s = line.strip()
                    if s:
                        records.append({"text": s})
        except Exception as e:  # noqa: BLE001
            _log(f"[{name}] TXT 读取失败：{e}", "ERROR")
            return None
        n = _write_jsonl(records, out_path)
        _log(f"[{name}] txt -> jsonl: {n} 条 -> {out_path}")
        return out_path if n > 0 else None

    # 未知格式：返回原文件路径，由 process_datasets 兜底处理
    _log(f"[{name}] 未知文件格式：{local_path}，直接返回路径", "WARN")
    return local_path


def _coerce_jsonable(v) -> object:
    """把 datasets 的各种字段类型（Value / Sequence / ClassLabel）转为 JSON 可序列化。"""
    # bytes -> str
    if isinstance(v, bytes):
        try:
            return v.decode("utf-8")
        except UnicodeDecodeError:
            return v.decode("utf-8", errors="ignore")
    # list / tuple：递归
    if isinstance(v, (list, tuple)):
        return [_coerce_jsonable(x) for x in v]
    # dict：递归
    if isinstance(v, dict):
        return {k: _coerce_jsonable(x) for k, x in v.items()}
    return v


# ---------------------------------------------------------------------------
# 主入口：download_source
# ---------------------------------------------------------------------------


def download_source(name: str, cfg: dict, raw_dir: str) -> Optional[str]:
    """下载单个数据源，返回本地 jsonl 文件路径，失败返回 None。

    Args:
        name: 数据源名称（用于日志与目录名）
        cfg:  数据源配置 dict（见 ENABLED_SOURCES）
        raw_dir: raw 根目录，下载到 ``raw_dir/<name>/``

    Returns:
        本地 jsonl 文件路径；失败返回 None
    """
    target_dir = os.path.join(raw_dir, name)
    _ensure_dir(target_dir)

    # 若已存在 data.jsonl 且非空，直接复用
    existing = os.path.join(target_dir, "data.jsonl")
    if _is_nonempty_file(existing):
        _log(f"[{name}] 已存在 {existing}，跳过下载（删除后可重下）")
        return existing

    _log(f"=== 开始下载数据源 [{name}] ===")
    _log(f"[{name}] 描述：{cfg.get('description', '(无)')}")
    _log(f"[{name}] 语言={cfg.get('lang')} 任务={cfg.get('task_type')} 格式={cfg.get('format')}")

    # 方法 1: huggingface_hub（推荐）
    try:
        result = _try_hf_hub(name, cfg, target_dir)
        if result:
            _log(f"=== [{name}] 下载完成（hf_hub）===\n")
            return result
    except Exception as e:  # noqa: BLE001
        _log(f"[{name}] hf_hub 异常：{e}", "ERROR")

    # 方法 2: datasets 库（兜底）
    try:
        result = _try_datasets(name, cfg, target_dir)
        if result:
            _log(f"=== [{name}] 下载完成（datasets）===\n")
            return result
    except Exception as e:  # noqa: BLE001
        _log(f"[{name}] datasets 异常：{e}", "ERROR")

    # 方法 3: urllib 直接下载（最后兜底）
    try:
        result = _try_urllib(name, cfg, target_dir)
        if result:
            _log(f"=== [{name}] 下载完成（urllib）===\n")
            return result
    except Exception as e:  # noqa: BLE001
        _log(f"[{name}] urllib 异常：{e}", "ERROR")

    _log(f"=== [{name}] 所有下载方式均失败，跳过 ===\n", "ERROR")
    return None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _list_sources() -> None:
    """列出所有可用数据源。"""
    print(f"{'名称':<15} {'启用':<6} {'语言':<6} {'任务':<6} {'repo_id':<40} 描述")
    print("-" * 120)
    for name, cfg in ENABLED_SOURCES.items():
        enabled = "✓" if cfg.get("enabled") else "✗"
        lang = cfg.get("lang", "?")
        task = cfg.get("task_type", "?")
        repo = cfg.get("repo_id", "?")
        desc = cfg.get("description", "")[:50]
        print(f"{name:<15} {enabled:<6} {lang:<6} {task:<6} {repo:<40} {desc}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="CometSpark-V0.2 数据下载脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--raw-dir",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "raw"),
        help="raw 目录（默认：./raw）",
    )
    parser.add_argument(
        "--only",
        nargs="+",
        default=None,
        help="仅下载指定的源（空格分隔，如 --only belle_chat firefly_zh）",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="仅列出可用数据源，不下载",
    )
    parser.add_argument(
        "--include-disabled",
        action="store_true",
        help="同时下载 enabled=False 的源（需配合 --only 使用）",
    )
    args = parser.parse_args()

    if args.list:
        _list_sources()
        return 0

    raw_dir = os.path.abspath(args.raw_dir)
    _ensure_dir(raw_dir)
    _log(f"raw 目录：{raw_dir}")

    # 选择要下载的源
    if args.only:
        names = args.only
        # 校验
        invalid = [n for n in names if n not in ENABLED_SOURCES]
        if invalid:
            _log(f"未知的数据源名称：{invalid}", "ERROR")
            _log(f"可用源：{list(ENABLED_SOURCES.keys())}", "ERROR")
            return 1
        sources = [(n, ENABLED_SOURCES[n]) for n in names]
    else:
        sources = [
            (n, cfg)
            for n, cfg in ENABLED_SOURCES.items()
            if cfg.get("enabled") or args.include_disabled
        ]

    if not sources:
        _log("没有需要下载的源（请检查 ENABLED_SOURCES 或 --only 参数）", "WARN")
        return 0

    _log(f"计划下载 {len(sources)} 个数据源：{[n for n, _ in sources]}\n")

    success: list[str] = []
    failed: list[str] = []
    skipped: list[str] = []

    for name, cfg in sources:
        try:
            result = download_source(name, cfg, raw_dir)
            if result:
                success.append(name)
            else:
                failed.append(name)
        except KeyboardInterrupt:
            _log(f"用户中断下载 [{name}]", "WARN")
            skipped.append(name)
            break
        except Exception as e:  # noqa: BLE001
            _log(f"[{name}] 下载异常：{e}", "ERROR")
            failed.append(name)

    # 汇总
    print("\n" + "=" * 60)
    print("下载汇总")
    print("=" * 60)
    print(f"成功：{len(success)} 个 -> {success}")
    print(f"失败：{len(failed)} 个 -> {failed}")
    if skipped:
        print(f"跳过：{len(skipped)} 个 -> {skipped}")
    print(f"\n下一步：运行 process_datasets.py 处理 raw/ 下的 jsonl 文件")
    return 0 if not failed else 2


if __name__ == "__main__":
    sys.exit(main())
