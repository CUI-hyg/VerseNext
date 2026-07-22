#!/usr/bin/env python3
"""CometSpark 命令行快捷入口

基于 VerseTrainer API，提供简单易用的训练/评估/生成/压缩命令。
所有命令都有合理的默认值，最小化用户配置。

用法示例：
    # 快速训练（小配置，10秒完成）
    python spark/run.py train --small

    # 正式训练（1B 配置）
    python spark/run.py train

    # 训练后自动评估
    python spark/run.py train --eval-after

    # 生成文本
    python spark/run.py generate --prompt "你好世界"

    # 交互式聊天
    python spark/run.py chat
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Optional, List

# ---------------------------------------------------------------------------
# 路径自举：确保 verse_torch / verse_nex / verse_infra 可被 import
# ---------------------------------------------------------------------------

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)
for _dep in ("verse_torch", "verse_nex", "verse_infra"):
    _dep_path = os.path.join(_REPO_ROOT, "packages", _dep)
    if os.path.isdir(_dep_path) and _dep_path not in sys.path:
        sys.path.insert(0, _dep_path)
# 把 repo root 加入 path 以便 import spark
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

_DEFAULT_CONFIG = "spark/config/cometspark_v05.yml"
_SMALL_CONFIG = "spark/config/cometspark_v05_small.yml"

# 颜色码（ANSI）
_COLOR_RESET = "\033[0m"
_COLOR_BOLD = "\033[1m"
_COLOR_GREEN = "\033[32m"
_COLOR_YELLOW = "\033[33m"
_COLOR_CYAN = "\033[36m"
_COLOR_RED = "\033[31m"
_COLOR_DIM = "\033[2m"


def _supports_color() -> bool:
    """检测终端是否支持彩色输出。"""
    return (
        hasattr(sys.stdout, "isatty")
        and sys.stdout.isatty()
        and os.environ.get("NO_COLOR") is None
    )


def _c(text: str, color: str) -> str:
    """给文本加颜色（不支持颜色时原样返回）。"""
    if not _supports_color():
        return text
    return f"{color}{text}{_COLOR_RESET}"


def _info(msg: str) -> None:
    """打印信息行。"""
    print(_c(f"[spark] {msg}", _COLOR_CYAN), flush=True)


def _ok(msg: str) -> None:
    """打印成功行。"""
    print(_c(f"[✓] {msg}", _COLOR_GREEN), flush=True)


def _warn(msg: str) -> None:
    """打印警告行。"""
    print(_c(f"[!] {msg}", _COLOR_YELLOW), flush=True)


def _error(msg: str) -> None:
    """打印错误行。"""
    print(_c(f"[✗] {msg}", _COLOR_RED), file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------


def _setup_paths() -> None:
    """路径自举（幂等，重复调用安全）。

    确保 verse_torch / verse_nex / verse_infra 在 sys.path 中。
    模块顶部的路径自举已执行过，这里做二次确认。
    """
    for _dep in ("verse_torch", "verse_nex", "verse_infra"):
        _dep_path = os.path.join(_REPO_ROOT, "packages", _dep)
        if os.path.isdir(_dep_path) and _dep_path not in sys.path:
            sys.path.insert(0, _dep_path)


def _resolve_config_path(config_arg: Optional[str], small: bool = False) -> str:
    """解析配置文件路径。

    优先级：
    1. --config 显式指定
    2. --small 标志 → 小配置
    3. 默认 1B 配置

    返回绝对路径。若文件不存在抛出 FileNotFoundError。
    """
    if config_arg:
        path = config_arg
    elif small:
        path = _SMALL_CONFIG
    else:
        path = _DEFAULT_CONFIG

    # 相对路径以 repo root 为基准
    if not os.path.isabs(path):
        path = os.path.join(_REPO_ROOT, path)

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"配置文件不存在：{path}\n"
            f"提示：使用 --config 指定路径，或 --small 使用小配置"
        )
    return path


def _load_yaml_config(path: str) -> dict:
    """加载 YAML 配置文件为 dict。

    优先用 PyYAML，不可用时用 verse_trainer 的极简解析器。
    """
    try:
        import yaml  # type: ignore
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except ImportError:
        from verse_infra.verse_trainer.trainer import _load_full_config
        return _load_full_config(path)


def _load_tokenizer_for_config(config_path: Optional[str], model_config=None):
    """根据配置加载 tokenizer。

    Args:
        config_path: 配置文件路径（可 None）。
        model_config: 模型配置对象（CometSparkV05Config），用于推断 tokenizer 类型。

    Returns:
        tokenizer 实例。

    策略：
    1. 若有 config_path，读取 tokenizer 段；
    2. 若 kind == "byte"，用 ByteTokenizer；
    3. 若有 from_hf，用 BPETokenizer.from_pretrained（网络不可用时 graceful skip）；
    4. 若无 config，根据 model_config.vocab_size 推断。
    """
    from verse_infra.verse_tokenizer import (
        BPETokenizer, ByteTokenizer, load_tokenizer,
    )

    tok_cfg = {}
    if config_path and os.path.exists(config_path):
        full_cfg = _load_yaml_config(config_path)
        tok_cfg = full_cfg.get("tokenizer", {})

    kind = tok_cfg.get("kind", "")
    from_hf = tok_cfg.get("from_hf")

    # 优先从 config 的 tokenizer 段判断
    if kind == "byte":
        return ByteTokenizer()

    if from_hf:
        try:
            return BPETokenizer.from_pretrained(from_hf)
        except Exception as e:
            _warn(f"无法从 HuggingFace 加载 tokenizer ({from_hf})：{e}")
            _warn("降级使用 ByteTokenizer")
            return ByteTokenizer()

    # 无 config 或无法判断：根据 vocab_size 推断
    vocab_size = getattr(model_config, "vocab_size", 256) if model_config else 256
    if vocab_size <= 260:
        return ByteTokenizer()

    tokenizer_repo = getattr(model_config, "tokenizer_repo", None)
    if tokenizer_repo:
        try:
            return BPETokenizer.from_pretrained(tokenizer_repo)
        except Exception as e:
            _warn(f"无法加载 tokenizer ({tokenizer_repo})：{e}")
            _warn("降级使用 ByteTokenizer")
            return ByteTokenizer()

    return ByteTokenizer()


def _load_model_and_tokenizer(
    checkpoint: str,
    config_path: Optional[str] = None,
):
    """加载模型和 tokenizer。

    Args:
        checkpoint: checkpoint 文件路径（.pt 或目录）。
        config_path: 配置文件路径（可选，用于加载 tokenizer）。

    Returns:
        (model, tokenizer) 元组。
    """
    if not os.path.exists(checkpoint):
        raise FileNotFoundError(
            f"checkpoint 不存在：{checkpoint}\n"
            f"提示：先运行 `python spark/run.py train` 训练模型"
        )

    from spark.model.model import CometSparkV05LM

    model = CometSparkV05LM.from_pretrained(checkpoint)

    # 加载 tokenizer：优先用 config_path，否则从模型 config 推断
    resolved_config = None
    if config_path:
        resolved_config = config_path if os.path.isabs(config_path) else os.path.join(_REPO_ROOT, config_path)

    tokenizer = _load_tokenizer_for_config(resolved_config, model.config)
    return model, tokenizer


def _print_model_info(model) -> None:
    """打印模型信息（参数量、配置摘要）。"""
    params = model.count_parameters()
    cfg = model.config
    _info(f"模型：CometSpark-V0.5 (arch={cfg.arch})")
    _info(
        f"参数量：{params:,} "
        f"(layers={cfg.n_layer}, dim={cfg.n_embd}, "
        f"heads={cfg.n_head}, kv_heads={cfg.n_kv_head}, "
        f"vocab={cfg.vocab_size})"
    )
    _info(f"设备：{model.device_info()}")


def _print_dry_run(action: str, **kwargs) -> None:
    """打印 dry-run 信息（只打印不执行）。"""
    _warn(f"[dry-run] 将执行：{action}")
    for k, v in kwargs.items():
        print(f"  {k} = {v}", flush=True)


def _generated_to_ids(generated) -> list:
    """把 generate 的输出（Tensor 或 ndarray）转为 id 列表。

    verse_torch.Tensor 的 ``.data`` 是 ndarray，可以直接 reshape；
    numpy ndarray 的 ``.data`` 是 memoryview（不能 reshape），所以
    需要先判断类型再取值。
    """
    import numpy as np
    if isinstance(generated, np.ndarray):
        return generated.reshape(-1).tolist()
    # verse_torch.Tensor：有 .data 属性且是 ndarray
    data = getattr(generated, "data", None)
    if isinstance(data, np.ndarray):
        return data.reshape(-1).tolist()
    # 兜底
    return np.asarray(generated).reshape(-1).tolist()


# ---------------------------------------------------------------------------
# 子命令：train
# ---------------------------------------------------------------------------


def cmd_train(args) -> int:
    """训练模型。

    调用 ``verse_infra.verse_trainer.train()``，训练后可选自动评估。
    """
    config_path = _resolve_config_path(args.config, small=args.small)

    # 打印模型信息
    if not args.quiet:
        _info(f"配置文件：{config_path}")
        if args.small:
            _info("模式：小配置（快速调试）")
        else:
            _info("模式：1B 配置（正式训练）")

    # dry-run：只打印不执行
    if args.dry_run:
        _print_dry_run(
            "train",
            config=config_path,
            max_steps=args.max_steps,
            batch_size=args.batch_size,
            device=args.device or "auto",
            resume=args.resume,
            amp=args.amp,
            eval_after=args.eval_after,
        )
        return 0

    # 实际训练
    import verse_infra.verse_trainer as _vt

    result = _vt.train(
        config_path=config_path,
        base_dir=os.path.dirname(config_path) or _REPO_ROOT,
        device=args.device,
        max_steps_override=args.max_steps,
        resume=args.resume,
        amp=args.amp,
        quiet=args.quiet,
        verbose=args.verbose,
    )

    best_val_loss = result.get("best_val_loss", float("inf"))
    _ok(f"训练完成！best_val_loss = {best_val_loss:.4f}")

    # 打印训练总结
    if not args.quiet:
        _info(f"checkpoint 保存目录：{result.get('save_dir', 'checkpoints')}")
        _info(f"训练步数：{result.get('total_steps', '?')}")

    # 训练后自动评估（默认开启）
    if args.eval_after:
        _info("开始自动评估...")
        try:
            eval_result = _vt.evaluate(
                config_path=config_path,
                base_dir=os.path.dirname(config_path) or _REPO_ROOT,
                checkpoint=result.get("best_checkpoint"),
            )
            n_samples = len(eval_result.get("results", []))
            _ok(f"评估完成，生成 {n_samples} 条样本")
        except Exception as e:
            _warn(f"自动评估失败（不影响训练结果）：{e}")

    return 0


# ---------------------------------------------------------------------------
# 子命令：eval
# ---------------------------------------------------------------------------


def cmd_eval(args) -> int:
    """评估模型。

    调用 ``verse_infra.verse_trainer.evaluate()``。
    """
    config_path = _resolve_config_path(args.config)

    if args.checkpoint and not os.path.exists(args.checkpoint):
        raise FileNotFoundError(
            f"checkpoint 不存在：{args.checkpoint}"
        )

    if args.dry_run:
        _print_dry_run(
            "eval",
            config=config_path,
            checkpoint=args.checkpoint or "auto",
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            score=args.score,
        )
        return 0

    import verse_infra.verse_trainer as _vt

    result = _vt.evaluate(
        config_path=config_path,
        base_dir=os.path.dirname(config_path) or _REPO_ROOT,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        score=args.score,
        references_file=args.references_file,
        checkpoint=args.checkpoint,
    )

    n_samples = len(result.get("results", []))
    _ok(f"评估完成，生成 {n_samples} 条样本")

    if args.score and "scores" in result:
        scores = result["scores"]
        _info(f"打分结果：{scores}")

    return 0


# ---------------------------------------------------------------------------
# 子命令：generate
# ---------------------------------------------------------------------------


def cmd_generate(args) -> int:
    """生成文本。

    加载模型后调用 ``model.generate()``。
    """
    if args.dry_run:
        _print_dry_run(
            "generate",
            checkpoint=args.checkpoint,
            prompt=args.prompt,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
        )
        return 0

    model, tokenizer = _load_model_and_tokenizer(
        args.checkpoint, config_path=args.config
    )

    if not args.quiet:
        _print_model_info(model)

    model.eval()

    # encode prompt
    prompt = args.prompt or ""
    try:
        prompt_ids = list(tokenizer.encode(prompt, add_special_tokens=False))
    except TypeError:
        prompt_ids = list(tokenizer.encode(prompt))

    if not prompt_ids:
        prompt_ids = [0]

    import numpy as np
    idx = np.asarray(prompt_ids, dtype=np.int64).reshape(1, -1)

    # 推断 eos_id
    eos_id = getattr(tokenizer, "eos_id", None)
    if eos_id is None:
        vocab = getattr(tokenizer, "vocab", None)
        if isinstance(vocab, dict):
            for _eos_str in ("<|im_end|>", "<|eos|>", "<eos>", ""):
                if _eos_str in vocab:
                    eos_id = int(vocab[_eos_str])
                    break

    generated = model.generate(
        idx,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        eos_id=eos_id,
    )

    # decode
    gen_ids = _generated_to_ids(generated)

    try:
        text = tokenizer.decode(list(gen_ids), strip_special=True)
    except TypeError:
        text = tokenizer.decode(list(gen_ids))

    print(text, flush=True)
    return 0


# ---------------------------------------------------------------------------
# 子命令：chat
# ---------------------------------------------------------------------------


def _chat_render_messages(messages, tokenizer):
    """用 ChatML 模板渲染消息列表为 token id 列表。

    优先用 tokenizer.apply_chat_template，降级用手动 ChatML 渲染。
    """
    try:
        rendered = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True
        )
        if isinstance(rendered, str):
            try:
                ids = list(tokenizer.encode(rendered, add_special_tokens=False))
            except TypeError:
                ids = list(tokenizer.encode(rendered))
        else:
            ids = list(rendered)
        return ids
    except Exception:
        pass

    # 降级：手动 ChatML 渲染
    text_parts = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        text_parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")
    text_parts.append("<|im_start|>assistant\n")
    full_text = "".join(text_parts)
    try:
        ids = list(tokenizer.encode(full_text, add_special_tokens=False))
    except TypeError:
        ids = list(tokenizer.encode(full_text))
    return ids


def cmd_chat(args) -> int:
    """交互式聊天。

    加载模型后进入交互循环，支持 /quit /clear /save 命令。
    """
    if args.dry_run:
        _print_dry_run(
            "chat",
            checkpoint=args.checkpoint,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )
        return 0

    model, tokenizer = _load_model_and_tokenizer(
        args.checkpoint, config_path=args.config
    )

    _print_model_info(model)
    model.eval()

    # 推断 eos_id
    eos_id = getattr(tokenizer, "eos_id", None)
    if eos_id is None:
        vocab = getattr(tokenizer, "vocab", None)
        if isinstance(vocab, dict):
            for _eos_str in ("<|im_end|>", "<|eos|>", "<eos>", ""):
                if _eos_str in vocab:
                    eos_id = int(vocab[_eos_str])
                    break

    import numpy as np

    messages: list = []
    print(_c("CometSpark 聊天模式", _COLOR_BOLD))
    print(_c("输入 /quit 退出 | /clear 清空历史 | /save <path> 保存对话",
              _COLOR_DIM))
    print()

    while True:
        try:
            user_input = input(_c("你: ", _COLOR_GREEN)).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            _info("再见！")
            break

        if not user_input:
            continue

        # 命令处理
        if user_input == "/quit":
            _info("再见！")
            break

        if user_input == "/clear":
            messages = []
            _ok("对话历史已清空")
            continue

        if user_input.startswith("/save"):
            parts = user_input.split(maxsplit=1)
            save_path = parts[1] if len(parts) > 1 else "chat_history.json"
            import json
            try:
                with open(save_path, "w", encoding="utf-8") as f:
                    json.dump(messages, f, ensure_ascii=False, indent=2)
                _ok(f"对话已保存到 {save_path}")
            except Exception as e:
                _error(f"保存失败：{e}")
            continue

        # 正常对话
        messages.append({"role": "user", "content": user_input})

        # 渲染 + 生成
        prompt_ids = _chat_render_messages(messages, tokenizer)
        if not prompt_ids:
            prompt_ids = [0]

        idx = np.asarray(prompt_ids, dtype=np.int64).reshape(1, -1)

        try:
            generated = model.generate(
                idx,
                max_new_tokens=args.max_tokens,
                temperature=args.temperature,
                eos_id=eos_id,
            )
        except Exception as e:
            _error(f"生成失败：{e}")
            messages.pop()  # 移除刚加的 user 消息
            continue

        # 取 prompt 之后的 token
        gen_ids = _generated_to_ids(generated)
        n_prompt = len(prompt_ids)
        gen_only_ids = gen_ids[n_prompt:] if len(gen_ids) > n_prompt else []

        # 逐 token decode 并打印（流式效果）
        print(_c("CometSpark: ", _COLOR_CYAN), end="", flush=True)
        decoded_text = ""
        for i, tid in enumerate(gen_only_ids):
            try:
                piece = tokenizer.decode([int(tid)], strip_special=True)
            except TypeError:
                piece = tokenizer.decode([int(tid)])
            except Exception:
                piece = ""
            if piece:
                print(piece, end="", flush=True)
                decoded_text += piece

        print(flush=True)

        messages.append({"role": "assistant", "content": decoded_text})

    return 0


# ---------------------------------------------------------------------------
# 子命令：compress
# ---------------------------------------------------------------------------


def cmd_compress(args) -> int:
    """压缩模型。

    调用 ``compress_pipeline()``，支持 prune/quantize/lora 组合。
    """
    # 解析 --method
    methods = [m.strip() for m in args.method.split(",") if m.strip()]
    compress_config = {}
    for m in methods:
        if m == "prune":
            compress_config["prune"] = {"sparsity": args.sparsity}
        elif m == "quantize":
            compress_config["quantize"] = {"qtype": args.qtype}
        elif m == "lora":
            compress_config["lora"] = {
                "r": args.lora_r,
                "alpha": args.lora_alpha,
            }
        elif m == "ternary":
            compress_config["ternary"] = {}
        else:
            _warn(f"未知压缩方法：{m}（跳过）")

    if args.dry_run:
        _print_dry_run(
            "compress",
            checkpoint=args.checkpoint,
            methods=methods,
            config=compress_config,
            output=args.output or "auto",
        )
        return 0

    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"checkpoint 不存在：{args.checkpoint}")

    from spark.model.model import CometSparkV05LM
    from verse_torch.compress import compress_pipeline

    model = CometSparkV05LM.from_pretrained(args.checkpoint)

    if not args.quiet:
        _print_model_info(model)
        _info(f"压缩方法：{', '.join(methods)}")

    original_params = model.count_parameters()
    compressed_model, stats = compress_pipeline(
        model.net, compress_config, return_stats=True
    )

    # 构造新的 CometSparkV05LM，替换内部 net
    new_model = CometSparkV05LM(model.config)
    new_model.net = compressed_model

    compressed_params = new_model.count_parameters()
    ratio = original_params / compressed_params if compressed_params > 0 else 1.0

    _ok(
        f"压缩完成：{original_params:,} → {compressed_params:,} "
        f"(压缩比 {ratio:.2f}x)"
    )

    # 保存
    output_path = args.output
    if not output_path:
        base, ext = os.path.splitext(args.checkpoint)
        output_path = f"{base}_compressed{ext or '.pt'}"

    new_model.save(output_path)
    _ok(f"压缩模型已保存到 {output_path}")

    if not args.quiet and stats:
        _info(f"压缩统计：{stats}")

    return 0


# ---------------------------------------------------------------------------
# 子命令：convert
# ---------------------------------------------------------------------------


def cmd_convert(args) -> int:
    """转换模型格式（.pt ↔ .vn）。"""
    src_lower = args.input.lower()
    dst_lower = args.output.lower()

    if args.dry_run:
        _print_dry_run(
            "convert",
            input=args.input,
            output=args.output,
            direction=(
                ".pt → .vn" if src_lower.endswith(".pt")
                else ".vn → .pt" if src_lower.endswith(".vn")
                else "auto"
            ),
        )
        return 0

    if not os.path.exists(args.input):
        raise FileNotFoundError(f"输入文件不存在：{args.input}")

    from verse_torch.vn_format import pt_to_vn, vn_to_pt, convert_format

    # 读取 chat_template（仅 pt→vn 有意义）
    chat_template = None
    if args.chat_template:
        with open(args.chat_template, "r", encoding="utf-8") as f:
            chat_template = f.read()

    try:
        if src_lower.endswith(".pt") and dst_lower.endswith(".vn"):
            pt_to_vn(
                args.input, args.output,
                arch=args.arch,
                chat_template=chat_template,
                tokenizer=args.tokenizer,
            )
            _ok(f".pt → .vn 完成：{args.input} → {args.output}")
        elif src_lower.endswith(".vn") and dst_lower.endswith(".pt"):
            vn_to_pt(args.input, args.output)
            _ok(f".vn → .pt 完成：{args.input} → {args.output}")
        else:
            convert_format(args.input, args.output)
            _ok(f"转换完成：{args.input} → {args.output}")
    except Exception as e:
        _error(f"转换失败：{type(e).__name__}: {e}")
        return 1

    return 0


# ---------------------------------------------------------------------------
# 子命令：download
# ---------------------------------------------------------------------------


def cmd_download(args) -> int:
    """下载数据集。"""
    if not args.url and not args.hf:
        _error("必须指定 --url 或 --hf 之一")
        return 1

    if args.dry_run:
        _print_dry_run(
            "download",
            url=args.url or args.hf,
            output=args.output or "auto",
            to_npz=args.to_npz,
        )
        return 0

    from verse_infra import DatasetDownloader

    downloader = DatasetDownloader(num_workers=args.workers)

    source = args.url or args.hf
    if args.to_npz:
        npz_path = downloader.download_and_cache(
            source, output_path=args.output, text_key=args.text_key,
        )
        _ok(f"已下载并缓存：{npz_path}")
    elif args.url:
        path = downloader.download_url(
            args.url, output_path=args.output,
            resume=not args.no_resume,
        )
        _ok(f"已下载：{path}")
    else:
        path = downloader.download_hf(
            args.hf, split=args.split, output_dir=args.output,
        )
        _ok(f"已下载：{path}")

    return 0


# ---------------------------------------------------------------------------
# 参数解析
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """构建命令行参数解析器。"""
    parser = argparse.ArgumentParser(
        prog="spark/run.py",
        description="CometSpark 命令行快捷入口（基于 VerseTrainer API）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "用法示例：\n"
            "  python spark/run.py train --small          # 快速训练\n"
            "  python spark/run.py train                    # 1B 训练\n"
            "  python spark/run.py eval --checkpoint ck.pt  # 评估\n"
            "  python spark/run.py generate --prompt '你好' # 生成\n"
            "  python spark/run.py chat --checkpoint ck.pt  # 聊天\n"
            "  python spark/run.py compress --checkpoint ck.pt --method prune,quantize\n"
            "  python spark/run.py convert --input m.pt --output m.vn\n"
            "  python spark/run.py download --url <URL>\n"
        ),
    )
    subparsers = parser.add_subparsers(dest="command", help="子命令")

    # --- train ---
    p_train = subparsers.add_parser("train", help="训练模型")
    p_train.add_argument("--config", default=None, help="配置文件路径")
    p_train.add_argument("--small", action="store_true", help="使用小配置（快速调试）")
    p_train.add_argument("--max-steps", type=int, default=None, help="覆盖 max_steps")
    p_train.add_argument("--batch-size", type=int, default=None, help="覆盖 batch_size")
    p_train.add_argument("--device", default=None, choices=["cpu", "cuda", "npu"],
                         help="设备（默认从 config 读取）")
    p_train.add_argument("--resume", action="store_true", help="从 checkpoint 断点续训")
    p_train.add_argument("--amp", action="store_true", help="启用混合精度")
    p_train.add_argument("--eval-after", dest="eval_after", action="store_true",
                         default=True, help="训练后自动评估（默认开启）")
    p_train.add_argument("--no-eval", dest="eval_after", action="store_false",
                         help="禁用训练后自动评估")
    p_train.add_argument("--dry-run", action="store_true", help="只打印不执行")
    p_train.add_argument("--quiet", action="store_true", help="静默模式")
    p_train.add_argument("--verbose", action="store_true", help="详细日志")
    p_train.set_defaults(func=cmd_train)

    # --- eval ---
    p_eval = subparsers.add_parser("eval", help="评估模型")
    p_eval.add_argument("--config", default=None, help="配置文件路径")
    p_eval.add_argument("--checkpoint", default=None, help="checkpoint 文件路径")
    p_eval.add_argument("--max-tokens", type=int, default=None,
                        help="每条 prompt 最大生成 token 数（默认不限）")
    p_eval.add_argument("--temperature", type=float, default=1.0, help="采样温度")
    p_eval.add_argument("--score", action="store_true", help="启用打分模式")
    p_eval.add_argument("--references-file", default=None, help="参考答案文件")
    p_eval.add_argument("--dry-run", action="store_true", help="只打印不执行")
    p_eval.set_defaults(func=cmd_eval)

    # --- generate ---
    p_gen = subparsers.add_parser("generate", help="生成文本")
    p_gen.add_argument("--checkpoint", default="checkpoints/best.pt",
                       help="checkpoint 文件路径")
    p_gen.add_argument("--config", default=None, help="配置文件路径（加载 tokenizer）")
    p_gen.add_argument("--prompt", default="", help="生成提示文本")
    p_gen.add_argument("--max-tokens", type=int, default=None,
                       help="最大生成 token 数（默认不限，EOS 自然停止）")
    p_gen.add_argument("--temperature", type=float, default=0.8, help="采样温度（默认 0.8）")
    p_gen.add_argument("--top-k", type=int, default=None, help="top-k 采样")
    p_gen.add_argument("--quiet", action="store_true", help="不打印模型信息")
    p_gen.add_argument("--dry-run", action="store_true", help="只打印不执行")
    p_gen.set_defaults(func=cmd_generate)

    # --- chat ---
    p_chat = subparsers.add_parser("chat", help="交互式聊天")
    p_chat.add_argument("--checkpoint", default="checkpoints/best.pt",
                        help="checkpoint 文件路径")
    p_chat.add_argument("--config", default=None, help="配置文件路径（加载 tokenizer）")
    p_chat.add_argument("--max-tokens", type=int, default=512,
                        help="每轮最大生成 token 数（默认 512）")
    p_chat.add_argument("--temperature", type=float, default=0.8, help="采样温度")
    p_chat.add_argument("--dry-run", action="store_true", help="只打印不执行")
    p_chat.set_defaults(func=cmd_chat)

    # --- compress ---
    p_comp = subparsers.add_parser("compress", help="压缩模型")
    p_comp.add_argument("--checkpoint", required=True, help="checkpoint 文件路径")
    p_comp.add_argument("--method", default="prune,quantize",
                        help="压缩方法（逗号分隔：prune,quantize,lora,ternary）")
    p_comp.add_argument("--sparsity", type=float, default=0.3, help="剪枝稀疏度（默认 0.3）")
    p_comp.add_argument("--qtype", default="int4", help="量化类型（默认 int4）")
    p_comp.add_argument("--lora-r", type=int, default=8, help="LoRA 秩（默认 8）")
    p_comp.add_argument("--lora-alpha", type=float, default=16.0, help="LoRA alpha（默认 16）")
    p_comp.add_argument("--output", default=None, help="输出路径（默认自动生成）")
    p_comp.add_argument("--quiet", action="store_true", help="静默模式")
    p_comp.add_argument("--dry-run", action="store_true", help="只打印不执行")
    p_comp.set_defaults(func=cmd_compress)

    # --- convert ---
    p_conv = subparsers.add_parser("convert", help="转换模型格式（.pt ↔ .vn）")
    p_conv.add_argument("--input", required=True, help="输入文件路径")
    p_conv.add_argument("--output", required=True, help="输出文件路径")
    p_conv.add_argument("--chat-template", default=None, help="chat_template.jinja 路径")
    p_conv.add_argument("--tokenizer", default=None, help="tokenizer.json 路径")
    p_conv.add_argument("--arch", default=None, help="覆盖架构名")
    p_conv.add_argument("--dry-run", action="store_true", help="只打印不执行")
    p_conv.set_defaults(func=cmd_convert)

    # --- download ---
    p_dl = subparsers.add_parser("download", help="下载数据集")
    p_dl.add_argument("--url", default=None, help="下载 URL")
    p_dl.add_argument("--hf", default=None, help="HuggingFace dataset repo ID")
    p_dl.add_argument("--split", default="train", help="HF dataset split（默认 train）")
    p_dl.add_argument("--output", "-o", default=None, help="输出路径")
    p_dl.add_argument("--to-npz", action="store_true", help="下载后转 .npz 缓存")
    p_dl.add_argument("--text-key", default="text", help="文本字段名")
    p_dl.add_argument("--workers", type=int, default=4, help="下载线程数")
    p_dl.add_argument("--no-resume", action="store_true", help="禁用断点续传")
    p_dl.add_argument("--dry-run", action="store_true", help="只打印不执行")
    p_dl.set_defaults(func=cmd_download)

    return parser


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------


def main(argv: Optional[List[str]] = None) -> int:
    """主入口：解析参数并分发到对应子命令。"""
    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.command:
        parser.print_help()
        return 1

    # 路径自举（幂等）
    _setup_paths()

    # 执行对应命令
    func = getattr(args, "func", None)
    if func is None:
        parser.print_help()
        return 1

    try:
        return func(args)
    except FileNotFoundError as e:
        _error(str(e))
        return 1
    except KeyboardInterrupt:
        _warn("已中断")
        return 130
    except Exception as e:
        import traceback
        _error(f"执行失败：{type(e).__name__}: {e}")
        traceback.print_exc(file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
