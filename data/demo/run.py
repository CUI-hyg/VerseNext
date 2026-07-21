"""CometSpark-v0.1 一键入口。

流程：
    1. set_seed + ensure_dir
    2. （可选）build_tokenizer
    3. （可选）train
    4. （可选）evaluate
    5. （可选）visualize

用法：
    python run.py                              # 全流程
    python run.py --skip-train                 # 仅 build + eval
    python run.py --skip-train --skip-eval     # 仅 build
    python run.py --config config/my.yml       # 自定义配置
    python run.py --verbose                    # 异常时打印完整 traceback
    python run.py --prompt "床前明月光，,你好，"  # 自定义评估 prompt
    python run.py --prompts-file prompts.txt   # 从文件读取 prompt
    python run.py --max-tokens 50              # 每条 prompt 生成 50 token
    python run.py --temperature 0.8 --top-k 10 # 采样参数
    python run.py --arch hybrid                # 覆盖 config 的 arch
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
import time
import traceback

# 临时缓解：提升 Python 递归上限到 2000，避免评估/训练阶段某些遗留递归路径
# （如 backward 的 DFS、对象 __repr__ 等）触发 RecursionError。
# 根因已在 verse_torch.tensor.Tensor.backward 改为迭代式 DFS 修复，
# 此处 setrecursionlimit 仅作兜底防御。
sys.setrecursionlimit(2000)

# 把当前目录加入 sys.path，使 model/ src/ train/ 包可被 import
_DEMO_DIR = os.path.dirname(os.path.abspath(__file__))
if _DEMO_DIR not in sys.path:
    sys.path.insert(0, _DEMO_DIR)

# 自动定位 verse_torch / verse_nex / verse_tokenizer / verse_inference / verse_compat
# 等包目录：run.py 位于 <repo>/data/demo/run.py，packages 在 <repo>/packages/。
# 若包未安装到 site-packages（开发模式），则自动注入 sys.path。
_REPO_ROOT = os.path.dirname(os.path.dirname(_DEMO_DIR))
_PACKAGES_DIR = os.path.join(_REPO_ROOT, "packages")
for _pkg in ("verse_torch", "verse_nex", "verse_tokenizer",
             "verse_inference", "verse_compat"):
    _pkg_path = os.path.join(_PACKAGES_DIR, _pkg)
    if os.path.isdir(_pkg_path) and _pkg_path not in sys.path:
        sys.path.insert(0, _pkg_path)

# 限制 BLAS 线程数：4 核 CPU 上设为 4 避免过度并行
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")

from src.utils import set_seed, ensure_dir
from model.config import load_full_config, save_full_config
from model.tokenizer import build_tokenizer, load_tokenizer
from train.trainer import train as train_fn
from train.evaluate import evaluate as evaluate_fn
from train.visualize import visualize as visualize_fn


def _resolve(base_dir: str, path_str: str) -> str:
    p = os.path.join(base_dir, path_str) if not os.path.isabs(path_str) else path_str
    return os.path.abspath(p)


def _parse_prompts_from_cli(prompt_arg: str | None,
                            prompts_file_arg: str | None,
                            base_dir: str) -> list[str] | None:
    """根据 --prompt / --prompts-file 构造 prompts 列表。

    优先级：
        1. --prompt（逗号分隔，如 ``"床前明月光，,你好，"`` → ``["床前明月光，", "你好，"]``）
        2. --prompts-file（每行一个 prompt，忽略空行与 ``#`` 注释行）
        3. 都未指定 → 返回 None（由 evaluate 使用默认 5 条）

    Args:
        prompt_arg: --prompt 的值（可能为 None）
        prompts_file_arg: --prompts-file 的值（可能为 None）
        base_dir: 用于解析 --prompts-file 的相对路径

    Returns:
        prompts 列表，或 None（表示使用默认 prompt）
    """
    if prompt_arg is not None:
        # 逗号分隔；保留空字符串 prompt（用户可能用 ",,," 表达多条空 prompt）
        # 但末尾/开头的空字符串视为分隔产物，过滤掉
        # 注意：用 ",," 分隔会得到 ["", "", ""]，这里只过滤首尾空字符串
        parts = prompt_arg.split(",")
        # 去掉首尾空字符串，但保留中间的空字符串
        while parts and parts[0] == "":
            parts.pop(0)
        while parts and parts[-1] == "":
            parts.pop()
        return parts if parts else None

    if prompts_file_arg is not None:
        path = _resolve(base_dir, prompts_file_arg)
        if not os.path.exists(path):
            raise FileNotFoundError(f"--prompts-file 指定的文件不存在：{path}")
        prompts = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                # 去掉行尾换行；忽略空行与 # 开头的注释行
                line = line.rstrip("\n").rstrip("\r")
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                prompts.append(line)
        return prompts if prompts else None

    return None


def _override_config_arch(config_path: str, arch: str | None) -> tuple[str, str | None]:
    """若 arch 指定，创建一个临时 config 文件覆盖 model.arch 字段。

    Args:
        config_path: 原始 config.yml 路径
        arch: 覆盖的 arch 值（"transformer" / "hybrid"）；None 表示不覆盖

    Returns:
        (effective_config_path, temp_path_to_cleanup)
        - effective_config_path: 实际使用的 config 路径（原始或临时）
        - temp_path_to_cleanup: 临时文件路径（需用完后删除）；None 表示未创建临时文件
    """
    if arch is None:
        return config_path, None

    if arch not in ("transformer", "hybrid"):
        raise ValueError(
            f"--arch 必须为 'transformer' 或 'hybrid'，得到 {arch!r}"
        )

    # 读取原始 config，覆盖 arch 字段，写入临时文件
    full_cfg = load_full_config(config_path)
    model_cfg = full_cfg.setdefault("model", {})
    model_cfg["arch"] = arch

    fd, tmp_path = tempfile.mkstemp(
        suffix=".yml", prefix="cometspark_arch_override_"
    )
    os.close(fd)
    save_full_config(full_cfg, tmp_path)
    print(f"[run.py] --arch {arch!r} 已覆盖 config.yml 的 model.arch 字段", flush=True)
    print(f"[run.py] 临时 config 文件：{tmp_path}", flush=True)
    return tmp_path, tmp_path


def _override_config_parallel_chunks(config_path: str, parallel_chunks: int | None) -> tuple[str, str | None]:
    """Part3K2 Task 1.8: 若 --parallel-chunks 指定，覆盖 training.parallel_chunks 字段。

    与 ``_override_config_arch`` 类似，创建临时 config 文件实现覆盖。

    Args:
        config_path: 原始 config.yml 路径
        parallel_chunks: 覆盖的 parallel_chunks 值；None 表示不覆盖

    Returns:
        (effective_config_path, temp_path_to_cleanup)
    """
    if parallel_chunks is None:
        return config_path, None

    if parallel_chunks < 1:
        raise ValueError(
            f"--parallel-chunks 必须 >= 1，得到 {parallel_chunks!r}"
        )

    full_cfg = load_full_config(config_path)
    train_cfg = full_cfg.setdefault("training", {})
    train_cfg["parallel_chunks"] = int(parallel_chunks)

    fd, tmp_path = tempfile.mkstemp(
        suffix=".yml", prefix="cometspark_pchunks_override_"
    )
    os.close(fd)
    save_full_config(full_cfg, tmp_path)
    print(f"[run.py] --parallel-chunks {parallel_chunks} 已覆盖 config.yml 的 "
          f"training.parallel_chunks 字段", flush=True)
    print(f"[run.py] 临时 config 文件：{tmp_path}", flush=True)
    return tmp_path, tmp_path


def stage_build_tokenizer(config_path: str, base_dir: str, force: bool = False) -> str:
    """构建并保存 tokenizer。"""
    print("=" * 70, flush=True)
    print("[stage 1/3] 构建 tokenizer", flush=True)
    print("=" * 70, flush=True)
    t0 = time.time()

    full_cfg = load_full_config(config_path)
    tok_cfg = full_cfg.get("tokenizer", {})
    ckpt_cfg = full_cfg.get("checkpoint", {})
    data_cfg = full_cfg.get("data", {})

    tok_kind = str(tok_cfg.get("kind", "byte"))
    vocab_size = int(tok_cfg.get("vocab_size", 259))
    save_dir = _resolve(base_dir, str(ckpt_cfg.get("save_dir", "checkpoints")))
    ensure_dir(save_dir)
    tok_path = os.path.join(save_dir, "tokenizer.json")

    if os.path.exists(tok_path) and not force:
        print(f"[build_tokenizer] 已存在 {tok_path}，跳过（--force 可覆盖）", flush=True)
        return tok_path

    if tok_kind == "byte":
        # ByteTokenizer 不需要 corpus
        print(f"[build_tokenizer] kind=byte, vocab_size=259 (固定)", flush=True)
        build_tokenizer(
            corpus_path="",
            vocab_size=259,
            save_path=tok_path,
            kind="byte",
        )
    else:
        # BPE 需要从训练语料构建
        train_path = _resolve(base_dir, str(data_cfg.get("train_path", "data/train.jsonl")))
        if not os.path.exists(train_path):
            raise FileNotFoundError(f"BPE 训练语料不存在：{train_path}")
        print(f"[build_tokenizer] kind=bpe, vocab_size={vocab_size}, corpus={train_path}", flush=True)
        build_tokenizer(
            corpus_path=train_path,
            vocab_size=vocab_size,
            save_path=tok_path,
            kind="bpe",
        )

    # 验证：加载并打印 vocab_size
    tok = load_tokenizer(tok_path, kind=tok_kind)
    print(f"[build_tokenizer] 完成，vocab_size={len(tok)}，wall_clock={time.time()-t0:.2f}s", flush=True)
    return tok_path


def stage_train(config_path: str, base_dir: str) -> dict:
    """训练阶段。"""
    print("=" * 70, flush=True)
    print("[stage 2/3] 训练", flush=True)
    print("=" * 70, flush=True)
    return train_fn(config_path, base_dir=base_dir)


def stage_evaluate(
    config_path: str,
    base_dir: str,
    prompts: list[str] | None = None,
    max_new_tokens: int = 30,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    score: bool = False,
    references_file: str | None = None,
) -> dict:
    """评估阶段。

    Args:
        config_path: 配置文件路径
        base_dir: 相对路径基准
        prompts: 自定义 prompt 列表；None 用 evaluate.py 默认 5 条
        max_new_tokens: 每条 prompt 生成 token 数（默认 30）
        temperature: 采样温度（默认 1.0，等价 greedy）
        top_k: top-k 采样；None 表示无限制
        top_p: nucleus sampling 阈值 (0,1)；None 表示不限制
        score: Part3K2 Task 1.7 是否启用评分模式
        references_file: 参考答案文件路径（仅 score=True 时生效）
    """
    print("=" * 70, flush=True)
    print("[stage 3/3] 评估", flush=True)
    print("=" * 70, flush=True)
    return evaluate_fn(
        config_path,
        base_dir=base_dir,
        prompts=prompts,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        score=score,
        references_file=references_file,
    )


def stage_visualize(loss_history_path: str, save_dir: str) -> str:
    """可视化阶段。"""
    print("=" * 70, flush=True)
    print("[extra] 可视化 loss 曲线", flush=True)
    print("=" * 70, flush=True)
    curve_path = os.path.join(save_dir, "loss_curve.png")
    return visualize_fn(loss_history_path, save_path=curve_path)


def _print_stage_error(stage_name: str, exc: BaseException, verbose: bool) -> None:
    """统一格式化打印阶段错误：阶段名 + 异常类型 + 消息，verbose 时附 traceback。

    Args:
        stage_name: 阶段名（如 "build_tokenizer" / "train" / "evaluate" / "visualize"）
        exc: 捕获的异常
        verbose: 是否打印完整 traceback
    """
    print("", file=sys.stderr, flush=True)
    print("=" * 70, file=sys.stderr, flush=True)
    print(f"[{stage_name}] 阶段失败", file=sys.stderr, flush=True)
    print(f"  异常类型 : {type(exc).__name__}", file=sys.stderr, flush=True)
    print(f"  异常消息 : {exc}", file=sys.stderr, flush=True)
    if verbose:
        print("-" * 70, file=sys.stderr, flush=True)
        traceback.print_exc(file=sys.stderr)
    print("=" * 70, file=sys.stderr, flush=True)


def main():
    parser = argparse.ArgumentParser(
        description="CometSpark-v0.1 一键训练 + 评估入口"
    )
    parser.add_argument(
        "--config", default="config/config.yml",
        help="配置文件路径（默认 config/config.yml）",
    )
    parser.add_argument(
        "--skip-train", action="store_true",
        help="跳过训练阶段（仅 build + eval）",
    )
    parser.add_argument(
        "--skip-eval", action="store_true",
        help="跳过评估阶段",
    )
    parser.add_argument(
        "--skip-build", action="store_true",
        help="跳过 tokenizer 构建（已有 tokenizer.json 时使用）",
    )
    parser.add_argument(
        "--force-build", action="store_true",
        help="强制重建 tokenizer（覆盖已有文件）",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="异常时打印完整 traceback（用于调试）",
    )
    # Task 7: 自定义 prompt 支持
    parser.add_argument(
        "--prompt", default=None,
        help="自定义评估 prompt，逗号分隔多条（如 --prompt \"床前明月光，,你好，\"）",
    )
    parser.add_argument(
        "--prompts-file", default=None,
        help="从文件读取 prompt（每行一个，忽略空行与 # 注释行）",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=30,
        help="每条 prompt 生成最大 token 数（默认 30）",
    )
    parser.add_argument(
        "--temperature", type=float, default=1.0,
        help="采样温度（默认 1.0 等价 greedy；>1 增加随机性，<1 收敛）",
    )
    parser.add_argument(
        "--top-k", type=int, default=None,
        help="top-k 采样（默认 None 表示 greedy）",
    )
    # Task 9: --arch 覆盖 config 的 arch 字段
    parser.add_argument(
        "--arch", default=None, choices=["transformer", "hybrid"],
        help="覆盖 config 的 model.arch 字段（transformer / hybrid）",
    )
    # Part3K2 Task 1.8: 新增 CLI 参数
    parser.add_argument(
        "--top-p", type=float, default=None,
        help="nucleus sampling 阈值 (0,1)；None 表示不限制。"
             "注意：CometSparkLM.generate 当前不支持 top_p，会自动降级。",
    )
    parser.add_argument(
        "--parallel-chunks", type=int, default=None,
        help="覆盖 config 的 training.parallel_chunks 字段（1=标准 Trainer，"
             ">1=ParallelTrainer chunk 拆分训练）",
    )
    parser.add_argument(
        "--score", action="store_true",
        help="启用评分模式：对生成结果与参考答案计算 5 个指标"
             "（exact_match/prefix_accuracy/char_f1/bleu/rouge_l）",
    )
    parser.add_argument(
        "--references-file", default=None,
        help="参考答案文件路径（每行一个，与 prompts 一一对应）；仅 --score 时生效",
    )
    args = parser.parse_args()

    base_dir = _DEMO_DIR
    config_path = _resolve(base_dir, args.config)
    if not os.path.exists(config_path):
        print(f"错误：配置文件不存在 {config_path}", file=sys.stderr)
        sys.exit(1)

    print(f"CometSpark-v0.1 端到端流程开始", flush=True)
    print(f"  base_dir       = {base_dir}", flush=True)
    print(f"  config_path    = {config_path}", flush=True)
    print(f"  skip_build     = {args.skip_build}", flush=True)
    print(f"  skip_train     = {args.skip_train}", flush=True)
    print(f"  skip_eval      = {args.skip_eval}", flush=True)
    print(f"  verbose        = {args.verbose}", flush=True)
    print(f"  prompt         = {args.prompt!r}", flush=True)
    print(f"  prompts_file   = {args.prompts_file!r}", flush=True)
    print(f"  max_tokens     = {args.max_tokens}", flush=True)
    print(f"  temperature    = {args.temperature}", flush=True)
    print(f"  top_k          = {args.top_k}", flush=True)
    print(f"  top_p          = {args.top_p}", flush=True)
    print(f"  arch           = {args.arch!r}", flush=True)
    print(f"  parallel_chunks= {args.parallel_chunks}", flush=True)
    print(f"  score          = {args.score}", flush=True)
    print(f"  references_file= {args.references_file!r}", flush=True)
    print("", flush=True)

    overall_t0 = time.time()
    set_seed(42)

    # Task 9.3: 若 --arch 指定，覆盖 config 的 arch 字段（创建临时 config 文件）
    # Part3K2 Task 1.8: 若 --parallel-chunks 指定，覆盖 training.parallel_chunks 字段
    # 两层覆盖可叠加：先 arch 再 parallel_chunks（第二个临时文件基于第一个生成）
    tmp_configs_to_cleanup: list[str] = []
    try:
        effective_config, tmp_arch = _override_config_arch(
            config_path, args.arch
        )
        if tmp_arch is not None:
            tmp_configs_to_cleanup.append(tmp_arch)
        effective_config, tmp_pchunks = _override_config_parallel_chunks(
            effective_config, args.parallel_chunks
        )
        if tmp_pchunks is not None:
            tmp_configs_to_cleanup.append(tmp_pchunks)
    except Exception as e:
        print(f"[run.py] config 覆盖失败：{type(e).__name__}: {e}", file=sys.stderr)
        # 清理已创建的临时文件
        for tmp_path in tmp_configs_to_cleanup:
            try:
                os.remove(tmp_path)
            except OSError:
                pass
        sys.exit(1)

    # 顶层 try/except：捕获任何未处理异常，打印友好错误后以非 0 退出
    try:
        # Stage 1: build tokenizer
        if not args.skip_build:
            try:
                tok_path = stage_build_tokenizer(
                    effective_config, base_dir, force=args.force_build
                )
            except Exception as e:
                _print_stage_error("build_tokenizer", e, args.verbose)
                raise
        else:
            print("[stage 1/3] 跳过 tokenizer 构建", flush=True)

        # Stage 2: train
        train_result = None
        if not args.skip_train:
            try:
                train_result = stage_train(effective_config, base_dir)
            except Exception as e:
                _print_stage_error("train", e, args.verbose)
                raise
        else:
            print("[stage 2/3] 跳过训练", flush=True)

        # Stage 3: evaluate
        # Task 7.3: 构造 prompts 列表（--prompt > --prompts-file > 默认 5 条）
        eval_result = None
        if not args.skip_eval:
            try:
                prompts = _parse_prompts_from_cli(
                    args.prompt, args.prompts_file, base_dir
                )
                if prompts is not None:
                    print(f"[run.py] 使用 {len(prompts)} 条自定义 prompt", flush=True)
                    for i, p in enumerate(prompts):
                        print(f"  [{i+1}] {p!r}", flush=True)
                else:
                    print("[run.py] 未指定 --prompt / --prompts-file，使用默认 5 条 prompt", flush=True)
                eval_result = stage_evaluate(
                    effective_config,
                    base_dir,
                    prompts=prompts,
                    max_new_tokens=args.max_tokens,
                    temperature=args.temperature,
                    top_k=args.top_k,
                    top_p=args.top_p,
                    score=args.score,
                    references_file=args.references_file,
                )
            except Exception as e:
                # 评估失败不应让整体流程退出码非 0，但打印友好错误
                _print_stage_error("evaluate", e, args.verbose)
                print("[evaluate] 跳过评估（继续后续步骤）", flush=True)

        # Extra: visualize
        if train_result is not None:
            try:
                stage_visualize(
                    train_result["loss_history_path"],
                    train_result["checkpoint_dir"],
                )
            except Exception as e:
                _print_stage_error("visualize", e, args.verbose)

    except Exception as e:
        # 顶层兜底：阶段内部已 raise 的异常在此统一处理
        # （评估阶段已吞掉异常，不会到此处）
        print(f"\n[run.py] 流程因错误终止：{type(e).__name__}: {e}", file=sys.stderr)
        if args.verbose:
            traceback.print_exc(file=sys.stderr)
        sys.exit(1)
    finally:
        # 清理 --arch / --parallel-chunks 创建的临时 config 文件
        for tmp_path in tmp_configs_to_cleanup:
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass

    # 汇总
    print("", flush=True)
    print("=" * 70, flush=True)
    print("CometSpark-v0.1 流程汇总", flush=True)
    print("=" * 70, flush=True)
    print(f"  总耗时 wall_clock = {time.time() - overall_t0:.2f}s", flush=True)
    if train_result is not None:
        print(f"  训练初始 loss = {train_result['initial_loss']:.4f}", flush=True)
        print(f"  训练最终 loss = {train_result['final_loss']:.4f}", flush=True)
        print(f"  最佳验证 loss = {train_result['best_val_loss']:.4f}", flush=True)
        print(f"  检查点目录    = {train_result['checkpoint_dir']}", flush=True)
    if eval_result is not None:
        print(f"  评估输出 {len(eval_result['results'])} 条样本：", flush=True)
        for r in eval_result["results"]:
            print(f"    [prompt] {r['prompt']!r} -> [output] {r['generated']!r}", flush=True)
    print("完成。", flush=True)


if __name__ == "__main__":
    main()
