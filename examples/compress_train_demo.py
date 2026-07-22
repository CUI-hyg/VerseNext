"""压缩训练演示：prune → quantize → finetune → evaluate。

用法：
    cd /workspace
    python examples/compress_train_demo.py

或显式指定 PYTHONPATH（无需 pip install）：
    PYTHONPATH=packages/verse_torch:packages/verse_nex:packages/verse_infra \
        python examples/compress_train_demo.py
"""
import os
import sys

# 优先使用已安装的 verse_torch / verse_nex / verse_infra（pip install -e .）
# 若未安装，则回退到 PYTHONPATH 风格的路径
_HERE = os.path.dirname(os.path.abspath(__file__))
_WORKSPACE = os.path.dirname(_HERE)
for _sub in ("verse_torch", "verse_nex", "verse_infra"):
    _p = os.path.join(_WORKSPACE, "packages", _sub)
    if os.path.isdir(_p):
        sys.path.insert(0, _p)
# spark 是顶层包，需要 _WORKSPACE 在 sys.path
if _WORKSPACE not in sys.path:
    sys.path.insert(0, _WORKSPACE)

from spark.model.model import CometSparkV05Small, CometSparkV05LM


def main():
    print("=" * 60)
    print("CometSpark 压缩训练演示")
    print("=" * 60)

    # 1. 创建基准模型
    print("\n[1] 创建基准模型 CometSparkV05Small...")
    model = CometSparkV05Small()
    baseline_params = model.count_parameters()
    print(f"    参数量: {baseline_params}")

    # 2. 压缩：50% 稀疏度 + INT4 量化
    print("\n[2] 压缩：50% 通道稀疏 + INT4 量化...")
    compress_config = {
        "prune": {"sparsity": 0.5},
        "quantize": {"bits": 4},
    }
    compressed = model.compress(compress_config)
    stats = compressed.compression_stats()

    # 3. 显示统计
    print("\n[3] 压缩统计:")
    print(f"    原始参数量: {stats['original_params']}")
    print(f"    压缩后参数量: {stats['compressed_params']}")
    print(f"    稀疏度: {stats['sparsity']:.2%}")
    print(f"    平均 bit: {stats['bits']:.1f}")
    print(f"    压缩比: {stats['compression_ratio']:.2f}x")

    # 4. 简单 forward 验证
    print("\n[4] Forward 验证（输入随机 token）...")
    import numpy as np
    x = np.random.randint(0, 256, size=(1, 16))
    from verse_torch import Tensor
    out = compressed.forward(Tensor(x))
    print(f"    输入 shape: {x.shape}")
    print(f"    输出 shape: {out.shape}")

    print("\n" + "=" * 60)
    print("演示完成")
    print("=" * 60)


if __name__ == "__main__":
    main()
