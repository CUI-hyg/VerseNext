"""MNIST MLP 训练示例（VerseTorch）。

任务：用 2 层 MLP（784 -> 128 -> 10）在 MNIST 上训练 5 epoch，
测试集准确率应 ≥ 95%。

数据来源：https://ossci-datasets.s3.amazonaws.com/mnist/（PyTorch 镜像）
缓存路径：datasets/raw/mnist/

依赖：仅 NumPy + 标准库（urllib, gzip, struct, pathlib）。
"""

from __future__ import annotations

import gzip
import os
import struct
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

# 让 examples/ 目录能直接 import verse_torch（无需 pip install）
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "packages" / "verse_torch"))

from verse_torch import Tensor, nn, optim, losses, no_grad


# ---------------------------------------------------------------------------
# MNIST 数据加载
# ---------------------------------------------------------------------------

MNIST_FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}

MNIST_URL_BASE = "https://ossci-datasets.s3.amazonaws.com/mnist/"


def get_mnist_dir() -> Path:
    """返回 MNIST 缓存目录（不存在则创建）。"""
    base = _REPO_ROOT / "datasets" / "raw" / "mnist"
    base.mkdir(parents=True, exist_ok=True)
    return base


def download_mnist() -> None:
    """下载 MNIST 数据集（如果尚未缓存）。"""
    mnist_dir = get_mnist_dir()
    for name, fname in MNIST_FILES.items():
        fpath = mnist_dir / fname
        if fpath.exists():
            continue
        url = MNIST_URL_BASE + fname
        print(f"[download] {url} -> {fpath}")
        try:
            urllib.request.urlretrieve(url, fpath)
        except Exception as e:
            print(f"[error] failed to download {url}: {e}")
            raise


def parse_idx_images(fpath: Path) -> np.ndarray:
    """解析 IDX 格式的图像文件（gzip 压缩）。

    返回 (N, 28, 28) uint8 数组。
    """
    with gzip.open(fpath, "rb") as f:
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
        assert magic == 2051, f"bad magic for images: {magic}"
        data = np.frombuffer(f.read(), dtype=np.uint8)
        return data.reshape(num, rows, cols)


def parse_idx_labels(fpath: Path) -> np.ndarray:
    """解析 IDX 格式的标签文件（gzip 压缩）。"""
    with gzip.open(fpath, "rb") as f:
        magic, num = struct.unpack(">II", f.read(8))
        assert magic == 2049, f"bad magic for labels: {magic}"
        return np.frombuffer(f.read(), dtype=np.uint8)


def load_mnist() -> tuple:
    """加载 MNIST，返回 (train_x, train_y, test_x, test_y)。

    train_x: (60000, 784) float32，归一化到 [0, 1]
    train_y: (60000,) int64
    test_x:  (10000, 784) float32
    test_y:  (10000,) int64
    """
    download_mnist()
    mnist_dir = get_mnist_dir()
    train_img = parse_idx_images(mnist_dir / MNIST_FILES["train_images"])
    train_lbl = parse_idx_labels(mnist_dir / MNIST_FILES["train_labels"])
    test_img = parse_idx_images(mnist_dir / MNIST_FILES["test_images"])
    test_lbl = parse_idx_labels(mnist_dir / MNIST_FILES["test_labels"])

    # 展平为 (N, 784)，并归一化到 [0, 1]
    train_x = train_img.reshape(-1, 784).astype(np.float32) / 255.0
    test_x = test_img.reshape(-1, 784).astype(np.float32) / 255.0
    train_y = train_lbl.astype(np.int64)
    test_y = test_lbl.astype(np.int64)
    return train_x, train_y, test_x, test_y


# ---------------------------------------------------------------------------
# MLP 模型
# ---------------------------------------------------------------------------


class MLP(nn.Module):
    """2 层 MLP：784 -> 128 -> 10。"""

    def __init__(self, in_features: int = 784, hidden: int = 128, num_classes: int = 10):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden)
        self.fc2 = nn.Linear(hidden, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        # x: (N, 784)
        x = self.fc1(x)
        x = x.relu()
        x = self.fc2(x)
        return x  # logits: (N, 10)


# ---------------------------------------------------------------------------
# 训练循环
# ---------------------------------------------------------------------------


def iterate_minibatches(X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool = True):
    """生成 (X_batch, y_batch) 的迭代器。"""
    n = X.shape[0]
    if shuffle:
        idx = np.random.permutation(n)
    else:
        idx = np.arange(n)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        sel = idx[start:end]
        yield X[sel], y[sel]


def accuracy(model: nn.Module, X: np.ndarray, y: np.ndarray, batch_size: int = 1000) -> float:
    """计算模型在 (X, y) 上的准确率。"""
    model.eval()
    correct = 0
    total = 0
    with no_grad():
        for xb, yb in iterate_minibatches(X, y, batch_size, shuffle=False):
            x = Tensor(xb, requires_grad=False)
            logits = model(x)
            preds = np.argmax(logits.data, axis=1)
            correct += int((preds == yb).sum())
            total += len(yb)
    model.train()
    return correct / total


def main():
    np.random.seed(42)
    print("=" * 60)
    print("MNIST MLP Training with VerseTorch")
    print("=" * 60)

    # 加载数据
    print("\n[1/4] Loading MNIST dataset...")
    t0 = time.time()
    try:
        train_x, train_y, test_x, test_y = load_mnist()
        print(f"  Train: {train_x.shape}, Test: {test_x.shape}")
        print(f"  Loaded in {time.time() - t0:.2f}s")
    except Exception as e:
        print(f"  [error] MNIST download failed: {e}")
        print("  Falling back to synthetic data for sanity check.")
        # 合成数据：3 类，每类 200 个样本
        n_per = 500
        Xs = []
        Ys = []
        for c in range(10):
            center = np.random.randn(784) * 3
            Xc = center + np.random.randn(n_per, 784) * 0.5
            Xs.append(Xc)
            Ys.append(np.full(n_per, c))
        train_x = np.concatenate(Xs).astype(np.float32)
        train_y = np.concatenate(Ys).astype(np.int64)
        # 测试集
        Xs = []
        Ys = []
        for c in range(10):
            center = np.random.randn(784) * 3
            Xc = center + np.random.randn(50, 784) * 0.5
            Xs.append(Xc)
            Ys.append(np.full(50, c))
        test_x = np.concatenate(Xs).astype(np.float32)
        test_y = np.concatenate(Ys).astype(np.int64)

    # 模型与优化器
    print("\n[2/4] Building model...")
    model = MLP(in_features=784, hidden=128, num_classes=10)
    n_params = sum(p.data.size for p in model.parameters())
    print(f"  Model: MLP(784 -> 128 -> 10), params={n_params}")

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    print(f"  Optimizer: Adam(lr=1e-3)")

    # 训练
    epochs = 5
    batch_size = 64
    print(f"\n[3/4] Training {epochs} epochs, batch_size={batch_size}...")

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        t_epoch = time.time()
        for xb, yb in iterate_minibatches(train_x, train_y, batch_size, shuffle=True):
            x = Tensor(xb, requires_grad=False)
            logits = model(x)
            loss = losses.cross_entropy(logits, yb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += float(loss.item())
            n_batches += 1

        avg_loss = epoch_loss / n_batches
        # 训练集准确率（小样本评估）
        sample_size = min(5000, train_x.shape[0])
        train_acc = accuracy(model, train_x[:sample_size], train_y[:sample_size])
        test_acc = accuracy(model, test_x, test_y)
        elapsed = time.time() - t_epoch
        print(f"  Epoch {epoch + 1}/{epochs} | loss={avg_loss:.4f} | "
              f"train_acc={train_acc:.4f} | test_acc={test_acc:.4f} | {elapsed:.1f}s")

    # 最终评估
    print("\n[4/4] Final evaluation...")
    final_test_acc = accuracy(model, test_x, test_y)
    print(f"  Final test accuracy: {final_test_acc * 100:.2f}%")

    if final_test_acc >= 0.95:
        print("  ✓ PASS: accuracy >= 95%")
    else:
        print(f"  ✗ FAIL: accuracy {final_test_acc * 100:.2f}% < 95%")

    return final_test_acc


if __name__ == "__main__":
    main()
