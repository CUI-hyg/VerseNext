"""CometSpark 模型定义。

基于 VerseNext 框架的 ``VerseTransformer``（decoder-only：RoPE + GQA +
SwiGLU + RMSNorm），注册为 ``cometspark``，通过 CometSparkConfig 提供本
实验模型的默认超参。版本号由项目根目录的 ``VERSION`` 文件区分：

- 0.1：d_model=128 稠密注意力（sdpa）基线（~35M）
- 0.2：d_model=512 + GQA + DSA（DeepSeek 稀疏注意力 + 索引器辅助损失，~200M）
- 0.3：**正式采用 Dense（sdpa）**，d_model=1024 / 32 层 / GQA(16:4)，
  参数量约 0.6B，开启梯度检查点以适配 CPU 内存
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from verse_core import global_registry
from verse_nn.transformer import VerseTransformer, VerseTransformerConfig

# 项目根目录（CometSpark/）
ROOT_DIR = Path(__file__).resolve().parent.parent
VERSION_FILE = ROOT_DIR / "VERSION"

__version__ = VERSION_FILE.read_text().strip() if VERSION_FILE.exists() else "0.0"
"""模型版本号，来自 CometSpark/VERSION 文件。"""

MODEL_NAME = f"CometSpark-Exp-{__version__}"
"""对外模型名，如 CometSpark-Exp-0.3。"""


@dataclass
class CometSparkConfig(VerseTransformerConfig):
    """CometSpark-Exp 系列的架构配置。

    词表大小与 tokenizer/tokenizer.json 对齐（248044 个 BPE token + 33 个
    特殊 token，最大 id 248076）。默认超参为 **v0.3 规模（约 0.6B）**：
    Dense 注意力（``sdpa``）+ GQA（16 Q 头 / 4 KV 头）+ SwiGLU。历史版本
    的 yaml/checkpoint 均显式携带 ``attn_type``/``d_model`` 等字段，不受默认
    值变化影响。
    """

    vocab_size: int = 248077
    d_model: int = 1024
    num_layers: int = 32
    num_heads: int = 16
    num_kv_heads: int | None = 4      # GQA：16 Q 头共享 4 KV 头
    ffn_hidden_dim: int | None = None  # None = 自动 8/3*d_model 对齐 256 -> 2816
    max_seq_len: int = 512
    dropout: float = 0.0
    tie_weights: bool = True
    gradient_checkpointing: bool = True  # 0.6B 在 CPU 上需靠它压住激活内存
    attn_type: str = "sdpa"            # v0.3 起正式采用 Dense
    dsa_index_head_dim: int = 32
    dsa_top_k: int = 64
    dsa_index_loss_weight: float = 0.1

    def validate(self) -> None:
        super().validate()
        if self.vocab_size < 248077:
            raise ValueError(
                "vocab_size 必须 >= 248077，以覆盖 tokenizer.json 的全部 token id"
            )
        if self.attn_type == "dsa" and self.dsa_top_k > self.max_seq_len:
            raise ValueError(
                f"dsa_top_k({self.dsa_top_k}) 不应超过 max_seq_len({self.max_seq_len})，"
                "否则 DSA 退化为稠密注意力且白付索引器开销"
            )


@global_registry.register_model("cometspark")
class CometSparkTransformer(VerseTransformer):
    """CometSpark 系列模型（结构继承 VerseTransformer）。"""

    config_cls = CometSparkConfig
