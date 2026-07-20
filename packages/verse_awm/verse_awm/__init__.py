"""VerseAWM: World Model package (JEPA + RSSM + H-JEPA)."""

__version__ = "0.1.0"

from .jepa import (
    JEPABase,
    ContextEncoder,
    TargetEncoder,
    Predictor,
    MultiHeadAttention,
    MLP,
    TransformerBlock,
    update_target_encoder,
    ema_decay_schedule,
    jepa_loss,
)
from .ijepa import IJEPA, PatchEmbed, random_masking
from .vjepa import VJEPA, SpatioTemporalPatchEmbed, video_random_masking
from .rssm import RSSM, VideoRSSM, GRUCell, gumbel_softmax, categorical_kl
from .hjepa import HJEPA

__all__ = [
    # Task 4.1, 4.3, 4.4: JEPA base
    "JEPABase",
    "ContextEncoder",
    "TargetEncoder",
    "Predictor",
    "MultiHeadAttention",
    "MLP",
    "TransformerBlock",
    "update_target_encoder",
    "ema_decay_schedule",
    "jepa_loss",
    # Task 4.2: I-JEPA
    "IJEPA",
    "PatchEmbed",
    "random_masking",
    # Task 4.5: V-JEPA
    "VJEPA",
    "SpatioTemporalPatchEmbed",
    "video_random_masking",
    # Task 4.6: RSSM
    "RSSM",
    "VideoRSSM",
    "GRUCell",
    "gumbel_softmax",
    "categorical_kl",
    # Task 4.7: H-JEPA
    "HJEPA",
]
