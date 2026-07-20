"""CometSpark train 模块：训练、评估、可视化入口。"""

from .trainer import train
from .evaluate import evaluate
from .visualize import visualize

__all__ = ["train", "evaluate", "visualize"]
