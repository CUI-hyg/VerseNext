# RSI（递归自进化）指南

## 概念

VerseNext 的 RSI 把「模型架构 + 训练超参」整体视为一个**个体**，通过
递归循环自动寻找更优的训练配方与网络结构：

```
        ┌────────────────────────────────────────┐
        │   1. 评估（短训练预算，算适应度）          │
        │            ↓                           │
        │   2. 精英选择（fitness 最优的 k 个）      │
        │            ↓                           │
        │   3. 变异（超参 + 架构）→ 子代            │
        │            ↓                           │
        └──── 子代进入下一代（递归）────────────────┘
```

## 适应度

```
fitness = eval_loss + speed_weight * max(0, ref_tps / tps - 1)
```

- `eval_loss`：个体在验证集上的损失（越小越好）
- 速度项：比基准个体慢则受罚；`speed_weight=0` 退化为纯 loss 优化
- 这使 RSI 同时优化**模型质量**与**训练性能**（更小的结构、GQA 等会被偏好）

## 变异算子

| 算子 | 注册名 | 变异内容 |
|---|---|---|
| `HyperparamMutator` | `mutator/hyperparam` | lr 乘性扰动 [0.5x, 2x]；batch_size 跳变 |
| `ArchMutator` | `mutator/arch` | 层数 ±1/±2；头数、d_model 离散跳变；MHA→GQA |
| `CombinedMutator` | `mutator/combined` | 两者叠加（默认） |

架构变异始终保证合法性（整除约束），非法变异自动跳过或修正。

## 自定义算子

```python
from verse_core.registry import register_mutator
from verse_rsi.mutator import BaseMutator

@register_mutator("my_mutator")
class MyMutator(BaseMutator):
    name = "my_mutator"
    def mutate(self, individual, config, rng):
        child = individual.copy()
        child.trainer_overrides.setdefault("optim", {})["lr"] = 1e-3
        return child

# 使用
Evolver(EvolveConfig(mutator="my_mutator"))
```

## 自定义模型参与进化

```python
from verse_core.registry import register_model
from verse_nn.transformer import VerseTransformer

@register_model("my_model")
class MyModel(VerseTransformer):
    pass

# 个体中指定
Individual(model_config=..., trainer_overrides={"model_name": "my_model"})
```

注意：个体覆盖项中 `model_name` 需在 Trainer 中被消费（当前 CLI 场景默认
`verse_transformer`；自定义训练入口可读取该覆盖项传给 `build_model`）。

## 进化产物

- `rsi_archive/population.json`：全部个体（含配置、fitness、lineage 祖先链），
  可用 `Population.load()` 恢复继续进化。
- 事件：每代结束发出 `rsi/generation_end`，可接自定义回调（记录、早停、通知）。
