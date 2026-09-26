"""种群档案：保存历史个体与适应度，支持 JSON 持久化。"""

from __future__ import annotations

import json
from pathlib import Path

from verse_rsi.individual import Individual


class Population:
    """个体档案（允许重复 id——同结构多代重评时覆盖 fitness）。"""

    def __init__(self) -> None:
        self.individuals: dict[str, Individual] = {}

    def add(self, individual: Individual) -> None:
        self.individuals[individual.id] = individual

    def best(self) -> Individual | None:
        evaluated = [i for i in self.individuals.values() if i.fitness is not None]
        if not evaluated:
            return None
        return min(evaluated, key=lambda i: i.fitness)

    def topk(self, k: int) -> list[Individual]:
        evaluated = sorted(
            (i for i in self.individuals.values() if i.fitness is not None),
            key=lambda i: i.fitness,
        )
        return evaluated[:k]

    def __len__(self) -> int:
        return len(self.individuals)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"individuals": [i.to_dict() for i in self.individuals.values()]}
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "Population":
        payload = json.loads(Path(path).read_text())
        pop = cls()
        for item in payload.get("individuals", []):
            pop.add(Individual.from_dict(item))
        return pop
