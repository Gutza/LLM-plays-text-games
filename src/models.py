from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class AuxPanelSpec:
    title: str
    command: str


@dataclass(frozen=True, slots=True)
class AuxPanelResult:
    title: str
    command: str
    content: str

    def to_dict(self) -> dict[str, Any]:
        return {"title": self.title, "command": self.command, "content": self.content}


@dataclass(frozen=True, slots=True)
class AuxMeta:
    moves: int | None = None
    score: int | None = None
    reward: int | float | None = None
    done: bool | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {}
        if self.moves is not None:
            data["moves"] = self.moves
        if self.score is not None:
            data["score"] = self.score
        if self.reward is not None:
            data["reward"] = self.reward
        if self.done is not None:
            data["done"] = self.done
        if self.extra:
            data.update(self.extra)
        return data


@dataclass(frozen=True, slots=True)
class AuxSnapshot:
    meta: AuxMeta
    panels: list[AuxPanelResult] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "meta": self.meta.to_dict(),
            "panels": [panel.to_dict() for panel in self.panels],
        }


@dataclass(slots=True)
class RepeatActionTracker:
    last_key: tuple[str, str, str, str] | None = None
    repeat_count: int = 0
    last_augmented_observation: str | None = None


@dataclass(slots=True)
class MoveTracker:
    logical_moves: int = 0
    last_engine_moves: int | None = None

    def advance(self, engine_moves: int | None) -> int:
        if not isinstance(engine_moves, int):
            self.logical_moves += 1
            return self.logical_moves
        if self.last_engine_moves is None:
            self.last_engine_moves = engine_moves
            self.logical_moves = max(self.logical_moves, engine_moves)
            return self.logical_moves
        if engine_moves > self.last_engine_moves:
            self.last_engine_moves = engine_moves
            if engine_moves > self.logical_moves:
                self.logical_moves = engine_moves
                return self.logical_moves
        self.logical_moves += 1
        return self.logical_moves
