import time

from .models import MoveTracker


class UserTimeTracker:
    def __init__(self) -> None:
        self.session_start = 0.0
        self.last_step_time = 0.0
        self.reset()

    def _now(self) -> float:
        return time.time()

    def _sanitize_clock(self, now: float) -> float:
        if now < self.session_start:
            self.session_start = now
        if now < self.last_step_time:
            self.last_step_time = now
        return now

    def reset(self) -> None:
        now = self._now()
        self.session_start = now
        self.last_step_time = now

    def snapshot(self) -> dict[str, float]:
        now = self._sanitize_clock(self._now())
        return {
            "session_seconds": now - self.session_start,
            "since_last_step_seconds": now - self.last_step_time,
        }

    def mark_step(self) -> dict[str, float]:
        now = self._sanitize_clock(self._now())
        delta = now - self.last_step_time
        total = now - self.session_start
        self.last_step_time = now
        return {"session_seconds": total, "since_last_step_seconds": delta}


class GameTimeTracker:
    def __init__(self) -> None:
        self._move_tracker = MoveTracker()

    @property
    def logical_moves(self) -> int:
        return self._move_tracker.logical_moves

    def seed_from_engine_moves(self, engine_moves: int | None) -> None:
        if not isinstance(engine_moves, int):
            return
        self._move_tracker.logical_moves = engine_moves
        self._move_tracker.last_engine_moves = engine_moves

    def advance(self, engine_moves: int | None) -> int:
        return self._move_tracker.advance(engine_moves)

    def snapshot(self, engine_moves: int | None, logical_moves: int | None = None) -> dict[str, int | None]:
        if logical_moves is None:
            logical_moves = self.logical_moves
        return {"logical_moves": logical_moves, "engine_moves": engine_moves}
