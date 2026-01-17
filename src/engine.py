from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from jericho import FrotzEnv

from .models import AuxMeta, AuxPanelResult, AuxPanelSpec, AuxSnapshot, RepeatActionTracker
from .timekeeping import GameTimeTracker, UserTimeTracker

DEFAULT_AUX_PANELS = [
    AuxPanelSpec(title="INVENTORY", command="inventory"),
    AuxPanelSpec(title="ENVIRONMENT", command="look"),
]


@dataclass(slots=True)
class StepResult:
    command: str
    observation: str
    augmented_observation: str
    reward: int | float
    done: bool
    info: Any
    aux_snapshot: AuxSnapshot
    llm_journal: dict[str, Any] | None
    repeat_count: int


def format_aux_snapshot(aux: AuxSnapshot, *, header: str) -> str:
    parts = [header, ""]
    meta = aux.meta.to_dict()
    if meta:
        parts.append("META:")
        for key in sorted(meta.keys()):
            parts.append(f"- {key}: {meta.get(key)}")
        parts.append("")
    for panel in aux.panels:
        title = panel.title or "PANEL"
        content = (panel.content or "").strip()
        parts.append(f"{title}:")
        parts.append(content)
        parts.append("")
    return "\n".join(parts).strip()


def get_panel_content(aux: AuxSnapshot | None, title: str) -> str:
    if not aux or not title:
        return ""
    for panel in aux.panels:
        if panel.title == title:
            return panel.content or ""
    return ""


class GameEngine:
    def __init__(
        self,
        game_path: Path,
        *,
        seed: int,
        aux_panels: Iterable[AuxPanelSpec] | None = None,
    ) -> None:
        self.game_path = Path(game_path)
        self.seed = seed
        self.aux_panels = list(aux_panels) if aux_panels is not None else []
        self.env: FrotzEnv | None = None
        self.current_state = None
        self.observation = ""
        self.reward: int | float = 0
        self.done = False
        self.info: Any = None
        self.repeat_tracker = RepeatActionTracker()
        self.user_time = UserTimeTracker()
        self.game_time = GameTimeTracker()

    def initialize(self) -> tuple[str, Any]:
        self.env = FrotzEnv(str(self.game_path), seed=self.seed)
        self.observation, self.info = self.env.reset()
        self.reward = 0
        self.done = False
        self.current_state = self.env.get_state()
        self.user_time.reset()
        self.game_time.seed_from_engine_moves(self._engine_moves_from_info(self.info))
        return self.observation, self.info

    def close(self) -> None:
        if not self.env:
            return
        close_fn = getattr(self.env, "close", None)
        if callable(close_fn):
            close_fn()
        self.env = None

    def replay(self, steps: list[dict[str, Any]]) -> None:
        if not self.env:
            raise RuntimeError("Engine must be initialized before replay.")
        command_queue: list[str] = []
        for step in steps:
            command = step.get("command", "")
            if not isinstance(command, str) or not command.strip():
                continue
            command_queue.append(command)
        executed = 0
        for command in command_queue:
            executed += 1
            self._execute_command(
                command,
                llm_journal=None,
                track_time=False,
                update_repeat=False,
            )
            if self.done:
                break
        if self.done and executed < len(command_queue):
            raise RuntimeError("Replay ended early; savegame does not match engine state.")

    def get_observation_for_llm(self) -> str:
        return self.repeat_tracker.last_augmented_observation or self.observation

    def build_aux_snapshot(self, *, include_aux_snapshot: bool) -> AuxSnapshot:
        if not self.env:
            raise RuntimeError("Engine is not initialized.")
        extra_meta = {
            "user_time": self.user_time.snapshot(),
            "game_time": self.game_time.snapshot(
                self._engine_moves_from_info(self.info),
                self.game_time.logical_moves,
            ),
        }
        return self._build_aux_snapshot(
            info=self.info,
            reward=self.reward,
            done=self.done,
            include_aux_snapshot=include_aux_snapshot,
            main_observation=self.observation,
            moves_override=self.game_time.logical_moves,
            extra_meta=extra_meta,
        )

    def step(self, command: str, llm_journal: dict[str, Any] | None) -> StepResult:
        if not command or not command.strip():
            raise ValueError("Command is required.")
        return self._execute_command(
            command,
            llm_journal=llm_journal,
            track_time=True,
            update_repeat=True,
        )

    def _execute_command(
        self,
        command: str,
        *,
        llm_journal: dict[str, Any] | None,
        track_time: bool,
        update_repeat: bool,
    ) -> StepResult:
        if not self.env:
            raise RuntimeError("Engine is not initialized.")
        self.env.set_state(self.current_state)
        observation, reward, done, info = self.env.step(command)
        engine_moves = self._engine_moves_from_info(info)
        logical_moves = self.game_time.advance(engine_moves)
        extra_meta: dict[str, Any] = {
            "game_time": self.game_time.snapshot(engine_moves, logical_moves),
        }
        if track_time:
            extra_meta["user_time"] = self.user_time.mark_step()

        aux_post = self._build_aux_snapshot(
            info=info,
            reward=reward,
            done=done,
            include_aux_snapshot=True,
            main_observation=observation,
            moves_override=logical_moves,
            extra_meta=extra_meta,
        )
        state_after = self.env.get_state()
        self.env.set_state(state_after)
        self.current_state = state_after
        self.observation = observation
        self.reward = reward
        self.done = done
        self.info = info

        augmented_observation = (observation or "").strip()
        if update_repeat:
            environment_text = get_panel_content(aux_post, "ENVIRONMENT").strip()
            inventory_text = get_panel_content(aux_post, "INVENTORY").strip()
            observation_text = augmented_observation
            repeat_key = (command, observation_text, environment_text, inventory_text)
            if self.repeat_tracker.last_key == repeat_key:
                self.repeat_tracker.repeat_count += 1
            else:
                self.repeat_tracker.last_key = repeat_key
                self.repeat_tracker.repeat_count = 1
            if self.repeat_tracker.repeat_count > 1:
                augmented_observation = (
                    f"{observation_text}\n"
                    f"(You did the exact same thing {self.repeat_tracker.repeat_count} times in a row. This is not a warning, just a reminder.)"
                )
            self.repeat_tracker.last_augmented_observation = augmented_observation
        else:
            self.repeat_tracker.last_augmented_observation = augmented_observation

        return StepResult(
            command=command,
            observation=observation,
            augmented_observation=augmented_observation,
            reward=reward,
            done=done,
            info=info,
            aux_snapshot=aux_post,
            llm_journal=llm_journal,
            repeat_count=self.repeat_tracker.repeat_count,
        )

    def _engine_moves_from_info(self, info: Any) -> int | None:
        if isinstance(info, dict):
            return info.get("moves")
        return None

    def _extract_aux_meta(
        self,
        info: Any,
        *,
        reward: int | float,
        done: bool,
        moves_override: int | None,
        extra_meta: dict[str, Any] | None,
    ) -> AuxMeta:
        moves = None
        score = None
        if isinstance(info, dict):
            if "moves" in info:
                moves = info.get("moves")
            if "score" in info:
                score = info.get("score")
        if moves_override is not None:
            moves = moves_override
        return AuxMeta(
            moves=moves,
            score=score,
            reward=reward,
            done=done,
            extra=extra_meta or {},
        )

    def _build_aux_snapshot(
        self,
        *,
        info: Any,
        reward: int | float,
        done: bool,
        include_aux_snapshot: bool,
        main_observation: str,
        moves_override: int | None,
        extra_meta: dict[str, Any] | None,
    ) -> AuxSnapshot:
        if not self.env:
            raise RuntimeError("Engine is not initialized.")
        aux = AuxSnapshot(
            meta=self._extract_aux_meta(
                info,
                reward=reward,
                done=done,
                moves_override=moves_override,
                extra_meta=extra_meta,
            )
        )
        if include_aux_snapshot and not done:
            saved_state = self.env.get_state()
            main_text = (main_observation or "").strip()
            aux = AuxSnapshot(
                meta=aux.meta,
                panels=[
                    panel
                    for panel in self._collect_aux_panels()
                    if panel.content and panel.content not in main_text
                ],
            )
            self.env.set_state(saved_state)
        return aux

    def _collect_aux_panels(self) -> list[AuxPanelResult]:
        if not self.env:
            raise RuntimeError("Engine is not initialized.")
        if not self.aux_panels:
            return []
        collected: list[AuxPanelResult] = []
        for panel in self.aux_panels:
            if not isinstance(panel, AuxPanelSpec):
                continue
            if not panel.title or not panel.command:
                continue
            observation, _, _, _ = self.env.step(panel.command)
            collected.append(
                AuxPanelResult(title=panel.title, command=panel.command, content=observation)
            )
        return collected
