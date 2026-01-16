import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from jericho import FrotzEnv
from openai import OpenAI

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


DEFAULT_AUX_PANELS = [
    AuxPanelSpec(title="INVENTORY", command="inventory"),
    AuxPanelSpec(title="LOOK AROUND", command="look"),
]


def parse_args():
    parser = argparse.ArgumentParser(description="Play Z-machine games via LLM.")
    parser.add_argument(
        "--game",
        required=True,
        help="Game image filename (e.g. games/zork1.z5)",
    )
    parser.add_argument(
        "--load",
        help="Path to a savegame JSON log to replay and then append to",
    )
    parser.add_argument(
        "--human",
        action="store_true",
        help="Start in human mode; type LLM to hand over control",
    )
    return parser.parse_args()


def resolve_game_path(game_filename):
    if not game_filename:
        raise ValueError("Game filename is required.")
    game_path = Path(game_filename)
    if not game_path.exists():
        raise FileNotFoundError(f"Game image not found: {game_path}")
    return game_path


def ensure_savegames_dir():
    save_dir = Path("savegames")
    save_dir.mkdir(parents=True, exist_ok=True)
    return save_dir


def game_stem(game_path):
    return Path(game_path).stem


def timestamp_now():
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def iso_now():
    return datetime.now(timezone.utc).isoformat()


def make_json_safe(value):
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): make_json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [make_json_safe(v) for v in value]
    return str(value)


def write_log_atomic(log_path, log_data):
    tmp_path = log_path.with_suffix(log_path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(log_data, handle, indent=2, ensure_ascii=True)
    os.replace(tmp_path, log_path)


def create_new_log(game_path, model_name, initial_observation):
    save_dir = ensure_savegames_dir()
    file_name = f"{game_stem(game_path)}_{timestamp_now()}.json"
    log_path = save_dir / file_name
    log_data = {
        "meta": {
            "game_file": str(game_path),
            "created_at": iso_now(),
            "model": model_name,
        },
        "initial_observation": initial_observation,
        "steps": [],
    }
    write_log_atomic(log_path, log_data)
    return log_path, log_data


def load_log(log_path):
    path = Path(log_path)
    if not path.exists():
        raise FileNotFoundError(f"Savegame not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError("Savegame format is invalid (expected object).")
    data.setdefault("steps", [])
    if not isinstance(data["steps"], list):
        raise ValueError("Savegame format is invalid (steps must be a list).")
    return path, data


def append_step_with_aux(
    log_data, actor, command, observation, reward, done, info, aux: AuxSnapshot | None
):
    steps = log_data.setdefault("steps", [])
    aux_dict = aux.to_dict() if aux is not None else None
    steps.append(
        {
            "i": len(steps),
            "actor": actor,
            "command": command,
            "observation": observation,
            "reward": reward,
            "done": done,
            "info": make_json_safe(info),
            "aux": make_json_safe(aux_dict),
            "ts": iso_now(),
        }
    )


class LLMAgent:
    def __init__(self, client, game_name):
        self.client = client
        self.model = "gpt-5-mini"
        self.game_history = [
            {
                "role": "system",
                "content": (
                    f"You are playing the game {game_name}. You are the player and you "
                    "are trying to win the game. Commands should be short and concise "
                    "(never more than one short sentence, typically just one or two "
                    "words). You are connected directly to the game engine."
                ),
            }
        ]

    def get_action(self, observation, *, temporary_snapshot=None):
        self.game_history.append({"role": "user", "content": observation})
        messages = list(self.game_history)
        if temporary_snapshot:
            messages.append({"role": "user", "content": temporary_snapshot})
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
        )
        model_response = response.choices[0].message.content
        self.game_history.append({"role": "assistant", "content": model_response})
        return model_response


def extract_aux_meta(info, *, reward, done):
    moves = None
    score = None
    if isinstance(info, dict):
        if "moves" in info:
            moves = info.get("moves")
        if "score" in info:
            score = info.get("score")
    return AuxMeta(moves=moves, score=score, reward=reward, done=done)


def format_aux_snapshot(aux, *, header):
    parts = [header, ""]
    meta = aux.meta.to_dict()
    if meta:
        parts.append("META:")
        for key in sorted(meta.keys()):
            parts.append(f"- {key}: {meta.get(key)}")
        parts.append("")
    for panel in aux.panels:
        title = panel.title or "PANEL"
        content = (panel.content or "").rstrip()
        parts.append(f"{title}:")
        parts.append(content)
        parts.append("")
    return "\n".join(parts).rstrip() + "\n"


def collect_aux_panels(env, panels):
    if not panels:
        return []
    collected: list[AuxPanelResult] = []
    for panel in panels:
        if not isinstance(panel, AuxPanelSpec):
            continue
        if not panel.title or not panel.command:
            continue
        observation, _, _, _ = env.step(panel.command)
        collected.append(
            AuxPanelResult(title=panel.title, command=panel.command, content=observation)
        )
    return collected


def simulate_from_scratch(game_path, commands, *, aux_panels, include_aux_snapshot):
    env = FrotzEnv(str(game_path))
    try:
        observation, info = env.reset()
        reward = 0
        done = False
        executed = 0

        for command in commands:
            executed += 1
            observation, reward, done, info = env.step(command)
            if done:
                break

        aux = AuxSnapshot(meta=extract_aux_meta(info, reward=reward, done=done))
        if include_aux_snapshot and not done:
            aux = AuxSnapshot(
                meta=aux.meta,
                panels=collect_aux_panels(env, aux_panels),
            )

        return observation, reward, done, info, aux, executed
    finally:
        close_fn = getattr(env, "close", None)
        if callable(close_fn):
            close_fn()


def main():
    args = parse_args()
    game_path = resolve_game_path(args.game)

    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    llm_agent = LLMAgent(client, game_stem(game_path))

    if args.load:
        log_path, log_data = load_log(args.load)
    else:
        env = FrotzEnv(str(game_path))
        try:
            observation, info = env.reset()
        finally:
            close_fn = getattr(env, "close", None)
            if callable(close_fn):
                close_fn()
        log_path, log_data = create_new_log(game_path, llm_agent.model, observation)

    if not log_data.get("initial_observation"):
        env = FrotzEnv(str(game_path))
        try:
            observation, info = env.reset()
        finally:
            close_fn = getattr(env, "close", None)
            if callable(close_fn):
                close_fn()
        log_data["initial_observation"] = observation

    command_queue = []
    for step in log_data.get("steps", []):
        command = step.get("command", "")
        if not isinstance(command, str) or not command.strip():
            continue
        command_queue.append(command)

    observation, _, done, _, _, executed = simulate_from_scratch(
        game_path,
        command_queue,
        aux_panels=DEFAULT_AUX_PANELS,
        include_aux_snapshot=False,
    )
    if done and executed < len(command_queue):
        print("Replay ended early; savegame does not match engine state.")
        sys.exit(1)

    print(observation)
    if done:
        return

    human_mode = args.human

    while True:
        observation, _, done, _, aux, _ = simulate_from_scratch(
            game_path,
            command_queue,
            aux_panels=DEFAULT_AUX_PANELS,
            include_aux_snapshot=True,
        )
        if done:
            return

        if human_mode:
            print(
                format_aux_snapshot(
                    aux,
                    header="[AUXILIARY STATE - DEBUG VIEW - NOT RECORDED IN HISTORY]",
                )
            )
            command = input("Your command (type LLM to hand over): ").strip()
            if not command:
                print("Please enter a command.")
                continue
            if command.upper() == "LLM":
                human_mode = False
                print("Handing over control to LLM.")
                continue
            actor = "human"
        else:
            temporary_snapshot = format_aux_snapshot(
                aux,
                header="[TEMPORARY SNAPSHOT - DO NOT ADD TO HISTORY]",
            )
            command = llm_agent.get_action(
                observation, temporary_snapshot=temporary_snapshot
            )
            print(f"LLM command: {command}")
            actor = "llm"

        print("<" * 70)
        if not command or not command.strip():
            print("Empty command; please try again.")
            print(">" * 70)
            continue

        command_queue.append(command)
        observation, reward, done, info, aux_post, executed = simulate_from_scratch(
            game_path,
            command_queue,
            aux_panels=DEFAULT_AUX_PANELS,
            include_aux_snapshot=True,
        )
        if done and executed < len(command_queue):
            print("Engine ended early; command history is inconsistent.")
            sys.exit(1)
        print(observation)
        print(f"Reward: {reward}")
        print(f"Done: {done}")
        print(f"Info: {info}")
        print(">" * 70)

        append_step_with_aux(log_data, actor, command, observation, reward, done, info, aux_post)
        write_log_atomic(log_path, log_data)

        if done:
            return


if __name__ == "__main__":
    main()