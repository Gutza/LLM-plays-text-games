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


DEFAULT_AUX_PANELS = [
    AuxPanelSpec(title="INVENTORY", command="inventory"),
    AuxPanelSpec(title="ENVIRONMENT", command="look"),
]

MAX_STM_SIZE = 30
LTM_BATCH_SIZE = 10
SUMMARY_ACTOR = "summary"


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
        json.dump(log_data, handle, indent=2, ensure_ascii=False)
    os.replace(tmp_path, log_path)


def is_summary_step(step: dict[str, Any] | None) -> bool:
    if not isinstance(step, dict):
        return False
    if step.get("actor") != SUMMARY_ACTOR:
        return False
    return isinstance(step.get("summary"), str)


def get_ltm_summary_and_stm_steps(
    steps: list[dict[str, Any]] | None,
) -> tuple[str, list[dict[str, Any]], int]:
    if not steps:
        return "", [], -1
    summary_text = ""
    last_summary_index = -1
    for idx, step in enumerate(steps):
        if not is_summary_step(step):
            continue
        summary_value = step.get("summary")
        summary_text = summary_value if isinstance(summary_value, str) else ""
        last_summary_index = idx
    stm_steps = [
        step
        for step in steps[last_summary_index + 1 :]
        if isinstance(step, dict) and not is_summary_step(step)
    ]
    return summary_text, stm_steps, last_summary_index


def collect_stm_batch(
    steps: list[dict[str, Any]],
    last_summary_index: int,
    batch_size: int,
) -> tuple[list[dict[str, Any]], int]:
    batch: list[dict[str, Any]] = []
    if batch_size <= 0:
        return batch, len(steps)
    insert_index = len(steps)
    count = 0
    for idx in range(last_summary_index + 1, len(steps)):
        step = steps[idx]
        if is_summary_step(step):
            continue
        batch.append(step)
        count += 1
        if count >= batch_size:
            insert_index = idx + 1
            break
    return batch, insert_index


def format_turns_for_summary(steps: list[dict[str, Any]]) -> str:
    if not steps:
        return ""
    parts: list[str] = []
    for step in steps:
        command = step.get("command", "")
        observation = step.get("observation", "")
        actor = step.get("actor", "")
        command_text = command.strip() if isinstance(command, str) else str(command)
        observation_text = (
            observation.strip() if isinstance(observation, str) else str(observation)
        )
        actor_text = actor.strip() if isinstance(actor, str) else ""
        if command_text:
            if actor_text:
                parts.append(f"Command ({actor_text}): {command_text}")
            else:
                parts.append(f"Command: {command_text}")
        if observation_text:
            parts.append(f"Observation: {observation_text}")
        parts.append("")
    return "\n".join(parts).strip()


def format_recent_turns(steps: list[dict[str, Any]]) -> str:
    if not steps:
        return ""
    parts = ["Recent turns (oldest to newest):"]
    for idx, step in enumerate(steps, start=1):
        command = step.get("command", "")
        observation = step.get("observation", "")
        actor = step.get("actor", "")
        command_text = command.strip() if isinstance(command, str) else str(command)
        observation_text = (
            observation.strip() if isinstance(observation, str) else str(observation)
        )
        actor_text = actor.strip() if isinstance(actor, str) else ""
        if command_text:
            if actor_text:
                parts.append(f"{idx}. Command ({actor_text}): {command_text}")
            else:
                parts.append(f"{idx}. Command: {command_text}")
        if observation_text:
            parts.append(f"Observation: {observation_text}")
        parts.append("")
    return "\n".join(parts).strip()


def build_summary_step(summary_text: str, batch_steps: list[dict[str, Any]]) -> dict[str, Any]:
    summarized_step_ids = [
        step_id
        for step in batch_steps
        for step_id in [step.get("i")]
        if isinstance(step_id, int)
    ]
    return {
        "i": None,
        "actor": SUMMARY_ACTOR,
        "summary": summary_text,
        "summarized_step_ids": summarized_step_ids,
        "summarized_step_count": len(batch_steps),
        "ts": iso_now(),
    }


def next_step_id(steps: list[dict[str, Any]]) -> int:
    step_ids = [step.get("i") for step in steps if isinstance(step.get("i"), int)]
    if not step_ids:
        return 0
    return max(step_ids) + 1


def summarize_ltm_batch(
    client,
    *,
    model: str,
    current_summary: str,
    batch_steps: list[dict[str, Any]],
) -> str:
    if not batch_steps:
        return current_summary
    system_prompt = (
        "You are the chronicler of a text adventure game. Your job is to update the "
        "story so far. You will be given a 'Previous Summary' and a sequence of "
        "'New Game Turns'.\n"
        "1. Incorporate the significant events from the New Game Turns into the "
        "Previous Summary.\n"
        "2. Drop trivial details (e.g., typos, 'look' commands that revealed "
        "nothing new, failed movement attempts).\n"
        "3. Maintain a coherent narrative flow.\n"
        "4. Output ONLY the updated summary."
    )
    turns_text = format_turns_for_summary(batch_steps)
    user_prompt = (
        "Previous Summary:\n"
        f"{current_summary.strip() if isinstance(current_summary, str) else ''}\n\n"
        "New Game Turns:\n"
        f"{turns_text}"
    )
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    summary_text = response.choices[0].message.content

    if not isinstance(summary_text, str):
        summary_text = ""
    summary_text = summary_text.strip()
    if summary_text:
        return summary_text
    return current_summary.strip() if isinstance(current_summary, str) else ""


def maybe_insert_ltm_summary(
    log_data,
    *,
    client,
    model: str,
) -> str | None:
    steps = log_data.get("steps")
    if not isinstance(steps, list):
        return None
    summary_text, stm_steps, last_summary_index = get_ltm_summary_and_stm_steps(steps)
    if len(stm_steps) <= MAX_STM_SIZE:
        return None
    batch_steps, insert_index = collect_stm_batch(
        steps, last_summary_index, LTM_BATCH_SIZE
    )
    if len(batch_steps) < LTM_BATCH_SIZE:
        return None
    updated_summary = summarize_ltm_batch(
        client,
        model=model,
        current_summary=summary_text,
        batch_steps=batch_steps,
    )
    summary_step = build_summary_step(updated_summary, batch_steps)
    summary_step["i"] = next_step_id(steps)
    steps.insert(insert_index, summary_step)
    return updated_summary


def create_new_log(game_path, model_name, initial_observation, *, seed: int):
    save_dir = ensure_savegames_dir()
    file_name = f"{game_stem(game_path)}_{timestamp_now()}.json"
    log_path = save_dir / file_name
    log_data = {
        "meta": {
            "game_file": str(game_path),
            "created_at": iso_now(),
            "model": model_name,
            "seed": seed,
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
    log_data,
    actor,
    command,
    observation,
    reward,
    done,
    info,
    aux: AuxSnapshot | None,
    *,
    llm_journal: dict[str, Any] | None = None,
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
            "llm": make_json_safe(llm_journal),
            "ts": iso_now(),
        }
    )


class LLMAgent:
    def __init__(self, client, game_name):
        self.client = client
        self.model = "gpt-5-mini"
        self.system_prompt = (
            f"You are playing the game {game_name}. You are the player and you "
            "are trying to win the game. Commands should be short and concise "
            "(never more than one short sentence, typically just one or two "
            "words). You are connected directly to the game engine."
        )

    @staticmethod
    def _approx_tokens_from_text(text: str) -> int:
        # Heuristic: ~4 chars/token for English-like text. Useful when real usage
        # numbers aren't available.
        if not text:
            return 0
        return max(1, (len(text) + 3) // 4)

    @staticmethod
    def _approx_tokens_from_chars(char_count: int) -> int:
        if not char_count:
            return 0
        return max(1, (int(char_count) + 3) // 4)

    @classmethod
    def _summarize_messages(cls, messages: list[dict[str, str]]) -> dict[str, Any]:
        chars_total = 0
        chars_by_role: dict[str, int] = {}
        for msg in messages:
            role = str(msg.get("role", "unknown"))
            content = msg.get("content", "")
            if not isinstance(content, str):
                content = str(content)
            n = len(content)
            chars_total += n
            chars_by_role[role] = chars_by_role.get(role, 0) + n
        return {
            "messages": len(messages),
            "chars_total": chars_total,
            "chars_by_role": dict(sorted(chars_by_role.items(), key=lambda kv: kv[0])),
            "approx_tokens_total": cls._approx_tokens_from_chars(chars_total),
        }

    def build_messages(
        self,
        *,
        ltm_summary: str,
        stm_steps: list[dict[str, Any]],
        current_observation: str,
        temporary_snapshot: str | None,
    ) -> list[dict[str, str]]:
        messages = [{"role": "system", "content": self.system_prompt}]
        if ltm_summary:
            messages.append(
                {"role": "user", "content": f"Summary so far:\n{ltm_summary}"}
            )
        recent_turns = format_recent_turns(stm_steps)
        if recent_turns:
            messages.append({"role": "user", "content": recent_turns})
        if not recent_turns and current_observation:
            messages.append(
                {"role": "user", "content": f"Current observation:\n{current_observation}"}
            )
        if temporary_snapshot:
            messages.append({"role": "user", "content": temporary_snapshot})
        return messages

    def get_action(self, messages: list[dict[str, str]]):
        if not messages:
            raise ValueError("LLM prompt messages are required.")
        history_summary_before = self._summarize_messages(messages)
        request_summary = history_summary_before
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
        )
        model_response = response.choices[0].message.content
        usage = getattr(response, "usage", None)
        usage_dict: dict[str, Any] | None = None
        if usage is not None:
            usage_dict = {
                "prompt_tokens": getattr(usage, "prompt_tokens", None),
                "completion_tokens": getattr(usage, "completion_tokens", None),
                "total_tokens": getattr(usage, "total_tokens", None),
            }
        history_summary_after = self._summarize_messages(
            messages + [{"role": "assistant", "content": model_response or ""}]
        )
        journal = {
            "model": self.model,
            "request": request_summary,
            "history_before": history_summary_before,
            "history_after": history_summary_after,
            "usage": usage_dict,
        }
        return model_response, journal


def extract_aux_meta(info, *, reward, done, moves_override: int | None = None):
    moves = None
    score = None
    if isinstance(info, dict):
        if "moves" in info:
            moves = info.get("moves")
        if "score" in info:
            score = info.get("score")
    if moves_override is not None:
        moves = moves_override
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


def build_aux_snapshot(
    env,
    *,
    info,
    reward,
    done,
    aux_panels,
    include_aux_snapshot,
    main_observation,
    moves_override: int | None = None,
):
    aux = AuxSnapshot(
        meta=extract_aux_meta(
            info, reward=reward, done=done, moves_override=moves_override
        )
    )
    if include_aux_snapshot and not done:
        saved_state = env.get_state()
        main_text = (main_observation or "").strip()
        aux = AuxSnapshot(
            meta=aux.meta,
            panels=[
                panel
                for panel in collect_aux_panels(env, aux_panels)
                if panel.content and panel.content not in main_text
            ],
        )
        env.set_state(saved_state)
    return aux


def main():
    args = parse_args()
    game_path = resolve_game_path(args.game)
    DEFAULT_GAME_SEED = 42

    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    llm_agent = LLMAgent(client, game_stem(game_path))
    env = None
    try:
        if args.load:
            log_path, log_data = load_log(args.load)
        else:
            env = FrotzEnv(str(game_path), seed=DEFAULT_GAME_SEED)
            try:
                initial_observation, info = env.reset()
            finally:
                close_fn = getattr(env, "close", None)
                if callable(close_fn):
                    close_fn()
            log_path, log_data = create_new_log(
                game_path, llm_agent.model, initial_observation, seed=DEFAULT_GAME_SEED
            )

        game_seed = DEFAULT_GAME_SEED
        meta = log_data.setdefault("meta", {})
        seed_persisted = False
        if isinstance(meta, dict):
            saved_seed = meta.get("seed")
            if isinstance(saved_seed, int):
                game_seed = saved_seed
            else:
                meta["seed"] = game_seed
                seed_persisted = True
        else:
            log_data["meta"] = {"seed": game_seed}
            seed_persisted = True

        env = FrotzEnv(str(game_path), seed=game_seed)
        initial_observation, info = env.reset()

        if not log_data.get("initial_observation"):
            log_data["initial_observation"] = initial_observation
            seed_persisted = True
        if seed_persisted:
            write_log_atomic(log_path, log_data)

        command_queue = []
        for step in log_data.get("steps", []):
            command = step.get("command", "")
            if not isinstance(command, str) or not command.strip():
                continue
            command_queue.append(command)

        observation = initial_observation
        reward = 0
        done = False
        executed = 0
        for command in command_queue:
            executed += 1
            observation, reward, done, info = env.step(command)
            if done:
                break
        if done and executed < len(command_queue):
            print("Replay ended early; savegame does not match engine state.")
            sys.exit(1)

        current_state = env.get_state()

        print(observation)
        if done:
            return

        human_mode = args.human
        repeat_tracker = RepeatActionTracker()
        move_tracker = MoveTracker()
        if isinstance(info, dict):
            initial_moves = info.get("moves")
            if isinstance(initial_moves, int):
                move_tracker.logical_moves = initial_moves
                move_tracker.last_engine_moves = initial_moves

        while True:
            llm_journal: dict[str, Any] | None = None
            env.set_state(current_state)
            aux = build_aux_snapshot(
                env,
                info=info,
                reward=reward,
                done=done,
                aux_panels=DEFAULT_AUX_PANELS,
                include_aux_snapshot=True,
                main_observation=observation,
                moves_override=move_tracker.logical_moves,
            )
            if done:
                return

            print("\n" + "=" * 70 + "\n")

            temporary_snapshot = format_aux_snapshot(
                aux,
                header="Game state: ",
            )
            print(temporary_snapshot)

            print("\n" + "- " * 35 + "\n")

            if human_mode:
                command = input("Your command (type LLM to hand over)> ").strip()
                if not command:
                    print("Please enter a command.")
                    continue
                if command.upper() == "LLM":
                    human_mode = False
                    print("Handing over control to LLM.")
                    continue
                actor = "human"
            else:
                observation_for_llm = (
                    repeat_tracker.last_augmented_observation or observation
                )
                ltm_summary, stm_steps, _ = get_ltm_summary_and_stm_steps(
                    log_data.get("steps", [])
                )
                messages = llm_agent.build_messages(
                    ltm_summary=ltm_summary,
                    stm_steps=stm_steps,
                    current_observation=observation_for_llm,
                    temporary_snapshot=temporary_snapshot,
                )
                command, llm_journal = llm_agent.get_action(messages)
                print(f"LLM> {command}")
                actor = "llm"

            if not command or not command.strip():
                print("Empty command; please try again.")
                continue

            command_queue.append(command)
            env.set_state(current_state)
            observation, reward, done, info = env.step(command)
            engine_moves = None
            if isinstance(info, dict):
                engine_moves = info.get("moves")
            logical_moves = move_tracker.advance(engine_moves)
            state_after = env.get_state()
            aux_post = build_aux_snapshot(
                env,
                info=info,
                reward=reward,
                done=done,
                aux_panels=DEFAULT_AUX_PANELS,
                include_aux_snapshot=True,
                main_observation=observation,
                moves_override=logical_moves,
            )
            env.set_state(state_after)
            current_state = state_after

            environment_text = get_panel_content(aux_post, "ENVIRONMENT").strip()
            inventory_text = get_panel_content(aux_post, "INVENTORY").strip()
            observation_text = (observation or "").strip()
            repeat_key = (command, observation_text, environment_text, inventory_text)
            if repeat_tracker.last_key == repeat_key:
                repeat_tracker.repeat_count += 1
            else:
                repeat_tracker.last_key = repeat_key
                repeat_tracker.repeat_count = 1

            if repeat_tracker.repeat_count > 1:
                augmented_observation = (
                    f"{observation_text}\n"
                    f"(You did the exact same thing {repeat_tracker.repeat_count} times in a row. This is not a warning, just a reminder.)"
                )
            else:
                augmented_observation = observation_text
            repeat_tracker.last_augmented_observation = augmented_observation

            print(augmented_observation)

            append_step_with_aux(
                log_data,
                actor,
                command,
                augmented_observation,
                reward,
                done,
                info,
                aux_post,
                llm_journal=(llm_journal if actor == "llm" else None),
            )
            updated_summary = maybe_insert_ltm_summary(
                log_data,
                client=llm_agent.client,
                model=llm_agent.model,
            )

            if updated_summary:
                print("~" * 80)
                print("Updated summary:")
                print(updated_summary)
                print("~" * 80)

            write_log_atomic(log_path, log_data)

            if done:
                return
    finally:
        close_fn = getattr(env, "close", None)
        if callable(close_fn):
            close_fn()


if __name__ == "__main__":
    main()