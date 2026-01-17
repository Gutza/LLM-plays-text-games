import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from .models import AuxSnapshot

MAX_STM_SIZE = 30
LTM_BATCH_SIZE = 10
SUMMARY_ACTOR = "summary"


def ensure_savegames_dir() -> Path:
    save_dir = Path("savegames")
    save_dir.mkdir(parents=True, exist_ok=True)
    return save_dir


def game_stem(game_path: Path | str) -> str:
    return Path(game_path).stem


def timestamp_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def make_json_safe(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): make_json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [make_json_safe(v) for v in value]
    return str(value)


def write_log_atomic(log_path: Path, log_data: dict[str, Any]) -> None:
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


def get_latest_strategy(
    steps: list[dict[str, Any]] | None,
) -> tuple[str, int]:
    if not steps:
        return "", -1
    latest_strategy = ""
    latest_index = -1
    for idx, step in enumerate(steps):
        if not isinstance(step, dict):
            continue
        strategy = step.get("strategy")
        if not isinstance(strategy, str):
            continue
        strategy_text = strategy.strip()
        if not strategy_text:
            continue
        latest_strategy = strategy_text
        latest_index = idx
    return latest_strategy, latest_index


def count_non_summary_steps_since(
    steps: list[dict[str, Any]] | None,
    last_index: int,
) -> int:
    if not steps:
        return 0
    count = 0
    for idx in range(last_index + 1, len(steps)):
        step = steps[idx]
        if not isinstance(step, dict):
            continue
        if is_summary_step(step):
            continue
        count += 1
    return count


def should_refresh_strategy(
    steps: list[dict[str, Any]] | None,
    *,
    strategy_length: int,
) -> bool:
    if not steps:
        return False
    if strategy_length <= 0:
        return False
    _, latest_index = get_latest_strategy(steps)
    steps_since = count_non_summary_steps_since(steps, latest_index)
    return (steps_since + 1) >= strategy_length


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


def build_summary_prompts(
    current_summary: str,
    batch_steps: list[dict[str, Any]],
) -> tuple[str, str]:
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
    return system_prompt, user_prompt


def summarize_ltm_batch(
    summarizer: Callable[[str, str], str],
    *,
    current_summary: str,
    batch_steps: list[dict[str, Any]],
) -> str:
    if not batch_steps:
        return current_summary
    system_prompt, user_prompt = build_summary_prompts(current_summary, batch_steps)
    summary_text = summarizer(system_prompt, user_prompt)
    if not isinstance(summary_text, str):
        summary_text = ""
    summary_text = summary_text.strip()
    if summary_text:
        return summary_text
    return current_summary.strip() if isinstance(current_summary, str) else ""


def maybe_insert_ltm_summary(
    log_data: dict[str, Any],
    *,
    summarizer: Callable[[str, str], str],
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
        summarizer,
        current_summary=summary_text,
        batch_steps=batch_steps,
    )
    summary_step = build_summary_step(updated_summary, batch_steps)
    summary_step["i"] = next_step_id(steps)
    steps.insert(insert_index, summary_step)
    return updated_summary


def create_new_log(
    game_path: Path,
    model_name: str,
    initial_observation: str,
    *,
    seed: int,
) -> tuple[Path, dict[str, Any]]:
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


def load_log(log_path: str | Path) -> tuple[Path, dict[str, Any]]:
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
    log_data: dict[str, Any],
    actor: str,
    command: str,
    observation: str,
    reward: int | float,
    done: bool,
    info: Any,
    aux: AuxSnapshot | None,
    *,
    strategy: str | None = None,
    llm_journal: dict[str, Any] | None = None,
) -> None:
    steps = log_data.setdefault("steps", [])
    aux_dict = aux.to_dict() if aux is not None else None
    step_data = {
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
    if isinstance(strategy, str):
        strategy_text = strategy.strip()
        if strategy_text:
            step_data["strategy"] = strategy_text
    steps.append(step_data)
