import argparse
import os
import sys
from pathlib import Path
from wakepy import keep

from openai import OpenAI

from src.llm import LLMManager
from src.savegame import (
    game_stem,
    get_latest_strategy,
    get_ltm_summary_and_stm_steps,
    is_summary_step,
    load_log,
)

ANALYSIS_BATCH_SIZE = 50


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze a savegame and augment a persistent strategy file."
    )
    parser.add_argument(
        "savegame",
        help="Full path to a savegame JSON file.",
    )
    parser.add_argument(
        "--model",
        help="Override the model used for post-mortem strategy updates.",
    )
    return parser.parse_args()


def get_created_at(log_data: dict) -> str:
    meta = log_data.get("meta")
    if not isinstance(meta, dict):
        return "unknown"
    created_at = meta.get("created_at")
    if not isinstance(created_at, str) or not created_at.strip():
        return "unknown"
    return created_at.strip()


def get_game_name(log_path: Path, log_data: dict) -> str:
    meta = log_data.get("meta")
    if isinstance(meta, dict):
        game_file = meta.get("game_file")
        if isinstance(game_file, str) and game_file.strip():
            return game_stem(game_file)
    return log_path.stem


def collect_non_summary_steps(steps: list[dict]) -> list[tuple[int, dict]]:
    if not steps:
        return []
    return [
        (idx, step)
        for idx, step in enumerate(steps)
        if isinstance(step, dict) and not is_summary_step(step)
    ]


def normalize_strategy_text(strategy: object) -> str:
    if not isinstance(strategy, str):
        return ""
    return strategy.strip()


def find_latest_strategy_before_or_at(steps: list[dict], start_index: int) -> str:
    if not steps:
        return ""
    if start_index < 0:
        return ""
    for idx in range(start_index, -1, -1):
        step = steps[idx]
        if not isinstance(step, dict):
            continue
        strategy_text = normalize_strategy_text(step.get("strategy"))
        if strategy_text:
            return strategy_text
    return ""


def format_batch_steps(
    batch: list[tuple[int, dict]],
    *,
    start_index: int,
    initial_strategy: str,
) -> str:
    if not batch:
        return ""
    parts: list[str] = []
    current_strategy = normalize_strategy_text(initial_strategy)
    if current_strategy:
        parts.append("Strategy in effect at batch start:")
        parts.append(current_strategy)
        parts.append("")
    for offset, (_, step) in enumerate(batch):
        command = step.get("command", "")
        observation = step.get("observation", "")
        step_strategy = normalize_strategy_text(step.get("strategy"))
        if step_strategy and step_strategy != current_strategy:
            parts.append("Strategy update:")
            parts.append(step_strategy)
            parts.append("")
            current_strategy = step_strategy
        command_text = command.strip() if isinstance(command, str) else str(command)
        observation_text = (
            observation.strip() if isinstance(observation, str) else str(observation)
        )
        step_number = start_index + offset
        if command_text:
            parts.append(f"{step_number}. Command: {command_text}")
        if observation_text:
            parts.append(f"Observation: {observation_text}")
        parts.append("")
    return "\n".join(parts).strip()


def load_existing_strategy(strategy_path: Path) -> str:
    if not strategy_path.exists():
        return ""
    return strategy_path.read_text(encoding="utf-8").strip()


def write_strategy(strategy_path: Path, content: str) -> None:
    if not content.strip():
        return
    with strategy_path.open("w", encoding="utf-8") as handle:
        handle.write(content.strip())
        handle.write("\n")


def main() -> None:
    args = parse_args()
    savegame_path, log_data = load_log(args.savegame)
    steps = log_data.get("steps")
    if not isinstance(steps, list):
        print("Savegame format is invalid (steps must be a list).")
        sys.exit(1)

    non_summary_steps = collect_non_summary_steps(steps)
    if not non_summary_steps:
        print("No non-summary steps found; nothing to analyze.")
        return

    created_at = get_created_at(log_data)
    game_name = get_game_name(savegame_path, log_data)

    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    model_name = args.model.strip() if isinstance(args.model, str) and args.model else "gpt-5.1"
    llm_manager = LLMManager(
        client,
        game_name=game_name,
        gameplay_model=model_name,
        summary_model=model_name,
        strategy_model=model_name,
    )

    ltm_summary, _, _ = get_ltm_summary_and_stm_steps(steps)
    strategy_dir = Path("strategies")
    strategy_dir.mkdir(parents=True, exist_ok=True)
    strategy_path = strategy_dir / f"{game_name}.md"
    existing_strategy = load_existing_strategy(strategy_path)

    total_steps = len(non_summary_steps)
    batch_size = ANALYSIS_BATCH_SIZE
    for batch_start in range(0, total_steps, batch_size):
        batch_end = min(batch_start + batch_size, total_steps)
        batch = non_summary_steps[batch_start:batch_end]
        first_original_index = batch[0][0] if batch else -1
        initial_strategy = find_latest_strategy_before_or_at(steps, first_original_index)
        start_index = batch_start + 1
        end_index = batch_end
        context_line = (
            "These are steps "
            f"{start_index}...{end_index} of {total_steps} "
            f"from a playthrough which started on {created_at}."
        )
        batch_text = format_batch_steps(
            batch,
            start_index=start_index,
            initial_strategy=initial_strategy,
        )
        post_mortem_context = existing_strategy
        messages = llm_manager.post_mortem_agent.build_messages(
            context_line=context_line,
            ltm_summary=ltm_summary,
            batch_text=batch_text,
            existing_strategy=post_mortem_context,
        )
        update_text, _ = llm_manager.post_mortem_agent.get_strategy(messages)
        if not update_text:
            continue
        write_strategy(strategy_path, update_text)
        existing_strategy = update_text.strip()


if __name__ == "__main__":
    # with keep.running(): # Not working in WSL2
        main()
