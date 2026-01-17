import argparse
import sys
from pathlib import Path
from wakepy import keep

from dataclasses import replace

from src.llm import LLMManager, build_default_llm_config, load_llm_config
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
        description="Analyze savegames for a game and augment a persistent strategy file."
    )
    parser.add_argument(
        "game",
        help="Game name (no extension), e.g. zork1.",
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


def read_processed_index(index_path: Path) -> set[str]:
    if not index_path.exists():
        return set()
    lines = index_path.read_text(encoding="utf-8").splitlines()
    return {line.strip() for line in lines if line.strip()}


def append_processed_index(index_path: Path, filename: str) -> None:
    if not filename:
        return
    with index_path.open("a", encoding="utf-8") as handle:
        handle.write(filename)
        handle.write("\n")


def collect_candidate_savegames(save_dir: Path, game_name: str) -> list[Path]:
    if not save_dir.exists():
        return []
    prefix = f"{game_name}_"
    candidates = [
        path
        for path in save_dir.iterdir()
        if path.is_file()
        and path.suffix == ".json"
        and path.name.startswith(prefix)
    ]
    return sorted(candidates, key=lambda path: path.name)


def process_savegame(
    savegame_path: Path,
    *,
    llm_manager: LLMManager,
    existing_strategy: str,
) -> str:
    print(f"Processing savegame: {savegame_path.name}")
    _, log_data = load_log(savegame_path)
    steps = log_data.get("steps")
    if not isinstance(steps, list):
        print(f"Savegame format is invalid (steps must be a list): {savegame_path}")
        return existing_strategy

    non_summary_steps = collect_non_summary_steps(steps)
    if not non_summary_steps:
        print(f"No non-summary steps found; skipping: {savegame_path.name}")
        return existing_strategy

    ltm_summary, _, _ = get_ltm_summary_and_stm_steps(steps)
    total_steps = len(non_summary_steps)
    batch_size = ANALYSIS_BATCH_SIZE
    updated_strategy = existing_strategy
    for batch_start in range(0, total_steps, batch_size):
        batch_end = min(batch_start + batch_size, total_steps)
        batch = non_summary_steps[batch_start:batch_end]
        first_original_index = batch[0][0] if batch else -1
        initial_strategy = find_latest_strategy_before_or_at(steps, first_original_index)
        start_index = batch_start + 1
        end_index = batch_end
        context_line = (
            "These are steps "
            f"{start_index}...{end_index} of a total of {total_steps} "
            f"in this playthrough."
        )
        batch_text = format_batch_steps(
            batch,
            start_index=start_index,
            initial_strategy=initial_strategy,
        )
        messages = llm_manager.post_mortem_agent.build_messages(
            context_line=context_line,
            ltm_summary=ltm_summary,
            batch_text=batch_text,
            existing_strategy=updated_strategy,
        )
        update_text, _ = llm_manager.post_mortem_agent.get_strategy(messages)
        if not update_text:
            continue
        updated_strategy = update_text.strip()
    return updated_strategy


def main() -> None:
    args = parse_args()
    game_name = args.game.strip()
    if not game_name:
        print("Game name is required.")
        sys.exit(1)

    model_name = (
        args.model.strip()
        if isinstance(args.model, str) and args.model
        else "gpt-5.1"
    )
    llm_defaults = build_default_llm_config(
        gameplay_model=model_name,
        summary_model=model_name,
        strategy_model=model_name,
        post_mortem_model=model_name,
    )
    llm_config = load_llm_config(Path("llm_config.json"), llm_defaults)
    if args.model:
        llm_config = replace(
            llm_config,
            post_mortem=replace(llm_config.post_mortem, model=model_name),
        )
    llm_manager = LLMManager(
        game_name=game_name,
        gameplay=llm_config.gameplay,
        summary=llm_config.summary,
        strategy=llm_config.strategy,
        post_mortem=llm_config.post_mortem,
    )

    strategy_dir = Path("strategies")
    strategy_dir.mkdir(parents=True, exist_ok=True)
    strategy_path = strategy_dir / f"{game_name}.md"
    index_path = strategy_dir / f"{game_name}.csv"
    existing_strategy = load_existing_strategy(strategy_path)

    save_dir = Path("savegames")
    candidates = collect_candidate_savegames(save_dir, game_name)
    if not candidates:
        print(f"No savegames found for {game_name} in {save_dir}.")
        return

    processed = read_processed_index(index_path)
    pending = [path for path in candidates if path.name not in processed]
    if not pending:
        print(f"No new savegames to process for {game_name}.")
        return

    for savegame_path in pending:
        updated_strategy = process_savegame(
            savegame_path,
            llm_manager=llm_manager,
            existing_strategy=existing_strategy,
        )
        if updated_strategy:
            write_strategy(strategy_path, updated_strategy)
            existing_strategy = updated_strategy
        append_processed_index(index_path, savegame_path.name)


if __name__ == "__main__":
    # with keep.running(): # Not working in WSL2
        main()
