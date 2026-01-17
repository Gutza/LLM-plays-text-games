import argparse
import os
import sys
from pathlib import Path

from openai import OpenAI

from src.engine import DEFAULT_AUX_PANELS, GameEngine, format_aux_snapshot
from src.llm import LLMManager
from src.savegame import (
    append_step_with_aux,
    create_new_log,
    get_ltm_summary_and_stm_steps,
    load_log,
    maybe_insert_ltm_summary,
    write_log_atomic,
)


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


def resolve_seed(log_data, default_seed: int) -> int:
    if not isinstance(log_data, dict):
        return default_seed
    meta = log_data.get("meta")
    if not isinstance(meta, dict):
        raise ValueError("Savegame meta is required (backward compatibility disabled).")
    seed = meta.get("seed")
    if not isinstance(seed, int):
        raise ValueError("Savegame seed is required (backward compatibility disabled).")
    return seed


def main():
    args = parse_args()
    game_path = resolve_game_path(args.game)
    default_seed = 42

    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    llm_manager = LLMManager(
        client,
        game_name=game_path.stem,
        gameplay_model="gpt-5-mini",
        summary_model="gpt-5-mini",
    )

    log_path = None
    log_data = None
    if args.load:
        log_path, log_data = load_log(args.load)
        game_seed = resolve_seed(log_data, default_seed)
    else:
        game_seed = default_seed

    engine = GameEngine(game_path, seed=game_seed, aux_panels=DEFAULT_AUX_PANELS)
    try:
        engine.initialize()
        if args.load:
            try:
                engine.replay(log_data.get("steps", []))
            except RuntimeError as exc:
                print(str(exc))
                sys.exit(1)
        else:
            log_path, log_data = create_new_log(
                game_path,
                llm_manager.gameplay_model,
                engine.observation,
                seed=game_seed,
            )

        if not log_data.get("initial_observation"):
            log_data["initial_observation"] = engine.observation
            write_log_atomic(log_path, log_data)

        print(engine.observation)
        if engine.done:
            return

        human_mode = args.human
        while True:
            aux_snapshot = engine.build_aux_snapshot(include_aux_snapshot=True)
            if engine.done:
                return

            print("\n" + "=" * 70 + "\n")
            temporary_snapshot = format_aux_snapshot(
                aux_snapshot,
                header="Game state: ",
            )
            print(temporary_snapshot)

            print("\n" + "- " * 35 + "\n")

            llm_journal = None
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
                ltm_summary, stm_steps, _ = get_ltm_summary_and_stm_steps(
                    log_data.get("steps", [])
                )
                messages = llm_manager.gameplay_agent.build_messages(
                    ltm_summary=ltm_summary,
                    stm_steps=stm_steps,
                    current_observation=engine.get_observation_for_llm(),
                    temporary_snapshot=temporary_snapshot,
                )
                command, llm_journal = llm_manager.gameplay_agent.get_action(messages)
                print(f"LLM> {command}")
                actor = "llm"

            if not command or not command.strip():
                print("Empty command; please try again.")
                continue

            step_result = engine.step(
                command,
                llm_journal if actor == "llm" else None,
            )
            print(step_result.augmented_observation)

            append_step_with_aux(
                log_data,
                actor,
                step_result.command,
                step_result.augmented_observation,
                step_result.reward,
                step_result.done,
                step_result.info,
                step_result.aux_snapshot,
                llm_journal=step_result.llm_journal if actor == "llm" else None,
            )
            updated_summary = maybe_insert_ltm_summary(
                log_data,
                summarizer=llm_manager.get_summarizer(),
            )
            if updated_summary:
                print("~" * 80)
                print("Updated summary:")
                print(updated_summary)
                print("~" * 80)

            write_log_atomic(log_path, log_data)

            if step_result.done:
                return
    finally:
        engine.close()


if __name__ == "__main__":
    main()
