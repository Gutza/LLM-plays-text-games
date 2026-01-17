import argparse
import sys
from pathlib import Path
from wakepy import keep

from src.engine import (
    DEFAULT_AUX_PANELS,
    GameEngine,
    format_aux_snapshot,
    get_panel_content,
)
from src.llm import LLMManager, build_default_llm_config, load_llm_config
from src.savegame import (
    append_step_with_aux,
    create_new_log,
    get_ltm_summary_and_stm_steps,
    get_latest_strategy,
    load_log,
    maybe_insert_ltm_summary,
    should_refresh_strategy,
    write_log_atomic,
)

MAX_REPEAT_COUNT = 7
STRATEGY_LENGTH = 7


def load_strategy_guide(game_name: str) -> str:
    if not game_name:
        return ""
    strategy_path = Path("strategies") / f"{game_name}.md"
    if not strategy_path.exists():
        return ""
    return strategy_path.read_text(encoding="utf-8").strip()


def build_strategy_context(guide_text: str, latest_strategy: str) -> str:
    guide = guide_text.strip() if isinstance(guide_text, str) else ""
    latest = latest_strategy.strip() if isinstance(latest_strategy, str) else ""

    result = ""
    if guide:
        result += f"Strategy guide:\n{guide}\n\n"
    if latest:
        result += f"Latest short-term strategy:\n{latest}\n\n"
    return result.strip()


def build_strategy_aux_context(aux_snapshot) -> str:
    if not aux_snapshot:
        return ""
    environment = get_panel_content(aux_snapshot, "ENVIRONMENT").strip()
    inventory = get_panel_content(aux_snapshot, "INVENTORY").strip()
    if not environment and not inventory:
        return ""
    parts: list[str] = []
    if environment:
        parts.append(f"Environment:\n{environment}")
    if inventory:
        parts.append(f"Inventory:\n{inventory}")
    return "\n\n".join(parts)

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
    parser.add_argument(
        "--ignore-strategy",
        action="store_true",
        help="Ignore the strategy guide and start with an empty strategy; useful for creative exploration",
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

    strategy_guide = ""
    if not args.ignore_strategy:
        strategy_guide = load_strategy_guide(game_path.stem)

    llm_defaults = build_default_llm_config(
        gameplay_model="gpt-5-mini",
        summary_model="gpt-5-mini",
        strategy_model="gpt-5-mini",
        post_mortem_model="gpt-5-mini",
    )
    llm_config = load_llm_config(Path("llm_config.json"), llm_defaults)
    llm_manager = LLMManager(
        game_name=game_path.stem,
        gameplay=llm_config.gameplay,
        summary=llm_config.summary,
        strategy=llm_config.strategy,
        post_mortem=llm_config.post_mortem,
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

        ltm_summary, stm_steps, _ = get_ltm_summary_and_stm_steps(
            log_data.get("steps", [])
        )
        current_strategy, _ = get_latest_strategy(log_data.get("steps", []))
        preemptive_strategy = ""
        initial_aux_snapshot = engine.build_aux_snapshot(include_aux_snapshot=True)
        strategy_messages = llm_manager.strategy_agent.build_messages(
            ltm_summary=ltm_summary,
            stm_steps=stm_steps,
        )
        strategy_context = build_strategy_context(strategy_guide, current_strategy)
        strategy_aux_context = build_strategy_aux_context(initial_aux_snapshot)
        if strategy_aux_context:
            strategy_messages.append(
                {"role": "user", "content": strategy_aux_context}
            )
        preemptive_strategy, _ = llm_manager.strategy_agent.get_strategy(
            strategy_messages,
            strategy_context,
        )
        if preemptive_strategy:
            print("~" * 80)
            print("Initial strategy:")
            print(preemptive_strategy)
            print("~" * 80)

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
                current_strategy, _ = get_latest_strategy(log_data.get("steps", []))
                if not current_strategy and preemptive_strategy:
                    current_strategy = preemptive_strategy
                strategy_to_store = None
                if should_refresh_strategy(
                    log_data.get("steps", []),
                    strategy_length=STRATEGY_LENGTH,
                ):
                    strategy_messages = llm_manager.strategy_agent.build_messages(
                        ltm_summary=ltm_summary,
                        stm_steps=stm_steps,
                    )
                    strategy_context = build_strategy_context(
                        strategy_guide, current_strategy
                    )
                    strategy_aux_context = build_strategy_aux_context(aux_snapshot)
                    if strategy_aux_context:
                        strategy_messages.append(
                            {"role": "user", "content": strategy_aux_context}
                        )
                    current_strategy, _ = llm_manager.strategy_agent.get_strategy(
                        strategy_messages,
                        strategy_context,
                    )
                    if current_strategy:
                        print("~" * 80)
                        print("Updated strategy:")
                        print(current_strategy)
                        print("~" * 80)
                        strategy_to_store = current_strategy
                messages = llm_manager.gameplay_agent.build_messages(
                    ltm_summary=ltm_summary,
                    stm_steps=stm_steps,
                    current_observation=engine.get_observation_for_llm(),
                    temporary_snapshot=temporary_snapshot,
                    strategy=current_strategy,
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

            step_data = append_step_with_aux(
                log_data,
                actor,
                step_result.command,
                step_result.augmented_observation,
                step_result.reward,
                step_result.done,
                step_result.info,
                step_result.aux_snapshot,
                strategy=strategy_to_store if actor == "llm" else None,
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

            if step_result.repeat_count > MAX_REPEAT_COUNT:
                print("~" * 80)
                print("Repeat count exceeded; handing over control to the human.")
                print("~" * 80)
                human_mode = True
                continue
    finally:
        engine.close()


if __name__ == "__main__":
    # with keep.running(): # Not working in WSL2
        main()
