import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from jericho import FrotzEnv
from openai import OpenAI

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


def append_step(log_data, actor, command, observation, reward, done, info):
    steps = log_data.setdefault("steps", [])
    steps.append(
        {
            "i": len(steps),
            "actor": actor,
            "command": command,
            "observation": observation,
            "reward": reward,
            "done": done,
            "info": make_json_safe(info),
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

    def get_action(self, observation):
        self.game_history.append({"role": "user", "content": observation})
        response = self.client.chat.completions.create(
            model=self.model,
            messages=self.game_history,
        )
        model_response = response.choices[0].message.content
        self.game_history.append({"role": "assistant", "content": model_response})
        return model_response


def replay_steps(env, steps):
    if not steps:
        return None, False
    observation = None
    done = False
    for index, step in enumerate(steps):
        command = step.get("command", "")
        observation, reward, done, info = env.step(command)
        print(f"Replay command: {command}")
        print("<" * 70)
        print(observation)
        print(f"Reward: {reward}")
        print(f"Done: {done}")
        print(f"Info: {info}")
        print(">" * 70)
        if done and index < len(steps) - 1:
            print("Replay ended early; savegame does not match engine state.")
            sys.exit(1)
    return observation, done


def main():
    args = parse_args()
    game_path = resolve_game_path(args.game)

    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    llm_agent = LLMAgent(client, game_stem(game_path))

    env = FrotzEnv(str(game_path))
    observation, info = env.reset()

    if args.load:
        log_path, log_data = load_log(args.load)
    else:
        log_path, log_data = create_new_log(game_path, llm_agent.model, observation)

    if not log_data.get("initial_observation"):
        log_data["initial_observation"] = observation

    print(observation)

    replay_observation, replay_done = replay_steps(env, log_data.get("steps", []))
    if replay_observation is not None:
        observation = replay_observation
    if replay_done:
        return

    human_mode = args.human

    while not env.game_over():
        if human_mode:
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
            command = llm_agent.get_action(observation)
            print(f"LLM command: {command}")
            actor = "llm"

        print("<" * 70)
        observation, reward, done, info = env.step(command)
        print(observation)
        print(f"Reward: {reward}")
        print(f"Done: {done}")
        print(f"Info: {info}")
        print(">" * 70)

        append_step(log_data, actor, command, observation, reward, done, info)
        write_log_atomic(log_path, log_data)

        if done:
            return


if __name__ == "__main__":
    main()