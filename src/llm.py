import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from openai import OpenAI


@dataclass
class ProviderConfig:
    model: str
    base_url: str | None = None
    api_key: str | None = None

    def resolved_api_key(self) -> str | None:
        if not self.api_key:
            return None
        if self.api_key.strip() == "${OPENAI_API_KEY}":
            return os.environ.get("OPENAI_API_KEY")
        return self.api_key


@dataclass
class LLMConfig:
    gameplay: ProviderConfig
    summary: ProviderConfig
    strategy: ProviderConfig
    post_mortem: ProviderConfig


def build_default_llm_config(
    *,
    gameplay_model: str,
    summary_model: str,
    strategy_model: str,
    post_mortem_model: str,
) -> LLMConfig:
    return LLMConfig(
        gameplay=ProviderConfig(model=gameplay_model),
        summary=ProviderConfig(model=summary_model),
        strategy=ProviderConfig(model=strategy_model),
        post_mortem=ProviderConfig(model=post_mortem_model),
    )


def load_llm_config(config_path: Path, defaults: LLMConfig) -> LLMConfig:
    if not config_path.exists():
        return defaults
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("llm_config.json must contain a JSON object.")

    def _require_model(value: object, role: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{role} model name is required.")
        return value.strip()

    def _load_role(role: str, default: ProviderConfig) -> ProviderConfig:
        raw = payload.get(role)
        if raw is None:
            return default
        if not isinstance(raw, dict):
            raise ValueError(f"{role} config must be a JSON object.")
        model = _require_model(raw.get("model", default.model), role)
        base_url = raw.get("base_url", default.base_url)
        api_key = raw.get("api_key", default.api_key)
        if base_url is not None and not isinstance(base_url, str):
            raise ValueError(f"{role}.base_url must be a string.")
        if api_key is not None and not isinstance(api_key, str):
            raise ValueError(f"{role}.api_key must be a string.")
        return ProviderConfig(model=model, base_url=base_url, api_key=api_key)

    return LLMConfig(
        gameplay=_load_role("gameplay", defaults.gameplay),
        summary=_load_role("summary", defaults.summary),
        strategy=_load_role("strategy", defaults.strategy),
        post_mortem=_load_role("post_mortem", defaults.post_mortem),
    )


def build_client(config: ProviderConfig) -> OpenAI:
    api_key = config.resolved_api_key() or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY is required to initialize the LLM client.")
    if config.base_url:
        return OpenAI(base_url=config.base_url, api_key=api_key)
    return OpenAI(api_key=api_key)


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


class LLMAgent:
    def __init__(self, client, *, game_name: str, model: str) -> None:
        self.client = client
        self.model = model
        self.system_prompt = (
            f"You are playing the game {game_name}. You are the player and you "
            "are trying to win the game. Commands should be short and concise "
            "(never more than one short sentence, typically just one or two "
            "words). You are connected directly to the game engine."
        )

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
        strategy: str | None = None,
    ) -> list[dict[str, str]]:
        messages = [{"role": "system", "content": self.system_prompt}]
        if ltm_summary:
            messages.append(
                {"role": "user", "content": f"Summary so far:\n{ltm_summary}"}
            )
        if strategy:
            messages.append({"role": "user", "content": f"Current strategy:\n{strategy}"})
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

    def get_action(self, messages: list[dict[str, str]]) -> tuple[str, dict[str, Any]]:
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


class StrategyAgent:
    def __init__(self, client, *, game_name: str, model: str) -> None:
        self.client = client
        self.model = model
        self.system_prompt = (
            f"You are a strategist for the game {game_name}. You are given the story "
            "so far and recent turns. Synthesize a concise short-term strategy for the "
            "next few moves. Focus on short-term goals and exploration. "
            "Look for red herrings and repetitive actions and make a note to avoid them. "
            "Output 2-4 short bullet points or a brief paragraph. Do not output commands."
        )

    def build_messages(
        self,
        *,
        ltm_summary: str,
        stm_steps: list[dict[str, Any]],
    ) -> list[dict[str, str]]:
        messages = [{"role": "system", "content": self.system_prompt}]
        if ltm_summary:
            messages.append(
                {"role": "user", "content": f"Summary so far:\n{ltm_summary}"}
            )
        recent_turns = format_recent_turns(stm_steps)
        if recent_turns:
            messages.append({"role": "user", "content": recent_turns})
        return messages

    def get_strategy(self, messages: list[dict[str, str]], previous_strategy: str) -> tuple[str, dict[str, Any]]:
        if not messages:
            raise ValueError("LLM prompt messages are required.")
        if previous_strategy:
            messages.append({"role": "user", "content": f"Previous strategy:\n{previous_strategy}"})
        history_summary_before = LLMAgent._summarize_messages(messages)
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
        history_summary_after = LLMAgent._summarize_messages(
            messages + [{"role": "assistant", "content": model_response or ""}]
        )
        journal = {
            "model": self.model,
            "request": request_summary,
            "history_before": history_summary_before,
            "history_after": history_summary_after,
            "usage": usage_dict,
        }
        if not isinstance(model_response, str):
            return "", journal
        return model_response.strip(), journal


class PostMortemStrategyAgent:
    def __init__(self, client, *, game_name: str, model: str) -> None:
        self.client = client
        self.model = model
        self.system_prompt = (
            f"You've been playing the text adventure game {game_name}. "
            "You will be given the latest long-term summary and a batch of turns "
            "from a completed or partial playthrough. Your task is to augment it into a "
            "self-contained, concise, markdown-rich strategy document for beating the game. "
            "Document title: `# (Game Name) Strategy Guide`. "
            "Keep in mind that new playthroughs might show that previous insights were wrong or incomplete, "
            "so be prepared to update the strategy accordingly. "
            "This is meant to be a self-contained, internally coherent, high-level set of notes for the player, "
            "not a detailed step-by-step guide, and not a journal of the playthroughs. "
            "Avoid mentioning \"new\"/\"updated\" insights, you'll fall into the \"new new\", \"newest last final\" silliness. "
            "Extract concrete learnings, pitfalls, and actionable guidance. "
            "Avoid repeating obvious or already-known facts unless the batch "
            "adds new evidence. Output the full UPDATED strategy in markdown. "
            "The strategy should only be about the game universe, never about saving, pausing, or other game mechanics. "
            "Do NOT mention step numbers or step batches in the strategy; those are just for your reference, "
            "so you can have an idea of how much time has passed since the start of the current playthrough, "
            "but things might happen in a completely different order in the future, or the player "
            "might not get stuck, etc. Avoid excessive verbosity; this is not a magazine article describing the game, "
            "it's a self-contained, ever-improving, internally coherent cheat sheet for the player to avoid common pitfalls, mistakes, and bumbling around."
        )

    def build_messages(
        self,
        *,
        context_line: str,
        ltm_summary: str,
        batch_text: str,
        existing_strategy: str | None,
    ) -> list[dict[str, str]]:
        messages = [{"role": "system", "content": self.system_prompt}]
        if context_line:
            messages.append({"role": "user", "content": context_line})
        if ltm_summary:
            messages.append(
                {"role": "user", "content": f"Latest LTM summary:\n{ltm_summary}"}
            )
        if existing_strategy:
            messages.append(
                {
                    "role": "user",
                    "content": f"Existing strategy (partial or full):\n{existing_strategy}",
                }
            )
        if batch_text:
            messages.append({"role": "user", "content": f"Batch steps:\n{batch_text}"})
        return messages

    def get_strategy(self, messages: list[dict[str, str]]) -> tuple[str, dict[str, Any]]:
        if not messages:
            raise ValueError("LLM prompt messages are required.")
        history_summary_before = LLMAgent._summarize_messages(messages)
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
        history_summary_after = LLMAgent._summarize_messages(
            messages + [{"role": "assistant", "content": model_response or ""}]
        )
        journal = {
            "model": self.model,
            "request": request_summary,
            "history_before": history_summary_before,
            "history_after": history_summary_after,
            "usage": usage_dict,
        }
        if not isinstance(model_response, str):
            return "", journal
        return model_response.strip(), journal


class LLMManager:
    def __init__(
        self,
        *,
        game_name: str,
        gameplay: ProviderConfig,
        summary: ProviderConfig,
        strategy: ProviderConfig,
        post_mortem: ProviderConfig,
    ) -> None:
        if not isinstance(game_name, str) or not game_name.strip():
            raise ValueError("Game name is required.")
        self.gameplay_client = build_client(gameplay)
        self.summary_client = build_client(summary)
        self.strategy_client = build_client(strategy)
        self.post_mortem_client = build_client(post_mortem)
        self.gameplay_model = gameplay.model
        self.summary_model = summary.model
        self.strategy_model = strategy.model
        self.post_mortem_model = post_mortem.model
        self.gameplay_agent = LLMAgent(
            self.gameplay_client, game_name=game_name, model=self.gameplay_model
        )
        self.strategy_agent = StrategyAgent(
            self.strategy_client, game_name=game_name, model=self.strategy_model
        )
        self.post_mortem_agent = PostMortemStrategyAgent(
            self.post_mortem_client, game_name=game_name, model=self.post_mortem_model
        )

    def set_gameplay_model(self, model: str) -> None:
        if not model:
            raise ValueError("Gameplay model name is required.")
        self.gameplay_model = model
        self.gameplay_agent.model = model

    def set_summary_model(self, model: str) -> None:
        if not model:
            raise ValueError("Summary model name is required.")
        self.summary_model = model

    def set_strategy_model(self, model: str) -> None:
        if not model:
            raise ValueError("Strategy model name is required.")
        self.strategy_model = model
        self.strategy_agent.model = model

    def get_summarizer(self) -> Callable[[str, str], str]:
        def summarize(system_prompt: str, user_prompt: str) -> str:
            response = self.summary_client.chat.completions.create(
                model=self.summary_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            summary_text = response.choices[0].message.content
            if not isinstance(summary_text, str):
                return ""
            return summary_text.strip()

        return summarize
