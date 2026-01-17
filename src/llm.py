from typing import Any, Callable


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
            "next several moves. Focus on medium-term goals and exploration. "
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


class LLMManager:
    def __init__(
        self,
        client,
        *,
        game_name: str,
        gameplay_model: str,
        summary_model: str,
        strategy_model: str,
    ) -> None:
        self.client = client
        self.gameplay_model = gameplay_model
        self.summary_model = summary_model
        self.strategy_model = strategy_model
        self.gameplay_agent = LLMAgent(
            client, game_name=game_name, model=gameplay_model
        )
        self.strategy_agent = StrategyAgent(
            client, game_name=game_name, model=strategy_model
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
            response = self.client.chat.completions.create(
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
