from __future__ import annotations

from langchain_core.language_models.chat_models import BaseChatModel

from .base import LLMProvider


class AnthropicProvider(LLMProvider):
    """Talks to the Claude API via langchain-anthropic.

    Config: api_key (defaults to the ANTHROPIC_API_KEY env var when omitted
    or None — see langchain_anthropic.ChatAnthropic).

    Per Anthropic guidance, `default_model` is Opus (the most capable
    widely-available model) unless the user names a different one via
    --model/--judge-model.
    """

    name = "anthropic"
    default_model = "claude-opus-5"

    def build(self, model: str, temperature: float = 0) -> BaseChatModel:
        try:
            from langchain_anthropic import ChatAnthropic
        except ImportError as e:
            raise ImportError(
                "The 'anthropic' provider requires langchain-anthropic "
                "(uv pip install langchain-anthropic, or `uv pip install -r requirements.txt`)."
            ) from e
        api_key = self.config.get("api_key")
        return ChatAnthropic(model=model, temperature=temperature, api_key=api_key)
