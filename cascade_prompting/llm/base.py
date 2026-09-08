"""Provider abstraction (Strategy pattern) for constructing chat models.

Every generation strategy in `cascade_prompting.strategies` is written
against the LangChain `BaseChatModel` interface (`invoke` /
`with_structured_output`), never against a concrete provider class. That's
what makes providers swappable: adding a new one (e.g. OpenAI, Bedrock) means
adding a new `LLMProvider` subclass and registering it in
`cascade_prompting.llm.factory` — nothing in `strategies/` or `benchmark.py`
has to change (Open/Closed Principle).

Constructor config is a free-form kwargs bag (`self.config`) rather than
named parameters, so `cascade_prompting.llm.factory.build_llm` can pass the
same kwargs to any provider without branching on its name — each provider
just reads the keys it cares about (e.g. Ollama reads "base_url", Anthropic
reads "api_key") and ignores the rest.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel


class LLMProvider(ABC):
    """One chat-model provider (e.g. Ollama, Anthropic)."""

    #: Registry key, e.g. "ollama" — used for --provider/--judge-provider.
    name: str

    #: Model used when the user doesn't pass --model / --judge-model.
    default_model: str

    def __init__(self, **config: Any) -> None:
        self.config: dict[str, Any] = config

    @abstractmethod
    def build(self, model: str, temperature: float = 0) -> BaseChatModel:
        """Construct a chat model instance for `model` at `temperature`."""
        raise NotImplementedError
