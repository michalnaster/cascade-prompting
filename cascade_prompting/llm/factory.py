"""Provider registry and construction entry point.

`build_llm` is the one function the rest of the codebase calls; it never
branches on provider name — that knowledge lives entirely inside each
`LLMProvider` subclass. Adding a new provider is: write the subclass, call
`register_provider` on it. No other file needs to change.
"""

from __future__ import annotations

from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel

from .anthropic_provider import AnthropicProvider
from .base import LLMProvider
from .ollama_provider import OllamaProvider

_PROVIDERS: dict[str, type[LLMProvider]] = {}


def register_provider(provider_cls: type[LLMProvider]) -> type[LLMProvider]:
    """Register a provider class under its `.name`. Usable as a decorator."""
    _PROVIDERS[provider_cls.name] = provider_cls
    return provider_cls


def available_providers() -> list[str]:
    return sorted(_PROVIDERS)


def get_provider(name: str, **config: Any) -> LLMProvider:
    try:
        provider_cls = _PROVIDERS[name]
    except KeyError:
        raise ValueError(
            f"Unknown provider {name!r} (expected one of {available_providers()})"
        ) from None
    return provider_cls(**config)


def get_default_model(name: str) -> str:
    return get_provider(name).default_model


def build_llm(
    provider_name: str, model: str, temperature: float = 0, **config: Any
) -> BaseChatModel:
    """Construct a chat model for `provider_name`/`model`.

    Extra keyword args are forwarded to the provider's config bag (e.g.
    base_url for ollama, api_key for anthropic); a provider ignores config
    keys it doesn't use.
    """
    provider = get_provider(provider_name, **config)
    return provider.build(model=model, temperature=temperature)


register_provider(OllamaProvider)
register_provider(AnthropicProvider)
