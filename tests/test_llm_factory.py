from __future__ import annotations

import pytest
from langchain_ollama import ChatOllama

from cascade_prompting.llm.factory import (
    available_providers,
    build_llm,
    get_default_model,
    get_provider,
)


def test_available_providers_includes_ollama_and_anthropic():
    assert set(available_providers()) == {"ollama", "anthropic"}


def test_get_default_model():
    assert get_default_model("ollama") == "qwen3.6"
    assert get_default_model("anthropic") == "claude-opus-5"


def test_unknown_provider_raises_value_error():
    with pytest.raises(ValueError, match="Unknown provider"):
        get_provider("openai")


def test_build_llm_ollama_returns_chat_ollama():
    llm = build_llm("ollama", "qwen3.6", base_url="http://localhost:11434")
    assert isinstance(llm, ChatOllama)
    assert llm.model == "qwen3.6"


def test_build_llm_anthropic_returns_chat_anthropic(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-key")
    llm = build_llm("anthropic", "claude-opus-5")
    assert type(llm).__name__ == "ChatAnthropic"
    assert llm.model == "claude-opus-5"


def test_ollama_provider_ignores_unrelated_config():
    # Anthropic's api_key kwarg should be harmlessly ignored by OllamaProvider.
    llm = build_llm("ollama", "qwen3.6", base_url="http://localhost:11434", api_key="unused")
    assert isinstance(llm, ChatOllama)
