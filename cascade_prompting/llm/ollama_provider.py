from __future__ import annotations

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_ollama import ChatOllama

from .base import LLMProvider


class OllamaProvider(LLMProvider):
    """Talks to a local/remote Ollama server.

    Config: base_url (default "http://localhost:11434").
    """

    name = "ollama"
    default_model = "qwen3.6"

    def build(self, model: str, temperature: float = 0) -> BaseChatModel:
        base_url = self.config.get("base_url") or "http://localhost:11434"
        return ChatOllama(model=model, base_url=base_url, temperature=temperature)
