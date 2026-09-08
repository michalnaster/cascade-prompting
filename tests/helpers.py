from __future__ import annotations

from langchain_core.messages import AIMessage


def usage(input_tokens: int, output_tokens: int) -> dict:
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
    }


def ai_message(content: str, input_tokens: int = 1, output_tokens: int = 1) -> AIMessage:
    return AIMessage(content=content, usage_metadata=usage(input_tokens, output_tokens))
