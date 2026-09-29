"""Unified LLM client — abstracts over Gemini, Groq, and OpenAI.

Usage:
    from rag.llm import get_client

    client = get_client()   # reads LLM_PROVIDER from .env
    response = client.complete(system="...", user="...")

To switch providers, edit .env:
    LLM_PROVIDER=gemini     # or groq, or openai
"""

import os
from typing import Protocol

from dotenv import load_dotenv

load_dotenv()


# ============================================================
# Protocol — every provider must implement this
# ============================================================

class LLMClient(Protocol):
    """The interface all LLM clients must satisfy."""

    def complete(self, system: str, user: str) -> str:
        """Send a system prompt + user message. Return the response text."""
        ...


# ============================================================
# Google Gemini (free tier, no credit card)
# Get key: https://aistudio.google.com/apikey
# Uses the new Interactions API (recommended for all new projects)
# ============================================================

class GeminiClient:
    DEFAULT_MODEL = "gemini-3.8-flash"

    def __init__(self, model: str | None = None):
        from google import genai  # lazy import

        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError(
                "GOOGLE_API_KEY is not set in .env.\n"
                "Get a free key at https://aistudio.google.com/apikey"
            )
        self.client = genai.Client(api_key=api_key)
        self.model = model or os.getenv("LLM_MODEL", self.DEFAULT_MODEL)

    def complete(self, system: str, user: str) -> str:
        # Interactions API — the new recommended way to call Gemini
        # See: https://ai.google.dev/gemini-api/docs/interactions-overview
        interaction = self.client.interactions.create(
            model=self.model,
            input=user,
            system_instruction=system,
        )
        return (interaction.output_text or "").strip()


# ============================================================
# Groq (free tier, blazing fast, OpenAI-compatible)
# Get key: https://console.groq.com/keys
# ============================================================

class GroqClient:
    DEFAULT_MODEL = "openai/gpt-oss-120b"   # was: llama-3.3-70b-versatile
    
    def __init__(self, model: str | None = None):
        from groq import Groq  # lazy import

        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError(
                "GROQ_API_KEY is not set in .env.\n"
                "Get a free key at https://console.groq.com/keys"
            )
        self.client = Groq(api_key=api_key)
        self.model = model or os.getenv("LLM_MODEL", self.DEFAULT_MODEL)

    def complete(self, system: str, user: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.0,
        )
        return (response.choices[0].message.content or "").strip()


# ============================================================
# OpenAI (paid; kept here in case you top up later)
# Get key: https://platform.openai.com/api-keys
# ============================================================

class OpenAIClient:
    DEFAULT_MODEL = "gpt-4o-mini"

    def __init__(self, model: str | None = None):
        from openai import OpenAI  # lazy import

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY is not set in .env.\n"
                "Get a key at https://platform.openai.com/api-keys"
            )
        self.client = OpenAI(api_key=api_key)
        self.model = model or os.getenv("LLM_MODEL", self.DEFAULT_MODEL)

    def complete(self, system: str, user: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.0,
        )
        return (response.choices[0].message.content or "").strip()


# ============================================================
# Factory
# ============================================================

_PROVIDERS = {
    "gemini": GeminiClient,
    "groq": GroqClient,
    "openai": OpenAIClient,
}


def get_client(provider: str | None = None) -> LLMClient:
    """
    Return an LLM client based on LLM_PROVIDER (default: gemini).

    Override with argument:
        get_client("groq")
    """
    provider = (provider or os.getenv("LLM_PROVIDER", "gemini")).lower()
    if provider not in _PROVIDERS:
        raise ValueError(
            f"Unknown LLM_PROVIDER: {provider!r}. "
            f"Choose from: {', '.join(_PROVIDERS)}"
        )
    return _PROVIDERS[provider]()


# ============================================================
# Manual test
# ============================================================

if __name__ == "__main__":
    provider_name = os.getenv("LLM_PROVIDER", "gemini")
    print(f"Provider: {provider_name}")

    try:
        client = get_client()
        response = client.complete(
            system="You are a helpful assistant. Answer briefly.",
            user="What is 2 + 2?",
        )
        print(f"Response: {response}")
    except Exception as e:
        print(f"ERROR: {type(e).__name__}: {e}")
        raise SystemExit(1)