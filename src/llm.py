import os
from typing import Optional

import requests

from src.config import MODEL, OPENROUTER_URL, SYSTEM_PROMPTS, DEFAULT_LANGUAGE


def get_api_key() -> str:
    env_key = os.getenv("OPENROUTER_API_KEY")
    if env_key:
        return env_key.strip()
    raise RuntimeError(
        "API key not found. Set OPENROUTER_API_KEY in .env or as an environment variable."
    )


def query_openrouter(
    message: str,
    context: Optional[str] = None,
    language: str = "en",
    conversation: Optional[list[dict]] = None,
) -> str:
    api_key = get_api_key()
    # Use configurable system prompts defined in src/config.py
    system_prompt = SYSTEM_PROMPTS.get(language) or SYSTEM_PROMPTS.get(DEFAULT_LANGUAGE)
    messages = [{"role": "system", "content": system_prompt}]
    if context:
        messages.append({"role": "system", "content": context})

    if conversation:
        for entry in conversation:
            role = entry.get("role")
            content = entry.get("content")
            if role and content:
                messages.append({"role": role, "content": content})

        if not conversation or conversation[-1].get("content") != message:
            messages.append({"role": "user", "content": message})
    else:
        messages.append({"role": "user", "content": message})

    response = requests.post(
        OPENROUTER_URL,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={"model": MODEL, "messages": messages},
        timeout=30,
    )
    response.raise_for_status()
    result = response.json()
    return result["choices"][0]["message"]["content"]
