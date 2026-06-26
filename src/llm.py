import os
from typing import Optional

import requests

from src.config import MODEL, OPENROUTER_URL


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
    system_prompt = "You are a helpful school helpdesk assistant. "
    if language == "yue":
        system_prompt += (
            "Answer in Cantonese using traditional Chinese characters. "
            "Always format your answer in Markdown. Use headings, bold text, lists, tables, and line breaks as needed. "
            "Do not provide any HTML or code wrapper; return only Markdown text."
        )
    else:
        system_prompt += (
            "Answer in English. "
            "Always format your answer in Markdown. Use headings, bold text, lists, tables, and line breaks as needed. "
            "Do not provide any HTML or code wrapper; return only Markdown text."
        )

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
