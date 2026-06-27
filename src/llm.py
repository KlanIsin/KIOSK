import os
from typing import Optional

import requests

from src.config import MODEL, OPENROUTER_URL, SYSTEM_PROMPTS, DEFAULT_LANGUAGE, ANTHROPIC_MAX_TOKENS


# --- Provider registry ----------------------------------------------------
# Allow ANY provider API key to work without code changes.
# Priority order is the order we look up env vars in. The first non-empty
# value wins. Set LLM_API_KEY for a provider-neutral key, or use the
# provider-specific names below.
#
# `base_url` is the OpenAI-compatible chat-completions endpoint. The
# `Authorization: Bearer <key>` header is sent for every provider.
# Override any provider's URL via the matching `_BASE_URL` env var, or
# globally via LLM_BASE_URL.

PROVIDERS = (
    # (provider_id, env_var, default_base_url)
    # Explicit provider keys. Listed in the order checked; explicit
    # provider-specific names take priority over the legacy
    # OPENROUTER_API_KEY slot, so users who set OPENAI_API_KEY +
    # OPENROUTER_API_KEY get OpenAI.
    ("openai",     "OPENAI_API_KEY",     "https://api.openai.com/v1/chat/completions"),
    ("anthropic",  "ANTHROPIC_API_KEY",  "https://api.anthropic.com/v1/messages"),
    ("gemini",     "GEMINI_API_KEY",     "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"),
    ("groq",       "GROQ_API_KEY",       "https://api.groq.com/openai/v1/chat/completions"),
    ("mistral",    "MISTRAL_API_KEY",    "https://api.mistral.ai/v1/chat/completions"),
    ("deepseek",   "DEEPSEEK_API_KEY",   "https://api.deepseek.com/v1/chat/completions"),
    ("together",   "TOGETHER_API_KEY",   "https://api.together.xyz/v1/chat/completions"),
    # Legacy OpenRouter slot — kept last among explicit keys so existing
    # deployments that only set OPENROUTER_API_KEY still work unchanged.
    ("openrouter", "OPENROUTER_API_KEY", OPENROUTER_URL),
)

# Neutral provider-neutral alias: only consulted when no provider-specific
# key is set. We assume OpenAI-compatible chat-completions; users can
# override with LLM_BASE_URL.
NEUTRAL_ALIAS = ("openrouter", "LLM_API_KEY", OPENROUTER_URL)

# When using a provider that requires a different request/response shape
# (Anthropic, Gemini native), we still send OpenAI-compatible payloads for
# the simple chat case via Gemini's OpenAI-compat endpoint above.
# Anthropic uses x-api-key + different body — we detect it from the URL.
# OPENAI_COMPAT_OVERRIDES was previously defined here but was never read.


# Heuristic prefixes for each provider — used to warn when MODEL looks like
# it was written for a different provider than the active API key. These
# are *advisory only*; many providers (OpenRouter, Together, Groq, DeepSeek)
# accept OpenAI-style model names so we only warn on the clearest mismatches.
_PROVIDER_MODEL_HINTS = {
    "openai":     ("gpt-", "o1", "o3", "o4", "chatgpt-"),
    "anthropic":  ("claude-",),
    "gemini":     ("gemini-",),
    "groq":       ("llama", "mixtral", "gemma-", "whisper-"),
    "mistral":    ("mistral", "mixtral", "codestral"),
    "deepseek":   ("deepseek-",),
    "together":   ("meta-llama", "mistralai", "togethercomputer"),
    "openrouter": (),  # accepts everything; nothing to validate.
}


def validate_model_for_provider(provider: str, model: str) -> str | None:
    """Return a warning string if ``model`` doesn't look right for ``provider``.

    Returns ``None`` if everything looks fine. Used at startup to log a
    one-line heads-up so misconfiguration is visible in the server log
    instead of failing the first user request.
    """
    if not provider or not model:
        return None
    hints = _PROVIDER_MODEL_HINTS.get(provider)
    if hints is None or not hints:
        return None  # Unknown provider or universal-acceptor — skip.
    lower = model.lower()
    if any(lower.startswith(h) for h in hints):
        return None
    return (
        f"MODEL={model!r} doesn't look like a typical {provider} model name. "
        f"Expected one of: {', '.join(hints)}. "
        f"Continuing anyway — set MODEL in your .env to silence this."
    )


def warn_on_misconfigured_model(logger=None) -> None:
    """Check current env (MODEL + active API key) and log a warning if mismatched.

    Safe to call at app startup. Uses the standard ``logging`` module by
    default so it integrates with whatever the host configures.
    """
    import logging
    log = logger or logging.getLogger(__name__)
    try:
        key = get_api_key()
    except Exception:
        return  # No key — already logged elsewhere; nothing to compare against.
    warning = validate_model_for_provider(key.provider, MODEL)
    if warning:
        log.warning(warning)


class ApiKeyResult(str):
    """Backwards-compatible return type: compares equal to the raw API key.

    Existing callers/tests treat ``get_api_key()`` as ``str``. New callers
    can also read ``.provider`` and ``.base_url`` to know which endpoint
    to hit. The string value is the API key itself.
    """

    def __new__(cls, api_key: str, provider: str, base_url: str):
        instance = super().__new__(cls, api_key)
        instance.provider = provider
        instance.api_key = api_key
        instance.base_url = base_url
        return instance

    def __repr__(self) -> str:
        # Never leak the key in logs / repr.
        return f"ApiKeyResult(provider={self.provider!r}, base_url={self.base_url!r}, api_key=***)"


def _read_env(name: str) -> Optional[str]:
    val = os.getenv(name)
    if val is None:
        return None
    val = val.strip()
    return val or None


def get_api_key() -> ApiKeyResult:
    """Find any usable API key in the environment.

    Scans for `LLM_API_KEY` (neutral), `OPENROUTER_API_KEY` (legacy), and
    common provider-specific keys. The first non-empty value wins. Returns
    an :class:`ApiKeyResult` with the matched provider and base URL.

    Raises ``RuntimeError`` if no key is found.
    """
    # Generic base-URL override applies to whichever provider is selected.
    base_override = _read_env("LLM_BASE_URL")

    # Phase 1: look for explicit provider keys. These always win so that a
    # user who sets both a provider-specific key and the neutral alias
    # (e.g. OPENAI_API_KEY + LLM_API_KEY) gets the provider-specific one.
    for provider_id, env_var, default_url in PROVIDERS:
        key = _read_env(env_var)
        if not key:
            continue
        url = base_override or _read_env(f"{provider_id.upper()}_BASE_URL") or default_url
        return ApiKeyResult(api_key=key, provider=provider_id, base_url=url)

    # Phase 2: fall back to the neutral alias. This is what lets a user
    # drop in any OpenAI-compatible API key as LLM_API_KEY without
    # editing code.
    provider_id, env_var, default_url = NEUTRAL_ALIAS
    key = _read_env(env_var)
    if key:
        url = base_override or _read_env(f"{provider_id.upper()}_BASE_URL") or default_url
        return ApiKeyResult(api_key=key, provider=provider_id, base_url=url)

    raise RuntimeError(
        "API key not found. Set one of: LLM_API_KEY, OPENROUTER_API_KEY, "
        "OPENAI_API_KEY, ANTHROPIC_API_KEY, GEMINI_API_KEY, GROQ_API_KEY, "
        "MISTRAL_API_KEY, DEEPSEEK_API_KEY, or TOGETHER_API_KEY."
    )


def _post_chat(key_result: ApiKeyResult, payload: dict) -> requests.Response:
    """POST a chat-completion request, translating payload for non-OAI providers."""
    if key_result.provider == "anthropic":
        # Translate OpenAI-style {messages:[{role,content}]} -> Anthropic
        # {system, messages:[{role,content}]}. Strip the leading system
        # message and pass the rest verbatim; prepended system goes into
        # the top-level `system` field.
        oai_messages = payload.get("messages", [])
        system_parts = [m["content"] for m in oai_messages if m.get("role") == "system"]
        non_system = [m for m in oai_messages if m.get("role") != "system"]
        anthropic_body = {
            "model": payload["model"],
            "max_tokens": ANTHROPIC_MAX_TOKENS,
            "messages": non_system,
        }
        if system_parts:
            anthropic_body["system"] = "\n\n".join(system_parts)
        return requests.post(
            key_result.base_url,
            headers={
                "x-api-key": key_result.api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json=anthropic_body,
            timeout=30,
        )

    # Default: OpenAI-compatible chat-completions endpoint.
    return requests.post(
        key_result.base_url,
        headers={
            "Authorization": f"Bearer {key_result.api_key}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=30,
    )


def _extract_reply(key_result: ApiKeyResult, response: requests.Response) -> str:
    """Pull the assistant text out of the provider response."""
    data = response.json()
    if key_result.provider == "anthropic":
        # Anthropic returns {"content":[{"type":"text","text":"..."}], ...}
        content = data.get("content") or []
        parts = [c.get("text", "") for c in content if c.get("type") == "text"]
        return "\n".join(parts).strip()
    # OpenAI-compatible shape.
    return data["choices"][0]["message"]["content"]


def query_openrouter(
    message: str,
    context: Optional[str] = None,
    language: str = "en",
    conversation: Optional[list[dict]] = None,
) -> str:
    """Send a chat request to whichever provider the active API key is for.

    Function name kept as ``query_openrouter`` for backwards compatibility
    with existing callers (src/app.py, tests).
    """
    key_result = get_api_key()
    # Use configurable system prompts defined in src/config.py
    system_prompt = SYSTEM_PROMPTS.get(language) or SYSTEM_PROMPTS.get(DEFAULT_LANGUAGE)
    messages = [{"role": "system", "content": system_prompt}]
    if context:
        messages.append({"role": "system", "content": context})

    if conversation:
        for entry in conversation:
            role = entry.get("role")
            content = entry.get("content")
            # Server owns the system prompt + RAG context. Strip any
            # client-supplied system message to prevent the prompt from
            # being duplicated or, worse, overridden by the client.
            if role == "system":
                continue
            if role and content:
                messages.append({"role": role, "content": content})

        if not conversation or conversation[-1].get("content") != message:
            messages.append({"role": "user", "content": message})
    else:
        messages.append({"role": "user", "content": message})

    payload = {"model": MODEL, "messages": messages}
    response = _post_chat(key_result, payload)
    response.raise_for_status()
    return _extract_reply(key_result, response)
