import os
from unittest.mock import patch

import pytest

from src.llm import get_api_key, query_openrouter


def test_get_api_key_from_env(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    assert get_api_key() == "sk-test"


def test_get_api_key_missing(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(RuntimeError):
        get_api_key()


def test_query_openrouter_builds_messages(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    expected = {"choices": [{"message": {"content": "Hello"}}]}
    with patch("src.llm.requests.post") as post:
        post.return_value.status_code = 200
        post.return_value.json.return_value = expected
        post.return_value.raise_for_status.return_value = None
        content = query_openrouter("Hi", language="yue")
        assert content == "Hello"
        assert post.called
