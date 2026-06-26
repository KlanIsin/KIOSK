import io
from unittest.mock import patch

import pytest

from src.app import app as flask_app
from src.config import DEFAULT_LANGUAGE


@pytest.fixture(autouse=True)
def set_env_api_key(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")


def test_health_endpoint():
    client = flask_app.test_client()
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.get_json()["status"] == "ok"


def test_suggestions_endpoint():
    client = flask_app.test_client()
    response = client.get(f"/api/suggestions?language={DEFAULT_LANGUAGE}")
    assert response.status_code == 200
    data = response.get_json()
    assert data["language"] == DEFAULT_LANGUAGE
    assert "suggestions" in data


def test_upload_no_file():
    client = flask_app.test_client()
    response = client.post("/api/upload")
    assert response.status_code == 400
    assert response.get_json()["error"] == "No file provided."


def test_upload_unsupported_file():
    client = flask_app.test_client()
    data = {"file": (io.BytesIO(b"some data"), "image.png")}
    response = client.post("/api/upload", data=data, content_type="multipart/form-data")
    assert response.status_code == 400
    assert response.get_json()["error"] == "Unsupported file type."


def test_chat_endpoint_with_empty_message():
    client = flask_app.test_client()
    response = client.post("/api/chat", json={"message": ""})
    assert response.status_code == 400
    assert "cannot be empty" in response.get_json()["error"].lower()


def test_chat_endpoint_calls_model(monkeypatch):
    client = flask_app.test_client()

    def fake_load_documents(directory):
        return [{"name": "school.txt", "text": "School schedule and attendance policy."}]

    def fake_query_openrouter(message, context=None, language=DEFAULT_LANGUAGE, conversation=None):
        assert language == DEFAULT_LANGUAGE
        assert "School schedule" in context
        assert isinstance(conversation, list)
        assert conversation[0]["role"] == "user"
        assert conversation[0]["content"] == "What is the attendance policy?"
        return "Test answer"

    monkeypatch.setattr("src.app.load_documents", fake_load_documents)
    monkeypatch.setattr("src.app.query_openrouter", fake_query_openrouter)

    response = client.post(
        "/api/chat",
        json={
            "message": "What is the attendance policy?",
            "language": DEFAULT_LANGUAGE,
            "conversation": [{"role": "user", "content": "What is the attendance policy?"}],
        },
    )
    assert response.status_code == 200
    data = response.get_json()
    assert data["reply"] == "Test answer"
    assert data["language"] == DEFAULT_LANGUAGE
    assert data["source_documents"] == ["school.txt"]


def test_chat_endpoint_trims_long_conversation(monkeypatch):
    client = flask_app.test_client()

    def fake_load_documents(directory):
        return []

    def fake_query_openrouter(message, context=None, language=DEFAULT_LANGUAGE, conversation=None):
        assert isinstance(conversation, list)
        assert len(conversation) == 10
        assert conversation[0]["content"] == "Turn 2"
        assert conversation[-1]["content"] == "Turn 11"
        return "Trimmed reply"

    monkeypatch.setattr("src.app.load_documents", fake_load_documents)
    monkeypatch.setattr("src.app.query_openrouter", fake_query_openrouter)

    conversation = [
        {"role": "user", "content": f"Turn {i}"}
        for i in range(1, 12)
    ]

    response = client.post(
        "/api/chat",
        json={
            "message": "What is the attendance policy?",
            "language": DEFAULT_LANGUAGE,
            "conversation": conversation,
        },
    )
    assert response.status_code == 200
    data = response.get_json()
    assert data["reply"] == "Trimmed reply"


def test_chat_endpoint_invalid_language(monkeypatch):
    client = flask_app.test_client()

    def fake_load_documents(directory):
        return []

    def fake_query_openrouter(message, context=None, language=DEFAULT_LANGUAGE, conversation=None):
        assert language == DEFAULT_LANGUAGE
        assert conversation == [{"role": "user", "content": "Hello"}]
        return "Reply"

    monkeypatch.setattr("src.app.load_documents", fake_load_documents)
    monkeypatch.setattr("src.app.query_openrouter", fake_query_openrouter)

    response = client.post(
        "/api/chat",
        json={"message": "Hello", "language": "invalid", "conversation": [{"role": "user", "content": "Hello"}]},
    )
    assert response.status_code == 200
    data = response.get_json()
    assert data["language"] == DEFAULT_LANGUAGE
