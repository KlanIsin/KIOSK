import json
import os
import re
from pathlib import Path

from dotenv import load_dotenv
from PyPDF2 import PdfReader
import requests
from flask import Flask, jsonify, request, send_from_directory

load_dotenv()

MODEL = os.getenv("MODEL", "cohere/north-mini-code:free")
URL = os.getenv("OPENROUTER_URL", "https://openrouter.ai/api/v1/chat/completions")
DOCS_DIR = os.getenv("DOCS_DIR", "docs")
PORT = int(os.getenv("PORT", "5000"))
DEBUG_MODE = os.getenv("DEBUG", "True").lower() in ("1", "true", "yes")
MAX_DOCS = 3
MAX_DOC_SNIPPET = 1500

app = Flask(__name__, static_folder="static", static_url_path="")


def get_api_key():
    env_key = os.getenv("OPENROUTER_API_KEY")
    if env_key:
        return env_key.strip()

    api_file = Path("api.txt")
    if api_file.exists():
        for line in api_file.read_text(encoding="utf-8").splitlines():
            token = line.strip()
            if token.startswith("sk-"):
                return token

    raise RuntimeError(
        "API key not found. Set OPENROUTER_API_KEY or place a key in api.txt."
    )


API_KEY = get_api_key()


def extract_text_from_pdf(path: Path) -> str:
    try:
        reader = PdfReader(path)
        text_parts = []
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text)
        return "\n\n".join(text_parts).strip()
    except Exception as exc:
        print(f"Warning: Failed to extract text from PDF '{path.name}': {exc}")
        return ""


def load_documents(directory):
    documents = []
    docs_path = Path(directory)
    if not docs_path.is_dir():
        return documents

    for path in sorted(docs_path.iterdir()):
        if path.suffix.lower() == ".txt":
            text = path.read_text(encoding="utf-8").strip()
        elif path.suffix.lower() == ".pdf":
            text = extract_text_from_pdf(path)
        else:
            continue

        if text:
            documents.append({"name": path.name, "text": text})

    return documents


def tokenize(text):
    return set(re.findall(r"\w+", text.lower()))


def score_document(query, document):
    query_tokens = tokenize(query)
    doc_tokens = tokenize(document["text"])
    return len(query_tokens & doc_tokens)


def retrieve_documents(query, documents, top_k=MAX_DOCS):
    scored = [
        {"score": score_document(query, doc), "doc": doc}
        for doc in documents
    ]
    scored = [item for item in scored if item["score"] > 0]
    scored.sort(key=lambda item: item["score"], reverse=True)
    return [item["doc"] for item in scored[:top_k]]


def build_rag_context(documents):
    if not documents:
        return None

    snippets = []
    for doc in documents:
        snippet = doc["text"][:MAX_DOC_SNIPPET].strip()
        if len(doc["text"]) > MAX_DOC_SNIPPET:
            snippet += "\n\n[...truncated...]"
        snippets.append(f"### {doc['name']}\n{snippet}")

    return (
        "Use the following reference documents to answer the user's question. "
        "If the answer is not contained in these documents, say you don't know."
        "\n\n"
        + "\n\n".join(snippets)
    )


@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.route("/api/health")
def health():
    return jsonify({"status": "ok"})


@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json(silent=True)
    if not data or "message" not in data:
        return jsonify({"error": "Request JSON must include a 'message' field."}), 400

    user_message = str(data["message"]).strip()
    if not user_message:
        return jsonify({"error": "Message cannot be empty."}), 400

    documents = load_documents(DOCS_DIR)
    relevant_docs = retrieve_documents(user_message, documents)
    rag_context = build_rag_context(relevant_docs)

    messages = []
    if rag_context:
        messages.append({"role": "system", "content": rag_context})
    messages.append({"role": "user", "content": user_message})

    payload = {
        "model": MODEL,
        "messages": messages,
    }

    response = requests.post(
        URL,
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=30,
    )

    if response.status_code != 200:
        return (
            jsonify(
                {
                    "error": "Model request failed.",
                    "status_code": response.status_code,
                    "details": response.text,
                }
            ),
            502,
        )

    result = response.json()
    assistant_message = result["choices"][0]["message"]["content"]

    return jsonify(
        {
            "reply": assistant_message,
            "source_documents": [doc["name"] for doc in relevant_docs],
        }
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT, debug=DEBUG_MODE)
