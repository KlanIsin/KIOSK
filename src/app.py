from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory

from src.config import (
    DEFAULT_LANGUAGE,
    DEBUG_MODE,
    DOCS_DIR,
    LANGUAGES,
    PORT,
    SUGGESTED_QUESTIONS,
)
from src.llm import query_openrouter
from src.rag import (
    build_rag_context,
    load_documents,
    retrieve_similar_documents,
    supported_upload_extension,
)

BASE_DIR = Path(__file__).resolve().parent.parent
app = Flask(__name__, static_folder=BASE_DIR / "static", static_url_path="")
MAX_CONVERSATION_HISTORY = 10


@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.route("/api/health")
def health():
    return jsonify({"status": "ok"})


@app.route("/api/suggestions")
def suggestions():
    language = request.args.get("language", DEFAULT_LANGUAGE)
    if language not in LANGUAGES:
        language = DEFAULT_LANGUAGE
    suggested = SUGGESTED_QUESTIONS.get(language, SUGGESTED_QUESTIONS[DEFAULT_LANGUAGE])
    return jsonify(
        {
            "language": language,
            "suggestions": [{"text": suggestion} for suggestion in suggested],
        }
    )


@app.route("/api/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify({"error": "No file provided."}), 400
    file = request.files["file"]
    filename = file.filename
    if not filename or not supported_upload_extension(filename):
        return jsonify({"error": "Unsupported file type."}), 400

    destination = Path(DOCS_DIR)
    destination.mkdir(parents=True, exist_ok=True)
    saved_path = destination / filename
    file.save(saved_path)
    return jsonify({"message": f"Uploaded {filename}", "file": filename})


@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json(silent=True)
    if not data or "message" not in data:
        return jsonify({"error": "Request JSON must include a 'message' field."}), 400

    language = data.get("language", DEFAULT_LANGUAGE)
    if language not in LANGUAGES:
        language = DEFAULT_LANGUAGE

    user_message = str(data["message"]).strip()
    if not user_message:
        return jsonify({"error": "Message cannot be empty."}), 400

    documents = load_documents(DOCS_DIR)
    relevant_docs = retrieve_similar_documents(user_message, documents)
    rag_context = build_rag_context(relevant_docs)

    conversation = data.get("conversation", [])
    if not isinstance(conversation, list):
        conversation = []

    conversation = conversation[-MAX_CONVERSATION_HISTORY:]

    reply = query_openrouter(
        user_message,
        context=rag_context,
        language=language,
        conversation=conversation,
    )

    return jsonify(
        {
            "reply": reply,
            "language": language,
            "source_documents": [doc["name"] for doc in relevant_docs],
        }
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT, debug=DEBUG_MODE)
