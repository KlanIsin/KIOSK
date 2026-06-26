import os
from pathlib import Path

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env")

MODEL = os.getenv("MODEL", "cohere/north-mini-code:free")
OPENROUTER_URL = os.getenv(
    "OPENROUTER_URL", "https://openrouter.ai/api/v1/chat/completions"
)
DOCS_DIR = os.getenv("DOCS_DIR", "knowledge_base")
PORT = int(os.getenv("PORT", "5000"))
DEBUG_MODE = os.getenv("DEBUG", "True").lower() in ("1", "true", "yes")
ALLOWED_EXTENSIONS = {".txt", ".pdf"}
LANGUAGES = {"en": "English", "yue": "Cantonese"}
DEFAULT_LANGUAGE = "en"
SUGGESTED_QUESTIONS = {
    "en": [
        "What are the school hours?",
        "How do I contact the main office?",
        "What is the attendance policy?",
        "When is the next holiday?",
    ],
    "yue": [
        "學校幾點上學？",
        "我點樣聯絡辦公室？",
        "考勤政策係點？",
        "下一個假期係幾時？",
    ],
}
MAX_DOCS = 3
MAX_DOC_SNIPPET = 1200
CHUNK_SIZE = 150
CHUNK_OVERLAP = 30
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)

# System prompts used as the assistant's instruction (configurable).
# By default we load `system_prompts.json` from the repository root if present,
# otherwise fall back to these built-in defaults.
PROMPTS_FILE = ROOT_DIR / "system_prompts.json"

DEFAULT_SYSTEM_PROMPTS = {
    "en": (
        "You are a helpful school helpdesk assistant. "
        "Answer in English. "
        "Always format your answer in Markdown. Use headings, bold text, lists, tables, and line breaks as needed. "
        "Do not provide any HTML or code wrapper; return only Markdown text."
    ),
    "yue": (
        "You are a helpful school helpdesk assistant. "
        "Answer in Cantonese using traditional Chinese characters. "
        "Always format your answer in Markdown. Use headings, bold text, lists, tables, and line breaks as needed. "
        "Do not provide any HTML or code wrapper; return only Markdown text."
    ),
}

try:
    if PROMPTS_FILE.is_file():
        import json

        with PROMPTS_FILE.open("r", encoding="utf-8") as f:
            loaded = json.load(f)
        # Expect a dict mapping language keys to prompt strings
        if isinstance(loaded, dict) and loaded:
            SYSTEM_PROMPTS = loaded
        else:
            SYSTEM_PROMPTS = DEFAULT_SYSTEM_PROMPTS
    else:
        SYSTEM_PROMPTS = DEFAULT_SYSTEM_PROMPTS
except Exception:
    SYSTEM_PROMPTS = DEFAULT_SYSTEM_PROMPTS
