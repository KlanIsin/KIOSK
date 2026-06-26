import re
from pathlib import Path
from typing import Dict, List, Optional

from PyPDF2 import PdfReader
import numpy as np
try:
    import faiss
except ImportError:
    faiss = None

from sentence_transformers import SentenceTransformer

from src.config import (
    ALLOWED_EXTENSIONS,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    DOCS_DIR,
    EMBEDDING_MODEL,
    MAX_DOC_SNIPPET,
    MAX_DOCS,
)

EMBEDDING_MODEL_INSTANCE = None


def get_embedding_model() -> SentenceTransformer:
    global EMBEDDING_MODEL_INSTANCE
    if EMBEDDING_MODEL_INSTANCE is None:
        EMBEDDING_MODEL_INSTANCE = SentenceTransformer(EMBEDDING_MODEL)
    return EMBEDDING_MODEL_INSTANCE


def extract_text_from_pdf(path: Path) -> str:
    try:
        reader = PdfReader(path)
        text_parts = []
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text)
        return "\n\n".join(text_parts).strip()
    except Exception:
        return ""


def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    words = text.split()
    if not words:
        return []

    chunks = []
    start = 0
    while start < len(words):
        end = min(start + size, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start += size - overlap
    return chunks


def load_documents(directory: str = DOCS_DIR) -> List[Dict[str, str]]:
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


def supported_upload_extension(filename: str) -> bool:
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS


def score_document(query: str, document: Dict[str, str]) -> int:
    query_tokens = set(re.findall(r"\w+", query.lower()))
    doc_tokens = set(re.findall(r"\w+", document["text"].lower()))
    return len(query_tokens & doc_tokens)


def retrieve_documents(query: str, documents: List[Dict[str, str]], top_k: int = MAX_DOCS) -> List[Dict[str, str]]:
    scored = [{"score": score_document(query, doc), "doc": doc} for doc in documents]
    scored = [item for item in scored if item["score"] > 0]
    scored.sort(key=lambda item: item["score"], reverse=True)
    return [item["doc"] for item in scored[:top_k]]


def build_embedding_index(documents: List[Dict[str, str]]):
    if not faiss:
        return None, []

    chunks = []
    texts = []
    for doc in documents:
        for chunk in chunk_text(doc["text"]):
            chunks.append({"text": chunk, "doc_name": doc["name"]})
            texts.append(chunk)

    if not texts:
        return None, []

    model = get_embedding_model()
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return index, chunks


def retrieve_similar_documents(query: str, documents: List[Dict[str, str]], top_k: int = MAX_DOCS) -> List[Dict[str, str]]:
    index, chunks = build_embedding_index(documents)
    if index is None or not chunks:
        return retrieve_documents(query, documents, top_k)

    model = get_embedding_model()
    query_embedding = model.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(query_embedding)
    _, ids = index.search(query_embedding, min(len(chunks), top_k * 3))

    found = []
    seen = set()
    for idx in ids[0]:
        if idx < 0 or idx >= len(chunks):
            continue
        doc_name = chunks[idx]["doc_name"]
        if doc_name in seen:
            continue
        seen.add(doc_name)
        for doc in documents:
            if doc["name"] == doc_name:
                found.append(doc)
                break
        if len(found) >= top_k:
            break

    if found:
        return found
    return retrieve_documents(query, documents, top_k)


def build_rag_context(documents: List[Dict[str, str]]) -> Optional[str]:
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
