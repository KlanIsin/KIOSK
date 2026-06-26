import os
from pathlib import Path

from src.rag import (
    build_rag_context,
    chunk_text,
    extract_text_from_pdf,
    load_documents,
    retrieve_documents,
    retrieve_similar_documents,
    score_document,
    supported_upload_extension,
)

BASE_DIR = Path(__file__).resolve().parent.parent
TEMP_DOCS = BASE_DIR / "tests_data"


def setup_module():
    TEMP_DOCS.mkdir(exist_ok=True)
    (TEMP_DOCS / "sample.txt").write_text("This is a school policy document about attendance.", encoding="utf-8")


def teardown_module():
    for path in TEMP_DOCS.iterdir():
        path.unlink()
    TEMP_DOCS.rmdir()


def test_supported_upload_extension():
    assert supported_upload_extension("policy.txt")
    assert supported_upload_extension("schedule.pdf")
    assert not supported_upload_extension("image.png")


def test_load_documents_txt():
    docs = load_documents(str(TEMP_DOCS))
    assert len(docs) == 1
    assert docs[0]["name"] == "sample.txt"


def test_build_rag_context_empty():
    assert build_rag_context([]) is None


def test_score_document():
    doc = {"name": "sample.txt", "text": "attendance policy for school"}
    assert score_document("school attendance", doc) == 2


def test_retrieve_documents():
    docs = [{"name": "sample.txt", "text": "school attendance policy"}]
    results = retrieve_documents("attendance", docs)
    assert results[0]["name"] == "sample.txt"


def test_chunk_text():
    text = "one two three four five six seven eight"
    chunks = chunk_text(text, size=4, overlap=2)
    assert chunks[0] == "one two three four"
    assert chunks[1] == "three four five six"


def test_retrieve_similar_documents_fallback():
    docs = [{"name": "sample.txt", "text": "school attendance policy"}]
    results = retrieve_similar_documents("attendance", docs)
    assert len(results) == 1


def test_extract_text_from_pdf_no_file():
    assert extract_text_from_pdf(Path("nonexistent.pdf")) == ""
