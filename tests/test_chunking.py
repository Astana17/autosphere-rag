"""Tests for recursive and article-aware chunking strategies."""

from __future__ import annotations

from app.chunking import chunk_by_article, chunk_recursive, run_chunking, summarize_chunks


SAMPLE_RECORDS = [
    {
        "document_id": "road_traffic_law",
        "document_title": "Закон о дорожном движении",
        "source_file": "road_traffic_law.docx",
        "chapter": "Глава 1. ОБЩИЕ ПОЛОЖЕНИЯ",
        "article": "Статья 1",
        "title": "Основные понятия",
        "text": "Статья 1. Основные понятия\n1) дорога - ...\n2) водитель - ...",
        "language": "ru",
        "jurisdiction": "Kazakhstan",
        "document_version": "2026-07",
        "source_url": "https://example.test/law",
        "article_id": "road_traffic_law_1",
    },
    {
        "document_id": "road_traffic_law",
        "document_title": "Закон о дорожном движении",
        "source_file": "road_traffic_law.docx",
        "chapter": "Глава 1. ОБЩИЕ ПОЛОЖЕНИЯ",
        "article": "Статья 2",
        "title": "Законодательство",
        "text": "Статья 2. Законодательство\nНастоящий Закон регулирует отношения.",
        "language": "ru",
        "jurisdiction": "Kazakhstan",
        "document_version": "2026-07",
        "source_url": "https://example.test/law",
        "article_id": "road_traffic_law_2",
    },
    {
        "document_id": "road_traffic_rules",
        "document_title": "Правила дорожного движения",
        "source_file": "road_traffic_rules.docx",
        "chapter": "Глава 8. Остановка",
        "article": "Пункт 84",
        "title": "Глава 8. Остановка",
        "text": "84. Остановка и стоянка транспортных средств запрещаются.",
        "language": "ru",
        "jurisdiction": "Kazakhstan",
        "document_version": "2026-07",
        "source_url": "https://example.test/rules",
        "article_id": "road_traffic_rules_84",
    },
]


def test_article_chunking_preserves_units_and_metadata():
    chunks = chunk_by_article(SAMPLE_RECORDS, max_chars=1500)
    assert len(chunks) == 3
    ids = {c["chunk_id"] for c in chunks}
    assert "road_traffic_law_1" in ids
    assert "road_traffic_rules_84" in ids
    for chunk in chunks:
        assert chunk["strategy"] == "article"
        assert chunk["source_file"]
        assert chunk["section_title"]
        assert chunk["char_start"] is not None
        assert chunk["char_end"] is not None
        assert chunk["document_version"] == "2026-07"
        assert chunk["parent_article_id"]


def test_article_chunking_subsplits_long_units():
    long_body = "Абзац о правилах. " * 200  # well over 1500 chars
    record = {
        **SAMPLE_RECORDS[0],
        "text": f"Статья 1. Основные понятия\n{long_body}",
        "article_id": "road_traffic_law_1",
    }
    chunks = chunk_by_article([record], max_chars=500, chunk_overlap=50)
    assert len(chunks) > 1
    assert all(c["chunk_id"].startswith("road_traffic_law_1_p") for c in chunks)
    assert all(c["parent_article_id"] == "road_traffic_law_1" for c in chunks)
    assert all("Статья 1" in c["text"] for c in chunks)
    assert chunks[0]["part_count"] == len(chunks)


def test_recursive_chunking_baseline():
    chunks = chunk_recursive(SAMPLE_RECORDS, chunk_size=120, chunk_overlap=20)
    assert len(chunks) >= 2
    assert all(c["strategy"] == "recursive" for c in chunks)
    assert all(c["chunk_id"].startswith("road_traffic_") for c in chunks)
    # Both documents should appear.
    docs = {c["document_id"] for c in chunks}
    assert docs == {"road_traffic_law", "road_traffic_rules"}


def test_run_chunking_dispatch_and_summary():
    article = run_chunking(SAMPLE_RECORDS, strategy="article")
    recursive = run_chunking(SAMPLE_RECORDS, strategy="recursive", chunk_size=200)
    assert summarize_chunks(article)["count"] == 3
    assert summarize_chunks(recursive)["count"] >= 1
    assert "recursive" in summarize_chunks(recursive)["strategies"]
