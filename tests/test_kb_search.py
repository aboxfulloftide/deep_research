from deep_research.config import Config
from deep_research.tools.kb_search import kb_search, kb_search_records


async def _make_chunked_source(kb_db, *, canonical_uri: str, chunk_text: str, page_number: int | None = None) -> dict:
    source, _ = await kb_db.get_or_create_source(
        source_type_code="web", canonical_uri=canonical_uri, canonical_key=canonical_uri,
    )
    version, _ = await kb_db.add_source_version(
        source["id"], content_hash=f"h-{canonical_uri}", snapshot_path="/tmp/x",
        http_status=200, mime_type="text/html",
    )
    artifact, _ = await kb_db.upsert_artifact(
        artifact_id=f"art-{canonical_uri}", source_version_id=version["id"], artifact_type="clean_text",
        storage_path="/tmp/x.txt", content_hash=f"h-{canonical_uri}", chunk_params_hash="p1",
    )
    await kb_db.add_chunk(
        artifact["id"], 0, chunk_text, f"chash-{canonical_uri}", page_number=page_number,
    )
    return source


async def test_kb_search_records_returns_structured_fields(kb_db, monkeypatch):
    await _make_chunked_source(
        kb_db, canonical_uri="https://example.test/widgets",
        chunk_text="Widgets are small mechanical devices used in many products.",
        page_number=3,
    )

    async def no_embeddings(*args, **kwargs):
        raise ConnectionError("embedding backend unreachable in this test")

    import deep_research.tools.kb_search as kb_search_module
    monkeypatch.setattr(kb_search_module, "embed_texts", no_embeddings)

    records = await kb_search_records("widgets", kb_db, Config())

    assert len(records) == 1
    record = records[0]
    assert record["title"] == "https://example.test/widgets"
    assert record["page_number"] == 3
    assert "Widgets" in record["chunk_text"]
    assert record["rrf_score"] > 0
    assert "chunk_id" in record


async def test_kb_search_formats_records_the_same_way_as_before(kb_db, monkeypatch):
    await _make_chunked_source(
        kb_db, canonical_uri="https://example.test/gadgets",
        chunk_text="Gadgets are small novel electronic devices.",
    )

    async def no_embeddings(*args, **kwargs):
        raise ConnectionError("embedding backend unreachable in this test")

    import deep_research.tools.kb_search as kb_search_module
    monkeypatch.setattr(kb_search_module, "embed_texts", no_embeddings)

    result = await kb_search("gadgets", kb_db, Config())

    assert "https://example.test/gadgets" in result
    assert "chunk 0" in result


async def test_kb_search_reports_no_results_for_an_unmatched_query(kb_db, monkeypatch):
    async def no_embeddings(*args, **kwargs):
        raise ConnectionError("embedding backend unreachable in this test")

    import deep_research.tools.kb_search as kb_search_module
    monkeypatch.setattr(kb_search_module, "embed_texts", no_embeddings)

    result = await kb_search("no such term anywhere zzyzx", kb_db, Config())

    assert result == "No results found in the local knowledge base."
