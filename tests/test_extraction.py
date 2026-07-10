from deep_research.config import load_config
from deep_research.kb.extraction import run_extraction


async def _make_artifact_with_chunks(kb_db, chunk_texts):
    source, _ = await kb_db.get_or_create_source(
        source_type_code="web", canonical_uri="http://scoped-extraction.example", canonical_key="scoped",
    )
    version, _ = await kb_db.add_source_version(
        source["id"], content_hash="h1", snapshot_path="/tmp/scoped", http_status=200, mime_type="text/html",
    )
    artifact, _ = await kb_db.upsert_artifact(
        artifact_id="art-scoped", source_version_id=version["id"], artifact_type="clean_text",
        storage_path="/tmp/scoped.txt", content_hash="h1", chunk_params_hash="p1",
    )
    chunks = [
        await kb_db.add_chunk(artifact["id"], i, text, f"chash-{i}")
        for i, text in enumerate(chunk_texts)
    ]
    return artifact, chunks


async def test_run_extraction_with_chunk_ids_only_processes_those_chunks(kb_db, monkeypatch):
    """The bug this guards against: verify_claim's web-fallback only wants
    extraction run on the 1-3 chunks relevant to the claim being checked, not
    an entire page -- a real page extracted in full during this session
    produced 1072 tangential claims from one source. chunk_ids must actually
    restrict which chunks get sent to the extraction LLM."""
    import deep_research.kb.extraction as extraction_module

    artifact, chunks = await _make_artifact_with_chunks(kb_db, ["chunk zero", "chunk one", "chunk two"])

    seen_chunk_texts = []

    async def fake_chat(self, messages):
        seen_chunk_texts.append(messages[-1]["content"])
        return {"choices": [{"message": {"content": "[]"}}]}

    monkeypatch.setattr("deep_research.llm.LLMClient.chat", fake_chat)

    config = load_config()
    result = await run_extraction(kb_db, config, artifact["id"], chunk_ids=[chunks[1]["id"]])

    assert result.chunk_count == 1
    assert len(seen_chunk_texts) == 1
    assert "chunk one" in seen_chunk_texts[0]


async def test_partial_extraction_run_does_not_satisfy_full_extraction_cache(kb_db, monkeypatch):
    """A chunk-scoped (partial) run must never be mistaken later for "this
    artifact was already fully extracted" -- otherwise a real extract-source
    call could silently skip most of a page's chunks forever."""
    async def fake_chat(self, messages):
        return {"choices": [{"message": {"content": "[]"}}]}

    monkeypatch.setattr("deep_research.llm.LLMClient.chat", fake_chat)

    artifact, chunks = await _make_artifact_with_chunks(kb_db, ["chunk zero", "chunk one"])
    config = load_config()

    partial = await run_extraction(kb_db, config, artifact["id"], chunk_ids=[chunks[0]["id"]])
    assert partial.chunk_count == 1

    full = await run_extraction(kb_db, config, artifact["id"])
    assert full.status != "unchanged"
    assert full.chunk_count == 2
