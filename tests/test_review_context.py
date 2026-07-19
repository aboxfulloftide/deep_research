from web.kb_routes import _evidence_summary


async def test_bulk_claim_evidence_includes_bounded_surrounding_context(kb_db):
    claim, _ = await kb_db.get_or_create_claim("fact", "The key sentence.")
    source, _ = await kb_db.get_or_create_source(
        source_type_code="web",
        canonical_uri="https://context.example/article",
        canonical_key="web:https://context.example/article",
        title="Context source",
    )
    version, _ = await kb_db.add_source_version(
        source["id"], content_hash="context-hash", snapshot_path="/tmp/context",
        http_status=200, mime_type="text/plain",
    )
    artifact, _ = await kb_db.upsert_artifact(
        artifact_id="context-artifact", source_version_id=version["id"],
        artifact_type="clean_text", storage_path="/tmp/context.txt",
        content_hash="context-hash", chunk_params_hash="context-params",
    )
    before = "Earlier explanation. " * 30
    quote = "The key sentence."
    after = " Later explanation." * 60
    chunk_text = before + quote + after
    chunk = await kb_db.add_chunk(artifact["id"], 0, chunk_text, "context-chunk-hash")
    await kb_db.add_claim_evidence(
        claim_id=claim["id"], artifact_chunk_id=chunk["id"], source_id=source["id"],
        source_version_id=version["id"], excerpt_text=quote,
        char_start=len(before), char_end=len(before) + len(quote),
    )

    rows = (await kb_db.get_claims_evidence_bulk([claim["id"]]))[claim["id"]]

    assert len(rows[0]["context_excerpt"]) <= 1000
    assert "Earlier explanation." in rows[0]["context_excerpt"]
    assert quote in rows[0]["context_excerpt"]
    assert "Later explanation." in rows[0]["context_excerpt"]
    assert rows[0]["context_truncated_start"] is True
    assert rows[0]["context_truncated_end"] is True


def test_evidence_summary_exposes_source_links_and_marks_clipped_context():
    result = _evidence_summary([{
        "source_id": "source-1",
        "source_title": "A source",
        "canonical_uri": "https://example.com/source",
        "excerpt_text": "Exact quote.",
        "context_excerpt": "Words around the exact quote.",
        "context_truncated_start": True,
        "context_truncated_end": True,
        "section_label": "Speaker 2",
    }])

    assert result == [{
        "source_id": "source-1",
        "source_title": "A source",
        "canonical_uri": "https://example.com/source",
        "excerpt": "Exact quote.",
        "context": "…Words around the exact quote.…",
        "section_label": "Speaker 2",
    }]
