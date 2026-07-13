from deep_research.kb.conversation import get_topic_conversation_transcript, parse_conversation_turns


def test_parses_alternating_speaker_turns():
    text = (
        "User: Is the Eiffel Tower taller than the Statue of Liberty?\n"
        "Assistant: Yes, the Eiffel Tower is about 330 meters tall.\n"
        "User: What about the Great Pyramid?\n"
        "Assistant: The Great Pyramid is about 146 meters tall.\n"
    )
    turns = parse_conversation_turns(text)
    assert [speaker for speaker, _ in turns] == ["User", "Assistant", "User", "Assistant"]
    assert "Eiffel Tower" in turns[0][1]
    assert "330 meters" in turns[1][1]


def test_multiline_turn_stays_with_its_speaker():
    text = "Alice: This is line one\nand this continues the same turn.\nBob: A reply.\n"
    turns = parse_conversation_turns(text)
    assert turns[0] == ("Alice", "This is line one\nand this continues the same turn.")
    assert turns[1] == ("Bob", "A reply.")


def test_plain_prose_with_no_speaker_lines_returns_one_untagged_turn():
    text = "Just a plain paragraph with no speakers at all."
    assert parse_conversation_turns(text) == [(None, text)]


def test_empty_text_returns_no_turns():
    assert parse_conversation_turns("") == []
    assert parse_conversation_turns("   \n  ") == []


# -- get_topic_conversation_transcript: needs a real DB ----------------------

async def _make_chunked_source(
    kb_db, *, canonical_uri: str, turns: list[tuple[str | None, str]], artifact_type: str = "conversation_turns",
) -> tuple[dict, dict, list[dict]]:
    """Returns (source, version, chunks) -- version and chunks are needed by
    callers that also want to attach claim evidence to a specific turn."""
    source, _ = await kb_db.get_or_create_source(
        source_type_code="web", canonical_uri=canonical_uri, canonical_key=canonical_uri,
    )
    version, _ = await kb_db.add_source_version(
        source["id"], content_hash="h1", snapshot_path="/tmp/x", http_status=200, mime_type="text/html",
    )
    artifact, _ = await kb_db.upsert_artifact(
        artifact_id=f"art-{canonical_uri}", source_version_id=version["id"], artifact_type=artifact_type,
        storage_path="/tmp/x.txt", content_hash="h1", chunk_params_hash="p1",
    )
    chunks = [
        await kb_db.add_chunk(artifact["id"], i, text, f"chash-{canonical_uri}-{i}", section_label=speaker)
        for i, (speaker, text) in enumerate(turns)
    ]
    return source, version, chunks


async def test_transcript_returns_ordered_turns_for_a_conversation_source(kb_db):
    topic = await kb_db.create_topic("A Conversation", topic_type="conversation")
    source, _, _ = await _make_chunked_source(
        kb_db, canonical_uri="conversation:test-1",
        turns=[("User", "Is the sky blue?"), ("Assistant", "Yes, due to Rayleigh scattering.")],
    )
    await kb_db.attach_source_to_topic(topic["id"], source["id"])

    turns = await get_topic_conversation_transcript(kb_db, topic["id"])

    assert [(t["speaker"], t["text"]) for t in turns] == [
        ("User", "Is the sky blue?"),
        ("Assistant", "Yes, due to Rayleigh scattering."),
    ]


async def test_transcript_includes_untagged_turns_from_a_conversation_source(kb_db):
    # A paste with no recognizable "Name:" lines at all still gets ingested
    # as artifact_type="conversation_turns" (parse_conversation_turns falls
    # back to one untagged turn, speaker=None) -- it must still show up,
    # just without a speaker label, not be silently dropped.
    topic = await kb_db.create_topic("Untagged Conversation", topic_type="conversation")
    source, _, _ = await _make_chunked_source(
        kb_db, canonical_uri="conversation:untagged", turns=[(None, "Some pasted text with no speaker lines.")],
    )
    await kb_db.attach_source_to_topic(topic["id"], source["id"])

    turns = await get_topic_conversation_transcript(kb_db, topic["id"])

    assert [(t["speaker"], t["text"]) for t in turns] == [(None, "Some pasted text with no speaker lines.")]


async def test_transcript_skips_non_conversation_sources(kb_db):
    # A regular (non-conversation) attached source's artifact_type is
    # "clean_text", not "conversation_turns" -- it must not show up in the
    # transcript at all, even if (as here) it never happens to set a
    # section_label either.
    topic = await kb_db.create_topic("Mixed Topic")
    plain_source, _, _ = await _make_chunked_source(
        kb_db, canonical_uri="https://example.com/article", turns=[(None, "Some article text.")],
        artifact_type="clean_text",
    )
    await kb_db.attach_source_to_topic(topic["id"], plain_source["id"])

    turns = await get_topic_conversation_transcript(kb_db, topic["id"])

    assert turns == []


async def test_transcript_includes_claims_evidenced_from_their_turn(kb_db):
    topic = await kb_db.create_topic("Conversation With Claims", topic_type="conversation")
    source, version, chunks = await _make_chunked_source(
        kb_db, canonical_uri="conversation:with-claims",
        turns=[("User", "The sky is blue."), ("Assistant", "That's correct, due to Rayleigh scattering.")],
    )
    claim, _ = await kb_db.get_or_create_claim("fact", "The sky is blue.")
    await kb_db.add_claim_evidence(
        claim_id=claim["id"], artifact_chunk_id=chunks[0]["id"], source_id=source["id"], source_version_id=version["id"],
    )
    # Evidence must exist before attach -- attach_source_to_topic auto-attaches
    # every claim already evidenced by the source at that moment.
    await kb_db.attach_source_to_topic(topic["id"], source["id"])

    turns = await get_topic_conversation_transcript(kb_db, topic["id"])

    assert [c["id"] for c in turns[0]["claims"]] == [claim["id"]]
    assert turns[1]["claims"] == []


async def test_transcript_excludes_claims_rejected_from_the_topic(kb_db):
    # A claim can have evidence anchored to a turn's chunk without actually
    # belonging to this topic (e.g. a human reviewed and rejected it) -- it
    # must not resurface inline just because the chunk still references it.
    topic = await kb_db.create_topic("Conversation With Rejected Claim", topic_type="conversation")
    source, version, chunks = await _make_chunked_source(
        kb_db, canonical_uri="conversation:with-rejected-claim", turns=[("User", "A tangential remark.")],
    )
    claim, _ = await kb_db.get_or_create_claim("fact", "A tangential remark.")
    await kb_db.add_claim_evidence(
        claim_id=claim["id"], artifact_chunk_id=chunks[0]["id"], source_id=source["id"], source_version_id=version["id"],
    )
    await kb_db.attach_source_to_topic(topic["id"], source["id"])
    await kb_db.review_topic_claim_link(topic["id"], claim["id"], "rejected")

    turns = await get_topic_conversation_transcript(kb_db, topic["id"])

    assert turns[0]["claims"] == []


async def test_transcript_concatenates_multiple_conversations_oldest_first(kb_db):
    topic = await kb_db.create_topic("Multi-conversation Topic", topic_type="conversation")
    first, _, _ = await _make_chunked_source(
        kb_db, canonical_uri="conversation:first", turns=[("User", "First conversation.")],
    )
    second, _, _ = await _make_chunked_source(
        kb_db, canonical_uri="conversation:second", turns=[("User", "Second conversation.")],
    )
    # Attach in reverse order -- source creation time (not attach order)
    # should still put "first" ahead of "second".
    await kb_db.attach_source_to_topic(topic["id"], second["id"])
    await kb_db.attach_source_to_topic(topic["id"], first["id"])

    turns = await get_topic_conversation_transcript(kb_db, topic["id"])

    assert [t["text"] for t in turns] == ["First conversation.", "Second conversation."]
