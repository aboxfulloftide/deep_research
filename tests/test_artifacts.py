import io

from pypdf import PdfWriter

from deep_research.kb.artifacts import build_artifact_for_version
from deep_research.kb.storage import SnapshotStore


def _minimal_pdf_bytes() -> bytes:
    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)
    buf = io.BytesIO()
    writer.write(buf)
    return buf.getvalue()


async def test_a_pdf_fetched_at_an_ordinary_url_is_routed_to_the_pdf_extractor(kb_db, tmp_path):
    """kb/ingest.py's ingest_web_page() always registers a web-fetched
    source as source_type_code="web" (the MIME type isn't known until after
    the source is created) -- build_artifact_for_version() must still route
    a PDF response to the pypdf extractor by checking the version's own
    stored mime_type, not the parent source's type."""
    snapshot_store = SnapshotStore(tmp_path)
    pdf_bytes = _minimal_pdf_bytes()

    source, _ = await kb_db.get_or_create_source(
        source_type_code="web", canonical_uri="https://example.test/paper",
        canonical_key="web:https://example.test/paper",
    )
    snapshot_path = snapshot_store.write(source["id"], 1, pdf_bytes, ".pdf")
    version, _ = await kb_db.add_source_version(
        source["id"], content_hash="h1", snapshot_path=str(snapshot_path),
        http_status=200, mime_type="application/pdf",
    )

    result = await build_artifact_for_version(kb_db, snapshot_store, source, version)

    # A blank pypdf-generated page has no extractable text, so chunk_count
    # is legitimately 0 ("empty") -- the routing itself (not chunk count) is
    # what this test verifies.
    assert result.status in ("chunked", "empty")
    artifact = await kb_db.get_artifact(result.artifact_id)
    assert artifact["artifact_type"] == "parsed_pdf"


async def test_an_html_response_at_a_web_source_is_still_routed_to_clean_text(kb_db, tmp_path):
    """Confirms the PDF-routing fix is additive -- an ordinary HTML web
    source is unaffected."""
    snapshot_store = SnapshotStore(tmp_path)
    html_bytes = b"<html><body><p>Hello world</p></body></html>"

    source, _ = await kb_db.get_or_create_source(
        source_type_code="web", canonical_uri="https://example.test/page",
        canonical_key="web:https://example.test/page",
    )
    snapshot_path = snapshot_store.write(source["id"], 1, html_bytes, ".html")
    version, _ = await kb_db.add_source_version(
        source["id"], content_hash="h2", snapshot_path=str(snapshot_path),
        http_status=200, mime_type="text/html",
    )

    result = await build_artifact_for_version(kb_db, snapshot_store, source, version)

    assert result.status == "chunked"
    artifact = await kb_db.get_artifact(result.artifact_id)
    assert artifact["artifact_type"] == "clean_text"
