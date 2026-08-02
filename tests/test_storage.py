import gzip

from deep_research.kb.storage import SnapshotStore


def test_archive_verification_evidence_compresses_and_preserves_content(tmp_path):
    store = SnapshotStore(tmp_path / "snapshots")
    original = store.write("source-1", 1, b"<html>some fetched page content</html>", ".html")

    archived = store.archive_verification_evidence(original, "claim-123", ".html")

    assert archived.exists()
    assert archived.name.endswith(".html.gz")
    assert gzip.decompress(archived.read_bytes()) == b"<html>some fetched page content</html>"


def test_archive_verification_evidence_is_keyed_by_claim_id(tmp_path):
    store = SnapshotStore(tmp_path / "snapshots")
    original = store.write("source-1", 1, b"content", ".html")

    archived = store.archive_verification_evidence(original, "claim-abc", ".html")

    assert "claim-abc" in archived.parts


def test_archive_verification_evidence_normalizes_extension_without_dot(tmp_path):
    store = SnapshotStore(tmp_path / "snapshots")
    original = store.write("source-1", 1, b"content", ".html")

    archived = store.archive_verification_evidence(original, "claim-abc", "html")

    assert archived.name.endswith(".html.gz")


def test_archive_verification_evidence_does_not_touch_the_original(tmp_path):
    store = SnapshotStore(tmp_path / "snapshots")
    original = store.write("source-1", 1, b"content", ".html")

    store.archive_verification_evidence(original, "claim-abc", ".html")

    assert original.exists()
    assert original.read_bytes() == b"content"
