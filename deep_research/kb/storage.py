"""Raw snapshot storage on disk.

Per decision 9 in PLAN_KB_ARCHITECTURE.md: raw sources and snapshots live as
files on disk; the database only stores paths, hashes, and metadata.
"""

import gzip
import uuid
from pathlib import Path


class SnapshotStore:
    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def path_for(self, source_id: str, version_number: int, ext: str) -> Path:
        ext = ext if ext.startswith(".") else f".{ext}"
        return self.root / source_id / f"v{version_number}{ext}"

    def write(self, source_id: str, version_number: int, content: bytes, ext: str) -> Path:
        path = self.path_for(source_id, version_number, ext)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        return path

    def write_artifact(self, artifact_id: str, content: bytes, ext: str = ".txt") -> Path:
        ext = ext if ext.startswith(".") else f".{ext}"
        path = self.root / "artifacts" / f"{artifact_id}{ext}"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        return path

    def archive_verification_evidence(self, original_path: Path | str, claim_id: str, ext: str) -> Path:
        """Gzip-compresses and copies a fetched page's snapshot to a
        permanent location keyed by the claim it helped verify, independent
        of the transient source/version row that fetched it (which gets
        deleted once verification finishes -- see kb.verification's
        "claims only live on sources the user added" rule). Lets a settled
        supported/contradicted verdict still point back to the original page
        even if the live site later changes or disappears, without keeping
        the source around in the KB. HTML/text compresses well (typically
        70-90% smaller), and only claims that actually settle on this
        evidence get archived at all -- the (much larger) set of fetched-
        but-never-used pages are just deleted, not compressed and kept.
        """
        ext = ext if ext.startswith(".") else f".{ext}"
        content = Path(original_path).read_bytes()
        dest = self.root / "verification_evidence" / claim_id / f"{uuid.uuid4().hex}{ext}.gz"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(gzip.compress(content))
        return dest

    def delete(self, path: Path | str) -> None:
        p = Path(path)
        p.unlink(missing_ok=True)
