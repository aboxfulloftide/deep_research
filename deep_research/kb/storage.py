"""Raw snapshot storage on disk.

Per decision 9 in PLAN_KB_ARCHITECTURE.md: raw sources and snapshots live as
files on disk; the database only stores paths, hashes, and metadata.
"""

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

    def delete(self, path: Path | str) -> None:
        p = Path(path)
        p.unlink(missing_ok=True)
