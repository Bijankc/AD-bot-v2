"""Frame store: imported images (named by content hash) + a manifest + append-only label files.

Append-only JSONL keeps every labelling decision; the latest entry for an id wins. Nothing here is ever committed.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

from PIL import Image

FULL_SIZE = (1920, 1080)


@dataclass
class FrameRecord:
    id: str                    # first 16 hex chars of sha256 of the file
    file: str                  # name inside raw_dir
    source: str                # e.g. "old_shots"
    orig_name: str
    width: int
    height: int
    full_frame: bool           # 1920x1080 screen frame (else a crop)
    time: str | None = None    # ISO time the frame was taken, if known
    reason: str | None = None  # old bot's screenshot reason
    hint: str | None = None    # suggested screen class (weak)
    context: list[str] | None = None  # old bot's log lines before the shot
    root: str = "raw"          # "raw": file is in raw_dir; "captures": file is <day>/<name> under the captures dir
    rect: list[int] | None = None  # captured client rect (left, top, width, height), physical pixels
    dpi: int | None = None
    rule: str | None = None    # weak-label rule that gave the hint


def frame_path(rec: FrameRecord, home: Path) -> Path:
    """Where a record's image lives (home = ADBOT2_HOME)."""
    if rec.root == "captures":
        return home / "captures" / rec.file
    return home / "data" / "raw" / rec.file


def file_id(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def append_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def load_manifest(path: Path) -> dict[str, FrameRecord]:
    return {r["id"]: FrameRecord(**r) for r in read_jsonl(path)}


def import_image(src: Path, raw_dir: Path, manifest: Path, source: str, *, time: datetime | None = None,
                 reason: str | None = None, hint: str | None = None, context: list[str] | None = None,
                 known: dict[str, FrameRecord] | None = None) -> tuple[FrameRecord, bool]:
    """Copy one image into the store. Returns (record, added); a duplicate (same bytes) is not added again."""
    known = load_manifest(manifest) if known is None else known
    fid = file_id(src)
    if fid in known:
        return known[fid], False
    with Image.open(src) as im:
        w, h = im.size
    raw_dir.mkdir(parents=True, exist_ok=True)
    name = fid + src.suffix.lower()
    shutil.copy2(src, raw_dir / name)
    rec = FrameRecord(fid, name, source, src.name, w, h, (w, h) == FULL_SIZE,
                      time.isoformat() if time else None, reason, hint, context)
    append_jsonl(manifest, [asdict(rec)])
    known[fid] = rec
    return rec, True


class LabelStore:
    """Screen labels: append-only JSONL, latest row per id wins."""

    def __init__(self, path: Path):
        self.path = path
        self._latest: dict[str, dict] = {}
        for r in read_jsonl(path):
            self._latest[r["id"]] = r

    def get(self, fid: str) -> str | None:
        r = self._latest.get(fid)
        return r["screen"] if r else None

    def set(self, fid: str, screen: str, by: str = "human") -> None:
        row = {"id": fid, "screen": screen, "by": by, "at": datetime.now().isoformat(timespec="seconds")}
        append_jsonl(self.path, [row])
        self._latest[fid] = row

    def all(self) -> dict[str, str]:
        return {k: v["screen"] for k, v in self._latest.items()}
