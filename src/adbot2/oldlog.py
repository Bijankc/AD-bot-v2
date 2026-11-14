"""Read the old AHK bot's logs (raid.log, logs/raid_YYYYMMDD.log) as weak labels for its screenshots.

Line format: ``dd.MM HH:mm:ss.mmm<spaces>message``. Screenshot lines look like
``screenshot: <path>\\<yyyyMMdd_HHmmss>_<reason>.png``. The lines just before a screenshot say what the old bot believed
was happening, which is a useful *hint* for the labeller, never a label by itself (a shot named "restart" can show the lobby).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path, PureWindowsPath

LINE_RE = re.compile(r"^(\d{2})\.(\d{2}) (\d{2}):(\d{2}):(\d{2})\.(\d{3})\s+(.*)$")
SHOT_NAME_RE = re.compile(r"^(\d{8})_(\d{6})_(.+)\.png$", re.IGNORECASE)


@dataclass(frozen=True)
class LogLine:
    time: datetime
    text: str


@dataclass
class ShotContext:
    name: str
    time: datetime
    context: list[str] = field(default_factory=list)  # preceding log messages, oldest first


def parse_line(line: str, year: int) -> LogLine | None:
    m = LINE_RE.match(line.lstrip("﻿").rstrip("\r\n"))
    if not m:
        return None
    dd, mo, hh, mi, ss, ms, text = m.groups()
    t = datetime(year, int(mo), int(dd), int(hh), int(mi), int(ss), int(ms) * 1000)
    return LogLine(t, text.strip())


def read_log(path: Path, year: int) -> list[LogLine]:
    out = []
    with open(path, encoding="utf-8-sig", errors="replace") as f:
        for raw in f:
            ln = parse_line(raw, year)
            if ln:
                out.append(ln)
    return out


def shot_reason(name: str) -> str | None:
    """'20261114_095321_raid_end.png' -> 'raid_end'."""
    m = SHOT_NAME_RE.match(name)
    return m.group(3) if m else None


def shot_time(name: str) -> datetime | None:
    m = SHOT_NAME_RE.match(name)
    if not m:
        return None
    return datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S")


def screenshot_contexts(lines: list[LogLine], keep: int = 4) -> dict[str, ShotContext]:
    """Map screenshot file name -> the `keep` log messages before it (screenshot lines themselves skipped)."""
    out: dict[str, ShotContext] = {}
    recent: list[str] = []
    for ln in lines:
        if ln.text.startswith("screenshot:"):
            name = PureWindowsPath(ln.text.split(":", 1)[1].strip()).name
            out[name] = ShotContext(name, ln.time, recent[-keep:])
        else:
            recent.append(ln.text)
            if len(recent) > keep * 4:
                del recent[: len(recent) - keep]
    return out


def read_all_contexts(bot_dir: Path, keep: int = 4) -> dict[str, ShotContext]:
    """Contexts from logs/raid_*.log and raid.log of an old-bot folder."""
    out: dict[str, ShotContext] = {}
    files = sorted((bot_dir / "logs").glob("raid_*.log"))
    if (bot_dir / "raid.log").exists():
        files.append(bot_dir / "raid.log")
    for f in files:
        m = re.search(r"(\d{4})\d{4}", f.name)
        year = int(m.group(1)) if m else datetime.fromtimestamp(f.stat().st_mtime).year
        out.update(screenshot_contexts(read_log(f, year), keep))
    return out


# Reasons whose shot is reliably taken on one screen. Everything else gets no hint.
REASON_HINTS: dict[str, str] = {
    "raid_end": "results_raid",
    "raid_rewards": "results_raid",
    "raid_world": "raid_world",
    "titan_start": "fight_dimension",
}
REASON_PREFIX_HINTS: dict[str, str] = {
    "titan_leg": "fight_dimension",
    "xboss_leg": "fight_dimension",
}


def screen_hint(reason: str | None) -> str | None:
    if not reason:
        return None
    if reason in REASON_HINTS:
        return REASON_HINTS[reason]
    for prefix, hint in REASON_PREFIX_HINTS.items():
        if reason.startswith(prefix):
            return hint
    return None
