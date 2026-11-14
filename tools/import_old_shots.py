"""Import the old AHK bot's screenshots into the v2 frame store, with weak hints from its logs.

    python tools/import_old_shots.py --bot-dir "..\\AD-bot"

Read-only on the old bot's folder. Safe to run again: duplicates (same bytes) are skipped, so new shots are picked up
each time (the old bot keeps only its newest 100).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from adbot2 import paths
from adbot2.dataset import import_image, load_manifest
from adbot2.oldlog import read_all_contexts, screen_hint, shot_reason, shot_time


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--bot-dir", type=Path, required=True, help="old AD-bot folder (has shots/ and logs/)")
    args = ap.parse_args(argv)

    shots = args.bot_dir / "shots"
    contexts = read_all_contexts(args.bot_dir)
    known = load_manifest(paths.manifest_path())
    added = skipped = 0
    for p in sorted(shots.glob("*.png")):
        reason = shot_reason(p.name)
        ctx = contexts.get(p.name)
        _, new = import_image(p, paths.raw_dir(), paths.manifest_path(), "old_shots",
                              time=shot_time(p.name), reason=reason, hint=screen_hint(reason),
                              context=ctx.context if ctx else None, known=known)
        added += new
        skipped += not new
    print(f"added {added}, already had {skipped}, store has {len(known)} frames -> {paths.data_dir()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
