"""Clip replay: a recorded raid hour through tracker + policy must give the old bot's entry, in order.

Uses the newest replay (reports/replay) and the old bot's logs; skipped without real data.
"""

from datetime import datetime
from pathlib import Path

import pytest

from adbot2 import paths

BOT = Path(__file__).resolve().parents[2] / "AD-bot"


def actions_between(t0: str, t1: str) -> list[str]:
    import shadow
    from adbot2.oldlog import read_all_lines

    if not (paths.home() / "reports" / "replay" / "latest.json").exists() or not BOT.exists():
        pytest.skip("no replay or no old bot logs")
    window = (datetime.fromisoformat(t0).timestamp(), datetime.fromisoformat(t1).timestamp())
    decisions, _, _ = shadow.replay_decisions(2700, read_all_lines(BOT), window)
    out = []
    for _, name in decisions:
        if name not in ("wait", "unhandled") and (not out or out[-1] != name):
            out.append(name)
    return out


def in_order(seq: list[str], want: list[str]) -> bool:
    it = iter(seq)
    return all(w in it for w in want)


@pytest.mark.data
def test_group_finder_raid_hour_17_nov():
    # 15:40 Titan LEAVE -> loadout 1 -> PLAY -> raid world -> PLAY -> Group Finder -> queued -> raid at 15:45
    got = actions_between("2026-11-17T15:39:30", "2026-11-17T15:48:00")
    assert in_order(got, ["press_play", "open_raid", "press_play", "queue_gf"]), got
    assert "reset" not in got and "restart_roblox" not in got, got
