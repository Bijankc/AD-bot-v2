from datetime import datetime

import numpy as np

from adbot2.perception import Observation
from adbot2.shadowing import LiveShadow, LogFollower, OldBotState
from adbot2.oldlog import LogLine


def line(text: str, h=10, m=0, s=0) -> LogLine:
    return LogLine(datetime(2026, 11, 20, h, m, s), text)


def test_old_bot_state_follows_its_log():
    st = OldBotState()
    st.feed(line("start v0.4.1-150, order: 5 4, loadouts=1, raid=1, infinity=1, speed=0, raidEntry=gf, soloRaid=0"))
    assert st.settings.speed is False and st.settings.infinity is True
    st.feed(line("loadout 3: opening Bag"))
    st.feed(line("Speed Raid: start (character x=521 y=693, run 17 of 50 this week)"))
    assert (st.mode, st.loadout) == ("speed", 3)
    st.feed(line("goal: infinity (next Speed Raid at 10:30)", m=1))
    assert st.speed_due == datetime(2026, 11, 20, 10, 30).timestamp()
    st.feed(line("pressing PLAY", m=2))
    assert st.mode is None


def test_log_follower_reads_only_new_whole_lines(tmp_path):
    f = tmp_path / "raid.log"
    f.write_bytes("20.11 10:00:00.000  old line\n".encode())
    fol = LogFollower(f)                       # starts at the end
    assert fol.new_lines() == []
    with open(f, "ab") as h:
        h.write("20.11 10:00:01.000  pressing PLAY\n20.11 10:00:02".encode())
    got = fol.new_lines()
    assert [ln.text for ln in got] == ["pressing PLAY"]
    with open(f, "ab") as h:
        h.write(".000  entering Dimension\n".encode())
    assert [ln.text for ln in fol.new_lines()] == ["entering Dimension"]
    f.write_bytes("20.11 10:05:00.000  start v1\n".encode())   # rotated: a new, shorter file
    assert [ln.text for ln in fol.new_lines()] == ["start v1"]


def test_live_shadow_writes_on_change_and_restarts_tracker_after_a_gap():
    classes = ["fight", "lobby"]
    sh = LiveShadow(classes, opens=2700)
    lobby = Observation(screen_probs=np.array([0.02, 0.98]))
    t = 1_800_000_000.0
    writes = [sh.step(lobby, t + k * 0.3)[2] for k in range(10)]
    assert writes[0] and sum(writes) <= 3                       # first row, then only on a change
    f, a, w = sh.step(lobby, t + 100)                            # after a gap
    assert sh.tracker.facts.raid_opens_at == 2700
