import time

import numpy as np

from adbot2.perception import Observation
from adbot2.tracker import Tracker

C = ["fight", "lobby"]
T0 = 1_800_000_000.0 - time.localtime(1_800_000_000).tm_gmtoff  # a whole local hour: second-of-hour = offset from T0


def obs(**kw) -> Observation:
    o = Observation(screen_probs=np.array([0.95, 0.05]))
    for k, v in kw.items():
        setattr(o, k, v)
    return o


def test_auto_off_needs_three_confident_frames():
    tr = Tracker(C)
    assert not tr.update(obs(auto_p_off=0.99), T0).auto_off
    assert not tr.update(obs(auto_p_off=0.99), T0 + 0.3).auto_off
    assert tr.update(obs(auto_p_off=0.99), T0 + 0.6).auto_off
    assert not tr.update(obs(auto_p_off=0.5), T0 + 0.9).auto_off       # one unsure frame: not off any more
    assert not tr.update(obs(auto_p_off=0.99), T0 + 1.2).auto_off


def test_auto_off_only_counts_in_a_fight():
    tr = Tracker(C)
    lobby = np.array([0.05, 0.95])
    for k in range(5):
        assert not tr.update(obs(auto_p_off=0.99, screen_probs=lobby), T0 + k * 0.3).auto_off


def test_enemy_count_flicker_and_cleared():
    tr = Tracker(C)
    assert tr.update(obs(enemies=True, enemy_count=1), T0).enemy_count is None   # one reading: not yet
    for k, n in enumerate([1, 0, 1, 1]):                                # 1 -> 0 -> 1 flicker between waves
        f = tr.update(obs(enemies=True, enemy_count=n), T0 + 1 + k)
        assert f.enemy_count == 1 and not f.cleared
    for k in range(6):
        f = tr.update(obs(enemies=True, enemy_count=0), T0 + 10 + k)
    assert f.enemy_count == 0 and f.cleared
    assert tr.update(obs(enemies=False), T0 + 17).enemy_count is None


def test_raid_clock_needs_two_agreeing_readings():
    tr = Tracker(C)
    # at second 600 of the hour the card says 35:00 -> opens at second 2700 (:45:00)
    assert tr.update(obs(menu_clock=(35, 0)), T0 + 600).raid_opens_at is None
    assert tr.update(obs(menu_clock=(3, 30)), T0 + 610).raid_opens_at is None   # a dropped digit: no agreement
    assert tr.update(obs(menu_clock=(34, 49)), T0 + 611).raid_opens_at is None
    assert tr.update(obs(menu_clock=(34, 48)), T0 + 612).raid_opens_at == 2700


def test_screen_unknown_until_convincing():
    tr = Tracker(C)
    assert tr.update(obs(screen_probs=np.array([0.55, 0.45])), T0).screen is None
    assert tr.update(obs(), T0 + 0.3).screen == "fight"


def test_group_finder_queue_is_debounced():
    tr = Tracker(C)
    assert not tr.update(obs(gf_banner=True), T0).gf_queued          # one banner frame: not yet
    assert tr.update(obs(gf_banner=True), T0 + 2).gf_queued
    assert tr.update(obs(gf_banner=False), T0 + 4).gf_queued         # dimmed by a window for a moment
    assert tr.update(obs(gf_banner=None), T0 + 8).gf_queued         # not read: no change
    assert not tr.update(obs(gf_banner=False), T0 + 13).gf_queued    # gone for 10 s


def test_raid_clock_is_in_local_time(monkeypatch):
    import adbot2.tracker as tracker
    monkeypatch.setattr(tracker, "local_time", lambda t: t + 5 * 3600 + 45 * 60)
    tr = Tracker(C)
    utc_whole_hour = 1_800_000_000
    tr.update(obs(menu_clock=(2, 0)), utc_whole_hour - 120)     # 2:00 left at UTC :58 = local :43
    assert tr.update(obs(menu_clock=(1, 58)), utc_whole_hour - 118).raid_opens_at == 2700


def test_start_button_needs_two_frames():
    tr = Tracker(C)
    assert not tr.update(obs(start_button=True), T0).start_shown
    assert tr.update(obs(start_button=True), T0 + 0.3).start_shown
    assert not tr.update(obs(start_button=False), T0 + 0.6).start_shown


def test_still_since_follows_picture_changes():
    tr = Tracker(C)
    a, b = np.zeros((18, 32), np.uint8), np.full((18, 32), 40, np.uint8)
    assert tr.update(obs(thumb=a), T0).still_since == T0
    assert tr.update(obs(thumb=a + 1), T0 + 60).still_since == T0          # tiny change: still
    assert tr.update(obs(thumb=b), T0 + 120).still_since == T0 + 120
