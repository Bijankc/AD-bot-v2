import numpy as np

from adbot2.perception import Observation
from adbot2.tracker import Tracker

C = ["fight", "lobby"]
T0 = 1_800_000_000.0  # a whole hour in epoch seconds: second-of-hour = offset from T0


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
