import time

from adbot2.policy import (GF_FALLBACK, INF_STOP, SPEED_EVERY, SPEED_MAX, Memory, Policy, Settings, decide, goal,
                           raid_hour, raid_time)
from adbot2.tracker import Facts

OPENS = 2700                 # :45:00
HOUR = 1_800_000_000 - time.localtime(1_800_000_000).tm_gmtoff   # a whole local hour (any time zone)


def at(raid_t: int, hour: int = 0) -> float:
    """Wall time at a given raid time (seconds since the raid opened)."""
    return HOUR + OPENS + raid_t + 3600 * hour


def facts(screen, **kw) -> Facts:
    return Facts(screen=screen, raid_opens_at=OPENS, **kw)


def mem(**kw) -> Memory:
    """Memory with the raid of hour 0 done and the right loadout unless a test says otherwise."""
    kw.setdefault("raid_done", raid_hour(at(0), OPENS))
    kw.setdefault("speed_due", at(10 ** 6))
    return Memory(**kw)


NO_LOADOUTS = Settings(loadouts=False)


def test_raid_time_and_hour():
    assert raid_time(HOUR + 2700, 2700) == 0
    assert raid_time(HOUR + 2690, 2700) == 3590
    assert raid_time(HOUR, None) is None
    # the raid hour turns at INF_STOP: 57:30 already belongs to the next raid
    assert raid_hour(at(5 * 60), OPENS) == raid_hour(at(INF_STOP - 3600 + 1), OPENS)
    assert raid_hour(at(INF_STOP), OPENS) == raid_hour(at(5 * 60), OPENS) + 1


def test_goal_follows_the_raid_hour():
    m = mem()
    assert goal(at(20 * 60), facts("lobby"), Settings(), m) == "infinity"
    assert goal(at(INF_STOP), facts("lobby"), Settings(), m) == "raid"
    assert goal(at(INF_STOP - 100), facts("lobby"), Settings(), m) == "raid"     # raid soon, not in a run
    assert goal(at(INF_STOP - 100), facts("fight"), Settings(), m) == "infinity"  # in a run: until INF_STOP
    assert goal(at(5 * 60), facts("lobby"), Settings(), m) == "infinity"         # this hour's raid is done
    assert goal(at(5 * 60), facts("lobby"), Settings(), Memory()) == "raid"      # not done yet
    assert goal(at(5 * 60), facts("lobby"), Settings(raid=False), m) == "infinity"
    assert goal(at(11 * 60), facts("lobby"), Settings(), Memory(speed_due=at(10 ** 6))) == "infinity"


def test_speed_raid_schedule():
    assert goal(at(20 * 60), facts("lobby"), Settings(), mem(speed_due=None)) == "speed"
    assert goal(at(20 * 60), facts("lobby"), Settings(), mem(speed_due=at(21 * 60))) == "infinity"
    assert goal(at(53 * 60), facts("lobby"), Settings(), mem(speed_due=None)) == "raid"      # raid soon wins
    assert goal(at(52 * 60), facts("fight"), Settings(), mem(speed_due=None)) == "infinity"
    assert goal(at(53 * 60), facts("lobby"), Settings(raid=False), mem(speed_due=None)) == "speed"


def test_infinity_loop():
    t, m = at(20 * 60), mem()
    assert decide(t, facts("lobby"), NO_LOADOUTS, m).name == "press_play"
    assert decide(t, facts("modes_menu"), NO_LOADOUTS, m).name == "open_dimension"
    assert decide(t, facts("dimension_window"), NO_LOADOUTS, m).name == "pick_infinite"
    assert decide(t, facts("party"), NO_LOADOUTS, m).name == "start"
    assert decide(t, facts("fight"), NO_LOADOUTS, m).name == "wait"
    assert decide(t, facts("fight", auto_off=True), NO_LOADOUTS, m).name == "press_auto"
    assert decide(t, facts("results_infinity"), NO_LOADOUTS, m).name == "leave"


def test_loadout_before_play():
    a = decide(at(20 * 60), facts("lobby"), Settings(), mem())
    assert (a.name, a.arg) == ("set_loadout", 2)
    assert decide(at(20 * 60), facts("lobby"), Settings(), mem(loadout=2)).name == "press_play"


def test_leave_infinity_only_when_entered_as_infinity():
    assert decide(at(INF_STOP + 5), facts("fight"), m=mem(mode="infinity")).name == "reset"
    assert decide(at(INF_STOP + 5), facts("fight"), m=mem()).name == "wait"            # not known to be Infinity
    assert decide(at(INF_STOP + 5), facts("fight"), m=mem(mode="raid")).name == "wait"
    assert decide(at(20 * 60), facts(None)).name == "wait"
    assert decide(at(20 * 60), facts("esc_menu"), m=mem()).name == "unhandled"


def test_leave_infinity_for_speed_raid_unless_waiting_pays():
    t = at(20 * 60)
    m = mem(mode="infinity", speed_due=t - 5, fight_since=t - 600)
    assert decide(t, facts("fight"), m=m).name == "reset"
    m = mem(mode="infinity", speed_due=t - 5, fight_since=t - 31 * 60)               # ends soon by itself
    assert decide(t, facts("fight"), m=m).name == "wait"
    m = mem(mode="infinity", speed_due=t - 13 * 60, fight_since=t - 31 * 60)         # waited long enough
    assert decide(t, facts("fight"), m=m).name == "reset"
    t = at(48 * 60)                                                                   # raid near: wait
    assert decide(t, facts("fight"), m=mem(mode="infinity", speed_due=t - 5, fight_since=t - 60)).name == "wait"


def test_group_finder_raid():
    m = Memory(loadout=1, speed_due=at(10 ** 6))
    t = at(INF_STOP + 30)
    assert decide(t, facts("lobby"), m=m).name == "press_play"
    assert decide(t, facts("modes_menu"), m=m).name == "open_raid"
    assert decide(t, facts("raid_world"), m=m).name == "press_play"
    assert decide(t, facts("raid_window"), m=m).name == "queue_gf"
    assert decide(t, facts("raid_world", gf_queued=True), m=m).name == "wait"
    assert decide(at(60), facts("raid_world", gf_queued=True), m=m).name == "wait"
    assert decide(at(GF_FALLBACK), facts("raid_world", gf_queued=True), m=m).name == "cancel_queue"
    # after the cancel, or once Group Finder is too late, the raid is created
    assert decide(at(GF_FALLBACK + 5), facts("raid_window"), m=m).name == "create_raid"
    assert decide(at(30), facts("raid_window"), Settings(raid_entry="create"), m).name == "create_raid"
    assert decide(t, facts("raid_window"), Settings(raid_entry="create"), m).name == "wait"   # not open yet


def test_raid_world_with_another_goal_restarts():
    a = decide(at(20 * 60), facts("raid_world"), m=mem())
    assert a.name == "restart_roblox"
    assert decide(at(20 * 60), facts("raid_window"), m=mem()).name == "close_window"


def test_speed_raid_entry_and_cap():
    t = at(20 * 60)
    m = mem(speed_due=None, loadout=3)
    assert decide(t, facts("lobby"), m=m).name == "press_play"
    assert decide(t, facts("modes_menu"), m=m).name == "open_speed"
    assert decide(t, facts("speed_raid_window"), m=m).name == "speed_start"
    assert decide(t, facts("speed_raid_window"), m=mem()).name == "close_window"
    m = mem(mode="speed", fight_since=t - SPEED_MAX - 1)
    assert decide(t, facts("fight"), m=m).name == "restart_roblox"


def test_policy_remembers_the_run():
    p = Policy(Settings(loadouts=False))
    p.m.raid_done = raid_hour(at(0), OPENS)
    p.m.speed_due = at(10 ** 6)
    p.step(at(20 * 60), facts("dimension_window"))
    assert p.mode == "infinity"
    p.step(at(20 * 60 + 30), facts("fight"))
    assert p.step(at(INF_STOP + 5), facts("fight")).name == "reset"
    p.step(at(INF_STOP + 60), facts("lobby"))
    assert p.mode is None and p.m.fight_since is None


def test_policy_marks_the_raid_done_and_times_speed_raid():
    p = Policy(Settings(loadouts=False))
    p.step(at(INF_STOP + 30), facts("raid_window"))           # queue_gf
    assert p.mode == "raid"
    p.step(at(3600 + 40), facts("fight"))                     # the raid of hour 1 begins
    assert p.m.raid_done == raid_hour(at(3600 + 40), OPENS)
    p.step(at(3600 + 500), facts("results_raid"))
    p.step(at(3600 + 520), facts("lobby"))
    assert goal(at(3600 + 530), facts("lobby"), p.s, p.m) == "speed"   # raid done, Speed Raid never run
    assert p.step(at(3600 + 540), facts("speed_raid_window")).name == "speed_start"
    p.step(at(3600 + 560), facts("fight"))
    p.step(at(3600 + 700), facts("lobby"))
    assert p.m.speed_due == at(3600 + 560) + SPEED_EVERY + 15
    assert goal(at(3600 + 710), facts("lobby"), p.s, p.m) == "infinity"


def test_dimension_goal_comes_before_infinity():
    s = Settings(dimension=True, loadouts=False)
    assert goal(at(20 * 60), facts("lobby"), s, mem()) == "dimension"
    assert decide(at(20 * 60), facts("modes_menu"), s, mem()).name == "open_dimension"
    assert decide(at(20 * 60), facts("dimension_window"), s, mem()).name == "unhandled"
    assert goal(at(INF_STOP), facts("lobby"), s, mem()) == "raid"


def test_raid_time_uses_the_local_hour(monkeypatch):
    """UTC+5:45 (this PC): the raid at local :45:00 opens at UTC :00:00."""
    import adbot2.policy as policy
    monkeypatch.setattr(policy, "local_time", lambda t: t + 5 * 3600 + 45 * 60)
    utc_whole_hour = 1_800_000_000
    assert policy.raid_time(utc_whole_hour, 2700) == 0
    assert policy.raid_time(utc_whole_hour - 150, 2700) == policy.INF_STOP


def test_results_leave_only_for_speed_raid():
    assert decide(at(20 * 60), facts("results_raid"), m=mem(mode="speed")).name == "leave"
    assert decide(at(5 * 60), facts("results_raid"), m=mem(mode="raid")).name == "wait"
