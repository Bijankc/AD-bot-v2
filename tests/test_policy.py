from adbot2.policy import INF_STOP, Settings, decide, goal, raid_time
from adbot2.tracker import Facts

OPENS = 2700                 # :45:00
HOUR = 1_800_000_000         # a whole hour in epoch seconds


def at(raid_t: int) -> float:
    """Wall time at a given raid time (seconds since the raid opened)."""
    return HOUR + OPENS + raid_t


def facts(screen, **kw) -> Facts:
    return Facts(screen=screen, raid_opens_at=OPENS, **kw)


def test_raid_time_wraps():
    assert raid_time(HOUR + 2700, 2700) == 0
    assert raid_time(HOUR + 2690, 2700) == 3590
    assert raid_time(HOUR, None) is None


def test_goal_follows_the_raid_hour():
    assert goal(at(20 * 60), facts("lobby"), Settings()) == "infinity"
    assert goal(at(INF_STOP), facts("lobby"), Settings()) == "raid"
    assert goal(at(5 * 60), facts("lobby"), Settings()) == "raid"
    assert goal(at(5 * 60), facts("lobby"), Settings(raid=False)) == "infinity"


def test_infinity_loop():
    t = at(20 * 60)
    assert decide(t, facts("lobby")).name == "press_play"
    assert decide(t, facts("modes_menu")).name == "open_dimension"
    assert decide(t, facts("dimension_window")).name == "pick_infinite"
    assert decide(t, facts("party")).name == "start"
    assert decide(t, facts("fight")).name == "wait"
    assert decide(t, facts("fight", auto_off=True)).name == "press_auto"
    assert decide(t, facts("results_infinity")).name == "leave"


def test_leave_before_the_raid_and_wait_when_unsure():
    assert decide(at(INF_STOP + 5), facts("fight")).name == "reset"
    assert decide(at(INF_STOP - 100), facts("lobby")).name == "unhandled"   # raid soon: no new Infinity run
    assert decide(at(20 * 60), facts(None)).name == "wait"
    assert decide(at(20 * 60), facts("esc_menu")).name == "unhandled"
