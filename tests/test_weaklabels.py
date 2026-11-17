from datetime import datetime, timedelta

from adbot2.oldlog import LogLine
from adbot2.weaklabels import HintIndex, context_at, intervals, is_state_line, match_rule

T0 = datetime(2026, 11, 15, 0, 45, 0)


def log(*rows):
    return [LogLine(T0 + timedelta(seconds=s), text) for s, text in rows]


def hint(lines, s):
    return HintIndex(intervals(lines)).at(T0 + timedelta(seconds=s))[0]


def test_info_lines_are_not_state_lines():
    assert not is_state_line("ui: frame 31 ms (page home)")
    assert not is_state_line("Infinity: Auto: the button pictures do not match - using its colours (level 137)")
    assert not is_state_line("goal: infinity (Speed Raid is off)")
    assert is_state_line("something the rules have never seen")  # unknown = ends intervals
    assert match_rule("in a run (AUTO button on screen) - waiting").screen is None


def test_hold_until_the_next_state_line_with_settle_and_lead():
    lines = log((0, "raid: Auto, waiting for the end"), (5, "goal: infinity (Speed Raid is off)"),
                (100, "results: reading rewards"))
    assert hint(lines, 0.5) is None           # settling
    assert hint(lines, 5) == "fight"     # an info line does not end it
    assert hint(lines, 97.5) == "fight"
    assert hint(lines, 98.5) is None          # lead: the screen changes before the bot logs it
    assert hint(lines, 100.5) == "results_raid"


def test_hold_is_capped():
    lines = log((0, "raid: Auto, waiting for the end"))
    assert hint(lines, 597) == "fight"
    assert hint(lines, 601) is None


def test_same_screen_lines_join_without_gaps():
    lines = log((0, "Titan: route leg 1/17"), (3, "Titan: clearing after leg 1"), (4, "Titan: leg 1 done, 8 left"))
    for s in (2.0, 2.9, 3.0, 3.5):
        assert hint(lines, s) == "fight"


def test_break_lines_end_and_give_nothing():
    lines = log((0, "raid: Auto, waiting for the end"), (50, "death: waiting for respawn (Auto stays on)"),
                (51, "raid: Auto, waiting for the end"))
    assert hint(lines, 47) == "fight"
    assert hint(lines, 50.5) is None
    assert hint(lines, 53) == "fight"


def test_at_rule_covers_just_before_the_action():
    lines = log((0, "lobby loaded in 9.8 s"), (20, "pressing PLAY"))
    assert hint(lines, 19) == "lobby"
    assert hint(lines, 10) is None
    assert hint(lines, 20.5) is None


def test_raid_auto_logged_again_on_the_results_is_not_a_fight():
    lines = log((0, "results: reading rewards"), (3, "raid: Auto, waiting for the end"),
                (25, "lobby (PLAY visible) - not a fight"))
    assert hint(lines, 10) is None


def test_conflicting_overlap_gives_no_hint():
    lines = log((0, "pressing PLAY"), (0.5, "Infinity results: pressing LEAVE"))
    assert hint(lines, -0.05) is None  # PLAY covers -1.5..0, LEAVE -0.1..0.5


def test_context_marks_lines_after_the_frame():
    lines = log((0, "pressing PLAY"), (1, "ui: frame 3 ms"), (10, "entering Dimension"))
    ctx = context_at(lines, T0 + timedelta(seconds=5))
    assert ctx[0].endswith(" pressing PLAY") and ctx[-1].startswith("00:45:10 >")


def test_rules_fixed_by_the_first_labels():
    # "Titan results:" comes after RETRY: no hint; "Create tab" is in both windows: no hint
    assert match_rule("Titan results: REWARDS Time: 4:45").screen is None
    assert match_rule("Create tab").screen is None
    lines = log((0, "Group Finder: queued, waiting for the raid"), (100, "waiting (screen not recognised), goal: raid"))
    assert hint(lines, 30) == "lobby" and hint(lines, 50) is None   # the raid starts ~a minute before the log says
    lines = log((0, "leaving Infinity (raid): Esc-R-Enter"), (22, "Infinity results: pressing LEAVE"))
    assert hint(lines, 10) == "respawn" and hint(lines, 21.7) == "results_infinity"
