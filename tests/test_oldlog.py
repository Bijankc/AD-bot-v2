from datetime import datetime

from adbot2.oldlog import (parse_line, read_all_contexts, screen_hint, screenshot_contexts, shot_reason,
                           shot_time)

LOG = """\ufeff13.11 02:28:07.923  Titan: run over after 594 s (results)
13.11 02:28:10.569    screenshot: C:\\Users\\someone\\AD-bot\\shots\\20261113_022810_titan_end.png
13.11 02:28:10.967    Titan results: REWARDS Time: 10:00 Damage: 1296
not a log line
13.11 02:28:10.968  Titan: RETRY
"""


def test_parse_line():
    ln = parse_line("13.11 02:28:07.923  Titan: run over after 594 s (results)", 2026)
    assert ln.time == datetime(2026, 11, 13, 2, 28, 7, 923000)
    assert ln.text == "Titan: run over after 594 s (results)"
    assert parse_line("garbage", 2026) is None


def test_bom_and_crlf():
    assert parse_line("\ufeff13.11 02:28:07.923  hi\r\n", 2026).text == "hi"


def test_shot_name():
    assert shot_reason("20261114_095321_raid_end.png") == "raid_end"
    assert shot_reason("20261114_181754_count_82_0_30_16_12.png") == "count_82_0_30_16_12"
    assert shot_reason("device_check.png") is None
    assert shot_time("20261114_095321_raid_end.png") == datetime(2026, 11, 14, 9, 53, 21)


def test_contexts_skip_screenshot_lines():
    lines = [parse_line(x, 2026) for x in LOG.splitlines()]
    ctx = screenshot_contexts([x for x in lines if x])
    c = ctx["20261113_022810_titan_end.png"]
    assert c.context == ["Titan: run over after 594 s (results)"]
    assert c.time == datetime(2026, 11, 13, 2, 28, 10, 569000)


def test_context_window_is_bounded():
    lines = [parse_line(f"13.11 02:00:{i:02d}.000  msg {i}", 2026) for i in range(40)]
    lines.append(parse_line("13.11 02:01:00.000  screenshot: x\\a.png", 2026))
    assert screenshot_contexts(lines, keep=3)["a.png"].context == ["msg 37", "msg 38", "msg 39"]


def test_read_all_contexts_uses_year_from_file_name(tmp_path):
    (tmp_path / "logs").mkdir()
    (tmp_path / "logs" / "raid_20251113.log").write_text(LOG, encoding="utf-8")
    ctx = read_all_contexts(tmp_path)
    assert ctx["20261113_022810_titan_end.png"].time.year == 2025


def test_hints():
    assert screen_hint("raid_end") == "results_raid"
    assert screen_hint("xboss_leg5_20s") == "fight"
    assert screen_hint("restart") is None  # a "restart" shot can show any screen
    assert screen_hint(None) is None
