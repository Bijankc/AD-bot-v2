"""Live shadow mode: read the Roblox window, decide what v2 would do, write it down. Sends no input, ever.

    pythonw tools/shadow_live.py [--bot-dir ..\\AD-bot] [--opens 2700] [--interval 0.33] [--hours 0]

Like the frame logger it only looks while Roblox is the foreground window, at below-normal priority, and never
changes focus. Each frame: perception (CPU, 4 threads), tracker, policy (adbot2.shadowing.LiveShadow). The run memory
(entered mode, loadout, Speed Raid schedule, settings) follows the old bot's raid.log, read with a quick open/close
every few seconds. v2's decision goes to <ADBOT2_HOME>/reports/shadow_live/<yyyymmdd>.jsonl whenever it changes
(and every 30 s): time, screen, facts, goal, action, reason, timings. Compare with the old bot's log afterwards.

Stops on reports/shadow_live/STOP, the time limit, or under the RAM floor (adbot2.resources). Status, an "alive"
line every 10 min and crashes (with a traceback) go to reports/shadow_live/shadow.log. A PID lock stops a second copy.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

from adbot2 import paths
from adbot2.capture import Grabber, foreground_roblox, lower_priority, make_dpi_aware, process_exe
from adbot2.geometry import UnknownLayout, to_canvas
from adbot2.perception import Perception
from adbot2.policy import goal, raid_time
from adbot2.resources import ram_message, ram_ok
from adbot2.shadowing import LiveShadow, LogFollower

LOG_EVERY_S = 3.0     # read new old-bot log lines this often
RAM_EVERY_S = 30.0


def out_root() -> Path:
    return paths.home() / "reports" / "shadow_live"


def other_running(pid_file: Path) -> int:
    try:
        pid = int(pid_file.read_text().strip())
    except (OSError, ValueError):
        return 0
    if pid == os.getpid():
        return 0
    return pid if process_exe(pid).startswith("python") else 0


def row(t: float, o, f, sh: LiveShadow, a) -> dict:
    return {"time": datetime.fromtimestamp(t).isoformat(timespec="milliseconds"),
            "screen": f.screen, "raw": o.screen, "p": round(o.screen_p, 3),
            "auto_off": f.auto_off, "p_off": round(o.auto_p_off, 3), "count": f.enemy_count, "cleared": f.cleared,
            "gf_queued": f.gf_queued, "start": f.start_shown, "opens": f.raid_opens_at,
            "raid_time": raid_time(t, f.raid_opens_at), "mode": sh.pol.mode,
            "goal": goal(t, f, sh.pol.s, sh.pol.m), "action": a.name, "why": a.why,
            "ms": {k: round(v, 1) for k, v in o.ms.items()}}


def run(args, root: Path, log) -> int:
    make_dpi_aware()
    lower_priority()
    stop_file = root / "STOP"
    stop_file.unlink(missing_ok=True)
    per = Perception(paths.home() / "models", gpu=False)
    if per.screen is None:
        log("no screen model under models/screens: nothing to run")
        return 1
    sh = LiveShadow(per.screen.classes, args.opens, LogFollower(args.bot_dir / "raid.log"))
    grabber = Grabber()
    started = last_report = last_log = last_ram = time.monotonic()
    looks = away = rows = bad_layout = 0
    log(f"started pid {os.getpid()} (interval {args.interval}s, raid opens at second {args.opens} of the hour "
        f"until the clock is read; no input is ever sent)")
    reason = "stopped"
    try:
        while True:
            now = time.monotonic()
            if stop_file.exists():
                reason = "STOP file"
                break
            if args.hours and now - started >= args.hours * 3600:
                reason = "time limit"
                break
            if now - last_ram >= RAM_EVERY_S:
                last_ram = now
                if not ram_ok():
                    reason = ram_message()
                    break
            if now - last_log >= LOG_EVERY_S:
                last_log = now
                sh.read_log()
            if now - last_report >= 600:
                log(f"alive: {looks} looks, {rows} rows, {away} not in front, {bad_layout} unknown layout")
                last_report = now
            win = foreground_roblox()
            if win is None:
                away += 1
                time.sleep(1.0)
                continue
            try:
                img, _ = grabber.grab(win)
                area = to_canvas(img, win.dpi)
            except UnknownLayout:
                bad_layout += 1
                time.sleep(1.0)
                continue
            except Exception as e:  # display off, desktop locked, window moving...
                log(f"grab failed: {e!r}")
                time.sleep(5)
                continue
            looks += 1
            t = time.time()
            o = per.observe(area)
            f, a, write = sh.step(o, t)
            if write:
                rows += 1
                with open(root / f"{datetime.now():%Y%m%d}.jsonl", "a", encoding="utf-8") as fh:
                    fh.write(json.dumps(row(t, o, f, sh, a)) + "\n")
            time.sleep(max(0.0, args.interval - (time.monotonic() - now)))
    except KeyboardInterrupt:
        reason = "Ctrl+C"
    log(f"ended ({reason}): {looks} looks, {rows} rows")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--bot-dir", type=Path, default=Path("..") / "AD-bot")
    ap.add_argument("--opens", type=int, default=2700, help="second of the local hour the raid opens, until read")
    ap.add_argument("--interval", type=float, default=0.33, help="seconds between looks")
    ap.add_argument("--hours", type=float, default=0, help="stop after this many hours (0 = no limit)")
    args = ap.parse_args(argv)
    if sys.stdout is None:  # pythonw: libraries that print need somewhere to write
        sys.stdout = sys.stderr = open(os.devnull, "w")
    root = out_root()
    root.mkdir(parents=True, exist_ok=True)

    def log(msg: str) -> None:
        with open(root / "shadow.log", "a", encoding="utf-8") as fh:
            fh.write(f"{datetime.now():%d.%m %H:%M:%S} {msg}\n")

    if not ram_ok():
        log("not started: " + ram_message())
        return 2
    pid_file = root / "shadow.pid"
    if other := other_running(pid_file):
        log(f"not started: another shadow is running (pid {other})")
        return 1
    pid_file.write_text(str(os.getpid()))
    try:
        return run(args, root, log)
    except Exception:
        log("crashed:\n" + traceback.format_exc())
        return 2
    finally:
        try:
            if pid_file.read_text().strip() == str(os.getpid()):
                pid_file.unlink()
        except OSError:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
