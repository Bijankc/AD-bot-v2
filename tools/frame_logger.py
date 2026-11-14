"""Read-only frame logger: saves the Roblox client area while Roblox is the foreground window.

    python tools/frame_logger.py               (runs until Ctrl+C, a STOP file, or the disk budget)

Sends no input and never changes focus. Frames go to <ADBOT2_HOME>/captures/<yyyymmdd>/<HHMMSS_mmm>.jpg with an
index.csv (time, file, client rect, DPI, change score). Create <ADBOT2_HOME>/captures/STOP to end it cleanly.
Does not open the old bot's log (an open handle once blocked its writes).
"""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2

from adbot2 import paths
from adbot2.capture import Grabber, foreground_roblox, lower_priority, make_dpi_aware
from adbot2.recorder import SavePolicy, thumb


def captures_root() -> Path:
    return paths.home() / "captures"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="read-only Roblox frame logger")
    ap.add_argument("--interval", type=float, default=0.5, help="seconds between looks")
    ap.add_argument("--quality", type=int, default=90, help="JPEG quality")
    ap.add_argument("--max-gb", type=float, default=3.0)
    ap.add_argument("--min-free-gb", type=float, default=6.0)
    ap.add_argument("--hours", type=float, default=0, help="stop after this many hours (0 = no limit)")
    args = ap.parse_args(argv)

    make_dpi_aware()
    lower_priority()
    root = captures_root()
    root.mkdir(parents=True, exist_ok=True)
    stop_file = root / "STOP"
    stop_file.unlink(missing_ok=True)
    policy = SavePolicy(max_bytes=int(args.max_gb * (1 << 30)), min_free=int(args.min_free_gb * (1 << 30)))
    grabber = Grabber()
    started = time.monotonic()
    saved = looked = away = 0
    last_report = started

    def log(msg: str) -> None:
        line = f"{datetime.now():%d.%m %H:%M:%S} {msg}"
        print(line, flush=True)
        with open(root / "logger.log", "a", encoding="utf-8") as f:
            f.write(line + "\n")

    log(f"started (interval {args.interval}s, budget {args.max_gb} GB)")
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
            over = policy.out_of_budget(shutil.disk_usage(root).free)
            if over:
                reason = over
                break
            if now - last_report >= 600:
                log(f"alive: {saved} saved, {policy.bytes_written >> 20} MB, {looked} looks, {away} not in front")
                last_report = now

            win = foreground_roblox()
            if win is None:
                away += 1
                time.sleep(max(args.interval, 1.0))
                continue
            looked += 1
            try:
                img, rect = grabber.grab(win)
            except Exception as e:  # display off, desktop locked, window moving...
                log(f"grab failed: {e!r}")
                time.sleep(5)
                continue
            th = thumb(img)
            keep, d = policy.should_save(now, th)
            if keep:
                ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, args.quality])
                if ok:
                    stamp = datetime.now()
                    day = root / f"{stamp:%Y%m%d}"
                    day.mkdir(exist_ok=True)
                    name = f"{stamp:%H%M%S_%f}"[:10] + ".jpg"
                    (day / name).write_bytes(buf.tobytes())
                    new_index = not (day / "index.csv").exists()
                    with open(day / "index.csv", "a", newline="", encoding="utf-8") as f:
                        w = csv.writer(f)
                        if new_index:
                            w.writerow(["time", "file", "left", "top", "width", "height", "dpi", "change"])
                        w.writerow([stamp.isoformat(timespec="milliseconds"), name, *rect, win.dpi, f"{d:.1f}"])
                    policy.saved(now, th, len(buf))
                    saved += 1
            time.sleep(max(0.0, args.interval - (time.monotonic() - now)))
    except KeyboardInterrupt:
        reason = "Ctrl+C"
    log(f"ended ({reason}): {saved} saved, {policy.bytes_written >> 20} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
