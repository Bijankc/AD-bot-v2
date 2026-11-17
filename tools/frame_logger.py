"""Read-only frame logger: saves the Roblox client area while Roblox is the foreground window.

    pythonw tools/frame_logger.py              (hidden; runs until a STOP file, the time limit or the disk budget)
    python  tools/frame_logger.py              (same, in a console; Ctrl+C ends it)

Sends no input and never changes focus. Frames go to <ADBOT2_HOME>/captures/<yyyymmdd>/<HHMMSS_mmm>.jpg (and a
lossless .png every few minutes) with an index.csv (time, file, client rect, DPI, change score, kind).
JPEG keeps full colour resolution (4:4:4): the default 4:2:0 halves it and smears thin red digits.
Create <ADBOT2_HOME>/captures/STOP to end it cleanly. Status and errors go to captures/logger.log.
Does not open the old bot's log (an open handle once blocked its writes).
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

import cv2

from adbot2 import paths
from adbot2.capture import Grabber, foreground_roblox, lower_priority, make_dpi_aware, process_exe
from adbot2.recorder import SavePolicy, thumb

INDEX_HEADER = ["time", "file", "left", "top", "width", "height", "dpi", "change", "kind"]


def captures_root() -> Path:
    return paths.home() / "captures"


def other_logger_running(pid_file: Path) -> int:
    """PID of another live logger (a python process named in the PID file), else 0."""
    try:
        pid = int(pid_file.read_text().strip())
    except (OSError, ValueError):
        return 0
    if pid == os.getpid():
        return 0
    return pid if process_exe(pid).startswith("python") else 0


def append_index(day: Path, row: list) -> None:
    """Append to day/index.csv; an index in an older column layout is renamed aside first."""
    idx = day / "index.csv"
    if idx.exists():
        with open(idx, encoding="utf-8") as f:
            header = f.readline().strip().split(",")
        if header != INDEX_HEADER:
            n = 1
            while (day / f"index.v{n}.csv").exists():
                n += 1
            idx.rename(day / f"index.v{n}.csv")
    new = not idx.exists()
    with open(idx, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(INDEX_HEADER)
        w.writerow(row)


def encode(img, png: bool, quality: int) -> bytes | None:
    if png:
        ok, buf = cv2.imencode(".png", img, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    else:
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality,
                                             cv2.IMWRITE_JPEG_SAMPLING_FACTOR, cv2.IMWRITE_JPEG_SAMPLING_FACTOR_444])
    return buf.tobytes() if ok else None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="read-only Roblox frame logger")
    ap.add_argument("--interval", type=float, default=0.5, help="seconds between looks")
    ap.add_argument("--quality", type=int, default=92, help="JPEG quality")
    ap.add_argument("--max-gb", type=float, default=15.0, help="stop after writing this much in one run")
    ap.add_argument("--min-free-gb", type=float, default=15.0, help="stop when the disk has less free than this")
    ap.add_argument("--png-min", type=float, default=5.0, help="a lossless PNG every N minutes (0 = never)")
    ap.add_argument("--hours", type=float, default=0, help="stop after this many hours (0 = no limit)")
    args = ap.parse_args(argv)

    root = captures_root()
    root.mkdir(parents=True, exist_ok=True)

    def log(msg: str) -> None:
        line = f"{datetime.now():%d.%m %H:%M:%S} {msg}"
        if sys.stdout is not None:  # None under pythonw
            print(line, flush=True)
        with open(root / "logger.log", "a", encoding="utf-8") as f:
            f.write(line + "\n")

    pid_file = root / "logger.pid"
    other = other_logger_running(pid_file)
    if other:
        log(f"not started: another logger is running (pid {other})")
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


def run(args, root: Path, log) -> int:
    make_dpi_aware()
    lower_priority()
    stop_file = root / "STOP"
    stop_file.unlink(missing_ok=True)
    policy = SavePolicy(max_bytes=int(args.max_gb * (1 << 30)), min_free=int(args.min_free_gb * (1 << 30)),
                        png_every=args.png_min * 60)
    grabber = Grabber()
    started = time.monotonic()
    saved = looked = away = skipped = pngs = 0
    last_report = started

    log(f"started pid {os.getpid()} (interval {args.interval}s, JPEG q{args.quality} 4:4:4, PNG every "
        f"{args.png_min} min, budget {args.max_gb} GB, stop under {args.min_free_gb} GB free)")
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
            free = shutil.disk_usage(root).free
            over = policy.out_of_budget(free)
            if over:
                reason = over
                break
            if now - last_report >= 600:
                log(f"alive: {saved} saved ({pngs} png), {skipped} thinned, {policy.bytes_written >> 20} MB, "
                    f"{looked} looks, {away} not in front, {free >> 30} GB free")
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
            keep, d, kind = policy.should_save(now, th)
            png = policy.png_due(now)
            if keep or png:
                kind = kind if keep else "png"
                data = encode(img, png, args.quality)
                if data:
                    stamp = datetime.now()
                    day = root / f"{stamp:%Y%m%d}"
                    day.mkdir(exist_ok=True)
                    name = f"{stamp:%H%M%S_%f}"[:10] + (".png" if png else ".jpg")
                    (day / name).write_bytes(data)
                    append_index(day, [stamp.isoformat(timespec="milliseconds"), name, *rect, win.dpi,
                                       f"{d:.1f}", kind])
                    policy.saved(now, th, len(data), kind, png)
                    saved += 1
                    pngs += png
            elif kind == "dup":
                skipped += 1  # changed since the last save, but a near-repeat of a recent one
            time.sleep(max(0.0, args.interval - (time.monotonic() - now)))
    except KeyboardInterrupt:
        reason = "Ctrl+C"
    log(f"ended ({reason}): {saved} saved ({pngs} png), {skipped} thinned, {policy.bytes_written >> 20} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
