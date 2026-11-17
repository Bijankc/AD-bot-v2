"""Find the Roblox window and grab its client area. Read-only: no input, no focus changes.

Uses GDI screen capture (via mss), the same kind of capture the old AHK bot relies on, which kept working at night
with the laptop lid shut. The process is made per-monitor DPI aware so rectangles are physical pixels at any
Windows scale (100% or 150%).
"""

from __future__ import annotations

import ctypes
import ctypes.wintypes as wt
from dataclasses import dataclass

import numpy as np

ROBLOX_EXE = "robloxplayerbeta.exe"

user32 = ctypes.WinDLL("user32", use_last_error=True)
kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

# Handles are pointer-sized; without these, ctypes passes and returns 32-bit ints on 64-bit Windows.
user32.GetForegroundWindow.restype = wt.HWND
for _f in ("IsIconic", "IsWindowVisible", "IsHungAppWindow"):
    getattr(user32, _f).argtypes = [wt.HWND]
user32.GetWindowThreadProcessId.argtypes = [wt.HWND, ctypes.POINTER(wt.DWORD)]
user32.GetClientRect.argtypes = [wt.HWND, ctypes.POINTER(wt.RECT)]
user32.ClientToScreen.argtypes = [wt.HWND, ctypes.POINTER(wt.POINT)]
user32.GetDpiForWindow.argtypes = [wt.HWND]
user32.GetDpiForWindow.restype = wt.UINT
kernel32.GetCurrentProcess.restype = wt.HANDLE
kernel32.SetPriorityClass.argtypes = [wt.HANDLE, wt.DWORD]
kernel32.OpenProcess.argtypes = [wt.DWORD, wt.BOOL, wt.DWORD]
kernel32.OpenProcess.restype = wt.HANDLE
kernel32.QueryFullProcessImageNameW.argtypes = [wt.HANDLE, wt.DWORD, wt.LPWSTR, ctypes.POINTER(wt.DWORD)]
kernel32.CloseHandle.argtypes = [wt.HANDLE]

_dpi_set = False


def make_dpi_aware() -> None:
    """Per-monitor v2 DPI awareness (must happen before any window or capture call)."""
    global _dpi_set
    if _dpi_set:
        return
    try:
        user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4))  # PER_MONITOR_AWARE_V2
    except (AttributeError, OSError):
        pass
    _dpi_set = True


def lower_priority() -> None:
    """Below-normal priority so capture never competes with the game."""
    kernel32.SetPriorityClass(kernel32.GetCurrentProcess(), 0x4000)


def _process_name(hwnd: int) -> str:
    pid = wt.DWORD()
    user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
    return process_exe(pid.value)


def process_exe(pid: int) -> str:
    """Lower-case exe name of a running process, or "" if there is none (or it cannot be opened)."""
    h = kernel32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
    if not h:
        return ""
    try:
        buf = ctypes.create_unicode_buffer(1024)
        size = wt.DWORD(len(buf))
        if not kernel32.QueryFullProcessImageNameW(h, 0, buf, ctypes.byref(size)):
            return ""
        return buf.value.rsplit("\\", 1)[-1].lower()
    finally:
        kernel32.CloseHandle(h)


@dataclass(frozen=True)
class GameWindow:
    hwnd: int
    left: int     # client area, physical screen pixels
    top: int
    width: int
    height: int
    dpi: int


def foreground_roblox() -> GameWindow | None:
    """The Roblox window if it is the foreground window, visible, not minimised and not hung; else None.

    Requiring the foreground keeps capture away from anything else the user has open.
    """
    make_dpi_aware()
    hwnd = user32.GetForegroundWindow()
    if not hwnd or _process_name(hwnd) != ROBLOX_EXE:
        return None
    if user32.IsIconic(hwnd) or not user32.IsWindowVisible(hwnd) or user32.IsHungAppWindow(hwnd):
        return None
    rc = wt.RECT()
    if not user32.GetClientRect(hwnd, ctypes.byref(rc)):
        return None
    pt = wt.POINT(0, 0)
    user32.ClientToScreen(hwnd, ctypes.byref(pt))
    dpi = user32.GetDpiForWindow(hwnd) or 96
    w, h = rc.right - rc.left, rc.bottom - rc.top
    if w < 200 or h < 200:
        return None
    return GameWindow(hwnd, pt.x, pt.y, w, h, dpi)


class Grabber:
    """Grabs a screen rectangle as BGR. Parts outside the virtual screen are clipped (the 150% client starts at y=-6)."""

    def __init__(self):
        make_dpi_aware()
        import mss  # imported here so tests and tools that never capture don't need it

        self._sct = mss.mss()
        v = self._sct.monitors[0]
        self._vx, self._vy = v["left"], v["top"]
        self._vr, self._vb = v["left"] + v["width"], v["top"] + v["height"]

    def grab(self, win: GameWindow) -> tuple[np.ndarray, tuple[int, int, int, int]]:
        l, t = max(win.left, self._vx), max(win.top, self._vy)
        r, b = min(win.left + win.width, self._vr), min(win.top + win.height, self._vb)
        shot = self._sct.grab({"left": l, "top": t, "width": r - l, "height": b - t})
        img = np.asarray(shot)[:, :, :3]  # BGRA -> BGR
        return np.ascontiguousarray(img), (l, t, r - l, b - t)
