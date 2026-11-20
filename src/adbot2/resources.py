"""Guards for batch jobs that run next to the game: never take the RAM Roblox needs.

18 Nov: a replay that decoded every frame up front used up the RAM and Roblox crashed mid-farm. Every tool that
decodes frames checks free RAM before it starts and between batches, and stops cleanly (keeping what it has done)
under MIN_FREE_GB.
"""

from __future__ import annotations

import ctypes
import sys

MIN_FREE_GB = 4.0


class _MemoryStatus(ctypes.Structure):
    _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]


def free_ram_gb() -> float:
    """Available physical memory in GB (Windows; elsewhere a large number, so tests never stop)."""
    if sys.platform != "win32":
        return 1e9
    m = _MemoryStatus()
    m.dwLength = ctypes.sizeof(_MemoryStatus)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
    return m.ullAvailPhys / 1e9


def ram_ok(min_gb: float = MIN_FREE_GB) -> bool:
    return free_ram_gb() >= min_gb


def ram_message(min_gb: float = MIN_FREE_GB) -> str:
    return f"only {free_ram_gb():.1f} GB RAM free (needs {min_gb:.0f}): stopping so Roblox keeps its memory"
