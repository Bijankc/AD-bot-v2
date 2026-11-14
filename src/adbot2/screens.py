"""Screen classes: the answer to "which screen is this?" (one label per frame).

Things that can appear on several screens (the Group Finder queue banner, the AUTO button, the boss HP bar, the enemy
count) are *not* screens: they are detector classes, read separately.

The list is a first version from the old bot's notes and screenshots. Adding a class later is cheap; renaming one means
relabelling, so names describe what is on screen, not what the bot does there.
"""

from __future__ import annotations

from enum import Enum


class Screen(str, Enum):
    LOADING = "loading"                      # Roblox / teleport loading, black or splash screen
    LOBBY = "lobby"                          # hub world with the PLAY button
    MODES_MENU = "modes_menu"                # Dimension / Speed Raid / RAID cards
    RAID_WINDOW = "raid_window"              # raid Join / Create / Group Finder window
    DIMENSION_WINDOW = "dimension_window"    # Dimension Join / Create window
    SPEED_RAID_WINDOW = "speed_raid_window"  # Speed Raid character picker
    PARTY = "party"                          # party room before Start
    RAID_WORLD = "raid_world"                # raid world, waiting for the raid to open
    FIGHT_AUTO = "fight_auto"                # Raid / Speed Raid / Infinity fight (has the AUTO button)
    FIGHT_DIMENSION = "fight_dimension"      # Dimension fight (no AUTO; red enemy count)
    REVIVE = "revive"                        # death screen with REVIVE (never press it: costs gems)
    RESULTS_RAID = "results_raid"            # RAID RESULTS (ranking + rewards)
    RESULTS_INFINITY = "results_infinity"    # Infinity results, RETRY / LEAVE / AUTO RETRY
    RESULTS_DIMENSION = "results_dimension"  # Dimension rank + REWARDS, RETRY / LEAVE
    BAG = "bag"                              # inventory / loadouts
    ESC_MENU = "esc_menu"                    # Roblox Esc menu
    DISCONNECTED = "disconnected"            # Roblox error / connection lost dialog
    OTHER = "other"                          # a known-but-unneeded screen (shop, quests, update notes...)
    UNSURE = "unsure"                        # labeller could not tell; excluded from training


# Keyboard shortcuts for the labelling page, in display order.
HOTKEYS: dict[str, Screen] = {
    "1": Screen.LOBBY,
    "2": Screen.MODES_MENU,
    "3": Screen.RAID_WINDOW,
    "4": Screen.DIMENSION_WINDOW,
    "5": Screen.SPEED_RAID_WINDOW,
    "6": Screen.PARTY,
    "7": Screen.RAID_WORLD,
    "8": Screen.FIGHT_AUTO,
    "9": Screen.FIGHT_DIMENSION,
    "0": Screen.LOADING,
    "q": Screen.REVIVE,
    "w": Screen.RESULTS_RAID,
    "e": Screen.RESULTS_INFINITY,
    "r": Screen.RESULTS_DIMENSION,
    "t": Screen.BAG,
    "y": Screen.ESC_MENU,
    "u": Screen.DISCONNECTED,
    "o": Screen.OTHER,
    "p": Screen.UNSURE,
}

assert set(HOTKEYS.values()) == set(Screen), "every screen needs a hotkey"
