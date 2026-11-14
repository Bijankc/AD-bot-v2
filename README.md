# AD Bot v2

> **Based on the idea of [AD-bot](https://github.com/Samogub/AD-bot) by Samogub.**

A farming bot for the Roblox game *Anime Dimensions Simulator*, which allows macros. It only looks at the screen and
presses keys and the mouse like a player. It does not read or change game memory, inject code or use exploits.

**Status: early development.** Nothing here plays the game yet. The first milestone is reading the screen reliably,
tested offline on saved screenshots. See [PLAN.md](PLAN.md).

## How it works (planned)

- **Capture** the Roblox window (Windows Graphics Capture).
- **Perception:** a small screen classifier ("which screen is this"), an object detector for buttons, HP bars, enemies and
  the player, and small readers for numbers (enemy count, timers). These are trained on the author's own screenshots,
  instead of pixel templates that break on every redraw or scale change.
- **State machine:** explicit screens and goals (raid > Speed Raid > Infinity / Dimensions), unit-tested on saved
  screenshots and short clips.
- **Fighting:** Raid, Speed Raid and Infinity use the game's own Auto button, and the bot only keeps it on. Dimensions are
  played closed-loop: the bot steers toward what it sees and re-targets after knockback.
- **Fallback (optional, off by default):** an unknown screen can be described by a vision-language model.

## Setup (development)

Windows 10/11, Python 3.12+, NVIDIA GPU recommended.

```powershell
py -m venv C:\adbot2\venv
C:\adbot2\venv\Scripts\pip install -e .[dev]
C:\adbot2\venv\Scripts\python -m pytest
```

Large data (captures, datasets, models) lives outside the repo, in `ADBOT2_HOME` (default `C:\adbot2`).

## Privacy

Private server links, logs and screenshots (which show the Roblox username) never go into this repository. See
`.gitignore`. Put your own settings in `config.local.toml` (ignored).

## Credits / acknowledgements

- **Samogub**, the author of [AD-bot](https://github.com/Samogub/AD-bot): the original idea of farming Anime Dimensions
  with a screen-reading macro bot, and much of the game knowledge behind it.
- v2 is a from-scratch rewrite by [Bijankc](https://github.com/Bijankc). It grew out of running a fork of AD-bot
  ([Bijankc/AD-bot](https://github.com/Bijankc/AD-bot)). AD-bot has no licence file, so **no code, image templates or other
  material from it is included here** unless its author has given permission (any such material will be listed in this
  section).

## License

[Apache License 2.0](LICENSE) for the code in this repository.

Not affiliated with Roblox or the developers of Anime Dimensions Simulator.
