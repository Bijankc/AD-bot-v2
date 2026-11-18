# AD Bot v2: plan (draft, 14 Nov 2026)

Based on the idea of AD-bot by Samogub (https://github.com/Samogub/AD-bot). Written from scratch. No upstream code or
image templates are used.

## Decisions (14 Nov)

- Public repo, Apache-2.0. Therefore no Ultralytics (AGPL); the detector will be an Apache-licensed one (RF-DETR or D-FINE).
- Venv, data, captures and models live in `C:\adbot2\` (outside OneDrive).
- No game access this weekend (farming time). Work offline on the existing screenshots and logs only.
- v2 uses screen reading and normal input only. No exploits or injected scripts. A script the author brings may be read as
  reference text only. It is never run, and nothing from it goes into the repo.

**Scope (17 Nov):** Hourly Raid, Speed Raid, Infinity, Titan and other Dimensions where possible, Extreme Dimension
on Easy, and Boss Rush only when asked (rarely). Not wanted: Time Challenge, daily claims, codes, gacha and upgrades
(the user does those; the bot only recognises and closes them).
Research and the reading plan: docs/RESEARCH.md (fixed UI = crop heads, digits = own reader, detector only for moving
things).

## Starting point

**Hardware:** RTX 4060 Laptop GPU with 8 GB VRAM, driver 621.01 (supports CUDA 13.x). i7-14700HX (20 cores), 16 GB RAM.
The CUDA toolkit is not installed, and it is not needed because PyTorch and onnxruntime wheels ship their own CUDA runtime.
Python 3.14, 3.10 and 3.7 are installed.
- 8 GB is enough to train small detectors (YOLO-n/s class at 640-960 px) and to run them in a few ms with TensorRT or ONNX-CUDA.
- Disk (14 Nov): only 22.9 GB free on C:. **17 Nov: no longer the main limit** (~45 GB free, more to come); jobs keep a
  15 GB floor so Windows and Roblox never run out.
- **The repo sits in OneDrive.** The venv, the datasets and the raw captures must not sync, so they go outside OneDrive in
  `C:\adbot2\` (venv, data, captures, models). Only code, configs and tiny test fixtures stay in the repo.

**Data already available:**
| Source | What | Use |
|---|---|---|
| `shots/` | 82 full 1920x1080 frames + 18 crops (1320x520), named by reason (titan_fail, xboss_leg*, raid_end, gf_banner, restart...). Rotated to 100 files. | First screen-classifier and detector labels, plus test fixtures |
| `raid.log` + `logs/` | 9 days, about 2.9 MB, timestamped state lines | Weak labels: map a frame's time to the state the old bot believed it was in |
| `dimension/*/log.txt` | death (969 rows, 95 `cnt`), extreme_titan (426, 37 `cnt`), giant/giant3 (no `cnt`). No recorder screenshots saved. | Movement and enemy-count timelines for the Dimension work and for behaviour cloning later |
| `img/` (50 templates) | Partly upstream's | Reference only. Not copied into v2 |

82 frames is far too few. The main data source has to be **new passive captures** (see M1).

## Design (refined)

```
capture (WGC window capture, game-area crop, DPI-aware)
   -> perception: screen classifier  -> which screen (lobby, modes menu, raid fight, results, GF queue, ...)
                  element detector    -> boxes for AUTO on/off, PLAY, RETRY/LEAVE, Create/Start, boss bar, enemy count, enemies, boss, self
                  readers             -> digits (enemy count, timers) with a small CNN; text with Windows OCR (no install)
   -> observation (typed dataclass, confidence per field)
   -> state machine (explicit screens + debounce: N agreeing frames) -> goal scheduler (raid > speed raid > infinity/dimension)
   -> actions (input layer: the only module that may send keys or clicks; disabled unless --live)
   -> optional VLM fallback for unknown screens
```

Changes to the original idea, and why:
1. **Screen classifier plus detector, not detector alone.** "Which screen is this" is one label per image, so it is cheap to
   label and copes with redraws. The detector is only needed where the bot must click or steer.
2. **Digits get their own tiny reader.** Enemy counts and raid timers are where OCR failed before ("41:28" read as "4:28",
   "22:02" read as "2:02"). A digit CNN trained on crops, cross-checked over frames, is more reliable than general OCR.
3. **Capture with Windows Graphics Capture** (e.g. the `windows-capture` package), aimed at the Roblox window. It does not
   need Roblox in front and handles the 100%/150% DPI layouts. DXcam or bettercam is the fallback. I'll check current
   maintenance before choosing.
4. **Detector licence matters for the repo licence.** Ultralytics YOLO (code and weights) is AGPL-3.0. Apache-2.0 options
   with similar quality: RF-DETR, D-FINE, YOLOX. Inference always runs through onnxruntime (MIT) either way.
5. **Shadow mode before any takeover.** v2 runs next to the AHK bot, reads the screen, and logs what it would do, without
   sending input. The two logs are then compared. This is how the "step by step" replacement happens.
6. **VLM fallback** uses the Claude API (`claude-opus-5-5`; a cheaper model only if you pick one). It sends one cropped
   screenshot and gets back JSON `{screen, suggested_action, confidence}`. It is off by default. It needs your API key and
   your OK, because screenshots leave the PC, and the username area is blurred first. The bot only takes its suggestion
   if the action is in an allow-list (Esc, LEAVE, close...).

## Milestones

**M0 Setup (small).** Folder, README with credits, .gitignore (raid.ini, logs, shots, captures, data, `*.ini`), licence
chosen with you, GitHub repo created after your answer on visibility, venv in `C:\adbot2\venv` (asked before installing).

**M1 Reliable screen reading, tested offline.**
- `capture/`: one interface for live frames and for files or clips, so every test runs on saved data.
- **Passive frame logger** (running since 14 Nov): saves the Roblox window about every 2 s and whenever the screen changes,
  while the AHK bot farms. It is read-only: no input, no focus change. Frames go to `C:\adbot2\captures\`, de-duplicated.
  17 Nov: 15 GB per run, JPEG q92 4:4:4, near-duplicate thinning against the last 20 saves, a PNG every 5 min.
  The old bot's `raid.log` is then joined by timestamp to pre-label them.
- Labelling workflow: (a) a local sorting page (one key per screen class) for classifier labels; (b) boxes in an
  off-the-shelf tool such as X-AnyLabeling or Label Studio (to pick with you), pre-filled by the current model so you only fix
  mistakes.
- Train classifier, detector and digit reader; export to ONNX, and TensorRT if it is worth it.
- `pytest` on a held-out set from both scales. Targets: 99%+ screen accuracy; **AUTO ON never read as OFF** (a wrong click
  turns Auto off); enemy count exact on 99%+ of frames; no false "lobby" or LEAVE on white flashes or bright floors (the
  failures in CLAUDE.md become named regression tests).

**M2 State machine and shadow mode.** Rewrite the decision rules in CLAUDE.md (raid clock and `raidOfs`, Group Finder
queue, one raid an hour, restart rules, hang checks) as plain, tested Python. Clip-replay tests: recorded frame sequences
→ expected transitions. Then shadow mode live (read-only) next to the AHK bot.

**M3 Take over the Auto modes.** Raid, Speed Raid and Infinity, with v2 sending input (only when you say the bot can run;
the AHK bot is stopped for that session). The game's Auto does the fighting.

**M4 Closed-loop Dimensions.** Steer with Shift Lock toward the nearest detected enemy or the boss (mouse yaw to centre it,
W until it is in range, skills), re-target every frame, detect being stuck (W held, image not moving) and strafe or back
off. The enemy count and boss HP check progress. Recorded runs are used for evaluation.

**M5 VLM fallback** for unknown screens (see design point 6).

**M6 (research) Behaviour cloning** from your recorded play (e.g. Pixels2Play-style). Only after M4 gives a baseline to beat.

## Rules kept throughout
No input to Roblox and no Roblox process actions unless you say so; tests never open windows or dialogs; the server link
and anything with your username never go into the repo; small commits; ask before every push.

## Progress

- **14-15 Nov:** M0 done. Frame logger ran overnight: 8,441 frames, 0 capture failures, stopped at its 3 GB budget.
- **17 Nov, step 1 (data flowing):** logger limits raised and thinning added (see M1); restarted hidden at below-normal
  priority. Old-bot shot import re-run (store 198 frames).
- **17 Nov, step 2 (weak labels and batches):** log lines turned into screen intervals (`weaklabels.py`); 10,731
  captures registered with hints (89% hinted: mostly fights, as expected; rare screens are rare). Embeddings for all
  10,911 frames in 59 s on the GPU. Batch picker and batch mode on the labelling page. First batch (120) waiting for labels.
- **17 Nov, step 3 (game area):** `geometry.py` maps 100%, 150% and old-bot screen grabs to one canonical frame and blanks
  everything outside the game band. Bottom UI aligns within 1 px; top-anchored UI (HP panel) is 29 px higher and 0.95x at
  150%, so readers must not use one fixed box. 8 mid-resize frames excluded. Embeddings redone on the canonical view.
- **17 Nov, step 4 (evaluation):** frozen split by farm day and layout with a 3-min guard band; per-class accuracy,
  confusion, worst examples to `C:\adbot2\reports\`; 13 named regression cases from the old bot's failures, 91
  candidates proposed (none yet for only-LEAVE results and the raid clock; 1 for Revive). Numbers wait for labels.
- **17 Nov, labels:** 441 labelled in four rounds. They added classes `dead` (no respawn/REVIVE offer) and
  `extreme_window`, and corrected ten weak-label rules (timings around resets, queues, Speed Raid start, results).
- **17 Nov, step 5a (screen classifier, first model):** MobileNet-V3-large (ImageNet, BSD) at 448x240 on the canonical
  view; 176 labelled + 642 hinted train frames; 1 min of training, 1.6 GB VRAM. Held-out results (test = 17 Nov morning
  and old-bot shots of 14 Nov; val chose the epoch, so it is slightly optimistic):

  | | old bot's log | kNN | model |
  |---|---|---|---|
  | test, 100% captures | 63% | 85% | **97%** (69) |
  | test, old-bot shots | 87% | 35% | **92%** (24) |
  | val, 100% captures | 79% | 78% | 94% (128) |
  | regression failures | 50 | 21 | **3** (2 GF banner, 1 Revive) |

  Speed on the 4060: 7 ms to shrink a 1080p frame + 12 ms on the GPU (17 ms CPU) via ONNX Runtime. Weak spots: Speed
  Raid window (no train examples yet), respawn, raid_world shots. No 150% frames are labelled on a held-out day yet.
- **17 Nov, step 5b (crop heads, started):** fixed UI is read from fixed boxes by tiny CNNs (`crops.py`; 0.1 M
  weights, ~0.4 ms per crop on the CPU): AUTO (on / off / gone), ENEMIES LEFT diamond, Group Finder banner. All
  19,841 frames cut once into a cache (2 min). Labelling is a grid of 60 pre-filled crops (`/crops` on the labelling
  page): fix the wrong tiles, Enter. First models are trained on rule guesses only; they already find the OFF
  diamond. Next: crop labels, then the enemy-count digit reader.
- **18 Nov, enemy count reader:** the count is red digits on an opaque diamond in a fixed font (every digit 17 px
  tall), so it is read at native pixels with no OCR or network (`digits.py`): red pixels -> glyphs (touching digits
  split) -> nearest mean shape, from 24 glyph groups named once. Reads 17,215 of 17,221 crops (the 6 unread are dimmed
  by a menu: refused, not guessed); 160 checked by eye, 0 wrong. The count is not monotonic: it flickers (1->0->1
  between waves, 13->18->13), so the bot must never treat one 0 as cleared.
  AUTO head with the first 21 train labels: val 94%, test 55% (rules 5%); more labels needed.
- **18 Nov, raid clock (modes menu RAID card):** a text-line reader (`textline.py`): text pixels on the dark bar ->
  glyphs -> nearest example. Digits are labelled by the clock itself: this server's raid opened at :45:00 every hour
  14-17 Nov, so each frame's countdown is known from its time (the card's text can lag ~5 s, so only digits certain
  under that lag are used). Single readings on held-out days: 55 right, 6 wrong; with the old bot's rule (accept a
  raid time only when two readings agree) 17 accepted, 0 wrong. More menu frames will sharpen single readings.
  The Group Finder banner clock (smaller font) is not done yet; the menu clock already gives the opening second.
