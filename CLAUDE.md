# AD Bot v2: working notes

Python rewrite of the AutoHotkey farming bot (sibling folder `..\AD-bot`, whose CLAUDE.md is the behaviour spec and the
source of regression tests). Public repo, Apache-2.0. Plan and milestones: PLAN.md. Keep this file short and public-safe:
no user names, no server links, no paths under the user profile.

## Hard rules

- The old bot farms while v2 is built. v2 sends **no input** to Roblox and never focuses, moves, restarts or closes it or the
  old bot, until the user says so for a given night (old bot stopped). Never edit `..\AD-bot`.
- Read the old bot's `raid.log` with a quick open/close only (a `tail -f` once blocked its writes for hours).
- Nothing pops up: detached jobs use `pythonw`, scripts open no windows. Big jobs: below-normal priority, leave 2 GB VRAM
  for Roblox, watch the old bot's log while they run, ask before anything over ~1 h.
- Bound memory: decode frames in small batches, never all at once, and stop under 4 GB free RAM (18 Nov: a replay
  that decoded every frame up front used up the RAM and Roblox crashed mid-farm).
- Privacy: never commit `*.ini`, logs, screenshots, captures, reports or weights trained on the user's frames. Stage
  explicit paths only (`git add <file>`, never `-A`) and read `git status` before every commit. Ask before every push.
- No cheats, no executors. No code or image templates from upstream AD-bot (no licence). Keep the credit line in README.

## Where things are

- Data root `ADBOT2_HOME` (default `C:\adbot2`, outside OneDrive, never in git): `venv\`, `data\` (frame store,
  `manifest.jsonl`, `labels\`), `captures\<yyyymmdd>\` (logger frames + `index.csv`), `reports\`.
- Tests: `C:\adbot2\venv\Scripts\python.exe -m pytest -q` (synthetic data only, fast).

## Frame logger (`tools/frame_logger.py`)

- Start hidden: `Start-Process C:\adbot2\venv\Scripts\pythonw.exe -ArgumentList '"tools\frame_logger.py"'
  -WorkingDirectory <repo> -WindowStyle Hidden`. The venv's pythonw is a launcher; the worker is a second pythonw at
  BelowNormal. A `captures\logger.pid` lock stops a second copy. Stop: create `captures\STOP`.
- Check `captures\logger.log` each session ("alive" line every 10 min; crashes are logged there with a traceback).
- Saves only while Roblox is the foreground window. JPEG q92 4:4:4 (~450 KB), a PNG every 5 min, 15 GB per run, stops
  under 15 GB free. Index column `kind`: first | change | periodic | png.
- Lesson (17 Nov): the game is rarely static (median frame-to-frame change 61/255 on 8.4k frames), so thinning compares
  against the last 20 saves, not just the last one; static screens back off 20 s -> 160 s.
- Each session also run `tools/import_old_shots.py --bot-dir ..\AD-bot` (the old bot keeps only its newest 100 shots).

## Screen geometry (`adbot2/geometry.py`)

- Canonical frame = the old bot's 1920x1080 screen coordinates. 100%: client 1920x1017 at (0,23); 150% (lid shut at
  night): 1920x1046 at (0,-6), saved by the logger as 1920x1040 (top 6 rows off screen). Both end at row 1040.
- Only rows 23..1039 are game in every layout; `to_canvas` zeroes the rest (title bar, window behind, taskbar) and
  `model_view` is that band (1920x1017). Unknown client sizes (mid-resize) raise `UnknownLayout` and are not used.
- Measured 17 Nov: bottom/centre-anchored UI (hotbar, Update Notes, gear, banners) lines up within 1 px across layouts;
  the top-left HP/XP panel is pinned to the client top: 29 px higher and 0.95x at 150%. The old bot's AUTO templates
  also differ per scale. So never read with one fixed box for both layouts: locate with the detector, or per-layout boxes.
- Tests marked `data` use the real frames and skip when `ADBOT2_HOME` has none.

## Weak labels and labelling (step 2)

- `adbot2/weaklabels.py`: the old bot logs a line when its state *changes*, so a state line holds until the next one
  (hold rules: settle after, lead before a screen change, max length; at rules: just before a click; break rules: end
  intervals, no hint). Unknown lines end intervals; known chatter is in `INFO`. Conflicts = no hint.
- Lessons from the log: an Infinity run is ~30 silent minutes; Infinite Mode is picked in the Dimension window;
  "raid: Auto, waiting for the end" is logged again on the results screen after "results: reading rewards";
  "in a run (AUTO button on screen)" was logged on a Create window (false AUTO), so it gives no hint.
- Session routine: `tools/import_old_shots.py --bot-dir ..\AD-bot`, `tools/import_captures.py --bot-dir ..\AD-bot`
  (frames older than 15 min, not copied; manifest points at captures), `tools/embed_frames.py` (MobileNet-V3-small,
  ~1 min for 11k frames on the GPU), `tools/pick_batch.py` (120 frames: kNN-uncertain + diverse/balanced, >=10 s apart),
  then the user labels at http://127.0.0.1:8765 (`tools/label_screens.py`, start hidden with pythonw).
  `tools/pick_batch.py --report` prints class counts and per-rule hint agreement.

## Evaluation (step 4, `adbot2/evaluation.py`, `tools/evaluate.py`)

- Split by session = one farm day (noon to noon) of one layout; per layout sessions take turns train, test, val, train,
  train. Frozen in `data\splits.json` (never edit by hand to move a session). Val/test frames within 3 min of a train
  frame of the same layout are "guard" and not scored. First session plan: 100% 14 Nov train, 17 Nov morning test;
  150% 14-15 Nov night train, the next 150% night test.
- Regression set: `tools/regression_candidates.py` proposes frames per named case (old-bot shot reasons, log moments,
  very bright fight frames) as a batch; a candidate counts once the user labels it. No log marker exists for Revive,
  only-LEAVE results or (since GF) the raid clock: those come from labelling / kNN later.
  "Auto: off" lines exist only in older logs; Auto-off frames are the ones just before "Auto pressed (try N)".
- `tools/evaluate.py --predictor hint|knn` writes `reports\<time>_<pred>\` (report.md, confusion CSVs, worst/*.jpg,
  confidently wrong first). `hint` = the old bot's own belief, a baseline worth knowing in itself.
- The labelling page opens the oldest batch that still has unlabelled frames.

## Screen classes (changed 17 Nov after the first labels)

- One `fight` class (key 8) for every mode; the mode comes from the detector (AUTO button, red enemy count). New
  `respawn` (key 9): dead or resetting inside a run (pink sky, no enemies, "None" where AUTO is). `dead` (key d):
  dead with neither respawn nor REVIVE on offer. Old labels were
  merged with `by: merge_fight_classes` rows; backups of the manifest and labels from before are in `data\`.
- Rules the first 120 labels proved wrong and were fixed: old-bot "raid_world" shots show the modes menu; "Titan
  results:" is logged after RETRY (loading); the GF queue hint must stop ~60 s before the next log line; "Create tab"
  and Friends Only appear in both windows. `tools/import_captures.py --rehint` recomputes every stored hint.
- Review batches (`"relabel": true` in the batch file) count only labels given after the batch was made.

## Crop heads (step 5b, `adbot2/crops.py`)

- Fixed UI is read from fixed canonical boxes (one box serves both layouts): `auto` (on/off/gone), `enemies`,
  `gf_banner`. The "None" sign while dead is on the hotbar, not in the AUTO box.
- Routine: `tools/cache_crops.py` (new frames only), label at http://127.0.0.1:8765/crops?head=auto (grid, Enter
  saves the page), `tools/train_crops.py --head auto` (seconds; writes preds.npz that pre-fills the next grid).
- AUTO acts only on P(off) >= 0.98; the trainer reports ON read as OFF, which must stay 0.
- Enemy count: `adbot2/digits.py` + `tools/digit_templates.py cluster | name RUN MAP | check`; templates live in
  `models\digits\` (never committed). The count flickers (0 between waves): debounce it, never act on one 0.

## Decisions (M2: `tracker.py`, `policy.py`, `shadowing.py`)

- Raid time = second of the *local* hour (`tracker.local_time`); zones are not all whole hours, so never take it
  from epoch seconds (20 Nov: v2's raid hour was 45 min off until the clock was read twice).
- The PARTY window (Start / Leave) is labelled as the window it opens from, so Start is read by colour at a fixed box
  (`buttons.py`, check with `tools/button_check.py`). Speed Raid results look like Dimension results (only LEAVE).
- Shadow on stored frames: `tools/shadow.py` (`--why ACTION` prints each miss with v2's view). Mode, loadout,
  Speed Raid schedule and settings come from the old bot's log (`OldBotState`). Live: `tools/shadow_live.py`
  (read-only, like the logger; rows in `reports\shadow_live\`; stop with its STOP file).
- Restarts the screen cannot see (not responding, no window, Titan rules) belong to the input layer, not the policy.
- Every tool that decodes frames checks `adbot2.resources.ram_ok()` (4 GB floor) before and between batches.
