# Research: how v2 should read and handle the whole game (17 Nov)

The question: after the first screen classifier (97% on a held-out day, 19 ms per frame), what is the most efficient way
to read *everything* the game shows and handle every mode? This note compares the options, records licences, and ends
with a recommended order. Sources are at the end; fan-wiki facts about the game are unofficial and must be checked
against our own frames.

## 1. The main finding: most of the game does not need a detector

The game's UI is anchored: the hotbar, AUTO button, banners, results buttons and windows line up within 1 px between
the 100% and 150% layouts (measured 17 Nov), and the top-left panel moves by a fixed rule. A general object detector
(boxes anywhere) is the right tool only for things whose position *varies*. So perception splits in three:

| What | Where it is | Best tool | Labelling cost |
|---|---|---|---|
| Which screen | whole frame | screen classifier (done) | 1 key per frame |
| AUTO on / off / none, Group Finder banner, results buttons (RETRY / LEAVE / AUTO RETRY countdown), REVIVE, Start prompt, "Teleporting in N" | fixed spot per layout | **crop classifiers**: tiny CNN heads on fixed crops | 1 key per crop |
| Enemy count, raid clock ("Raid Opens in MM:SS"), gems / tokens, run count | fixed spot | **digit reader** (CTC) on the crop | type the number, pre-filled |
| Boss HP bar, HP / XP | fixed spot | pixel measurement (fraction of bar filled) | none |
| Enemies, boss, own character, portals in the 3D world; entries in scrolling lists (raid names, Speed Raid characters, dimmed = used); pop-ups with a close X at varying places | anywhere | **object detector** | boxes, pre-filled |

Why crop heads beat a detector for fixed UI:
- **More accurate:** the classifier sees the button at full resolution, not shrunk to 448 px wide. AUTO ON must never
  read as OFF; a crop at native pixels with its own head and a safety margin is the safest way to guarantee that.
- **Faster:** a 64x64 crop through a tiny CNN is well under 1 ms. All heads can run in one batched call.
- **Cheaper to label:** one key per crop, not drawing boxes. Weak labels already exist ("Auto pressed (try N)" = OFF just
  before, the old bot's GF banner reads, OCR'd clock values).

## 2. Detector choice (for the things that move)

Only Apache-2.0 (or equally permissive) code **and** weights; no Ultralytics (AGPL).

| Model | Code | Weights | Fit for us |
|---|---|---|---|
| **RF-DETR N / S / M / L** (Roboflow, ICLR 2026) | Apache 2.0 | Apache 2.0 (N, S, M, L, all Seg sizes); backbone DINOv2 (Apache 2.0) | **Recommended.** `pip install rfdetr`, actively maintained (1.12.1, 12 Nov 2026), built for fine-tuning on small custom sets. Dry run on our Python 3.14 venv: every dependency has a wheel, torch untouched (~45 small packages incl. transformers, supervision, scipy). N: 2.3 ms, S: 3.5 ms on a T4 with TensorRT. |
| RF-DETR A / F / P / XL / 2XL | - | **PML 1.0** (Roboflow platform licence; commercial self-hosting needs a paid add-on) | Avoid (installed only by `rfdetr[plus]`). |
| **D-FINE N-X** | Apache 2.0 | COCO weights fine; the authors warn that **Objects365** weights may carry Objects365 dataset terms | Good second choice. Trained from a cloned repo with config files (not a pip package), last release Nov 2024, tested on Python 3.11. More upkeep than RF-DETR. |
| DEIM (training method for D-FINE) | Apache 2.0 | as D-FINE | Faster convergence; same upkeep issue. |
| **DEIMv2** | "DEIMv2 License" (custom; commercial use needs contacting the authors); S-X use DINOv3 | unclear | **Excluded:** not Apache. |
| RT-DETR / RT-DETRv2 | Apache 2.0 | Apache 2.0 | Older and slower than the two above; fallback only. |

Plan: RF-DETR-Small at 512 px on the fight view, run only on fight frames and only as often as the fight logic needs
(5-10 Hz), exported to ONNX. Train first on boxes pre-filled from the old bot's knowledge (fixed UI boxes come free; the
detector then only needs world objects).

## 3. Box labelling workflow (proposal; nothing installed yet)

| Tool | Licence | For us |
|---|---|---|
| **Our own labelling page + a box mode** | ours | **Recommended.** Same page, same keyboard flow; boxes arrive pre-filled from the current model; you only drag a corner or press a key to accept / delete / change class. No server, no Docker, no pop-ups. About a day of work. |
| Label Studio | Apache 2.0 (core) | Strong, ML-backend pre-labels, but a Django server and a bigger install; mouse-heavy. |
| CVAT | MIT | Best box tooling, but self-hosting needs Docker. |
| X-AnyLabeling | GPL-3.0 (as far as known) | Desktop app with built-in models; GPL is fine for a tool we only use, but it opens windows. |

Workflow: pick frames by uncertainty (as now) -> model pre-fills boxes -> you correct (target: 10 s a frame) ->
retrain (minutes) -> repeat. Fixed-UI boxes never need drawing: they come from geometry.

## 4. Digits and text

- **Own CTC reader per field** (enemy count, raid clock, gems, tokens): a small CNN+BiLSTM/CTC on the crop, trained on
  crops whose numbers come from Windows OCR **plus consistency checks** (a clock counts down one second per second, the
  enemy count only drops within a wave, two reads must agree) and your corrections. This directly targets the old bot's
  dropped-digit failures ("41:28" as "4:28", "22:02" as "2:02"): the reader is trained on exactly that font, and any
  reading that breaks the time rule is rejected.
- Generic fallbacks if a field is too varied: **PARSeq-tiny** (Apache 2.0 code and weights on Hugging Face) or
  **PP-OCRv5** as ONNX (Apache 2.0; community ONNX conversions, so check the upstream licence of the file used).
- Game fonts change (13 Nov AUTO redraw, 12 Nov sub-pixel text change): synthetic training crops rendered from samples
  plus augmentations keep the reader robust, and the regression set catches the next change.

## 5. Making the whole loop fast and safe

- **Capture:** mss (GDI) is fine for the 2 fps logger. For the live bot, **Windows Graphics Capture** (e.g. the
  `windows-capture` package, Rust-backed; or DXcam's Desktop Duplication) delivers only *new* frames, straight from the
  GPU, and WGC can capture the Roblox window even when another window covers it (input still needs focus). Licences of
  these packages to be checked before install; the yellow capture border can be turned off in Windows 11 settings.
- **One network pass per frame:** screen classifier + all crop heads in one batched ONNX call; the detector only on
  fight frames. Target: about 10 ms total on the 4060.
- **Speed-ups, in order of effort:** do the shrink on the GPU (now 7 ms on the CPU); FP16 export; ONNX Runtime
  IO binding; the **TensorRT-RTX execution provider** (`onnxruntime-ep-nv-tensorrt-rtx-cu13`, RTX 30-series or newer)
  if still needed. That is a large NVIDIA download: ask first.
- **Never act on one frame:** smooth the screen over time (an HMM / forward filter over allowed screen transitions:
  "fight" cannot jump to "modes_menu" without "loading"), and require N agreeing frames plus a confidence margin for any
  click that can hurt (AUTO, LEAVE, REVIVE is never pressed).
- **Know when it does not know:** an out-of-distribution score (energy of the logits, or distance to the nearest labelled
  frames in embedding space) marks screens it has never seen (a new pop-up after an update). Those frames are saved
  for labelling and the bot falls back to a safe action (wait, Esc, or the old bot's rule), never a guess.

## 6. Everything the game offers, and what the bot needs for each

From the old bot's spec (tested on this server) and the fan wiki (unofficial, Sept-Oct 2026). "Have" = the old bot
handles it today.

| Mode / feature | Have | What v2 must read | Notes |
|---|---|---|---|
| Hourly Raid (Divine Guardian here; 10 raids in the list), Group Finder or Create | yes | raid clock (modes menu card, raid-world banner), GF banner, raid list by name, results + rewards | opens at a server-specific minute; 8-min fight; create only in ~first 10 min |
| Speed Raid | yes | character picker (dimmed = used this week), results | 1 per character per week; can drag on |
| Infinity / Infinite Mode | yes | AUTO, results with AUTO RETRY countdown, reset (respawn -> dead -> results) | ~30 min a run on Auto; lid-shut nights run only this |
| Titan Dimension (walked route) | yes (100% only) | enemy count, boss bar, legs, respawn, results (short: "Teleporting in N") | the hard one: no AUTO; needs navigation (M4); fails at 150% today |
| Extreme Dimension | partly | difficulty window (Easy / Normal / Nightmare / Extreme, run N today, reward %), results | revamped in the Halloween update (no entry cap, lower rewards per run) |
| Death Dimension (recorded run) | partly | respawn, revive screen (never press), results | recordings break after a respawn |
| Boss Rush | no | entry (1 free a day, then tickets), 3 bosses, results | a daily the old bot does not do |
| Time Challenge | no | solo ~8 min, kills -> Virtual cards | unlocks at level 5 |
| Daily quests and daily rewards | no | claim pop-ups (now "other") | gems; reset times to be read from the game |
| Codes | no | CODES button, redeem box | new codes come with updates |
| Event raids and keys | no | event banners, key counts | rotate (e.g. a Halloween raid until 8 Nov) |
| Summons, fruit banner, pet capsules, upgrades, cards, traits | no (on purpose) | recognise as "other" and close | spending decisions stay with you |
| Loadouts (Bag) | yes | bag screen, loadout number | 1 raid, 2 Infinity, 3 Speed Raid |
| Roblox trouble: freeze, Not Responding, disconnect, update notes | yes | disconnected screen, frozen picture, Esc menu | preventive restart after ~2 h |

So "handle everything" adds four new activities (Boss Rush, Time Challenge, dailies / claims, codes) and two fixes
(Titan at 150%, Extreme Dimension). Each needs only a few new screen classes and crop heads, plus decision rules.

## 7. Further out

- **Navigation in Dimensions (M4):** the detector finds enemies and the boss; the bot turns towards the nearest group,
  walks until in range, uses Auto-like attacks, and detects being stuck (picture not moving while walking). This
  replaces fixed routes that break at 150% or after map changes.
- **Vision-language model as a helper, not a driver:** Florence-2 (MIT, 0.23-0.77 B) or SmolVLM (Apache 2.0) can run
  locally to *pre-label* new screen types or read odd text offline. Not in the live loop (too slow, can hallucinate).
  Qwen2.5-VL-3B is stronger but its licence must be checked first (reported as research-only).
- **Behaviour cloning** from your own play: only after M4 gives a baseline to beat.

## 8. Recommended order

1. **Crop heads + digit readers** for AUTO, GF banner, results buttons, enemy count, raid clock. No new installs;
   labels by key; weak labels from the log already exist. Biggest win for M1 (AUTO never OFF, count 99%+).
2. **Temporal smoothing + "unknown" detection** on top of the screen classifier.
3. **More screen classes** for the new activities (Boss Rush, Time Challenge, claim pop-ups, codes) as frames of them
   appear in labelling.
4. **RF-DETR-Small** + the box mode on the labelling page, for world objects and lists (install ~45 small packages:
   ask first).
5. **WGC capture** test for the live bot (small install: ask first).
6. Speed work only if the whole frame is over budget.

## Sources

- RF-DETR: https://github.com/roboflow/rf-detr , https://pypi.org/project/rfdetr/
- D-FINE: https://github.com/Peterande/D-FINE ; DEIM: https://github.com/Intellindust-AI-Lab/DEIM ;
  DEIMv2: https://github.com/Intellindust-AI-Lab/DEIMv2 ; RT-DETR: https://github.com/lyuwenyu/RT-DETR
- PARSeq: https://github.com/baudm/parseq ; PP-OCR ONNX: https://huggingface.co/monkt/paddleocr-onnx
- Capture: https://pypi.org/project/windows-capture/ , https://github.com/ra1nty/DXcam ,
  https://learn.microsoft.com/en-us/uwp/api/windows.graphics.capture.graphicscapturesession.isborderrequired
- ONNX Runtime TensorRT EPs: https://onnxruntime.ai/docs/execution-providers/TensorRT-ExecutionProvider.html ,
  https://pypi.org/project/onnxruntime-ep-nv-tensorrt-rtx/
- Labelling tools: https://roboflow.com/compare-labeling-tools/cvat-vs-label-studio
- Small VLMs: https://huggingface.co/blog/vlms-2025
- OOD for GUI agents: https://arxiv.org/abs/2505.12842
- Game (fan wikis, unofficial): https://animedimensionssimulator-wiki.wiki/beginner-guide/ ,
  https://animedimensionssimulator-wiki.wiki/guides/dimensions-and-levels/ , https://anime-dimensions.fandom.com/wiki/Raid ,
  https://anime-dimensions.fandom.com/wiki/Speed_Raid
