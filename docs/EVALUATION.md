# Evaluation

**No held-out or own-clip result exists yet.** Held-out numbers will come only from `results/final_metrics.json`. The MOT17 **val** sections below cite their own files under `results/`. Every number must trace to a saved file; do not hand-type metrics.

## Protocol
1. Build `configs/manifests/val.json` and `configs/manifests/test.json`; `sightline split-check` must pass.
2. `sightline lock-manifest` on the test manifest.
3. `sightline detect-eval` and `sightline tune` on val only, writing `results/frozen_config.yaml` (with provenance).
4. `sightline evaluate` once on the locked held-out set. Re-runs require `--allow-rerun` and are logged in `final_run_log.jsonl`.
5. `sightline benchmark` on the demo machine; `sightline stress` on the frozen config (no retuning); `sightline errors` and `plots`.

## Acceptance bar (from the brief)
| Metric | Bar | Result |
|---|---|---|
| Detection precision / recall (IoU 0.5), own clips in scope (A1) | >= 0.80 / >= 0.80 | not run |
| IDF1 | >= 0.65 and > baseline | not run |
| Event precision / recall | >= 0.85 / >= 0.85 | not run |
| False events per 5 min (negative footage) | <= 1 | not run |
| Median event latency | <= 5 frames | not run |
| Throughput | >= 10 fps | not run on the held-out set |
| Held-out clips | >= 12 | 2 listed in `test.json` (MOT17-09, 10), 0 evaluated, not locked |

## What exists today
- **Tests on a synthetic clip:** unit and end-to-end tests cover geometry, config validation, tracker expiry and re-entry, the event state machine, health states, manifest locking and CLI failure paths. On that clip, `tests/test_e2e.py` checks that the ByteTrack-style tracker uses no more ids than the IoU baseline. `sightline demo` prints both id lists. This is a correctness check of the code, **not** a performance claim, and no file saves those counts.
- **MOT17 val measurements:** see the sections below.

## Scope: moving cameras count for detection and tracking only
The corridor is an image-space polygon, so it means nothing once the camera moves. MOT17-05, 10, 11 and 13 are filmed from a moving platform. Their manifest entries say `"metrics": ["detection", "tracking"]`, and every event number (precision, recall, latency, false events, false CLEAR) pools only clips scored on events. Decided 2026-10-12, before any event metric was computed.

What this does to the held-out event numbers. Sources: `results/split/split_check.json`, and for the GT event counts, `results/labels/mot17_gt_events_draft.json`.

| | Before scoping | After scoping |
|---|---|---|
| Held-out clips scored on events | 2 (09, 10) | 1 (09) |
| Label-derived held-out GT events from MOT17 | 27 (14 + 13) | 14 |
| Difficult conditions covered *for events* | camera motion, low light (both from 10) | none |
| Val clips used for event tuning | 5 | 2 (02, 04) |

Consequences:
- **Coarser numbers.** Until own clips are added, held-out event recall moves in steps of 1/14 (about 7 points). Reaching the 0.85 bar needs at least 12 of the 14 events.
- **No hard conditions for events.** Detection and tracking still see camera motion and night (MOT17-10). The event metrics see neither until own clips cover them.
- **Event coverage must come from own clips.** That means negatives, crowding and low light. A handheld clip has a moving camera, so it is excluded from events too. Camera-motion coverage for events needs a static camera on a blurred scene, or a ground-plane corridor (PLAN section 11).
- MOT17-10's 13 label-derived events are not lost evidence. Under a drifting polygon they were never valid labels.

## Val results so far (MOT17 only; untuned; held-out set not locked)
These are **val** measurements on public MOT17 footage. They compare configurations;
they do **not** select one. Under PLAN.md amendment A4 the detector and detection
interval are chosen on own val clips, and MOT17 does not enter that choice. They say
nothing about the held-out set or the bar. Under A1 the detection bar is judged on own
corridor clips, none of which exist yet.

**Detector F1 against speed.** Sources:
- F1: `results/detector_selection/summary.md` (all 4137 val frames, IoU 0.5, thresholds 0.15-0.90).
- Speed: `results/runtime/interval/*` (first 150 frames of each val scene, every frame processed). This is **one offline run per cell**: frames are not paced in real time, which is not yet the A4 protocol of a median of 3 runs.
- Each cell gives fps / worst per-scene p95 latency / maximum latency, in ms.

| Detector | Best F1 in scope (P / R) | Best F1, all boxes | k = 1 | k = 2 | k = 3 |
|---|---|---|---|---|---|
| MobileNetV3-320 @640x360 | 0.607 (0.745 / 0.512) | 0.389 | 5.83 fps / 195 / 205 | 11.41 / 181 / 205 | 16.53 / 189 / 220 |
| YOLOX-tiny @416 | 0.778 (0.805 / 0.753) | 0.531* | 21.50 / 52 / 123 | 39.22 / 51 / 63 | not run |
| YOLOX-s @640 | 0.853 (0.842 / 0.863) | 0.625* | 6.03 / 245 / 291 | 13.08 / 158 / 179 | 18.89 / 167 / 194 |
| ResNet50-FPN-v2 @960x540 (reference) | 0.857 (0.824 / 0.891) | 0.673 | 0.19 fps (earlier session, `results/runtime/candidates/`) | not run | not run |

\* At 0.15, the lowest threshold of the grid. A lower threshold might score higher. The grid was widened from 0.30 after one-stage best F1 values sat on the old floor (PLAN.md A5).

- **What the table shows.** On MOT17 val, YOLOX-s at k = 2 is the only measured
  setting whose detector reaches in-scope P and R >= 0.80 **and** whose pipeline
  ran at >= 10 fps offline. That is one run, not paced in real time.
- **Real time is different.** Paced in real time and synchronous, the same setting
  ran at 6.25 fps and dropped most frames (worker-thread section below).
- **Slow frames remain.** The frames on which YOLOX-s runs take up to 291 ms (k = 1)
  and 179 ms (k = 2). They individually miss the 100 ms budget; throughput is an
  average.
- **Timing is noisy.** A MobileNet control re-timed in two sessions ran at 4.99 and
  5.81 fps with the same code, so expect about 15% run-to-run variation.

**Events against detection interval.** Source: `results/interval/*_time_params.json`.
Event clips MOT17-02 and 04 only (20 GT events), cached detections, untuned
defaults (score threshold 0.5, A3 time-based settings).
- **Rows shown:** a selection. The files also hold k = 4 and 6 for each detector.
  For example, YOLOX-s at k = 6 has precision 0.952 and recall 1.00, with 21 predicted events.
- **Not shown:** the YOLOX-tiny results (`val_interval_yolox_tiny_416x416_time_params.json`)
  and the frame-based settings (`*_frame_params.json`).

| Detector | k | Event recall | Event precision | Median latency (frames) | Opened on prediction |
|---|---|---|---|---|---|
| YOLOX-s @640 | 1 | 0.95 | 0.655 | 4.0 | 0 / 29 |
| YOLOX-s @640 | 2 | 0.95 | 0.760 | 4.0 | 5 / 25 |
| YOLOX-s @640 | 3 | 1.00 | 0.769 | 3.5 | 16 / 26 |
| MobileNetV3-320 @640x360 | 2 | 1.00 | 0.588 | 5.5 | 6 / 34 |

- Twenty events is far too few to tune on. Recall moves in 5-point steps, and the
  medians are noisy.
- Event precision is below the 0.85 bar for every row at k <= 3.
- Many events end UNRESOLVED: 10-16 per row for YOLOX-s and 14-22 for MobileNet, in
  the same files. Not all of these are failures. For YOLOX-s at k = 2, 6 of the 11
  are tracks lost while the event was open; the other 5 were still open when the clip
  ended (`stream_end`). See the error analysis below.
- Why tracks are lost is not yet written up. Tracker tuning on val, after the held-out
  set is locked, is where it gets addressed.
- `rests_on_prediction` was 0 in every row shown, and 1 at k = 4 and 6 for
  YOLOX-s: almost every event had at least one detector frame inside the
  corridor.

## Detection on a worker thread: latency before and after
Source: `results/runtime/async/{before_sync,after_sync,after_async}/summary.json`, produced by `sightline runtime-summary`.

**Protocol.** `benchmark --realtime`: frames arrive on the clip's own clock. A frame that arrives while a detector frame is still being processed waits, or is dropped once it lags past the 100 ms budget. Each config ran 3 times on the first 150 frames of the five val scenes.
- "before" = the synchronous pipeline, at commit fe15d93.
- "after_sync" = the same synchronous code path at 9ff5dad, as a control.
- "after_async" = 9ff5dad with `--async-detection`.
- Values are the median of 3 runs (min-max). Latency is the worst per-clip p95 and the overall maximum.

| Config | Mode | Throughput fps | Worst clip p95 ms | Max ms | Frames dropped (of 750) | Detector frames skipped, worker busy |
|---|---|---|---|---|---|---|
| YOLOX-s @640, k = 2 | before | 6.25 (6.13-7.61) | 354 (315-551) | 700 (465-721) | 546 | 0 of 375 |
| | after_sync | 6.74 (5.87-7.25) | 380 (350-381) | 393 (364-440) | 530 | 0 of 375 |
| | after_async | 23.78 | 15 (12-36) | 27 (17-93) | 0 | 237 of 375 |
| YOLOX-s @640, k = 3 | before | 10.46 (10.23-10.65) | 294 (291-314) | 347 (334-470) | 415 | 0 of 250 |
| | after_sync | 12.27 (12.18-12.39) | 254 (253-265) | 336 (263-362) | 357 | 0 of 250 |
| | after_async | 23.78 | 26 (21-27) | 54 (42-79) | 0 | 120 of 250 |
| MobileNetV3-320 @640x360, k = 2 | before | 6.36 (6.19-6.49) | 330 (310-350) | 397 (350-430) | 543 | 0 of 375 |
| | after_sync | 7.18 (6.85-7.34) | 296 (295-300) | 314 (304-318) | 517 | 0 of 375 |
| | after_async | 23.78 | 13 (13-17) | 20 (17-34) | 0 | 259 of 375 |

What this does and does not show:
- **It removes blocking.** With the worker, no frame waits for the detector: no frame is dropped, and p95 latency falls from about 250-380 ms to 13-26 ms.
- **It does not make the detector faster.** In async runs, 23.78 fps is the source rate of these clips (30, 30, 14, 30 and 25 fps), so every frame was processed in time. But 48-69% of the detector frames were skipped because the worker was still busy, so detections happen only as often as the detector can run. Between them the tracker predicts for longer than k frames.
- **Its effect on accuracy is not measured.** When a result arrives depends on wall-clock timing, so cached-detection evaluation cannot reproduce async runs. Event recall and latency in async mode are untested.
- **Synchronous real time does not keep up.** Even YOLOX-s at k = 3 dropped about half of all frames (415 of 750) in sync mode.
- **Timing is noisy.** The control (same synchronous code) differs from "before" by up to 17% in throughput.

## Error analysis
`sightline errors` selects failures worst-first from saved results. The "why" column is written by a human after looking at the frames; it is not generated.

- **Unresolved events on val** (YOLOX-s @640, k = 2, untuned):
  `results/error_analysis/val_unresolved_yolox_s_640x640_k2/INDEX.md`.
  - 10 were requested and 6 met the rule (track lost while the event was open).
    These are the same 6 `track_lost` events behind the interval run's 11
    unresolved.
  - The "why" column is still TODO for a human.
  - No setting was changed from these strips. The frame images stay out of git
    because they show faces.
