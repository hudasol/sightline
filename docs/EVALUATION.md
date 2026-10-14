# Evaluation

**No real-data results exist yet.** This file will be filled from `results/final_metrics.json` only; every number must trace to a saved file. Do not hand-type metrics.

## Protocol
1. Build `manifests/val.json` and `manifests/test.json`; `sightline split-check` must pass.
2. `sightline lock-manifest` on the test manifest.
3. `sightline detect-eval` and `sightline tune` on val only → `results/frozen_config.yaml` (with provenance).
4. `sightline evaluate` once on the locked held-out set. Re-runs require `--allow-rerun` and are logged in `final_run_log.jsonl`.
5. `sightline benchmark` on the demo machine; `sightline stress` on the frozen config (no retuning); `sightline errors` and `plots`.

## Acceptance bar (from the brief)
| Metric | Bar | Result |
|---|---|---|
| Detection precision / recall (IoU 0.5) | ≥ 0.80 / ≥ 0.80 | not run |
| IDF1 | ≥ 0.65 and > baseline | not run |
| Event precision / recall | ≥ 0.85 / ≥ 0.85 | not run |
| False events per 5 min (negative footage) | ≤ 1 | not run |
| Median event latency | ≤ 5 frames | not run |
| Throughput | ≥ 10 fps | not run |
| Held-out clips | ≥ 12 | 0 |

## What exists today
Unit and end-to-end tests on a **synthetic** clip only (geometry, config validation, tracker expiry/re-entry, event state machine, health states, manifest locking, CLI failure paths). On that clip ByteTrack-style keeps 3 ids where the IoU baseline makes 4. This is a correctness check of the code, **not** a performance claim.

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
These are **val** measurements on public MOT17 footage. They choose a configuration;
they say nothing about the held-out set or the bar. Under PLAN.md amendment A1 the
detection bar is judged on own corridor clips, none of which exist yet.

**Detector F1 against speed.** Sources: `results/detector_selection/summary.md`
(all 4137 val frames, IoU 0.5, thresholds 0.15-0.90) and
`results/runtime/interval/*` (first 150 frames of each val scene, every frame
processed, same session).

| Detector | Best F1 in scope (P / R) | Best F1, all boxes | fps, k = 1 | k = 2 | k = 3 |
|---|---|---|---|---|---|
| MobileNetV3-320 @640x360 | 0.607 (0.745 / 0.512) | 0.389 | 5.83 | 11.41 | 16.53 |
| YOLOX-tiny @416 | 0.778 (0.805 / 0.753) | 0.531 | 21.50 | 39.22 | not run |
| **YOLOX-s @640** | **0.853 (0.842 / 0.863)** | 0.625 | 6.03 | **13.08** | 18.89 |
| ResNet50-FPN-v2 @960x540 (reference) | 0.857 (0.824 / 0.891) | 0.673 | 0.19 (k = 1, earlier session) | not run | not run |

- YOLOX-s with `detect_every: 2` is the first setting whose detector reaches
  P/R >= 0.80 in scope on val **and** whose pipeline sustains >= 10 fps.
- Frames on which YOLOX-s runs still take up to 158-245 ms (worst per-scene p95).
  Those frames individually miss the 100 ms budget; throughput is an average.
- A MobileNet control re-timed in two sessions ran at 4.99 and 5.81 fps with the
  same code, so expect about 15% run-to-run variation.

**Events against detection interval.** Source: `results/interval/*_time_params.json`.
Event clips MOT17-02 and 04 only (20 GT events), cached detections, untuned
defaults (score threshold 0.5, A3 time-based settings).

| Detector | k | Event recall | Event precision | Median latency (frames) | Opened on prediction |
|---|---|---|---|---|---|
| YOLOX-s @640 | 1 | 0.95 | 0.655 | 4.0 | 0 / 29 |
| YOLOX-s @640 | 2 | 0.95 | 0.760 | 4.0 | 5 / 25 |
| YOLOX-s @640 | 3 | 1.00 | 0.769 | 3.5 | 16 / 26 |
| MobileNetV3-320 @640x360 | 2 | 1.00 | 0.588 | 5.5 | 6 / 34 |

- Twenty events is far too few to tune on. Recall moves in 5-point steps, and the
  medians are noisy.
- Event precision is below the 0.85 bar for every row at k <= 3. Many events end
  UNRESOLVED (10-16 per row for YOLOX-s, 14-22 for MobileNet, in the same files),
  which points to fragmented tracks re-opening events. That has not yet been
  checked frame by frame (`sightline errors`). Tracker tuning on val, after the
  held-out set is locked, is where it gets addressed.
- `rests_on_prediction` was at most 1 event per row: almost every event had at
  least one detector frame inside the corridor.

## Error analysis
`sightline errors` selects failures worst-first from saved results. The "why" column is written by a human after looking at the frames; it is not generated.
