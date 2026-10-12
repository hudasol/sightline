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

## Error analysis
`sightline errors` selects failures worst-first from saved results. The "why" column is written by a human after looking at the frames; it is not generated.
