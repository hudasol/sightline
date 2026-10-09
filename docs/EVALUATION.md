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

## Error analysis
`sightline errors` selects failures worst-first from saved results. The "why" column is written by a human after looking at the frames; it is not generated.
