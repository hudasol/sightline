# Sightline

Advisory video perception for an indoor inspection robot. It detects people,
keeps anonymous track ids through motion and short occlusion, turns tracks into
**one event per corridor entry/crossing**, and says so when the video is too
poor to trust. It never reports "clear" when it is not sure.

> **Advisory only.** No robot connection, no motion control, no face or
> biometric processing. Tracks are anonymous integers.

**Status: under construction.** `docs/PLAN.md` was committed first and is the
contract. No held-out metrics exist yet; none are claimed anywhere in this
repo until `results/` contains them. The table at the bottom is kept current.

## What is in the box

| Stage | Module | What it does |
|---|---|---|
| Ingest | `sightline/ingest.py` | One command records fps, frame count, per-frame timestamps and source metadata |
| Detect | `sightline/detector.py` | Pretrained torchvision person detector at a frozen confidence threshold |
| Track (baseline) | `sightline/tracking/iou_tracker.py` | Greedy IoU matching, no motion model |
| Track (chosen) | `sightline/tracking/bytetrack.py` | ByteTrack-style two-stage association, Kalman + Hungarian, geometry only |
| Track state | `sightline/tracking/state.py` | id, age, last seen, confidence, trajectory, explicit lost/expired policy |
| Events | `sightline/events.py` | Debounced open / active / close, re-entry merge, `UNRESOLVED` on lost track |
| Health | `sightline/health.py` | `OK / DEGRADED / UNAVAILABLE` from corrupt frames, stalls, low fps, low quality |
| Evaluation | `sightline/evaluation/` | Detection P/R/F1, IDF1 and ID switches, event P/R, latency, per-condition tables |
| Split safety | `sightline/manifest.py` | Clip manifests with hashes and a leakage check |
| Runtime | `sightline/runtime.py` | fps, p50/p95 latency, inference time, dropped/skipped frames |

## Setup

Python 3.12+ (developed against 3.13).

**Windows (PowerShell)**
```powershell
git clone https://github.com/hudasol/sightline
cd sightline
py -3.13 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
```

**macOS / Linux**
```bash
git clone https://github.com/hudasol/sightline && cd sightline
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

Model weights download on first use (torchvision, COCO). They are git-ignored.

## Data preparation

Raw video and frames are **not** in git (licences differ from the code's).
Put them under `data/` and describe them in a manifest. See
`docs/DATA_CARD.md` for sources, licences and the annotation method.

```text
data/
  mot17/MOT17-02-FRCNN/{img1,gt/gt.txt,seqinfo.ini}   # one detector copy per scene only
  own/<clip_id>.mp4
annotations/
  <clip_id>/gt.txt          # MOTChallenge format: frame,id,x,y,w,h,conf,class,vis
  <clip_id>/events.json     # ground-truth corridor events
configs/
  corridors/<camera_id>.yaml
  manifests/{val.json,test.json}
```

Check the split is leakage-safe and lock the held-out manifest **before** the
final tuning pass:

```bash
python -m sightline split-check --val configs/manifests/val.json --test configs/manifests/test.json
python -m sightline lock-manifest configs/manifests/test.json --out results/heldout_manifest.json
```

`split-check` fails if val and test share a scene, camera, session or MOT17
base scene. The evaluation refuses to run if a locked clip hash has changed.

## Ingest

```bash
python -m sightline ingest data/own/clip_01.mp4 --out results/ingest
python -m sightline ingest data/own/ --out results/ingest      # a folder
```

Writes `<clip>.ingest.json` with fps, frame count, resolution, codec, and a
timestamp per frame. Corrupt or non-monotonic frames are recorded, not hidden.

## Try it with no data and no model

```bash
python -m sightline demo          # generated clip, both trackers, writes outputs/demo
python -m pytest                  # unit + end-to-end tests
```

The demo is synthetic. It checks the code, not real-world performance.

## Splits, tuning, evaluation (order matters)

```bash
python -m sightline split-check --val configs/manifests/val.json --test configs/manifests/test.json
python -m sightline lock-manifest configs/manifests/test.json      # BEFORE tuning
python -m sightline detect-eval --manifest configs/manifests/val.json   # detector P/R/F1 vs threshold
python -m sightline tune                                           # val only -> results/frozen_config.yaml
python -m sightline evaluate                                       # locked held-out set, once
python -m sightline benchmark <clips> --corridor <corridor.yaml>   # fps / latency on THIS machine
python -m sightline stress                                         # blur/compression/low light/shake, no retuning
python -m sightline errors                                         # picks failures worst-first; you write the "why"
python -m sightline plots                                          # all plots from saved results
```

`tune` refuses to run without a locked held-out manifest and writes
`results/frozen_config.yaml` with provenance. `evaluate` refuses if the lock or
config provenance does not match, and a second run needs `--allow-rerun` (logged).

## Replay one clip

```bash
python -m sightline replay clip.mp4 --corridor configs/corridors/<yours>.yaml --config results/frozen_config.yaml --overlay
```

Exit code 2 means UNAVAILABLE (bad config, unreadable video, no detector); the
system says so instead of reporting a clear corridor.

Outputs, all under `results/`: detection metrics, IDF1 / ID switches for the
baseline and chosen tracker, event precision / recall / latency, false events
per 5 minutes of negative footage, per-condition tables, runtime report.

## Replay one clip

```bash
python -m sightline replay data/own/clip_01.mp4 \
    --config results/frozen_config.yaml --tracker bytetrack --out outputs/replay
```

Writes an overlay video, `events.jsonl`, `tracks.jsonl`, `health.jsonl` and a
runtime report. Use `--tracker iou` to run the baseline on the same clip.

## Break it on purpose

```bash
python -m sightline replay does_not_exist.mp4 --config results/frozen_config.yaml
python -m sightline replay data/own/clip_01.mp4 --config configs/invalid.yaml
```

Both must end in `UNAVAILABLE`, never `clear`. A stalled stream and a
corrupt file are covered by `tests/test_health.py` and the end-to-end test.

## Stress suite

```bash
python -m sightline stress --split test --config results/frozen_config.yaml
```

Degrades held-out clips (blur, compression, low light, shake) at several
levels using the **frozen** config and plots the degradation curve. It does not
retune anything.

## Tests

```bash
pytest -q
```

Covers event state, geometry, tracker expiry and re-entry, invalid
configuration, degraded states, split leakage, and an end-to-end replay on a
generated synthetic clip (no downloads needed).

## Repository layout

```text
docs/        PLAN, DATA_CARD, MODEL_CARD, EVALUATION, PROCESS_LOG, RETROSPECTIVE
sightline/   package
configs/     corridors, manifests, default parameters
results/     machine-readable metrics, frozen config, held-out manifest
tests/
```

## Status

| Item | State |
|---|---|
| `docs/PLAN.md` | done |
| Pipeline code and tests (synthetic + unit) | done, passing |
| Real detector run (needs torch + weights) | not run yet |
| Real datasets assembled and annotated | not started (needs footage) |
| Held-out evaluation | not run |
| Demo video, retrospective, `v1.0.0` | not started |

## Licences

Code: MIT. Third-party: torchvision (BSD-3-Clause), COCO-trained weights,
MOT17 (non-commercial terms, see `docs/DATA_CARD.md`). Do not commit clips or
weights.
