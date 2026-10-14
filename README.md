# Sightline

Advisory video perception for an indoor inspection robot. It detects people,
keeps anonymous track ids through motion and short occlusion, turns tracks into
**one event per corridor entry/crossing**, and says so when the video is too
poor to trust. It never reports "clear" when it is not sure.

> **Advisory only.** No robot connection, no motion control, no face or
> biometric processing. Tracks are anonymous integers.

**Status: under construction.** `docs/PLAN.md` was committed first and is the
contract. Only MOT17 **val** measurements exist (`docs/EVALUATION.md`). No held-out
or own-clip metric exists, and none is claimed until `results/` contains it. The status
table at the bottom was last updated on 2026-10-14.

## What is in the box

| Stage | Module | What it does |
|---|---|---|
| Ingest | `sightline/ingest.py` | One command records fps, frame count, per-frame timestamps and source metadata |
| Detect | `sightline/detector.py` | Pretrained COCO person detectors (torchvision, or YOLOX on ONNX Runtime); the operating threshold is frozen on val, which has not happened yet |
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

## Own footage: from phone to labelled clip

Phones record variable frame rate (VFR), often HEVC, sometimes with rotation
metadata, and `check-clip` flags all three. Convert every recording once with
ffmpeg, which is free (ffmpeg.org; on Windows `winget install Gyan.FFmpeg`). This
gives a constant 30 fps, 1920x1080, H.264 file and **keeps the original
untouched**.

Keep originals in `data/own/originals/` (git-ignored). `-n` refuses to overwrite
an existing file.

```bash
ffmpeg -n -i data/own/originals/IMG_1234.MOV -vf "scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2,fps=30" -c:v libx264 -preset slow -crf 18 -pix_fmt yuv420p -an -movflags +faststart data/own/own-corridorA-20261014-03.mp4
```

- `fps=30` makes the frame rate constant by repeating or dropping frames. Timing
  shifts by at most one frame (33 ms).
- `scale` and `pad` keep the aspect ratio. Record in landscape, because a
  portrait clip ends up with black side bars.
- ffmpeg applies the phone's rotation metadata while it re-encodes.
- `-an` drops the audio track. It is not used, and voices are personal data.

Then, per clip:

```bash
python -m sightline check-clip data/own/own-corridorA-20261014-03.mp4
python -m sightline draw-corridor data/own/own-corridorA-20261014-03.mp4 --out configs/corridors/own-corridorA-tripod1.yaml
python -m sightline label-events data/own/own-corridorA-20261014-03.mp4 --corridor configs/corridors/own-corridorA-tripod1.yaml --out annotations/own-corridorA-20261014-03/events.json
python -m sightline import-cvat export.zip --out annotations/own-corridorA-20261014-03 --frames 1-900:10
```

- `check-clip` exits 1 if anything would break ingest.
- Draw the corridor once per camera position, not once per clip.
- `draw-corridor` and `label-events` open an OpenCV window. The project installs
  `opencv-python-headless`, which has no window support. On the machine where you
  label, swap it for the GUI build of the same version:
  `pip uninstall -y opencv-python-headless` and then
  `pip install opencv-python==<version>`.
- See `docs/RECORDING_GUIDE.md` for what to film, and
  `configs/manifests/own_clip.template.json` for the manifest entry.

## Try it with no data and no model

```bash
python -m sightline demo          # generated clip, both trackers, writes outputs/demo
python -m pytest                  # unit + end-to-end tests
```

The demo is synthetic. It checks the code, not real-world performance.

## Splits, tuning, evaluation (order matters)

```bash
python -m sightline split-check --val configs/manifests/val.json --test configs/manifests/test.json --out results/split/split_check.json
python -m sightline detect-eval --manifest configs/manifests/val.json --out <file>   # val only: P/R/F1 vs threshold
python -m sightline interval-eval --every 1 2 3 --out <file>      # val only: event recall/latency vs detection interval
python -m sightline lock-manifest configs/manifests/test.json      # BEFORE tuning; "draft" until coverage is met
python -m sightline tune                                           # val only -> results/frozen_config.yaml
python -m sightline evaluate                                       # locked held-out set, once
python -m sightline benchmark <clips> --corridor <corridor.yaml>   # fps / latency on THIS machine
python -m sightline stress                                         # blur/compression/low light/shake, no retuning
python -m sightline errors                                         # picks failures worst-first; you write the "why"
python -m sightline plots                                          # all plots from saved results
```

Every frame is processed. `pipeline.detect_every: k` runs the detector on every k-th
frame, and the tracker predicts in between. Events record how much of their evidence
came from predicted boxes (PLAN.md A2). Tracker and event settings are given in
seconds and converted at each clip's fps (A3).

Detectors (`--detector`):
- torchvision: `fasterrcnn_mobilenet_v3_large_320_fpn`, `fasterrcnn_resnet50_fpn_v2`,
  `ssdlite320_mobilenet_v3_large`
- YOLOX through ONNX Runtime: `yolox_nano`, `yolox_tiny`, `yolox_s`. Install with
  `pip install -e ".[onnx]"`, and put the 0.1.1rc0 `.onnx` files in `weights/yolox/`.

`detect-eval`, `interval-eval`, `benchmark` and `threads` accept `--detector NAME` and
`--input-size W H` to compare detector candidates without editing the config.
`benchmark --max-frames N` times only the first N frames of each clip and says so
in the report. `threads <clip>` times the detector at several torch thread counts.

`benchmark --async-detection` runs the detector on a worker thread, so other frames
never wait for it (`pipeline.async_detection`). It is off by default, and its effect on
accuracy is not yet measured (`docs/EVALUATION.md`). `runtime-summary <folder>` gives
the median and range over repeated benchmark runs (PLAN.md A4).

`tune` refuses to run unless the held-out lock is **final** (coverage met) and
unchanged, and writes `results/frozen_config.yaml` with provenance. `evaluate` refuses if the lock or
config provenance does not match, and a second run needs `--allow-rerun` (logged).
None of this has run yet. Once it has, `evaluate` writes under `results/`: detection
metrics, IDF1 / ID switches for the baseline and the chosen tracker, event precision,
recall and latency, false events per 5 minutes of negative footage, per-condition
tables and the runtime report.

## Replay one clip

```bash
python -m sightline replay data/own/clip_01.mp4 --corridor configs/corridors/<yours>.yaml --config configs/default.yaml --tracker bytetrack --out outputs/replay --overlay
```

This writes an overlay video, `events.jsonl`, `tracks.jsonl`, `health.jsonl` and a
runtime report to `--out`. Use `--tracker iou` to run the baseline on the same
clip. Once a frozen config exists, pass `--config results/frozen_config.yaml`.

Exit code 2 means UNAVAILABLE (bad config, unreadable video, no detector). The
system says so instead of reporting a clear corridor.

## Break it on purpose

```bash
python -m sightline replay does_not_exist.mp4 --corridor configs/corridors/synthetic.yaml --config configs/default.yaml
```

This ends in `UNAVAILABLE` (exit 2), never `clear`. An invalid config does the
same. For example, write `detector: {scor_threshold: 0.4}` (a typo in a key) to
`bad.yaml` and pass `--config bad.yaml`: unknown keys are rejected, not ignored.
A stalled stream and a corrupt file are covered by `tests/test_health.py` and the
end-to-end tests.

## Stress suite

```bash
python -m sightline stress --lock results/heldout_manifest.json --config results/frozen_config.yaml
```

This needs the locked held-out manifest and the frozen config, so it cannot run yet.

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
docs/        PLAN, DATA_CARD, MODEL_CARD, EVALUATION, RECORDING_GUIDE, PROCESS_LOG, RETROSPECTIVE
sightline/   package
configs/     corridors, manifests (+ own-clip template), default parameters
results/     saved measurements (see results/README.md); later the frozen config and locked held-out manifest
tests/
```

## Status

| Item | State |
|---|---|
| `docs/PLAN.md` and amendments A1-A4 | done |
| Pipeline code and tests (synthetic + unit) | done, passing |
| Real detectors on MOT17 **val** (7 configurations) | done, val only: `results/detector_selection/`, `results/interval/`, `results/runtime/` |
| MOT17 downloaded, ingested and split | done (`docs/DATA_CARD.md`) |
| Own corridor clips | not recorded yet (`docs/RECORDING_GUIDE.md`) |
| Detector and detection interval selected | no: chosen on own val clips by PLAN.md A4 |
| Held-out manifest locked, tuning, held-out evaluation | not done |
| Demo video, retrospective, `v1.0.0` | not started |

## Licences

This code is MIT. Third-party parts:
- torchvision (BSD-3-Clause)
- YOLOX ONNX exports (Apache-2.0)
- ONNX Runtime (MIT)
- detector weights trained on COCO (annotations CC BY 4.0, images under the Flickr terms). Files and hashes are in `docs/MODEL_CARD.md`.
- MOT17 (commonly cited as CC BY-NC-SA 3.0, not re-verified; see `docs/DATA_CARD.md`)

Do not commit clips or weights.
