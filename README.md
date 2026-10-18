# Sightline

Advisory video perception for an indoor inspection robot. It detects people,
tracks them with anonymous ids through motion and short occlusion, turns tracks into
**one event per corridor entry/crossing**, and reports DEGRADED or UNAVAILABLE when the
input is invalid, stalled or of poor quality. It reports CLEAR only when health is OK and
no track is inside the corridor. That is not a guarantee that nobody is inside: on the
held out set, 78 of 890 occupied frames were reported CLEAR (see Known limitations).

> **Advisory only.** No robot connection, no motion control, no face or
> biometric processing. Tracks are anonymous integers.

`docs/PLAN.md` was committed first and is the contract; its dated amendments (A1-A8)
record every later change with what was known when it was made. The held out set (14
clips: MOT17-09, MOT17-10 and 12 IndoorCrowd recordings) was locked before tuning, the
configuration was tuned on val only, and the held out evaluation ran once. Results:
`docs/EVALUATION.md` (interpretation) and `docs/RESULTS.md` (tables generated from
`results/`). Every number comes from a file under `results/`.

**Data: sourced only (PLAN.md A8).** The evaluation uses public sourced data only, MOT17 and IndoorCrowd. No own footage was recorded, and none was required. The own footage tools below are optional.

## What is in the box

| Stage | Module | What it does |
|---|---|---|
| Ingest | `sightline/ingest.py` | One command records fps, frame count, per frame timestamps and source metadata |
| Detect | `sightline/detector.py` | Pretrained COCO person detectors (torchvision, or YOLOX on ONNX Runtime). Shipped: YOLOX-s @640 on every 3rd frame, threshold frozen on val |
| Track (baseline) | `sightline/tracking/iou_tracker.py` | Greedy IoU matching, no motion model |
| Track (chosen) | `sightline/tracking/bytetrack.py` | ByteTrack-style two stage association, Kalman + Hungarian, geometry only |
| Track state | `sightline/tracking/state.py` | id, age, last seen, confidence, trajectory, explicit lost/expired policy |
| Events | `sightline/events.py` | Debounced open / active / close, re-entry merge, `UNRESOLVED` on lost track |
| Health | `sightline/health.py` | `OK / DEGRADED / UNAVAILABLE` from corrupt frames, stalls, low fps, low quality, stale detections |
| Evaluation | `sightline/evaluation/` | Detection P/R/F1, IDF1 and ID switches, event P/R, latency, per condition tables |
| Split safety | `sightline/manifest.py` | Clip manifests with hashes and a leakage check |
| Runtime | `sightline/runtime.py` | fps, p50/p95 latency, inference time, dropped/skipped frames |
| Analysis | `sightline/{ablation,calibration,errors,timeline,report}.py` | Tracker ablation, confidence calibration, worst first failures, one event end to end, generated result tables |

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

Model weights download on first use (torchvision, COCO). They are git ignored.

### OpenCV with a GUI (for `draw-corridor` and `label-events`)

The default install uses `opencv-python-headless`, which has no window support.
The interactive tools need `opencv-python`, which the `gui` extra installs. Both
packages ship the same `cv2` folder, so they must never be installed together.
- A plain `pip install -e ".[...]"` (re)installs the headless package as a core
  dependency, and it overwrites the GUI build's files (`GUI: NONE`), even though
  `opencv-python` still shows in `pip list`.
- Uninstalling only one of the two deletes files the other needs.

So the swap is always the **last** step, in this order (Windows PowerShell, venv
active):

```powershell
pip install -e ".[detector,onnx,dev,gui]"
$v = (pip show opencv-python-headless | Select-String '^Version:').ToString().Split(':')[1].Trim()
pip uninstall -y opencv-python-headless opencv-python
pip install "opencv-python==$v"
python -c "import cv2; print([l.strip() for l in cv2.getBuildInformation().splitlines() if l.strip().startswith('GUI')])"
```

On macOS / Linux, the same steps:

```bash
pip install -e ".[detector,onnx,dev,gui]"
v=$(pip show opencv-python-headless | sed -n 's/^Version: //p')
pip uninstall -y opencv-python-headless opencv-python
pip install "opencv-python==$v"
python -c "import cv2; print([l.strip() for l in cv2.getBuildInformation().splitlines() if l.strip().startswith('GUI')])"
```

- **Check:** the last line must print a GUI backend (`WIN32UI` on Windows), not `NONE`.
- **Expected warning:** `pip check` then reports that sightline requires
  `opencv-python-headless`. That is expected here and harmless: `opencv-python`
  provides the same `cv2`.
- **Reinstalling the project code later:** use `pip install -e . --no-deps`, which
  leaves OpenCV alone.
- **Reinstalling with dependencies or extras:** this brings the headless package
  back, so repeat the last four lines.
- **Servers and CI:** keep the headless default and skip this section.

## Data preparation

Raw video and frames are **not** in git (licences differ from the code's).
Put them under `data/` and describe them in a manifest. See
`docs/DATA_CARD.md` for sources, licences and the annotation method.

```text
data/
  mot17/MOT17-02-FRCNN/{img1,gt/gt.txt,seqinfo.ini}   # one detector copy per scene only
  mot17_5fps/MOT17-02-FRCNN-5fps/...                  # made by `decimate` (val only)
  indoorcrowd/<site>/<recording>/...                  # made by `import-indoorcrowd`
  own/<clip_id>.mp4
annotations/
  <clip_id>/gt.txt          # MOTChallenge format: frame,id,x,y,w,h,conf,class,vis
  <clip_id>/events.json     # ground-truth corridor events
configs/
  corridors/<camera_id>.yaml
  manifests/{val.json,test.json}
```

**MOT17.** Download `https://motchallenge.net/data/MOT17.zip` and extract only
`train/MOT17-{02,04,05,09,10,11,13}-FRCNN/{img1,gt,seqinfo.ini}` into `data/mot17/` (DATA_CARD).

**IndoorCrowd** is gated. Accept its terms on Hugging Face under your own account and read
`docs/licences/INDOORCROWD_LICENSE.txt` first. Download the `tracking` config only:

```bash
pip install -e ".[indoorcrowd]"
hf auth login
hf download sebnae/IndoorCrowd --repo-type dataset --revision 87049647f27c4419dcba051474dcfc0fcecad8e0 --include "tracking/*" --local-dir data/indoorcrowd/raw
python -m sightline import-indoorcrowd --revision 87049647f27c4419dcba051474dcfc0fcecad8e0
```

**Frame-rate-matched val copies** (PLAN.md A6.3) run at about 5 fps, like the IndoorCrowd clips.
Make them after the val detection caches exist, so the caches are renumbered rather than recomputed:

```bash
python -m sightline decimate data/mot17/MOT17-02-FRCNN data/mot17/MOT17-04-FRCNN data/mot17/MOT17-05-FRCNN data/mot17/MOT17-11-FRCNN data/mot17/MOT17-13-FRCNN --summary results/ingest/val_5fps_decimation.json
```

Check the split is leakage safe and lock the held out manifest **before** the
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

## Own footage: from phone to labelled clip (optional)

Optional (PLAN.md A8): the reported evaluation uses sourced data only and no own footage was recorded. This section is for anyone who later wants to evaluate on their own corridor.

Phones record variable frame rate (VFR), often HEVC, sometimes with rotation
metadata, and `check-clip` flags all three. Convert every recording once with
ffmpeg, which is free (ffmpeg.org; on Windows `winget install Gyan.FFmpeg`). This
gives a constant 30 fps, 1920x1080, H.264 file and **keeps the original
untouched**.

Keep originals in `data/own/originals/` (git ignored). `-n` refuses to overwrite
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
- `draw-corridor` and `label-events` open an OpenCV window. The default headless
  OpenCV has none: on the machine where you label, follow "OpenCV with a GUI" under
  Setup, in that order. Uninstalling only the headless package is not enough.
- See `docs/RECORDING_GUIDE.md` for what to film, and
  `configs/manifests/own_clip.template.json` for the manifest entry.

## Try it with no data and no model

```bash
python -m sightline demo          # generated clip, both trackers, writes outputs/demo
python -m pytest                  # unit + end-to-end tests
```

The demo is synthetic. It checks the code, not real world performance.

## Splits, tuning, evaluation (order matters)

This is the exact sequence that produced `results/`. Each step reads only what the steps before it wrote.

```bash
python -m sightline split-check --val configs/manifests/val.json --test configs/manifests/test.json --out results/split/split_check.json
python -m sightline detect-eval --manifest configs/manifests/val.json --detector yolox_s --input-size 640 640 --out results/detector_selection/detect_eval_val_yolox_s_640x640.json   # val only, one file per candidate
python -m sightline runtime-summary results/runtime/a4_selection        # median of 3 benchmark runs per (detector, k)
python -m sightline select-detector                                     # PLAN.md A4 -> results/detector_selection/a4_choice.json
python -m sightline lock-manifest configs/manifests/test.json           # BEFORE tuning; "draft" until coverage is met
python -m sightline tune --config configs/default.yaml                  # val only -> results/frozen_config.yaml
python -m sightline benchmark --manifest results/heldout_manifest.json  # fps / latency on THIS machine -> results/runtime_report.json
python -m sightline benchmark --manifest results/heldout_manifest.json --realtime --out results/runtime/heldout_realtime.json
python -m sightline evaluate                                            # locked held-out set, once
python -m sightline ablate                                              # tracker parts on/off, cached detections, analysis only
python -m sightline calibration                                         # does confidence predict correctness? analysis only
python -m sightline errors                                              # failures worst-first; a human writes the "why"
python -m sightline stress                                              # blur/compression/low light/shake, frozen, no retuning
python -m sightline plots                                               # all plots from saved results
python -m sightline report                                              # docs/RESULTS.md, tables from saved results only
python -m sightline report-html                                         # results/report/index.html: one static page, opens by double-click
python -m sightline timeline <clip> --event <N>                         # one corridor event from approach to close, from saved results
```

The A4 throughput measurement was three synchronous offline runs per k. One of them looks like this:
`python -m sightline benchmark data/mot17/MOT17-02-FRCNN --corridor configs/corridors/MOT17-02.yaml --config configs/default.yaml --detector yolox_s --input-size 640 640 --detect-every 3 --max-frames 150 --out results/runtime/a4_selection/yolox_s_640x640_k3_run1/MOT17-02.json`.
It ran for each val scene, k = 1, 2 and 3, and runs 1-3 (`results/runtime/a4_selection/README.md`).

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
never wait for it (`pipeline.async_detection`). It is off by default and its effect on
accuracy is not measured. In both modes health checks how old the detection behind the
corridor status is (`health.max_detection_age_s`, 0.5 s). A detector that falls behind turns
the system DEGRADED instead of leaving it CLEAR on stale detections. `runtime-summary <folder>` gives
the median and range over repeated benchmark runs (PLAN.md A4).

`tune` refuses to run unless the held out lock is **final** (coverage met) and
unchanged, and writes `results/frozen_config.yaml` with provenance. `evaluate` refuses if the lock or
config provenance does not match. A second run needs `--allow-rerun` and is logged in
`results/final_run_log.jsonl`. `evaluate` writes:
- `results/final_metrics.json`: detection, IDF1 and ID switches for the baseline and the chosen
  tracker, event precision, recall and latency, false events per 5 minutes of negative footage,
  per condition tables and the bar check;
- `results/per_clip/`.

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
end to end tests.

## Stress suite

```bash
python -m sightline stress --lock results/heldout_manifest.json --config results/frozen_config.yaml
```

Degrades held out clips (blur, compression, low light, shake) at several
levels using the **frozen** config and plots the degradation curve. It does not
retune anything.

## Tests

```bash
python -m pytest -q
```

The tests cover:
- event state, geometry, and tracker expiry and re-entry;
- invalid configuration;
- degraded states: corrupt frames, stalls, low fps and stale detections;
- split leakage, one key at a time, and the lock and `evaluate` refusals;
- the importers;
- end to end replays on a generated synthetic clip (no downloads needed).

CI runs the same suite on Python 3.12 and 3.13 for every push (`.github/workflows/tests.yml`; runs at
https://github.com/hudasol/sightline/actions/workflows/tests.yml).

Last local run (2026-10-18, the demo laptop, Python 3.13): `python -m pytest -q` -> **194 passed** in 58 s.
The async tests no longer depend on wall clock speed: the detector is blocked on a gate the test controls
(`tests/test_async_detect.py`), so they pass on a busy or battery-powered machine too.

## Repository layout

```text
docs/        PLAN, DATA_CARD, MODEL_CARD, EVALUATION, RESULTS (generated), DEMO, RECORDING_GUIDE,
             SOLO_RECORDING_GUIDE, PROCESS_LOG, RETROSPECTIVE, licences/
sightline/   package
configs/     corridors, manifests (+ own-clip template), default parameters
results/     saved measurements (see results/README.md); later the frozen config and locked held-out manifest
tests/
```

## Known limitations

Each item is measured or recorded in the named file; `docs/EVALUATION.md` has the detail.

- **ACS-EC corridor drawn on one camera view (found after the evaluation, PLAN.md A7.1).** IndoorCrowd's ACS-EC site has two camera views. The ACS-EC corridor was drawn on one view (`clip_00001_00051`). On the other two recordings (`clip_00001_00082`, `clip_00026_00091`) it lands on the cafe counter. These two recordings account for **10 of the 17 missed events** (`results/per_clip/bytetrack/`) and **56 of the 78 false CLEAR frames** (`results/safety/false_clear_by_clip.json`). The corridor was **not** redrawn for the reported result, and the held out evaluation was not rerun. A post hoc sensitivity table without these two recordings is in `docs/RESULTS.md`; it is not a held out result.
- **Bars missed.** IDF1 0.512 (bar 0.65) and event precision 0.840 and recall 0.787 (bar 0.85)
  (`results/bar_check.json`). The recorded throughput check failed on a run with a 549 s stall, during which the laptop was in Modern Standby
  (PLAN.md A7.2, A9; `results/runtime/heldout_offline/STALL.md`).
- **Track fragmentation at 5 fps with k = 3.** On IndoorCrowd's ground level ACS-EG view, people close to the camera
  fragment into several tracks; ACS-EG has 10 of the 12 false events (`results/per_clip/bytetrack/`).
- **CLEAR means "no track inside", not "nobody inside".** 78 of 890 occupied frames were reported CLEAR with health OK
  (`results/final_metrics.json`, `safety`).
- **Short negative set.** 218 s of negative footage (3.6 min) with 0 false events: it cannot show a rate below 1 per
  5 minutes.
- **Sourced data only, not robot footage** (PLAN.md A8). Three of IndoorCrowd's four views look down from above.

## Status (2026-10-17)

| Item | State |
|---|---|
| `docs/PLAN.md`, amendments A1-A7 | done; A7 records two errata found after the evaluation |
| Data: MOT17 (7 scenes) and IndoorCrowd (12 recordings, 4 sites) | done (`docs/DATA_CARD.md`); own robot height footage not recorded (sourced data only, none required: PLAN.md A8) |
| Detector and interval (A4 on MOT17 val) | YOLOX-s @640, k = 3 (`results/detector_selection/a4_choice.json`) |
| Held out manifest | locked FINAL before tuning (`results/heldout_manifest.json`) |
| Tuning (val only) | done (`results/frozen_config.yaml`) |
| Held out evaluation | run once (`results/final_metrics.json`; tables in `docs/RESULTS.md`) |
| Bar | detection, clip count, conditions, negatives, latency and "beats the baseline" pass; IDF1 and event precision/recall fail; throughput recorded as fail on a stalled run, median of 3 runs 14.79 fps (`docs/EVALUATION.md`) |
| Ablation, calibration, error analysis, stress | done (`results/`) |
| Tests | all pass locally and in CI (`.github/workflows/tests.yml`) |
| Demo video | script in `docs/DEMO.md`; recording is Huda's |
| Retrospective | drafted from the record; Huda to finish |

## Licences

This code is MIT. Third party parts:
- torchvision (BSD-3-Clause)
- YOLOX ONNX exports (Apache-2.0)
- ONNX Runtime (MIT)
- detector weights trained on COCO (annotations CC BY 4.0, images under the Flickr terms). Files and hashes are in `docs/MODEL_CARD.md`.
- MOT17 (commonly cited as CC BY-NC-SA 3.0, not re-verified; see `docs/DATA_CARD.md`)
- IndoorCrowd (`fair-noncommercial-research-license`): non-commercial computer vision research only,
  no surveillance, no re-identification (`docs/licences/INDOORCROWD_LICENSE.txt`). It is not
  redistributed: no frame or label of it is in this repository.

Do not commit clips or weights.
