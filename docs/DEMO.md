# Demo script (4-6 minutes)

Each step answers one item of the brief's acceptance demo. Every number shown comes from a
file under `results/`. Nothing is changed between steps: no threshold, label or clip. Run from
the repo root with the venv active. IndoorCrowd frames are face-blurred by the dataset authors,
but **record the screen only on your own machine and do not upload a video that shows MOT17 or
IndoorCrowd frames** (DATA_CARD: no frames published). For a shareable video, show the terminal,
the JSON and the plots, and keep overlay windows off the recording.

## 0. Setup (before recording)
```bash
python -m sightline verify-lock results/heldout_manifest.json
python -m pytest -q
```

## 1. The manifest, and proof the test clips were not used for tuning (30 s)
```bash
python -m sightline split-check --val configs/manifests/val.json --test configs/manifests/test.json
```
Show `results/heldout_manifest.json`: `status: final`, `locked_at_utc`, and a hash per clip.
Then `results/frozen_config.yaml` -> `provenance`:
- `tuned_on_split: val`
- `val_clip_ids` (no IndoorCrowd, no MOT17-09 or 10)
- `locked_test_manifest_sha256`, which equals the lock's `manifest_sha256`

`git log --format='%h %ad %s' --date=iso -- results/heldout_manifest.json results/frozen_config.yaml`
shows the lock committed before the frozen config.

## 2. One ordinary held-out clip (45 s)
```bash
python -m sightline replay data/indoorcrowd/acs_eg/clip_00001_00061 --corridor configs/corridors/IC-acs_eg.yaml --config results/frozen_config.yaml --out outputs/demo/acs_eg --overlay
```
Show the overlay (locally): detections, stable ids, and the corridor events in `events.jsonl`. Compare
them with the clip's row in `results/per_clip/bytetrack/IC-acs_eg-clip_00001_00061.json`.

## 3. One difficult held-out clip: crowding and occlusion (45 s)
```bash
python -m sightline replay data/indoorcrowd/acs_ec/clip_00001_00082 --corridor configs/corridors/IC-acs_ec.yaml --config results/frozen_config.yaml --out outputs/demo/acs_ec --overlay
```
Point at where `health.jsonl` changes state and where ids change (`tracks.jsonl`).

## 4. Baseline against the chosen tracker on the same clip (30 s)
From `results/final_metrics.json`: `tracking.iou.ALL` against `tracking.bytetrack.ALL` (IDF1, ID
switches), and the per-clip rows in `results/per_clip/{iou,bytetrack}/<clip>.json`. The
ablation in `results/ablation/ablation_heldout.json` shows which part of the tracker makes the
difference.

## 5. Three failures, and why (60 s)
`results/error_analysis/heldout/INDEX.md`: one false detection, one missed person, and one ID
switch or duplicate track, each with the "why" written after looking at the frame.

## 6. One corridor event end to end (30 s)
```bash
python -m sightline timeline IC-acs_eg-clip_00001_00061             # lists the clip's events and latencies
python -m sightline timeline IC-acs_eg-clip_00001_00061 --event <N>  # one event, approach to close or loss
```
The output shows when the track was first reported, the GT first-inside frame, the open frame and its
latency in frames and seconds, the evidence while active, and how the event ended (closed, or UNRESOLVED
when the track was lost). It is built from `results/per_clip/` and the saved tracks only.

## 7. Break the input (30 s)
```bash
python -m sightline replay does_not_exist.mp4 --corridor configs/corridors/synthetic.yaml --config results/frozen_config.yaml
python -m sightline replay data/indoorcrowd/acs_eg/clip_00001_00061 --corridor configs/corridors/IC-acs_eg.yaml --config configs/broken.yaml
```
Both end UNAVAILABLE with exit code 2, never "clear" (`configs/broken.yaml` has a typo'd key). The
stalled-stream and stale-detection cases are pinned by tests:
`python -m pytest -q tests/test_health.py tests/test_async_detect.py`.

## 8. Runtime on this machine (20 s)
`results/runtime_report.json`: throughput, p50/p95 latency, inference time, dropped and skipped
frames, the machine, and the commit it ran on.

## 9. The tests that pin the behaviour (20 s)
```bash
python -m pytest -q tests/test_events.py tests/test_geometry.py tests/test_tracking.py tests/test_health.py tests/test_e2e.py
```
