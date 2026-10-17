# results/

Every number in the docs comes from a file here. Each file's provenance (command, commit, settings) is in the commit message that added it, and newer runtime and evaluation files also record their commit inside; `git log -- <path>` shows it. The held-out evaluation ran once on 2026-10-17. `docs/RESULTS.md` is generated from these files by `sightline report`.

## The held-out evaluation (2026-10-17)

| Path | What it is |
|---|---|
| `heldout_manifest.json` | The locked held-out manifest (FINAL): a hash per source, ground truth, corridor and the eval rules |
| `frozen_config.yaml` | The configuration tuned on val only, with provenance (val clips, lock hash, commit) |
| `tuning/` | The val grids `tune` searched: detector threshold, both trackers, events, image quality |
| `final_metrics.json`, `bar_check.json`, `final_run_log.jsonl` | The single held-out evaluation, its mechanical bar check, and the run log (one line = one run) |
| `per_clip/{bytetrack,iou}/` | Per held-out entry: detections, ID switches, predicted and GT events, matches, health |
| `runtime_report.json` | Offline runtime on the held-out clips that `evaluate` read (run 1, with a 549 s stall, PLAN A7.2) |
| `runtime/heldout_offline/` | Runs 1-3 of the same measurement and their median |
| `runtime/heldout_realtime.json` | The same clips paced in real time (synchronous) |
| `runtime/a4_selection/` | The A4 throughput runs on val (3 per k) and their summary |
| `detector_selection/a4_choice.json` | The A4 choice built from them by `select-detector` |
| `ablation/`, `calibration/`, `safety/` | Tracker ablation, confidence calibration (held-out and val), per-clip false-CLEAR breakdown. Analysis only |
| `stress/` | The frozen system on degraded held-out clips (nothing retuned) |
| `error_analysis/heldout/` | The 12 held-out failures with their "why" |
| `plots/` | Every plot, from `sightline plots` |
| `ingest/indoorcrowd_import.json`, `ingest/val_5fps_decimation.json` | The IndoorCrowd import and the about-5 fps val copies |

## Before the lock (MOT17 val and synthetic data)

| Path | What it is |
|---|---|
| `ingest/` | `sightline ingest` records of the 7 MOT17 sequences: fps, frames, timestamps and hashes |
| `labels/` | Ground-truth event counts derived from MOT17 boxes with the draft corridors. Label statistics, not system output |
| `split/` | `split-check` result: leakage, and held-out coverage per metric family |
| `smoke/` | Real detector on the **synthetic** clip, to check that it runs. Not a performance figure |
| `runtime/pretune_default_config_*` | Val benchmarks of the default detector **before the speed changes** (resize, decode thread, thread count). "pretune" does not mean anything was tuned |
| `runtime/postchange_default_config_*` | The same benchmarks after those speed changes |
| `runtime/diag_input_resize_*` | 30-frame diagnostic: full-resolution input against input resized to the declared size |
| `runtime/torch_threads_*` | `sightline threads` sweeps (detector time against intra-op thread count) |
| `runtime/candidates/<detector>/` | End-to-end timing of each detector candidate on the first 150 frames of each val scene, detector on every frame |
| `runtime/control_1b39626/` | The MobileNet candidate re-timed in a later session, to show run-to-run variation |
| `runtime/interval/<detector>_k<k>/` | End-to-end fps against detection interval k (offline: frames are not paced in real time) |
| `runtime/async/{before_sync,after_sync,after_async}/` | Latency on the clip's own clock (`--realtime`), 3 runs per config, before and after detection moved to a worker thread. Each has a `summary.json` from `runtime-summary`. The JSON files do not record the commit; the commit messages give fe15d93 (before) and 9ff5dad (after) |
| `detector_selection/` | `detect-eval` on all val frames for 7 candidates, and `summary.md`/`summary.json` from `detector-summary`. The summary's verdict line applies the PLAN section 5 rule, which amendment A4 replaces for the final choice; it is **not** a selection |
| `interval/` | `interval-eval`: event recall, precision and latency against detection interval, on cached detections. `_frame_params` = settings counted in frames, `_time_params` = settings in seconds (A3) |
| `stride/` | Older `stride-eval` results (frame skipping), kept for history; superseded by `interval/` (A2) |
| `error_analysis/` | Indexes of failures selected by `sightline errors`. The frame images are not committed (they show faces); each README says how to recreate them |
