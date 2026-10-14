# results/

Every number in the docs comes from a file here. **All of it is MOT17 validation or synthetic-clip data.** There are no held-out results and no own-clip results. Nothing here is tuned, locked or frozen. Each file's provenance (command, commit, settings) is in the commit message that added it; `git log -- <path>` shows it.

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
