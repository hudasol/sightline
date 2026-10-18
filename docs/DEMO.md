# Demo script (4 to 6 minutes)

Each step answers one item of the brief's acceptance demo. Every number shown comes from a file under
`results/`. Nothing changes between steps: no threshold, label or clip. Run from the repo root with the venv active.

There are two cuts.
- **The public cut** shows no dataset frame and may be shared.
- **The private cut** shows held out frames. It is for the mentor only, and only if the mentor confirms.

Both follow the map below.

## Acceptance demo map

**How the clips were chosen.** The clips were picked by a rule, not by how good they look. The rule is the first
clean clip and the first occluded clip in the locked manifest order (`results/heldout_manifest.json`):
- **Clean clip:** `MOT17-09-FRCNN`, the first entry tagged `clear`.
- **Hard clip:** `IC-acs_ec-clip_00001_00051`, the first entry tagged `occlusion_crowding`.

The failures are the first row of each type in `results/error_analysis/heldout/INDEX.md`. That file lists them in
the order the `errors` tool selected them, worst first.

| Brief item | Held out clip | Command (public cut) | Private cut adds |
|---|---|---|---|
| A clean held out clip | `MOT17-09-FRCNN` | `python -m sightline replay data/mot17/MOT17-09-FRCNN --corridor configs/corridors/MOT17-09.yaml --config results/frozen_config.yaml --tracker bytetrack --out outputs/demo/clean` | `--overlay`, then play `outputs/demo/clean/overlay.mp4` |
| A hard held out clip | `IC-acs_ec-clip_00001_00051` | `python -m sightline replay data/indoorcrowd/acs_ec/clip_00001_00051 --corridor configs/corridors/IC-acs_ec.yaml --config results/frozen_config.yaml --tracker bytetrack --out outputs/demo/hard` | `--overlay`, then play `outputs/demo/hard/overlay.mp4` |
| Baseline against chosen tracker, same clip | `MOT17-09-FRCNN` | `python -m sightline replay data/mot17/MOT17-09-FRCNN --corridor configs/corridors/MOT17-09.yaml --config results/frozen_config.yaml --tracker iou --out outputs/demo/clean_iou`, then `python -m sightline compare-trackers MOT17-09-FRCNN` | `--overlay` on both replays |
| A false detection | `MOT17-09-FRCNN`, frame 16 (INDEX row 1) | Read row 1 of `results/error_analysis/heldout/INDEX.md` | Open `results/error_analysis/heldout/01_false_detection_MOT17-09-FRCNN_f16.png` |
| A missed person | `MOT17-09-FRCNN`, frame 220 (INDEX row 3) | Read row 3 | Open `03_missed_person_MOT17-09-FRCNN_f220.png` |
| An ID switch or duplicate track | `IC-acs_ec-clip_00001_00082`, frame 19 (row 5); duplicate: `MOT17-09-FRCNN`, frame 223 (row 11) | Read rows 5 and 11 | Open `05_id_switch_IC-acs_ec-clip_00001_00082_f19.png` and `11_duplicate_track_MOT17-09-FRCNN_f223.png` |
| One corridor event, approach to close | `IC-ie_central-clip_00001_00061`, event 3 | `python -m sightline timeline IC-ie_central-clip_00001_00061 --event 3` | Nothing (text) |
| A deliberately broken input | none | `python -m sightline replay does_not_exist.mp4 --corridor configs/corridors/synthetic.yaml --config results/frozen_config.yaml` and `python -m sightline replay data/indoorcrowd/acs_eg/clip_00001_00061 --corridor configs/corridors/IC-acs_eg.yaml --config configs/broken.yaml` (both exit 2, UNAVAILABLE) | Nothing |
| The runtime report | all held out clips | `python -m sightline runtime-summary results/runtime/heldout_offline --out outputs/demo/runtime_summary.json`, then open `results/runtime_report.json` and `results/runtime/heldout_offline/STALL.md` | Nothing |
| The tests | none | `python -m pytest -q` | Nothing |

**Notes on the map.**
- **The event walkthrough clip.** It is not chosen by the clean and hard rule. It is the report page's walkthrough:
  a matched, closed crossing whose track is reported before it enters, so the approach is visible.
- **The failure images already exist.** They are on this machine only and git ignored. They were rendered by
  `python -m sightline errors --lock results/heldout_manifest.json --results results --config results/frozen_config.yaml`,
  which is not rerun for the demo.
- **The replays run the live detector.**
  - On MOT17-09 the live replay reports DEGRADED for part of the clip, because the processing rate on this laptop
    falls below the declared 10 fps (`low_fps` in `health.jsonl`). That is health working as designed.
  - The held out metrics come from the saved evaluation (`results/`), not from these replays.
- **Commands checked on 2026-10-18, on this machine.**
  - Every command in the public cut column ran: the replays and `runtime-summary` exit 0, both broken-input commands
    exit 2 with UNAVAILABLE, the tests pass, and `compare-trackers` and `timeline` print from saved results.
  - The check wrote no image or video, and its outputs were deleted afterwards.
  - The `--overlay` variants were **not** run, because they write a video of held out frames.
  - `timeline` and the replays need the held out data in `data/`.

## Public cut (no dataset frames)

### What may appear on screen
- **In the public cut, never a MOT17 or IndoorCrowd frame.** No overlay window, no error-analysis image, no video of them. Their
  licences and the project's privacy rule do not allow it (`docs/DATA_CARD.md`,
  `docs/licences/INDOORCROWD_LICENSE.txt`).
- **Held out clips appear only as text:** the terminal output of a run, the saved JSON, the plots, and the
  report page (`results/report/index.html`, which has no frames). Run held out replays **without** `--overlay`.
- **Moving pictures come only from the approved Pexels clip** (`docs/licences/DEMO_FOOTAGE.md`).
  - **Pending: no clip is approved yet.** Until one is, skip step 3b.
  - The clip has no ground truth, so it shows the system working, never an accuracy number.
  - Credit its creator in the video description.

### Shot list (about 5 min 30 s)

Only four things go on screen:
- the report page (`results/report/index.html`);
- the plots (`results/plots/`);
- the terminal;
- the approved Pexels clip, once Huda approves one.

Read every number off the screen; none is typed here. Rows marked **MISSED BAR** show a bar the system did not
meet: say so plainly and do not move on quickly.

| # | Time | Open or run | Say | Length |
|---|---|---|---|---|
| 1 | 0:00 | `README.md`, the "Start here" block | What Sightline is: advisory corridor perception, no robot control, anonymous ids. It was evaluated once, on sourced data (MOT17, IndoorCrowd). | 20 s |
| 2 | 0:20 | Terminal: `python -m sightline split-check --val configs/manifests/val.json --test configs/manifests/test.json`, then the `git log` line in step 1 below | The held out clips were locked before any tuning, and tuning used val only. | 30 s |
| 3 | 0:50 | Report page: the bar table | Go down the rows. Read each PASS. | 25 s |
| 4 | 1:15 | Report page: the same table, the FAIL rows. **MISSED BAR** | IDF1 fails the 0.65 bar, event precision and recall fail 0.85, and the recorded throughput check fails. Read the measured values off the page. Say that the throughput run had a stall while the laptop was in standby, and that the later timing checks are separate, not a replacement. | 35 s |
| 5 | 1:50 | Report page: the ablation table | The tracker beats its baseline, and the motion model is what carries that gain. That is still short of the IDF1 bar. | 25 s |
| 6 | 2:15 | Terminal: `python -m sightline timeline IC-ie_central-clip_00001_00061 --event 3` | One event from approach to close: when the track appears, when the feet are first inside, when the event opens and its latency, and how it closes. | 30 s |
| 7 | 2:45 | Approved clip: the step 3b replay with `--overlay`. Until a clip is approved, run step 3a instead (text only) | Boxes, anonymous ids, the corridor, an event and the health state. This clip has no ground truth, so no accuracy number. | 30 s |
| 8 | 3:15 | `results/error_analysis/heldout/INDEX.md` (text only, no images) | Three failures and why: a person cut off by the image edge, an ID switch in a tight group, and false events from tracks breaking up at 5 fps with the detector on every third frame. **MISSED BAR** context: the clip these false events come from holds most of the false events in the held out set. | 40 s |
| 9 | 3:55 | `README.md`, "Known limitations": the ACS-EC item | The ACS-EC corridor was drawn on one camera view and lands on the cafe counter on two recordings. It was not redrawn after the evaluation. **MISSED BAR** context: those recordings hold most of the missed events. | 20 s |
| 10 | 4:15 | Terminal: the two commands in step 8 below | Bad input ends UNAVAILABLE with exit code 2, never CLEAR. | 30 s |
| 11 | 4:45 | `results/plots/stress_curves.png`, then `results/plots/reliability_heldout.png` | Tracking gets worse under blur, compression, low light and shake. Confidence did not predict recall on held out data. | 25 s |
| 12 | 5:10 | Terminal: `python -m pytest -q` | The tests that pin events, geometry, expiry and failure behaviour. | 20 s |

Without an approved clip, row 7 shows text only and the timing stays the same.

The steps below give the full commands for each row.

### 0. Setup (before recording)
```bash
python -m sightline verify-lock results/heldout_manifest.json
python -m pytest -q
```

### 1. The manifest, and proof the test clips were not used for tuning (30 s)
```bash
python -m sightline split-check --val configs/manifests/val.json --test configs/manifests/test.json
```
Show `results/heldout_manifest.json`: `status: final`, `locked_at_utc`, and a hash per clip. Then
`results/frozen_config.yaml` -> `provenance`:
- `tuned_on_split: val`
- `val_clip_ids` (no IndoorCrowd, no MOT17-09 or 10)
- `locked_test_manifest_sha256`, which equals the lock's `manifest_sha256`

```bash
git log --format="%h %ad %s" --date=iso -- results/heldout_manifest.json results/frozen_config.yaml
```
This shows the lock was committed before the frozen config.

### 2. The held out results at a glance (30 s)
Open `results/report/index.html` by double-clicking it. Show:
- the bar table, pass or fail per bar;
- the stress plot and the ablation;
- the event walkthrough.

### 3a. The clean held out clip, as text (45 s)
```bash
python -m sightline replay data/mot17/MOT17-09-FRCNN --corridor configs/corridors/MOT17-09.yaml --config results/frozen_config.yaml --tracker bytetrack --out outputs/demo/clean
```
No `--overlay`. Show the terminal summary, then `outputs/demo/clean/events.jsonl` and the first lines of
`tracks.jsonl` (stable ids). Compare them with the clip's saved row,
`results/per_clip/bytetrack/MOT17-09-FRCNN.json`. The clip was chosen by the map's rule.

### 3b. The system on footage that may be shown (45 s, only once a clip is approved)
```bash
python -m sightline replay data/demo/<approved clip>.mp4 --corridor configs/corridors/demo.yaml --config results/frozen_config.yaml --out outputs/demo/pexels --overlay
```
Show the overlay video: boxes, anonymous track ids, the corridor, events and the health state.
- The corridor is drawn once with `draw-corridor` after approval.
- There is no ground truth for this clip, so say so and show no accuracy number.

### 4. The hard held out clip: crowding and occlusion, as text (45 s)
```bash
python -m sightline replay data/indoorcrowd/acs_ec/clip_00001_00051 --corridor configs/corridors/IC-acs_ec.yaml --config results/frozen_config.yaml --tracker bytetrack --out outputs/demo/hard
```
No `--overlay`. Show where `health.jsonl` changes state and where ids change in `tracks.jsonl`. The clip was
chosen by the map's rule. It is the ACS-EC view the corridor was drawn on; the other two ACS-EC recordings are
not (README, Known limitations).

### 5. Baseline against the chosen tracker (30 s)
```bash
python -m sightline compare-trackers MOT17-09-FRCNN
```
This prints the clip's ID switches for both trackers, and the pooled IDF1 (per clip IDF1 is not saved). In
`results/final_metrics.json`, compare `tracking.iou.ALL` with `tracking.bytetrack.ALL` (IDF1, ID switches).
The per clip rows are in `results/per_clip/{iou,bytetrack}/<clip>.json`. The ablation
(`results/ablation/ablation_heldout.json`) shows that the motion model carries the gain.

### 6. Three failures, and why (60 s)
Read from `results/error_analysis/heldout/INDEX.md`: one false detection, one missed person, and one ID switch or
duplicate track, each with its "why". Read the text only; do not show the images.

### 7. One corridor event end to end (30 s)
```bash
python -m sightline timeline IC-acs_eg-clip_00001_00061             # lists the clip's events and latencies
python -m sightline timeline IC-ie_central-clip_00001_00061 --event 3  # one event, approach to close (the report page's walkthrough)
```
This prints, from the saved results only (`results/per_clip/` and the saved tracks):
- when the track was first reported;
- the GT first inside frame;
- the open frame, and its latency in frames and seconds;
- the evidence while the event was active;
- how the event ended: closed, or UNRESOLVED when the track was lost.

### 8. Break the input (30 s)
```bash
python -m sightline replay does_not_exist.mp4 --corridor configs/corridors/synthetic.yaml --config results/frozen_config.yaml
python -m sightline replay data/indoorcrowd/acs_eg/clip_00001_00061 --corridor configs/corridors/IC-acs_eg.yaml --config configs/broken.yaml
```
Both end UNAVAILABLE with exit code 2, never "clear" (`configs/broken.yaml` has a misspelt key). The stalled-stream
and stale-detection cases are pinned by tests:
```bash
python -m pytest -q tests/test_health.py tests/test_async_detect.py
```

### 9. Runtime on this machine (20 s)
Show:
- `results/runtime_report.json`: throughput, p50 and p95 latency, inference time, dropped and skipped frames,
  the machine, and the commit it ran on;
- `results/runtime/heldout_offline/summary.json`: the three repeated runs, and the 549 s stall in run 1;
- `results/runtime/heldout_realtime.json`: the same clips paced in real time.

### 10. The tests that pin the behaviour (20 s)
```bash
python -m pytest -q tests/test_events.py tests/test_geometry.py tests/test_tracking.py tests/test_health.py tests/test_e2e.py
```

## Private cut (held out frames shown, shared only with the mentor, never uploaded publicly; use only if the mentor confirms)

Use this cut only after the mentor confirms, in writing, that they want to see held out frames.
- **Why it is restricted.**
  - MOT17 frames show unblurred faces.
  - IndoorCrowd's licence is for non-commercial research only and forbids redistribution
    (`docs/licences/INDOORCROWD_LICENSE.txt`).
  - So this recording is never uploaded, never published and never linked publicly.
  - Keep it on this machine, share it with the mentor only, and delete it when the review is done.
- **What changes from the public cut:**
  - Add `--overlay` to the three replays in the map. The overlay video lands in `outputs/demo/<name>/overlay.mp4`,
    which is git ignored. Play it, and delete it after recording.
  - For the false detection, missed person, ID switch and duplicate rows, open the existing strip images named in
    the map instead of reading the text.
  - Everything else is the same as the public cut.
- **What stays out even here:** no frame is committed, put in a document or shown in the public cut.
