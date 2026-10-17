# Evaluation

The held-out set was locked, the configuration was tuned on val only, and the held-out evaluation ran
**once** on 2026-10-17 (`results/final_run_log.jsonl` has one line). This document interprets the
results. The tables are in `docs/RESULTS.md`, which `python -m sightline report` generates from the
files under `results/`. Every number here comes from one of those files, and each is named next to the
claim. The MOT17 **val** sections further down predate the lock and are kept as history.

## Verdict against the bar
Source: `results/bar_check.json`, `results/final_metrics.json`.

| Bar (brief) | Held-out result | Verdict |
|---|---|---|
| >= 12 held-out clips, >= 3 difficult conditions, a negative set | 14 clips (6 scenes); crowding, camera motion/blur, low light; 9 negative entries | **pass** |
| Detection P and R >= 0.80 at the frozen threshold (IndoorCrowd, in scope, A1/A6) | P 0.830, R 0.807 | **pass**, narrowly |
| IDF1 >= 0.65 | 0.512 (all boxes); 0.713 in scope | **fail** as the bar is defined |
| Tracker measurably better than the baseline | IDF1 0.512 vs 0.435; ID switches 235 vs 422 | **pass** |
| Event precision and recall >= 0.85 | P 0.840, R 0.787 (63 TP, 12 FP, 17 FN) | **fail** |
| <= 1 false event per 5 min of negative footage | 0 false events in 218 s | **pass**, but on 3.6 min only |
| Median event latency <= 5 frames | 0 frames (63 matched events) | **pass** |
| Degraded/unavailable on invalid video, stalls, low quality or performance | tested; on the held-out clips health was DEGRADED on 31 MOT17-10 frames and OK elsewhere | **pass** (by tests) |
| Throughput >= 10 fps at the declared resolution | recorded check: 3.14 fps (run 1, one 549 s stall); median of 3 offline runs 14.79 fps | recorded **fail**; see Runtime |
| Reproducible from frozen config; tests pass | `results/frozen_config.yaml` with provenance; all tests pass locally and in CI | **pass** |

**In short.** Detection meets the bar on the primary set. The tracker clearly beats its baseline, but
IDF1 over all boxes does not reach 0.65. Events miss both 0.85 bars. Most event errors come from two
sources, described below, and both are visible in the error analysis. Throughput meets 10 fps in the
median of three runs, but the run the bar check used has a stall, and the check is left as recorded.

## What was done, in order
1. **Held-out set fixed** before any output existed on it: MOT17-09 and 10 (2026-10-12) and the 12
   IndoorCrowd recordings, split by site (PLAN.md A6, 2026-10-17). Corridors and negative corridors
   were drawn from frames and ground-truth foot traces only.
2. **Detector and interval chosen on MOT17 val by A4:** YOLOX-s @640 with k = 3
   (`results/detector_selection/a4_choice.json`). k = 3 is the smallest interval whose median of 3
   synchronous offline runs reached 10 fps.
3. **Locked (FINAL)** at fabd2d3 (`results/heldout_manifest.json`, hash `865a3313`).
4. **Tuned on val only** at f07eb40: the 5 MOT17 val scenes plus their about-5 fps copies (A6.3) gave
   `results/frozen_config.yaml`.
5. **Runtime measured, then the single evaluation**, then analysis that changes nothing: ablation,
   calibration, error analysis, stress.

## Detection
Sources: `docs/RESULTS.md` "Detection"; `results/final_metrics.json`.
- **IndoorCrowd in scope (the judged set): P 0.830, R 0.807.** It passes, but only just: about 1 in 5
  tall-enough people is missed. Because IndoorCrowd has no visibility field, "in scope" there includes
  occluded people (A6.4). The crowded group (crowding, in scope) is the weakest: P 0.768, R 0.793.
- **MOT17 in scope: P 0.827, R 0.957.** Reported, not judged (A1).
- **All boxes: P 0.856, R 0.550.** Most misses are small or partly visible people, many of them MOT17
  people cut off by the image edge. MOT17 labels the full body even off-frame, so a correct box on the
  visible part scores IoU < 0.5. All four MOT17-09 detection failures in the error analysis are this
  case (`results/error_analysis/heldout/INDEX.md` #1-#4).

## Tracking
Sources: `docs/RESULTS.md` "Tracking" and "Ablation"; `results/ablation/ablation_heldout.json`.
- **ByteTrack-style: IDF1 0.512, 235 ID switches. IoU baseline: 0.435, 422** (all boxes, every frame).
  In scope: 0.713 against 0.619, and 166 against 338 switches. The bar (0.65) is checked on all boxes,
  so it fails. The in-scope value is reported, not substituted.
- **Per condition:** crowding 0.569 (0.804 in scope), camera motion and night (MOT17-10) 0.374.
- **What carries the gain: the motion model.**
  - Without Kalman prediction, IDF1 falls to 0.438 with 423 switches, the baseline's level.
  - Without the low-score second stage, it is 0.500 with 229 switches, and the event metrics do not
    change.
  - At k = 3 the tracker predicts two frames out of three, so a motion model matters more than any
    other part.
- **ID switches** in the error analysis happen inside tight groups standing shoulder to shoulder (#5,
  #6). Geometry alone cannot tell overlapping neighbours apart.

## Events
Sources: `docs/RESULTS.md` "Events" and "Sensitivity"; `results/per_clip/bytetrack/*.json`.
- **P 0.840, R 0.787: 63 true events, 12 false, 17 missed, 24 UNRESOLVED** over the 13 scored clips
  and 9 negative entries. The IoU baseline: P 0.811, R 0.750.
- **Where the 12 false events come from.**
  - 10 are on ACS-EG, the ground-level corridor, and 9 of them on one clip (`clip_00011_00093`: 15
    predicted events for 6 GT events).
  - People there walk close to the camera. At 5 fps with the detector every 3rd frame (one detection
    every 0.6 s), they move too far between detections for the predicted box to match the next one, so
    tracks fragment. With `min_hits` 1, each new track counts at once, and a new event opens on almost
    every detector frame (#9, #10).
  - 18 of the 24 UNRESOLVED events are on ACS-EG for the same reason.
- **Where the 17 misses come from.**
  - 10 are on the two ACS-EC lounge-view recordings, whose corridor was drawn on another camera view
    (PLAN.md A7.1).
  - Without those two recordings, recall would be 0.891 and precision 0.826. That is a post-hoc
    sensitivity check, not a result.
- **Latency.** Median 0 frames over the 63 matches. MOT17-09 (30 fps): median 4, with 5 of 14 events
  later than 5 frames, the worst 66 frames (2.2 s), inside a dense group (#7). IndoorCrowd (5 fps):
  median 0, with 10 of 49 negative, because the system's foot point entered before the ground truth's.
  The open debounce is one frame at 5 fps. Latency is the frame the event opened, not the backdated
  first-inside frame.
- **Negatives.** No false event in the 9 negative entries (218 s). That is under 5 minutes, so the bar
  could only be met with zero; one false event would have failed it. The ACS-EC negative entries on
  the lounge view lie mostly on a wall, so they test little (A7.1).
- **Coverage for events.** The only difficult condition the event metrics rest on is crowding. Camera
  motion and night come only from MOT17-10, which is scored on detection and tracking only.

## Safety: false CLEAR
Sources: `results/final_metrics.json` "safety"; `results/safety/false_clear_by_clip.json`.
- **78 of 890 occupied frames (8.8%) were reported CLEAR** while a GT person's feet were inside. Health
  was OK on all of them, so the system did not know it was wrong.
- **56 of the 78 are on the two ACS-EC lounge-view recordings** (46 of 82 and 10 of 40). The other 22
  are spread over 8 clips (at most 9 on one clip).
- In the frozen system, CLEAR means "no track inside", not "nobody inside". A person the detector
  misses, or whose track has not been confirmed, leaves the corridor CLEAR. This is the most important
  limit of the advisory output, and the health checks do not cover it.

## Degraded state
- **Pinned by tests.** Invalid video, invalid config, corrupt frames, stalled streams, low fps, low
  light and stale detections are pinned by tests (`tests/test_health.py`, `tests/test_async_detect.py`,
  `tests/test_e2e.py`). A bad file or config ends UNAVAILABLE with exit code 2 (`docs/DEMO.md` step 7).
- **On the held-out clips.** Health was DEGRADED on 31 frames of MOT17-10 (night) and OK elsewhere. The
  `min_luminance` threshold tuned on val (101.1) did not trigger on IndoorCrowd.
- **The low-confidence trigger is unsupported.** A frame's mean confidence does not predict its recall
  on held-out data (Spearman -0.008; 0.261 on val; `results/calibration/heldout.json`).

## Runtime (demo machine)
Sources: `results/runtime_report.json`, `results/runtime/heldout_offline/`,
`results/runtime/heldout_realtime.json`.
- **Offline, every frame processed, 2,484 held-out frames.**
  - Run 1 ran at 3.14 fps because one MOT17-09 frame took 549 s (PLAN.md A7.2). Runs 2 and 3 ran at
    14.79 and 14.87 fps, so the median is 14.79 fps.
  - Detector inference p50 is about 225 ms on the frames it runs on; with k = 3, the other two frames
    cost a few ms.
  - The worst-clip p95 frame latency is 235-457 ms across the runs: the detector frames miss a
    100 ms budget, and throughput is an average.
- **Paced in real time, synchronous.**
  - On the 5 fps IndoorCrowd clips the pipeline kept up: 38 of 1,305 frames dropped.
  - On the 25-30 fps MOT17 clips it did not: 888 of 1,179 frames dropped.
  - Real-time 30 fps input needs the async worker (whose accuracy is not measured) or a faster machine.
- **Run-to-run variation is large on this laptop.** The A4 runs on 17 Oct were about half as fast as
  the 14 Oct runs of the same code.

## Beyond the bar
Each item addresses a named risk.
- **Event-state redesign** (risk: a lost track silently closing an event). Events are candidate,
  active, closing, CLOSED or UNRESOLVED, with entry and exit edges. 24 held-out events ended
  UNRESOLVED instead of being reported as "left". How they are scored is described in "How UNRESOLVED
  events are scored" below.
- **Ablation** (risk: crediting the wrong component). The motion model carries the gain; the
  low-score stage adds little (above).
- **Confidence calibration** (risk: trusting scores that do not predict failure). ECE is 0.115 on
  held-out. Precision is below the score at low scores and above it at high scores. Frame confidence
  does not predict recall on held-out (`results/plots/reliability_heldout.png`).
- **Edge optimisation** (risk: missing the 10 fps bar on a CPU). YOLOX through ONNX Runtime, plus a
  measured detection interval, choose the speed and accuracy point: in-scope val F1 0.853 at k = 3 and
  10.13 fps, against 3.98 fps at k = 1 (`docs/RESULTS.md`). The cost of k = 3 at 5 fps is the track
  fragmentation above.
- **Stress suite** (risk: unknown behaviour under blur, compression, low light and shake). The frozen
  system was rerun on degraded held-out clips, with nothing retuned (`results/stress/`,
  `docs/RESULTS.md`).
  - **Known limit of the suite:** only the detector sees the degraded frames. The pipeline replay that
    computes tracking, events and health reads the clean frames (`sightline/stress.py` passes
    degraded detections to `evaluate_clip`, which reads the original source). So the stress rows show
    how detection, tracking and events degrade, but their "non-OK fraction" does not show whether
    health would flag the degraded video.
  - Health's reaction to dark, blurry and corrupt input is pinned by tests instead
    (`tests/test_health.py`).

## What these results do not show
- **Not robot footage.** No own corridor footage at robot height was evaluated. IndoorCrowd is the
  nearest public match; three of its four views look down from above.
- **The negative set is short:** 3.6 minutes.
- **Event labels are derived** from ground-truth boxes and the corridor polygon. Nobody labelled the
  events by hand.
- **One machine, one evaluation run.** Throughput varies by a factor of about 2 between sessions on
  this laptop.

---

# Before the lock: protocol history and MOT17 val results

Everything below was written before the held-out set was locked. It is kept as the record of how the protocol and the configuration were reached.

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
These are **val** measurements on public MOT17 footage. They compare configurations;
they do **not** select one. Under PLAN.md amendment A4 the detector and detection
interval are chosen on own val clips, and MOT17 does not enter that choice. They say
nothing about the held-out set or the bar. Under A1 the detection bar is judged on own
corridor clips, none of which exist yet.

**Detector F1 against speed.** Sources:
- F1: `results/detector_selection/summary.md` (all 4137 val frames, IoU 0.5, thresholds 0.15-0.90).
- Speed: `results/runtime/interval/*` (first 150 frames of each val scene, every frame processed). This is **one offline run per cell**: frames are not paced in real time, which is not yet the A4 protocol of a median of 3 runs.
- Each cell gives fps / worst per-scene p95 latency / maximum latency, in ms.

| Detector | Best F1 in scope (P / R) | Best F1, all boxes | k = 1 | k = 2 | k = 3 |
|---|---|---|---|---|---|
| MobileNetV3-320 @640x360 | 0.607 (0.745 / 0.512) | 0.389 | 5.83 fps / 195 / 205 | 11.41 / 181 / 205 | 16.53 / 189 / 220 |
| YOLOX-tiny @416 | 0.778 (0.805 / 0.753) | 0.531* | 21.50 / 52 / 123 | 39.22 / 51 / 63 | not run |
| YOLOX-s @640 | 0.853 (0.842 / 0.863) | 0.625* | 6.03 / 245 / 291 | 13.08 / 158 / 179 | 18.89 / 167 / 194 |
| ResNet50-FPN-v2 @960x540 (reference) | 0.857 (0.824 / 0.891) | 0.673 | 0.19 fps (earlier session, `results/runtime/candidates/`) | not run | not run |

\* At 0.15, the lowest threshold of the grid. A lower threshold might score higher. The grid was widened from 0.30 after one-stage best F1 values sat on the old floor (PLAN.md A5).

- **What the table shows.** On MOT17 val, YOLOX-s at k = 2 is the only measured
  setting whose detector reaches in-scope P and R >= 0.80 **and** whose pipeline
  ran at >= 10 fps offline. That is one run, not paced in real time.
- **Real time is different.** Paced in real time and synchronous, the same setting
  ran at 6.25 fps and dropped most frames (worker-thread section below).
- **Slow frames remain.** The frames on which YOLOX-s runs take up to 291 ms (k = 1)
  and 179 ms (k = 2). They individually miss the 100 ms budget; throughput is an
  average.
- **Timing is noisy.** A MobileNet control re-timed in two sessions ran at 4.99 and
  5.81 fps with the same code, so expect about 15% run-to-run variation.

**Events against detection interval.** Source: `results/interval/*_time_params.json`.
Event clips MOT17-02 and 04 only (20 GT events), cached detections, untuned
defaults (score threshold 0.5, A3 time-based settings).
- **Rows shown:** a selection. The files also hold k = 4 and 6 for each detector.
  For example, YOLOX-s at k = 6 has precision 0.952 and recall 1.00, with 21 predicted events.
- **Not shown:** the YOLOX-tiny results (`val_interval_yolox_tiny_416x416_time_params.json`)
  and the frame-based settings (`*_frame_params.json`).

| Detector | k | Event recall | Event precision | Median latency (frames) | Opened on prediction |
|---|---|---|---|---|---|
| YOLOX-s @640 | 1 | 0.95 | 0.655 | 4.0 | 0 / 29 |
| YOLOX-s @640 | 2 | 0.95 | 0.760 | 4.0 | 5 / 25 |
| YOLOX-s @640 | 3 | 1.00 | 0.769 | 3.5 | 16 / 26 |
| MobileNetV3-320 @640x360 | 2 | 1.00 | 0.588 | 5.5 | 6 / 34 |

- Twenty events is far too few to tune on. Recall moves in 5-point steps, and the
  medians are noisy.
- Event precision is below the 0.85 bar for every row at k <= 3.
- Many events end UNRESOLVED: 10-16 per row for YOLOX-s and 14-22 for MobileNet, in
  the same files. Not all of these are failures. For YOLOX-s at k = 2, 6 of the 11
  are tracks lost while the event was open; the other 5 were still open when the clip
  ended (`stream_end`). See the error analysis below.
- Why tracks are lost is not yet written up. Tracker tuning on val, after the held-out
  set is locked, is where it gets addressed.
- `rests_on_prediction` was 0 in every row shown, and 1 at k = 4 and 6 for
  YOLOX-s: almost every event had at least one detector frame inside the
  corridor.

## Detection on a worker thread: latency before and after
Source: `results/runtime/async/{before_sync,after_sync,after_async}/summary.json`, produced by `sightline runtime-summary`.

**Protocol.** `benchmark --realtime`: frames arrive on the clip's own clock. A frame that arrives while a detector frame is still being processed waits, or is dropped once it lags past the 100 ms budget. Each config ran 3 times on the first 150 frames of the five val scenes.
- "before" = the synchronous pipeline, at commit fe15d93.
- "after_sync" = the same synchronous code path at 9ff5dad, as a control.
- "after_async" = 9ff5dad with `--async-detection`.
- Values are the median of 3 runs (min-max). Latency is the worst per-clip p95 and the overall maximum.

| Config | Mode | Throughput fps | Worst clip p95 ms | Max ms | Frames dropped (of 750) | Detector frames skipped, worker busy |
|---|---|---|---|---|---|---|
| YOLOX-s @640, k = 2 | before | 6.25 (6.13-7.61) | 354 (315-551) | 700 (465-721) | 546 | 0 of 375 |
| | after_sync | 6.74 (5.87-7.25) | 380 (350-381) | 393 (364-440) | 530 | 0 of 375 |
| | after_async | 23.78 | 15 (12-36) | 27 (17-93) | 0 | 237 of 375 |
| YOLOX-s @640, k = 3 | before | 10.46 (10.23-10.65) | 294 (291-314) | 347 (334-470) | 415 | 0 of 250 |
| | after_sync | 12.27 (12.18-12.39) | 254 (253-265) | 336 (263-362) | 357 | 0 of 250 |
| | after_async | 23.78 | 26 (21-27) | 54 (42-79) | 0 | 120 of 250 |
| MobileNetV3-320 @640x360, k = 2 | before | 6.36 (6.19-6.49) | 330 (310-350) | 397 (350-430) | 543 | 0 of 375 |
| | after_sync | 7.18 (6.85-7.34) | 296 (295-300) | 314 (304-318) | 517 | 0 of 375 |
| | after_async | 23.78 | 13 (13-17) | 20 (17-34) | 0 | 259 of 375 |

What this does and does not show:
- **It removes blocking.** With the worker, no frame waits for the detector: no frame is dropped, and p95 latency falls from about 250-380 ms to 13-26 ms.
- **It does not make the detector faster.** In async runs, 23.78 fps is the source rate of these clips (30, 30, 14, 30 and 25 fps), so every frame was processed in time. But 48-69% of the detector frames were skipped because the worker was still busy, so detections happen only as often as the detector can run. Between them the tracker predicts for longer than k frames.
- **Its effect on accuracy is not measured.** When a result arrives depends on wall-clock timing, so cached-detection evaluation cannot reproduce async runs. Event recall and latency in async mode are untested.
- **Health could not see a slow detector in async mode (fixed 2026-10-17).** The low-fps check uses per-frame processing time, which is a few ms when the detector runs elsewhere. Health now also checks how old the detection behind the corridor status is (`health.max_detection_age_s`, 0.5 s), in both modes, and a test pins it with a 0.8 s detector in async mode. Async accuracy is still not measured.
- **Synchronous real time does not keep up.** Even YOLOX-s at k = 3 dropped about half of all frames (415 of 750) in sync mode.
- **Timing is noisy.** The control (same synchronous code) differs from "before" by up to 17% in throughput.

## How UNRESOLVED events are scored
**Rule** (`sightline/evaluation/events.py`, `match_events`):
- **Matching ignores status.** Every predicted event is matched the same way, whatever its status (CLOSED, UNRESOLVED or still ACTIVE). Matching is one-to-one, on temporal overlap of the [first inside, last inside] frames, largest overlap first.
- **What counts as what.** A matched predicted event is a true positive, an unmatched one a false event, an unmatched GT event a miss.
- **UNRESOLVED changes nothing in the score.** An UNRESOLVED event that overlaps a not-yet-matched GT event counts as a TP; otherwise it is a false event. `events_unresolved` is reported alongside as a count only.

**The six track-lost UNRESOLVED events** (val, YOLOX-s @640, k = 2): `results/error_analysis/val_unresolved_yolox_s_640x640_k2/index.json`.
- **3 were matched to a GT event and count as TPs:** MOT17-02 event 1 (GT 1), MOT17-02 event 10 (GT 6) and MOT17-04 event 6 (GT 2).
- **3 were not matched and count as false events:** MOT17-02 events 5, 7 and 12.

**The 31-frame gap.** All six went UNRESOLVED exactly 31 frames after their last inside frame, which `max_age` and k explain:
- That run's `max_age` was 15 detector frames (1.0 s at 30 fps with k = 2, recorded in the run's `policy.resolved_timing`). A track expires on its 16th missed detector frame (`tracking/state.py`), 32 frames after its last detection.
- Every last-inside frame is even. At k = 2 the detector runs on odd frames, so each one was the predicted frame just after the last detection, which gives 31.
- **Lost inside the zone.** UNRESOLVED (`track_lost`) is only assigned to an event whose phase was "active" at expiry, so each track's last observed position was inside the zone. Where the person went after that is not observed.

## Error analysis
`sightline errors` selects failures worst-first from saved results. The "why" column is written by a human after looking at the frames; it is not generated.

- **Held-out failures** (frozen system): `results/error_analysis/heldout/INDEX.md`.
  - 12 failures, two of each type, selected worst-first by the tool:
    - false detections, missed people, ID switches;
    - late crossings, false events, duplicate tracks.
  - The "why" for each was written after looking at the frame and checking the saved boxes, tracks and
    timelines. It was drafted by Claude for Huda to review.
  - Three causes cover most of them:
    1. MOT17 labels the off-frame part of people cut by the image edge (#1-#4).
    2. Geometry-only association inside tight groups (#5, #6, #7, #11, #12).
    3. Track fragmentation at 5 fps with k = 3 on people close to the camera (#9, #10).
  - #8 is on the ACS-EC view the corridor was not drawn for (PLAN.md A7.1).
- **False events on val** (M1): `results/error_analysis/val_false_events_yolox_s_640x640_k2/`.

- **Unresolved events on val** (YOLOX-s @640, k = 2, untuned):
  `results/error_analysis/val_unresolved_yolox_s_640x640_k2/INDEX.md`.
  - 10 were requested and 6 met the rule (track lost while the event was open).
    These are the same 6 `track_lost` events behind the interval run's 11
    unresolved.
  - The "why" column is still TODO for a human.
  - No setting was changed from these strips. The frame images stay out of git
    because they show faces.
