# Sightline: PLAN

Status: committed **before any implementation**, as the brief requires.
Author: Huda Mueen. Written 9 Oct 2026 (day 1 of 7, buffer 3).

Principle: choose the evidence first, then the model. Every number marked
`FROZEN-ON-VAL` is a placeholder rule, not a value. It gets its value on the
validation clips only, is written to `results/frozen_config.yaml`, and is never
touched again after the held-out manifest is locked.

---

## 1. Decision: what is one Sightline event?

One event = **one continuous occupancy of the configured corridor by one
anonymous track**. It covers both "entry" and "crossing"; the type is a label on
the event, not two separate detectors.

- **Reference point:** bottom-centre of the person box (the feet). It is the
  only point that is geometrically meaningful for a floor-level corridor.
- **Inside test:** reference point inside the corridor polygon.
- **Open (debounce):** the event opens when the reference point has been inside
  for `N_open` consecutive processed frames (default 3, `FROZEN-ON-VAL`). The
  event timestamp is backdated to the first inside frame for latency accounting.
- **Active:** while the track stays inside, or has been outside for fewer than
  `M_close` consecutive frames (hysteresis, default 5, `FROZEN-ON-VAL`).
- **Close:** after `M_close` consecutive frames outside, the event closes with a
  type: `crossing` if the exit edge differs from the entry edge, otherwise `entry`.
- **Re-entry:** the same track re-entering within `T_cooldown` frames reopens
  the **same** event rather than creating a second one. This is what stops
  jitter on the boundary from producing duplicate events.
- **Lost while active:** if the track expires while the event is active, the
  event becomes `UNRESOLVED`. It is **never** silently closed as "left". An
  unresolved event is reported as such and counted separately in evaluation.
- **Per-frame output** is never an event. Only state transitions are events.

## 2. Data

Hybrid, because no single public source gives indoor-robot footage plus
trackable identities plus a corridor.

| Source | Use | Licence / permission | Caveat |
|---|---|---|---|
| MOT17 **train** sequences (7 base scenes: 02, 04, 05, 09, 10, 11, 13) | Credible tracking metrics (IDF1, ID switches) with moving and static cameras, crowding, a night-ish scene | MOTChallenge terms; the dataset is listed as CC BY-NC-SA 3.0 (non-commercial), fine for coursework and not for the company. **Re-check at download time and record it in `DATA_CARD.md`.** | Mostly outdoor street footage. Test labels are hidden, so only train sequences are usable. |
| Self-recorded indoor corridor clips | The actual scenario, negative corridor cases, and clips whose rights I own | Filmed by me, with written consent from anyone visible. No faces are processed or stored beyond what raw video contains, and clips stay out of git. | Annotation is slow, so the clip count is bounded (see section 3). |

Not used: private company footage, anything from EDGE, face recognition,
demographic or biometric attributes. Tracking is anonymous integer ids only.

**MOT17 trap I am designing around.** Each MOT17 scene ships three times (DPM,
FRCNN, SDP detector variants) with identical frames. These are **one scene**.
I use exactly one copy per scene, and all copies are assigned to the same split.

**Corridor labels for MOT17.** MOT17 has no corridor, so I define one polygon
per sequence in `configs/corridors/` and derive ground-truth events from the
ground-truth boxes with the same geometric rule as section 1. This measures the
pipeline against ground-truth geometry. It is stated as a limitation, because it
means MOT17 event labels are rule-derived and not independently human-labelled.
Self-recorded clips are labelled by hand (first frame inside, exit frame).

**Condition coverage** (every clip gets a tag in the manifest):

| Condition | Source |
|---|---|
| clear / ordinary motion | MOT17 static scenes + own clips |
| partial occlusion / crowding | MOT17-02, 04, 09 + own clips with crossing people |
| camera motion / blur | MOT17 moving-camera scenes + one deliberately handheld own clip |
| low light / compression / poor contrast | real low-light own clips if time allows, otherwise the stress suite (section 8) |
| negative corridor | own clips where people are visible but never enter the polygon |

## 3. Split (leakage-safe)

Split unit = **whole scene or whole recording session**, never frames, never a
clip alone if a sibling clip shows the same place.

- `train`: unused for fitting, because nothing is trained. Reserved as the pool
  from which tuning scenes are drawn.
- `val` (tuning): thresholds, tracker parameters, corridor debounce and
  degraded-state thresholds are tuned here and only here.
- `test` (held-out): at least **12 clips**, at least 3 difficult conditions and a
  negative set. The manifest (`results/heldout_manifest.json`) is written and
  committed **before** the final tuning pass, with a SHA-256 per clip. The
  evaluation script refuses to run if a clip hash is missing or changed, or if
  any test clip id also appears in the val manifest.
- A leakage test (`tests/test_split.py`) fails the build if val and test share a
  scene id, a camera id, a recording session id, or an MOT17 base scene.

MOT17 has only 7 base training scenes, so its held-out share is small (about 2
scenes). The remaining held-out clips must come from my own recordings. This is
the main schedule risk, and annotation time is budgeted accordingly (section 10).

## 4. Baselines (what the final pipeline must justify replacing)

- **Detector baseline:** one pretrained COCO person detector at a frozen
  confidence threshold. Reported on the labelled eval subset as precision,
  recall and F1 at IoU 0.5.
- **Tracking baseline:** greedy IoU matching with a fixed `max_age`, no motion
  model, no low-confidence recovery. Nearest-centroid is the fallback if IoU
  proves degenerate on moving-camera clips.
- **Chosen tracker must beat the baseline** on IDF1, ID switches, or both, on
  held-out data, from saved results.

## 5. Detector selection (evidence first)

Candidates, all pretrained, none trained by me:

1. torchvision Faster R-CNN MobileNetV3-Large FPN (small, CPU-friendly)
2. torchvision Faster R-CNN ResNet-50 FPN v2 (accuracy reference, slower)
3. A third candidate only if neither reaches the bar on val

**Decision rule, written now:** pick the fastest candidate that reaches
precision >= 0.80 and recall >= 0.80 on **val** at some confidence threshold
**and** sustains >= 10 fps on the demo machine at the declared input size. If no
candidate does both, document a frame-sampling strategy and evaluate its effect
on event recall and latency, rather than silently lowering the bar.

Licensing: torchvision code is BSD-3-Clause. Weights are COCO-trained; the
weight name and the COCO annotation licence go in `MODEL_CARD.md`. Ultralytics
YOLO is deliberately not the default because of its AGPL-3.0 licence. If I use
it for an optional shadow detector, the licence is recorded.

## 6. Tracker

**Chosen: ByteTrack-style two-stage association**, implemented in this repo
from the published algorithm, so the code is mine and testable.

- **Information used:** geometry only: Kalman-predicted box and IoU, with
  Hungarian assignment. **No appearance, no biometrics, no re-identification.**
- **Why it fits:** its second stage matches low-confidence detections to tracks
  that high-confidence matching left unmatched. Occluded or blurred people often
  score low, so this is the mechanism that should reduce lost identities.
- **Known failure modes (to be demonstrated, not just asserted):**
  - People crossing paths can swap ids, because geometry alone cannot tell them apart.
  - Long occlusions exceed `max_age` and the person returns as a new id.
  - Camera motion shifts every box and can break IoU matching, so camera-motion
    compensation is a candidate improvement if the moving-camera results are poor.
  - A false detection can seed a ghost track. `min_hits` mitigates this at the
    cost of latency.
- **Track state, per active track:** id, age, hits, time since update,
  last-seen timestamp, confidence (smoothed), recent trajectory (bounded deque
  of reference points), and state `tentative | confirmed | lost | expired`.
- **Expiry policy:** `lost` after one missed frame; `expired` after `max_age`
  missed frames (`FROZEN-ON-VAL`). Expired ids are never reused.

## 7. Event geometry

- **Baseline (committed to now): image-space polygon**, one per camera, in
  config. Assumption: the camera is fixed relative to the corridor, or moves
  little enough that the polygon still lands on the floor region. This is
  **false** for MOT17 moving-camera scenes, and I will say so in the report
  rather than hide it.
- **No metric distances** are claimed from the monocular camera. Any "metres"
  claim requires a calibrated homography and is out of scope unless I do the
  ground-plane improvement (section 11).
- Config validation rejects self-intersecting polygons, fewer than 3 vertices,
  zero area and out-of-frame vertices (tested).

## 8. Confidence, thresholds, degraded state

**Frozen on val, then never changed:** detector confidence, tracker high/low
score split, `N_open`, `M_close`, `T_cooldown`, `max_age`, `min_hits`, and the
degraded-state thresholds below. Written to `results/frozen_config.yaml` with
its git commit hash. The final evaluation reads that file and nothing else.

**System state machine: `OK -> DEGRADED -> UNAVAILABLE`.** The output is never
"corridor clear" unless the state is `OK`. Each frame/interval carries a state.

| Trigger | State |
|---|---|
| frame decode fails or frame is empty/corrupt | `DEGRADED`; `UNAVAILABLE` after `K_corrupt` consecutive |
| timestamp gap > `gap_factor` x nominal frame interval, or timestamps non-monotonic | `DEGRADED`; `UNAVAILABLE` if the stream stalls > `stall_seconds` |
| rolling processed fps < declared minimum (10 fps) | `DEGRADED` |
| frame too dark or too blurry by the image-quality check | `DEGRADED` |
| detections systematically low-confidence over a window | `DEGRADED` |
| invalid config or unreadable source | `UNAVAILABLE` at startup, no events emitted |

Image-quality thresholds (mean luminance, Laplacian variance) are chosen on val
clips by looking at where the detector actually fails. They are not copied from
a blog. I will report whether low confidence actually predicts failure; if it
does not, I will say that.

## 9. Performance

- **Hardware:** my Windows laptop (Lenovo Yoga, Python 3.13). Exact CPU/GPU/RAM
  go in `MODEL_CARD.md` from a script that records them, not from memory.
- **Target:** >= 10 fps end-to-end at the declared input resolution, on that
  machine. I do not assume it. The detector choice in section 5 is gated on it.
- **Measured per run:** end-to-end throughput, p50/p95 frame latency, detector
  inference time, and **dropped/skipped/late frames**, as counts and percentages.
- **No hidden frame dropping.** If frames are sampled or skipped, the policy is
  recorded in the run log and its effect on event recall and latency is
  evaluated.
- Warm-up frames are excluded from latency statistics, and I say so.

## 10. Evaluation plan

- **Detection:** precision, recall, F1 at IoU 0.5 at the frozen threshold.
- **Tracking:** IDF1 and ID switches via `motmetrics`, cross-checked against my
  own small hand-computed case in a unit test.
- **Events:** precision and recall **per event**, not per frame. False events
  per 5 minutes of negative footage. Median and p95 event latency in frames,
  measured from the first frame where the **ground-truth** reference point is
  inside. `UNRESOLVED` events are reported as their own count.
- **Per-condition table** for every metric. Hard clips are not cropped out.
- **Visual error analysis:** at least five concrete failures with saved overlay
  frames, chosen by a script from the results (worst clips first) rather than
  hand-picked flattering ones.
- **All plots regenerated from saved results** by one command.

**Stress suite (improvement):** degrade the held-out clips (blur, JPEG
compression, low light, camera shake) at several levels and plot the
degradation curve of event recall and IDF1. The frozen config is **not**
retuned afterward.

## 11. Improvements I commit to attempting (beyond the bar)

Only after the bar is met cleanly:

1. **Ablation**: remove the low-confidence stage, the Kalman motion model, and
   the IoU gating one at a time, to show what is actually responsible for any gain.
2. **Stress suite** as above.
3. Stretch, only if time remains: ground-plane homography for the corridor.

Not attempting: ONNX/TensorRT, shadow logger, event-state redesign. The
redesign is partly present already through `UNRESOLVED`.

## 12. Failure behaviour and safety boundary

- Advisory only. No connection to any robot, no motion control, ever.
- Any uncertainty surfaces as `DEGRADED` or `UNAVAILABLE`. Silence is not "clear".
- Anonymous ids only. No face, identity, demographic or biometric processing.
- Raw clips and weights are git-ignored. Only manifests, configs and metrics
  are tracked.

## 13. Schedule (7 days + 3 buffer)

| Day | Work |
|---|---|
| 1 | This plan, scaffold, ingest command, manifests, split and leakage test |
| 1-2 | Film and annotate own clips (the slow part, started first) |
| 2 | Detector baseline on val, frozen threshold |
| 3 | IoU baseline, ByteTrack-style tracker, track state |
| 4 | Event logic, degraded states; **mid-point check-in** |
| 5 | Lock held-out manifest, frozen test run, per-condition results |
| 6 | Error analysis, stress suite, ablation, performance report |
| 7 | Docs, retrospective, demo video, `v1.0.0` tag |

## 14. Known risks

1. **Not enough annotated held-out clips.** MOT17 supplies about 2 scenes of
   held-out data; the rest must be my own footage. Mitigation: start filming
   on day 1, and annotate sparsely only where the metric tolerates it, and
   report the rule used.
2. **MOT17 is outdoor and moving-camera**, so image-space polygons are
   questionable there. Mitigation: report image-space and, if time, ground-plane.
3. **10 fps on a laptop CPU with a two-stage detector** may fail. Mitigation:
   the lighter detector, a lower declared input size, or documented sampling.
4. **Rule-derived MOT17 event labels** are not independent. Stated in the report.

---

## Amendments

Changes to this plan after it was committed. Each is dated, says what was already
known when it was decided, and is in place **before the held-out manifest is
locked and before any tuning**. Nothing above is edited; it is superseded here.

### A1 (2026-10-14): where the detection bar is judged, and who is "in scope"

**What was known.** Val-only detector measurements on MOT17 (scenes 02, 04, 05,
11, 13) were already saved in `results/detector_selection/`. The best recall at
any threshold was 0.282 / 0.423 for MobileNetV3-320 at 640x360 / 960x540, and
0.643 for the ResNet50-FPN-v2 reference. **This amendment was decided after seeing
those numbers**, and is reported as such. No own corridor clip had been recorded,
so no held-out or own-clip number had been seen by anyone.

1. **The detection bar (P >= 0.80, R >= 0.80 at IoU 0.5) is judged on the held-out
   own corridor clips, on in-scope people.** Those clips are the deployment
   domain: indoor, a static robot-height camera, people at corridor distances.
   MOT17 is reported as a **harder public benchmark** with full numbers, in scope
   and all boxes, failures included. MOT17 detection numbers neither pass nor
   fail the bar, and they are never left out of the report.
2. **In scope** = a scored pedestrian whose box height is **>= 10% of the source
   frame height** and whose visibility is **>= 0.65**. This follows the Caltech
   Pedestrian "Reasonable" subset (>= 50 px of a 480 px frame, >= 65% visible),
   expressed relative to frame height so it means the same at 480p and 1080p.
   It was taken from that precedent, not chosen from our results. No number per
   height or visibility had been computed when it was fixed.
   - For in-scope scoring, out-of-scope people become **ignore regions**. A
     detection on one is neither a hit nor a false positive.
   - Every detection and tracking result is reported twice: **in scope** and
     **all boxes**.
   - The two thresholds live in `configs/eval.yaml` under `in_scope`. That file is
     hashed into the lock, so they cannot change after locking.
   - Own-clip annotations must record visibility (see `docs/DATA_CARD.md`).
3. **Ground-truth classes, confirmed on 2026-10-14 over all seven MOT17 scenes.**
   Every class-1 (pedestrian) row has conf=1. Every row of classes 2-12 has conf=0.
   - **Scored:** class 1 with conf=1.
   - **Ignore regions:** the distractor classes 2 (person on vehicle), 7 (static
     person), 8 (distractor) and 12 (reflection), plus any conf-0 pedestrian.
   - **Dropped:** everything else (vehicles, bicycles, occluders). A detection on
     one is a false positive.
   - This matches the MOTChallenge protocol (TrackEval). A bug fixed the same day
     had made *all* conf-0 rows ignore regions, so detections on vehicles and
     occluders were excused. Every detection number saved before that fix is
     re-scored from the cached detections.

### A2 (2026-10-14): a detection interval replaces frame skipping

**What was known.** Frame skipping was measured on val at strides 1-3 (saved in
`results/stride/`). It processes only every s-th frame, so output and health also
exist only every s-th frame. And the detector, at about 5 fps, still could not keep
up with a 30 fps source at stride 3.

**Decision.** `pipeline.frame_stride` is removed. It is replaced by
`pipeline.detect_every` (k):
- **Every frame is processed.** The detector runs on every k-th frame. In between,
  the tracker predicts each track one step (the Kalman filter for the chosen
  tracker; the IoU baseline holds its last box). Events, the corridor status and
  health are updated on every frame.
- **A non-detector frame is not a missed detection.** Track misses and
  confirmations count detector frames only.
- **Events mark predicted evidence.** Each event records how many of its inside
  frames came from detected and from predicted boxes, whether it opened or closed
  on a predicted box, and `rests_on_prediction` (no detector frame has seen the
  person inside yet).
- **Detection metrics** score the detector on the frames it ran. **Tracking
  metrics** score the boxes reported on every frame, predicted ones included.
- k is a sampling policy. It is recorded in every run and judged by event recall,
  latency and the share of events resting on prediction, never by fps alone.

### A3 (2026-10-14): time-based, detection-interval-aware settings

**What was known.** The interval results with frame-based settings were saved
(`results/interval/*_frame_params.json`). No tuning had run. MOT17 val clips run
at 14, 25 and 30 fps, so a setting counted in frames meant a different time on
each clip, and at k > 1 `max_age` (counted in detector frames) meant k times more
video.

**Decision.** The tracker and event settings are given in seconds and converted
per clip at its own fps:
- `events.open_s`, `close_s` and `cooldown_s` count every processed frame,
  detected or predicted.
- `tracker.max_age_s` counts **detector** frames, at fps / k, so a track's
  patience in seconds is the same at any k.
- `tracker.min_hits` stays a count of real detections. It is evidence, not time.
  Its time cost grows with k, and that is reported, not hidden.
- Seconds win over the old frame counts when both are given. Every run records
  the frame counts that actually ran.
- The defaults equal the old frame values at 30 fps and k = 1: 0.1 s (3 frames),
  0.167 s (5), 0.5 s (15) and 1.0 s (30). So nothing is retuned by this change.
  The `tune` grids are the old frame grids expressed in seconds.

### A4 (2026-10-14): how the final detector and detection interval are chosen

**What was known.** MOT17 val results for seven detector configurations and for
detection intervals 1-6 (`docs/EVALUATION.md`, "Val results so far"). **No own
corridor clip has been recorded**, so no own-clip number exists. Nothing is tuned,
locked or frozen. `configs/default.yaml` keeps its current detector until this
rule is applied.

**Rule.** It replaces the selection rule in section 5 for the final choice.
1. **Candidates are fixed now.** Detectors: MobileNetV3-320 FPN at 640x360 and
   960x540, SSDLite320, YOLOX-nano, YOLOX-tiny and YOLOX-s, each at its stated
   input size. Detection intervals k in {1, 2, 3, 4, 6}. ResNet50-FPN-v2 is an
   accuracy reference only. No candidate is added after own-clip results are
   seen.
2. **Accuracy.** For each detector, take its in-scope detection F1 (IoU 0.5,
   amendment A1) on the **own val clips**: the best F1 over the fixed threshold
   grid (`evaluation.detection.THRESHOLDS`). That threshold becomes its operating
   point. Detection is scored on the frames the detector runs on, so F1 does not
   depend on k.
3. **Speed.** For each (detector, k) pair, take the end-to-end pipeline
   throughput on the own val clips. Every frame is processed, and the throughput
   is the **median of 3 separate `benchmark` runs** on the demo machine. The
   **range (min-max)** of the 3 runs is reported next to it. A pair qualifies only
   if its median is >= 10 fps.
4. **Choice.** Pick the qualifying pair with the **highest in-scope F1**. If
   several qualify for the same detector, pick the **smallest k**: it rests least
   on prediction. If two detectors tie on F1 at three decimals, pick the faster
   median.
5. **If no pair qualifies**, no detector is selected. That is reported as a
   failure of the throughput bar. The bar is not lowered and no candidate is
   added.
6. **Latency.** Every throughput figure in a report is shown with the **p95 frame
   latency** and the **worst-case frame latency** next to it. A throughput average
   alone hides the slow frames on which the detector runs.
7. MOT17 numbers do not enter the choice. They stay in the report as a harder
   public benchmark (A1).

### A5 (2026-10-14): changes made after seeing val data, recorded late

A repo review on 2026-10-14 found these changes in commits but not in this plan.
They are recorded here with what was known when each was made. None used held-out
or own-clip data, because none exists.

1. **Threshold grid 0.15-0.90** (commit 56d408c). **What was known:** every
   SSDLite and YOLOX best F1 on val sat at 0.30, the lowest value of the old
   0.30-0.90 grid. No result below 0.30 had been computed. The grid starts at 0.15
   because the operating threshold must stay above the 0.10 detection floor. This
   is the "fixed threshold grid" A4 refers to. On all boxes, four of seven
   candidates still have their best F1 at 0.15, so that column is limited by the
   grid.
2. **Detection on a worker thread** (`pipeline.async_detection`, commit d93b2f2).
   **What was known:** synchronous real-time runs dropped most frames
   (`results/runtime/async/before_sync/`). With the worker, a detector frame is
   **skipped** whenever the worker is still busy (48-69% of detector frames in the
   val runs). So "the detector runs on every k-th frame" (A2) holds only for the
   synchronous pipeline, which is the default. Async accuracy is not measured.
   **Until it is, A4's throughput is measured with the default synchronous
   pipeline** (`benchmark` without `--async-detection`), and async runs are
   reported separately. Otherwise every candidate would qualify at the source frame
   rate.
3. **ONNX:** section 11 said "Not attempting: ONNX/TensorRT". YOLOX now runs on
   ONNX Runtime (commit 1b39626), added for decision (3) of 2026-10-14 to evaluate
   faster permissively licensed detectors. TensorRT is still not attempted.
4. **A1's recall figures** (0.282 / 0.423 / 0.643) are as saved at commit
   a2c6a10, on the 0.30-0.90 grid with all boxes. The files in
   `results/detector_selection/` now hold the re-scored values (A1 ground-truth
   fix, A5.1 grid). The A1 decision does not depend on the exact figures.
5. Section 3 names `tests/test_split.py`. The leakage tests are in
   `tests/test_manifest.py`.


### A6 (2026-10-17): IndoorCrowd as the held-out evaluation set

**What was known.** No own corridor clip has been recorded, and the deadline is
2026-10-18. The IndoorCrowd `tracking` config was imported: 12 recordings from 4
sites, 1,305 frames at 5 fps, one fixed 1280x720 camera per site
(`results/ingest/indoorcrowd_import.json`). Its frames, ground-truth boxes and
ground-truth foot traces were looked at to draw corridors and tag conditions. **No
detector, tracker or event output on IndoorCrowd existed** when this was decided.
Nothing is tuned, locked or frozen. The MOT17 val results are those in
`docs/EVALUATION.md`.

1. **Role.** IndoorCrowd is the primary labelled evaluation set. Its fixed indoor
   cameras and corridor-distance people are the closest public match to the
   deployment. The A1 detection bar is judged on held-out IndoorCrowd, in scope.
   Own clips, once recorded, join in a later version with a fresh lock. They are
   not part of v1.0.
2. **Split by site, never by frame or recording. All four sites go to test.**
   - The bar asks for at least 12 held-out clips, and MOT17 gives 2. Each site
     moved to val would take its 3 recordings out of test, leaving 11 or fewer.
   - The held-out set is then 14 clips from 6 scenes: MOT17-09, MOT17-10 and the
     12 IndoorCrowd recordings.
   - In the manifest, site = scene = camera, so `split-check` enforces the
     site boundary.
   - **Cost:** tuning sees no IndoorCrowd clip. The test therefore measures
     transfer from MOT17 tuning to indoor 5 fps video. A failure there is
     reported, not tuned away.
3. **Frame-rate-matched val.** The test clips run at 5 fps and the val clips at
   14-30 fps. So each val scene also enters val as a decimated copy at about
   5 fps: every round(fps / 5)-th frame, with the ground truth renumbered.
   - The copies are the same scenes in the same split, so no data crosses the
     split.
   - They count for tracking and events only. Detection is already scored on
     every original val frame, and counting the copies too would score those
     frames twice.
   - Tuning pools the originals and the copies.
4. **In scope on IndoorCrowd.** IndoorCrowd gives no visibility, so boxes carry
   visibility -1 (unknown). In scope = height >= 10% of the frame (72 px), judged
   by height alone. This is the stricter reading for the detector: an occluded
   person still counts.
5. **Ground-truth event labels in seconds.** `configs/eval.yaml` gives the label
   rule as `min_s: 0.1` and `merge_gap_s: 0.1667`, converted per clip at its fps,
   rounding half up. The rule now means the same time on every clip, as the
   system's own settings already do (A3).
   - **Unchanged labels:** every clip at 25 or 30 fps keeps its label-derived
     event count (MOT17-02 11, 04 9, 09 14, 10 13, 11 40, 13 21).
   - **One change, unused:** MOT17-05 (14 fps) goes from 53 to 64 derived
     events. MOT17-05 is scored on detection and tracking only, so no metric
     uses them.
   - **At 5 fps** the rule is 1 frame inside and a 1-frame merge gap.
6. **Corridors.** One per site (one fixed camera each): a band across the main
   walkway, drawn from the first frame and the ground-truth foot traces before
   any system output existed (`configs/corridors/IC-<site>.yaml`). Every
   recording has ground-truth foot points inside its site's corridor.
7. **Negative corridor set.** No recording is naturally negative, so a second,
   negative corridor was placed at three sites:
   - **Where:** visible floor next to a walkway (ACS-EC seating area, IE-Central
     floor in front of the back wall, R-Central stairs). It was checked against
     the ground truth before any system output: no ground-truth foot point lies
     inside it in any recording of that site. ACS-EG has no such floor area (the
     candidate was wall), so it has none.
   - **How it is scored:** each recording with its site's negative corridor is a
     negative clip entry. It counts for **events only**, so detection and
     tracking are not counted twice. These entries are reported as negative
     corridor configurations on the same footage and do not count towards the
     12 clips.
   - **Size:** 9 entries, 1,090 frames, about 3.6 minutes of negative footage.
     Under 5 minutes, a single false event already exceeds the bar of 1 per
     5 minutes. That is stated with the result, not hidden.
8. **The detector and interval rule (A4) on MOT17 val.** A4 picks the detector and
   k on own val clips, and none exist. It is applied unchanged otherwise on MOT17
   val:
   - **Accuracy:** in-scope F1 over all val frames, from
     `results/detector_selection/`.
   - **Speed:** median of 3 synchronous `benchmark` runs on the first 150 frames
     of each val scene.
   - **Fixed:** the candidates, the grid and the >= 10 fps bar do not change.

### A7 (2026-10-17): errata found after the held-out evaluation

**What was known.** The single held-out evaluation had run (`results/final_metrics.json`,
`results/final_run_log.jsonl`). These two problems were found afterwards, during the error analysis
and the runtime write-up. **Nothing below changed a setting, a corridor, a label or the evaluation**,
which was not rerun. They are recorded so that the results can be read correctly.

1. **ACS-EC has two camera views, not one.**
   - A6.6 assumed one fixed camera per IndoorCrowd site. ACS-EC's `clip_00001_00051` is an elevated
     view of the atrium. Its `clip_00001_00082` and `clip_00026_00091` are a ground-level view of the
     lounge. The import check had shown the difference (frames with the same number differed by 62 grey
     levels against about 19 between the other two), but it was read only as "not the same video".
   - The ACS-EC corridor was drawn on `clip_00001_00051`'s view. On the other two recordings it lies
     over the busy counter area, and the negative corridor lies mostly on a wall.
   - The labels use the same polygon, so the event metrics stay internally consistent. But those two
     clips test "entered this image region", not the walkway the corridor was drawn for.
   - The other three sites have one view each (checked on the first frame of all 12 recordings).
   - Redrawing the corridor now would be changing the test after seeing it, so it is not done. A
     post-hoc sensitivity table in `docs/RESULTS.md` shows the event metrics with and without these two
     recordings. The reported result is the full set.
2. **The runtime report `evaluate` read contains a 549 s stall.**
   - `results/runtime_report.json` (offline run 1) has one MOT17-09 frame that took 548.9 s. Its overall
     throughput is therefore 3.14 fps, and the mechanical bar check records the throughput as FAIL.
   - Two more runs at the same code, on the same clips, gave 14.79 and 14.87 fps with no such frame
     (`results/runtime/heldout_offline/`). The median of the three is 14.79 fps.
   - The cause of the stall was not identified. The bar check is left as recorded. The repeated runs are
     reported next to it, as A4 does for the detector choice.

### A8 (2026-10-18): the evaluation uses sourced data only

**What was known.** The held-out evaluation had run once (A6, A7), on sourced public data only. Huda's
mentor confirmed that the task required sourcing data and never required filming any.

1. **Data.** The evaluation used **sourced data only**: MOT17 (val and part of held-out) and IndoorCrowd
   (held-out, A6). **No own footage was recorded**, and none is part of v1.0.
2. **What this supersedes.** Every place in this plan that makes own corridor clips a requirement is
   superseded:
   - section 2 (self-recorded clips as a source);
   - section 3 ("the remaining held-out clips must come from my own recordings");
   - section 13 (filming on days 1-2);
   - section 14 risk 1;
   - A1.1 (the detection bar on own clips; A6.1 already moved it to IndoorCrowd);
   - A4 (choice on own val clips; A6.8 already applied it on MOT17 val).

   The original text is not edited, as for every amendment.
3. **The own-footage tools stay, as optional.** `draw-corridor`, `label-events`, `check-clip`,
   `import-cvat`, `docs/RECORDING_GUIDE.md` and `docs/SOLO_RECORDING_GUIDE.md` stay in the repository.
   They are optional, for anyone who later wants to evaluate on their own corridor, and are not part of
   this evaluation.
4. **Nothing in the evaluation changes.** No held-out number, setting, polygon or threshold changes
   with this amendment.
