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
