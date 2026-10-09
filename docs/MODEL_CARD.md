# Perception card (model card)

**Status: real-detector numbers not yet produced.** The sandbox the code was written in had no PyTorch/weights, so the detector path is implemented but only the cached-detection path is tested.

## Components
| Part | Choice | Why |
|---|---|---|
| Detector | torchvision Faster R-CNN, pretrained COCO, person class only. Default MobileNetV3-Large-320 FPN; ResNet50-FPN-v2 as accuracy reference | BSD-3 code; avoids AGPL (Ultralytics). Chosen by validation F1 vs speed per the PLAN rule, not by test results. |
| Tracker | Own ByteTrack-style two-stage association: Kalman (cx, cy, aspect, h) + IoU + Hungarian, no appearance model | Geometry only; cheap; handles low-score recovery |
| Baseline | Greedy IoU tracker | Required comparison; tuned on the same grid |
| Events | Bottom-centre point vs corridor polygon, debounced open (n), hysteresis close (m), cooldown merge | Deterministic and explainable |
| Health | OK / DEGRADED / UNAVAILABLE from frame validity, timestamps, fps, brightness, blur, detector confidence | Failure must be visible, never silent |

## Intended use
Advisory perception for a human or downstream system: "someone entered / crossed this region". **Not** a safety certificate.

## Output contract
Corridor status is `CLEAR` only when health is OK and no person is inside, a candidate, or a tentative track. Otherwise `UNKNOWN` (or `OCCUPIED`). A track lost mid-event yields an `UNRESOLVED` event, not a guess. Exit code 2 = UNAVAILABLE.

## Not supported / known failure modes
- No person re-identification: a person who leaves and returns after the track expires gets a new id.
- Heavy occlusion/crowds: ids switch and events may merge or split.
- Extreme low light, blur, shake: expected to degrade; health should flag it. Camera shake moves the corridor relative to the image (no stabilisation).
- Pose: only the bottom-centre of the box is used; a person leaning in without feet crossing counts as outside.
- Unvalidated beyond the clips in the manifest.

## Evaluation
See `EVALUATION.md` (currently no results). Bars come from the brief and are checked mechanically by `bar_check`.
