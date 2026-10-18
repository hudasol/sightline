# Perception card (model card)

**Status (2026-10-17): frozen.** The shipped system is `results/frozen_config.yaml`: YOLOX-s @640x640 on ONNX Runtime, detector on every 3rd frame, ByteTrack-style tracker, all thresholds tuned on val only after the held-out manifest was locked. Its held-out numbers are in `docs/RESULTS.md` (generated) and interpreted in `docs/EVALUATION.md`.

## How the shipped configuration was chosen
1. **Detector and detection interval: PLAN.md A4, applied on MOT17 val (A6.8).** YOLOX-s @640 has the highest in-scope val F1 of the candidates (0.853). The smallest interval whose median of 3 synchronous offline runs reaches 10 fps is k = 3 (10.13 fps, range 9.29-11.15; k = 2: 7.30, k = 1: 3.98). Source: `results/detector_selection/a4_choice.json`.
2. **Thresholds and timings: `sightline tune` on val only** (5 MOT17 val scenes plus their about-5 fps copies, A6.3), after the held-out lock. The frozen values and the grids they came from are in `results/frozen_config.yaml` (`provenance`) and `results/tuning/`.

## Frozen operating point
Read these from `results/frozen_config.yaml`; they are not repeated here so that they cannot drift:
- `detector.score_threshold`: the operating threshold, also the tracker's high/low split.
- `tracker.max_age_s`, `tracker.min_hits`, `tracker.match_iou`, and the same three for `baseline_tracker`
  (the baseline is tuned on the same grid).
- `events.open_s`, `events.close_s`, `events.cooldown_s`: converted to frames at each clip's fps.
- `health.*`: the degraded-state thresholds. `max_detection_age_s` (0.5 s) is declared, not tuned.

## Components
| Part | Choice | Why |
|---|---|---|
| Detector | **YOLOX-s, 640x640, ONNX Runtime, COCO person class.** The other candidates are listed below. | Permissive licences only (BSD-3 / Apache-2.0 / MIT); avoids AGPL (Ultralytics). Chosen by PLAN.md A4 on val, never on test results. |
| Detection interval | k = 3: the detector runs on every 3rd frame, the tracker predicts in between (PLAN.md A2) | The smallest k at >= 10 fps on this laptop. At a 5 fps source it means a detection every 0.6 s. |
| Tracker | Own ByteTrack-style two-stage association: Kalman (cx, cy, aspect, h) + IoU + Hungarian, no appearance model | Geometry only; cheap; handles low-score recovery |
| Baseline | Greedy IoU tracker | Required comparison, tuned on the same grid as the chosen tracker |
| Events | Bottom-centre point vs corridor polygon, debounced open (n), hysteresis close (m), cooldown merge | Deterministic and explainable |
| Health | OK / DEGRADED / UNAVAILABLE from frame validity, timestamps, fps, brightness, blur, detector confidence, and the age of the detection the status rests on | Failure must be visible, never silent |

## Detector candidates: sources, licences, files

None of these weights is in git (`weights/` and `*.onnx`/`*.pth` are ignored). Each file below was hashed on 2026-10-14 on the demo machine. Every torchvision file name carries the first 8 hex digits of its own SHA-256, and those match the hashes here.

| Candidate | Runtime and code licence | Weights source | File | Size (bytes) | SHA-256 |
|---|---|---|---|---|---|
| Faster R-CNN MobileNetV3-Large-320 FPN | torchvision (BSD-3-Clause) | download.pytorch.org/models (`FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT`) | `fasterrcnn_mobilenet_v3_large_320_fpn-907ea3f9.pth` | 77,844,807 | `907ea3f91ff92242bc1baea8049276a3e76bca48ce7560bd268cc029f37977b5` |
| Faster R-CNN MobileNetV3-Large FPN (thread sweeps only) | torchvision (BSD-3-Clause) | download.pytorch.org/models | `fasterrcnn_mobilenet_v3_large_fpn-fb6a3cc7.pth` | 77,844,807 | `fb6a3cc702b1df54c18a44b26708cd083614211062d0c36d2ca7bf9270df3533` |
| Faster R-CNN ResNet50-FPN-v2 (accuracy reference) | torchvision (BSD-3-Clause) | download.pytorch.org/models | `fasterrcnn_resnet50_fpn_v2_coco-dd69338a.pth` | 175,221,657 | `dd69338a24b8d7381807e247652bdc356325bcbaf1cd3e092e00e0a1a58706bf` |
| SSDLite320 MobileNetV3-Large | torchvision (BSD-3-Clause) | download.pytorch.org/models (`SSDLite320_MobileNet_V3_Large_Weights.DEFAULT`) | `ssdlite320_mobilenet_v3_large_coco-a79551df.pth` | 14,069,355 | `a79551df90c79834bcd3bb3845ef9d966b5449a3a9b2833ae8404778ca5d65d2` |
| YOLOX-nano (416x416) | YOLOX (Apache-2.0) on ONNX Runtime 1.31.0 (MIT) | github.com/Megvii-BaseDetection/YOLOX release `0.1.1rc0` | `yolox_nano.onnx` | 3,659,407 | `c789161ed43c8269fcd4e67c67eeeb4e80c622da2eb296a20bc6007bd18a0b7d` |
| YOLOX-tiny (416x416) | YOLOX (Apache-2.0) on ONNX Runtime 1.31.0 (MIT) | same release | `yolox_tiny.onnx` | 20,219,662 | `427cc366d34e27ff7a03e2899b5e3671425c262ea2291f88bb942bc1cc70b0f7` |
| YOLOX-s (640x640) | YOLOX (Apache-2.0) on ONNX Runtime 1.31.0 (MIT) | same release | `yolox_s.onnx` | 35,858,002 | `c5c2d13e59ae883e6af3b45daea64af4833a4951c92d116ec270d9ddbe998063` |

- **Training data.** All weights are trained on COCO train2017. Its annotations are CC BY 4.0 and its images are under the Flickr terms of use. None of these models was trained or fine-tuned in this project.
- **Runtime.** The torchvision files are cached by torch in `~/.cache/torch/hub/checkpoints/`. The YOLOX files go in `weights/yolox/`; each ONNX run records its file's SHA-256 in the detector info.
- **YOLOX preprocessing** follows the official ONNX demo: letterbox to the fixed square input, BGR 0-255, grid decode, NMS IoU 0.45, person = COCO index 0.
- **SSDLite** is fed its fixed 320x320 input (aspect not kept, as in its training).

## Intended use
Advisory perception for a human or downstream system: "someone entered / crossed this region". **Not** a safety certificate.

## Output contract
Corridor status is `CLEAR` only when health is OK, no track is in a candidate, active or closing phase, and no observed tentative track is inside (`sightline/pipeline.py`). It is not evidence that nobody is inside: 78 of 890 occupied held-out frames were reported CLEAR (`results/final_metrics.json`, `safety`). Otherwise `UNKNOWN` (or `OCCUPIED`). A track lost mid-event yields an `UNRESOLVED` event, not a guess. Exit code 2 = UNAVAILABLE.

## Not supported / known failure modes
- No person re-identification: a person who leaves and returns after the track expires gets a new id.
- Heavy occlusion/crowds: ids switch and events may merge or split.
- Extreme low light, blur, shake: tracking degrades (stress suite, `results/stress/`). Health is designed to flag dark and blurry frames, and tests pin that. Whether it flags degraded held-out video was **not** measured: the stress replay reads clean frames (EVALUATION, stress suite). Camera shake moves the corridor relative to the image (no stabilisation).
- Pose: only the bottom-centre of the box is used; a person leaning in without feet crossing counts as outside.
- Detection interval k = 3 at a 5 fps source: a person can move far between detections, and the tracker
  predicts across the gap. Events opened or closed on a predicted box are marked in each event's evidence.
- IndoorCrowd is the only indoor footage, and it is not robot footage. Its cameras are fixed, and three of
  the four look down from an elevated or overhead position; only ACS-EG is at ground level. No own
  robot-height footage was evaluated.
- **ACS-EC corridor drawn on one camera view (found after the evaluation, PLAN.md A7.1).** IndoorCrowd's ACS-EC site has two camera views. The ACS-EC corridor was drawn on one view (`clip_00001_00051`). On the other two recordings (`clip_00001_00082`, `clip_00026_00091`) it lands on the cafe counter. These two recordings account for **10 of the 17 missed events** (`results/per_clip/bytetrack/`) and **56 of the 78 false-CLEAR frames** (`results/safety/false_clear_by_clip.json`). The corridor was **not** redrawn for the reported result, and the held-out evaluation was not rerun. A post-hoc sensitivity table without these two recordings is in `docs/RESULTS.md`; it is not a held-out result.
- Unvalidated beyond the clips in the manifest.

## Hardware benchmark
The runtime report on the demo machine (`results/runtime_report.json`, offline, every frame processed;
`results/runtime/heldout_realtime.json`, paced in real time) gives throughput, p50 and p95 latency,
inference time and dropped or skipped frames, with the machine and the commit. The tables are in
`docs/RESULTS.md`. The machine is a Windows 11 laptop with an AMD Family 25 Model 80 CPU (16 logical
cores), CPU only (the runtime reports `cuda_available: false`), Python 3.13, ONNX Runtime 1.31.0. Throughput on it varied between sessions: the same code
path ran 13.08 fps on 2026-10-14 and 7.30 fps on 2026-10-17 at k = 2 (`results/runtime/a4_selection/README.md`).

## Evaluation
See `EVALUATION.md` (interpretation) and `RESULTS.md` (tables generated from `results/`). The bars come
from the brief and are checked mechanically by `bar_check`. Under PLAN.md A1 and A6 the detection bar is
judged on held-out IndoorCrowd clips, in scope.
