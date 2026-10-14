# Perception card (model card)

**Status:** real detectors have been run on MOT17 **val** only (`docs/EVALUATION.md`, "Val results so far"). **No detector is selected yet.** The final detector and detection interval will be chosen on own val clips by the rule in PLAN.md amendment A4. `configs/default.yaml` still names MobileNetV3-Large-320 FPN as a placeholder. Nothing is tuned, locked or frozen.

## Components
| Part | Choice | Why |
|---|---|---|
| Detector | Pretrained COCO detectors, person class only. Candidates are listed below. | Permissive licences only (BSD-3 / Apache-2.0 / MIT); avoids AGPL (Ultralytics). Chosen on own val clips by PLAN.md A4, never on test results. |
| Tracker | Own ByteTrack-style two-stage association: Kalman (cx, cy, aspect, h) + IoU + Hungarian, no appearance model | Geometry only; cheap; handles low-score recovery |
| Baseline | Greedy IoU tracker | Required comparison; tuned on the same grid |
| Events | Bottom-centre point vs corridor polygon, debounced open (n), hysteresis close (m), cooldown merge | Deterministic and explainable |
| Health | OK / DEGRADED / UNAVAILABLE from frame validity, timestamps, fps, brightness, blur, detector confidence | Failure must be visible, never silent |

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
Corridor status is `CLEAR` only when health is OK and no person is inside, a candidate, or a tentative track. Otherwise `UNKNOWN` (or `OCCUPIED`). A track lost mid-event yields an `UNRESOLVED` event, not a guess. Exit code 2 = UNAVAILABLE.

## Not supported / known failure modes
- No person re-identification: a person who leaves and returns after the track expires gets a new id.
- Heavy occlusion/crowds: ids switch and events may merge or split.
- Extreme low light, blur, shake: expected to degrade; health should flag it. Camera shake moves the corridor relative to the image (no stabilisation).
- Pose: only the bottom-centre of the box is used; a person leaning in without feet crossing counts as outside.
- Unvalidated beyond the clips in the manifest.

## Evaluation
See `EVALUATION.md`. It has MOT17 val results only, and no held-out result exists. The bars come from the brief and are checked mechanically by `bar_check`. Under PLAN.md amendment A1 the detection bar is judged on own clips, in scope.
