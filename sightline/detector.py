"""Person detection. Pretrained only: nothing here trains a model.

The detector returns every person box above a low floor (cfg.min_score). The
FROZEN operating threshold (cfg.score_threshold) is applied downstream: it is
the P/R/F1 operating point and the tracker's high/low split. Keeping the floor
low is what lets the tracker's second stage use low-confidence boxes.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np

from sightline.config import DetectorConfig
from sightline.ingest import sha256_file
from sightline.types import Detection


class DetectorUnavailable(RuntimeError):
    pass


class Detector(Protocol):
    name: str

    def detect(self, image: np.ndarray) -> list[Detection]: ...

    def info(self) -> dict: ...


# model name -> (torchvision builder, weights enum name, family). Code BSD-3-Clause, weights COCO-trained.
#   rcnn: aspect-preserving resize; we pin torchvision's resize to the size we feed it
#   ssd:  always resized to a fixed square (aspect NOT kept, as in training); we feed exactly that square
_TORCHVISION_MODELS = {
    "fasterrcnn_mobilenet_v3_large_320_fpn": ("fasterrcnn_mobilenet_v3_large_320_fpn", "FasterRCNN_MobileNet_V3_Large_320_FPN_Weights", "rcnn"),
    "fasterrcnn_mobilenet_v3_large_fpn": ("fasterrcnn_mobilenet_v3_large_fpn", "FasterRCNN_MobileNet_V3_Large_FPN_Weights", "rcnn"),
    "fasterrcnn_resnet50_fpn_v2": ("fasterrcnn_resnet50_fpn_v2", "FasterRCNN_ResNet50_FPN_V2_Weights", "rcnn"),
    "ssdlite320_mobilenet_v3_large": ("ssdlite320_mobilenet_v3_large", "SSDLite320_MobileNet_V3_Large_Weights", "ssd"),
}
_SSD_SIZE = {"ssdlite320_mobilenet_v3_large": 320}

# YOLOX ONNX exports, Apache-2.0, COCO-trained: github.com/Megvii-BaseDetection/YOLOX/releases/tag/0.1.1rc0
_ONNX_YOLOX = ("yolox_nano", "yolox_tiny", "yolox_s")


def make_detector(cfg: DetectorConfig) -> "Detector":
    """The detector named by cfg.name. Raises DetectorUnavailable for an unknown name or missing runtime/weights."""
    if cfg.name in _TORCHVISION_MODELS:
        return TorchvisionPersonDetector(cfg)
    if cfg.name in _ONNX_YOLOX:
        return OnnxYoloxDetector(cfg)
    raise DetectorUnavailable(f"unknown detector {cfg.name!r}; choose from {sorted([*_TORCHVISION_MODELS, *_ONNX_YOLOX])}")


def fit_size(width: int, height: int, box_w: int, box_h: int) -> tuple[int, int]:
    """Largest (w, h) with the frame's aspect ratio that fits inside box_w x box_h. Never upscales."""
    scale = min(box_w / width, box_h / height, 1.0)
    return max(1, round(width * scale)), max(1, round(height * scale))


class TorchvisionPersonDetector:
    """COCO-pretrained torchvision detector, person class only (COCO id 1).

    Code: BSD-3-Clause (torchvision). Weights: trained on COCO train2017; record the
    exact weights name below in the model card.
    """

    PERSON_LABEL = 1

    def __init__(self, cfg: DetectorConfig, load_weights: bool = True) -> None:
        try:
            import torch
            import torchvision
        except ImportError as exc:
            raise DetectorUnavailable(
                "torch/torchvision are not installed: pip install -e '.[detector]'"
            ) from exc
        if cfg.name not in _TORCHVISION_MODELS:
            raise DetectorUnavailable(f"unknown detector {cfg.name!r}; choose from {sorted(_TORCHVISION_MODELS)}")
        if cfg.torch_threads:
            torch.set_num_threads(cfg.torch_threads)
        self._torch = torch
        self.cfg = cfg
        self.name = cfg.name
        builder_name, weights_name, self.family = _TORCHVISION_MODELS[cfg.name]
        builder = getattr(torchvision.models.detection, builder_name)
        weights = getattr(torchvision.models.detection, weights_name).DEFAULT if load_weights else None
        self._weights_name = f"{weights_name}.DEFAULT" if load_weights else "none (random init)"
        box_w, box_h = cfg.input_size
        if self.family == "ssd":
            size = _SSD_SIZE[cfg.name]
            if [box_w, box_h] != [size, size]:
                raise DetectorUnavailable(f"{cfg.name} only runs at a fixed {size}x{size} input; set input_size [{size}, {size}]")
            self.model = builder(weights=weights, weights_backbone=None, score_thresh=cfg.min_score)
        else:
            self.model = builder(
                weights=weights, weights_backbone=None, box_score_thresh=cfg.min_score,
                min_size=min(box_w, box_h), max_size=max(box_w, box_h),
            )
        self.model.eval().to(cfg.device)
        self._versions = {"torch": torch.__version__, "torchvision": torchvision.__version__}

    def detect(self, image: np.ndarray) -> list[Detection]:
        """Resize to fit cfg.input_size, detect, and return boxes in ORIGINAL frame pixels."""
        torch = self._torch
        h, w = image.shape[:2]
        if self.family == "ssd":  # the model's own fixed square; resizing here avoids a full-resolution tensor
            nw, nh = self.cfg.input_size
        else:
            nw, nh = fit_size(w, h, *self.cfg.input_size)
            # pin torchvision's own resize to this exact size so the frame is not resized a second time
            self.model.transform.min_size = (min(nw, nh),)
            self.model.transform.max_size = max(nw, nh)
        small = image if (nw, nh) == (w, h) else cv2.resize(image, (nw, nh), interpolation=cv2.INTER_AREA)
        rgb = small[:, :, ::-1]
        tensor = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1).float().div_(255.0).to(self.cfg.device)
        with torch.inference_mode():
            out = self.model([tensor])[0]
        sx, sy = w / nw, h / nh
        dets = []
        for box, label, score in zip(out["boxes"].cpu().numpy(), out["labels"].cpu().numpy(), out["scores"].cpu().numpy()):
            if int(label) != self.PERSON_LABEL or float(score) < self.cfg.min_score:
                continue
            x1, y1, x2, y2 = float(box[0]) * sx, float(box[1]) * sy, float(box[2]) * sx, float(box[3]) * sy
            x1, x2 = max(0.0, min(x1, w)), max(0.0, min(x2, w))
            y1, y2 = max(0.0, min(y1, h)), max(0.0, min(y2, h))
            if x2 - x1 >= 2 and y2 - y1 >= 2:
                dets.append(Detection((x1, y1, x2, y2), float(score)))
        return dets

    def info(self) -> dict:
        return {"detector": self.name, "weights": self._weights_name, "min_score": self.cfg.min_score,
                "input_size": list(self.cfg.input_size),
                "resize": ("stretched to the fixed square input (aspect not kept, as in training)" if self.family == "ssd"
                           else "fit inside input_size, aspect kept, no upscaling"),
                "device": self.cfg.device, "backend": "torchvision",
                "torch_threads": self._torch.get_num_threads(),
                "torch_interop_threads": self._torch.get_num_interop_threads(), **self._versions}


def _nms(boxes: np.ndarray, scores: np.ndarray, iou_thr: float) -> list[int]:
    """Greedy NMS on xyxy boxes; returns kept indices, highest score first."""
    order = scores.argsort()[::-1]
    keep: list[int] = []
    area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    while order.size:
        i = int(order[0])
        keep.append(i)
        rest = order[1:]
        xx1 = np.maximum(boxes[i, 0], boxes[rest, 0])
        yy1 = np.maximum(boxes[i, 1], boxes[rest, 1])
        xx2 = np.minimum(boxes[i, 2], boxes[rest, 2])
        yy2 = np.minimum(boxes[i, 3], boxes[rest, 3])
        inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
        iou = inter / np.maximum(area[i] + area[rest] - inter, 1e-12)
        order = rest[iou <= iou_thr]
    return keep


def yolox_decode(raw: np.ndarray, input_hw: tuple[int, int], strides: tuple[int, ...] = (8, 16, 32)) -> np.ndarray:
    """YOLOX head output (N, 5 + classes) on grid cells -> (cx, cy, w, h) in input pixels.

    Same as yolox/utils/demo_utils.py demo_postprocess for the 0.1.1rc0 ONNX exports.
    """
    grids, expanded = [], []
    for s in strides:
        hs, ws = input_hw[0] // s, input_hw[1] // s
        xv, yv = np.meshgrid(np.arange(ws), np.arange(hs))
        g = np.stack((xv, yv), 2).reshape(-1, 2)
        grids.append(g)
        expanded.append(np.full((g.shape[0], 1), s))
    grid, stride = np.concatenate(grids, 0), np.concatenate(expanded, 0)
    out = raw.astype(np.float32).copy()
    out[:, :2] = (out[:, :2] + grid) * stride
    out[:, 2:4] = np.exp(out[:, 2:4]) * stride
    return out


class OnnxYoloxDetector:
    """YOLOX (Apache-2.0) through ONNX Runtime (MIT), person class only (COCO index 0).

    Preprocessing follows the official ONNX demo: letterbox into the model's fixed square
    input (aspect kept, padded with 114), BGR, 0-255, no normalisation.
    """

    PERSON_INDEX = 0
    NMS_IOU = 0.45  # the official demo's value

    def __init__(self, cfg: DetectorConfig) -> None:
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise DetectorUnavailable("onnxruntime is not installed: pip install -e '.[onnx]'") from exc
        path = Path(cfg.weights_path or f"weights/yolox/{cfg.name}.onnx")
        if not path.is_file():
            raise DetectorUnavailable(f"YOLOX weights not found at {path}; download {cfg.name}.onnx from the YOLOX 0.1.1rc0 release")
        opts = ort.SessionOptions()
        if cfg.torch_threads:  # the one intra-op thread setting, whichever runtime runs the model
            opts.intra_op_num_threads = cfg.torch_threads
        self.session = ort.InferenceSession(str(path), sess_options=opts, providers=["CPUExecutionProvider"])
        inp = self.session.get_inputs()[0]
        self._input_name = inp.name
        self.input_hw = (int(inp.shape[2]), int(inp.shape[3]))
        if [self.input_hw[1], self.input_hw[0]] != list(cfg.input_size):
            raise DetectorUnavailable(f"{cfg.name} only runs at its fixed {self.input_hw[1]}x{self.input_hw[0]} input; "
                                      f"set input_size [{self.input_hw[1]}, {self.input_hw[0]}]")
        self.cfg = cfg
        self.name = cfg.name
        self._weights = {"path": path.as_posix(), "sha256": sha256_file(path)}
        self._versions = {"onnxruntime": ort.__version__}
        self._threads = opts.intra_op_num_threads

    def detect(self, image: np.ndarray) -> list[Detection]:
        h, w = image.shape[:2]
        ih, iw = self.input_hw
        r = min(ih / h, iw / w)
        nw, nh = int(w * r), int(h * r)
        padded = np.full((ih, iw, 3), 114, dtype=np.uint8)
        padded[:nh, :nw] = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
        blob = np.ascontiguousarray(padded.transpose(2, 0, 1)[None], dtype=np.float32)
        raw = self.session.run(None, {self._input_name: blob})[0][0]
        pred = yolox_decode(raw, self.input_hw)
        scores = pred[:, 4] * pred[:, 5 + self.PERSON_INDEX]
        keep = scores >= self.cfg.min_score
        if not keep.any():
            return []
        cx, cy, bw, bh = (pred[keep, i] for i in range(4))
        boxes = np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], 1) / r
        scores = scores[keep]
        dets = []
        for i in _nms(boxes, scores, self.NMS_IOU):
            x1, y1, x2, y2 = (float(v) for v in boxes[i])
            x1, x2 = max(0.0, min(x1, w)), max(0.0, min(x2, w))
            y1, y2 = max(0.0, min(y1, h)), max(0.0, min(y2, h))
            if x2 - x1 >= 2 and y2 - y1 >= 2:
                dets.append(Detection((x1, y1, x2, y2), float(scores[i])))
        return dets

    def info(self) -> dict:
        return {"detector": self.name, "weights": self._weights, "min_score": self.cfg.min_score,
                "input_size": list(self.cfg.input_size), "resize": "letterboxed into the fixed square input, aspect kept",
                "device": "cpu", "backend": "onnxruntime", "intra_op_threads": self._threads,
                "nms_iou": self.NMS_IOU, **self._versions}


# -- detection cache ---------------------------------------------------------------
#
# Running the detector once per clip and caching its raw output (above the floor)
# lets threshold sweeps and tracker tuning reuse it, and makes evaluation
# deterministic. The cache header records which model produced it; a cache from a
# different model/floor is refused.


def write_detection_cache(path: str | Path, meta: dict, per_frame: dict[int, list[Detection]]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w") as f:
        f.write(json.dumps({"_meta": meta}) + "\n")
        for frame in sorted(per_frame):
            dets = [[*[round(v, 2) for v in d.box], round(d.score, 4)] for d in per_frame[frame]]
            f.write(json.dumps({"frame": frame, "dets": dets}) + "\n")


def read_detection_cache(path: str | Path) -> tuple[dict, dict[int, list[Detection]]]:
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(p)
    meta: dict = {}
    per_frame: dict[int, list[Detection]] = {}
    with p.open() as f:
        for line in f:
            row = json.loads(line)
            if "_meta" in row:
                meta = row["_meta"]
                continue
            per_frame[int(row["frame"])] = [Detection(tuple(d[:4]), float(d[4])) for d in row["dets"]]
    return meta, per_frame


class CachedDetector:
    """Replays cached detections by frame index. Not a model; used for tuning/eval speed."""

    def __init__(self, meta: dict, per_frame: dict[int, list[Detection]]) -> None:
        self.name = f"cache:{meta.get('detector', 'unknown')}"
        self._meta = meta
        self._frames = per_frame

    def detections_for(self, frame_index: int) -> list[Detection]:
        return self._frames.get(frame_index, [])

    def detect(self, image):  # pragma: no cover - cached detectors are queried by index
        raise RuntimeError("CachedDetector is queried by frame index via detections_for()")

    def info(self) -> dict:
        return {**self._meta, "source": "cache"}
