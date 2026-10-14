"""Detection on a worker thread, so frames the detector does not run on are never blocked by
one it does (pipeline.async_detection).

One worker, at most one frame in flight. A detector frame is copied and handed to the worker;
the main thread carries on predicting every frame. When the result comes back, possibly
several frames later, it is applied by REPLAY: the tracker state saved just before that frame
is updated with the detections at the frame they belong to, then predicted forward to now.
Events already emitted on predicted boxes are not rewritten (they are marked as predicted).
If the worker is still busy at the next detector frame, that detection is skipped and counted:
the frame itself is still processed.

torch and ONNX Runtime release the GIL while they run, so the worker really runs alongside
the main thread.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor

import numpy as np

from sightline.detector import Detector
from sightline.types import Detection


class DetectionWorker:
    def __init__(self, detector: Detector, clock: Callable[[], float] = time.perf_counter) -> None:
        self.detector = detector
        self.clock = clock
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="sightline-detect")
        self._pending: tuple[int, Future] | None = None

    @property
    def idle(self) -> bool:
        """True if a new frame can be submitted (nothing in flight and nothing waiting to be collected)."""
        return self._pending is None

    def submit(self, frame_index: int, image: np.ndarray) -> bool:
        if self._pending is not None:
            return False
        img = image.copy()  # the main thread may draw on its copy (overlay) while the worker reads this one
        self._pending = (frame_index, self._pool.submit(self._run, img))
        return True

    def _run(self, image: np.ndarray) -> tuple[list[Detection], float]:
        t0 = self.clock()
        dets = self.detector.detect(image)
        return dets, self.clock() - t0

    def poll(self, wait: bool = False) -> tuple[int, list[Detection], float] | None:
        """(frame index, detections, inference seconds) once the in-flight frame is done; detector errors re-raise."""
        if self._pending is None:
            return None
        frame_index, fut = self._pending
        if not wait and not fut.done():
            return None
        self._pending = None
        dets, infer_s = fut.result()
        return frame_index, dets, infer_s

    def close(self) -> None:
        self._pool.shutdown(wait=True, cancel_futures=True)
