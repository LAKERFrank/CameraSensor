"""Worker thread that consumes frames and publishes TensorRT pose results."""
from __future__ import annotations

import json
import logging
import threading
from typing import Any, Dict

from LayerCamera.CameraSystemC.recorder_module import Frame, ImageBuffer

from .tensorrt_engine import PoseInferenceResult, TensorRTPoseEngine

LOGGER = logging.getLogger(__name__)


class PoseWorker(threading.Thread):
    """Consumes frames from an :class:`ImageBuffer` and publishes pose estimates."""

    def __init__(
        self,
        nodename: str,
        image_buffer: ImageBuffer,
        data_handler: Any,
        _mqtt_client: Any,
        engine_path: str,
        *,
        camera_index: int,
        target_fps: float = 30.0,
        input_size: int = 640,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.65,
        max_det: int = 100,
        ) -> None:
        super().__init__(daemon=True)
        self.nodename = nodename
        self.image_buffer = image_buffer
        self.data_handler = data_handler
        self.camera_index = camera_index
        self.stop_event = threading.Event()
        base_camera_fps = 120.0
        self.target_fps = target_fps
        self.frame_stride = max(1, round(base_camera_fps / target_fps)) if target_fps > 0 else 1
        self.effective_fps = base_camera_fps / self.frame_stride

        self.engine = TensorRTPoseEngine(
            engine_path,
            input_shape=(3, input_size, input_size),
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
            max_det=max_det,
        )

    # ------------------------------------------------------------------
    def run(self) -> None:
        LOGGER.info(
            "%s pose worker started (target_fps=%s, stride=%s -> ~%.2f fps)",
            self.nodename,
            self.target_fps,
            self.frame_stride,
            self.effective_fps,
        )
        try:
            while not self.stop_event.is_set():
                frame = self.image_buffer.pop(True)
                if frame is None:
                    continue
                if frame.is_eos:
                    LOGGER.info("%s pose worker received EOS", self.nodename)
                    break
                if frame.index % self.frame_stride != 0:
                    continue
                try:
                    result = self.engine.predict(frame.image)
                    payload = self._format_payload(frame, result)
                    self.data_handler.publish("pose", json.dumps(payload))
                except Exception as exc:  # pragma: no cover - defensive logging
                    LOGGER.exception("Pose inference failed: %s", exc)
        finally:
            self.engine.close()
            LOGGER.info("%s pose worker terminated", self.nodename)

    # ------------------------------------------------------------------
    def request_stop(self, wait_for_eos: bool) -> None:
        self.stop_event.set()
        if not wait_for_eos:
            eos = Frame()
            eos.is_eos = True
            self.image_buffer.push(eos)

    # ------------------------------------------------------------------
    def _format_payload(self, frame: Frame, result: PoseInferenceResult) -> Dict[str, Any]:
        detections = [
            {
                "bbox": detection.bbox,
                "score": detection.score,
                "class_id": detection.class_id,
                "keypoints": detection.keypoints,
            }
            for detection in result.detections
        ]
        timings = {k: round(v, 3) for k, v in result.timings_ms.items()}
        payload = {
            "camera_index": self.camera_index,
            "frame_index": frame.index,
            "timestamp": frame.timestamp,
            "monotonic_timestamp": frame.monotonic_timestamp,
            "image_size": [frame.width, frame.height],
            "timings_ms": timings,
            "detections": detections,
        }
        return payload


__all__ = ["PoseWorker"]
