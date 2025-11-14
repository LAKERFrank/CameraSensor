"""Convenience wrapper mirroring the TrackNet MQTT predictor interface."""
from __future__ import annotations

from typing import Any

from LayerCamera.CameraSystemC.recorder_module import ImageBuffer

from .pose_worker import PoseWorker


class PoseMqtt(PoseWorker):
    """Drop-in replacement for TrackNet MQTT threads that performs pose inference."""

    def __init__(
        self,
        nodename: str,
        mqtt_client: Any,
        data_handler: Any,
        image_buffer: ImageBuffer,
        engine_path: str,
        *,
        camera_index: int,
        input_size: int = 640,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.65,
        max_det: int = 100,
        fallback_weights: str | None = None,
    ) -> None:
        super().__init__(
            nodename,
            image_buffer,
            data_handler,
            mqtt_client,
            engine_path,
            camera_index=camera_index,
            input_size=input_size,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
            max_det=max_det,
            fallback_weights=fallback_weights,
        )


__all__ = ["PoseMqtt"]
