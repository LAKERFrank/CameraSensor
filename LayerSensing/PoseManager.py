"""Manager for TensorRT-based pose estimation services."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import paho.mqtt.client as mqtt

from LayerCamera.CameraSystemC.recorder_module import ImageBuffer
from lib.common import ROOTDIR

from LayerSensing.Pose.pose_mqtt import PoseMqtt
from LayerSensing.Pose.datafeeder import PoseDatafeeder

LOGGER = logging.getLogger(__name__)


class PoseManager:
    """Control lifecycle of pose inference threads."""

    def __init__(
        self,
        device_name: str,
        data_handler,
        mqtt_client: mqtt.Client,
        imgbuf: ImageBuffer,
    ) -> None:
        self.device_name = device_name
        self.data_handler = data_handler
        self.mqtt_client = mqtt_client
        self.image_buffer = imgbuf
        self.pose_thread: Optional[PoseMqtt] = None
        self.pose_feeder: Optional[PoseDatafeeder] = None

    # ------------------------------------------------------------------
    def startPose(
        self,
        engine_filename: str,
        cam_idx: int,
        *,
        input_size: int = 640,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.65,
        max_det: int = 100,
    ):
        try:
            if self.pose_thread is not None:
                raise RuntimeError("Pose worker is already running")

            engine_path = Path(engine_filename)
            if not engine_path.is_absolute():
                engine_path = Path(ROOTDIR) / "LayerSensing" / "Pose" / "engine" / engine_filename
            engine_path = engine_path.resolve()

            self.pose_thread = PoseMqtt(
                f"Pose_{cam_idx}",
                self.mqtt_client,
                self.data_handler,
                self.image_buffer,
                str(engine_path),
                camera_index=cam_idx,
                input_size=input_size,
                conf_threshold=conf_threshold,
                iou_threshold=iou_threshold,
                max_det=max_det,
            )
            self.pose_thread.start()
            LOGGER.info("Pose worker started for camera %s using %s", cam_idx, engine_path)
            return {"status": "ready"}
        except Exception as exc:
            LOGGER.exception("Unable to start pose worker: %s", exc)
            self.pose_thread = None
            return {"status": "failure", "message": str(exc)}

    # ------------------------------------------------------------------
    def stopPose(self, wait_for_eos: bool = True):
        try:
            if self.pose_thread is None:
                raise RuntimeError("Pose worker is not running")

            self.pose_thread.request_stop(wait_for_eos)
            self.pose_thread.join()
            self.pose_thread = None
            LOGGER.info("Pose worker stopped")
            suffix = "(EOS reached)" if wait_for_eos else "(Force stop)"
            return {"status": f"stopped {suffix}"}
        except Exception as exc:
            LOGGER.exception("Unable to stop pose worker: %s", exc)
            return {"status": "failure", "message": str(exc)}

    # ------------------------------------------------------------------
    def startDatafeeder(self, filepath: str, *, playback_speed: float = 1.0):
        try:
            if self.pose_feeder is not None:
                raise RuntimeError("Pose datafeeder is already running")

            feeder = PoseDatafeeder(
                self.data_handler,
                self.device_name,
                filepath,
                playback_speed=playback_speed,
            )
            self.pose_feeder = feeder
            feeder.start()
            LOGGER.info(
                "Pose datafeeder started using %s (duration: %s s)",
                filepath,
                "unknown" if feeder.duration is None else f"{feeder.duration:.2f}",
            )
            payload = {"status": "ready", "frames": feeder.entry_count}
            if feeder.duration is not None:
                payload["duration"] = feeder.duration
            return payload
        except Exception as exc:
            LOGGER.exception("Unable to start pose datafeeder: %s", exc)
            self.pose_feeder = None
            return {"status": "failure", "message": str(exc)}

    # ------------------------------------------------------------------
    def stopDatafeeder(self):
        try:
            if self.pose_feeder is None:
                raise RuntimeError("Pose datafeeder is not running")

            self.pose_feeder.stop()
            self.pose_feeder.join()
            self.pose_feeder = None
            LOGGER.info("Pose datafeeder stopped")
            return {"status": "stopped"}
        except Exception as exc:
            LOGGER.exception("Unable to stop pose datafeeder: %s", exc)
            return {"status": "failure", "message": str(exc)}


__all__ = ["PoseManager"]
