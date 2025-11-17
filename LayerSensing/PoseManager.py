"""Manager for TensorRT-based pose estimation services."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import paho.mqtt.client as mqtt

from LayerCamera.CameraSystemC.recorder_module import ImageBuffer
from lib.common import ROOTDIR

from LayerSensing.Pose.pose_mqtt import PoseMqtt

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
        fallback_weights: Optional[str] = None,
    ):
        try:
            if self.pose_thread is not None:
                raise RuntimeError("Pose worker is already running")

            engine_path = Path(engine_filename)
            if not engine_path.is_absolute():
                engine_path = Path(ROOTDIR) / "LayerSensing" / "Pose" / "engine" / engine_filename
            engine_path = engine_path.resolve()

            engine_dir = Path(ROOTDIR) / "LayerSensing" / "Pose" / "engine"

            def _resolve_fallback(spec: str) -> Optional[str]:
                """Resolve a fallback weight specification to an absolute path if possible."""

                candidate = Path(spec)
                if candidate.is_absolute() and candidate.is_file():
                    return str(candidate.resolve())

                if not candidate.is_absolute():
                    local_candidate = (engine_dir / candidate).resolve()
                    if local_candidate.is_file():
                        return str(local_candidate)
                    if candidate.is_file():
                        return str(candidate.resolve())

                # Returning None tells the caller to fall back to the raw spec.
                return None

            fallback_path: Optional[str] = None
            if fallback_weights:
                resolved = _resolve_fallback(fallback_weights)
                fallback_path = resolved if resolved else fallback_weights
            else:
                # Auto-discover local PyTorch weights before falling back to public checkpoints.
                pt_candidates = sorted(engine_dir.glob("*.pt")) + sorted(engine_dir.glob("*.pth"))
                if pt_candidates:
                    fallback_path = str(pt_candidates[0])
                    LOGGER.info(
                        "Using pose fallback weights discovered at %s", fallback_path
                    )
                else:
                    fallback_path = "yolov8n-pose.pt"
                    LOGGER.info(
                        "No local pose fallback weights found; defaulting to %s", fallback_path
                    )

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
                fallback_weights=fallback_path,
            )
            self.pose_thread.start()
            LOGGER.info(
                "Pose worker started for camera %s using %s (fallback=%s)",
                cam_idx,
                engine_path,
                fallback_path or "disabled",
            )
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


__all__ = ["PoseManager"]
