import logging
import time
from datetime import datetime
from pathlib import Path

from LayerApplication.utils.Mqtt import MqttClient
from lib.common import ROOTDIR, loadConfig
from LayerApplication.Rpc.RpcStreamingBadminton import RpcStreamingBadminton
from LayerApplication.Rpc.RpcManager import RpcManager
from LayerCamera.camera.RpcCamera import RpcCamera
from LayerSensing.RpcSensing import RpcSensing

logging.getLogger().setLevel(logging.INFO)

cfg = loadConfig(f"{ROOTDIR}/config")


def _resolve_pose_engine(config) -> str:
    if config.has_section("Pose"):
        engine = config["Pose"].get("engine", "").strip()
        if engine:
            return engine

    engine_dir = Path(ROOTDIR) / "LayerSensing" / "Pose" / "engine"
    candidates = sorted(engine_dir.glob("*.engine"))
    if not candidates:
        raise FileNotFoundError(
            f"No TensorRT engine found in {engine_dir}. "
            "Configure the engine filename under the [Pose] section in config."
        )
    return candidates[0].name


def _pose_runtime_kwargs(config):
    kwargs = {}
    if not config.has_section("Pose"):
        return kwargs

    section = config["Pose"]
    if section.get("input_size"):
        kwargs["input_size"] = int(section.get("input_size"))
    if section.get("conf_threshold"):
        kwargs["conf_threshold"] = float(section.get("conf_threshold"))
    if section.get("iou_threshold"):
        kwargs["iou_threshold"] = float(section.get("iou_threshold"))
    if section.get("max_det"):
        kwargs["max_det"] = int(section.get("max_det"))
    return kwargs

mqtt = MqttClient(cfg["Project"]["mqtt_broker"], 1883)

camera = RpcCamera("test-0", mqtt.mqttc)

sensing = RpcSensing("test-0", mqtt.mqttc)

# save dir
replay_dirname = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

tracknet_started = False
pose_started = False
video_started = False

try:
    ret = sensing.startTrackNet((640, 480), "tracknet_v2", "no114_30.tar", replay_dirname, 0)
    print(f"TrackNet: {ret}")
    if ret.get("status") != "ready":
        raise RuntimeError(f"TrackNet failed to start: {ret}")
    tracknet_started = True

    pose_engine = _resolve_pose_engine(cfg)
    pose_kwargs = _pose_runtime_kwargs(cfg)
    pose_ret = sensing.startPose(pose_engine, 0, **pose_kwargs)
    print(f"Pose: {pose_ret}")
    if pose_ret.get("status") != "ready":
        raise RuntimeError(f"Pose failed to start: {pose_ret}")
    pose_started = True

    duration = camera.startVideoFeeder(f"{ROOTDIR}/replay/origin_court/CameraReader_0.mp4")
    video_started = True
    time.sleep(duration)
finally:
    if video_started:
        camera.stopVideoFeeder()
    if pose_started:
        sensing.stopPose()
    if tracknet_started:
        sensing.stopTrackNet()
