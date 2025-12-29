import json
import logging
import os
import threading
import time
from datetime import datetime
from pathlib import Path

from LayerApplication.utils.Mqtt import MqttClient
from lib.common import ROOTDIR, loadConfig
from LayerApplication.Rpc.RpcStreamingBadminton import RpcStreamingBadminton
from LayerApplication.Rpc.RpcManager import RpcManager
from LayerCamera.camera.RpcCamera import RpcCamera
from LayerSensing.RpcSensing import RpcSensing
from Tools.tracknet_pose_visualize import visualize

logging.getLogger().setLevel(logging.INFO)

cfg = loadConfig(f"{ROOTDIR}/config")

mqtt = MqttClient(cfg["Project"]["mqtt_broker"], 1883)

camera = RpcCamera("CameraReader_0", mqtt.mqttc)

sensing = RpcSensing("CameraReader_0", mqtt.mqttc)

device_name = "CameraReader_0"

# save dir
replay_dirname = os.environ.get("REPLAY_DIRNAME") or datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

replay_path = Path(ROOTDIR) / "replay" / replay_dirname
replay_path.mkdir(parents=True, exist_ok=True)

pose_topic = f"/DATA/{device_name}/SensingLayer/Pose"
pose_path = replay_path / "Pose_0.jsonl"
pose_lock = threading.Lock()


def _pose_logger(client, userdata, msg):
    try:
        payload = msg.payload.decode("utf-8")
        json.loads(payload)
    except Exception:
        return

    with pose_lock:
        with pose_path.open("a", encoding="utf-8") as fp:
            fp.write(payload)
            fp.write("\n")

# ret = sensing.startTrackNet((640, 480), "tracknet_v2", "no114_30.tar", replay_dirname, 0)
ret = sensing.startTrackNet((640, 480), "tracknet_1000", "best.pt", replay_dirname, 0)
print(f"TrackNet: {ret}")

# Launch pose inference with a TensorRT engine.
pose_ret = sensing.startPose("pose_int8_minmax.engine", 0)
print(f"Pose: {pose_ret}")

mqtt.mqttc.message_callback_add(pose_topic, _pose_logger)
mqtt.mqttc.subscribe(pose_topic)

duration = camera.startVideoFeeder(f"{ROOTDIR}/replay/{replay_dirname}/CameraReader_0.mp4")

time.sleep(duration)

camera.stopVideoFeeder()
sensing.stopTrackNet()
sensing.stopPose()

mqtt.mqttc.message_callback_remove(pose_topic)
mqtt.mqttc.unsubscribe(pose_topic)

def _wait_for_file(path: Path, *, timeout: float = 60.0, min_size: int = 1) -> bool:
    """Wait for a file to exist, reach a minimum size, and stop growing."""

    deadline = time.time() + timeout
    last_size = -1
    stable_checks = 0
    while time.time() < deadline:
        if path.exists():
            size = path.stat().st_size
            if size >= min_size:
                if size == last_size:
                    stable_checks += 1
                else:
                    stable_checks = 0
                last_size = size
                if stable_checks >= 2:  # stable for ~1s
                    logging.info("File ready: %s (size=%d)", path, size)
                    return True
            else:
                logging.debug("Waiting for %s to reach %d bytes (current=%d)", path, min_size, size)
        time.sleep(0.5)
    if path.exists():
        logging.warning("Timeout waiting for %s to stabilize (final size=%d)", path, path.stat().st_size)
    return False


video_path = replay_path / "CameraReader_0.mp4"
tracknet_csv = replay_path / "TrackNet_0.csv"

if not video_path.exists():
    logging.error("Video file not found: %s", video_path)
else:
    logging.info("Waiting for TrackNet CSV at %s", tracknet_csv)
    has_csv = _wait_for_file(tracknet_csv, timeout=90.0)
    has_pose = _wait_for_file(pose_path, timeout=10.0, min_size=10)

    if not has_csv:
        logging.error("TrackNet CSV not found; skip visualization: %s", tracknet_csv)
    else:
        pose_file = pose_path if has_pose else None
        if pose_file:
            logging.info("Using pose log: %s", pose_file)
        else:
            logging.info("Pose log missing or empty; visualizing TrackNet only")
        try:
            output_path = visualize(video_path, tracknet_csv, pose_path=pose_file)
            print(f"Visualization saved to: {output_path}")
        except Exception as exc:
            logging.error("Failed to visualize TrackNet/Pose outputs: %s", exc)
