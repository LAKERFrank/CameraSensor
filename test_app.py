import json
import logging
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
# replay_dirname = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
replay_dirname = "2025-11-18_11-41-07"

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

video_path = replay_path / "CameraReader_0.mp4"
tracknet_csv = replay_path / "TrackNet_0.csv"
pose_file = pose_path if pose_path.exists() else None

try:
    output_path = visualize(video_path, tracknet_csv, pose_path=pose_file)
    print(f"Visualization saved to: {output_path}")
except Exception as exc:
    logging.error(f"Failed to visualize TrackNet/Pose outputs: {exc}")
