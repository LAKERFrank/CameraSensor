import argparse
import json
import logging
import os
import threading
import time
from datetime import datetime
from pathlib import Path

from LayerApplication.utils.Mqtt import MqttClient
from LayerCamera.camera.RpcCamera import RpcCamera
from LayerSensing.RpcSensing import RpcSensing
from lib.common import ROOTDIR, loadConfig

logging.getLogger().setLevel(logging.INFO)

BLUE = "\033[94m"
ORANGE = "\033[38;5;208m"
RESET = "\033[0m"


def parse_args():
    parser = argparse.ArgumentParser(description="Offline content testing app")
    parser.add_argument("--device", default="CameraReader_0", help="device name")
    parser.add_argument("--tracknet-ver", default="tracknet_1000", choices=["tracknet_1000", "tracknet_v2"])
    parser.add_argument("--tracknet-weight", default="best.pt", help="tracknet weight filename")
    parser.add_argument("--pose-engine", default="pose_int8_minmax.engine", help="pose engine filename")
    parser.add_argument("--camera-width", type=int, default=640)
    parser.add_argument("--camera-height", type=int, default=480)
    parser.add_argument("--video", default=None, help="video path for feeder")
    parser.add_argument("--replay", default=None, help="replay folder name")
    return parser.parse_args()


def _resolve_feeder_video(video_arg: str | None) -> Path:
    env_video = video_arg or os.environ.get("VIDEO_PATH") or os.environ.get("FEEDER_VIDEO")
    if env_video:
        candidate = Path(env_video).expanduser()
        if not candidate.is_absolute():
            candidate = (Path(ROOTDIR) / candidate).resolve()
        if candidate.exists():
            return candidate
        raise FileNotFoundError(f"Specified video not found: {candidate}")

    replay_root = Path(ROOTDIR) / "replay"
    candidates = sorted(
        replay_root.glob("*/CameraReader_0.mp4"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if candidates:
        return candidates[0]

    raise FileNotFoundError("No replay video found. Set --video or VIDEO_PATH/FEEDER_VIDEO.")


def main():
    args = parse_args()
    cfg = loadConfig(f"{ROOTDIR}/config")

    mqtt = MqttClient(cfg["Project"]["mqtt_broker"], int(cfg["Project"]["mqtt_port"]))
    camera = RpcCamera(args.device, mqtt.mqttc)
    sensing = RpcSensing(args.device, mqtt.mqttc)

    replay_dirname = args.replay or os.environ.get("REPLAY_DIRNAME") or datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    replay_path = Path(ROOTDIR) / "replay" / replay_dirname
    replay_path.mkdir(parents=True, exist_ok=True)

    tracknet_topic = f"/DATA/{args.device}/SensingLayer/TrackNet"
    pose_topic = f"/DATA/{args.device}/SensingLayer/Pose"
    pose_path = replay_path / "Pose_0.jsonl"
    pose_lock = threading.Lock()

    def _print_topic(topic: str, payload_text: str):
        ts = datetime.now().strftime("%H:%M:%S")
        if topic.endswith("/TrackNet"):
            print(f"{BLUE}[{ts}] [TRACKNET]{RESET} {payload_text}")
        elif topic.endswith("/Pose"):
            print(f"{ORANGE}[{ts}] [POSE]{RESET} {payload_text}")

    def _tracknet_logger(client, userdata, msg):
        try:
            payload = msg.payload.decode("utf-8")
        except Exception:
            payload = str(msg.payload)
        _print_topic(msg.topic, payload)

    def _pose_logger(client, userdata, msg):
        try:
            payload = msg.payload.decode("utf-8")
            json.loads(payload)
        except Exception:
            return

        _print_topic(msg.topic, payload)
        with pose_lock:
            with pose_path.open("a", encoding="utf-8") as fp:
                fp.write(payload)
                fp.write("\n")

    mqtt.mqttc.message_callback_add(tracknet_topic, _tracknet_logger)
    mqtt.mqttc.subscribe(tracknet_topic)
    mqtt.mqttc.message_callback_add(pose_topic, _pose_logger)
    mqtt.mqttc.subscribe(pose_topic)

    logging.info("Replay output folder: %s", replay_path)

    ret = sensing.startTrackNet(
        (args.camera_width, args.camera_height),
        args.tracknet_ver,
        args.tracknet_weight,
        replay_dirname,
        0,
    )
    print(f"TrackNet: {ret}")

    pose_ret = sensing._call_rpc_sync("Pose/start", engine_filename=args.pose_engine)
    print(f"Pose: {pose_ret}")

    feeder_video = _resolve_feeder_video(args.video)
    logging.info("Using feeder video: %s", feeder_video)

    duration = camera.startVideoFeeder(str(feeder_video))
    time.sleep(duration)

    camera.stopVideoFeeder()
    sensing.stopTrackNet()
    sensing._call_rpc_sync("Pose/stop")

    mqtt.mqttc.message_callback_remove(tracknet_topic)
    mqtt.mqttc.unsubscribe(tracknet_topic)
    mqtt.mqttc.message_callback_remove(pose_topic)
    mqtt.mqttc.unsubscribe(pose_topic)
    mqtt.stop()


if __name__ == "__main__":
    main()
