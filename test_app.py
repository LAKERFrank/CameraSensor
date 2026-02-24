import argparse
import logging
import time
from datetime import datetime

from LayerApplication.utils.Mqtt import MqttClient
from lib.common import ROOTDIR, loadConfig
from LayerCamera.camera.RpcCamera import RpcCamera
from LayerSensing.RpcSensing import RpcSensing

logging.getLogger().setLevel(logging.INFO)


def parse_args():
    parser = argparse.ArgumentParser(description="Offline camera -> sensing -> content MQTT test")
    parser.add_argument("--device", default="test-0", help="MQTT device name for camera+sensing agent")
    parser.add_argument("--broker", default=None, help="MQTT broker host, default from config")
    parser.add_argument("--port", type=int, default=None, help="MQTT broker port, default from config")
    parser.add_argument("--video", default=f"{ROOTDIR}/replay/origin_court/CameraReader_0.mp4", help="video path for startVideoFeeder")
    parser.add_argument("--tracknet_ver", default="tracknet_v2")
    parser.add_argument("--tracknet_weights", default="no114_30.tar")
    parser.add_argument("--pose_engine", default="int8.engine")
    parser.add_argument("--cam_idx", type=int, default=0)
    parser.add_argument("--camera_width", type=int, default=640)
    parser.add_argument("--camera_height", type=int, default=480)
    return parser.parse_args()


def main():
    args = parse_args()

    cfg = loadConfig(f"{ROOTDIR}/config")
    broker = args.broker or cfg["Project"]["mqtt_broker"]
    port = args.port or int(cfg["Project"]["mqtt_port"])

    mqtt = MqttClient(broker, port)

    camera = RpcCamera(args.device, mqtt.mqttc)
    sensing = RpcSensing(args.device, mqtt.mqttc)

    replay_dirname = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    ret = sensing.startTrackNet((args.camera_width, args.camera_height), args.tracknet_ver, args.tracknet_weights, replay_dirname, args.cam_idx)
    print(f"TrackNet: {ret}")

    pose_ret = sensing.startPose((args.camera_width, args.camera_height), args.pose_engine, replay_dirname, args.cam_idx)
    print(f"Pose: {pose_ret}")

    duration = camera.startVideoFeeder(args.video)

    time.sleep(duration)

    camera.stopVideoFeeder()
    sensing.stopPose()
    sensing.stopTrackNet()
    mqtt.stop()


if __name__ == "__main__":
    main()
