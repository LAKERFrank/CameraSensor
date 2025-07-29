import logging
import time
from datetime import datetime

from LayerApplication.utils.Mqtt import MqttClient
from lib.common import ROOTDIR, loadConfig
from LayerApplication.Rpc.RpcStreamingBadminton import RpcStreamingBadminton
from LayerApplication.Rpc.RpcManager import RpcManager
from LayerCamera.camera.RpcCamera import RpcCamera
from LayerSensing.RpcSensing import RpcSensing

logging.getLogger().setLevel(logging.INFO)

cfg = loadConfig(f"{ROOTDIR}/config")

mqtt = MqttClient(cfg["Project"]["mqtt_broker"], 1885)

camera = RpcCamera("test-0", mqtt.mqttc)

sensing = RpcSensing("test-0", mqtt.mqttc)

# save dir
replay_dirname = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def startTask(use_tracknet: bool = False, use_pose: bool = False):
    if use_tracknet:
        ret = sensing.startTrackNet(
            (640, 480),
            "tracknet_v2",
            "no114_30.tar",
            replay_dirname,
            0,
            visualize=True,
        )
        print(f"TrackNet: {ret}")
    if use_pose:
        ret = sensing.startPose(
            "yolov8n-pose-gray-train.pt", replay_dirname, 0, visualize=True
        )
        print(f"Pose: {ret}")


startTask(use_tracknet=True, use_pose=True)

duration = camera.startVideoFeeder(f"{ROOTDIR}/replay/test_video/1_01_00.mp4")

time.sleep(duration)

camera.stopVideoFeeder()
sensing.stopTrackNet()
sensing.stopPose()
