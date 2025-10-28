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

mqtt = MqttClient(cfg["Project"]["mqtt_broker"], 1883)

camera = RpcCamera("test-0", mqtt.mqttc)

sensing = RpcSensing("test-0", mqtt.mqttc)

# save dir
replay_dirname = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

ret = sensing.startTrackNet((640, 480), "tracknet_v2", "no114_30.tar", replay_dirname, 0)
print(f"TrackNet: {ret}")

duration = camera.startVideoFeeder(f"{ROOTDIR}/replay/origin_court/CameraReader_0.mp4")

time.sleep(duration)

camera.stopVideoFeeder()
sensing.stopTrackNet()
