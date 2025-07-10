import logging

from LayerApplication.utils.Mqtt import MqttClient
from lib.common import ROOTDIR, loadConfig
from LayerApplication.Rpc.RpcStreamingBadminton import RpcStreamingBadminton
from LayerApplication.Rpc.RpcManager import RpcManager

logging.getLogger().setLevel(logging.INFO)

cfg = loadConfig(f"{ROOTDIR}/config")

mqtt = MqttClient(cfg["Project"]["mqtt_broker"], 1883)

rpcm = RpcManager(mqtt.mqttc)
rpcm.setDeviceNamesFromConfig(cfg)
rpcm.open()

badminton = RpcStreamingBadminton(rpcm)
badminton.test()

#badminton.start()
#time.sleep(3)
#badminton.stop()