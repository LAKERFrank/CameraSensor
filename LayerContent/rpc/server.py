import paho.mqtt.client as mqtt
import threading
import signal
import argparse
import json
import pickle
import re
import subprocess
import numpy as np

from lib.GracefulKiller import GracefulKiller
from lib.common import ROOTDIR, loadConfig
from lib.MqttAgent import MqttAgent
from LayerContent.Model3D_mqtt import MainThreadManager
  
class ContentLayerAgent(MqttAgent):
    def __init__(self, device_name:str):
        super().__init__(device_name, "ContentLayer")
        self.Model3DManager = MainThreadManager(self.mqttc)
        
    def on_connect(self, client:mqtt.Client, userdata, flags, reason_code, properties):
        super().on_connect(client, userdata, flags, reason_code, properties)

        super()._add_func_callback(self.Model3DManager.start_main_thread, 'Model3D/Start')
        super()._add_func_callback(self.Model3DManager.stop_main_thread, 'Model3D/Stop')
        super()._add_func_callback(self.Model3DManager.data_feeder, 'Model3D/Feeder')


if __name__ == "__main__":

    cfg = loadConfig(f"{ROOTDIR}/config")

    broker_ip = cfg["Project"]["mqtt_broker"]
    broker_port = int(cfg["Project"]["mqtt_port"])

    client = ContentLayerAgent("ContentDevice")
    client.start(broker_ip, broker_port)

    GracefulKiller().wait()

    print("stopping...")
    client.stop()



