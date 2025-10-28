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
        self.Model3DManager = MainThreadManager(self.data_handler, self.mqttc)
        
    def on_connect(self, client:mqtt.Client, userdata, flags, reason_code, properties):
        super().on_connect(client, userdata, flags, reason_code, properties)
        
        # 註冊 Content Layer 所提供的服務、格式化MQTT Topic(可設定suffix，預設使用function name)
        self.control_handler.register_function(self.Model3DManager.start_main_thread, 'Model3D/Start')
        self.control_handler.register_function(self.Model3DManager.stop_main_thread, 'Model3D/Stop')
        self.control_handler.register_function(self.Model3DManager.data_feeder, 'Model3D/Feeder')

        # 註冊 Content Layer 所發布的資料流的Topic，未來可以透過較短的稱呼(第一個參數)索引到Topic的全稱
        self.data_handler.register_topic("point", "Model3D/Point")
        self.data_handler.register_topic("event", "Model3D/Event")
        self.data_handler.register_topic("segment", "Model3D/Segment")
        # 註1：
            # 對於開發者而言要發布資料可以用 data_handler.publish("point", payload=<要發布的payload>)
            # 便會自動發布到 "/DATA/{device_name}/{layer}/Model3D/Point" 這個Topic
            # 其中{device_name}/{layer}會自動填入，以免輸入錯誤

if __name__ == "__main__":

    cfg = loadConfig(f"{ROOTDIR}/config")

    broker_ip = cfg["Project"]["mqtt_broker"]
    broker_port = int(cfg["Project"]["mqtt_port"])

    client = ContentLayerAgent("ContentDevice")
    client.start(broker_ip, broker_port)

    GracefulKiller().wait()

    print("stopping...")
    client.stop()



