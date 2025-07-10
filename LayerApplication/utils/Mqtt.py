import paho.mqtt.client as mqtt
import logging
import ipaddress
import json
import uuid

from datetime import datetime

class MqttClient:
    def __init__(self, broker_ip:str=None, broker_port:str=None):
        self.app_uuid = str(uuid.uuid4())
        self.timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        
        self.mqttc = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
        self.mqttc.on_connect = self.on_connect
        self.mqttc.on_message = self.on_message
        
        self.CONTROL_PLANE_QOS = 2
        
        if broker_ip == None:
            self.mqttc.connect("host.docker.internal")
        else:
            try:
                #ipaddress.ip_address(broker_ip)
                self.mqttc.connect(broker_ip, broker_port)
            except:
                logging.error("Fail to connect MQTT Broker.")   
        
        self.mqttc.loop_start()
        
    def on_connect(self, client:mqtt.Client, userdata, flags, reason_code, properties):
        
        print(f"Connected with result code {reason_code}")
                
    def on_message(client, userdata, msg):
        logging.warning(f"{__class__}.{__name__} unhandled message from MQTT topic: {msg.topic}, payload: {msg.payload[:20]}...")
        
        print(f"[ApplicationLayer] Reveived on topic '{msg.topic}': {json.loads(msg.payload)}")

    def stop(self):
        print("Stopping application...")
        try:
             self.mqttc.loop_stop()
             self.mqttc.disconnect()
             print("Application stopped.")
        except Exception as e:
           logging.error(f"Error while stopping applicatipn: {e}")

    def getClient(self):
        return self.mqttc

    def getAppId(self):
        return self.app_uuid
