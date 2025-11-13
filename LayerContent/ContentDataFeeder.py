import paho.mqtt.client as mqtt
import logging
import ipaddress
import json
import signal
import uuid
import time
import csv

from datetime import datetime

from LayerApplication.utils.CameraExplorer import CameraExplorer
from LayerContent.RpcContent import RpcContent

class ExampleAPP():
    
    def __init__(self, broker_ip:str=None, broker_port:str=None):
        self.app_uuid = str(uuid.uuid4())
        self.target_devices_name = "ContentDevice"
        self.has_explored = False
        self.timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        
        self.mqttc = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
        self.mqttc.on_connect = self.on_connect
        self.mqttc.on_message = self.on_message
        
        self.CONTROL_PLANE_QOS = 2
        
        if broker_ip == None:
            # self.mqttc.connect("host.docker.internal")
            # self.mqttc.connect("localhost")
            self.mqttc.connect("140.113.213.131", 1883, 60)
        else:
            try:
                ipaddress.ip_address(broker_ip)
                self.mqttc.connect(broker_ip, broker_port)
            except:
                logging.error("Fail to connect MQTT Broker.")   
        
        self.mqttc.loop_start()
        
        # self.start()
        
    def on_connect(self, client:mqtt.Client, userdata, flags, reason_code, properties):
        
        print(f"Connected with result code {reason_code}")

        self.mqttc.subscribe(f"/DATA/cam_0/SensingLayer/TrackNet")
        self.mqttc.subscribe(f"/DATA/cam_1/SensingLayer/TrackNet")
               
        # if self.has_explored:
        #     self.subscribeLayers()
                
    def on_message(self, client, userdata, msg):
        # logging.warning(f"{__class__}.{__name__} unhandled message from MQTT topic: {msg.topic}, payload: {msg.payload[:20]}...")
        
        print(f"[ContentLayer] Reveived on topic '{msg.topic}': {json.loads(msg.payload)}")
        
    def subscribeLayers(self):
        for device_name in self.target_devices_name:
            self.mqttc.subscribe(f"/RETURN/{device_name}/CameraLayer/+")
            self.mqttc.subscribe(f"/RETURN/{device_name}/SensingLayer/+")
            self.mqttc.subscribe(f"/RETURN/{device_name}/ContentLayer/+")
            
    def start(self):
        camera_explore = CameraExplorer(self.mqttc)
        camera_explore.explore(app_uuid = self.app_uuid, date = self.timestamp)
        self.target_devices_name = camera_explore.choose_target_cameras()
        
        self.subscribeLayers()
        
    def stop(self):
        print("Stopping application...")
        try:
                self.mqttc.loop_stop()
                self.mqttc.disconnect()
                print("Application stopped.")
        except Exception as e:
            logging.error(f"Error while stopping applicatipn: {e}")
    

    def run(self):
        output_topic = '/DATA/ContentDevice/ContentLayer/Model3D'
        file_path = '/workspaces/camerasensor/LayerContent/content_output_sample.csv'
        with open(file_path, 'r', newline='') as csvFile:
            rows = list(csv.DictReader(csvFile))
            for row in rows:
                json_str = {'shot_id': int(row['ShotID']), 
                            'timestamp': float(row['Timestamp']) if row['Timestamp'] else None,
                            'event': int(row['Event']) if row['Event'] else None,
                            'position': json.loads(row['Position']) if row['Position'] else None,
                            'speed': json.loads(row['Speed']) if row['Speed'] else None
                            }
                self.mqttc.publish(output_topic, json.dumps(json_str))
                print(f"[ContentLayer] Published on topic '{output_topic}': {json_str}")

        
if __name__ == "__main__":
    app = ExampleAPP()
    
    app.run()
    
    signal.sigwait([signal.SIGTERM, signal.SIGINT])
    app.stop()