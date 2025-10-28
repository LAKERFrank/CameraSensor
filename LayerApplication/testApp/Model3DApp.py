import paho.mqtt.client as mqtt
import logging
import ipaddress
import json
import signal
import uuid
import time

from datetime import datetime

from LayerApplication.utils.CameraExplorer import CameraExplorer
# from LayerCamera.camera.Rpc_camera import RpcCamera
from LayerContent.RpcContent import RpcContent

from lib.common import ROOTDIR, loadConfig

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
               
        # if self.has_explored:
        #     self.subscribeLayers()
                
    def on_message(self, client, userdata, msg):
        # logging.warning(f"{__class__}.{__name__} unhandled message from MQTT topic: {msg.topic}, payload: {msg.payload[:20]}...")
        
        print(f"[ApplicationLayer] Reveived on topic '{msg.topic}': {json.loads(msg.payload)}")
        
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
    
    def _readCamConfigForModel3D(self, path):
        cfg = loadConfig(path)

        return {
            "ks": eval(cfg["Other"].get("ks", [])),
            "poses": eval(cfg["Other"].get("poses", [])),
            "eye": eval(cfg["Other"].get("eye", [])),
            "hmtx": eval(cfg["Other"].get("hmtx", [])),
            "dist": eval(cfg["Other"].get("dist", [])),
            "newcameramtx": eval(cfg["Other"].get("newcameramtx", [])),
            "projection_mat": eval(cfg["Other"].get("projection_mat", [])),
            "extrinsic_mat": eval(cfg["Other"].get("extrinsic_mat", [])),
        }
    
    def run(self):
        # rpc_camera=RpcCamera(self.target_devices_name[0], self.mqttc)
        # print(rpc_camera.ping())

        rpc_content = RpcContent(self.target_devices_name, self.mqttc)

        self.mqttc.subscribe("/DATA/ContentDevice/ContentLayer/Model3D")

        # date = '2024-09-19_09-33-56'
        # data = [
        #         {'idx': 1, 'device': '39320296', 'fps': 120.0, 'parameters': self._readCamConfigForModel3D(f"{ROOTDIR}/replay/{date}/39320296.cfg")},
        #         {'idx': 2, 'device': '39320299', 'fps': 120.0, 'parameters': self._readCamConfigForModel3D(f"{ROOTDIR}/replay/{date}/39320299.cfg")}
        #     ]
        # mode = ''

        # CES
        date = '2025-01-07_07-23-01'
        data = [
                {'idx': 0, 'device': '16124946', 'fps': 120.0, 'parameters': self._readCamConfigForModel3D(f"{ROOTDIR}/replay/{date}/16124946.cfg")},
                {'idx': 1, 'device': '38224002', 'fps': 120.0, 'parameters': self._readCamConfigForModel3D(f"{ROOTDIR}/replay/{date}/38224002.cfg")}
            ]
        mode = 'CES'

        print(rpc_content.startModel3D(date, data, mode))

        # time.sleep(15)
        # file_path = '/workspaces/camerasensor/LayerContent/content_output_sample.csv'
        # print(rpc_content.dataFeeder(file_path))

        # time.sleep(60)
        # data = 'stop Model3D'
        # print(rpc_content.stopModel3D())

        
        # self.stop()
        
if __name__ == "__main__":
    app = ExampleAPP()
    
    app.run()
    
    signal.sigwait([signal.SIGTERM, signal.SIGINT])
    app.stop()