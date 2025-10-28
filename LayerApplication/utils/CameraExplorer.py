import paho.mqtt.client as mqtt
import json
import threading
import time    

class CameraExplorer():
    def __init__(self, mqtt_client: mqtt.Client, app_uuid: str):
        self.mqttc = mqtt_client
        self.avaliable_devices = []
        self.app_uuid = app_uuid
        
        self.CONTROL_PLANE_QOS = 2
        
    def explore(self, timeout=2, *args, **kwargs):
        self.avaliable_devices = []
        
        kwargs["app_uuid"] = self.app_uuid

        payload = json.dumps({"args":args, "kwargs":kwargs})
        
        call_topic = "/CALL/AllCamera/CameraLayer/getCameraStatus"
        return_topic = "/RETURN/+/CameraLayer/getCameraStatus"
        
        lock = threading.Event()
        
        def callback(client, userdata, msg):
            res = json.loads(msg.payload)
            if res['app_uuid'] == self.app_uuid:
                self.avaliable_devices.append(res)
                lock.set()
            
        self.mqttc.message_callback_add(return_topic, callback)
        self.mqttc.subscribe(return_topic)
        print(f"\033[94m Subscribed {return_topic}\033[0m")
        
        self.mqttc.publish(call_topic, payload, qos=self.CONTROL_PLANE_QOS)
        print(f"[ApplicationLayer] Published to topic '{call_topic}")
        
        time.sleep(timeout)
        
        self.mqttc.message_callback_remove(return_topic)
        self.mqttc.unsubscribe(return_topic)
            
        #if not self.avaliable_devices:
        #    raise Exception("Timeout exceeded, no responses received.")

    def isDeviceExists(self, name):
        return name in [d["name"] for d in self.avaliable_devices]

    def getAvailableDevice(self):
        return self.avaliable_devices

    def getAvailableDeviceName(self):
        return [d['name'] for d in self.avaliable_devices]
            
    def choose_target_cameras(self):
        self.list_all_camera()
        choices = input(">> ").split()
        
        target_cameras = []
        for choice in choices:
            target_cameras.append(self.avaliable_devices[int(choice)]['name'])
        return target_cameras
    
    def list_all_camera(self):
        print("==List all devices==")
        print("------------------")
        
        for i, device_info in enumerate(self.avaliable_devices):
            print(f"- {i} name: {device_info['name']} | cameraInfo: {device_info['cameraInfo']}")
            print("------------------")