import paho.mqtt.client as mqtt
import json
import time
import threading

from LayerCamera.camera.RpcCamera import RpcCamera
from lib.MqttAgent import MqttAgent
from LayerCamera.camera.Camera import Camera

class MetricUpdateThread(threading.Thread):
    def __init__(self, mqttc:mqtt.Client, device_name:str, camera:Camera):
        threading.Thread.__init__(self)
        self.mqttc = mqttc
        self.camera = camera
        self.deviceName = device_name
        self.stopEvent = threading.Event()

    def stop(self):
        self.stopEvent.set()

    def run(self):
        while not self.stopEvent.isSet():
            time.sleep(0.5)
            if self.camera.isStreaming():
                metric = self.camera.getMetricData()
                kf = metric.kf.copy()
                payload = json.dumps({
                    "fps": metric.fps,
                    "avg_fps": metric.avg_fps,
                    "frames_rendered": metric.frames_rendered,
                    "frames_dropped": metric.frames_dropped,
                    "kf_timestamp": kf[0],
                    "kf_dt": kf[1]
                })
                self.mqttc.publish(f"/DATA/{self.deviceName}/CameraLayer/Metrics", payload)

class HeartbeatUpdateThread(threading.Thread):

    UPDATE_DURATION=15 # seconds

    def __init__(self, mqttc:mqtt.Client, device_name:str, camera:Camera):
        threading.Thread.__init__(self)
        self.mqttc = mqttc
        self.camera = camera
        self.deviceName = device_name
        self.stopEvent = threading.Event()

    def stop(self):
        self.stopEvent.set()

    def run(self):
        while not self.stopEvent.isSet():
            if self.camera.isStreaming():
                metric = self.camera.getMetricData()
                kf = metric.kf.copy()
                payload = json.dumps({
                    "kf_timestamp": kf[0],
                    "kf_dt": kf[1]
                })
                self.mqttc.publish(f"/DATA/{self.deviceName}/CameraLayer/Heartbeat", payload)
            time.sleep(self.UPDATE_DURATION)

class CameraLayerAgent(MqttAgent):

    def __init__(self, device_name:str, serial:str):
        super().__init__(device_name, "CameraLayer")

        self.camera = Camera(serial, device_name)

    def on_connect(self, client:mqtt.Client, userdata, flags, reason_code, properties):
        super().on_connect(client, userdata, flags, reason_code, properties)

        client.message_callback_add("/CALL/AllCamera/CameraLayer/getCameraStatus", self.on_machine_message)

        client.subscribe('/CALL/AllCamera/CameraLayer/getCameraStatus')
        print("\033[94m Subscribed /CALL/AllCamera/CameraLayer/getCameraStatus\033[0m")

        # 填入的這個function應該要有return，回傳 簡短的單次答案 / API開啟狀態(Status Code, Error Msg).etc
        super()._add_func_callback(self.camera.init)
        super()._add_func_callback(self.camera.start)
        super()._add_func_callback(self.camera.getSnapshot)
        super()._add_func_callback(self.camera.release)
        super()._add_func_callback(self.camera.getCameraParameters)
        super()._add_func_callback(self.camera.setCameraParameters)
        super()._add_func_callback(self.camera.getCaptureFormats)
        super()._add_func_callback(self.camera.setCaptureFormat)
        super()._add_func_callback(self.camera.getDeviceInfo)
        super()._add_func_callback(self.camera.startRecording)
        super()._add_func_callback(self.camera.stopRecording)
        #super()._add_func_callback(self.camera.getFile)
        super()._add_func_callback(self.camera.startVideoFeeder)
        super()._add_func_callback(self.camera.stopVideoFeeder)
        super()._add_func_callback(self.camera.getIntrinsic)
        super()._add_func_callback(self.camera.setIntrinsic)
        super()._add_func_callback(self.camera.getExtrinsic)
        super()._add_func_callback(self.camera.setExtrinsic)
        super()._add_func_callback(self.camera.clip)
        super()._add_func_callback(self.camera.startUdp)
        super()._add_func_callback(self.camera.stopUdp)
        super()._add_func_callback(self.camera.isStreaming)
        super()._add_func_callback(self.camera.resync)
        super()._add_func_callback(lambda : self.camera.getMetricData().serialize(), "getMetricData")

    def start(self, broker_ip = None, broker_port = 1883):
        self.metricUpdateThread = MetricUpdateThread(self.mqttc, self.device_name, self.camera)
        self.metricUpdateThread.start()
        self.heartbeatUpdateThread = HeartbeatUpdateThread(self.mqttc, self.device_name, self.camera)
        self.heartbeatUpdateThread.start()

        return super().start(broker_ip, broker_port)

    def stop(self):
        self.metricUpdateThread.stop()
        self.metricUpdateThread.join()
        self.heartbeatUpdateThread.stop()
        self.heartbeatUpdateThread.join()
        return super().stop()

    def on_machine_message(self, client, userdata, msg):
        received_msg = json.loads(msg.payload)
        print(f"[CameraLayer] Reveived on topic '{msg.topic}': {received_msg}")

        payload = json.dumps({
            "name": self.device_name,
            "app_uuid": received_msg['kwargs'].get('app_uuid'),
            "cameraInfo": {
                "serial": self.camera.serial,
            }
        })
        return_topic = f"/RETURN/{self.device_name}/CameraLayer/getCameraStatus"
        client.publish(return_topic, payload, self.CONTROL_PLANE_QOS)

        print(f"[{self.layer}] Published to topic '{return_topic}")
