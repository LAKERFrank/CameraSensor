import paho.mqtt.client as mqtt
import json
import time
import threading

from LayerCamera.camera.RpcCamera import RpcCamera
from lib.MqttAgent import MqttAgent
from LayerCamera.camera.Camera import Camera

class MetricUpdateThread(threading.Thread):
    def __init__(self, data_handler, camera:Camera):
        threading.Thread.__init__(self)
        self.data_handler = data_handler
        self.camera = camera
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
                self.data_handler.publish("metrics", payload)

class HeartbeatUpdateThread(threading.Thread):

    UPDATE_DURATION=15 # seconds

    def __init__(self, data_handler, camera:Camera):
        threading.Thread.__init__(self)
        self.data_handler = data_handler
        self.camera = camera
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
                self.data_handler.publish("heartbeat", payload)
            time.sleep(self.UPDATE_DURATION)

class CameraLayerAgent(MqttAgent):

    def __init__(self, device_name:str, serial:str):
        super().__init__(device_name, "CameraLayer")

        self.camera = Camera(serial, device_name)

    def on_connect(self, client:mqtt.Client, userdata, flags, reason_code, properties):
        super().on_connect(client, userdata, flags, reason_code, properties)
        self.control_handler.register_custom_call("/CALL/AllCamera/CameraLayer/getCameraStatus", self.on_machine_message)

        # 註冊 Camera Layer 所提供的服務
        # 填入的這個function應該要有return，回傳 簡短的單次答案 / API開啟狀態(Status Code, Error Msg).etc
        self.control_handler.register_function(self.camera.init)
        self.control_handler.register_function(self.camera.start)
        self.control_handler.register_function(self.camera.getSnapshot)
        self.control_handler.register_function(self.camera.release)
        self.control_handler.register_function(self.camera.getCameraParameters)
        self.control_handler.register_function(self.camera.setCameraParameters)
        self.control_handler.register_function(self.camera.getCaptureFormats)
        self.control_handler.register_function(self.camera.setCaptureFormat)
        self.control_handler.register_function(self.camera.getDeviceInfo)
        self.control_handler.register_function(self.camera.startRecording)
        self.control_handler.register_function(self.camera.stopRecording)
        # self.control_handler.register_function(self.camera.getFile)
        self.control_handler.register_function(self.camera.startVideoFeeder)
        self.control_handler.register_function(self.camera.stopVideoFeeder)
        self.control_handler.register_function(self.camera.getIntrinsic)
        self.control_handler.register_function(self.camera.setIntrinsic)
        self.control_handler.register_function(self.camera.getExtrinsic)
        self.control_handler.register_function(self.camera.setExtrinsic)
        self.control_handler.register_function(self.camera.clip)
        self.control_handler.register_function(self.camera.startUdp)
        self.control_handler.register_function(self.camera.stopUdp)
        self.control_handler.register_function(self.camera.isStreaming)
        self.control_handler.register_function(self.camera.resync)
        self.control_handler.register_function(lambda : self.camera.getMetricData().serialize(), "getMetricData")
        
        # 註冊 Camera Layer 所發布的資料流的Topic [DATA]，未來可以透過較短的稱呼(第一個參數)索引到Topic的全稱
        self.data_handler.register_topic("metrics", "Metrics")
        self.data_handler.register_topic("heartbeat", "Heartbeat")
        # 註：
            # 對於開發者而言要發布資料可以用 data_handler.publish("metrics", payload=<要發布的payload>)
            # 便會自動發布到 "/DATA/{device_name}/{layer}/Model3D/Point" 這個Topic
            # 其中{device_name}/{layer}會自動填入，以免輸入錯誤s

    def start(self, broker_ip = None, broker_port = 1883):
        self.metricUpdateThread = MetricUpdateThread(self.data_handler, self.camera)
        self.metricUpdateThread.start()
        self.heartbeatUpdateThread = HeartbeatUpdateThread(self.data_handler, self.camera)
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
        client.publish(return_topic, payload, qos=2)

        print(f"[{self.layer}] Published to topic '{return_topic}")
