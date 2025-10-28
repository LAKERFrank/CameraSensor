import paho.mqtt.client as mqtt

from lib.MqttAgent import MqttAgent

from LayerCamera.CameraSystemC.recorder_module import ImageBuffer
from LayerSensing.TrackNetManager import TrackNetManager

class SensingLayerAgent(MqttAgent):
    def __init__(self, device_name:str, imgbuf:ImageBuffer):
        super().__init__(device_name, "SensingLayer")

        self.tracknetManager = TrackNetManager(device_name, self.data_handler, self.mqttc, imgbuf)

    def on_connect(self, client:mqtt.Client, userdata, flags, reason_code, properties):
        super().on_connect(client, userdata, flags, reason_code, properties)
        
        # 註冊 Sensing Layer 所提供的服務、格式化MQTT Topic(可設定suffix，預設使用function name)
        # 填入的這個function應該要有return，回傳 簡短的單次答案 / API開啟狀態(Status Code, Error Msg).etc
        self.control_handler.register_function(self.tracknetManager.startTrackNet, "TrackNet/start")
        self.control_handler.register_function(self.tracknetManager.stopTrackNet, "TrackNet/stop")    
        self.control_handler.register_function(self.tracknetManager.startDatafeeder, "TrackNet/startDatafeeder")
        self.control_handler.register_function(self.tracknetManager.stopDatafeeder, "TrackNet/stopDatafeeder")

        # 註冊 Sensing Layer 所發布的資料流的Topic，未來可以透過較短的稱呼(第一個參數)索引到Topic的全稱
        self.data_handler.register_topic("tracknet", "TrackNet")