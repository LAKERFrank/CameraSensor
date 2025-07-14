import paho.mqtt.client as mqtt

from lib.MqttAgent import MqttAgent

from LayerCamera.CameraSystemC.recorder_module import ImageBuffer
from LayerSensing.TrackNetManager import TrackNetManager
from LayerSensing.PoseManager import PoseManager

class SensingLayerAgent(MqttAgent):
    def __init__(self, device_name:str, imgbuf:ImageBuffer):
        super().__init__(device_name, "SensingLayer")

        self.tracknetManager = TrackNetManager(device_name, self.mqttc, imgbuf)
        self.poseManager = PoseManager(device_name, self.mqttc, imgbuf)
        
    def on_connect(self, client:mqtt.Client, userdata, flags, reason_code, properties):
        super().on_connect(client, userdata, flags, reason_code, properties)
        
        # 填入的這個function應該要有return，回傳 簡短的單次答案 / API開啟狀態(Status Code, Error Msg).etc
        super()._add_func_callback(self.tracknetManager.startTrackNet, "TrackNet/start")
        super()._add_func_callback(self.tracknetManager.stopTrackNet, "TrackNet/stop")    
        super()._add_func_callback(self.tracknetManager.startDatafeeder, "TrackNet/startDatafeeder")
        super()._add_func_callback(self.tracknetManager.stopDatafeeder, "TrackNet/stopDatafeeder")
        super()._add_func_callback(self.poseManager.startPose, "Pose/start")
        super()._add_func_callback(self.poseManager.stopPose, "Pose/stop")
