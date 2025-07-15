import paho.mqtt.client as mqtt

from lib.MqttAgent import MqttAgent

from LayerCamera.CameraSystemC.recorder_module import ImageBuffer
from LayerSensing.TrackNetManager import TrackNetManager
from LayerSensing.PoseManager import PoseManager
from LayerSensing.FrameDistributor import FrameDistributor

class SensingLayerAgent(MqttAgent):
    def __init__(self, device_name:str, imgbuf: ImageBuffer):
        super().__init__(device_name, "SensingLayer")

        # create internal buffers for tracknet and pose
        self._src_buf = imgbuf
        self._tracknet_buf = ImageBuffer()
        self._pose_buf = ImageBuffer()
        # distribute frames from the camera to TrackNet and Pose
        # Pose only receives every 4th frame
        self._distributor = FrameDistributor(self._src_buf,
                                             self._tracknet_buf,
                                             self._pose_buf,
                                             pose_interval=4)

        self.tracknetManager = TrackNetManager(device_name, self.mqttc, self._tracknet_buf)
        self.poseManager = PoseManager(device_name, self.mqttc, self._pose_buf)

    def start(self, broker_ip: str = None, broker_port: int = 1883):
        self._distributor.start()
        return super().start(broker_ip, broker_port)

    def stop(self):
        self._distributor.stop()
        self._distributor.join()
        return super().stop()
        
    def on_connect(self, client:mqtt.Client, userdata, flags, reason_code, properties):
        super().on_connect(client, userdata, flags, reason_code, properties)
        
        # 填入的這個function應該要有return，回傳 簡短的單次答案 / API開啟狀態(Status Code, Error Msg).etc
        super()._add_func_callback(self.tracknetManager.startTrackNet, "TrackNet/start")
        super()._add_func_callback(self.tracknetManager.stopTrackNet, "TrackNet/stop")    
        super()._add_func_callback(self.tracknetManager.startDatafeeder, "TrackNet/startDatafeeder")
        super()._add_func_callback(self.tracknetManager.stopDatafeeder, "TrackNet/stopDatafeeder")
        super()._add_func_callback(self.poseManager.startPose, "Pose/start")
        super()._add_func_callback(self.poseManager.stopPose, "Pose/stop")
