import os
from LayerSensing.TrackNet.Tracknet1000.Predict import TrackNet1000Mqtt
import paho.mqtt.client as mqtt
import time
import pandas as pd

from LayerCamera.CameraSystemC.recorder_module import ImageBuffer, Frame
from LayerSensing.TrackNet.TrackNetMqtt import TrackNetMqtt
from LayerSensing.Datafeeder import Datafeeder
from lib.common import ROOTDIR

class TrackNetManager:
    def __init__(self, device_name, data_handler, mqttc:mqtt.Client, imgbuf:ImageBuffer):

        self.deviceName = device_name
        self.data_handler = data_handler        
        self.mqttc = mqttc
        self.imageBuffer = imgbuf
        self.tracknetThread = None

    def startTrackNet(self, camera_origin_size: 'tuple[int, int]',
                      tracknet_ver:str, weights_filename: str,
                      replay_dirname: str, cam_idx: int):
        """Start Tracknet thread
           會存在 PROJECT_ROOT/replay/{replay_dirname}/TrackNet_{cam_idx}.csv

        Args:
            camera_origin_size (tuple[int, int]): 相機原始解析度 (Tracknet會回推)
            tracknet_ver (str): Tracknet版本 ("tracknet_v2" or "tracknet_1000")
            weights_filename (str): 模型檔案名稱
            replay_dirname (str): 儲存的資料夾名稱 
            cam_idx (int): 相機編號

        Returns:
            dict: 狀態
        """

        try:

            if self.tracknetThread is not None:
                raise Exception("There is another Tracknet thread is running.")

            # tracknet_topic = f"/DATA/{self.deviceName}/SensingLayer/TrackNet"

            replay_path = f"{ROOTDIR}/replay/{replay_dirname}"

            os.makedirs(replay_path, exist_ok=True)

            if tracknet_ver == "tracknet_v2":
                self.tracknetThread \
                    = TrackNetMqtt(f"TrackNet_{cam_idx}", self.mqttc, self.data_handler, camera_origin_size[0],
                                   camera_origin_size[1], replay_path, weights_filename,
                                   self.imageBuffer, True)
            elif tracknet_ver == "tracknet_1000":
                self.tracknetThread \
                    = TrackNet1000Mqtt(f"TrackNet_{cam_idx}", self.mqttc, self.data_handler, camera_origin_size[0],
                                   camera_origin_size[1], replay_path, weights_filename,
                                   self.imageBuffer, True)
            else:
                raise Exception(f"tracknet_ver={tracknet_ver} is not acceptable.")
            self.tracknetThread.start()
            return {"status": "ready"}
        except Exception as e:
            return {"status": "failure", "message": str(e)}

    def stopTrackNet(self, wait_for_eos=True):
        try:
            if self.tracknetThread is None:
                raise Exception("No tracknet is running")

            if not wait_for_eos:
                self.imageBuffer.clear()
                frame = Frame()
                frame.is_eos = True
                self.imageBuffer.push(frame)
            self.tracknetThread.join()
            self.tracknetThread = None
            return {"status": "stopped " + ("(EOS reached)" if wait_for_eos else "(Force stop)")}
        except Exception as e:
            return {"status": "failure", "message": str(e)}

    def startDatafeeder(self, filepath, metapath=None):
        self.feederThread = Datafeeder(self.mqttc, self.deviceName, filepath, metapath)
        self.feederThread.start()

        df = pd.read_csv(filepath)
        duration = float(df.iloc[-1].Timestamp) - float(df.iloc[0].Timestamp)

        return duration

    def stopDatafeeder(self):
        self.feederThread.join()
