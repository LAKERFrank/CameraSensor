import os
from typing import Optional

import paho.mqtt.client as mqtt
import pandas as pd

from LayerCamera.CameraSystemC.recorder_module import ImageBuffer, Frame
from LayerSensing.Pose.YOLOPoseMqtt import YOLOPoseMqtt
from LayerSensing.PoseDatafeeder import PoseDatafeeder
from lib.common import ROOTDIR

class PoseManager:
    def __init__(self, device_name: str, mqttc: mqtt.Client, imgbuf: ImageBuffer):
        self.deviceName = device_name
        self.mqttc = mqttc
        self.imageBuffer = imgbuf
        self.poseThread = None
        self.feederThread = None

    def startPose(self, weights_filename: str, replay_dirname: str, cam_idx: int,
                  visualize: bool = False):
        try:
            if self.poseThread is not None:
                raise Exception("There is another pose thread running.")

            pose_topic = f"/DATA/{self.deviceName}/LayerSensing/Pose"
            replay_path = f"{ROOTDIR}/replay/{replay_dirname}"
            os.makedirs(replay_path, exist_ok=True)

            self.poseThread = YOLOPoseMqtt(
                f"Pose_{cam_idx}", self.mqttc, pose_topic, replay_path,
                weights_filename, self.imageBuffer, True, visualize)
            self.poseThread.start()
            return {"status": "ready"}
        except Exception as e:
            return {"status": "failure", "message": str(e)}

    def stopPose(self, wait_for_eos: bool = True):
        try:
            if self.poseThread is None:
                raise Exception("No pose is running")

            if not wait_for_eos:
                self.imageBuffer.clear()
                frame = Frame()
                frame.is_eos = True
                self.imageBuffer.push(frame)
            self.poseThread.join()
            self.poseThread = None
            return {"status": "stopped " + ("(EOS reached)" if wait_for_eos else "(Force stop)")}
        except Exception as e:
            return {"status": "failure", "message": str(e)}

    def startDatafeeder(self, filepath: str, metapath: Optional[str] = None):
        """Start a Pose CSV data feeder."""
        self.feederThread = PoseDatafeeder(self.mqttc, self.deviceName,
                                           filepath, metapath)
        self.feederThread.start()

        df = pd.read_csv(filepath)
        duration = float(df.iloc[-1].Timestamp) - float(df.iloc[0].Timestamp)
        return duration

    def stopDatafeeder(self):
        if self.feederThread is not None:
            self.feederThread.join()
            self.feederThread = None
