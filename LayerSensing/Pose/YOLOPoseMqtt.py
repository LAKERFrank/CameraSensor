import os
import threading
import json
import logging

import cv2
import paho.mqtt.client as mqtt
from ultralytics import YOLO

from LayerCamera.CameraSystemC.recorder_module import ImageBuffer, Frame
from lib.writer import CSVWriter
from lib.point import Point
from lib.common import ROOTDIR

class YOLOPoseMqtt(threading.Thread):
    def __init__(self, nodename: str, mqttc: mqtt.Client, output_topic: str,
                 path: str, weights_filename: str, image_buffer: ImageBuffer,
                 save_csv: bool = True):
        super().__init__()
        self.nodename = nodename
        self.mqttc = mqttc
        self.output_topic = output_topic
        self.image_buffer = image_buffer
        weight_path = os.path.join(ROOTDIR, 'weights', weights_filename)
        self.model = YOLO(weight_path)
        if save_csv:
            os.makedirs(path, exist_ok=True)
            csv_path = os.path.join(path, f"{self.nodename}.csv")
            self.csv_writer = CSVWriter(name=self.nodename, filename=csv_path)
        else:
            self.csv_writer = None
        self._stopper = threading.Event()

    def stop(self):
        self._stopper.set()

    def _stopped(self):
        return self._stopper.is_set()

    def _publish(self, points):
        payload = {"linear": [p.toJson() for p in points]}
        self.mqttc.publish(self.output_topic, json.dumps(payload))

    def run(self):
        logging.info(f"{self.nodename} start processing...")
        while not self._stopped():
            frame = self.image_buffer.pop(True)
            if frame.is_eos:
                break
            results = self.model(frame.image, verbose=False)
            points = []
            for r in results:
                if r.keypoints is None:
                    continue
                # take first keypoint of first detection as example
                kp = r.keypoints.xy[0][0]
                p = Point(fid=frame.index, timestamp=frame.monotonic_timestamp,
                           visibility=1, x=float(kp[0]), y=float(kp[1]), z=0, event=0)
                points.append(p)
                if self.csv_writer:
                    self.csv_writer.writePoints(p)
            if points and self.mqttc is not None:
                self._publish(points)
        if self.csv_writer:
            self.csv_writer.close()
        if self.mqttc is not None:
            self.mqttc.publish(self.output_topic, json.dumps({"linear": [], "EOF": True}))
        logging.info(f"{self.nodename} terminated.")
