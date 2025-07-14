import os
import threading
import json
import logging

import cv2
import paho.mqtt.client as mqtt
from ultralytics import YOLO
from ultralytics.yolo.v8.pose.predict import PosePredictor

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
        # initialize predictor manually so we can warm up with the correct channel count
        self.model.predictor = PosePredictor()
        self.model.predictor.setup_model(model=self.model.model, verbose=False)
        # determine expected input channels from first layer weights
        try:
            m = self.model.model.model[0]
            self.expected_ch = m.conv.in_channels if hasattr(m, 'conv') else getattr(m, 'in_channels', 3)
        except Exception:
            self.expected_ch = 3
        logging.info(f"{self.nodename} expected_ch={self.expected_ch}")
        # warmup with one-frame input matching expected channels
        self.model.predictor.model.warmup(imgsz=(1, self.expected_ch, 640, 640))
        self.model.predictor.done_warmup = True
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
            img = frame.image
            # ensure shape matches model expectation
            if self.expected_ch == 3:
                if img.ndim == 2:
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            elif self.expected_ch == 1:
                if img.ndim == 3 and img.shape[2] == 3:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                if img.ndim == 2:
                    img = img[..., None]  # expand to (H, W, 1) for predictor
            results = self.model(img, verbose=False)
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
