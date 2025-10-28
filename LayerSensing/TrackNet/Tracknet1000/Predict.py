"""
TrackNet10 : find out the ball on frame on 2D Coordinate System
"""
from datetime import datetime
import logging
import os
import threading
import json

import cv2
import numpy as np
import torch
import paho.mqtt.client as mqtt
from LayerCamera.CameraSystemC.recorder_module import ImageBuffer
from LayerCamera.HPCameraWidget import HPCameraWidget

from LayerSensing.TrackNet.TrackNet10.TrackNet10 import TrackNet10

from LayerSensing.TrackNet.Tracknet1000 import ImageBufferDataset, ImageBufferPredictor
from lib.common import insertById, loadConfig, loadNodeConfig, ROOTDIR
from lib.point import Point, removeOutliers
from lib.writer import CSVWriter

class TrackNet1000Mqtt(threading.Thread):
    def __init__(self, nodename, mqttc:mqtt.Client, data_handler,
                 camera_origin_width:int, camera_origin_height:int,
                 path: str, weights_filename: str,
                 imgbuf: ImageBuffer, save_csv=True):
        """
        初始化 TrackNet1000 MQTT 推論執行緒。

        Args:
            nodename (str): 節點名稱，通常用於識別當前設備或攝影機來源。
            mqttc (mqtt.Client): 已建立連線的 paho-mqtt 用戶端，用於發送推論結果。
            output_topic (str): MQTT 推論結果發送的 topic。
            camera_origin_width (int): 原始攝影機影像的寬度，用於推論後轉換座標等應用。(目前無實作)
            camera_origin_height (int): 原始攝影機影像的高度。(目前無實作)
            path (str): 影片或影像來源路徑，用於載入輸入資料。(目前無實作)
            weights_filename (str): YOLOv8/TrackNet1000 權重檔路徑，用於載入已訓練模型。
            imgbuf (ImageBuffer): 與上游攝影機容器共用的影像 buffer，用於接收影像資料。
            save_csv (bool, optional): 是否儲存推論結果為 CSV 檔案，預設為 True。(目前無實作)
        """
        threading.Thread.__init__(self)

        self.camera_origin_width = camera_origin_width
        self.camera_origin_height = camera_origin_height

        # GPU device
        logging.info("GPU Use : {}".format(torch.cuda.is_available()))
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.device = device

        self.nodename = nodename
        model_path = f"{ROOTDIR}/LayerSensing/TrackNet/TrackNet1000/weights/{weights_filename}.pt"

        # wait for new image
        self.isProcessing = False

        overrides = {}
        overrides['model'] = model_path
        overrides['mode'] = 'predict_v2'
        overrides['data'] = 'tracknet.yaml'
        overrides['batch'] = 1
        self.predictor = ImageBufferPredictor.ImageBufferPredictor(
            weight=model_path,
            image_buffer=imgbuf,
            output_width=camera_origin_width,
            output_height=camera_origin_height,
            mqttc=mqttc,
            data_handler=data_handler,
            overrides=overrides,
            path=path,
        )

    def run(self):
        try:
            logging.info(f"{datetime.now()} {self.nodename} start processing...")
            self.predictor.start()
        except Exception as e:
            logging.error(e)
        finally:
            self.isProcessing = False
        logging.info(f"{self.nodename} EOF reached.")
        logging.info("{} terminated.".format(self.nodename))
