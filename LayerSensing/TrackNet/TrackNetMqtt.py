"""
TrackNet10 : find out the ball on frame on 2D Coordinate System
"""
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

from lib.common import insertById, loadConfig, loadNodeConfig, ROOTDIR
from lib.point import Point, removeOutliers
from lib.writer import CSVWriter

HEIGHT = 288
WIDTH = 512
TRACK_SIZE = 10

class BoundingBox():
    def __init__(self, left, top, right, bottom):
        self.left = left
        self.top = top
        self.right = right
        self.bottom = bottom

def isBlacklist(boxs, x, y):
    for box in boxs:
        if x >= box.left and x <= box.right and y >= box.top and y <= box.bottom:
            return True
    return False

def generateBlacklist(filename):
    blacklist = []
    with open(filename) as f:
        line = f.readline()
        while line:
            line = f.readline()
            l = line.split(',')
            if len(l) == 4:
                b = BoundingBox(int(l[0]), int(l[1]), int(l[2]), int(l[3]))
                blacklist.append(b)
    return blacklist

class TrackNetThread:
    def __init__(self, device:str, model:TrackNet10, mqttc:mqtt.Client,
                 data_handler, blacklist,
                 output_width:int, output_height:int,
                 csv_writer:CSVWriter):

        self.images = []
        self.fids = []
        self.timestamps = []
        self.device = device
        self.model = model
        self.points:'list[Point]' = []
        self.blacklist = blacklist
        self.output_width = output_width
        self.output_height = output_height
        self.csv_writer = csv_writer
        self.mqttc = mqttc
        self.data_handler = data_handler
        # wait for new image
        self.isProcessing = False

    def init(self):
        self.images = []
        self.fids = []
        self.timestamps = []
        self.points:'list[Point]' = []

    def prediction(self):
        # TrackNet prediction
        unit = np.stack(self.images, axis=2)
        unit = np.moveaxis(unit, -1, 0).astype('float32')/255
        unit = torch.from_numpy(np.asarray([unit])).to(self.device)
        with torch.no_grad():
            self.h_pred = self.model(unit)
        self.h_pred = self.h_pred > 0.5
        self.h_pred = self.h_pred.cpu().numpy()
        self.h_pred = self.h_pred.astype('uint8')
        self.h_pred = self.h_pred[0]*255

    def execute(self):
        # TrackNet
        for idx_f, (image, fid, timestamp) in enumerate(zip(self.images, self.fids, self.timestamps)):
            # height, width, channels = image.shape
            ratio_w = self.output_width / WIDTH
            ratio_h = self.output_height / HEIGHT
            #show = np.copy(image)
            #show = cv2.resize(show, (width, height))
            # Ball tracking
            if np.amax(self.h_pred[idx_f]) <= 0: # no ball
                point = Point(fid=fid, timestamp=timestamp, visibility=0, x=0, y=0, z=0, event=0, speed=0)
            else:
                (cnts, _) = cv2.findContours(self.h_pred[idx_f].copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                rects = [cv2.boundingRect(ctr) for ctr in cnts]
                max_area_idx = 0
                max_area = rects[max_area_idx][2] * rects[max_area_idx][3]
                for i in range(len(rects)):
                    area = rects[i][2] * rects[i][3]
                    if area > max_area:
                        max_area_idx = i
                        max_area = area
                target = rects[max_area_idx]
                (cx_pred, cy_pred) = (int(ratio_w*(target[0] + target[2] / 2)), int(ratio_h*(target[1] + target[3] / 2)))
                #logging.info("id: {} timestamp: {}, (x, y): ({}, {})".format(fid, timestamp, cx_pred, cy_pred))
                if (not isBlacklist(self.blacklist, cx_pred, cy_pred)):
                    point = Point(fid=fid, timestamp=timestamp, visibility=1, x=cx_pred, y=cy_pred, z=0, event=0, speed=0)
                    insertById(self.points, point)
                else:
                    point = Point(fid=fid, timestamp=timestamp, visibility=0, x=0, y=0, z=0, event=0, speed=0)

            if self.csv_writer is not None:
                self.csv_writer.writePoints(point)

        if self.mqttc is not None:
            self.points = removeOutliers(self.points)
            self._publishPoints()

    def _publishPoints(self):
        payload = {"linear": [p.toJson() for p in self.points]}
        self.data_handler.publish("tracknet", json.dumps(payload))

    def run(self):
        #logging.debug("TrackNetThread started.")
        try:
            if len(self.images) == TRACK_SIZE:
                self.isProcessing = True
                self.prediction()
                self.execute()
            else:
                logging.warning("images size {} is not correctly. ".format(len(self.images)))
        except Exception as e:
            logging.error(e)
        finally:
            self.isProcessing = False

        #logging.debug("TrackNetThread terminated.")

class TrackNetMqtt(threading.Thread):
    def __init__(self, nodename, mqttc:mqtt.Client, data_handler,
                 camera_origin_width:int, camera_origin_height:int,
                 path: str, weights_filename: str,
                 imgbuf: ImageBuffer, save_csv=True):
        """_summary_

        Args:
            path (str): path to video directory
            nodename (str): _description_
            weights_filename (str): _description_
            save_csv (bool, optional): _description_. Defaults to True.
        """
        threading.Thread.__init__(self)

        self.imageBuffer = imgbuf

        self.camera_origin_width = camera_origin_width
        self.camera_origin_height = camera_origin_height

        # GPU device
        logging.info("GPU Use : {}".format(torch.cuda.is_available()))
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.device = device

        self._loadModel(weights_filename)

        self.nodename = nodename

        self.data_handler = data_handler
        self.mqttc = mqttc

        # Setup CSV Writer
        if save_csv:
            path = os.path.join(path, self.nodename+'.csv')
            self.csv_writer = CSVWriter(name=self.nodename, filename=path)
        else:
            self.csv_writer = None

        self._stopper = threading.Event()

    def stop(self):
        self._stopper.set()  

    def _stopped(self):
        return self._stopper.is_set()

    def _loadModel(self, weights_filename:str):
        ### Track Net Model ###
        model = TrackNet10()
        model.to(self.device)

        weights = f"{ROOTDIR}/LayerSensing/TrackNet/TrackNet10/weights/{weights_filename}"
        checkpoint = torch.load(weights)
        model.load_state_dict(checkpoint['state_dict'])
        model.eval()
        self.model = model

    def run(self):
        size = 0
        list_images = []
        list_fids = []
        list_timestamps = []

        # TrackNet
        tracknetThread = TrackNetThread(device=self.device,
                                        model=self.model,
                                        mqttc=self.mqttc,
                                        data_handler=self.data_handler,
                                        blacklist=[],
                                        output_width=self.camera_origin_width,
                                        output_height=self.camera_origin_height,
                                        csv_writer=self.csv_writer)

        while not self._stopped():

            while (not self._stopped()) and size < TRACK_SIZE:
                frame = self.imageBuffer.pop(True)
                if frame.is_eos:
                    if self.data_handler is not None:
                        self.data_handler.publish("tracknet", json.dumps({"linear": [], "EOF": True}))
                    logging.info(f"{self.nodename} EOF reached.")
                    self.stop()
                    break
                list_images.append(frame.image)
                list_fids.append(frame.index)
                list_timestamps.append(frame.monotonic_timestamp)
                size += 1

            if self._stopped():
                break

            # run tracknet
            tracknetThread.init()
            tracknetThread.images = list_images.copy()
            tracknetThread.fids = list_fids.copy()
            tracknetThread.timestamps = list_timestamps.copy()
            tracknetThread.run()

            list_images.clear()
            list_fids.clear()
            list_timestamps.clear()
            size = 0
        
        if self.csv_writer is not None:
            self.csv_writer.close()

        logging.info("{} terminated.".format(self.nodename))
