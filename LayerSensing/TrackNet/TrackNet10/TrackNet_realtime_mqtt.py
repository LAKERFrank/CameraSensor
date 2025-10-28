"""
TrackNet10 : find out the ball on frame on 2D Coordinate System
"""
import argparse
import csv
import json
import logging
import os
import shutil
import sys
import threading
import re
import queue
import time
from typing import Optional

import cv2
import numpy as np
import paho.mqtt.client as mqtt
import torch
from sklearn.metrics import confusion_matrix
from torchvision import transforms
from LayerCamera.HPCameraWidget import HPCameraWidget

# Our System's library
from .TrackNet10 import TrackNet10

# A Python wrapper of libjpeg-turbo for decoding and encoding JPEG image.
#from turbojpeg import TurboJPEG, TJPF_GRAY, TJSAMP_GRAY, TJFLAG_PROGRESSIVE

DIRNAME = os.path.dirname(os.path.abspath(__file__))

from lib.common import insertById, loadConfig, loadNodeConfig, ROOTDIR
from lib.frame import Frame
from lib.point import Point
from lib.writer import CSVWriter
from lib.inspector import sendNodeStateMsg, sendPerformance
from lib.receiver import RawImgReceiver

BATCH_SIZE = 1
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

class TrackNetThread(threading.Thread):
    def __init__(self, nodename, device, model, broker, topic, blacklist, output_width, output_height, csv_writer):
        threading.Thread.__init__(self)

        self.nodename = nodename
        self.images = []
        self.fids = []
        self.timestamps = []
        self.device = device
        self.model = model
        self.points = []
        self.blacklist = blacklist
        self.output_width = output_width
        self.output_height = output_height
        self.csv_writer = csv_writer
        # wait for new image
        self.isProcessing = False

        # setup MQTT client
        client = mqtt.Client()
        client.on_publish = self.on_publish
        client.connect(broker)
        self.client = client
        self.topic = topic

    def on_publish(self, mosq, userdata, mid):
        logging.debug("send")

    def publish_points(self):
        points_left = len(self.points)
        if points_left < 1:
            return

        json_str = '{"linear":['
        fids = []
        for i in range(points_left):
            point = self.points.pop(0)
            fids.append(point.fid)
            json_str += point.build_string()
            if i+1 < points_left:
                json_str += ','
        json_str += ']}'

        # print(self.topic, json_str)
        # print("111111111111111111")
        # print(self.topic)
        self.client.publish(self.topic, json_str)
        sendPerformance(self.client, self.topic, 'none', 'send', fids)

    # def publish_points_csv_version(self, csvFile):
    #     self.csvFile = csvFile
    #     with open(self.csvFile, mode="r") as file:
    #         csv_reader = csv.reader(file)
    #         header = next(csv_reader)  # Skip the header if needed

    #         json_str = '{"linear":'
    #         count = 0
    #         for row in csv_reader:
    #             count += 1
    #             # Convert row to a dictionary if you want to send it as JSON
    #             json_str += row
    #             json_str += ","
                
    #             if(count == 10):
    #                 json_str += "}"
    #                 # Publish data as JSON
    #                 self.client.publish(self.topic, str(json_str))
    #                 json_str = '{"linear":'
    #                 count = 0

    def prediction(self):
        grays = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in self.images]
        # TrackNet prediction
        unit = np.stack(grays, axis=2)
        unit = cv2.resize(unit, (WIDTH, HEIGHT))
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

    def run(self):
        #logging.debug("TrackNetThread started.")
        try:
            if len(self.images) == TRACK_SIZE:
                self.isProcessing = True
                sendPerformance(self.client, self.nodename, 'prediction', 'start', self.fids)
                self.prediction()
                sendPerformance(self.client, self.nodename, 'prediction', 'end', self.fids)
                sendPerformance(self.client, self.nodename, 'tracking', 'start', self.fids)
                self.execute()
                sendPerformance(self.client, self.nodename, 'tracking', 'end', self.fids)
                self.publish_points()
            else:
                logging.warning("images size {} is not correctly. ".format(len(self.images)))
        except Exception as e:
            logging.error(e)
        finally:
            self.isProcessing = False
        
        sendPerformance(self.client, self.nodename, 'total', 'end', self.fids)
        #logging.debug("TrackNetThread terminated.")

class TrackNetRealtime(threading.Thread):
    # def __init__(self, path: str, nodename: str, weights_filename: str, camera_widget: HPCameraWidget, cam_idx: int, testVideo: str, save_csv=True):

    # no camera_widget for testing ContentLayer connection
    def __init__(self, path: str, nodename: str, weights_filename: str, cam_idx: int, testVideo: str, save_csv=False): 
        """_summary_

        Args:
            path (str): path to video directory
            nodename (str): _description_
            weights_filename (str): _description_
            save_csv (bool, optional): _description_. Defaults to True.
            testVideo (str):  _description_
        """
        threading.Thread.__init__(self)

        # self.cameraWidget = camera_widget
        self.cam_idx = cam_idx

        # Load configs
        projectCfg = f"{ROOTDIR}/config"
        settings = loadNodeConfig(projectCfg, nodename)

        # load camera cfg
        filecfg = loadNodeConfig(projectCfg, settings['file_name'])
        cameraCfg = f"{ROOTDIR}/Reader/{filecfg['brand']}/config/{filecfg['hw_id']}.cfg"
        camera_settings = loadConfig(cameraCfg)
    
        # GPU device
        logging.info("GPU Use : {}".format(torch.cuda.is_available()))
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        ### Track Net Model ###
        model = TrackNet10()
        model.to(device)

        weights = f"{DIRNAME}/weights/{weights_filename}"
        checkpoint = torch.load(weights)
        model.load_state_dict(checkpoint['state_dict'])
        model.eval()


        self.nodename = nodename
        self.settings = settings
        self.camera_settings = camera_settings
        self.device = device
        self.model = model

        self.testVideo = testVideo

        # setup mqtt detail
        self.output_topic = settings['output_topic']
        # print("22222222222222222")
        # print(self.output_topic)
        # self.broker = self.settings['mqtt_broker']
        self.broker = "localhost"

        # Setup CSV Writer
        if save_csv:
            path = os.path.join(path, nodename+'.csv')
            self.csv_writer = CSVWriter(name=nodename, filename=path)
        else:
            self.csv_writer = None

        # setup MQTT client
        client = mqtt.Client()
        client.on_connect = self.on_connect
        client.connect(self.broker)
        self.client = client

    def start_thread(self, nodename, broker = "localhost"):
        # self.thread = threading.Thread.__init__(self)

        # GPU device
        logging.info("GPU Use : {}".format(torch.cuda.is_available()))
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Track Net Model
        model = TrackNet10()

        # Output topic
        output_topic = f"/DATA/{self.cam_idx}/SensingLayer/TrackNet"

        #load camera cfg
        projectCfg = f"{ROOTDIR}/config"
        settings = loadNodeConfig(projectCfg, nodename)
        filecfg = loadNodeConfig(projectCfg, settings['file_name'])
        cameraCfg = f"{ROOTDIR}/Reader/{filecfg['brand']}/config/{filecfg['hw_id']}.cfg"
        camera_settings = loadConfig(cameraCfg)
        cam_origin_width,cam_origin_height = [int(i) for i in re.findall(r"\d+",camera_settings['Camera']['RecordResolution'])]

        self.tracknetThread = TrackNetThread(nodename = nodename,
                                            device = device,
                                            model = model,
                                            broker = broker,
                                            topic = output_topic,
                                            blacklist = [],
                                            output_width = cam_origin_width,
                                            output_height = cam_origin_height,
                                            csv_writer = None)
        self.tracknetThread.start()
        time.sleep(3)
        if(self.tracknetThread.is_alive()):
            # self.tracknetThread.stop()
            # self.tracknetThread.join()
            payload = json.dumps({"status": "Ready"})
        else:
            payload = json.dumps({"status": "Fail"})
        return payload

    def stop_thread(self):
        self.tracknetThread.join()
        if(self.tracknetThread.is_alive()):
            payload = json.dumps({"status": "Fail"})
        else:
            payload = json.dumps({"status": "Ready"})
        return payload

    def on_connect(self, client, userdata, flag, rc):
        logging.info(f"{self.nodename} Connected with result code: {str(rc)}")
        self.client.subscribe(self.nodename)
        self.client.subscribe('system_control')

    def on_message(self, client, userdata, msg):
        cmds = json.loads(msg.payload)
        if msg.topic == 'system_control':
            if 'stop' in cmds:
                if cmds['stop'] == True:
                    self.stop()
        else:
            if 'stop' in cmds:
                if cmds['stop'] == True:
                    self.stop()

    def stop(self):
        self.alive = False

    def run(self):
        #logging.debug("{} started.".format(self.nodename))

        # setup tracknetthreads
        tracknetThreads = []
        threads_size = int(self.settings['threads_size'])
        cam_origin_width,cam_origin_height = [int(i) for i in re.findall(r"\d+",self.camera_settings['Camera']['RecordResolution'])]
        #cam_origin_width = int(self.settings['width'])
        #cam_origin_height = int(self.settings['height'])
        # improve encode/decode performance
        #jpeg = TurboJPEG('/usr/lib/x86_64-linux-gnu/libturbojpeg.so.0')

        #logging.info(f"[{threading.get_ident()}] {self.nodename} is ready.")

        self.client.loop_start()
        logging.info(f"{self.nodename} is ready.")
        sendNodeStateMsg(self.client, self.nodename, "ready")

        size = 0
        list_images = []
        list_fids = []
        list_timestamps = []

        stop = False

        # while not stop:

        #     while (not stop) and size < TRACK_SIZE:
        #         frame = self.cameraWidget.popRecordingQueue(self.cam_idx, True)
        #         if frame.is_eos:
        #             print(f"{self.nodename} EOS reached.")
        #             stop = True
        #             break
        #         list_images.append(frame.image)
        #         list_fids.append(frame.index)
        #         list_timestamps.append(frame.timestamp)
        #         size += 1

        #     if stop:
        #         break

        #     # TrackNet
        #     tracknetThread = TrackNetThread(nodename=self.nodename,
        #                                     device=self.device,
        #                                     model=self.model,
        #                                     broker=self.broker,
        #                                     topic=self.output_topic,
        #                                     blacklist=[],
        #                                     output_width=cam_origin_width,
        #                                     output_height=cam_origin_height,
        #                                     csv_writer=self.csv_writer)

        #     tracknetThread.images = list_images.copy()
        #     tracknetThread.fids = list_fids.copy()
        #     tracknetThread.timestamps = list_timestamps.copy()

        #     list_images.clear()
        #     list_fids.clear()
        #     list_timestamps.clear()
        #     size = 0

        #     tracknetThread.start()
        #     tracknetThread.join()


        queue = []
        if os.path.exists(self.testVideo):
            cap = cv2.VideoCapture(self.testVideo)
            logging.debug(f"videofile={self.testVideo} total frames={cap.get(cv2.CAP_PROP_FRAME_COUNT)}")
            try:
                idx = 1
                while cap.isOpened():
                    ret, img = cap.read()
                    if not ret:
                        logging.info(f"{self.testVideo} file is end.")
                        break
                    timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                    frame = Frame(fid=idx, timestamp=timestamp, raw_data=img)
                    queue.append(frame)
                    idx += 1

                    if len(queue) >= TRACK_SIZE and len(tracknetThreads) < threads_size:
                        #logging.debug(f"{self.testVideo} do one predict")
                        tracknetThread = TrackNetThread(nodename=self.nodename,
                                                        device=self.device,
                                                        model=self.model,
                                                        broker=self.broker,
                                                        topic=self.output_topic,
                                                        blacklist=[],
                                                        output_width=cam_origin_width,
                                                        output_height=cam_origin_height,
                                                        csv_writer=self.csv_writer)
                        for i in range(TRACK_SIZE):
                            frame = queue.pop(0)
                            #logging.info("received fid:{}".format(frame.fid))
                            tracknetThread.images.append(frame.raw_data)
                            tracknetThread.fids.append(frame.fid)
                            tracknetThread.timestamps.append(frame.timestamp)
                        tracknetThread.start()
                        tracknetThreads.append(tracknetThread)
                    elif len(tracknetThreads) >= threads_size:
                        for i in range(len(tracknetThreads)):
                            if tracknetThreads[i].is_alive() == True:
                                tracknetThreads[i].join()
                            if tracknetThreads[i].is_alive() == False:
                                del tracknetThreads[i]
                                break

                for i in range(len(tracknetThreads)):
                    tracknetThreads[i].join()
            except Exception as e:
                logging.error(e)

        
        if self.csv_writer is not None:
            self.csv_writer.close()

        # sendNodeStateMsg(self.client, self.nodename, "terminated", self.page)
        self.client.loop_stop()

        logging.info("{} terminated.".format(self.nodename))

def parse_args() -> Optional[str]:
    # Args
    parser = argparse.ArgumentParser(description = 'TrackNet_realtime_mqtt')
    parser.add_argument('--path', type=str, default = '2024-10-28_11-42-21', help = 'project name (default: coachbox)')
    parser.add_argument('--nodename', type=str, default = 'Model3D', help = 'mqtt node name (default: 3DModel)')
    parser.add_argument('--weights_filename', type=str, default = 'no114_30.tar', help = 'csv path (default: replay/XX/)')
    # parser.add_argument('--camera_widget', type=HPCameraWidget, default = camear_widget, required=True, help = 'several cameras configs')
    parser.add_argument('--cam_idx', type=str, default = '0', required=True, help = 'several cameras configs')
    parser.add_argument('--testVideo', type=str, default = './replay/2024-10-17_00-00-19/CameraReader_0.mp4', required=True, help = 'several cameras configs')
    args = parser.parse_args()
    # print("done")

    return args

def main():
    # Parse arguments
    args = parse_args()
    # Start MainThread
    mainThread = TrackNetRealtime(path=args.path, nodename=args.nodename, weights_filename=args.weights_filename, cam_idx=args.cam_idx, testVideo=args.testVideo)
    mainThread.start()
    mainThread.join()

if __name__ == '__main__':
    main()