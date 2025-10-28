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
from typing import Optional

import cv2
import numpy as np
import paho.mqtt.client as mqtt
import torch
from sklearn.metrics import confusion_matrix
from torchvision import transforms

# Our System's library
from TrackNet10 import TrackNet10

# A Python wrapper of libjpeg-turbo for decoding and encoding JPEG image.
#from turbojpeg import TurboJPEG, TJPF_GRAY, TJSAMP_GRAY, TJFLAG_PROGRESSIVE

DIRNAME = os.path.dirname(os.path.abspath(__file__))

from lib.common import insertById, loadConfig, loadNodeConfig, ROOTDIR
from lib.frame import Frame
from lib.inspector import sendNodeStateMsg, sendPerformance
from lib.point import Point
from lib.receiver import RawImgReceiver
from lib.writer import CSVWriter

BATCH_SIZE = 1
HEIGHT = 288
WIDTH = 512
TRACK_SIZE = 10

DEBUG = False

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
        self.killswitch = threading.Event()

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

    def stop(self):
        self.killswitch.set()

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
        self.client.publish(self.topic, json_str)
        sendPerformance(self.client, self.topic, 'none', 'send', fids)

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
                logging.info("id: {} timestamp: {}, (x, y): ({}, {})".format(fid, timestamp, cx_pred, cy_pred))
                if (not isBlacklist(self.blacklist, cx_pred, cy_pred)):
                    point = Point(fid=fid, timestamp=timestamp, visibility=1, x=cx_pred, y=cy_pred, z=0, event=0, speed=0)
                    insertById(self.points, point)
                else:
                    point = Point(fid=fid, timestamp=timestamp, visibility=0, x=0, y=0, z=0, event=0, speed=0)

                if DEBUG:
                    height, width, channels = image.shape
                    ratio_input_w = width / WIDTH
                    ratio_input_h = height / HEIGHT
                    debug_png = image
                    cv2.circle(debug_png,(int(ratio_input_w*(target[0] + target[2] / 2)), int(ratio_input_h*(target[1] + target[3] / 2))),5,(0,0,255),-1)
                    cv2.imwrite(os.path.join(DIRNAME,'debug',self.nodename,str(fid)+'.png'), debug_png)

            if self.csv_writer is not None:
                self.csv_writer.writePoints(point)

    def run(self):
        # logging.debug("TrackNetThread started.")
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
        # logging.debug("TrackNetThread terminated.")

class MainThread(threading.Thread):
    def __init__(self, args, settings, camera_settings, device, model, blacklist_file):
        threading.Thread.__init__(self)

        self.nodename = args.nodename
        self.data = args.data
        self.settings = settings
        self.camera_settings = camera_settings
        self.device = device
        self.model = model
        self.page = args.page

        queue_size = int(self.settings['queue_size'])
        self.blacklist_file = blacklist_file

        input_topic = settings['input_topic']
        self.output_topic = settings['output_topic']
        self.broker = self.settings['mqtt_broker']
        # Setup Image Receiver
        self.tracks = threading.Event()
        self.imgReceiver = None
        self.videoFile = None
        # self.csvFile = None
        if (self.data == 'CameraReader'):
            self.imgReceiver = RawImgReceiver(self.broker, input_topic, queue_size, self.tracks)
        else:
            self.videoFile = args.data
            # self.csvFile = args.input_csv

        # Setup CSV Writer
        if args.save_csv != 'no':
            path = os.path.join(args.save_csv, args.nodename+'.csv')
            self.csv_writer = CSVWriter(name=args.nodename, filename=path)
        else:
            self.csv_writer = None

        # setup MQTT client
        client = mqtt.Client()
        client.on_connect = self.on_connect
        client.on_message = self.on_message
        client.connect(self.broker)
        self.client = client

        self.killswitch = threading.Event()

        if DEBUG:
            folder = os.path.join(DIRNAME,'debug',self.nodename)
            if os.path.exists(folder):
                shutil.rmtree(folder)
            os.makedirs(folder,exist_ok=True)

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
        logging.debug("{} started.".format(self.nodename))

        # setup tracknetthreads
        tracknetThreads = []
        threads_size = int(self.settings['threads_size'])
        blacklist = generateBlacklist(self.blacklist_file)
        cam_origin_width,cam_origin_height = [int(i) for i in re.findall(r"\d+",self.camera_settings['Camera']['RecordResolution'])]
        #cam_origin_width = int(self.settings['width'])
        #cam_origin_height = int(self.settings['height'])
        # improve encode/decode performance
        #jpeg = TurboJPEG('/usr/lib/x86_64-linux-gnu/libturbojpeg.so.0')

        self.client.loop_start()

        logging.info(f"{self.nodename} is ready.")
        sendNodeStateMsg(self.client, self.nodename, "ready")

        if self.data == 'CameraReader':
            self.alive = True
            self.imgReceiver.start()

            while self.alive:
                # TrackNet
                if len(self.imgReceiver.queue) >= TRACK_SIZE and len(tracknetThreads) < threads_size:
                    tracknetThread = TrackNetThread(nodename=self.nodename,
                                                    device=self.device,
                                                    model=self.model,
                                                    broker=self.broker,
                                                    topic=self.output_topic,
                                                    blacklist=blacklist,
                                                    output_width=cam_origin_width,
                                                    output_height=cam_origin_height,
                                                    csv_writer=self.csv_writer)
                    for i in range(TRACK_SIZE):
                        frame = self.imgReceiver.queue.pop(0)
                        #logging.info("received fid:{}".format(frame.fid))
                        tracknetThread.images.append(frame.coverToCV2())
                        tracknetThread.fids.append(frame.fid)
                        tracknetThread.timestamps.append(frame.timestamp)
                    tracknetThread.start()
                    tracknetThreads.append(tracknetThread)
                else:
                    for i in range(len(tracknetThreads)):
                        if tracknetThreads[i].is_alive() == False:
                            del tracknetThreads[i]
                            break
                if len(self.imgReceiver.queue) <= TRACK_SIZE and len(tracknetThreads) == 0:
                    # time.sleep(0.03)
                    self.tracks.wait()
                    self.tracks.clear()

            self.imgReceiver.stop()
            self.imgReceiver.join()

        else: #read video file directly
            queue = []
            if os.path.exists(self.videoFile):
                cap = cv2.VideoCapture(self.videoFile)
                logging.debug(f"videofile={self.videoFile} total frames={cap.get(cv2.CAP_PROP_FRAME_COUNT)}")
                try:
                    idx = 1
                    while cap.isOpened():
                        ret, img = cap.read()
                        if not ret:
                            logging.info(f"{self.videoFile} file is end.")
                            break
                        timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                        frame = Frame(fid=idx, timestamp=timestamp, raw_data=img)
                        queue.append(frame)
                        idx += 1

                        if len(queue) >= TRACK_SIZE and len(tracknetThreads) < threads_size:
                            #logging.debug(f"{self.videoFile} do one predict")
                            tracknetThread = TrackNetThread(nodename=self.nodename,
                                                            device=self.device,
                                                            model=self.model,
                                                            broker=self.broker,
                                                            topic=self.output_topic,
                                                            blacklist=blacklist,
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
        sendNodeStateMsg(self.client, self.nodename, "terminated", self.page)
        self.client.loop_stop()

        logging.info("{} terminated.".format(self.nodename))

def parse_args() -> Optional[str]:
    # Args
    parser = argparse.ArgumentParser(description = 'Pytorch TrackNet10')
    parser.add_argument('--nodename', type=str, default = 'TrackNet', help = 'mqtt node name (default: TrackNet)')
    parser.add_argument('--weights', type = str,default = 'TrackNet.tar', help = 'input model weight for predict')
    parser.add_argument('--data', type=str, default = 'CameraReader', help = 'load image from CameraReader or path of the video')
    parser.add_argument('--input_csv', type=str, default = None, help = 'if data is a path of a video, then the corresponding csv file is needed')
    parser.add_argument('--save_csv', type=str, default = 'no', help = 'saved csv filename (default: no), if you dont want to save enter no ')
    parser.add_argument('--page', type=str, default = None, help = 'the page that calls the startTracking function')
    args = parser.parse_args()

    return args

def main():
    # Parse arguments
    args = parse_args()
    # Load configs
    path = args.data.split('/')
    replyFile = [path[i+1] for i in range(len(path)) if path[i]=='replay']
    projectCfg = f"{ROOTDIR}/replay/"+replyFile[0]+"/config"
    settings = loadNodeConfig(projectCfg, args.nodename)
    
    #load camera cfg
    filecfg = loadNodeConfig(projectCfg, settings['file_name'])
    cameraCfg = f"{ROOTDIR}/replay/"+replyFile[0]+'/'+filecfg['hw_id']+'.cfg'
    camera_settings = loadConfig(cameraCfg)
    
    # GPU device
    logging.info("GPU Use : {}".format(torch.cuda.is_available()))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    ### Track Net Model ###
    model = TrackNet10()
    model.to(device)
    if settings['optimizer'] == 'Ada':
        optimizer = torch.optim.Adadelta(model.parameters(), lr=1e-1, rho=0.9, eps=1e-06, weight_decay=0)
        #optimizer = torch.optim.Adam(model.parameters(), lr = args.lr, weight_decay = args.weight_decay)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr = float(settings['lr']),
                                    weight_decay = float(settings['weight_decay']),
                                    momentum = float(settings['momentum']))

    weights = f"{DIRNAME}/weights/{args.weights}"
    blacklist = f"{DIRNAME}/weights/{settings['blacklist']}"
    checkpoint = torch.load(weights)
    model.load_state_dict(checkpoint['state_dict'])
    epoch = checkpoint['epoch']
    model.eval()

    # Start MainThread
    mainThread = MainThread(args, settings, camera_settings, device, model, blacklist)
    mainThread.start()
    mainThread.join()

if __name__ == '__main__':
    main()

