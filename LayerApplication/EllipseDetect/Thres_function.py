import cv2
import os
import numpy as np
import sys
import paho.mqtt.client as mqtt
import json
import queue
from turbojpeg import TurboJPEG
from datetime import datetime
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtGui import qRed, qGreen, qBlue
import math
from scipy import stats

DIRNAME = os.path.dirname(os.path.abspath(__file__))
ROOTDIR = os.path.dirname(DIRNAME)
sys.path.append(f"{ROOTDIR}/lib")
from nodes import CameraReader
from frame import Frame
from common import *
from message import *


lowThreshold = 90
max_lowThreshold = 500
ratio = 5
kernel_size = 3

def hsv_mask(image):
	hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
	lower = np.array([90, 31, 99])
	upper = np.array([129, 110, 149])
	mask = cv2.inRange(hsv, lower, upper)
	result = cv2.bitwise_and(image, image, mask=mask)
	result =  cv2.cvtColor(result, cv2.COLOR_HSV2BGR)
	return result

def Detect_Ellipse(image):
    masked = hsv_mask(image)

    blur = cv2.GaussianBlur(masked, (5,5), 0)
    sub = masked.astype(int) - blur
    detected_edges = np.clip(masked.astype(int) + sub*2, a_min = 0, a_max = 255).astype('uint8')

    detected_edges = cv2.Canny(detected_edges,100,500,apertureSize = kernel_size)

    ellipse = cv2.HoughCircles(detected_edges, cv2.HOUGH_GRADIENT, dp=1, minDist=0.01, param1=150, param2=25, minRadius=0, maxRadius=200)

    #output = image.copy()
    # ensure at least some circles were found
    if ellipse is not None:
        # convert the (x, y) coordinates and radius of the circles to integers
        ellipse = np.round(ellipse[0, :]).astype("int")
        # loop over the (x, y) coordinates and radius of the circles
        test = [(x**2 + y**2)**0.5 for (x,y,_) in ellipse]

        #Remove outliers using z-score
        z = np.abs(stats.zscore(test))
        vals = np.where(z > 3)
        rem = [x for x in test if x not in vals[0]]

        max_y = max(rem)
        min_y = min(rem)
        max_ = test.index(max_y)
        min_ = test.index(min_y)
        (x, y, r) = ellipse[min_]
        (x1, y1, r1) = ellipse[max_]

        if(math.dist((x,y),(x1,y1)) < 210):
            min_edge = (int(min(x-r,x1-r)), int(min(y-r,y1-r)))#(x,y) #kiri atas
            max_edge = (int(max(x+r,x1+r)), int(max(y+r,y1+r)))#(x,y) #kanan bawah

            min_edge2 = (int(max(x+r,x1+r)), int(min(y-r, y1-r))) #kanan atas
            max_edge2 = (int(min(x-r,x1-r)), int(max(y+r,y1+r))) #kiri bawah

            middle = (int((min_edge[0] + max_edge[0])/2), int((min_edge[1] + min_edge[1])/2)) #tengah


            # cv2.rectangle(output, min_edge, max_edge, (0, 255, 0), 2)
            return [min_edge, min_edge2, max_edge, max_edge2], middle
            # return ([(494, 622), (622, 622), (622, 754), (494, 754)], (558.0, 622.0))
    
    return [(0, 0), (0, 0), (0, 0), (0, 0)], (0.0, 0.0)

class DetectThread(QThread):
    def __init__(self, mqtt_broker, camera:CameraReader, detectSignal:pyqtSignal, index):
        super().__init__()
        self.camera = camera
        self.detectSignal = detectSignal
        self.index = index

        self.alive = True

        # frame Queue
        self.frameQueue = queue.Queue(maxsize=30)

        # Mqtt client for receive raw image
        client = mqtt.Client()
        client.on_connect = self.onConnect
        client.on_message = self.onMessage
        client.connect(mqtt_broker)
        self.client = client

        self.jpeg = TurboJPEG('/usr/lib/x86_64-linux-gnu/libturbojpeg.so.0')
        self.detectCounter = 0

    def onConnect(self, client, userdata, flag, rc):
        self.client.subscribe(self.camera.output_topic)

    def onMessage(self, client, userdata, msg):
        data = json.loads(msg.payload)
        if 'id' in data and 'timestamp' in data and 'raw_data' in data:
            if self.detectCounter == 0:
                frame = Frame(data['id'], data['timestamp'], data['raw_data'])
                self.frameQueue.put_nowait(frame)
            self.detectCounter = (self.detectCounter + 1) % 30

    def stop(self):
        self.alive = False

    def run(self):
        try:
            self.client.loop_start()
            while self.alive:
                try:
                    frame = self.frameQueue.get()
                    image = frame.coverToCV2ByTurboJPEG(self.jpeg)
                    ellipse_coords, midpoint = Detect_Ellipse(image)
                    self.detectSignal.emit(ellipse_coords, midpoint, self.index)
                except:
                    continue
        finally:
            self.client.loop_stop()

if __name__ == '__main__':
    image_name = sys.argv[1]
    image = cv2.imread(image_name)
    time1 = datetime.now()
    ellipse_coords, midpoint = Detect_Ellipse(image)
    time2 = datetime.now()
    delta = time2 - time1
    print(delta.total_seconds())

    print(ellipse_coords)