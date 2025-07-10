"""
EventDetector : To recognize the event (Hit, Land) by trajectory of 3D-Model
"""
import sys
import os
import logging
import configparser
import queue
import paho.mqtt.client as mqtt
import numpy as np
import json
import math
import time
import argparse
import random
import matplotlib.pyplot as plt
import threading
import csv
import pandas as pd

from numpy.linalg import norm
from datetime import datetime
from typing import Optional
from scipy.signal import find_peaks, peak_prominences
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline

'''
Our common function
'''
from lib.common import ROOTDIR, loadConfig, loadNodeConfig
from lib.inspector import sendPerformance, sendNodeStateMsg
from lib.point import Point, sendPoints
from lib.writer import CSVWriter

from LayerContent.isPhysics import trajectory_segment, optimize_velocity
from LayerContent.CES.smooth import detectBallType

HEIGHT_THRESHOLD = 1.5
SEQUENCE_LENGTH = 20
SHOT_LENGTH = 20
SERVE_LENGTH = 0

INLIER_THRESHOLD = 0.3
INLIER_RATIO = 0.7

def isHit(client, topic, v1, v2, v3, v4, v5, GROUND_HEIGHT, max_time=0.5):
    # Small Mid Large Mid Small or L M S M L
    if ((v3.y - v2.y) * (v4.y - v3.y) < 0 and
       (v3.y - v2.y) * (v2.y - v1.y) > 0 and
       (v4.y - v3.y) * (v5.y - v4.y) > 0 and
       v3.z > GROUND_HEIGHT and
       # If time diff too long means ball out-and-in cameras
       abs(v5.timestamp - v1.timestamp) < max_time):
        logging.info(f"Fid: {v3.fid} Time : {v3.timestamp:>.3f} --> HIT")
        v3.event = 1
        v3.color = 'blue'
        sendPoints(client,topic, v3)
        return True # hit point
    else:
        return False

def isLand(client, topic, v1, v2, v3, GROUND_HEIGHT):
    if (v3.z <= GROUND_HEIGHT and
        v1.z > GROUND_HEIGHT and
        v2.z > GROUND_HEIGHT):
        logging.info(f"Fid: {v3.fid} Time : {v3.timestamp:>.3f} --> LAND")
        v3.event = 2
        v3.color = 'red'
        sendPoints(client,topic, v3)
        return True
    else:
        return False

def isServe(client, topic, v1, v2, v3, GROUND_HEIGHT, SERVE_HEIGHT):
    if (v3.z >= SERVE_HEIGHT and
        v1.z <= GROUND_HEIGHT and
        v2.z <= GROUND_HEIGHT):
        # Serve ball TODO First Serve will miss
        logging.info(f"Fid: {v3.fid} Time : {v3.timestamp:>.3f} --> SERVE")
        v3.event = 3
        v3.color = 'green'
        sendPoints(client,topic, v3)
        return True
    else:
        return False


class CSVWriter():
    def __init__(self, filename):
        self.filename = filename
        self.csvfile = open(filename, 'w', newline='')
        self.writer = csv.writer(self.csvfile)
        self.writer.writerow(['Type', 'fid', 'event', 'timestamp', 'position', 'start_fid', 'start_timestamp', 'start_position', 'end_fid', 'end_timestamp', 'end_position', 'speed', 'ball_type', 'publish_time'])

    def close(self):
        self.csvfile.flush()
        self.csvfile.close()

    def writePublish(self, type, data, publish_time):
        if type == 'Event':
            self.writer.writerow(['Event', data['fid'], data['event'], data['timestamp'], json.dumps(data['position']), '', '', '', '', '', '', '', '', publish_time])
        elif type == 'Segment':
            self.writer.writerow(['Segment', '', '', '', '', data['start_fid'], data['start_timestamp'], json.dumps(data['start_position']), data['end_fid'], data['end_timestamp'], json.dumps(data['end_position']), json.dumps(data['speed']), data['ball_type'], publish_time])

        self.csvfile.flush()


class EventDetector():
    def __init__(self, name, client, topic, writer3D, fps):
        self.name = name
        self.points = [] # 3D point
        self.fps = fps

        # setup MQTT client
        self.client = client
        # self.topic = topic
        self.topic = '/DATA/ContentDevice/LayerContent/Model3D_peak'

        # 3d csv writer
        self.writer3D = writer3D

        self.GROUND_HEIGHT = 0.1
        self.MAX_DELAY_TIME = 1/self.fps
        self.SERVE_HEIGHT = 0.5 # Serving height should higher than this value
        self.pre_calculate_time = float("-inf")

    def addPoint(self, point):
        self.points.append(point)

    def detect(self):
        # 0: nothing, 1: Shot, 2: Land
        event = 0
        calculate_time = datetime.now().timestamp()
        if calculate_time - self.pre_calculate_time >= self.MAX_DELAY_TIME:
            self.pre_calculate_time = calculate_time
            #self.clear_queue()
        else:
            if len(self.points) >= 3:
                isLand(self.client, self.topic, self.points[0], self.points[1], self.points[2], self.GROUND_HEIGHT)
                isServe(self.client, self.topic, self.points[0], self.points[1], self.points[2], self.GROUND_HEIGHT, self.SERVE_HEIGHT)
            if len(self.points) >= 5:
                isHit(self.client, self.topic, self.points[0], self.points[1], self.points[2], self.points[3], self.points[4], self.GROUND_HEIGHT)
                self.writer3D.writePoints(self.points[0])
                # sendPoints(self.client,self.topic, self.points[0])
                del self.points[0]

    def close(self):
        self.clear_queue()

    def clear_queue(self):
        while self.points:
            # Duplicate Code TODO
            if len(self.points) >= 3:
                isLand(self.client, self.topic, self.points[0], self.points[1], self.points[2], self.GROUND_HEIGHT)
                isServe(self.client, self.topic, self.points[0], self.points[1], self.points[2], self.GROUND_HEIGHT, self.SERVE_HEIGHT)
            if len(self.points) >= 5:
                isHit(self.client, self.topic, self.points[0], self.points[1], self.points[2], self.points[3], self.points[4], self.GROUND_HEIGHT)

            self.writer3D.writePoints(self.points[0])
            # sendPoints(self.client,self.topic, self.points[0])
            del self.points[0]

    def publishHitEvent(self, points):
        # RNN need trajectory : {ball, "HIT, ball, ball, ball, ball"}
        sendPoints(self.client, self.topic, points)
        #sendPerformance(self.client, self.topic, 'none', 'send', fids)

    def publishLandEvent(self, point):
        sendPoints(self.client, self.topic, point)
        #sendPerformance(self.client, self.topic, 'none', 'send', [point.fid])





class flyEventDetector():
    def __init__(self, name, client, topic, writer3D, fps, start_time, save_path='./'):
        self.name = name
        self.points = [] # 3D point
        self.fps = fps
        self.save_path = save_path

        # setup MQTT client
        self.client = client
        self.topic = topic

        # 3D CSV writer
        self.writer3D = writer3D
        self.writer3Dbuffer = {}

        # Event and segment CSV writer
        self.save_path = save_path
        self.filePath_publish = f"{self.save_path}/Model3D_publish.csv"
        self.publish_writer = CSVWriter(self.filePath_publish)

        self.GROUND_HEIGHT = 0.1
        self.MAX_DELAY_TIME = 1/self.fps
        self.SERVE_HEIGHT = 0.5 # Serving height should higher than this value
        self.pre_calculate_time = float("-inf")

        self.shots = []

        self.max_delay = 360 * (1/self.fps)

        self.lock = threading.Lock()
        
        # Thread to monitor delay time
        self.last_point_time = time.time() # Time of the last received point
        self.alive = True
        self.monitor_thread = threading.Thread(target=self._monitor_delay)
        # self.monitor_thread.start()
        self.monitor_thread_started = False

        self.event_triggered = False
        self.new_rally = True

        self.merged_segments = []
        self.prev_start, self.prev_end, self.prev_v, self.prev_points = None, None, None, []
        self.last_valid_time = None

        self.points_cleared = True

        self.start_time = start_time
        self.timer = None    
    
    def _monitor_delay(self):
        """Thread to monitor if delay exceeds MAX_DELAY_TIME."""
        while self.alive:
            with self.lock:
                logging.debug(f"Check delay: {time.time() - self.start_time}")
                current_time = time.time()
                if (current_time - self.last_point_time) > self.max_delay:
                    if len(self.points) > 0:
                        logging.debug(f"LONG DELAY: {current_time - self.last_point_time}")
                        # print(f"current points:")
                        # print(f"{self.points[0].fid} - {self.points[-1 ].fid}")
                        self.process_trajectory(self.points)
                        self.points = []  # Reset points
                    if self.prev_v is not None and self.prev_points:
                        print(f"--> SHOT: DELAY: {self.prev_start} - {self.prev_end}")                      
                        self.complete_segment(dead=True)
                        self.reset_prev_flight()

                    self.event_triggered = False
                    self.new_rally = True
                else:
                    if self.monitor_thread_started:
                        logging.debug(f"Delay check passed: {time.time() - self.start_time}")
                        print(f"Delay check passed: {time.time() - self.start_time}")
            time.sleep(1)
    
    def addPoint(self, point):
        self.points.append(point)
        # write
        self.writer3Dbuffer[point.fid] = point
        self.last_point_time = time.time()
        if not self.monitor_thread_started:
            self.monitor_thread.start()
            self.monitor_thread_started = True

    def detect(self):
        with self.lock:
            if len(self.points) >= SEQUENCE_LENGTH:
                self.points_cleared = False
                print()
                print('========================================')
                print(f"Sequence: {self.points[0].fid} - {self.points[-1].fid}")
                print('--------------------')
                self.process_trajectory(self.points)
                self.points = []
                self.points_cleared = True

    def process_trajectory(self, points):
        # logging.debug(f"--- Start process_trajectory --- {time.time() - self.start_time}")
        results = trajectory_segment(self.points, fps=120, initial_V=np.array([0, 0, 0]), loss_function=1, evaluation_method='vote', inlier_threshold=INLIER_THRESHOLD, inlier_ratio=INLIER_RATIO)
        for traj, isFly, v in results:
            speed = np.linalg.norm(v) if v is not None else -1
            self.trajectory_merged(traj, isFly, v, speed)
        # logging.debug(f"--- Finish process_trajectory --- {time.time() - self.start_time}")

    def complete_segment(self, dead=False):
        if len(self.prev_points) >= SHOT_LENGTH:
            if self.new_rally:
                if len(self.prev_points) >= SERVE_LENGTH:
                    print(f"EVENT: SERVE: {self.prev_points[0].fid}")
                    sendPoints(self.client, f"{self.topic}/Event/Debug", self.prev_points[0])
                    self.publishEvent(2, self.prev_points[0])
                    self.new_rally = False
                    self.prev_points[0].event = 2
                    # write
                    # self.writer_update_event_in_buffer(self.prev_points[0], 2)        
            
            sendPoints(self.client, f"{self.topic}/Segment/Debug", self.prev_points)
            self.publishSegment(self.prev_points)
            self.merged_segments.append((self.prev_start, self.prev_end, self.prev_v, self.prev_points))
            if dead:
                print(f"EVENT: DEAD: {self.prev_points[-1].fid}")
                self.publishEvent(3, self.prev_points[-1])
                self.prev_points[-1].event = 3
                # write
                # self.writer_update_event_in_buffer(self.prev_points[-1], 3)
            else:
                if self.prev_points[-1].z < 0.2:
                    print('!!! Land !!!')
                    print(f"EVENT: DEAD: {self.prev_points[-1].fid}")
                    self.publishEvent(3, self.prev_points[-1])
                    self.prev_points[-1].event = 3
                    # write
                    # self.writer_update_event_in_buffer(self.prev_points[-1], 3)
                    self.reset_prev_flight()
                    self.new_rally = True


    def creat_new_segment(self):
        event_type = None
        if len(self.prev_points) >= SHOT_LENGTH and self.new_rally:
            print(f"EVENT: SERVE {self.prev_points[0].fid}")
            event_type = 2
            self.new_rally = False
            # self.event_triggered = True
        elif len(self.prev_points) >= SHOT_LENGTH and self.event_triggered == False:
            print(f"EVENT: HIT {self.prev_points[0].fid}")
            event_type = 1
            # self.event_triggered = True
            
        if event_type:
            self.event_triggered = True
            
            sendPoints(self.client, f"{self.topic}/Event/Debug", self.prev_points[0])
            self.publishEvent(event_type, self.prev_points[0])
            self.prev_points[0].event = event_type
            # write
            # self.writer_update_event_in_buffer(self.prev_points[0], event_type)
    
    def combine_segment(self):
        if len(self.prev_points) >= SHOT_LENGTH and self.event_triggered == False:
            sendPoints(self.client, f"{self.topic}/Event/Debug", self.prev_points[0])
            if self.new_rally:
                print(f"EVENT: SERVE: merge: {self.prev_points[0].fid}")
                event_type = 2
                self.new_rally = False
            else:
                event_type = 1
                print(f"EVENT: HIT {self.prev_points[0].fid}")
            
            self.event_triggered = True
            self.publishEvent(event_type, self.prev_points[0])
            self.prev_points[0].event = event_type
            # write
            # self.writer_update_event_in_buffer(self.prev_points[0], event_type)  

    def trajectory_merged(self, traj, isFly, v, speed):
        print()
        if v is not None:
            print(f"--> {traj[0].fid} - {traj[-1].fid} : {isFly}, Speed: {v}, {speed}")
        else:
            print(f"--> {traj[0].fid} - {traj[-1].fid} : {isFly}")

        traj_start_time = traj[0].timestamp
        median_height = np.median([point.z for point in traj])
        if self.last_valid_time is not None and traj_start_time - self.last_valid_time > 3.0:
            print("... No trajectory detected for over 3 seconds, resetting ...")
            if self.prev_v is not None:
                print(f"--> SHOT: DELAYNEW: {self.prev_start} - {self.prev_end}")
                print(self.prev_points[0].fid, self.prev_points[-1].fid)
                self.complete_segment(dead=True)
            self.reset_prev_flight()
            self.new_rally = True

        if isFly and (speed >= 3 or (0 < speed < 3 and median_height > HEIGHT_THRESHOLD)):
            if self.prev_v is None or len(self.prev_points) == 0: # Start new segment
                self.update_prev_flight(traj, v)
                self.creat_new_segment()
            else: # If a previous segment exists, check whether it meets the merge condition
                # print(f"Valid trajectory interval: ( {traj[0].fid} - {self.prev_points[-1].fid} ) {traj[0].timestamp - self.prev_points[-1].timestamp}")
                if self.can_merge_segments(traj, v):
                    print("... Merged with previous segment ...", end='')
                    self.merge_prev_flight(traj, v)
                    self.combine_segment()      
                else:
                    # If not mergeable, finalize the previous segment first
                    print(f"--> SHOT: NEW:{self.prev_start} - {self.prev_end}")
                    self.complete_segment()
                    self.event_triggered = False

                    # Start a new segment and report a HIT event
                    self.update_prev_flight(traj, v)
                    self.creat_new_segment()

            self.last_valid_time = traj[-1].timestamp  # Update the timestamp of the last valid trajectory 

        
    def can_merge_segments(self, traj, v, time_gap=2.0):
        # Same Y direction and time gap less than or equal to time_gap
        return (
            ((self.prev_v[1] >= 0 and v[1] >= 0) or (self.prev_v[1] < 0 and v[1] < 0)) and
            (traj[0].timestamp - self.prev_points[-1].timestamp <= time_gap)
        )

    
    def update_prev_flight(self, traj, v):
        self.prev_start, self.prev_end, self.prev_v, self.prev_points = traj[0].fid, traj[-1].fid, v, list(traj)
        print(f"Update Prev {self.prev_points[0].fid} - {self.prev_points[-1].fid}")
    
    def reset_prev_flight(self):
        self.prev_start, self.prev_end, self.prev_v, self.prev_points = None, None, None, [] 

    def merge_prev_flight(self, traj, v):
        self.prev_end = traj[-1].fid
        self.prev_v = v
        self.prev_points.extend(traj)
        print(f"{self.prev_points[0].fid} - {self.prev_points[-1].fid}")
   
    # event_type: 1: hit, 2: serve, 3: dead, 4: 不自由移動, 5: 靜止/無球
    def publishEvent(self, event_type, point):
        print('--- Publish Event ---')
        topic = f"{self.topic}/Event"
        if event_type == 1 or event_type == 2 or event_type == 3:
            position = [point.x, point.y, point.z]
        elif event_type == 4 or event_type == 5:
            position = [-1, -1, -1]
        payload = {'fid': point.fid, 'event': event_type, 'timestamp': round(point.timestamp, 3), 'position': [round(pos, 3) for pos in position]}
        self.client.publish(topic, json.dumps(payload))
        print(payload)
        publish_time = datetime.now().time().isoformat(timespec='milliseconds')
        self.publish_writer.writePublish('Event', payload, publish_time)
        logging.debug(f">> {publish_time} Publish Event: {payload}")
    
    def publishSegment(self, points):
        print('--- Publish Segment ---')
        topic = f"{self.topic}/Segment"
        start = points[0]
        end = points[-1]
        starting_point = [start.x, start.y, start.z, start.timestamp]
        flight_time = end.timestamp - start.timestamp
        if start.y > 0:
            initial_V = np.array([0, -10, 0])
        else:
            initial_V = np.array([0, 10, 0])
        optimized_V, loss = optimize_velocity(points, starting_point, self.fps, flight_time, initial_V)
        if optimized_V is not None:
            print(optimized_V, np.linalg.norm(optimized_V)*3.6)
            optimized_V = list(optimized_V)
        else:
            print('None', [0, 0, 0])
            optimized_V = [0, 0, 0]
        ball_type = detectBallType(points)
        payload = {'start_fid': start.fid, 'start_timestamp': round(start.timestamp, 3), 'start_position': [round(start.x, 3), round(start.y, 3), round(start.z, 3)], 
                   'end_fid': end.fid, 'end_timestamp': round(end.timestamp, 3), 'end_position': [round(end.x, 3), round(end.y, 3), round(end.z, 3)],
                   'speed': [round(v, 3) for v in optimized_V], 'ball_type': ball_type}
        self.client.publish(topic, json.dumps(payload))
        print(payload)
        publish_time = datetime.now().time().isoformat(timespec='milliseconds')
        self.publish_writer.writePublish('Segment', payload, publish_time)
        logging.debug(f">> {publish_time} Publish Segment: {payload}")
    
    def isLand(self, traj):
        if traj and traj[-1].z < 0.1:
            # sendPoints(self.client, f"{self.topic}/Event/Debug", traj[-1])
            print(f"EVENT: DEAD {traj[-1].fid}")
            self.publishEvent(3, traj[-1])
    
    def writer_update_event_in_buffer(self, point, event_type):
        point.event = event_type
        event_fid = point.fid
        if event_fid in self.writer3Dbuffer:
            self.writer3Dbuffer[event_fid] = point
        
        if event_type == 3:
            self.writer_write_buffer_to_csv()
    
    def writer_write_buffer_to_csv(self):
        points = list(self.writer3Dbuffer.values())
        # TODO
        self.writer3D.writePoints(points)
        self.writer3Dbuffer = {}


    def close(self):
        print('----- Event Detector Close -----')
        time.sleep(0.5)
        # self.clear_queue()

        self.alive = False  # Set flag to stop the loop in _monitor_delay
        # self.monitor_thread.join()  # Wait for the thread to finish

        if self.prev_v is not None and self.prev_points:
            print(f"--> SHOT: {self.prev_start} - {self.prev_end}")
            self.complete_segment(dead=True)

        # write
        # self.writer_write_buffer_to_csv()
        
        print()
        print('===== SHOT =====')
        for idx, (s, e, v, traj) in enumerate(self.merged_segments):
            print(f"{idx+1}: {traj[0].fid} - {traj[-1].fid}")

        self.publish_writer.close()
