"""
Triangulation : To combine two 2D points into 3D point
"""
import argparse
import json
import logging
import os
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import paho.mqtt.client as mqtt
from copy import deepcopy
import csv
import pandas as pd

# Our System's library
from LayerContent.CameraSync import CameraSync
# from LayerContent.CameraSync_Callback import CameraSync
# from LayerContent.Model3D.EventDetector import CSVWriter, EventDetector, flyEventDetector
from LayerContent.Model3D.EventDetector import flyEventDetector
from LayerContent.Model3D.MultiCamTriang import MultiCamTriang

from LayerContent.CES.smooth import removeOuterPoint_batch, detectEvent_batch, removeMotionOutliers_batch
from LayerContent.CES.gradientSpeed import gradient_speed

from LayerContent.tools.draw import drawPoints, drawTrajectory_points, drawGradientTrajectory_points, drawVisualization

from lib.common import ROOTDIR, loadConfig, loadNodeConfig
from lib.inspector import sendNodeStateMsg
from lib.point import Point, sendPoints, new_sendPoints
from lib.receiver import PointReceiver
from lib.writer import CSVWriter
from lib.MqttAgent import MqttAgent

# from LayerContent.ContentAgent import ContentLayerAgent


def denoise(l, a):
    if len(l) == 0:
       lastpoint = Point(0,0,0,0)
    else:
       lastpoint = l[len(l)-1]

    for i in range(len(l)):
        p = l[i]
        if p.fid > a.fid:
            if (   ((abs(a.x - lastpoint.x) > 300) or (abs(a.y - lastpoint.y) > 300) or (abs(a.z - lastpoint.z) > 300)) and ((a.fid -  lastpoint.fid) == 1)    ):
                logging.info("not order noise : x = {} y = {}".format(a.x, a.y))
                logging.info("not order lastpoint {} {} {}".format( lastpoint.fid, lastpoint.x, lastpoint.y, lastpoint.z))
                return
            else:
                l.insert(i, a)
                return

    if (   ((abs(a.x - lastpoint.x) > 300) or (abs(a.y - lastpoint.y) > 300) or (abs(a.z - lastpoint.z) > 300)) and ((a.fid -  lastpoint.fid) == 1) and (a.fid != 1)  ):
        logging.info("Noise : id = {} x = {} y = {} ".format(a.fid, a.x, a.y))
        logging.info('lastpoint : {} {} {} {}'.format(lastpoint.fid, lastpoint.x, lastpoint.y, lastpoint.z))
        return
    else:
        l.append(a)

def isWithinBounds(point):
    x = point.x
    y = point.y
    z = point.z
    if x > 4.5 or x < -4.5:
        return False
    if y > 8 or y < -8:
        return False
    if z > 10 or z < 0:
        return False
    return True

class RawTrack2D():
    def __init__(self, name, receiver):
        self.name = name
        self.points = []
        self.receiver = receiver
        self.inserted_count = 0
        logging.debug(f"[RawTrack2D] name: {self.name}")

    def startReceiver(self):
        self.receiver.start()

    def stopReceiver(self):
        self.receiver.stop()
        self.receiver.join()

    def remove(self, idx):
        del self.points[idx]

    # TODO: timestamp replace fid
    def doInterpolation(self):
        if len(self.receiver.queue) >= 2:
            startPoint = self.receiver.queue.pop(0)
            endPoint = self.receiver.queue[0]
            xp = np.linspace(startPoint.x, endPoint.x, endPoint.fid - startPoint.fid + 1)
            yp = np.linspace(startPoint.y, endPoint.y, endPoint.fid - startPoint.fid + 1)
            idp = np.linspace(startPoint.fid, endPoint.fid, endPoint.fid - startPoint.fid + 1)
            newarr = [0 for i in range(endPoint.fid - startPoint.fid + 1)  ]
            for i in range(len(xp)):
                newarr[i] = Point(idp[i], 1, xp[i], yp[i], 0)
            del newarr[0]
            self.points.extend(newarr)

    def insertPoints(self):
        if len(self.receiver.queue) > 0:
            # print(f'[pop] {self.name} :', self.receiver.queue[0].fid)
            # print(f'[insert] {self.name} :', self.receiver.queue[0].fid)
            # logging.debug(f"[insert] {self.name}: {self.receiver.queue[0].fid}")
            point = self.receiver.queue.pop(0)
            self.points.append(point)
            self.inserted_count += 1

class TriangulationThread(threading.Thread):
    def __init__(self, client, data_handler, main_thread, args, settings, ks, poses, eye, dist, newcameramtx, projection_mat):
        threading.Thread.__init__(self)

        self.fps = float(settings['fps'])

        self.main_thread = main_thread
        self.receivers_finished = 0

        self.ks = ks
        self.poses = poses
        self.eye = eye
        self.dist = dist
        self.newcameramtx = newcameramtx
        self.projection_mat = projection_mat
        self.nodename = args.nodename
        self.mode = args.mode

        self.WAIT_TIME = (1 / self.fps) * 10     # waiting for other camaras point (sec)
        self.SKEW_TOL = 1.0 / (2 * self.fps)      # 跨相機時間偏差容忍
        self.STALE_TIMEOUT = 1.0                  # 相機失活門檻秒數（可依環境調）
        # Heartbeat 狀態
        self.cam_last_ts = {}      # cam -> 最近 batch_end_ts（watermark）
        self.cam_last_seen = {}    # cam -> 最近收到心跳的 wall-clock 時間

        if os.path.isfile(args.save_path):
            self.save_path = os.path.dirname(os.path.abspath(args.save_path))
        else:
            self.save_path = os.path.abspath(args.save_path)
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        input_topics = settings['input_topic'].split(',')
        queue_size = int(settings['queue_size'])
        broker = (settings['mqtt_broker'], int(settings['mqtt_port']))
        self.topic = settings['output_topic']
        # event_topic = settings['output_event_topic']

        self.client = client
        self.data_handler = data_handler

        self.start_time = time.time()


        # 2D CSVWriters
        self.csv2DWriters = []

        self.max_FID = 0

        # Setup Event Detector
        filePath = f"{self.save_path}/{self.nodename}.csv"
        logging.debug("[{}]: Write 3d csv in {} by topic {}".format(self.nodename, filePath, self.topic))
        # self.csv3DWriter = CSVWriter(name=self.topic, filename=filePath)
        # self.eventDetector = EventDetector(self.nodename, self.client, self.topic, self.csv3DWriter, self.fps)
        
        filePath_event = f"{self.save_path}/{self.nodename}_event.csv"
        self.csv3DWriter_event = CSVWriter(name=self.topic, filename=filePath_event)
        self.flyEventDetector = flyEventDetector(self.nodename, self.data_handler, self.client, self.topic, self.csv3DWriter_event, self.fps, self.start_time, self.save_path)
        
        filePath_points = f"{self.save_path}/{self.nodename}_points.csv"
        self.csv3DWriter_points = CSVWriter(name=self.topic, filename=filePath_points)
        print('save 3D csv at:', filePath_points)

        # setup TrackNet Receiver
        self.rawTrack2Ds = []
        self.pose_lock = threading.Lock()
        self.pose_records = {}
        for in_topic in input_topics:
            print('--------------------------------------------')
            print('in_topic:', in_topic)
            in_name = in_topic.split('/')[2]
            devicename = in_name
            idx = [key for key, value in args.camera_mapping.items() if value == devicename][0]
            receiver = PointReceiver(self.data_handler, self, self.nodename, in_name, in_topic, queue_size)
            rawTrack2D = RawTrack2D(name=in_name, receiver=receiver)
            # print("rawTrack2D's name:", rawTrack2D.name)
            self.rawTrack2Ds.append(rawTrack2D)
            # Setup 2D track csv writer
            parts = in_topic.split('/')
            func = parts[4]
            filename = f"{func}_data_{idx}"
            filePath = f"{self.save_path}/{filename}.csv"
            # logging.debug("[{}]: Write 2d csv in {} by topic {}".format(self.nodename, filePath, in_topic))
            self.csv2DWriters.append(CSVWriter(name=in_name, filename=filePath))
            print('--------------------------------------------')

            pose_topic = in_topic.replace('TrackNet', 'Pose')
            self.pose_records.setdefault(in_name, [])
            self.data_handler.subscribe(pose_topic, self._make_pose_callback(in_name, pose_topic))
            logging.debug("[TriangulationThread] Listening pose topic %s", pose_topic)

        # Setup MultiCamTriang
        self.multiCamTriang = MultiCamTriang(poses, eye, self.newcameramtx)
        self.alive = False

        self.MAXIMUM_FRAME_TIMESTAMP_DELAY = 1/(self.fps*2) # according to FPS

        self.max_delay = 240 * (1/self.fps) # 30/60倍

        self.camera_source = pd.DataFrame(columns=["Frame", "Visibility", "X", "Y", "Z", "Event", "Timestamp", "Camera"])


    def on_connect(self, client, userdata, flag, rc):
        logging.info(f"{self.nodename} Connected with result code: {rc}")

    def on_publish(self, mosq, userdata, mid):
        logging.debug("send")

    def stop(self):
        self.alive = False

    def allPointsInserted(self):
        '''
        Batch Mode :
        Checks if all points received by PointReceiver have been inserted into their corresponding RawTrack2D
        '''
        for rawTrack2D in self.rawTrack2Ds:
            if rawTrack2D.inserted_count < rawTrack2D.receiver.point_count:
                return False
        return True

    def onReceiverFinished(self):
        '''
        Batch Mode :
        Check if all input topic data has been transmitted (EOF/EOS received),
        then wait until all points are inserted into the RawTrack2D before stopping the main thread.
        '''
        self.receivers_finished += 1
        if self.receivers_finished == len(self.rawTrack2Ds):
            while not self.allPointsInserted():
                print("Waiting for all points to be inserted into the RawTrack2D...")
                # time.sleep(0.1)  # Pause briefly before checking again
            print("All points have been inserted...")
            self.main_thread.stop()

    def setfps(self, fps):
        assert fps > 0, "[3DModel] FPS > 0"
        self.fps = fps

    
    def _save_camera_sources(self, point3d, cam_fid_pairs):
        src_map = {c: fid for c, fid in cam_fid_pairs}

        new_row = pd.DataFrame([[point3d.fid, point3d.visibility, point3d.x, point3d.y, point3d.z, point3d.event, point3d.timestamp, json.dumps(src_map, separators=(',', ':'))]], columns=["Frame", "Visibility", "X", "Y", "Z", "Event", "Timestamp", "Camera"])
        self.camera_source = pd.concat([self.camera_source, new_row], ignore_index=True)
    
    def update_heartbeat(self, cam: str, last_ts: float):
        """Receiver 回報 heartbeat（水位線）。"""
        now = time.time()
        prev = self.cam_last_ts.get(cam, float('-inf'))
        if last_ts is not None and last_ts >= prev:
            self.cam_last_ts[cam] = float(last_ts)
        self.cam_last_seen[cam] = now
        # logging.debug(f"[Update Heartbeat] cam: {cam}, {prev} -> {last_ts} at {now}")

    def _low_watermark(self):
        """只用「仍活著」的相機算最低水位線。"""
        now = time.time()
        active_vals = []
        for raw in self.rawTrack2Ds:
            cam = raw.name
            last_seen = self.cam_last_seen.get(cam)
            if last_seen is None:
                continue
            if (now - last_seen) <= self.STALE_TIMEOUT:
                ts = self.cam_last_ts.get(cam)
                if ts is not None:
                    active_vals.append(ts)
        if not active_vals:
            return None
        return min(active_vals)

    def _peek_min_candidate_ts(self):
        """看各相機 queue 頭部，找最早的候選時間。"""
        cands = []
        for raw in self.rawTrack2Ds:
            if len(raw.points) > 0:
                cands.append(raw.points[0].timestamp)
        return min(cands) if cands else None
    
    def run(self):
        logging.info("TriangulationThread started.")

        self.alive = True
        self.prev_calculate_time = time.time()
        # start receivers
        for rawTrack2D in self.rawTrack2Ds:
            rawTrack2D.startReceiver()

        prev_point3d = None
        prev_calculate_time = float("inf")

        sendNodeStateMsg(self.client, self.nodename, "ready")

        self._dbg_prev = {
            "min_ts": None,
            "min_cam": None,
            "min_fid": None,
            "force": None,
        }
        Track3D = []
        while self.alive:
            # logging.debug('===============================================================')
            # get point from receiver and to do interpolation
            for rawTrack2D in self.rawTrack2Ds:
                # rawTrack2D.doInterpolation()
                rawTrack2D.insertPoints()

            # Do Triangulation
            points_2D = []
            points_2D_info = []

            # Only Several Cameras detect the ball, [0,1,3] means idx 0,1,3 cams detect, 2 misses
            cam_detected_ball = []
            cam_fid_pairs = [] # (cam_idx, fid)

            # search smallest timestamp which have detected the ball
            queueAllNotEmpty = True
            force_3d_flag = False
            min_timestamp = float("inf")
            min_timestamp_cam = None
            min_timestamp_fid = None

            current_time = datetime.now().timestamp()
            for rawTrack2D in self.rawTrack2Ds:
                if len(rawTrack2D.points) > 0:
                    if min_timestamp > rawTrack2D.points[0].timestamp:
                        min_timestamp = rawTrack2D.points[0].timestamp
                        min_timestamp_cam = rawTrack2D.name
                        min_timestamp_fid = rawTrack2D.points[0].fid            

            reason = None
            low_wm = self._low_watermark()
            if low_wm is not None and (low_wm - min_timestamp) >= self.SKEW_TOL:
                # 最慢相機的處理進度已經「超過」這個候選時間 → 不必等了
                force_3d_flag = True
                reason = "WATERMARK"

            # If a camera always miss, no wait if the time exceed WAIT_TIME
            if current_time - prev_calculate_time >= self.WAIT_TIME:
               force_3d_flag = True
               reason = "TIMEOUT"

            for rawTrack2D in self.rawTrack2Ds:
                if len(rawTrack2D.points) <= 0:
                    queueAllNotEmpty = False

            # === Debug log ===
            # wm_val = low_wm if low_wm is not None else float("nan")
            # # 只在「狀態改變」或「到達節流時間」才印
            # changed = (
            #     self._dbg_prev["min_ts"] != min_timestamp or
            #     self._dbg_prev["min_cam"] != min_timestamp_cam or
            #     self._dbg_prev["min_fid"] != min_timestamp_fid or
            #     self._dbg_prev["force"]  != force_3d_flag
            # )
            # if changed:
            #     logging.debug(" ")
            #     logging.debug(
            #         f"[3D] min_timestamp: {min_timestamp:.6f}, by cam: {min_timestamp_cam}, fid: {min_timestamp_fid} "
            #         f"wm={wm_val:.6f} force={int(force_3d_flag)}"
            #     )
            #     if reason == "WATERMARK":
            #         logging.debug("[DECIDE] WATERMARK")
            #     elif reason == "TIMEOUT":
            #         logging.debug("[DECIDE] TIMEOUT")
            #     self._dbg_prev.update({
            #         "min_ts":  min_timestamp,
            #         "min_cam": min_timestamp_cam,
            #         "min_fid": min_timestamp_fid,
            #         "force":   force_3d_flag,
            #     })
            # =================
            
            point3d_fid = None

            if queueAllNotEmpty or force_3d_flag: # Do 3D Triangulation
                # logging.debug(f" ")
                # logging.debug(f"--- Start time-aligned --- {time.time() - self.start_time}")
                for i in range(len(self.rawTrack2Ds)):
                    # logging.debug(f"{self.rawTrack2Ds[i].name}, len: {len(self.rawTrack2Ds[i].points)}")
                    if len(self.rawTrack2Ds[i].points) > 0:
                        point = self.rawTrack2Ds[i].points[0]
                        # logging.debug(f"{point.fid}: {point.timestamp}")
                        cam_idx = self.rawTrack2Ds[i].name
                        # logging.debug('***')
                        # logging.debug(f"cam: {cam_idx}, fid: {point.fid}")
                        # logging.debug(f"{point.timestamp}, {abs(min_timestamp - point.timestamp)}")
                        # logging.debug(abs(min_timestamp - point.timestamp) <= self.MAXIMUM_FRAME_TIMESTAMP_DELAY)
                        # logging.debug('***')
                        if abs(min_timestamp - point.timestamp) <= self.MAXIMUM_FRAME_TIMESTAMP_DELAY:

                            if point3d_fid is None:
                                point3d_fid = point.fid # The FID in which Camera idx is the smallest (In coachbox, it's always CameraL)

                            points_2D.append([point.x, point.y])
                            points_2D_info.append([cam_idx, point.fid, point.timestamp])
                            cam_detected_ball.append(i)
                            cam_fid_pairs.append((cam_idx, point.fid))

                            writer = next((w for w in self.csv2DWriters if w.name == self.rawTrack2Ds[i].name), None)
                            if writer:
                                writer.writePoints(point)

                            # logging.debug('cam {} detected_ball in frame_id {} :'.format(i, self.rawTrack2Ds[i].points[0].fid))
                            self.rawTrack2Ds[i].points.pop(0)
                            prev_calculate_time = datetime.now().timestamp()
                # logging.debug(f"--- Finish time-aligned --- {time.time() - self.start_time}")
                if cam_detected_ball:
                    cam_detected_ball = np.stack(cam_detected_ball, axis=0)

                # debug
                # if points_2D_info:
                #     logging.debug('--------------------------------------------------------------')
                #     for i, point in enumerate(points_2D_info):
                #         logging.debug(f"[P{i+1}]: cam: {points_2D_info[i][0]}, fid: {points_2D_info[i][1]}, timestamp: {points_2D_info[i][2]}")
                #     logging.debug('--------------------------------------------------------------')
                
                # to generate point 3d
                if len(points_2D) >= 2:

                    track_2D = np.array(points_2D, dtype = np.float32) # shape:(num_cam,num_frame,2), num_frame=1
                    undistort_track2D_list = []
                    for i in range(len(points_2D)): # for each camera, do undistort
                        temp = cv2.undistortPoints(np.array(track_2D[i], np.float32),
                                                    np.array(self.ks[cam_detected_ball[i]], np.float32),
                                                    np.array(self.dist[cam_detected_ball[i]], np.float32),
                                                    None,
                                                    np.array(self.newcameramtx[cam_detected_ball[i]], np.float32)) # shape:(1,num_frame,2), num_frame=1
                        temp = temp.reshape(-1,2) # shape:(num_frame,2), num_frame=1
                        undistort_track2D_list.append(temp)
                    undistort_track2D = np.stack(undistort_track2D_list, axis=0) # shape:(num_cam,num_frame,2), num_frame=1

                    ###### Shao-Ping #####################################################################
                    # self.multiCamTriang.setTrack2Ds(track_2D)
                    # self.multiCamTriang.setPoses(self.poses[cam_detected_ball])
                    # self.multiCamTriang.setEye(self.eye[cam_detected_ball])
                    # self.multiCamTriang.setKs(self.ks[cam_detected_ball])
                    # track_3D = self.multiCamTriang.calculate3D()

                    ###### Ours ##########################################################################
                    # logging.debug('')
                    # logging.debug(f"--- Start 3D cal --- {time.time() - self.start_time}")
                    self.multiCamTriang.setTrack2Ds(undistort_track2D)
                    self.multiCamTriang.setCamInfo(cam_fid_pairs)
                    self.multiCamTriang.setProjectionMats(self.projection_mat[cam_detected_ball])
                    track_3D = self.multiCamTriang.rain_calculate3D_robust() # shape:(num_frame,3), num_frame=1
                    # logging.debug(f"--- Finish 3D cal --- {time.time() - self.start_time}")
                    if track_3D is None:
                        # logging.debug("[TriangulationThread] Skip frame due to all-outlier 3D.")
                        continue
                    ######################################################################################

                    # Use Timestamp to triangulation, so fid is not correct [*]
                    point3d = Point(fid=point3d_fid,
                                    timestamp=min_timestamp,
                                    visibility=1,
                                    x=track_3D[0][0],
                                    y=track_3D[0][1],
                                    z=track_3D[0][2],
                                    color='white')
                    is_in_bounds = isWithinBounds(point3d)
                    # logging.debug(f"[calPoint] {point3d.fid}, {point3d.timestamp}, ({point3d.x}, {point3d.y}, {point3d.z})")
                    if not is_in_bounds:
                        continue
                    # print('-->>>>> point', point3d.fid, point3d.timestamp, point3d.x, point3d.y, point3d.z)
                    
                    self.prev_calculate_time = time.time()

                    Track3D.append(deepcopy(point3d))
                    self.csv3DWriter_points.writePoints(point3d)
                    self._save_camera_sources(point3d, cam_fid_pairs)

                    new_sendPoints(self.data_handler, "point", point3d)
                    # sendPoints(self.client, f"{self.topic}/Point", point3d)

                    # Ball position (X,Z) when pass above the net (Y=0)
                    # if prev_point3d is not None and (prev_point3d.y*point3d.y) <= 0:
                    #     # TODO if the ball hit the net may detect serveral times
                    #     tmpx = (prev_point3d.x - point3d.x)/(prev_point3d.y - point3d.y) * (0 - point3d.y) + point3d.x
                    #     tmpz = (prev_point3d.z - point3d.z)/(prev_point3d.y - point3d.y) * (0 - point3d.y) + point3d.z
                    #     logging.info(f"Ball Pass Above the Net between fid({prev_point3d.fid},{point3d.fid}): ({tmpx:.2f},{tmpz:.2f})")
                    # prev_point3d = point3d

                    # event detect
                    # self.eventDetector.addPoint(point3d)
                    # self.eventDetector.detect()
                    # logging.debug(f"--- Send 3D {point3d.fid}: {point3d.timestamp} --- {time.time() - self.start_time}")
                    # print(f"--- Send 3D {point3d.fid}: {point3d.timestamp} --- {time.time() - self.start_time}")
                    self.flyEventDetector.addPoint(point3d)
                    self.flyEventDetector.detect()

            else:
                time.sleep(0.5/self.fps) # less than 1/fps

            # logging.debug('===============================================================')

        # stop receivers
        for rawTrack2D in self.rawTrack2Ds:
            rawTrack2D.stopReceiver()

        # self.eventDetector.close()
        for i in range(3):
            # logging.debug(f"Waiting for {5 - i} second to close detector")
            time.sleep(1)
        self.flyEventDetector.close()

        self._export_pose_records()

        # Output to CSV
        for w in self.csv2DWriters:
            w.close()

        filePath_source = f"{self.save_path}/{self.nodename}_source.csv"
        self.camera_source.to_csv(filePath_source, index=False)
        
        if len(Track3D) > 0:
            drawPoints(Track3D, self.save_path)

        if self.mode == 'CES':
            runFireBall(self.client, self.topic, self.save_path, Track3D, self.fps)

        logging.info("TriangulationThread terminated.")

    def _make_pose_callback(self, camera_name, topic):
        def _handle(payload):
            if isinstance(payload, dict) and payload.get("EOF"):
                logging.info("Pose stream EOF on %s", topic)
                return

            detections = []
            if isinstance(payload, dict):
                detections = payload.get("detections", []) or []
                frame_idx = payload.get("frame_index")
                logging.debug(
                    "[PoseReceiver] %s frame=%s detections=%d",
                    camera_name,
                    frame_idx,
                    len(detections),
                )
            else:
                logging.debug("[PoseReceiver] %s payload=%s", camera_name, payload)

            with self.pose_lock:
                self.pose_records.setdefault(camera_name, []).append(payload)

        return _handle

    def _export_pose_records(self):
        base_path = Path(self.save_path)
        with self.pose_lock:
            pose_items = list(self.pose_records.items())

        for camera_name, records in pose_items:
            if not records:
                continue
            output_path = base_path / f"{camera_name}_pose.jsonl"
            with output_path.open("w", encoding="utf-8") as fh:
                for record in records:
                    fh.write(json.dumps(record))
                    fh.write("\n")
            logging.info(
                "Saved %d pose frames for %s to %s",
                len(records),
                camera_name,
                output_path,
            )

class MainThread(threading.Thread):
    def __init__(self, client, data_handler, args, settings, ks, poses, eye, dist, newcameramtx, projection_mat, enable_camera_sync=False):
        threading.Thread.__init__(self)
        self.killswitch = threading.Event()
        self.client = client
        self.data_handler = data_handler
        self.nodename = args.nodename

        # ToDo: check num of ks, pose, eye is equal to number of input_topic
        self.triangulation = TriangulationThread(self.client, self.data_handler, self, args, settings, ks, poses, eye, dist, newcameramtx, projection_mat)
        
        self.canerasync = None
        if enable_camera_sync:
            self.canerasync = CameraSync(self.client, list(args.camera_mapping.values()))


    # def stop(self):
    #     finish = False
    #     while not finish:
    #         for rawTrack2D in self.triangulation.rawTrack2Ds:
    #             if len(rawTrack2D.points) <= 0:
    #                 finish = True
    #                 break
    #     self.killswitch.set()
    def stop(self):
        while any(len(rt.points) > 0 or len(rt.receiver.queue) > 0 for rt in self.triangulation.rawTrack2Ds):
            time.sleep(0.01)
        time.sleep(1)
        self.killswitch.set()

    def on_connect(self, client, userdata, flag, rc):
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

    def run(self):
        logging.info("{} started.".format(self.nodename))
        # start
        try:
            if self.canerasync:
                self.canerasync.start()
                logging.info("CameraSync started.")

            self.triangulation.start()
            self.client.loop_start()
            logging.info("{} is ready.".format(self.nodename))

            self.killswitch.wait()
        finally:
            if self.canerasync:
                self.canerasync.stop()
                self.canerasync.join()
                logging.info("CameraSync stopped.")

            self.triangulation.stop()
            self.triangulation.join()
            sendNodeStateMsg(self.client, self.nodename, "terminated")
            # self.client.loop_stop()

        # end
        logging.info("{} terminated.".format(self.nodename))


class MainThreadManager:
    def __init__(self, data_handler, mqttc:mqtt.Client):
        self.main_thread = None
        self.mqttc:mqtt.Client = mqttc
        self.data_handler = data_handler

    def start_main_thread(self, date, data, mode):
        print(f"!!!!!!!!!!{date}, {data}, {mode}")
        print(prepare_main_thread(date, data, mode))
        print('##### strat main thread #####')
        args, settings, ks, poses, eye, dist, newcameramtx, projection_mat = prepare_main_thread(date, data, mode)
        if self.main_thread is None or not self.main_thread.is_alive():
            self.main_thread = MainThread(self.mqttc, self.data_handler, args, settings, ks, poses, eye, dist, newcameramtx, projection_mat)
            self.main_thread.start()
            if self.main_thread.is_alive():
                return_payload = {"status": "Ready"}
            else:
                return_payload = {"status": "Fail"}
        else:
            return_payload = {"status": "Fail"}
        return return_payload
    
    def stop_main_thread(self):
        print('##### stop main thread #####')
        if self.main_thread is not None:
            self.main_thread.stop()
            self.main_thread.join()
            self.main_thread = None
            return_payload = {"status": "Stop"}
            print("MainThread has been stopped.")
            # self.mqttc.loop_stop()
        else:
            return_payload = {"status": "Fail"}
        return return_payload
    
    def data_feeder(self, file_path=f'{ROOTDIR}/LayerContent/content_output_sample.csv'):
        print('##### data feeder #####')
        # projectCfg = f"{ROOTDIR}/config"
        # settings = loadNodeConfig(projectCfg, 'Model3D')
        # broker = (settings['mqtt_broker'], int(settings['mqtt_port']))
        # self.client = mqtt.Client()
        # self.client.connect(*broker)
        # output_topic = '/DATA/ContentDevice/ContentLayer/Model3D'
        with open(file_path, 'r', newline='') as csvFile:
            rows = list(csv.DictReader(csvFile))
            for row in rows:
                json_str = {'shot_id': int(row['ShotID']), 
                            'timestamp': float(row['Timestamp']) if row['Timestamp'] else None,
                            'event': int(row['Event']) if row['Event'] else None,
                            'position': json.loads(row['Position']) if row['Position'] else None,
                            'speed': json.loads(row['Speed']) if row['Speed'] else None
                            }
                self.data_handler.publish("event", json_str)
                # print(f"[ContentLayer] Published on topic '{output_topic}': {json_str}")
        return_payload = {"status": "Finish"}
        return return_payload
            

def runFireBall(client, topic, save_path, points, fps):
    if len(points) > 0:
        points = removeOuterPoint_batch(save_path, points)
        points = removeMotionOutliers_batch(save_path, points)
        points = detectEvent_batch(save_path, points)
        drawTrajectory_points(save_path, points)
        result = gradient_speed(save_path, points, fps)
        print('====== SPEED ======')

        if isinstance(result, str):
            print(result)
            payload = {
                "shot_id": -1,
                "timestamp": -1.0,
                "event": -1,
                "position": [-1.0, -1.0, -1.0],
                "speed": [-1.0, -1.0, -1.0]         
            }
        else:
            hit_point = result[0]
            speed = list(result[1])
            points = result[2]
            drawGradientTrajectory_points(save_path, points)
            print(speed)
            payload = {
                "shot_id": 1,
                "timestamp": hit_point.timestamp,
                "event": hit_point.event,
                "position": [hit_point.x, hit_point.y, hit_point.z],
                "speed": speed          
            }
    else:
        payload = {
                "shot_id": -1,
                "timestamp": -1.0,
                "event": -1,
                "position": [-1.0, -1.0, -1.0],
                "speed": [-1.0, -1.0, -1.0]               
            }
    client.publish(topic, json.dumps(payload))
    print(f"[ContentLayer] Published on topic '{topic}': {payload}")

    output_path = os.path.join(save_path, 'Model3D_output.csv')
    fieldnames = ['ShotID', 'Timestamp', 'Event', 'Position', 'Speed']

    with open(output_path, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        data_to_write = {
            'ShotID': payload['shot_id'],
            'Timestamp': payload['timestamp'],
            'Event': payload['event'],
            'Position': payload['position'],
            'Speed': payload['speed']
        }
        writer.writerow(data_to_write)

    drawVisualization(os.path.basename(save_path))

def prepare_main_thread(date, data, mode=None):
    project = 'coachbox'
    nodename = 'Model3D'
    projectCfg = f"{ROOTDIR}/config"
    settings = loadNodeConfig(projectCfg, nodename)

    input_topics = []
    ks, poses, eye, dist, newcameramtx, projection_mat = [], [], [], [], [], []
    camera_mapping = {}
    fps = float('inf')

    if isinstance(data, list):
        for entry in data:
            camera_idx = entry.get('idx')
            camera_device = entry.get('device')
            camera_fps = entry.get('fps')
            parameters = entry.get('parameters')

            if camera_idx is not None and camera_device and camera_fps and parameters:
                camera_mapping[camera_idx] = camera_device

                tracknet_topic = f"/DATA/{camera_device}/SensingLayer/TrackNet"
                input_topics.append(tracknet_topic)

                ks.append(parameters.get('ks'))
                poses.append(parameters.get('poses'))
                eye.append(parameters.get('eye'))
                dist.append(parameters.get('dist'))
                newcameramtx.append(parameters.get('newcameramtx'))
                projection_mat.append(parameters.get('projection_mat'))

                fps = min(fps, camera_fps)

        ks = np.array(ks)
        poses = np.array(poses)
        eye = np.array(eye)
        dist = np.array(dist)
        newcameramtx = np.array(newcameramtx)
        projection_mat = np.array(projection_mat)

        settings['nodename'] = nodename
        settings['fps'] = fps
        settings['input_topic'] = ','.join(input_topics)
        settings['output_topic'] = '/DATA/ContentDevice/ContentLayer/Model3D'

    args = argparse.Namespace(
        project=project,
        nodename=nodename,
        camera_mapping=camera_mapping,
        save_path=f'{ROOTDIR}/replay/{date}/',
        mode=mode
    )

    return args, settings, ks, poses, eye, dist, newcameramtx, projection_mat


class ContentLayerAgent(MqttAgent):
    def __init__(self, device_name:str):
        super().__init__(device_name, "ContentLayer")
        
    def on_connect(self, client:mqtt.Client, userdata, flags, reason_code, properties):
        super().on_connect(client, userdata, flags, reason_code, properties)

        # 註冊 Content Layer 所發布的資料流的Topic，未來可以透過較短的稱呼(第一個參數)索引到Topic的全稱
        self.data_handler.register_topic("point", "Model3D/Point")
        self.data_handler.register_topic("event", "Model3D/Event")
        self.data_handler.register_topic("segment", "Model3D/Segment")

def parse_args() -> Optional[str]:
    # Args
    parser = argparse.ArgumentParser(description = 'Model3D')
    parser.add_argument('--project', type=str, default = 'coachbox', help = 'project name (default: coachbox)')
    parser.add_argument('--nodename', type=str, default = 'Model3D', help = 'mqtt node name (default: 3DModel)')
    # parser.add_argument('--save_path', type=str, default = './', help = 'csv path (default: replay/XX/)')
    parser.add_argument('--date', type=str, required=True, help='default: replay/XX/')
    parser.add_argument('--camera_idxs', type=int, nargs='+', required=True, help='camera indices (e.g., 0 1 2)')
    parser.add_argument('--camera_device', type=str, nargs='+', required=True, help='camera serial numbers (e.g., S1 S2 S3)')
    parser.add_argument('--fps', type=int, default = '-1.0')
    parser.add_argument('--mode', type=str, default = ' ')
    args = parser.parse_args()

    return args

def main():
    # Parse arguments
    args = parse_args()

    if len(args.camera_idxs) != len(args.camera_device):
        raise ValueError("The number of camera indices must match the number of camera serials.")
    
    camera_mapping = {idx: serial for idx, serial in zip(args.camera_idxs, args.camera_device)}
    
    projectCfg = f"{ROOTDIR}/config"
    settings = loadNodeConfig(projectCfg, args.nodename)
    logging.getLogger().setLevel(logging.DEBUG)

    if args.fps != -1.0:
        settings['fps'] = args.fps

    input_topics = []
    ks, poses, eye, dist, newcameramtx, projection_mat = [], [], [], [], [], []

    for idx, serial in zip(args.camera_idxs, args.camera_device):
        input_topic = f"/DATA/{serial}/SensingLayer/TrackNet"
        input_topics.append(input_topic)

        replay_dir = f"{ROOTDIR}/replay/{args.date}"
        if os.path.exists(f"{replay_dir}/CameraReader_{idx}.cfg"):
            camera_config_file = f"{replay_dir}/CameraReader_{idx}.cfg"
        elif os.path.exists(f"{replay_dir}/{serial}.cfg"):
            camera_config_file = f"{replay_dir}/{serial}.cfg"
        cfg = loadConfig(camera_config_file)
        ks.append(np.array(json.loads(cfg['Other']['ks'])))
        poses.append(np.array(json.loads(cfg['Other']['poses'])))
        eye.append(np.array(json.loads(cfg['Other']['eye'])))
        dist.append(np.array(json.loads(cfg['Other']['dist'])))
        newcameramtx.append(np.array(json.loads(cfg['Other']['newcameramtx'])))
        projection_mat.append(np.array(json.loads(cfg['Other']['projection_mat'])))

    ks = np.array(ks)
    poses = np.array(poses)
    eye = np.array(eye)
    dist = np.array(dist)
    newcameramtx = np.array(newcameramtx)
    projection_mat = np.array(projection_mat)

    settings['input_topic'] = ','.join(input_topics)
    settings['output_topic'] = '/DATA/ContentDevice/ContentLayer/Model3D'

    args = argparse.Namespace(
            project=args.project,
            nodename=args.nodename,
            camera_mapping=camera_mapping,
            save_path=f'{ROOTDIR}/replay/{args.date}/',
            mode=args.mode
        )
    
    # mqtt_agent = MqttAgent("ContentDevice", "ContentLayer")
    # mqtt_agent.start('140.113.208.125', 1884)

    mqtt_agent = ContentLayerAgent("ContentDevice")
    mqtt_agent.start('140.113.208.125', 1884)

    
    # Start MainThread
    mainThread = MainThread(mqtt_agent.mqttc, mqtt_agent.data_handler, args, settings, ks, poses, eye, dist, newcameramtx, projection_mat)
    mainThread.start()
    mainThread.join()

if __name__ == '__main__':
    main()
