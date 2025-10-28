import os
import sys
import numpy as np
import cv2

DIRNAME = os.path.dirname(os.path.abspath(__file__))
ROOTDIR = os.path.dirname(os.path.dirname(DIRNAME))
sys.path.append(f"{ROOTDIR}/lib")
from point import Point

class LandPointDetect():
    def __init__(self, ks, hmtx, cameramtx, dist):
        self.ks = ks
        self.hmtx = hmtx
        self.cameramtx = cameramtx
        self.dist = dist
        self.yaw = -1
        self.pitch = -1
        self.speed = -1

        self.trajectory = []
        self.landPoints = []
        self.duration = []
        self.dropping = False
        self.detected = False
        self.startPoint = None

    def settings(self, yaw, pitch, speed):
        self.yaw = yaw
        self.pitch = pitch
        self.speed = speed

    def insertPoint(self, point:Point):
        if point.visibility == 1:
            if self.startPoint == None:
                self.startPoint = point
            if (len(self.trajectory) > 3) and (not self.dropping) and (not self.detected):
                if (point.y > self.trajectory[-1].y) and (self.trajectory[-1].y > self.trajectory[-2].y) \
                    and (self.trajectory[-2].y > self.trajectory[-3].y):
                    self.dropping = True
            elif self.dropping:
                if point.y < self.trajectory[-1].y:
                    self.detected = True
                    self.dropping = False
                    landPoint_court = self.calLandPoint(self.trajectory[-1])
                    #self.landPoints.append(Point(fid=self.trajectory[-1].fid, x=landPoint_court[0], y=landPoint_court[1]))
                if np.linalg.norm(point.toXY() - self.trajectory[0].toXY()) < np.linalg.norm(point.toXY() - self.trajectory[-1].toXY()):
                    print(f'new trajectory at frame:{point.fid}')
                    if not self.detected:
                        landPoint_court = self.calLandPoint(self.trajectory[-1])
                        #self.landPoints.append(Point(fid=self.trajectory[-1].fid, x=landPoint_court[0], y=landPoint_court[1]))
                    self.dropping = False
                    self.detected = False
                    self.trajectory.clear()
            elif self.detected:
                if np.linalg.norm(point.toXY() - self.trajectory[0].toXY()) < np.linalg.norm(point.toXY() - self.trajectory[-1].toXY()):
                    print(f'new trajectory at frame:{point.fid}')
                    self.trajectory.clear()
                    self.detected = False

            self.trajectory.append(point)

    def calLandPoint(self, point:Point):
        self.duration.append(point.timestamp - self.startPoint.timestamp)
        landPoint_pixel = point.toXY()
        undistort_point = cv2.undistortPoints(landPoint_pixel, self.ks, self.dist, None, self.cameramtx)
        landPoint_court = self.hmtx @ np.append(undistort_point, 1)
        landPoint_court = landPoint_court / landPoint_court[2]
        self.landPoints.append(Point(fid=point.fid, x=landPoint_court[0], y=landPoint_court[1]))
        return landPoint_court

    def getLandPoints(self):
        if not self.detected and len(self.trajectory) > 1:
            landPoint_court = self.calLandPoint(self.trajectory[-1])
            #self.landPoints.append(Point(fid=self.trajectory[-1].fid, x=landPoint_court[0], y=landPoint_court[1]))
        self.dropping = False
        self.detected = False
        return self.landPoints, self.duration