import math
import numpy as np
from scipy import optimize
from calcDis import PredictDis, PredictLandDis
from scipy.optimize import curve_fit

def calibrateFunc(x, a, b):
    return a * x + b

class CircleFit():
    def __init__(self, points_x, points_y, dis=None):
        self.points_x = points_x
        self.points_y = points_y
        self.dis = dis

    def getCenter(self):
        center_estimate = 0, 0
        center_2, ier = optimize.leastsq(self.f_2, center_estimate)

        return center_2

    def calc_R(self, xc, yc):
        """ calculate the distance of each 2D points from the center (xc, yc) """
        return [np.sqrt((x-xc)**2 + (y-yc)**2) for x, y in zip(self.points_x, self.points_y)]

    def f_2(self, c):
        """ calculate the algebraic distance between the data points and the mean circle centered at c=(xc, yc) """
        Ri = self.calc_R(*c)
        result = np.array([], dtype=np.float32)
        if self.dis is None:
            for r in Ri:
                result = np.append(result, r-r.mean())
        else:
            for r, dis in zip(Ri, self.dis):
                result = np.append(result, r-dis)
        return result

class Calibrate():
    
    def __init__(self):
        self.X = None
        self.Y = None
        self.H = None
        self.yawPar = None
        self.pitchPar = None
        self.speedPar = None

    def setCoordinate(self, x, y, h):
        self.X = x
        self.Y = y
        self.H = h

    def distanceEstimate(self, speed, pitch, landTime):
        fps = 60
        h = 0
        while True:
            dis, time = PredictLandDis(h, [[speed, pitch]], fps=fps)
            loss = time[0] - landTime
            error = abs(loss)
            if error < (1 / fps) / 2:
                self.H = h
                return h, dis
            d_dis, d_time = PredictLandDis(h + 0.1, [[speed, pitch]], fps=fps)
            dh = d_time[0] - time[0]
            h = h - 0.1 * (loss / dh)

    def findStartPoint(self, dropPoints_x, dropPoints_y, dis=None):
        '''
        if dropPoints.shape[0] < 3:
            raise Exception('length of \"dropPoints\" should be larger than 3')
        if yaws.shape[0] != dropPoints.shape[0]:
            raise Exception('length of \"yaws\" and \"dropPoints\" should be the same')
        '''
        circleFit = CircleFit(dropPoints_x, dropPoints_y, dis)
        xc, yc = circleFit.getCenter()
        self.X = xc
        self.Y = yc

        '''
        realYaws = []
        for x, y in dropPoints:
            dx = x - self.X
            dy = y - self.Y
            if dy < 0:
                dx = -dx
                dy = -dy
            angle = 90 - math.atan2(dy, dx) / math.pi * 180 
            realYaws.append(angle)
        '''
        #self.yawPar, pcov = curve_fit(calibrateFunc, yaws, realYaws)
                
    def distanceCalibration(self, speeds, pitches, dis, height=1.5, lr=0.01):
        # speeds = np.array(settings of speed) # shape: (m, 2)
        # pitched = np.array(settings of pitch) # shape: (n, 2)
        # dropPoints = np.array(2D coordinate of drop points) # shape: (m, n, 2)
        '''
        if self.X is None or self.Y is None:
            raise Exception('You need the coordinate of starting point to do calibration')
        if dropPoints.shape[0] != (speeds.shape[0] or dropPoints.shape[1] != pitches.shape[0]):
            raise Exception('shape of \"settings_dropPoint\" should be ({length of speed},{length of pitches},2)')
        if dropPoints.shape[2] != 2:
            raise Exception('shape of \"settings_dropPoint\" should be ({length of speed},{length of pitches},2)')
        '''
        GTdistance = dis
        n = len(speeds)
        m = len(pitches)
        '''
        for i in range(dropPoints.shape[0]):
            for j in range(dropPoints.shape[1]):
                x, y = dropPoints[i][j]
                distance = np.linalg.norm([self.X-x, self.Y-y])
                GTdistance.append([distance])
        '''

        guessSpeeds = speeds
        guessPitches = pitches
        guessHeight = height
        iter = 0
        while(iter < 10000):
            settings = []
            for i in range(guessSpeeds.shape[0]):
                for j in range(guessPitches.shape[0]):
                    settings.append([guessSpeeds[i], guessPitches[j]])
            predictDistance = PredictDis(guessHeight, 0, settings)
            error = np.linalg.norm(np.array(predictDistance) - np.array(GTdistance))
            if error < 0.05:
                break
        
            #adjust Height
            prevHeight = guessHeight
            loss = (1 / n*m) * np.sum(np.array(predictDistance) - np.array(GTdistance))
            dh = (1 / n*m) * np.sum(np.array(PredictDis(guessHeight + 0.1, 0, settings)) - np.array(GTdistance))
            guessHeight = guessHeight - lr * 0.1 * (loss / dh)

            #adjust Speed
            prevSpeeds = guessSpeeds
            for i in range(n):
                newSpeed = guessSpeeds[i] + 0.1
                ori_settings = []
                new_settings = []
                GT = []
                for j in range(guessPitches.shape[0]):
                    ori_settings.append([guessSpeeds[i], guessPitches[j]])
                    new_settings.append([newSpeed, guessPitches[j]])
                    '''
                    x, y = dis[i][j]
                    distance = np.linalg.norm([self.X-x, self.Y-y])
                    '''
                    distance = dis[i][j]
                    GT.append([distance])
                ori_pred = PredictDis(prevHeight, 0, ori_settings)
                loss = (1 / n) * np.sum(np.array(ori_pred) - np.array(GT))
                new_pred = PredictDis(prevHeight, 0, new_settings)
                d_v = (1 / n) * np.sum(np.array(new_pred) - np.array(ori_pred))
                guessSpeeds[i] = guessSpeeds[i] - lr * 1 * (loss / d_v)

            #adjust Pitch
            for j in range(m):
                newPitch = guessPitches[j] + 0.1
                ori_settings = []
                new_settings = []
                GT = []
                for i in range(prevSpeeds.shape[0]):
                    ori_settings.append([prevSpeeds[i], guessPitches[j]])
                    new_settings.append([prevSpeeds[i], newPitch])
                    '''
                    x, y = dis[i][j]
                    distance = np.linalg.norm([self.X-x, self.Y-y])
                    '''
                    distance = dis[i][j]
                    GT.append([distance])
                ori_pred = PredictDis(prevHeight, 0, ori_settings)
                loss = (1 / m) * np.sum(np.array(ori_pred) - np.array(GT))
                new_pred = PredictDis(prevHeight, 0, new_settings)
                d_pitch = (1 / m) * np.sum(np.array(new_pred) - np.array(ori_pred))
                guessPitches[j] = guessPitches[j] - lr * 1 * (loss / d_pitch)

            iter += 1

        return guessHeight, guessSpeeds, guessPitches