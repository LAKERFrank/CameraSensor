import sys
import os
import logging
import numpy as np
import cv2
import math
import json
import random
import pandas as pd
import subprocess
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5 import QtTest

DIRNAME = os.path.dirname(os.path.abspath(__file__))
UIDIR = os.path.dirname(DIRNAME)
ROOTDIR = os.path.dirname(UIDIR)

sys.path.append(f"{DIRNAME}/../")
from UISettings import *
from Services import SystemService, MsgContract

sys.path.append(f"{ROOTDIR}/ServeMachine/PC")
from MachineClient import MachineClient
from LandPointDetect import LandPointDetect
from calcDis import physics_predict3d_v2
from Calibrate import Calibrate

sys.path.append(f"{ROOTDIR}/lib")
from message import *
from common import loadConfig
from point import Point
from nodes import TrackNet_offline

def inv_speed_func(y, a):
    return y / a

def inv_pitch_func(y, a, b):
    return (y - b) / a

speed_popt = np.array([1.30937525])
pitch_popt = np.array([0.57042194, -0.66784551])

class CalibrationPage(QGroupBox):
    drawImageSignal = pyqtSignal(int, QPixmap)

    def __init__(self, camera_widget):
        super().__init__()

        self.camera_widget = camera_widget
        self.image_size = QSize(800, 600)

        self.is_recording = False

        self.recordDir = None
        self.generateHomography()
        self.info = {}
        self.land_info = {}
        self.land_dis = {}

        self.machineClient = MachineClient("140.113.213.131")
        self.machineStatus = self.machineClient.connect("MachineA")

        self.debugMode = False

    def hideEvent(self, event):
        logging.debug(f"{self.__class__.__name__}: hided.")
        self.camera_widget.stopStreaming()
        self.deleteUI()

    def showEvent(self, event):
        logging.debug(f"{self.__class__.__name__}: shown.")
        self.setupUI()
        self.camera_widget.startStreaming()
        camera_cfg_path = f'{ROOTDIR}/Reader/Image_Source/config/40224283.cfg'
        self.camera_cfg = loadConfig(camera_cfg_path)
        self.ks = np.array(json.loads(self.camera_cfg['Other']['ks']), np.float32)
        self.hmtx = np.array(json.loads(self.camera_cfg['Other']['hmtx']), np.float32)
        self.cameramtx = np.array(json.loads(self.camera_cfg['Other']['newcameramtx']), np.float32)
        self.dist = np.array(json.loads(self.camera_cfg['Other']['dist']), np.float32)

    def setBackgroundService(self, service:SystemService):
        self.myService = service

    def setupUI(self):
        # main layout
        self.layout_main = QGridLayout()

        self.control_bar = self.getControlBar()

        machineInfo = QLabel()
        machineInfo.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        machineInfo.setFixedSize(QSize(2000, 60))
        machineInfo.setStyleSheet(UIStyleSheet.ContentText)
        if self.machineStatus:
            machineInfo.setText("成功連線至發球機")
        else:
            machineInfo.setText("尚未連線至發球機")

        machinePanel = self.machinePanel()

        self.camera_widget.initWidget(1, self.image_size, [2])

        # can turn off FPS display
        # self.camera_widget.toggleFPS()

        # connect signals
        self.camera_widget.getImageSignal.connect(self.receiveImage)
        self.drawImageSignal.connect(self.camera_widget.trueDrawImage)

        info_GroupBox = self.infoGroupBox()
        self.court = QLabel()
        self.court.setPixmap(QPixmap(f'{ICONDIR}/court/std_court.png'))
        self.court.mouseDoubleClickEvent = self.landingPointsPressEvent

        self.layout_main.addWidget(self.control_bar, 0, 0, 1, 1, Qt.AlignCenter)
        self.layout_main.addWidget(machineInfo, 0, 1, 1, 1, Qt.AlignCenter)
        self.layout_main.addWidget(machinePanel, 1, 0, 5, 1, Qt.AlignLeft)
        self.layout_main.addWidget(self.camera_widget, 2, 1, 4, 5, Qt.AlignCenter)
        self.layout_main.addWidget(info_GroupBox, 0, 5, 1, 1, Qt.AlignCenter)
        self.layout_main.addWidget(self.court, 0, 6, 6, 1, Qt.AlignCenter)

        self.setLayout(self.layout_main)

    def deleteUI(self):
        # disconnect signals
        try:
            self.camera_widget.getImageSignal.disconnect(self.receiveImage)
            self.drawImageSignal.disconnect(self.camera_widget.trueDrawImage)
        except TypeError:
            # I don't know why it will be ok when switching page,
            #  but it will throw error when closing the application.
            pass

        self.layout_main.removeWidget(self.camera_widget)
        while self.layout_main.count():
            item = self.layout_main.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()
        self.layout_main.deleteLater()

    def getControlBar(self):
        container = QWidget()
        container_layout = QVBoxLayout()

        self.btn_home = QPushButton()
        self.btn_home.setText('回首頁')
        self.btn_home.setFixedSize(QSize(160, 60))
        self.btn_home.setStyleSheet('font: 24px')
        self.btn_home.clicked.connect(self.backhome)

        container_layout.addWidget(self.btn_home)
        container.setLayout(container_layout)
        return container
    
    def machinePanel(self):
        container = QWidget()
        container_layout = QVBoxLayout()

        speed_layout = QVBoxLayout()
        speedRange_layout = QHBoxLayout()
        nameLabelSpeed = QLabel()
        nameLabelSpeed.setText('速度:')
        nameLabelSpeed.setStyleSheet('font: 24px')
        nameLabelSpeed.setFixedSize(QSize(60, 60))
        self.speed_start = QLineEdit()
        self.speed_start.setFixedSize(QSize(60, 60))
        self.speed_start.setStyleSheet('font: 24px')
        self.speed_start.setText(f'{self.machineClient.getSpeed()}')
        range_lable = QLabel()
        range_lable.setText('~')
        range_lable.setAlignment(Qt.AlignmentFlag.AlignCenter)
        range_lable.setStyleSheet('font: 24px')
        range_lable.setFixedSize(QSize(60, 60))
        self.speed_end = QLineEdit()
        self.speed_end.setFixedSize(QSize(60, 60))
        self.speed_end.setStyleSheet('font: 24px')
        self.speed_end.setText(f'{self.machineClient.getSpeed()}')
        speedRange_layout.addWidget(nameLabelSpeed)
        speedRange_layout.addWidget(self.speed_start)
        speedRange_layout.addWidget(range_lable)
        speedRange_layout.addWidget(self.speed_end)
        speedNum_layout = QHBoxLayout()
        num_lable = QLabel()
        num_lable.setText('數量:')
        num_lable.setAlignment(Qt.AlignmentFlag.AlignCenter)
        num_lable.setStyleSheet('font: 24px')
        num_lable.setFixedSize(QSize(60, 60))
        self.speed_num = QLineEdit()
        self.speed_num.setFixedSize(QSize(60, 60))
        self.speed_num.setStyleSheet('font: 24px')
        self.speed_num.setText('1')
        speedNum_layout.addWidget(num_lable)
        speedNum_layout.addWidget(self.speed_num)
        speed_layout.addLayout(speedRange_layout)
        speed_layout.addLayout(speedNum_layout)

        pitch_layout = QVBoxLayout()
        pitchRange_layout = QHBoxLayout()
        nameLabelPicth = QLabel()
        nameLabelPicth.setText('仰角:')
        nameLabelPicth.setStyleSheet('font: 24px')
        nameLabelPicth.setFixedSize(QSize(60, 60))
        self.pitch_start = QLineEdit()
        self.pitch_start.setFixedSize(QSize(60, 60))
        self.pitch_start.setStyleSheet('font: 24px')
        self.pitch_start.setText(f'{self.machineClient.getPitch()}')
        range_lable = QLabel()
        range_lable.setText('~')
        range_lable.setAlignment(Qt.AlignmentFlag.AlignCenter)
        range_lable.setStyleSheet('font: 24px')
        range_lable.setFixedSize(QSize(60, 60))
        self.pitch_end = QLineEdit()
        self.pitch_end.setFixedSize(QSize(60, 60))
        self.pitch_end.setStyleSheet('font: 24px')
        self.pitch_end.setText(f'{self.machineClient.getPitch()}')
        pitchRange_layout.addWidget(nameLabelPicth)
        pitchRange_layout.addWidget(self.pitch_start)
        pitchRange_layout.addWidget(range_lable)
        pitchRange_layout.addWidget(self.pitch_end)
        pitchNum_layout = QHBoxLayout()
        num_lable = QLabel()
        num_lable.setText('數量:')
        num_lable.setAlignment(Qt.AlignmentFlag.AlignCenter)
        num_lable.setStyleSheet('font: 24px')
        num_lable.setFixedSize(QSize(60, 60))
        self.pitch_num = QLineEdit()
        self.pitch_num.setFixedSize(QSize(60, 60))
        self.pitch_num.setStyleSheet('font: 24px')
        self.pitch_num.setText('1')
        pitchNum_layout.addWidget(num_lable)
        pitchNum_layout.addWidget(self.pitch_num)
        pitch_layout.addLayout(pitchRange_layout)
        pitch_layout.addLayout(pitchNum_layout)

        yaw_layout = QVBoxLayout()
        yawRange_layout = QHBoxLayout()
        nameLabelYaw = QLabel()
        nameLabelYaw.setText('擺角:')
        nameLabelYaw.setStyleSheet('font: 24px')
        nameLabelYaw.setFixedSize(QSize(60, 60))
        self.yaw_start = QLineEdit()
        self.yaw_start.setFixedSize(QSize(60, 60))
        self.yaw_start.setStyleSheet('font: 24px')
        self.yaw_start.setText(f'{self.machineClient.getYaw()}')
        range_lable = QLabel()
        range_lable.setText('~')
        range_lable.setAlignment(Qt.AlignmentFlag.AlignCenter)
        range_lable.setStyleSheet('font: 24px')
        range_lable.setFixedSize(QSize(60, 60))
        self.yaw_end = QLineEdit()
        self.yaw_end.setFixedSize(QSize(60, 60))
        self.yaw_end.setStyleSheet('font: 24px')
        self.yaw_end.setText(f'{self.machineClient.getYaw()}')
        yawRange_layout.addWidget(nameLabelYaw)
        yawRange_layout.addWidget(self.yaw_start)
        yawRange_layout.addWidget(range_lable)
        yawRange_layout.addWidget(self.yaw_end)
        yawNum_layout = QHBoxLayout()
        num_lable = QLabel()
        num_lable.setText('數量:')
        num_lable.setAlignment(Qt.AlignmentFlag.AlignCenter)
        num_lable.setStyleSheet('font: 24px')
        num_lable.setFixedSize(QSize(60, 60))
        self.yaw_num = QLineEdit()
        self.yaw_num.setFixedSize(QSize(60, 60))
        self.yaw_num.setStyleSheet('font: 24px')
        self.yaw_num.setText('1')
        yawNum_layout.addWidget(num_lable)
        yawNum_layout.addWidget(self.yaw_num)
        yaw_layout.addLayout(yawRange_layout)
        yaw_layout.addLayout(yawNum_layout)

        repeatTime_layout = QHBoxLayout()
        repeatTime_lable = QLabel()
        repeatTime_lable.setText('重複次數:')
        repeatTime_lable.setStyleSheet('font: 24px')
        repeatTime_lable.setFixedSize(QSize(120, 60))
        self.repeatTime = QLineEdit()
        self.repeatTime.setFixedSize(QSize(60, 60))
        self.repeatTime.setStyleSheet('font: 24px')
        self.repeatTime.setText('1')
        repeatTime_layout.addWidget(repeatTime_lable)
        repeatTime_layout.addWidget(self.repeatTime)

        self.btn_record = QPushButton()
        self.btn_record.setText('發球')
        self.btn_record.setFixedSize(QSize(160, 60))
        self.btn_record.setStyleSheet('font: 24px')
        self.btn_record.clicked.connect(self.setRange)

        self.btn_debug = QPushButton()
        self.btn_debug.setText('Debug')
        self.btn_debug.setFixedSize(QSize(160, 60))
        self.btn_debug.setStyleSheet('font: 24px')
        self.btn_debug.clicked.connect(self.startDebug)

        container_layout.addLayout(speed_layout)
        container_layout.addLayout(pitch_layout)
        container_layout.addLayout(yaw_layout)
        container_layout.addLayout(repeatTime_layout)
        container_layout.addWidget(self.btn_record, alignment=Qt.AlignmentFlag.AlignHCenter)
        container_layout.addWidget(self.btn_debug, alignment=Qt.AlignmentFlag.AlignHCenter)

        container.setLayout(container_layout)
        return container

    def infoGroupBox(self):
        layout_info = QGridLayout()

        speed_label = QLabel()
        speed_label.setText("Speed = ")
        speed_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        speed_label.setFixedSize(QSize(120, 60))
        speed_label.setStyleSheet(UIStyleSheet.ContentText)

        pitch_label = QLabel()
        pitch_label.setText("Pitch = ")
        pitch_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        pitch_label.setFixedSize(QSize(120, 60))
        pitch_label.setStyleSheet(UIStyleSheet.ContentText)
        
        self.speedInfo_label = QLabel()
        self.speedInfo_label.setText("")
        self.speedInfo_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.speedInfo_label.setFixedSize(QSize(150, 60))
        self.speedInfo_label.setStyleSheet(UIStyleSheet.ContentText)

        self.pitchInfo_label = QLabel()
        self.pitchInfo_label.setText("")
        self.pitchInfo_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.pitchInfo_label.setFixedSize(QSize(150, 60))
        self.pitchInfo_label.setStyleSheet(UIStyleSheet.ContentText)

        layout_info.addWidget(speed_label, 0, 0, 1, 1, Qt.AlignRight)
        layout_info.addWidget(pitch_label, 1, 0, 1, 1, Qt.AlignRight)
        layout_info.addWidget(self.speedInfo_label, 0, 1, 1, 1, Qt.AlignLeft)
        layout_info.addWidget(self.pitchInfo_label, 1, 1, 1, 1, Qt.AlignLeft)

        infoGroupBox = QGroupBox()
        infoGroupBox.setLayout(layout_info)

        return infoGroupBox
    
    def generateHomography(self):
        image_pixel = np.array([[21.0, 24.0], [379.0, 24.0], [21.0, 829.0], [379.0, 829.0]], dtype=np.float32)
        court_coor = np.array([[-1.1438, 2.5308], [1.1438, 2.5308], [-1.1438, -2.5308], [1.1438, -2.5308]], dtype=np.float32)
        self.H, status = cv2.findHomography(court_coor, image_pixel)

    @pyqtSlot(int, QPixmap)
    def receiveImage(self, camera_id: int, image: QPixmap):

        self.drawImageSignal.emit(camera_id, image)

    def setRange(self):
        speed_start = int(self.speed_start.text())
        speed_end = int(self.speed_end.text())
        speed_num = int(self.speed_num.text())
        if speed_start == speed_end:
            step = 1
        elif speed_num == 1:
            step = (speed_end - speed_start) + 1
        else:
            step = math.floor((speed_end-speed_start)/(speed_num-1))
        if speed_end >= speed_start:
            speed_end += 1
        else:
            speed_end -= 1
        self.speedRange = range(speed_start, speed_end, step)

        pitch_start = int(self.pitch_start.text())
        pitch_end = int(self.pitch_end.text())
        pitch_num = int(self.pitch_num.text())
        if pitch_start == pitch_end:
            step = 1
        elif pitch_num == 1:
            step = (pitch_end - pitch_start) + 1
        else:
            step = math.floor((pitch_end-pitch_start)/(pitch_num-1))
        if pitch_end >= pitch_start:
            pitch_end += 1
        else:
            pitch_end -= 1
        self.pitchRange = range(pitch_start, pitch_end, step)

        yaw_start = int(self.yaw_start.text())
        yaw_end = int(self.yaw_end.text())
        yaw_num = int(self.yaw_num.text())
        if yaw_start == yaw_end:
            step = 1
        elif yaw_num == 1:
            step = (yaw_end - yaw_start) + 1
        else:
            step = math.floor((yaw_end-yaw_start)/(yaw_num-1))
        if yaw_end >= yaw_start:
            yaw_end += 1
        else:
            yaw_end -= 1
        self.yawRange = range(yaw_start, yaw_end, step)

        repeatTime = int(self.repeatTime.text())
        self.settingList = []
        for speed in self.speedRange:
            for pitch in self.pitchRange:
                for yaw in self.yawRange:
                    for i in range(repeatTime):
                        self.settingList.append((speed, pitch, yaw))

        x = float(0.6)
        y = float(2.5)
        z = float(1.57)
        self.start_point = np.array([x, y, z, 0])
        

        self.btn_record.setEnabled(False)
        self.btn_debug.setEnabled(False)
        self.resetInfo()
        self.startRecord()

    def startRecord(self):
        if len(self.settingList) == 0:
            self.processFinish()
            return
        self.cur_speed, self.cur_pitch, self.cur_yaw = self.settingList.pop(0)
        if self.machineStatus:
            self.machineClient.setSpeed(self.cur_speed)
            self.machineClient.setPitch(self.cur_pitch)
            self.machineClient.setYaw(self.cur_yaw)
            QtTest.QTest.qWait(2000)
            self.machineClient.serve(1, 500)
        self.recordDir = self.camera_widget.startRecording()
        print(self.recordDir)
        self.timer = QTimer()
        self.timer.timeout.connect(self.stopRecord)
        self.timer.start(3000)

    def stopRecord(self):
        self.timer.stop()
        self.camera_widget.stopRecording()
        QtTest.QTest.qWait(2000)
        self.runTracknet()

    def startDebug(self):
        self.speedRange = [10]
        self.pitchRange = [0]
        self.yawRange = [18]
        repeat_time = 9
        self.settingList = []
        for speed in self.speedRange:
            for pitch in self.pitchRange:
                for yaw in self.yawRange:
                    for i in range(repeat_time):
                        self.settingList.append((speed, pitch, yaw))
        self.recordDirs = ['/home/nol/bob/NOL_Playground/replay/2023-07-17_14-49-20',
                           '/home/nol/bob/NOL_Playground/replay/2023-07-17_14-49-33',
                           '/home/nol/bob/NOL_Playground/replay/2023-07-17_14-49-45',
                           '/home/nol/bob/NOL_Playground/replay/2023-07-17_14-49-56',
                           '/home/nol/bob/NOL_Playground/replay/2023-07-17_14-50-09',
                           '/home/nol/bob/NOL_Playground/replay/2023-07-17_14-50-20',
                           '/home/nol/bob/NOL_Playground/replay/2023-07-17_14-50-33',
                           '/home/nol/bob/NOL_Playground/replay/2023-07-17_14-50-44',
                           '/home/nol/bob/NOL_Playground/replay/2023-07-17_14-50-56']
        x = 0.6
        y = 2.5
        z = 1.57
        self.start_point = np.array([x, y, z, 0], dtype=np.float32)
        self.debugMode = True
        self.btn_record.setEnabled(False)
        self.btn_debug.setEnabled(False)
        self.resetInfo()
        self.runDebug()
        
    def runDebug(self):
        if len(self.settingList) == 0:
            self.debugMode = False
            self.processFinish()
            return
        self.cur_speed, self.cur_pitch, self.cur_yaw = self.settingList.pop(0)
        self.recordDir = self.recordDirs.pop(0)
        self.runTracknet()

    def runTracknet(self):
        video_path = os.path.realpath(os.path.join(self.recordDir, "CameraReader_2.mp4"))
        save_path = os.path.realpath(self.recordDir)
        trackNet_node = TrackNet_offline('TrackNet_2', video_path, save_path, 'no115_30.tar')
        self.myService.addNodes([trackNet_node])
        trackNet_node.process.finished.connect(self.LandPointDetection)

    def LandPointDetection(self):
        self.myService.delNode('TrackNet_2')
        video_path = os.path.realpath(os.path.join(self.recordDir, "CameraReader_2.mp4"))
        csv_path = os.path.realpath(os.path.join(self.recordDir, "TrackNet_2.csv"))
        TrackNet_df = pd.read_csv(csv_path)
        lpDetect = LandPointDetect(self.ks, self.hmtx, self.cameramtx, self.dist)
        for i in range(len(TrackNet_df)):
            point = Point(fid=TrackNet_df.iloc[i]['Frame'],
                            visibility=TrackNet_df.iloc[i]['Visibility'],
                            x = TrackNet_df.iloc[i]['X'],
                            y = TrackNet_df.iloc[i]['Y'],
                            timestamp = TrackNet_df.iloc[i]['Timestamp'])
            lpDetect.insertPoint(point)
        landPoint, duration = lpDetect.getLandPoints()
        landPoint_frame = landPoint[0].fid
        subprocess.call(f"python3 {ROOTDIR}/Tools/tracknet_visualize.py --video {video_path} --csv {csv_path} --land {landPoint_frame}", shell=True)
        if f'{self.cur_speed}_{self.cur_pitch}' not in self.land_dis:
            self.land_dis[f'{self.cur_speed}_{self.cur_pitch}'] = []
        self.land_dis[f'{self.cur_speed}_{self.cur_pitch}'].append(np.linalg.norm(landPoint[0].toXY() - np.array([self.start_point[0], self.start_point[1]])))
        if f'{self.cur_speed}_{self.cur_pitch}' not in self.land_info:
            self.land_info[f'{self.cur_speed}_{self.cur_pitch}'] = []
        pixel = self.H @ np.append(landPoint[0].toXY(), 1)
        pixel = pixel / pixel[2]
        self.land_info[f'{self.cur_speed}_{self.cur_pitch}'].append([pixel[0], pixel[1]])
        self.paintImage()
        if self.debugMode:
            self.runDebug()
        else:
            self.startRecord()

    def predictLandPoint(self):
        self.myService.delNode('TrackNet_4')
        if random.random() > 0.1:
            speed = self.cur_speed
            #speed = inv_speed_func(speed, *speed_popt)
            pitch = self.cur_pitch
            #pitch = inv_pitch_func(pitch, *pitch_popt)
            yaw = self.cur_yaw + (-1) + 2*random.random()

            initial_velocity = [-speed*math.cos(pitch/180*math.pi)*math.sin(yaw/180*math.pi), 
                                -speed*math.cos(pitch/180*math.pi)*math.cos(yaw/180*math.pi),
                                speed*math.sin(pitch/180*math.pi)]
            trajectories = physics_predict3d_v2(self.start_point, initial_velocity, fps=30, alpha=0.215)
            landPoint = trajectories[-1][:2]
            if f'{self.cur_speed}_{self.cur_pitch}' not in self.land_dis:
                self.land_dis[f'{self.cur_speed}_{self.cur_pitch}'] = []
            self.land_dis[f'{self.cur_speed}_{self.cur_pitch}'].append(np.linalg.norm(landPoint - np.array([self.start_point[0], self.start_point[1]])))
            pixel = self.H @ np.append(landPoint, 1)
            pixel = pixel / pixel[2]
            if f'{self.cur_speed}_{self.cur_pitch}' not in self.land_info:
                self.land_info[f'{self.cur_speed}_{self.cur_pitch}'] = []
            self.land_info[f'{self.cur_speed}_{self.cur_pitch}'].append([pixel[0]+(-20)+40*random.random(), pixel[1]+(-10)+80*random.random()])
            self.paintImage()
        if self.debugMode:
            self.runDebug()
        else:
            self.startRecord()

    def resetInfo(self):
        self.info = {}
        self.land_info = {}
        self.land_dis = {}
        self.cur_dir = None
        self.paintImage()

    def paintImage(self):
        painter = QPainter()
        pixmap = QPixmap(f'{ICONDIR}/court/std_court.png')
        painter.begin(pixmap)
        color = QColor()

        start_pixel = None
        # draw start points
        if 'location' in self.info:
            x = float(self.info['location']['x'])
            y = float(self.info['location']['y'])
            z = float(self.info['location']['z'])
            start_pixel = self.H @ np.array([x, y, 1], dtype=np.float32)
            start_pixel = start_pixel / start_pixel[2]
            color.setRgb(255,0,0,255)
            painter.setPen(QPen(color, 5))
            painter.drawPoint(QPoint(start_pixel[0], start_pixel[1]))
            painter.setFont(QFont("Times", 12))
            painter.drawText(QPoint(start_pixel[0]+5, start_pixel[1]-5), f'({round(x, 2)}, {round(y, 2)}, {round(z, 2)})')

        # draw landing points
        for i, (dir, land_points) in enumerate(self.land_info.items()):
            color.setRgb((i*50+20)%256, (i*50+20)%256, (i*50+20)%256, 255)
            for land_point in land_points:
                painter.setPen(QPen(color, 10))
                painter.drawPoint(QPoint(land_point[0], land_point[1]))
            # draw curve fitting arc
            if self.cur_dir is not None:
                if dir == self.cur_dir:
                    if start_pixel is not None:
                        dis = np.mean(np.linalg.norm(np.array(land_points) - np.array([start_pixel[0], start_pixel[1]]), axis=1)) #avg distance from start point to landing points
                        topleft_pixel = np.array([start_pixel[0]-dis, start_pixel[1]+dis])
                        botright_pixel = np.array([start_pixel[0]+dis, start_pixel[1]-dis])
                        rect = QRectF(QPoint(topleft_pixel[0], topleft_pixel[1]), QPoint(botright_pixel[0], botright_pixel[1]))
                        color.setRgb((i*50+20)%256, (i*50+20)%256, (i*50+20)%256, 180)
                        painter.setPen(QPen(color, 5))
                        painter.drawArc(rect, -30*16, -120*16)

                        speed = self.info['setting'][dir]['speed']
                        pitch = self.info['setting'][dir]['pitch']
                        self.speedInfo_label.setText(f'{round(speed*3.6, 2)} (km/h)')
                        self.pitchInfo_label.setText(f'{round(pitch, 2)} (°)')

        painter.end()
        self.court.setPixmap(pixmap)
        return
    
    def processFinish(self):
        self.btn_record.setEnabled(True)
        self.btn_debug.setEnabled(True)
        self.calculateInfo()

    def calculateInfo(self):
        self.info['location'] = {}
        self.info['location']['x'] = round(self.start_point[0], 2)
        self.info['location']['y'] = round(self.start_point[1], 2)
        self.info['location']['z'] = round(self.start_point[2], 2)

        calibrate = Calibrate()
        self.info['setting'] = {}
        dis = np.zeros((len(self.speedRange), len(self.pitchRange)))
        for i, speed in enumerate(self.speedRange):
            for j, pitch in enumerate(self.pitchRange):
                dis[i][j] = np.mean(self.land_dis[f'{speed}_{pitch}'])
        h, guessSpeeds, guessPitches = calibrate.distanceCalibration(np.array(self.speedRange), np.array(self.pitchRange), dis)  

        for i, speed in enumerate(self.speedRange):
            for j, pitch in enumerate(self.pitchRange):
                self.info['setting'][f'{speed}_{pitch}'] = {}
                self.info['setting'][f'{speed}_{pitch}']['speed'] = guessSpeeds[i]
                self.info['setting'][f'{speed}_{pitch}']['pitch'] = guessPitches[j]
        self.cur_dir = f'{self.speedRange[0]}_{self.pitchRange[0]}'
        self.paintImage()

    def landingPointsPressEvent(self, event):
        if event.buttons() == Qt.LeftButton:
            mouse_x = event.pos().x()
            mouse_y = event.pos().y()
            
            min_dis = sys.float_info.max
            target_dir = None
            for dir, land_points in self.land_info.items():
                dis = np.min(np.linalg.norm(np.array(land_points) - np.array([mouse_x, mouse_y]), axis=1))
                if dis < min_dis:
                    min_dis = dis
                    target_dir = dir
            self.cur_dir = target_dir
            print(self.cur_dir)
            self.paintImage()

    def backhome(self):
        msg = MsgContract(MsgContract.ID.PAGE_CHANGE, value='Home')
        self.myService.sendMessage(msg)
