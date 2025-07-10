import logging
import time
import threading
import cv2
import os
import subprocess
import pandas as pd
import glob
import json
import numpy as np
import requests

from LayerApplication.Rpc.RpcStreamingBadminton import RpcStreamingBadminton
from LayerApplication.UI.TrajectoryAnalyzing.Trajectory import TrajectoryWidget
from lib.message import MsgContract

from ..Services import SystemService

from PyQt5.QtWidgets import QGroupBox, QVBoxLayout, QComboBox, QWidget, QGridLayout, QLabel, QPushButton, QHBoxLayout, QSpinBox, QScrollArea, QDialog, QDoubleSpinBox, QProgressDialog, QSizePolicy
from PyQt5.QtCore import QSize, QThread, pyqtSignal, pyqtSlot, QTimer, Qt
from PyQt5.QtGui import QPixmap, QImage, QGuiApplication, QPainter, QPen, QColor, QBrush
from PyQt5 import uic

from LayerContent.landingPointConsistency import calcLandingPoint
from LayerApplication.ServeMachine.PC.MachineClient import MachineClient

from LayerApplication.ServeMachine.PC.calcDis import physics_predict3d_v2

from .Trajectory3d import Trajectory3DVisualizeWidget

from lib.common import ROOTDIR
from lib.point import Point

class CESHomePage(QGroupBox):

    TRY_BALL=0
    RUN_BALL=1

    NET_HEIGHT = 1.524

    # ID
    if os.path.exists(f"{ROOTDIR}/LayerApplication/UI/CES_2024/ID.txt"):
        with open(f"{ROOTDIR}/LayerApplication/UI/CES_2024/ID.txt", "r") as file:
            ID = int(file.read().strip())
    else:
        ID = 1
    # 倒數秒數
    COUNTDOWN_SECONDS = 3
    # 錄影秒數
    RECORDING_SECONDS = 5
    # 發球機連線
    #machineClient = MachineClient('192.168.50.209')
    #machineClient.connect('MachineA')
    # landing point coordination
    X = []
    Y = []
    Height = []
    maxSpeed = 0

    # max speed
    maxSpeed = 0

    def __init__(self, rpcStreamingBadminton:RpcStreamingBadminton):
        super().__init__()
        self.rpcStreamingBadminton = rpcStreamingBadminton
        self.__setupUI()

        self.resultEvent = threading.Event()

        self.trajectory_list = []
        self.landing_list = []
        self.height_list = []

    def keyPressEvent(self, e):

        if e.key() == Qt.Key.Key_Escape:
            print("Esc")
            self.backhome()

        return super().keyPressEvent(e)

    def __setupUI(self):

        uic.loadUi(f'{ROOTDIR}/LayerApplication/UI/CES_2024/ces.ui', self)

        self.btn_debug:QPushButton
        self.btn_start:QPushButton
        self.btn_save:QPushButton
        self.label_uid:QLabel
        self.label_countdown:QLabel
        self.label_speed:QLabel
        self.label_height:QLabel
        self.label_landing_point:QLabel
        self.label_ball:QLabel
        self.logo_itri:QLabel
        self.logo_nycu:QLabel

        self.widget_speed:TrajectoryWidget
        self.widget_height:TrajectoryWidget
        self.widget_landing:TrajectoryWidget

        self.label_ball.setText("0/" + str(self.TRY_BALL + self.RUN_BALL))

        self.btn_debug.clicked.connect(self.onclick_debug)
        self.btn_start.clicked.connect(self.onclick_start)
        self.btn_save.clicked.connect(self.onclick_save)

        # Setup speed
        t = TrajectoryWidget()
        layout = self.widget_speed.parent().layout()  # Get the layout containing the placeholder
        layout.replaceWidget(self.widget_speed, t)
        self.widget_speed.deleteLater()
        self.widget_speed = t
        self.widget_speed.setCameraPosition([
            (10.99, 15.0, 9.7),
            (-0.58, 0.39, 1.04),
            (-0.17, -0.39, 0.90)
        ])

        # Setup Height
        t = TrajectoryWidget()
        layout = self.widget_height.parent().layout()  # Get the layout containing the placeholder
        layout.replaceWidget(self.widget_height, t)
        self.widget_height.deleteLater()
        self.widget_height = t
        self.widget_height.setCameraPosition([
            (0.37, 10.66, 1.36),
            (0.0, 0.0, 1.33),
            (0.0, 0.0, 1.0)
        ])

        # Setup Landing
        t = TrajectoryWidget()
        layout = self.widget_landing.parent().layout()  # Get the layout containing the placeholder
        layout.replaceWidget(self.widget_landing, t)
        self.widget_landing.deleteLater()
        self.widget_landing = t
        self.widget_landing.setCameraPosition([
            (0.08, -3.26, 13.58),
            (0.0, -3.66, 1.29),
            (0.0, -1, 0.0)
        ])
        self.widget_landing.setCameraZoom(0.8)
        self.widget_landing.set2DStyle()
        self.widget_landing.setBackgroundGreen()

        # Setup userID
        self.label_uid.setText(f"ID:    {self.ID:04d}")

        # Setup Net View Image
        #self.img_net = QPixmap(os.path.join(ROOTDIR, "LayerApplication/UI/CES_2024/net_view.png"))
        #self.label_netView.setPixmap(self.img_net.scaled(self.label_topView.size(), Qt.KeepAspectRatio))

        # Setup Top View Image
        #self.img_topView = QPixmap(os.path.join(ROOTDIR, "LayerApplication/UI/CES_2024/top_view.png"))
        #self.label_topView.setPixmap(self.img_topView.scaled(self.label_topView.size(), Qt.KeepAspectRatio))

        # Setup LOGO
        self.img_ITRI = QPixmap(os.path.join(ROOTDIR, "LayerApplication/UI/CES_2024/ITRI.png"))
        self.logo_itri.setPixmap(self.img_ITRI.scaled(self.logo_itri.size(), Qt.KeepAspectRatio))

        self.img_nycu = QPixmap(os.path.join(ROOTDIR, "LayerApplication/UI/CES_2024/nycu.png"))
        self.logo_nycu.setPixmap(self.img_nycu.scaled(self.logo_nycu.size(), Qt.KeepAspectRatio))

    def clear_result(self):
        #self.label_netView.setPixmap(self.img_net.scaled(self.label_topView.size(), Qt.KeepAspectRatio))
        #self.label_topView.setPixmap(self.img_topView.scaled(self.label_topView.size(), Qt.KeepAspectRatio))
        self.X.clear()
        self.Y.clear()
        self.Height.clear()
        self.maxSpeed = 0
        self.trajectory_list.clear()
        self.height_list.clear()
        self.landing_list.clear()
        QTimer.singleShot(0, self.update_trajectory_layout)

    def onclick_debug(self):
        self.clear_result()

        self.t = threading.Thread(target=self.__smashOne, args=(True,))
        self.t.start()

    def onclick_start(self):
        self.clear_result()

        self.t = threading.Thread(target=self.__smashOne)
        self.t.start()

    def onclick_save(self):
        print("speed:", self.widget_speed.plotter.camera_position)
        print("height:", self.widget_height.plotter.camera_position)
        print("landing:", self.widget_landing.plotter.camera_position)

        # For meichu ranking page
        # POST info to server
        url = "http://140.113.208.99:5003/api/machine_shot"

        if self.getSmashInfo():
            payload = json.dumps({
                "distance_cm": round(self.radius, 1),
                "in_opponent_court": 1 if self.in_opponent_court else 0,
                "net_height_cm": round(self.height_above_net, 1),
                "net_miss": 1 if not self.in_opponent_court or self.height_above_net < 0 else 0,
                "speed_kmh": round(self.speed, 1),
            })
        else:
            payload = json.dumps({
                "distance_cm": 999,
                "in_opponent_court": 0,
                "net_height_cm": 999,
                "net_miss": 1,
                "speed_kmh": 0,
            })
        
        headers = {
            'Content-Type': 'application/json'
        }

        try:
            response = requests.request("POST", url, headers=headers, data=payload, timeout=3)
            print(response.text)
        except Exception as e:
            print(e)

        #data = {
        #    'ID': [self.ID],
        #    'MaxSpeed': [round(self.maxSpeed, 2)],
        #    'Radius': [round(self.radius, 2)]
        #}

        #df = pd.DataFrame(data)
        #file_path = f'{ROOTDIR}/LayerApplication/UI/CES_2024/score.csv'

        #df.to_csv(file_path, mode='a', header=False, index=False)

    def __smashOne(self, debug=False):
        self.btn_start.setEnabled(False)
        
        self.ID += 1
        self.label_uid.setText(f"ID:    {self.ID:04d}")
        with open(f"{ROOTDIR}/LayerApplication/UI/CES_2024/ID.txt", "w") as file:
            file.write(str(self.ID))
        
        # 目前發球數
        ball = 0

        while ball < (self.TRY_BALL + self.RUN_BALL):
            self.label_ball.setText(str(ball+1) + "/" + str(self.TRY_BALL + self.RUN_BALL))

            for i in range(self.COUNTDOWN_SECONDS, -1, -1):
                time.sleep(1)
                self.label_countdown.setText(str(i))

            self.resultEvent.clear()
            self.content_payload = None

            if debug:
                self.rpcStreamingBadminton.testStart(read_video=False)
            else:
                self.rpcStreamingBadminton.start(content_mode="CES")

            time.sleep(0.1)
            # TODO: serve
            # self.machineClient.serve(1, 0)

            time.sleep(self.RECORDING_SECONDS)

            if debug:
                self.rpcStreamingBadminton.testStop(read_video=False)
            else:
                self.rpcStreamingBadminton.stop()

            self.label_countdown.setText(str(self.COUNTDOWN_SECONDS))

            self.resultEvent.wait(10)

            # check if TrackNet recognize the ball
            if self.getSmashInfo():
            # if so ball plus one
                # update numeric result
                self.label_speed.setText(str(f"{self.speed:.1f}"))
                self.label_speed_text.setText("km/h")
                self.label_height.setText(str(f"{self.height_above_net:.1f}"))
                self.label_height_text.setText("cm")
                self.label_landing_point.setText(str(f"{self.radius:.1f}"))
                self.label_landing_point_text.setText("cm")

            # if not ball remain the same
            else:
                self.label_speed.setText("MISS")
                self.label_speed_text.setText("")
                self.label_height.setText("MISS")
                self.label_height_text.setText("")
                self.label_landing_point.setText("MISS")
                self.label_landing_point_text.setText("")

            ball += 1

            if ball == self.TRY_BALL-1:
                #self.label_netView.setPixmap(self.img_net.scaled(self.label_topView.size(), Qt.KeepAspectRatio))
                #self.label_topView.setPixmap(self.img_topView.scaled(self.label_topView.size(), Qt.KeepAspectRatio))
                self.X.clear()
                self.Y.clear()
                self.Height.clear()
                self.maxSpeed = 0
                self.trajectory_list.clear()
                QTimer.singleShot(0, self.update_trajectory_layout)

        # TODO: visualize
        self.btn_start.setEnabled(True)

    def setBackgroundService(self, service:SystemService):
        self.myService = service

    def hideEvent(self, event):
        self.rpcStreamingBadminton.content_callback = None
        logging.debug(f"{self.__class__.__name__}: hided.")

    def showEvent(self, event):
        self.rpcStreamingBadminton.content_callback = self.content_callback
        logging.debug(f"{self.__class__.__name__}: shown.")

    def update_trajectory_layout(self):
        self.widget_speed.reset()
        for t in self.trajectory_list:
            track = []

            first = t[0]

            track.append(Point(0, first[3], 1, first[0], first[1], first[2], 1, 0, "blue"))

            for p in t[1:-1]:
                track.append(Point(0, p[3], 1, p[0], p[1], p[2]))

            final = t[-1]
            track.append(Point(0, final[3], 1, final[0], final[1], final[2], 2, 0, "red"))

            for point in track:
                self.widget_speed.addPointByTrackID(point, 0)
            self.widget_speed.render()

        self.widget_height.reset()
        for p in self.height_list:
            point = Point(x=p[0], y=0, z=p[2], color="red")
            self.widget_height.addPointByTrackID(point)
        self.widget_height.render()

        self.widget_landing.reset()
        for p in self.landing_list:
            point = Point(x=p[0], y=p[1], z=0.01, color="red")
            self.widget_landing.addPointByTrackID(point)
        self.widget_landing.render()

    def content_callback(self, data):
        payload = json.loads(data)
        self.content_payload = payload
        self.resultEvent.set()

    def getSmashInfo(self):
        print(self.content_payload)

        if self.content_payload is None or self.content_payload["shot_id"] == -1:
            return False
        else:
            pos = self.content_payload["position"]
            speed = self.content_payload["speed"]

            # Speed
            self.speed = np.linalg.norm(speed)*3.6

            trajectory = physics_predict3d_v2(pos+[0], speed, 120)

            # Net Height
            p = sorted(trajectory, key=lambda row: np.abs(row[1]))[:2]

            self.height_point = (p[0] + p[1]) / 2
            self.height_list.append(self.height_point)

            self.net_height = self.height_point[2]

            self.height_above_net = (self.net_height - 1.524) * 100
            # Fix weird height
            if -5<= self.height_above_net < 0:
                self.height_above_net = 1

            self.landing_list.append(trajectory[-1])

            self.X.append(trajectory[-1][0])
            self.Y.append(trajectory[-1][1])
            self.Height.append(self.net_height)
            self.maxSpeed = max(self.maxSpeed, self.speed)
            #self.radius = calcLandingPoint(self.X * 100, self.Y * 100)
            self.radius = np.linalg.norm(np.array([0, -3.96])-np.array([self.X[-1], self.Y[-1]]))*100

            self.in_opponent_court = 3.1 >= self.X[-1] >= -3.1 and -6.75 <= self.Y[-1] <= 0

            # # pixmap
            #self.pixmap_net = self.img_net
            #self.pixmap_top = self.img_topView

            # update trajectory
            self.trajectory_list.append(trajectory)
            QTimer.singleShot(0, self.update_trajectory_layout)

            # update netView result
            #for i in range(len(self.X)-1, max(-1, len(self.X)-6), -1):
            #    painter = QPainter()
            #    painter.begin(self.pixmap_net)
            #    color = QColor()
            #    color.setRgb(255,0,0,255)
            #    painter.setPen(QPen(color,2))
            #    painter.setBrush(QBrush(color, Qt.SolidPattern))
            #    x , self.height = self.paint_coor(self.X[i] ,0 ,self.Height[i] , view="net")
            #    r = 5 # Adjust this as needed
            #    painter.drawEllipse(x - r, self.height - r, 2 * r, 2 * r) # Draw a filled circle at (x, y) with the specified radius
            #    painter.end()

            #    # update topView result
            #    painter = QPainter()
            #    painter.begin(self.pixmap_top)
            #    color = QColor()
            #    color.setRgb(255,0,0,255)
            #    painter.setPen(QPen(color,2))
            #    painter.setBrush(QBrush(color, Qt.SolidPattern))
            #    x , y = self.paint_coor(self.X[i] , self.Y[i] ,0, view="top")
            #    r = 5 # Adjust this as needed
            #    painter.drawEllipse(x - r, y - r, 2 * r, 2 * r) # Draw a filled circle at (x, y) with the specified radius
            #    painter.end()

            #self.label_netView.setPixmap(self.pixmap_net)
            #self.label_topView.setPixmap(self.pixmap_top)
            return True

    def paint_coor(self , x , y , height, view):
        if view == "top":
            x = (3 - x) / 0.017045 + 100
            y = (y - (-6.7)) / 0.017179 + 20
            # print("pixel(x,y):",x,y)
            return x , y
        elif view == "net":
            x = (3 - x) / 0.01101 + 5
            height = 207 - (height - self.NET_HEIGHT) / 0.00598
            # print("pixel(x,height):",x,height)
            return x , height

    def backhome(self):
        msg = MsgContract(MsgContract.ID.PAGE_CHANGE, value='Home')
        self.myService.sendMessage(msg)
