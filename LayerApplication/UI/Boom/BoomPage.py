import sys
import json
import os
import logging
import ast
import time
import queue
import shutil
import threading
from datetime import datetime
from functools import partial
from UISettings import *
import paho.mqtt.client as mqtt
from vidgear.gears import WriteGear
from turbojpeg import TurboJPEG
from Services import SystemService, MsgContract
from message import *
from common import insertById, loadConfig, saveConfig, setIntrinsicMtx
from nodes import CameraReader
from gi.repository import GLib, Gst, Tcam

from PyQt5.QtWidgets import QGroupBox, QVBoxLayout, QComboBox, QWidget, QGridLayout, QLabel, QLineEdit, QPushButton, QHBoxLayout
from PyQt5.QtCore import QSize, QThread, pyqtSignal, pyqtSlot, QTimer, QLineF
from PyQt5.QtGui import QPixmap, QImage, QPainter, QPen, QColor
from PyQt5.Qt import *

DIRNAME = os.path.dirname(os.path.abspath(__file__))
ROOTDIR = os.path.dirname(os.path.dirname(DIRNAME))
ICONDIR = f"{DIRNAME}/icon"

sys.path.append(f"{ROOTDIR}/EllipseDetect")
import Thres_function as ellipsedetect

class BoomPage(QGroupBox):
    drawImageSignal = pyqtSignal(int, QPixmap)
    detectSignal = pyqtSignal(list, tuple, int)

    def __init__(self, camera_widget, broker_ip, cfg, cameras:list):
        super().__init__()

        self.cameraNumber = 2
        self.camera_widget = camera_widget
        self.broker_ip = broker_ip
        self.cfg = cfg
        self.cameras = cameras

        self.image_size = QSize(800, 600)

        # for ellipse detection
        self.detectSignal.connect(self.updateEllipseCoords)
        self.ellipse_coords = [None, None]
        self.midpoint = [None, None]

        self.detector = []

    def updateEllipseCoords(self, ellipse_coords, midpoint, index):
        self.ellipse_coords[index] = ellipse_coords
        self.midpoint[index] = midpoint
        print(index, ellipse_coords, midpoint)

    def hideEvent(self, event):
        logging.debug(f"{self.__class__.__name__}: hided.")
        for detector in self.detector:
            detector.stop()
        self.ellipse_coords = [None, None]
        self.midpoint = [None, None]
        self.camera_widget.stopStreaming()
        self.deleteUI()

    def showEvent(self, event):
        logging.debug(f"{self.__class__.__name__}: shown.")
        for i in range(self.cameraNumber):
            detector = ellipsedetect.DetectThread(self.broker_ip, self.cameras[i], self.detectSignal, i)
            self.detector.append(detector)
        for detector in self.detector:
            detector.start()
        self.setupUI()
        self.camera_widget.startStreaming()

    def setBackgroundService(self, service:SystemService):
        self.myService = service

    def setupUI(self):
        # main layout
        self.layout_main = QGridLayout()

        self.coords = self.getCoordsUI()
        self.control_bar = self.getControlBar()
        self.pic_btn = self.getPicBtn()
        self.camera_widget.initWidget(2, self.image_size)

        # can turn off FPS display
        # self.camera_widget.toggleFPS()

        # connect signals
        self.camera_widget.getImageSignal.connect(self.receiveImage)
        self.drawImageSignal.connect(self.camera_widget.trueDrawImage)

        self.layout_main.addWidget(self.coords, 0, 1, Qt.AlignLeft)
        # self.layout_main.addWidget(self.nameLabel, 0, 1, Qt.AlignCenter)
        self.layout_main.addWidget(self.control_bar, 1, 0, Qt.AlignCenter)
        self.layout_main.addWidget(self.camera_widget, 1, 1, Qt.AlignCenter)
        
        self.layout_main.addWidget(self.pic_btn, 2, 1, Qt.AlignCenter)


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

        btn_home = QPushButton()
        btn_home.setText('回首頁')
        btn_home.setFixedSize(QSize(160, 60))
        btn_home.setStyleSheet('font: 24px')
        btn_home.clicked.connect(self.backhome)

        container_layout.addWidget(btn_home)
        container.setLayout(container_layout)
        return container
    
    def getCoordsUI(self):
        container = QWidget()
        container_layout = QHBoxLayout()

        nameLabelX = QLabel(self)
        nameLabelX.setText('X:')
        nameLabelX.setStyleSheet('font: 24px')
        lineX = QLineEdit(self)

        nameLabelY = QLabel(self)
        nameLabelY.setText('Y:')
        nameLabelY.setStyleSheet('font: 24px')
        lineY = QLineEdit(self)

        nameLabelZ = QLabel(self)
        nameLabelZ.setText('Z:')
        nameLabelZ.setStyleSheet('font: 24px')
        lineZ = QLineEdit(self)

        container_layout.addWidget(nameLabelX)
        container_layout.addWidget(lineX)
        container_layout.addWidget(nameLabelY)
        container_layout.addWidget(lineY)
        container_layout.addWidget(nameLabelZ)
        container_layout.addWidget(lineZ)
        
        container_layout.setSpacing(30)
        container_layout.addStretch()
        container.setLayout(container_layout)
        return container
    
    def getPicBtn(self):
        container = QWidget()
        container_layout = QVBoxLayout()

        btn_home = QPushButton()
        btn_home.setText('拍照')
        btn_home.setFixedSize(QSize(160, 60))
        btn_home.setStyleSheet('font: 24px')
        btn_home.clicked.connect(self.toggleSnapshot)
        # contrain

        container_layout.addWidget(btn_home)
        container.setLayout(container_layout)
        return container

    def toggleSnapshot(self):
        self.camera_widget.takeSnapshot()

    @pyqtSlot(int, QPixmap)
    def receiveImage(self, camera_id: int, image: QPixmap):
        ellipse_coords = self.ellipse_coords[camera_id]
        if ellipse_coords is not None:
            painter = QPainter()
            painter.begin(image)
            color = QColor()
            color.setRgb(0,255,0,50)
            painter.setPen(QPen(color, 20))
            for i in range(len(ellipse_coords)):
                painter.drawLine(QLineF(ellipse_coords[i][0], ellipse_coords[i][1], ellipse_coords[(i+1)%len(ellipse_coords)][0], ellipse_coords[(i+1)%len(ellipse_coords)][1]))
            painter.end()

        self.drawImageSignal.emit(camera_id, image)

    def backhome(self):
        msg = MsgContract(MsgContract.ID.PAGE_CHANGE, value='Home')
        self.myService.sendMessage(msg)