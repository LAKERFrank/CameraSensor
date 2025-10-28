import os
import logging
import subprocess
import cv2
import time
from datetime import datetime

from PyQt5.QtWidgets import QGroupBox, QGridLayout, QLabel, QWidget, QHBoxLayout, QVBoxLayout, QComboBox, QPushButton, QStyle
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QSize, QThread, Qt, pyqtSignal, pyqtSlot, QElapsedTimer, QTimer
from PyQt5.Qt import *

from ..Services import SystemService, MsgContract
from ..UISettings import UIStyleSheet
from lib.message import *

from lib.common import REPLAYDIR

class RecordPage(QGroupBox):
    def __init__(self, camera_widget):
        super().__init__()

        self.camera_widget = camera_widget

        self.display = True
        self.isRecording = False

    def hideEvent(self, event):
        self.deleteUI()
        logging.debug(f"{self.__class__.__name__}: hided.")

    def showEvent(self, event):
        self.setupUI()
        logging.debug(f"{self.__class__.__name__}: shown.")

    def setBackgroundService(self, service:SystemService):
        self.myService = service

    def deleteUI(self):
        self.layout_main.removeWidget(self.camera_widget)
        while self.layout_main.count():
            item = self.layout_main.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()
        self.layout_main.deleteLater()

    def setupUI(self):

        self.control_bar = self.getControlBar()

        container_bar = QWidget()
        layout_bar = QVBoxLayout()
        layout_bar.addWidget(self.control_bar)
        container_bar.setLayout(layout_bar)

        self.layout_main = QHBoxLayout()
        self.layout_main.addWidget(container_bar, 1)
        self.layout_main.addWidget(self.camera_widget, 5, alignment=Qt.AlignmentFlag.AlignVCenter)

        self.setLayout(self.layout_main)

    def reset(self):
        self.btnRecording.setText("Start Record")
        self.isRecording = False
        self.timer.stop()

    def onTimer(self):
        elapsed = self.elapsedTimer.elapsed()
        msec = elapsed % 1000
        sec = elapsed // 1000 % 60
        min = elapsed // 60000 % 60
        hour = elapsed // 3600000

        self.labelTime.setText(f"{hour:02d}:{min:02d}:{sec:02d}.{msec:03d}")

    def toggleRecord(self):
        if self.isRecording:
            self.camera_widget.stopRecording()
            self.btnRecording.setText("Start Record")
            self.isRecording = False
            self.timer.stop()
        else:
            mode = self.comboMode.currentText()

            self.camera_widget.startRecording(mode)

            self.btnRecording.setText("Stop Record")
            self.isRecording = True
            self.elapsedTimer.restart()
            self.timer.start()

    def toggleDisplay(self):
        if self.display:
            self.camera_widget.hideEvent(None)
            self.btnDisplay.setText("Show Display")
        else:
            self.camera_widget.showEvent(None)
            self.btnDisplay.setText("Hide Display")
        self.display = not self.display

    def getControlBar(self):
        container = QWidget()
        container.setFixedWidth(300)
        container_layout = QVBoxLayout()

        # Home
        self.btn_home = QPushButton()
        self.btn_home.setText('返回')#回首頁
        self.btn_home.setFixedSize(QSize(180, 50))
        self.btn_home.setStyleSheet(UIStyleSheet.ReturnButton)
        self.btn_home.clicked.connect(self.backhome)

        self.gap = QLabel()
        self.gap.setStyleSheet("margin-top: 200px;")

        self.labelTime = QLabel()
        self.labelTime.setStyleSheet(UIStyleSheet.Check3DLabel)
        self.labelTime.setFixedSize(QSize(180, 50))
        self.labelTime.setText("00:00:00.000")

        self.btnRecording = QPushButton()
        self.btnRecording.setText("Start Record")
        self.btnRecording.setFixedSize(QSize(180, 50))
        self.btnRecording.setStyleSheet(UIStyleSheet.SelectButton)
        self.btnRecording.clicked.connect(self.toggleRecord)

        self.btnDisplay = QPushButton()
        self.btnDisplay.setText("Hide Display")
        self.btnDisplay.setFixedSize(QSize(180, 50))
        self.btnDisplay.setStyleSheet(UIStyleSheet.SelectButton)
        self.btnDisplay.clicked.connect(self.toggleDisplay)

        self.btnResync = QPushButton()
        self.btnResync.setText("Resync")
        self.btnResync.setFixedSize(QSize(180, 50))
        self.btnResync.setStyleSheet(UIStyleSheet.SelectButton)
        self.btnResync.clicked.connect(self.resync)

        self.comboMode = QComboBox()
        self.comboMode.setStyleSheet(UIStyleSheet.CameraSettingCameraCombobox)
        self.comboMode.setFixedSize(QSize(180, 50))
        self.comboMode.addItem("none")
        self.comboMode.addItem("h264_low")
        self.comboMode.addItem("h264_high")
        self.comboMode.addItem("lossless")
        self.comboMode.setCurrentText("h264_low")

        self.elapsedTimer = QElapsedTimer()
        self.timer = QTimer()
        self.timer.setInterval(66)
        self.timer.timeout.connect(self.onTimer)

        container_layout.addWidget(self.btn_home, alignment=Qt.AlignCenter)
        container_layout.addWidget(self.gap, alignment=Qt.AlignCenter)
        container_layout.addWidget(self.labelTime, alignment=Qt.AlignCenter)
        container_layout.addWidget(self.comboMode, alignment=Qt.AlignCenter)
        container_layout.addWidget(self.btnRecording, alignment=Qt.AlignCenter)
        container_layout.addWidget(self.btnDisplay, alignment=Qt.AlignCenter)
        container_layout.addWidget(self.btnResync, alignment=Qt.AlignCenter)
        container_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        container.setLayout(container_layout)
        return container

    def backhome(self):
        msg = MsgContract(MsgContract.ID.PAGE_CHANGE, value='CameraSystem')
        self.myService.sendMessage(msg)

    def resync(self):
        self.camera_widget.resync()
