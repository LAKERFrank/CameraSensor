import sys
import os
import logging
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *

from functools import partial
from .UISettings import *
from .Services import SystemService
from lib.message import *

from lib.common import ICONDIR

class HomePage(QGroupBox):
    def __init__(self):
        super().__init__()

        self.setupUI()

    def hideEvent(self, event):
        logging.debug(f"{self.__class__.__name__}: hided.")

    def showEvent(self, event):
        logging.debug(f"{self.__class__.__name__}: shown.")

    def setBackgroundService(self, service:SystemService):
        self.myService = service

    def setupButton(self, text, icon=f"{ICONDIR}/defaultappicon.png"):
        # logging.debug(icon)
        button = QPushButton()
        button.setMaximumWidth(200)
        button.setIcon(QIcon(icon))
        button.setIconSize(QSize(190,190))
        label = QLabel()
        label.setText(f'{text}')
        label.setStyleSheet(UIStyleSheet.SubtitleText)
        label.setAlignment(Qt.AlignCenter)
        return button, label

    def setupUI(self):
        # main layout
        layout_main = QGridLayout()

        # Page Title
        self.pagetitle = QLabel()
        self.pagetitle.setStyleSheet(UIStyleSheet.TitleText)
        line = QFrame()
        line.setFrameStyle(QFrame.HLine | QFrame.Sunken)

        btn, label = self.setupButton(text="Camera", icon = f"{ICONDIR}/button.png")
        btn.setStyleSheet(UIStyleSheet.HomePageButton)
        btn.clicked.connect(self.showCameraSystem)
        layout_main.addWidget(btn, 6,0 ,Qt.AlignBottom|Qt.AlignHCenter)
        layout_main.addWidget(label, 7,0,Qt.AlignTop|Qt.AlignHCenter)

        #btn, label = self.setupButton(text="RallySeg", icon = f"{ICONDIR}/button.png") # TrajectoryAnalyzing page
        #btn.setStyleSheet(UIStyleSheet.HomePageButton)
        #btn.clicked.connect(self.showTrajectoryAnalyzing)
        #layout_main.addWidget(btn, 6, 1, Qt.AlignBottom|Qt.AlignHCenter)
        #layout_main.addWidget(label, 7, 1, Qt.AlignTop|Qt.AlignHCenter)

        #btn, label = self.setupButton(text="NiceShot", icon = f"{ICONDIR}/button.png")
        #btn.setStyleSheet(UIStyleSheet.HomePageButton)
        #btn.clicked.connect(self.showNiceShot)
        #layout_main.addWidget(btn, 6,2 ,Qt.AlignBottom|Qt.AlignHCenter)
        #layout_main.addWidget(label,7,2, Qt.AlignTop|Qt.AlignHCenter)

        #btn, label = self.setupButton(text="Boom", icon = f"{ICONDIR}/button.png")
        #btn.setStyleSheet(UIStyleSheet.HomePageButton)
        #btn.clicked.connect(self.showBoomPage)
        #layout_main.addWidget(btn, 6,3,Qt.AlignBottom|Qt.AlignHCenter)
        #layout_main.addWidget(label,7,3, Qt.AlignTop|Qt.AlignHCenter)

        #btn, label = self.setupButton(text="Machine Calibration", icon = f"{ICONDIR}/button.png")
        #btn.setStyleSheet(UIStyleSheet.HomePageButton)
        #btn.clicked.connect(self.showMachineCalibration)
        #layout_main.addWidget(btn, 6,4,Qt.AlignBottom|Qt.AlignHCenter)
        #layout_main.addWidget(label,7,4, Qt.AlignTop|Qt.AlignHCenter)

        btn, label = self.setupButton(text="test", icon = f"{ICONDIR}/button.png")
        btn.setStyleSheet(UIStyleSheet.HomePageButton)
        #btn.clicked.connect(partial(self.showPage, name))
        layout_main.addWidget(btn, 6,5,Qt.AlignBottom|Qt.AlignHCenter)
        layout_main.addWidget(label,7,5, Qt.AlignTop|Qt.AlignHCenter)
        btn.clicked.connect(self.showTestPage)

        #btn, label = self.setupButton(text="Pitcher", icon=f"{ICONDIR}/button.png")
        #btn.setStyleSheet(UIStyleSheet.HomePageButton)
        #btn.clicked.connect(partial(self.showBaseballSystem))
        #layout_main.addWidget(btn, 6, 6, Qt.AlignBottom | Qt.AlignHCenter)
        #layout_main.addWidget(label, 7, 6, Qt.AlignTop | Qt.AlignHCenter)

        btn, label = self.setupButton(text="MQTT demo", icon = f"{ICONDIR}/button.png") # TrajectoryAnalyzing page
        btn.setStyleSheet(UIStyleSheet.HomePageButton)
        btn.clicked.connect(self.showStreamingDemo)
        layout_main.addWidget(btn, 6, 7, Qt.AlignBottom|Qt.AlignHCenter)
        layout_main.addWidget(label, 7, 7, Qt.AlignTop|Qt.AlignHCenter)

        btn, label = self.setupButton(text="CES demo", icon = f"{ICONDIR}/button.png") # TrajectoryAnalyzing page
        btn.setStyleSheet(UIStyleSheet.HomePageButton)
        btn.clicked.connect(self.showCES)
        layout_main.addWidget(btn, 6, 8, Qt.AlignBottom|Qt.AlignHCenter)
        layout_main.addWidget(label, 7, 8, Qt.AlignTop|Qt.AlignHCenter)

        self.setLayout(layout_main)

    def showCameraSystem(self):
        msg = MsgContract(MsgContract.ID.PAGE_CHANGE, value='CameraSystem')
        self.myService.sendMessage(msg)

    def showTrajectoryAnalyzing(self):
        # msg = MsgContract(MsgContract.ID.PAGE_CHANGE, value='SettingPreviewPage')
        # msg.data = 'TrajectoryAnalyzingPage'
        msg = MsgContract(MsgContract.ID.PAGE_CHANGE, value='Model3dPage')
        # msg = MsgContract(MsgContract.ID.PAGE_CHANGE, value='WaitPage')
        self.myService.sendMessage(msg)

    def showTestPage(self):
        msg = MsgContract(MsgContract.ID.PAGE_CHANGE, value='SettingPreviewPage')
        msg.data = 'TestPage'
        self.myService.sendMessage(msg)

    def showMachineCalibration(self):
        msg = MsgContract(MsgContract.ID.PAGE_CHANGE, value='MachineCalibration')
        self.myService.sendMessage(msg)

    def showBoomPage(self):
        msg = MsgContract(MsgContract.ID.PAGE_CHANGE, value='BoomPage')
        self.myService.sendMessage(msg)

    def showNiceShot(self):
        msg = MsgContract(MsgContract.ID.PAGE_CHANGE, value='Processing') # for test
        self.myService.sendMessage(msg)

    def showBaseballSystem(self):
        msg = MsgContract(MsgContract.ID.PAGE_CHANGE, value='BaseballSettingPage')
        self.myService.sendMessage(msg)

    def showStreamingDemo(self):
        msg = MsgContract(MsgContract.ID.PAGE_CHANGE, value='MqttDemoPage')
        self.myService.sendMessage(msg)

    def showCES(self):
        msg = MsgContract(MsgContract.ID.PAGE_CHANGE, value='CESHomePage')
        self.myService.sendMessage(msg)
