import logging
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from functools import partial
from UISettings import *
from Services import SystemService
from message import *
import os

class Result_angleDetail(QGroupBox):
    def __init__(self):
        super().__init__()
        self.setupUI()

    def hideEvent(self, event):
        logging.debug(f"{self.__class__.__name__}: hided.")

    def showEvent(self, event):
        logging.debug(f"{self.__class__.__name__}: shown.")
    def setBackgroundService(self, service:SystemService):
        self.myService = service
    def setupButton(self):
        # logging.debug(icon)
        button = QPushButton()
        button.setMaximumWidth(200)
        button.setText("回上一頁")
        button.setFixedSize(QSize(160, 60))
        button.setStyleSheet('font: 24px')
        return button


    def setupUI(self):
        print("Setup resultPage UI ======================")
        # main layout

        self.layout_main = QGridLayout()
        self.layout_main.setColumnStretch(1, 2)
        self.layout_main.setColumnStretch(2, 2)
        # Page Title
        self.pagetitle = QLabel()
        self.pagetitle.setStyleSheet(UIStyleSheet.TitleText)
        line = QFrame()
        line.setFrameStyle(QFrame.HLine | QFrame.Sunken)

        btn = self.setupButton()
        btn.clicked.connect(self.backPage)
        self.layout_main.addWidget(btn, 0, 0, Qt.AlignTop | Qt.AlignCenter)

        label = QLabel()
        label.setText("● Shoulder Abduction:\nRaise your arms into a lateral raise.\n A T-pose is 90, arms at your side is 0.")
        label.setStyleSheet(UIStyleSheet.SubtitleText)
        self.layout_main.addWidget(label, 0, 1, Qt.AlignTop | Qt.AlignHCenter)
        pixmap = QPixmap("./Pitcher/icon/VA.PNG").scaled(400, 300)
        label = QLabel(self)
        label.setPixmap(pixmap)
        frame = QFrame()
        frame.setFrameStyle(QFrame.Box)
        frame.setLineWidth(2)
        frameLayout = QGridLayout(frame)
        frameLayout.addWidget(label)
        self.layout_main.addWidget(frame, 1, 1, Qt.AlignVCenter | Qt.AlignHCenter)

        label = QLabel()
        label.setText("● Shoulder Horizontal Abduction:\nA T-pose is 0, moving behind you is positive.")
        label.setStyleSheet(UIStyleSheet.SubtitleText)
        self.layout_main.addWidget(label, 0, 2, Qt.AlignTop | Qt.AlignHCenter)
        pixmap = QPixmap("./Pitcher/icon/HA.PNG").scaled(400, 300)
        label = QLabel(self)
        label.setPixmap(pixmap)
        frame = QFrame()
        frame.setFrameStyle(QFrame.Box)
        frame.setLineWidth(2)
        frameLayout = QGridLayout(frame)
        frameLayout.addWidget(label)
        self.layout_main.addWidget(frame, 1, 2, Qt.AlignVCenter | Qt.AlignHCenter)

        label = QLabel()
        label.setText("● Shoulder External Rotation:\nMaking a goal post with your arms is 90,\n rotating them forward moves towards 0.")
        label.setStyleSheet(UIStyleSheet.SubtitleText)
        self.layout_main.addWidget(label, 2, 1, Qt.AlignTop | Qt.AlignHCenter)
        pixmap = QPixmap("./Pitcher/icon/ER.PNG").scaled(400, 300)
        label = QLabel(self)
        label.setPixmap(pixmap)
        frame = QFrame()
        frame.setFrameStyle(QFrame.Box)
        frame.setLineWidth(2)
        frameLayout = QGridLayout(frame)
        frameLayout.addWidget(label)
        self.layout_main.addWidget(frame, 3, 1, Qt.AlignVCenter | Qt.AlignHCenter)

        label = QLabel()
        label.setText("● Elbow Flexion:\nAngle of elbow, fully extended is 0 deg.")
        label.setStyleSheet(UIStyleSheet.SubtitleText)
        self.layout_main.addWidget(label, 2, 2, Qt.AlignTop | Qt.AlignHCenter)
        pixmap = QPixmap("./Pitcher/icon/EF.PNG").scaled(400, 300)
        label = QLabel(self)
        label.setPixmap(pixmap)
        frame = QFrame()
        frame.setFrameStyle(QFrame.Box)
        frame.setLineWidth(2)
        frameLayout = QGridLayout(frame)
        frameLayout.addWidget(label)
        self.layout_main.addWidget(frame, 3, 2, Qt.AlignVCenter | Qt.AlignHCenter)

        self.setLayout(self.layout_main)

    def backPage(self):
        msg = MsgContract(MsgContract.ID.PAGE_CHANGE, value='ResultPage_angleAnalyze')
        self.myService.sendMessage(msg)

