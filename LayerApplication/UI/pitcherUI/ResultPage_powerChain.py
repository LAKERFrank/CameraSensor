import logging
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from functools import partial
from UISettings import *
from Services import SystemService
from message import *
import os

class ResultPage_powerChain(QGroupBox):
    def __init__(self):
        super().__init__()

        # self.setupUI()

    def hideEvent(self, event):
        logging.debug(f"{self.__class__.__name__}: hided.")
        self.deleteUI()
    def showEvent(self, event):
        # self.updateData()
        self.setupUI()
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

        # Page Title
        self.pagetitle = QLabel()
        self.pagetitle.setStyleSheet(UIStyleSheet.TitleText)
        line = QFrame()
        line.setFrameStyle(QFrame.HLine | QFrame.Sunken)

        btn = self.setupButton()
        btn.clicked.connect(self.showBaseballSystem)
        self.layout_main.addWidget(btn, 0, 0, Qt.AlignTop | Qt.AlignLeft)

        pixmap = QPixmap("./Pitcher/resultData/Kinematic_Sequence_Chart.jpg").scaled(1200, 800)
        label = QLabel(self)
        label.setPixmap(pixmap)
        frame = QFrame()
        frame.setFrameStyle(QFrame.Box)
        frame.setLineWidth(2)
        frameLayout = QGridLayout(frame)
        frameLayout.addWidget(label)

        label1 = QLabel()
        label1.setText("Kinematic Sequencing")
        label1.setStyleSheet(UIStyleSheet.SubtitleText)
        self.layout_main.addWidget(label1, 0, 1, Qt.AlignCenter | Qt.AlignHCenter)

        self.layout_main.addWidget(frame, 1, 1, Qt.AlignCenter | Qt.AlignHCenter)
        label2 = QLabel()
        label2.setText("● Joint Angular Velocity Chain\nShould follow the Kinematic Sequence\n\n"
                       "Pelvis -> Torso -> Elbow Ext -> Shoulder IR")
        # label2.setStyleSheet(UIStyleSheet.SubtitleText)
        font = QFont()
        font.setPointSize(18)
        label2.setFont(font)
        self.layout_main.addWidget(label2, 1, 0, Qt.AlignTop | Qt.AlignHCenter)
        self.setLayout(self.layout_main)

    def showBaseballSystem(self):
        msg = MsgContract(MsgContract.ID.PAGE_CHANGE, value='BaseballSettingPage')
        self.myService.sendMessage(msg)

    def updateData(self):
        pixmap = QPixmap("./Pitcher/resultData/Kinematic_Sequence_Chart.jpg").scaled(1200, 800)
        label = self.layout_main.itemAtPosition(1, 1).widget()
        label.setPixmap(pixmap)
        #===
        frame = QFrame()
        frame.setFrameStyle(QFrame.Box)
        frame.setLineWidth(2)
        frameLayout = QGridLayout(frame)

        frameLayout.addWidget(label)

        #===
        self.layout_main.addWidget(frame, 1, 1,Qt.AlignCenter | Qt.AlignHCenter)
    def deleteUI(self):
        while self.layout_main.count():
            item = self.layout_main.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()
        self.layout_main.deleteLater()