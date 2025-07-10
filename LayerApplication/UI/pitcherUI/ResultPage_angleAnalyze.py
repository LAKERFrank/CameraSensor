import logging
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from functools import partial
from UISettings import *
from Services import SystemService
from message import *
import os

class ResultPage_angleAnalyze(QGroupBox):
    def __init__(self):
        super().__init__()

        # self.setupUI()

    def hideEvent(self, event):
        logging.debug(f"{self.__class__.__name__}: hided.")
        self.deleteUI()

    def showEvent(self, event):
        self.setupUI()
        # self.updateData()
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

    def detailButton(self):
        button = QPushButton()
        button.setMaximumWidth(200)
        button.setText("Detail")
        button.setFixedSize(QSize(160, 60))
        button.setStyleSheet('font: 24px')
        return button

    def setupUI(self):
        print("Setup resultPage UI ======================")
        # main layout

        self.layout_main = QGridLayout()
        self.layout_main.setColumnStretch(1,2)
        # Page Title
        self.pagetitle = QLabel()
        self.pagetitle.setStyleSheet(UIStyleSheet.TitleText)
        line = QFrame()
        line.setFrameStyle(QFrame.HLine | QFrame.Sunken)

        btn = self.setupButton()
        btn.clicked.connect(self.showBaseballSystem)

        # widget1 = QWidget()
        # widget1.setMaximumWidth(200)
        # widget1.setLayout(QVBoxLayout())
        # widget1.layout().addWidget(btn)


        pixmap = QPixmap("./Pitcher/resultData/line_Chart.jpg").scaled(1200,800)
        label = QLabel(self)
        label.setPixmap(pixmap)
        frame = QFrame()
        frame.setFrameStyle(QFrame.Box)
        frame.setLineWidth(2)
        frameLayout = QGridLayout(frame)
        frameLayout.addWidget(label)
        # self.control_panel = self.getControlPanel()
        # self.control_panel.setFixedHeight(400)
        label1 = QLabel()
        label1.setText("Arm Angle Kinematics Analysis (deg)")
        label1.setStyleSheet(UIStyleSheet.SubtitleText)
        self.layout_main.addWidget(label1, 0, 1, Qt.AlignCenter | Qt.AlignHCenter)
        # self.layout_main.addWidget(self.control_panel, 0, 0, 0, 1, alignment=Qt.AlignHCenter)
        self.layout_main.addWidget(frame, 1, 1, Qt.AlignCenter | Qt.AlignHCenter)
        self.layout_main.addWidget(btn, 0, 0, Qt.AlignTop | Qt.AlignHCenter)

        btn1 = self.detailButton()
        btn1.clicked.connect(self.showDetail)
        self.layout_main.addWidget(btn1, 1, 0, Qt.AlignTop | Qt.AlignCenter)


        # self.layout_main.setRowMinimumHeight()
        self.setLayout(self.layout_main)

    def showBaseballSystem(self):
        msg = MsgContract(MsgContract.ID.PAGE_CHANGE, value='BaseballSettingPage')
        self.myService.sendMessage(msg)

    def showDetail(self):
        msg = MsgContract(MsgContract.ID.PAGE_CHANGE, value='Result_angleDetail')
        self.myService.sendMessage(msg)

    def updateData(self):
        pixmap = QPixmap("./Pitcher/resultData/line_Chart.jpg").scaled(1200, 800)
        label = self.layout_main.itemAtPosition(1, 1).widget()
        label.setPixmap(pixmap)
        #===
        frame = QFrame()
        frame.setFrameStyle(QFrame.Box)
        frame.setLineWidth(2)
        frameLayout = QGridLayout(frame)
        frameLayout.addWidget(label)
        #===
        self.layout_main.addWidget(frame, 1, 1, Qt.AlignCenter | Qt.AlignHCenter)

    def deleteUI(self):
        while self.layout_main.count():
            item = self.layout_main.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()
        self.layout_main.deleteLater()

    def getControlPanel(self):
        container = QWidget()
        container_layout = QVBoxLayout()

        # btn = self.setupButton()
        self.button = QPushButton()
        self.button.setMaximumWidth(200)
        self.button.setText("回上一頁")
        self.button.setFixedSize(QSize(160, 60))
        self.button.setStyleSheet('font: 24px')
        self.button.clicked.connect(self.showBaseballSystem)
        # btn1 = self.detailButton()
        self.button1 = QPushButton()
        self.button1.setMaximumWidth(200)
        self.button1.setText("Detail")
        self.button1.setFixedSize(QSize(160, 60))
        self.button1.setStyleSheet('font: 24px')
        self.button1.clicked.connect(self.showDetail)

        container_layout.addWidget(self.button)
        container_layout.addWidget(self.button1)
        container.setLayout(container_layout)
        return container