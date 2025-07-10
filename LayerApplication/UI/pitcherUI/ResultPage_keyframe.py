import logging
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from functools import partial
from UISettings import *
from Services import SystemService
from message import *
import os

class ResultPage_keyframe(QGroupBox):
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
    # def setupButton1(self):
    #     # logging.debug(icon)
    #     button = QPushButton()
    #     button.setMaximumWidth(200)
    #     button.setText("結果產生")
    #     button.setFixedSize(QSize(160, 60))
    #     button.setStyleSheet('font: 24px')
    #
    #     return button

    def setupUI(self):
        print("Setup resultPage UI ======================")
        # main layout

        self.layout_main = QGridLayout()

        # Page Title
        self.pagetitle = QLabel()
        self.pagetitle.setStyleSheet(UIStyleSheet.TitleText)
        line = QFrame()
        line.setFrameStyle(QFrame.HLine | QFrame.Sunken)

        #


        btn = self.setupButton()
        btn.clicked.connect(self.showBaseballSystem)
        # btn1 = self.setupButton1()
        # btn1.clicked.connect(self.updateData)

        keyframe = ["./Pitcher/resultData/peak_lag.jpg", "./Pitcher/resultData/foot_plant.jpg",
                    "./Pitcher/resultData/max_shoulder_external_rotation.jpg", "./Pitcher/resultData/Ball_release.jpg"]
        keyframe_name = ['1.peak_leg',"2.foot_plant",'3.max shoulder external','4.Ball release']
        que_row = [1,1,3,3]
        label_row = [0,0,2,2]
        que_col = [1,2,1,2]
        num = 0
        for img in keyframe:
            pixmap = QPixmap(img).scaled(600, 400)
            label = QLabel(self)
            label.setPixmap(pixmap)
            label1 = QLabel()
            label1.setText(keyframe_name[num])
            label1.setStyleSheet(UIStyleSheet.SubtitleText)
            frame = QFrame()
            frame.setFrameStyle(QFrame.Box)
            frame.setLineWidth(2)
            frameLayout = QGridLayout(frame)
            frameLayout.addWidget(label)
            self.layout_main.addWidget(label1, label_row[num], que_col[num], Qt.AlignBottom | Qt.AlignHCenter)
            self.layout_main.addWidget(frame, que_row[num], que_col[num], Qt.AlignBottom | Qt.AlignHCenter)
            num += 1
        current_dir = os.getcwd()
        print(current_dir)
        # pixmap = QPixmap("./resultData/line_Chart.jpg").scaled(1200,800)
        # label = QLabel(self)
        # label.setPixmap(pixmap)
        # self.layout_main.addWidget(label, 0, 1, 2, 3, Qt.AlignBottom | Qt.AlignHCenter)
        self.layout_main.addWidget(btn, 0, 0, Qt.AlignTop | Qt.AlignLeft)
        # self.layout_main.addWidget(btn1, 0, 1, Qt.AlignTop | Qt.AlignLeft)
        label2 = QLabel()
        label2.setText("Pitch Sequence Image Frames \n"
                       "1. Peak Leg Kick (PK)\n"
                       "2. Foot Plant (FP)\n"
                       "3. Maximum External Rotation (MER)\n"
                       "4. Ball Release (BR)\n")
        label2.setStyleSheet(UIStyleSheet.SubtitleText)
        # font = QFont()
        # font.setPointSize(16)
        # label2.setFont(font)
        self.layout_main.addWidget(label2, 1, 0, Qt.AlignCenter | Qt.AlignHCenter)
        label3 = QLabel()
        label3.setText("The pictures on this view are of the athlete throwing \nat 4 events throuhgout the Pitching sequence.")
        font = QFont()
        font.setPointSize(14)
        label3.setFont(font)
        self.layout_main.addWidget(label3, 2, 0, Qt.AlignCenter | Qt.AlignHCenter)

        self.setLayout(self.layout_main)

    def showBaseballSystem(self):
        msg = MsgContract(MsgContract.ID.PAGE_CHANGE, value='BaseballSettingPage')
        self.myService.sendMessage(msg)

    def deleteUI(self):
        while self.layout_main.count():
            item = self.layout_main.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()
        self.layout_main.deleteLater()

    def updateData(self):
        keyframe = ["./Pitcher/resultData/peak_lag.jpg", "./Pitcher/resultData/foot_plant.jpg",
                    "./Pitcher/resultData/max_shoulder_external_rotation.jpg", "./Pitcher/resultData/Ball_release.jpg"]
        num = 0
        que_row = [1, 1, 3, 3]
        label_row = [0, 0, 2, 2]
        que_col = [1, 2, 1, 2]
        for img in keyframe:
            pixmap = QPixmap(img).scaled(600, 400)
            label = self.layout_main.itemAtPosition(que_row[num], que_col[num]).widget()
            label.setPixmap(pixmap)
            frame = QFrame()
            frame.setFrameStyle(QFrame.Box)
            frame.setLineWidth(2)
            frameLayout = QGridLayout(frame)
            frameLayout.addWidget(label)
            self.layout_main.addWidget(frame, que_row[num], que_col[num], Qt.AlignBottom | Qt.AlignHCenter)
            num += 1
        # pixmap = QPixmap("./resultData/line_Chart.jpg").scaled(860, 500)
        # label = self.layout_main.itemAtPosition(1,0).widget()
        # label.setPixmap(pixmap)
        # self.layout_main.addWidget(label, 1, 0, 1, 2, Qt.AlignBottom | Qt.AlignHCenter)