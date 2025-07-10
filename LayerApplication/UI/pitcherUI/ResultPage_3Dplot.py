import logging
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtMultimedia import *
from PyQt5.QtMultimediaWidgets import *
from PyQt5.QtCore import QUrl
from functools import partial
from UISettings import *
from Services import SystemService
from message import *

import os

class ResultPage_3Dplot(QGroupBox):
        def __init__(self):
            super().__init__()

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
        def playButton(self):
            # logging.debug(icon)
            button = QPushButton()
            button.setMaximumWidth(200)
            button.setText("Play")
            button.setFixedSize(QSize(160, 40))
            button.setStyleSheet('font: 20px')

            return button
        def setupUI(self):
            print("Setup resultPage UI ======================")
            # main layout

            self.layout_main = QGridLayout()
            self.layout_main.setColumnStretch(1, 2)
            # Page Title
            self.pagetitle = QLabel()
            self.pagetitle.setStyleSheet(UIStyleSheet.TitleText)

            btn = self.setupButton()
            btn.clicked.connect(self.showBaseballSystem)
            self.layout_main.addWidget(btn, 0, 0, Qt.AlignTop | Qt.AlignLeft)

            btn1 = self.playButton()
            btn1.clicked.connect(self.playfunction)
            self.layout_main.addWidget(btn1, 1, 0, Qt.AlignTop | Qt.AlignLeft)

            label1 = QLabel()
            label1.setText("3D骨架影片")
            label1.setStyleSheet(UIStyleSheet.SubtitleText)
            self.layout_main.addWidget(label1, 0, 1, Qt.AlignCenter | Qt.AlignHCenter)

            self.media_player = QMediaPlayer()
            media_content = QMediaContent(QUrl.fromLocalFile("/home/nol/demo_barry/Pitcher/resultData/output_3D.mp4"))
            self.media_player.setMedia(media_content)

            self.video_widget = QVideoWidget()
            # self.video_widget.setGeometry(0, 0, 800, 800)
            self.video_widget.setFixedSize(800,800)
            # self.video_widget.resize(1000,1000)
            self.media_player.setPlaybackRate(0.5)
            self.media_player.setVideoOutput(self.video_widget)
            self.layout_main.addWidget(self.video_widget, 1, 1, 2, 2, Qt.AlignCenter|Qt.AlignHCenter)
            # self.media_player.play()
            self.setLayout(self.layout_main)
        def playfunction(self):
            self.media_player.play()

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

