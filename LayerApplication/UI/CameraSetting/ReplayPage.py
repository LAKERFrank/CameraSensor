import os
import logging
import subprocess
import cv2
import time

from PyQt5.QtWidgets import QGroupBox, QGridLayout, QLabel, QWidget, QHBoxLayout, QVBoxLayout, QComboBox, QPushButton, QStyle
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QSize, QThread, Qt, pyqtSignal, pyqtSlot
from PyQt5.Qt import *

from ..Services import SystemService, MsgContract
from ..UISettings import UIFont
from lib.message import *

from lib.common import REPLAYDIR

class ReplayPage(QGroupBox):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg

        # replay thread
        self.replay_thread = []

        # setup UI
        self.setupUI()

    def hideEvent(self, event):
        self.stopThread()
        logging.debug(f"{self.__class__.__name__}: hided.")

    def showEvent(self, event):
        # get all date in replay directory
        self.date_cb.clear()
        dates = [f for f in os.listdir(REPLAYDIR)]
        dates.sort()
        self.date_cb.addItem('')
        self.date_cb.addItems(dates)

        logging.debug(f"{self.__class__.__name__}: shown.")

    def setBackgroundService(self, service:SystemService):
        self.myService = service

    def setupUI(self):
        self.control_bar = self.getControlBar()
        self.video_block = self.videoReplay(number=4)

        container_bar = QWidget()
        layout_bar = QVBoxLayout()
        layout_bar.addWidget(self.control_bar)
        container_bar.setLayout(layout_bar)

        layout_main = QVBoxLayout()
        layout_main.addWidget(container_bar, stretch=1)
        layout_main.addWidget(self.video_block, stretch=9)

        self.setLayout(layout_main)

    def getControlBar(self):
        container = QWidget()
        container_layout = QHBoxLayout()

        self.btn_home = QPushButton()
        self.btn_home.setText('回首頁')
        self.btn_home.setFixedSize(QSize(120, 50))
        self.btn_home.setStyleSheet('font: 24px')
        self.btn_home.clicked.connect(self.backhome)

        self.date_cb = QComboBox()
        self.date_cb.currentTextChanged.connect(self.date_cb_changed)
        self.date_cb.setFixedSize(1400,50)

        self.btn_play = QPushButton(self)
        self.btn_play.setFixedSize(50,50)
        self.btn_play.setIcon(QWidget().style().standardIcon(QStyle.SP_MediaPlay))
        self.btn_play.clicked.connect(self.playVideo)

        self.btn_pause = QPushButton(self)
        self.btn_pause.setFixedSize(50,50)
        self.btn_pause.setIcon(QWidget().style().standardIcon(QStyle.SP_MediaPause))
        self.btn_pause.clicked.connect(self.pauseVideo)

        self.btn_next = QPushButton(self)
        self.btn_next.setFixedSize(80,50)
        self.btn_next.setText('Next')
        self.btn_next.setStyleSheet('font: 24px')
        self.btn_next.clicked.connect(self.nextFrame)

        container_layout.addWidget(self.btn_home)
        container_layout.addWidget(self.date_cb)
        container_layout.addWidget(self.btn_play)
        container_layout.addWidget(self.btn_pause)
        container_layout.addWidget(self.btn_next)
        container.setLayout(container_layout)
        return container

    def videoReplay(self, number=4):
        container = QWidget()
        container_layout = QGridLayout()

        self.video_label = []
        for i in range(number):
            tmp_video = QLabel()
            text = "Video_Replay_" + str(i)
            tmp_video.setText(text)
            self.video_label.append(tmp_video)
            y = int(i / 2)
            x = i % 2
            container_layout.addWidget(tmp_video, y, x, alignment=Qt.AlignCenter) # 直、橫

        container.setLayout(container_layout)
        return container

    def playVideo(self):
        for c in self.replay_thread:
            c.restart()

    def pauseVideo(self):
        for c in self.replay_thread:
            c.pause()

    def nextFrame(self):
        for c in self.replay_thread:
            c.next()

    def stopThread(self):
        for c in self.replay_thread:
            c.stop()
        for i in range(len(self.video_label)):
            text = "Video_Replay_" + str(i)
            self.video_label[i].setText(text)

    def date_cb_changed(self):
        self.stopThread()
        if self.date_cb.currentText() != '':
            date = self.date_cb.currentText()
            video_list = [f for f in os.listdir(os.path.join(REPLAYDIR, date)) if f.endswith('.mp4')]
            if len(video_list) > 2:
                w = 480
                h = 360
            else:
                w = 800
                h = 600

            self.replay_thread = []
            i = 0
            for video in video_list:
                v = PlayVideo(camera_id=i, video=os.path.join(REPLAYDIR, date, video), image_width=w, image_height=h)
                v.drawImage_signal.connect(self.updateCameraImage)
                v.clearImage_signal.connect(self.clearCameraImage)
                v.load_video(v.video)
                self.replay_thread.append(v)
                v.start()
                self.video_label[i].setHidden(False)
                i+=1

            while i < len(self.video_label):
                self.video_label[i].setHidden(True)
                i += 1

            self.replay_dir = os.path.join(REPLAYDIR, date)

    def weights_cb_changed(self):
        self.weights = self.weights_cb.currentText()

    def backhome(self):
        msg = MsgContract(MsgContract.ID.PAGE_CHANGE, value='CameraSystem')
        self.myService.sendMessage(msg)

    @pyqtSlot(int, QPixmap)
    def updateCameraImage(self, camera_id: int, image: QPixmap):
        self.video_label[camera_id].setPixmap(image)

    @pyqtSlot(int)
    def clearCameraImage(self, camera_id: int):
        self.video_label[camera_id].clear()

class PlayVideo(QThread):
    # to communicate with replay page
    drawImage_signal = pyqtSignal(int, QPixmap)
    clearImage_signal = pyqtSignal(int)

    def __init__(self, camera_id,image_width,image_height,video):
        super().__init__()
        self.camera_id = camera_id
        self.image_width = image_width
        self.image_height = image_height
        self.playing = False
        self.alive = False
        self.next_frame = False
        self.cap = None
        self.video = video
        self.fps = None

    def stop(self):
        self.alive = False

    def pause(self):
        self.playing = False

    def restart(self):
        self.playing = True

    def next(self):
        self.playing = False
        self.next_frame = True

    def load_video(self,video):
        self.playing = False
        self.cap = cv2.VideoCapture(video)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) * 1.2
        # Load Video and show the first frame
        ret, frame = self.cap.read()
        if ret:
            height, width, channel = frame.shape
            bytesPerLine = 3 * width
            Q_image = QImage(frame.data, width, height, bytesPerLine, QImage.Format_RGB888).rgbSwapped().scaled(QSize(self.image_width,self.image_height),Qt.KeepAspectRatio)
            self.drawImage_signal.emit(self.camera_id, QPixmap(Q_image))
        else:
            self.clearImage_signal.emit(self.camera_id)
            logging.warning("[Replay] No Video Files: {}".format(video))

    def run(self):
        # Maybe better?
        # https://stackoverflow.com/questions/44404349/pyqt-showing-video-stream-from-opencv/44404713
        self.alive = True
        while self.alive:
            if self.playing:
                ret, frame = self.cap.read()
                if ret:
                    height, width, channel = frame.shape
                    bytesPerLine = 3 * width
                    Q_image = QImage(frame.data, width, height, bytesPerLine, QImage.Format_RGB888).rgbSwapped().scaled(QSize(self.image_width,self.image_height),Qt.KeepAspectRatio)
                    self.drawImage_signal.emit(self.camera_id, QPixmap(Q_image))
                else:
                    self.clearImage_signal.emit(self.camera_id)
                    self.load_video(self.video)
                time.sleep(1/self.fps) # Video speed
            elif self.next_frame:
                ret, frame = self.cap.read()
                if ret:
                    height, width, channel = frame.shape
                    bytesPerLine = 3 * width
                    Q_image = QImage(frame.data, width, height, bytesPerLine, QImage.Format_RGB888).rgbSwapped().scaled(QSize(self.image_width,self.image_height),Qt.KeepAspectRatio)
                    self.drawImage_signal.emit(self.camera_id, QPixmap(Q_image))
                else:
                    self.clearImage_signal.emit(self.camera_id)
                    self.load_video(self.video)
                self.next_frame = False
            else:
                time.sleep(0.05)
        self.cap.release()
