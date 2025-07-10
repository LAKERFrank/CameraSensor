import sys
import os
import logging
import numpy as np
import pandas as pd
import cv2
import json
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtMultimedia import *
from PyQt5.QtMultimediaWidgets import *

DIRNAME = os.path.dirname(os.path.abspath(__file__))
UIDIR = os.path.dirname(DIRNAME)
ICONDIR = os.path.join(UIDIR, 'icon')

sys.path.append(f"{UIDIR}")
from UISettings import *
from Services import SystemService, MsgContract
sys.path.append(f"{UIDIR}/../lib")
from message import *

SCREEN_SIZE = QSize(800, 600)

class VisualizePage(QGroupBox):
    def __init__(self):
        super().__init__()

        self.generateHomography()
        self.land_info = {}

    def hideEvent(self, event):
        logging.debug(f"{self.__class__.__name__}: hided.")
        self.deleteUI()

    def showEvent(self, event):
        logging.debug(f"{self.__class__.__name__}: shown.")
        self.setupUI()
        self.refreshDir()

    def setBackgroundService(self, service:SystemService):
        self.myService = service

    def setupUI(self):
        # main layout
        self.layout_main = QGridLayout()

        self.control_bar = self.getControlBar()

        self.cbo_dir = QComboBox()
        self.cbo_dir.setFixedSize(300, 50)
        self.cbo_dir.setFont(UIFont.Combobox)
        self.cbo_dir.currentTextChanged.connect(self.refreshVideoDir)

        self.cbo_video = QComboBox()
        self.cbo_video.setFixedSize(300, 50)
        self.cbo_video.setFont(UIFont.Combobox)
        self.cbo_video.currentTextChanged.connect(self.refreshVideo)

        video_GroupBox = self.videoGroupBox()
        info_GroupBox = self.infoGroupBox()

        self.court = QLabel()
        self.court.setPixmap(QPixmap(f'{ICONDIR}/court/std_court.png'))
        self.court.mouseDoubleClickEvent = self.landingPointsPressEvent

        self.layout_main.addWidget(self.control_bar, 0, 0, 1, 1, Qt.AlignCenter)
        self.layout_main.addWidget(self.cbo_dir, 1, 1, 1, 1, Qt.AlignCenter)
        self.layout_main.addWidget(self.cbo_video, 1, 2, 1, 1, Qt.AlignCenter)
        self.layout_main.addWidget(video_GroupBox, 2, 0, 4, 4, Qt.AlignCenter)
        self.layout_main.addWidget(info_GroupBox, 5, 4, 1, 1, Qt.AlignCenter)
        self.layout_main.addWidget(self.court, 0, 5, 6, 1, Qt.AlignCenter)

        self.setLayout(self.layout_main)

    def getControlBar(self):
        container = QWidget()
        container_layout = QVBoxLayout()

        self.btn_home = QPushButton()
        self.btn_home.setText('回首頁')
        self.btn_home.setFixedSize(QSize(160, 60))
        self.btn_home.setStyleSheet('font: 24px')
        self.btn_home.clicked.connect(self.backhome)

        container_layout.addWidget(self.btn_home)
        container.setLayout(container_layout)
        return container

    def refreshDir(self):
        dirs = []
        dirs.append('')
        self.replay_dir = f'{UIDIR}/../replay'

        for dir in os.listdir(self.replay_dir):
            if os.path.isdir(os.path.join(self.replay_dir, dir)):
                dirs.append(dir)
        self.cbo_dir.clear()
        self.cbo_dir.addItems(dirs)

    def refreshVideoDir(self, dirName):
        if dirName == '':
            return
        with open(os.path.join(self.replay_dir, dirName, 'serveMachine.json')) as jsonfile:
            self.info = json.load(jsonfile)

        dirs = []
        self.land_info.clear()
        for dir in os.listdir(os.path.join(self.replay_dir, dirName)):
            if os.path.isdir(os.path.join(self.replay_dir, dirName, dir)):
                dirs.append(dir)
                land_df = pd.read_csv(os.path.join(self.replay_dir, dirName, dir, 'detect.csv'))
                land_points = np.array(land_df[['X','Y']])
                land_pixel = []
                for land_point in land_points:
                    pixel = self.H @ np.array([land_point[0], land_point[1], 1], dtype=np.float32)
                    pixel = pixel / pixel[2]
                    land_pixel.append([pixel[0], pixel[1]])
                self.land_info[dir] = np.array(land_pixel)
        self.cbo_video.clear()
        self.cbo_video.addItems(dirs)

    def refreshVideo(self, videoDir):
        if videoDir == '':
            return
        self.paintImage()
        videoName = os.path.join(self.replay_dir, self.cbo_dir.currentText(), videoDir, 'CameraReader_0.mp4')
        self.playlist.clear()
        self.playlist.addMedia(QMediaContent(QUrl.fromLocalFile(os.path.abspath(videoName))))
        self.videoplayer.play()

    def infoGroupBox(self):
        layout_info = QGridLayout()

        speed_label = QLabel()
        speed_label.setText("Speed = ")
        speed_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        speed_label.setFixedSize(QSize(120, 60))
        speed_label.setStyleSheet(UIStyleSheet.ContentText)

        pitch_label = QLabel()
        pitch_label.setText("Pitch = ")
        pitch_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        pitch_label.setFixedSize(QSize(120, 60))
        pitch_label.setStyleSheet(UIStyleSheet.ContentText)
        
        self.speedInfo_label = QLabel()
        self.speedInfo_label.setText("")
        self.speedInfo_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.speedInfo_label.setFixedSize(QSize(150, 60))
        self.speedInfo_label.setStyleSheet(UIStyleSheet.ContentText)

        self.pitchInfo_label = QLabel()
        self.pitchInfo_label.setText("")
        self.pitchInfo_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.pitchInfo_label.setFixedSize(QSize(150, 60))
        self.pitchInfo_label.setStyleSheet(UIStyleSheet.ContentText)

        layout_info.addWidget(speed_label, 0, 0, 1, 1, Qt.AlignRight)
        layout_info.addWidget(pitch_label, 1, 0, 1, 1, Qt.AlignRight)
        layout_info.addWidget(self.speedInfo_label, 0, 1, 1, 1, Qt.AlignLeft)
        layout_info.addWidget(self.pitchInfo_label, 1, 1, 1, 1, Qt.AlignLeft)

        infoGroupBox = QGroupBox()
        infoGroupBox.setLayout(layout_info)

        return infoGroupBox

    def videoGroupBox(self):
        layout_video = QGridLayout()

        videowidget = QVideoWidget()
        videowidget.setFixedSize(SCREEN_SIZE)
        self.videoplayer = QMediaPlayer(None, QMediaPlayer.VideoSurface)
        self.videoplayer.setVideoOutput(videowidget)
        self.videoplayer.durationChanged.connect(self.onDurationChanged)
        self.videoplayer.positionChanged.connect(self.onPositionChanged)
        self.playlist = QMediaPlaylist()
        self.playlist.setPlaybackMode(QMediaPlaylist.CurrentItemInLoop)
        self.videoplayer.setPlaylist(self.playlist)

        self.slider_video = QSlider(Qt.Horizontal)
        self.slider_video.sliderMoved.connect(self.dragSlider)
        self.slider_video.setSingleStep(1)
        self.slider_video.setMinimum(0)
        self.slider_video.setFixedWidth(800)

        self.play_btn = QPushButton()
        self.play_btn.setFixedSize(50,50)
        self.play_btn.setIcon(QIcon(QIcon(f"{ICONDIR}/play.png")))
        self.play_btn.clicked.connect(self.onBtnPlayClicked)

        self.playrate_cbbox = QComboBox()
        self.playrate_cbbox.addItems(['2x', '1.5x', '1x', '0.75x', '0.5x'])
        self.playrate_cbbox.setFont(QFont('Arial', 18))
        self.playrate_cbbox.setCurrentIndex(0)
        self.playrate_cbbox.setFixedWidth(120)
        self.playrate_cbbox.currentIndexChanged.connect(self.playRateChange)

        layout_video.addWidget(videowidget, 0, 0, 5, 10, Qt.AlignCenter)
        layout_video.addWidget(self.slider_video, 5, 0, 1, 10, Qt.AlignCenter)
        layout_video.addWidget(self.play_btn, 6, 4, 1, 1)
        layout_video.addWidget(self.playrate_cbbox, 6, 5, 1, 1)

        videoGroupBox = QGroupBox()
        videoGroupBox.setLayout(layout_video)
        videoGroupBox.setStyleSheet(UIStyleSheet.InfoFrame)

        return videoGroupBox
        
    def playRateChange(self):
        if self.playrate_cbbox.currentText() == "2x":
            playrate = 2
        elif self.playrate_cbbox.currentText() == "1.5x":
            playrate = 1.5
        elif self.playrate_cbbox.currentText() == "1x":
            playrate = 1
        elif self.playrate_cbbox.currentText() == "0.75x":
            playrate = 0.75
        elif self.playrate_cbbox.currentText() == "0.5x":
            playrate = 0.5

        self.videoplayer.setPlaybackRate(playrate)
        self.videoplayer.play()

    def onBtnPlayClicked(self):
        if self.videoplayer.state() == QMediaPlayer.PlayingState:
            self.videoplayer.pause()
            self.play_btn.setIcon(QIcon(f"{ICONDIR}/play.png"))
        else:
            self.videoplayer.play()
            self.play_btn.setIcon(QIcon(f"{ICONDIR}/pause.png"))

    def dragSlider(self, position):
        self.videoplayer.setPosition(position)

    def onDurationChanged(self, duration):
        self.slider_video.setRange(0, duration)

    def onPositionChanged(self):
        current_num = self.videoplayer.position()
        self.slider_video.setValue(current_num)

    def generateHomography(self):
        image_pixel = np.array([[21.0, 24.0], [379.0, 24.0], [21.0, 829.0], [379.0, 829.0]], dtype=np.float32)
        court_coor = np.array([[-3.01, 6.66], [3.01, 6.66], [-3.01, -6.66], [3.01, -6.66]], dtype=np.float32)
        self.H, status = cv2.findHomography(court_coor, image_pixel)

    def paintImage(self):
        painter = QPainter()
        pixmap = QPixmap(f'{ICONDIR}/court/std_court.png')
        painter.begin(pixmap)
        color = QColor()

        # draw start points
        x = self.info['location']['x']
        y = self.info['location']['y']
        z = self.info['location']['z']
        start_pixel = self.H @ np.array([x, y, 1], dtype=np.float32)
        start_pixel = start_pixel / start_pixel[2]
        color.setRgb(255,0,0,255)
        painter.setPen(QPen(color, 5))
        painter.drawPoint(QPoint(start_pixel[0], start_pixel[1]))
        painter.setFont(QFont("Times", 12))
        painter.drawText(QPoint(start_pixel[0]+5, start_pixel[1]-5), f'({x}, {y}, {z})')

        # draw landing points
        for i, (dir, land_points) in enumerate(self.land_info.items()):
            color.setRgb((i*50+20)%256, (i*50+20)%256, (i*50+20)%256, 255)
            for land_point in land_points:
                painter.setPen(QPen(color, 5))
                painter.drawPoint(QPoint(land_point[0], land_point[1]))
            # draw curve fitting arc
            if dir == self.cbo_video.currentText():
                dis = np.mean(np.linalg.norm(land_points - [start_pixel[0], start_pixel[1]], axis=1)) #avg distance from start point to landing points
                topleft_pixel = np.array([start_pixel[0]-dis, start_pixel[1]+dis])
                botright_pixel = np.array([start_pixel[0]+dis, start_pixel[1]-dis])
                rect = QRectF(QPoint(topleft_pixel[0], topleft_pixel[1]), QPoint(botright_pixel[0], botright_pixel[1]))
                color.setRgb((i*50+20)%256, (i*50+20)%256, (i*50+20)%256, 180)
                painter.setPen(QPen(color, 5))
                painter.drawArc(rect, -30*16, -120*16)

                speed = self.info['setting'][dir]['speed']
                pitch = self.info['setting'][dir]['pitch']
                self.speedInfo_label.setText(f'{round(speed*3.6, 2)} (km/h)')
                self.pitchInfo_label.setText(f'{round(pitch, 2)} (°)')

        painter.end()
        self.court.setPixmap(pixmap)

    def landingPointsPressEvent(self, event):
        if event.buttons() == Qt.LeftButton:
            mouse_x = event.pos().x()
            mouse_y = event.pos().y()
            
            min_dis = sys.float_info.max
            target_dir = None
            for dir, land_points in self.land_info.items():
                dis = np.min(np.linalg.norm(land_points - [mouse_x, mouse_y], axis=1))
                if dis < min_dis:
                    min_dis = dis
                    target_dir = dir
            self.cbo_video.setCurrentText(target_dir)

    def deleteUI(self):
        while self.layout_main.count():
            item = self.layout_main.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()
        self.layout_main.deleteLater()

    def backhome(self):
        msg = MsgContract(MsgContract.ID.PAGE_CHANGE, value='Home')
        self.myService.sendMessage(msg)
