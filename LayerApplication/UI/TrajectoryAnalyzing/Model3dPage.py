import os
import sys
import csv
import time
import logging

from PyQt5.QtWidgets import QGroupBox, QVBoxLayout, QComboBox, QWidget, QHBoxLayout, QPushButton, QTableWidget, QGridLayout, QTableWidgetItem, QLabel, QCheckBox, QGraphicsScene, QGraphicsView
from PyQt5.QtCore import QSize, QUrl, QSizeF
from PyQt5.QtGui import QIcon, QFont, QPixmap
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtMultimediaWidgets import QVideoWidget, QGraphicsVideoItem
from UISettings import *

from lib.common import ICONDIR, REPLAYDIR

from ..Services import SystemService, MsgContract
from ..TrajectoryAnalyzing.Trajectory3DVisualizeWidget import Trajectory3DVisualizeWidget

from lib.point import Point
from lib.smooth import removeOuterPoint, detectEvent, smoothByEvent, detectBallTypeByEvent
from lib.trajectory_pred import placement_predict
from lib.BallInfo import getBallInfo, writeBallInfo

from LayerContent.model3D_cut import cut_csv

class Model3dPage(QGroupBox):
    def __init__(self):
        super().__init__()

        # 用來放每次讀檔的軌跡，一段存一起
        # visibility = 1, event = 0 是球在飛行
        # event = 1 是擊球
        # event = 2 是發球
        # event = 3 是死球
        self.hit_points = []
        self.updating_player = False
        self.tracks = []
        self.ball_type_a = {
            '小球': 0,
            '挑球': 0,
            '推球': 0,
            '平球': 0,
            '切球': 0,
            '殺球': 0,
            '長球': 0,
            '例外': 0
        }
        self.ball_type_b = {
            '小球': 0,
            '挑球': 0,
            '推球': 0,
            '平球': 0,
            '切球': 0,
            '殺球': 0,
            '長球': 0,
            '例外': 0
        }

        # current points in video
        self.cur_points = 1
        self.max_points = 1

        self.TracksByTrackID = {}
        self.track_duration = {}
        self.TableButton_dict = {}
        self.filtered_long_shots = []
        self.filtered_A_shots = []
        self.filtered_B_shots = []
        self.play_or_pause = 1

        self.long_shot = 1
        self.A_shot = 1
        self.B_shot = 1
        self.tableOtherButtonIsClicked = False
        self.tableButtonIsClicked = False

        # setup UI
        self.setupUI()

        self.widget_court.A_checkbox.clicked.connect(self.updateTrackView)
        self.widget_court.B_checkbox.clicked.connect(self.updateTrackView)

    def updateTrackView(self):
        if self.widget_court.A_checkbox.isChecked() and self.widget_court.B_checkbox.isChecked():
            self.updateTrackViewAB()

        elif self.widget_court.A_checkbox.isChecked() and not self.widget_court.B_checkbox.isChecked():
            self.updateTrackViewA()
            for i in self.filtered_B_shots:
                button_name = str(i)
                TableButton = self.TableButton_dict.get(button_name)
                if TableButton:
                    self.ButtonDisable(TableButton)
                    TableButton.setEnabled(False)

            self.TableButton = self.TableButton_dict.get(str(self.filtered_A_shots[0]))#當只有A checkbox勾起來顯示A的第一拍被點擊
            self.ButtonClicked(self.TableButton)

        elif self.widget_court.B_checkbox.isChecked() and not self.widget_court.A_checkbox.isChecked():
            self.updateTrackViewB()
            for i in self.filtered_A_shots:
                button_name = str(i)
                TableButton = self.TableButton_dict.get(button_name)
                if TableButton:
                    self.ButtonDisable(TableButton)
                    TableButton.setEnabled(False)

            self.TableButton = self.TableButton_dict.get(str(self.filtered_B_shots[0]))#當只有B checkbox勾起來顯示A的第一拍被點擊
            self.ButtonClicked(self.TableButton)  
        else:
            for i in self.filtered_long_shots:
                button_name = str(i)
                TableButton = self.TableButton_dict.get(button_name)
                if TableButton:
                    self.ButtonDisable(TableButton)
                    TableButton.setEnabled(False)

    def ButtonDisable(self,button):
        button.setStyleSheet(UIStyleSheet.Model3dTableButtonDisable)
    
    def ButtonClicked(self,button):
        button.setStyleSheet(UIStyleSheet.Model3dTableButtonClicked)
    
    def updateTrackViewA(self):
        date = self.date_cb.currentText()
        current_mode = self.widget_court.mode
        self.A_shot = 1
        self.widget_court.current_shot = self.widget_court.filtered_A_shots[self.A_shot - 1]
        self.cur_points = self.widget_court.current_shot
        # print('cur:', self.widget_court.current_shot)
        self.showTrajectory(date)
        self.widget_court.mode = current_mode

    def updateTrackViewB(self):
        date = self.date_cb.currentText()
        current_mode = self.widget_court.mode
        self.B_shot = 1
        self.widget_court.current_shot = self.widget_court.filtered_B_shots[self.B_shot - 1]
        self.cur_points = self.widget_court.current_shot
        # print('cur:', self.widget_court.current_shot)
        self.showTrajectory(date)
        self.widget_court.mode = current_mode

    def updateTrackViewAB(self):
        date = self.date_cb.currentText()
        current_mode = self.widget_court.mode
        self.long_shot = 1
        self.widget_court.current_shot = self.widget_court.filtered_long_shots[self.long_shot - 1]
        self.cur_points = self.widget_court.current_shot
        self.showTrajectory(date)
        self.widget_court.mode = current_mode


    def hideEvent(self, event):
        self.widget_court.clearAll()
        logging.debug(f"{self.__class__.__name__}: hided.")

    def showEvent(self, event):
        self.date_cb.blockSignals(True)
        self.date_cb.clear()
        dates = [f for f in os.listdir(REPLAYDIR)]
        dates.sort()
        self.date_cb.addItem('')
        self.date_cb.addItems(dates)
        self.date_cb.setCurrentText(dates[-1])
        date = self.date_cb.currentText()
        self.date_cb.blockSignals(False)
        date_dir = os.path.join(REPLAYDIR, date)
        self.mediaPlayer.setMedia(QMediaContent(QUrl.fromLocalFile(os.path.join(date_dir, "CameraReader_1.mp4"))))
        removeOuterPoint(date)
        # cut_csv(date)
        # files = [f for f in os.listdir(os.path.join(REPLAYDIR, date)) if f.startswith('Model3D_mod_')]
        self.max_points = 1
        self.showTrajectory(date)
        logging.debug(f"{self.__class__.__name__}: shown.")
        self.play_video()

    def setBackgroundService(self, service:SystemService):
        self.myService = service

    def setupButton(self, icon=f"{ICONDIR}/defaultappicon.png"):
        # logging.debug(icon)
        button = QPushButton()
        # button.setMaximumWidth(200)
        button.setIcon(QIcon(icon))
        button.setIconSize(QSize(30,30))
        return button

    def setupUI(self):
        layout_main = QHBoxLayout()

        control_widget = QWidget()
        control_layout = QVBoxLayout()

        self.control_bar = self.getControlBar()
        self.control_3d = self.getControl3d()
        control_layout.addWidget(self.control_bar)
        control_layout.addWidget(self.control_3d)
        control_widget.setLayout(control_layout)

        table_widget = QWidget()
        table_layout = QVBoxLayout()
        self.balltype_tb = self.getTable()
        self.ballinfo_tb = self.getInfoTable()
        table_layout.addWidget(self.balltype_tb)
        # table_layout.addWidget(self.ballinfo_tb)
        table_widget.setLayout(table_layout)

        rightHandSide_widget = QWidget()
        rightHandSide_layout = QVBoxLayout()

        self.mediaPlayer = QMediaPlayer(None, QMediaPlayer.VideoSurface)
        self.mediaPlayer.positionChanged.connect(self.positionChanged)
        self.mediaPlayer.mediaStatusChanged.connect(self.check_media_status)
        self.mediaPlayer.setNotifyInterval(100)
        # self.videoWidget = QVideoWidget()
        self.videoWidget = QGraphicsVideoItem()
        # self.videoWidget.setFixedSize(480,360)
        # self.videoWidget.setFixedSize(360,480)
        self.videoWidget.setSize(QSizeF(330, 440))
        scene = QGraphicsScene(self)
        graphicsView = QGraphicsView(scene)
        scene.addItem(self.videoWidget)
        self.videoWidget.setRotation(270)

        buttonList_widget = QWidget()
        buttonList_layout = QHBoxLayout()
        self.checkbox = QCheckBox('單拍播放', self)
        self.checkbox.setStyleSheet(UIStyleSheet.Model3dCheckBox)
        self.checkbox.setChecked(False)
        self.checkbox.setEnabled(True)
        self.checkbox.stateChanged.connect(self.positionChanged)
        self.backwardButton = self.setupButton(icon = f"{ICONDIR}/Model3dbackward.png")
        self.backwardButton.setStyleSheet(UIStyleSheet.SelectButton)
        self.backwardButton.clicked.connect(self.mediaBackward)
        self.playButton = self.setupButton(icon = f"{ICONDIR}/pause_model3D.png")
        self.playButton.setStyleSheet(UIStyleSheet.ReturnButton)
        self.playButton.clicked.connect(self.mediaButtonClicked)
        self.forwardButton = self.setupButton(icon = f"{ICONDIR}/Model3dforward.png")
        self.forwardButton.setStyleSheet(UIStyleSheet.SelectButton)
        self.forwardButton.clicked.connect(self.mediaForward)
        buttonList_layout.addWidget(self.checkbox) 
        buttonList_layout.addWidget(self.backwardButton)
        buttonList_layout.addWidget(self.playButton)
        buttonList_layout.addWidget(self.forwardButton)
        buttonList_widget.setLayout(buttonList_layout)

        player_widget = QWidget()
        player_layout = QVBoxLayout()
        # player_layout.addWidget(self.videoWidget)
        player_layout.addWidget(graphicsView)
        player_layout.addWidget(buttonList_widget)
        player_widget.setLayout(player_layout)

        self.mediaPlayer.setVideoOutput(self.videoWidget)

        self.ABTable = self.getABTable()

        rightHandSide_layout.addWidget(player_widget)
        rightHandSide_layout.addWidget(self.ABTable)
        rightHandSide_widget.setLayout(rightHandSide_layout)

        layout_main.addWidget(control_widget)
        layout_main.addWidget(rightHandSide_widget)
        # layout_main.addWidget(table_widget)
        self.setLayout(layout_main)

    def getControlBar(self):
        container = QWidget()
        container_layout = QHBoxLayout()

        self.btn_home = QPushButton()
        self.btn_home.setText('回首頁')
        self.btn_home.setFixedSize(QSize(120, 50))
        self.btn_home.setStyleSheet(UIStyleSheet.ReturnButton)
        self.btn_home.clicked.connect(self.backhome)

        # replay 日期選擇
        self.date_cb = QComboBox()
        self.date_cb.setStyleSheet(UIStyleSheet.Model3dCombobox)
        self.date_cb.currentTextChanged.connect(self.date_cb_changed)
        self.date_cb.setFixedSize(1100,50)

        container_layout.addWidget(self.btn_home)
        container_layout.addWidget(self.date_cb)
        container.setLayout(container_layout)
        return container

    def getControl3d(self):
        container = QWidget()
        container_layout = QHBoxLayout()

        # 3D 軌跡圖
        self.widget_court = Trajectory3DVisualizeWidget()
        self.widget_court.setMinimumSize(800, 800)

        # self.nextbtn = QPushButton(self)
        # self.nextbtn.clicked.connect(self.onClickNext)
        # self.nextbtn.setFixedSize(QSize(60, 96))
        # self.nextbtn.setIcon(QIcon(f"{ICONDIR}/right_arrow.png"))
        # self.nextbtn.setIconSize(QSize(55, 55))

        # change the court to show next shot
        # self.lastbtn = QPushButton(self)
        # self.lastbtn.clicked.connect(self.onClickLast)
        # self.lastbtn.setFixedSize(QSize(60, 96))
        # self.lastbtn.setIcon(QIcon(f"{ICONDIR}/left_arrow.png"))
        # self.lastbtn.setIconSize(QSize(55, 55))

        # container_layout.addWidget(self.lastbtn)
        container_layout.addWidget(self.widget_court)
        # container_layout.addWidget(self.nextbtn)
        container.setLayout(container_layout)
        return container

    def mediaButtonClicked(self):
        # Change the icon when the button is clicked
        # self.button.setIcon(QIcon('new_icon.png'))
        if self.play_or_pause == 1:
            self.playButton.setIcon(QIcon(f"{ICONDIR}/button.png"))
            self.mediaPlayer.pause()
            self.play_or_pause = 0
        else:
            self.playButton.setIcon(QIcon(f"{ICONDIR}/pause_model3D.png"))
            self.mediaPlayer.play()
            self.play_or_pause = 1

    def play_video(self):
        if self.mediaPlayer.state() == QMediaPlayer.PlayingState:
            self.mediaPlayer.pause()
        else:
            self.mediaPlayer.play()
    
    def positionChanged(self, position):
        if self.widget_court.A_checkbox.isChecked() and not self.widget_court.B_checkbox.isChecked():
            self.checkbox.setChecked(True)
            self.checkbox.setEnabled(False)
            self.checkbox.setStyleSheet(UIStyleSheet.Model3dCheckBoxDisable)
            self.check_position(position)
            self.tableOtherButtonIsClicked = False
        elif self.widget_court.B_checkbox.isChecked() and not self.widget_court.A_checkbox.isChecked():
            self.checkbox.setChecked(True)
            self.checkbox.setEnabled(False)
            self.checkbox.setStyleSheet(UIStyleSheet.Model3dCheckBoxDisable)
            self.check_position(position)
            self.tableOtherButtonIsClicked = False
        elif not self.widget_court.A_checkbox.isChecked() and not self.widget_court.B_checkbox.isChecked():
            self.checkbox.setChecked(False)
            self.checkbox.setEnabled(False)
            self.checkbox.setStyleSheet(UIStyleSheet.Model3dCheckBoxDisable)
            self.mediaReplay(position)
            self.tableOtherButtonIsClicked = False
        else:
            self.checkbox.setEnabled(True) 
            self.checkbox.setStyleSheet(UIStyleSheet.Model3dCheckBox)
            if self.checkbox.isChecked():
                self.check_position(position)
                if(len(self.TableButton_dict)!=0 and not self.tableOtherButtonIsClicked):
                    if len(self.filtered_long_shots) > 0:
                        self.TableButton = self.TableButton_dict.get(str(self.filtered_long_shots[0]))
                        self.ButtonClicked(self.TableButton)
            else:
                self.mediaReplay(position)
                # if(len(self.TableButton_dict)!=0 and self.tableOtherButtonIsClicked):
                #     self.TableButton.setStyleSheet(UIStyleSheet.Model3dTableButton)
                if(len(self.TableButton_dict)!=0 and not self.tableOtherButtonIsClicked):
                    if len(self.filtered_long_shots) > 0:
                        self.TableButton = self.TableButton_dict.get(str(self.filtered_long_shots[0]))
                        self.ButtonClicked(self.TableButton)

    # def singlePlayerMediaReplay(self, position, cur_shot):
    #     print('cur:::', cur_shot)
    #     if self.updating_player:
    #         return
    #     if len(self.hit_points) != 0 and cur_shot!=len(self.filtered_long_shots):#過濾最後一球，之後長球寫好可以拿掉後面這個條件
    #         self.l = int(self.hit_points[cur_shot - 1] / 120 * 1000)
    #         self.r = int(self.hit_points[cur_shot] / 120 * 1000)
    #         if position > self.r - 100:
    #             self.updating_player = True
    #             self.mediaPlayer.pause()
    #             time.sleep(1)
                
    #             # self.mediaPlayer.play()
    #             # self.mediaPlayer.setPosition(self.l)
    #             # self.mediaPlayer.play()
    #             self.updating_player = False

    def mediaReplay(self,position):
        if self.updating_player:
            return
        if len(self.hit_points) != 0:
            self.l = int(self.hit_points[0] / 120 * 1000)
            self.r = int(self.mediaPlayer.duration())
            # print("right: ",self.r)
            # print("position: ",position + 100)
            if position + 300 > self.r:
                self.updating_player = True
                self.mediaPlayer.pause()
                time.sleep(1)
                self.mediaPlayer.play()
                self.mediaPlayer.setPosition(0)
                self.updating_player = False

    def mediaForward(self):
        self.mediaPlayer.setPosition(self.mediaPlayer.position() + 1000)
    def mediaBackward(self):
        self.mediaPlayer.setPosition(self.mediaPlayer.position() - 1500)   

    def check_media_status(self, status):
        if status == 3:
            # if self.checkbox.isChecked():
            self.check_position(0)
            self.play_video()
            # else:
            #     self.mediaPlayer.setPosition(0)
            #     self.play_video()
        # print(QMediaPlayer.EndOfMedia)

    def check_position(self, position=None):
        if self.updating_player:
            return
        # print(QMediaPlayer.PlayingState)
        # if self.mediaPlayer.state() != QMediaPlayer.PlayingState:
        #     self.play_video()
        # if len(self.hit_points) < self.max_points + 1 and self.mediaPlayer.duration() != 0:
        #     self.hit_points.append(int(self.mediaPlayer.duration() / 1000 * 120) - 1)
        # print(self.hit_points)
        # print(self.max_points)

        if len(self.hit_points) != 0:
            self.l = int(self.hit_points[self.cur_points - 1] / 120 * 1000)
            self.r = int(self.hit_points[self.cur_points] / 120 * 1000)
            if position+100 > self.hit_points[-1] / 120 * 1000:
                self.updating_player = True
                self.mediaPlayer.pause()
                time.sleep(1)
                self.mediaPlayer.play()
                self.mediaPlayer.setPosition(max(0, self.l - 500))
                self.updating_player = False
            if position > min(self.hit_points[-1] / 120 * 1000, self.r - 50):
                self.updating_player = True
                self.mediaPlayer.pause()
                time.sleep(1)
                self.mediaPlayer.play()
                self.mediaPlayer.setPosition(max(0, self.l - 500))
                self.updating_player = False

    def getABTable(self):
        self.table = QTableWidget(10, 2)
        self.table.setStyleSheet(UIStyleSheet.Model3dTable)
        self.table.setColumnWidth(0, 223)
        self.table.setColumnWidth(1, 223)
        for i in range (10):
            self.table.setRowHeight(i,50)

        verticalHeader = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
        self.table.setVerticalHeaderLabels(verticalHeader)
        self.table.verticalHeader().setFont(QFont('Arial', 15))

        horizontalHeader = ['A', 'B']
        self.table.setHorizontalHeaderLabels(horizontalHeader)
        self.table.horizontalHeader().setFont(QFont('Arial', 15))
        return self.table

    def getABInfoTable(self):
        self.table.clearContents()
        for k in range (max(len(self.filtered_A_shots),len(self.filtered_B_shots))):
            if k<len(self.filtered_A_shots):
                button_name =f"{self.filtered_A_shots[k]}"
                button_info = "{:.2f}".format(self.maxHeightDict[int(button_name)])
                TableButton = QPushButton(button_name + "  (" + str(button_info) + "m)")
                TableButton.setStyleSheet(UIStyleSheet.Model3dTableButton)
                TableButton.setObjectName(button_name)
                item = QTableWidgetItem()
                self.table.setCellWidget(k, 0, TableButton)
                TableButton.clicked.connect(self.tableButtonClicked)
                self.TableButton_dict[button_name] = TableButton
            if k<len(self.filtered_B_shots):
                button_name =f"{self.filtered_B_shots[k]}"
                button_info = "{:.2f}".format(self.maxHeightDict[int(button_name)])
                TableButton = QPushButton(button_name + "  (" + str(button_info) + "m)")
                TableButton.setStyleSheet(UIStyleSheet.Model3dTableButton)
                TableButton.setObjectName(button_name)
                self.table.setCellWidget(k, 1, TableButton)
                TableButton.clicked.connect(self.tableButtonClicked)
                self.TableButton_dict[button_name] = TableButton
        return self.table
    
    def tableButtonClicked(self):
        sender = self.sender()  # 获取发送信号的对象
        button_name = sender.objectName()
        self.tableButtonIsClicked = True
        self.cur_points = int(button_name)
        # logging.debug(f'current = {self.cur_points}')
        date = self.date_cb.currentText()
        current_mode = self.widget_court.mode
        self.showTrajectory(date)
        self.widget_court.mode = current_mode
        self.TableButton = self.TableButton_dict.get(button_name)
        self.ButtonClicked(self.TableButton)
        self.tableOtherButtonIsClicked = True
        self.tableButtonIsClicked = False

    def getTable(self):
        table = QTableWidget(len(self.ball_type_a), 3)
        table.setColumnWidth(0, 150)
        table.setColumnWidth(1, 150)
        table.setColumnWidth(2, 150)
        i = 0
        for k in self.ball_type_a:
            table.setRowHeight(i, 49)
            type_item = QTableWidgetItem(k)
            type_item.setFont(QFont('Times', 20))
            count_item_a = QTableWidgetItem(str(0))
            count_item_a.setFont(QFont('Times', 20))
            count_item_b = QTableWidgetItem(str(0))
            count_item_b.setFont(QFont('Times', 20))
            table.setItem(i, 0, type_item)
            table.setItem(i, 1, count_item_a)
            table.setItem(i, 2, count_item_b)
            i += 1

        header = ['球種', '發球方', '接發球方']
        table.setHorizontalHeaderLabels(header)
        table.horizontalHeader().setFont(QFont('Times', 12))
        return table

    def getInfoTable(self):
        table = QTableWidget()
        table.setColumnCount(5)
        table.setRowCount(1)
        table.setColumnWidth(0, 95)
        table.setColumnWidth(1, 95)
        table.setColumnWidth(2, 80)
        table.setColumnWidth(3, 90)
        table.setColumnWidth(4, 90)

        header = ['球種', '球速(km/h)', '角度', '過網高度', '最大高度']
        table.setHorizontalHeaderLabels(header)
        table.horizontalHeader().setFont(QFont('Times', 12))
        return table

    def showTrack(self):
        total_tracks = 0
        self.widget_court.clearAll()
        # self.
        for track in self.tracks:
            for point in track:
                if point.event == 1:
                    point.color = 'blue'
                    total_tracks += 1
                if point.event == 2:
                    point.color = 'red'
                    total_tracks += 1
                    # (這邊還沒做事件偵測，先都通時呈現不分段)
                if point.event == 3:
                    point.color = 'green'
                    total_tracks += 1
                    # (理論上這段是雜訊)
                self.widget_court.addPointByTrackID(point, total_tracks)
        self.max_points = total_tracks - 1
        # logging.debug(f"total = {total_tracks}")

        if total_tracks > 0:
            self.TracksByTrackID = self.widget_court.getTracksByTrackID()
            self.filterTrack()
            self.ABtable = self.getABInfoTable()    #等filterTrack 統計好A、B的球再顯示數據在ABTable上
            if self.widget_court.A_checkbox.isChecked() and not self.widget_court.B_checkbox.isChecked():
                for i in self.filtered_B_shots:
                    button_name = str(i)
                    TableButton = self.TableButton_dict.get(button_name)
                    if TableButton:
                        self.ButtonDisable(TableButton)
                        TableButton.setEnabled(False)
            elif self.widget_court.B_checkbox.isChecked() and not self.widget_court.A_checkbox.isChecked():
                for i in self.filtered_A_shots:
                    button_name = str(i)
                    TableButton = self.TableButton_dict.get(button_name)
                    if TableButton:
                        self.ButtonDisable(TableButton)
                        TableButton.setEnabled(False)
            elif not self.widget_court.A_checkbox.isChecked() and not self.widget_court.B_checkbox.isChecked():
                for i in self.filtered_long_shots:
                    button_name = str(i)
                    TableButton = self.TableButton_dict.get(button_name)
                    if TableButton:
                        self.ButtonDisable(TableButton)
                        TableButton.setEnabled(False)
            self.trackDuration()       

    def filterTrack(self):
        self.filtered_long_shots = []
        self.filtered_A_shots = []
        self.filtered_B_shots = []
        self.maxHeightDict = {}
        for shot, track in self.TracksByTrackID.items():
            if shot <= self.max_points and self.ball_type[shot - 1] == '長球':
                if shot not in self.filtered_long_shots:
                    self.filtered_long_shots.append(shot)
                highest_point = max(track, key=lambda point: point.z)
                self.maxHeightDict[shot] = highest_point.z
        for shot, track in self.TracksByTrackID.items():
            if shot in self.filtered_long_shots:
                if track[0].y <= 0:
                    if shot not in self.filtered_A_shots:
                        self.filtered_A_shots.append(shot)
                else:
                    if shot not in self.filtered_B_shots:
                        self.filtered_B_shots.append(shot)

    def trackDuration(self):
        self.track_duration = {}
        timestamp = []
        for shot, track in self.TracksByTrackID.items():
            for point in track:
                timestamp.append(point.timestamp)
                # if point.event == 3:
                #     end_timestamp = point.timestamp

        for i in self.filtered_long_shots:
            self.track_duration[i] = timestamp[self.hit_points[i]] - timestamp[self.hit_points[i-1]]
        self.widget_court.shotDuration(self.track_duration)

    def readPoints(self, tracks_file):
        points = []
        with open(tracks_file, 'r', newline='') as csvFile:
            rows = csv.DictReader(csvFile)
            for row in rows:
                if int(float(row['Visibility'])) != 0:
                    point = Point(fid=row['Frame'], timestamp=row['Timestamp'], visibility=row['Visibility'],
                                    x=row['X'], y=row['Y'], z=row['Z'],
                                    event=row['Event'])
                    points.append(point)
        self.tracks.append(points)

    def showBallType(self, ball_type):
        if ball_type != []:
            for i, type in enumerate(ball_type):
                if i % 2 == 0:
                    self.ball_type_a[type] += 1
                else:
                    self.ball_type_b[type] += 1

        for i in range(self.balltype_tb.rowCount()):
            type = self.balltype_tb.item(i, 0).text()
            count_a = self.ball_type_a[type]
            count_a_item = QTableWidgetItem(str(count_a))
            count_a_item.setFont(QFont('Times', 20))
            count_b = self.ball_type_b[type]
            count_b_item = QTableWidgetItem(str(count_b))
            count_b_item.setFont(QFont('Times', 20))
            self.balltype_tb.setItem(i, 1, count_a_item)
            self.balltype_tb.setItem(i, 2, count_b_item)

    def showBallInfo(self, ball_info, ball_type):
        try:
            self.ballinfo_tb.setRowCount(len(ball_info))
            for i, r in enumerate(ball_info):
                self.ballinfo_tb.setRowHeight(i, 40)
                type_item = QTableWidgetItem(f'{ball_type[i]}')
                type_item.setFont(QFont('Times', 18))
                speed_item = QTableWidgetItem(f'{r["Speed"]:.2f}')
                speed_item.setFont(QFont('Times', 18))
                angle_item = QTableWidgetItem(f'{r["Angle"]:.1f}°')
                angle_item.setFont(QFont('Times', 18))
                midheight_item = QTableWidgetItem(f'{r["MidHeight"]:.2f}m')
                midheight_item.setFont(QFont('Times', 18))
                maxheight_item = QTableWidgetItem(f'{r["MaxHeight"]:.2f}m')
                maxheight_item.setFont(QFont('Times', 18))

                self.ballinfo_tb.setItem(i, 0, type_item)
                self.ballinfo_tb.setItem(i, 1, speed_item)
                self.ballinfo_tb.setItem(i, 2, angle_item)
                self.ballinfo_tb.setItem(i, 3, midheight_item)
                self.ballinfo_tb.setItem(i, 4, maxheight_item)
        except Exception as e:
            logging.error(f"error in showBallInfo {e}")

    def date_cb_changed(self):
        if self.date_cb.currentText() == '':
            self.widget_court.clearAll()
            self.cur_points = 1
            self.max_points = 1
            for k in self.ball_type_a:
                self.ball_type_a[k] = 0
                self.ball_type_b[k] = 0
        else:
            self.cur_points = 1
            self.max_points = 1
            date = self.date_cb.currentText()
            date_dir = os.path.join(REPLAYDIR, date)
            self.mediaPlayer.setMedia(QMediaContent(QUrl.fromLocalFile(os.path.join(date_dir, "CameraReader_1.mp4"))))
            removeOuterPoint(date)
            # do something to cut mod to mod_1, mod_2, ...
            # cut_csv(date)
            # files = [f for f in os.listdir(os.path.join(REPLAYDIR, date)) if f.startswith('Model3D_mod_')]
            self.max_points = 1
            # self.showTrajectory(date)
            self.long_shot = 1
            self.A_shot = 1
            self.B_shot = 1
            self.widget_court.filtered_long_shots = []
            self.widget_court.filtered_A_shots = []
            self.widget_court.filtered_B_shots = []
            current_mode = self.widget_court.mode
            self.showTrajectory(date)
            self.widget_court.mode = current_mode

    def showTrajectory(self, date):
        self.widget_court.clearAll()
        for k in self.ball_type_a:
            self.ball_type_a[k] = 0
            self.ball_type_b[k] = 0
        self.tracks = []
        detectEvent(date, 1)
        smoothByEvent(date, 1)
        self.ball_type = detectBallTypeByEvent(date, 1)
        self.widget_court.ballType(self.ball_type)
        ball_info = getBallInfo(date, 1)
        self.widget_court.ballInfo(ball_info)
        self.hit_points = []

        # for i in ball_info:
        #     self.hit_points.append(i['Frame'])
        # self.hit_points.append(9999999999)
        
        
        # print(ball_info)

        # ball_info_file = 'Model3D_info_' + str(self.cur_points) + '.csv'
        # ball_info_path = os.path.join(REPLAYDIR, date, ball_info_file)
        # writeBallInfo(ball_info_path, ball_info, ball_type)

        file = 'Model3D_smooth_' + str(1) + '.csv'
        track_file = os.path.join(REPLAYDIR, date, file)
        #predict trajectory
        # placement_predict(date,file)

        try:
            self.readPoints(tracks_file = track_file)
            for i in ball_info:
                self.hit_points.append(i['Frame'])
            for track in self.tracks:
                for p in range(len(track)-1,-1,-1):
                    if track[p].event == 3:
                        self.hit_points.append(track[p].fid)
                        break

            self.showTrack()
            if(not self.tableButtonIsClicked):
                if len(self.filtered_long_shots) > 0:
                    if self.widget_court.A_checkbox.isChecked() and self.widget_court.B_checkbox.isChecked():
                        self.cur_points = self.filtered_long_shots[self.long_shot - 1]
                    elif self.widget_court.A_checkbox.isChecked() and not self.widget_court.B_checkbox.isChecked():
                        self.cur_points = self.filtered_A_shots[self.A_shot - 1]
                    elif self.widget_court.B_checkbox.isChecked() and not self.widget_court.A_checkbox.isChecked():
                        self.cur_points = self.filtered_B_shots[self.B_shot - 1]
                else:
                    self.cur_points = 0
            self.widget_court.setShotText(self.cur_points)
            self.widget_court.setShot(self.cur_points)
            # self.showBallType(ball_type = ball_type)
            # self.showBallInfo(ball_info=ball_info, ball_type=ball_type)
        except Exception as e:
            logging.warning("[Model3D_smooth] No Csv Files: {} err = {}".format(track_file, e))
        # self.hit_points.append(self.mediaPlayer.duration())

    def backhome(self):
        msg = MsgContract(MsgContract.ID.PAGE_CHANGE, value='TrajectoryAnalyzingPage')
        self.myService.sendMessage(msg)

    def onClickNext(self):
        if self.widget_court.A_checkbox.isChecked() and self.widget_court.B_checkbox.isChecked():
            self.long_shot += 1
            if self.long_shot > len(self.filtered_long_shots):
                self.long_shot = 1
            self.cur_points = self.filtered_long_shots[self.long_shot - 1]
            if self.cur_points > self.max_points:
                self.long_shot = 1
                self.cur_points = self.filtered_long_shots[self.long_shot - 1]
            logging.debug(f'current = {self.cur_points}')
        elif self.widget_court.A_checkbox.isChecked():
            self.A_shot += 1
            if self.A_shot > len(self.filtered_A_shots):
                self.A_shot = 1
            self.cur_points = self.filtered_A_shots[self.A_shot - 1]
            if self.cur_points > self.max_points:
                self.A_shot = 1
                self.cur_points = self.filtered_A_shots[self.A_shot - 1]
            logging.debug(f'current = {self.cur_points}')
        elif self.widget_court.B_checkbox.isChecked():
            self.B_shot += 1
            if self.B_shot > len(self.filtered_B_shots):
                self.B_shot = 1
            self.cur_points = self.filtered_B_shots[self.B_shot - 1]
            if self.cur_points > self.max_points:
                self.B_shot = 1
                self.cur_points = self.filtered_B_shots[self.B_shot - 1]
            logging.debug(f'current = {self.cur_points}')
        date = self.date_cb.currentText()
        current_mode = self.widget_court.mode
        self.showTrajectory(date)
        self.widget_court.mode = current_mode
        # self.mediaPlayer.pause()
        self.mediaPlayer.setPosition(int(self.hit_points[self.cur_points - 1] / 120 * 1000))

    def onClickLast(self):
        if self.widget_court.A_checkbox.isChecked() and self.widget_court.B_checkbox.isChecked():
            self.long_shot -= 1
            if self.long_shot <= 0:
                self.long_shot = len(self.filtered_long_shots)
            self.cur_points = self.filtered_long_shots[self.long_shot - 1]
            if self.cur_points > self.max_points:
                self.long_shot -= 1
                self.cur_points = self.filtered_long_shots[self.long_shot - 1]
            logging.debug(f'current = {self.cur_points}')
        elif self.widget_court.A_checkbox.isChecked():
            self.A_shot -= 1
            if self.A_shot <= 0:
                self.A_shot = len(self.filtered_A_shots)
            self.cur_points = self.filtered_A_shots[self.A_shot - 1]
            if self.cur_points > self.max_points:
                self.A_shot -= 1
                self.cur_points = self.filtered_A_shots[self.A_shot - 1]
            logging.debug(f'current = {self.cur_points}')
        elif self.widget_court.B_checkbox.isChecked():
            self.B_shot -= 1
            if self.B_shot <= 0:
                self.B_shot = len(self.filtered_B_shots)
            self.cur_points = self.filtered_B_shots[self.B_shot - 1]
            if self.cur_points > self.max_points:
                self.B_shot -= 1
                self.cur_points = self.filtered_B_shots[self.B_shot - 1]
            logging.debug(f'current = {self.cur_points}')
        date = self.date_cb.currentText()
        current_mode = self.widget_court.mode
        self.showTrajectory(date)
        self.widget_court.mode = current_mode
        # self.mediaPlayer.pause()
        self.mediaPlayer.setPosition(int(self.hit_points[self.cur_points - 1] / 120 * 1000))
