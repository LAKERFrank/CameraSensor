import logging
import threading
import time
import json
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np

from LayerApplication.Rpc.RpcStreamingBadminton import RpcStreamingBadminton
from LayerApplication.UI.TrajectoryAnalyzing.Trajectory import TrajectoryWidget
from LayerCamera.RpcCameraWidget import RpcCameraWidget
from lib.point import Point
from .BallPointWidget import BallPointWidget

from ..UISettings import *
from ..Services import SystemService
from lib.message import *

from lib.common import ICONDIR

class StreamingDemoPage(QGroupBox):
    signal_tracknet_0 = pyqtSignal(bytes)
    signal_tracknet_1 = pyqtSignal(bytes)
    signal_tracknet_2 = pyqtSignal(bytes)
    signal_tracknet_3 = pyqtSignal(bytes)
    signal_content_point = pyqtSignal(bytes)
    signal_content_event = pyqtSignal(bytes)
    signal_content_segment = pyqtSignal(bytes)

    def __init__(self, rpcStreamingBadminton:RpcStreamingBadminton, rpcCameraWidget:RpcCameraWidget):
        super().__init__()

        self.is_recording = False
        self.mode = ""  # 初始化 mode 變數

        self.rpcStreamingBadminton = rpcStreamingBadminton
        self.rpcCameraWidget = rpcCameraWidget

        self.signal_tracknet_0.connect(self.sensing_callback_0)
        self.signal_tracknet_1.connect(self.sensing_callback_1)
        self.signal_tracknet_2.connect(self.sensing_callback_2)
        self.signal_tracknet_3.connect(self.sensing_callback_3)
        self.signal_content_point.connect(self.content_callback_point)
        self.signal_content_event.connect(self.content_callback_event)
        self.signal_content_segment.connect(self.content_callback_segment)

    def hideEvent(self, event):
        logging.debug(f"{self.__class__.__name__}: hided.")
        self.deleteUI()

        self.rpcStreamingBadminton.sensing_callbacks[0] = None
        self.rpcStreamingBadminton.sensing_callbacks[1] = None
        self.rpcStreamingBadminton.sensing_callbacks[2] = None
        self.rpcStreamingBadminton.sensing_callbacks[3] = None
        self.rpcStreamingBadminton.content_callback_point = None
        self.rpcStreamingBadminton.content_callback_event = None
        self.rpcStreamingBadminton.content_callback_segment = None

    def deleteUI(self):
        self.rpcCameraWidget.show()
        self.stack_layout.removeWidget(self.rpcCameraWidget)
        self.layout_main.deleteLater()

    def showEvent(self, event):
        self.setupUI()

        self.rpcStreamingBadminton.sensing_callbacks[0] = lambda x: self.signal_tracknet_0.emit(x)
        self.rpcStreamingBadminton.sensing_callbacks[1] = lambda x: self.signal_tracknet_1.emit(x)
        self.rpcStreamingBadminton.sensing_callbacks[2] = lambda x: self.signal_tracknet_2.emit(x)
        self.rpcStreamingBadminton.sensing_callbacks[3] = lambda x: self.signal_tracknet_3.emit(x)
        self.rpcStreamingBadminton.content_callback_point = lambda x: self.signal_content_point.emit(x)
        self.rpcStreamingBadminton.content_callback_event = lambda x: self.signal_content_event.emit(x)
        self.rpcStreamingBadminton.content_callback_segment = lambda x: self.signal_content_segment.emit(x)

        logging.debug(f"{self.__class__.__name__}: shown.")

    def setBackgroundService(self, service:SystemService):
        self.myService = service

    def backhome(self):
        msg = MsgContract(MsgContract.ID.PAGE_CHANGE, value='Home')
        self.myService.sendMessage(msg)

    def setupUI(self):
        # main layout
        self.layout_main = QHBoxLayout()

        self.rpcCameraWidget.initWidget(6)

        self.layout_main.addWidget(self.getControlBar(), 1, Qt.AlignLeft)
        self.layout_main.addWidget(self.getStackLayout(), 2, Qt.AlignCenter)

        self.setLayout(self.layout_main)

        self.tab_list.setCurrentRow(0)


    def _clearOutput(self):
        # clear output
        self.sensing_output_0.setText("")
        self.sensing_output_0.update()
        self.sensing_output_1.setText("")
        self.sensing_output_1.update()
        self.content_output.setText("")
        self.content_output.update()

        self.trajectory_widget.reset()
        self.ball_point_widget.reset()

    def startTest(self):
        self._clearOutput()
        self.ball_point_widget.reset()

        self.mode = self.combo_mode.currentText()

        duration = self.rpcStreamingBadminton.testStart(read_video=False, mode=self.mode)

        self.btn_debug_run.setText(f"Waiting for {duration:.2f} sec")
        self.btn_debug_run.setEnabled(False)

        QTimer.singleShot(duration*1000, self.stopTest)

    def stopTest(self):
        self.rpcStreamingBadminton.testStop(read_video=False)
        self.btn_debug_run.setText("Debug Run")
        self.btn_debug_run.setEnabled(True)
        self.trajectory_widget.render()

    def start(self):

        if not self.is_recording:
            self._clearOutput()

            content_mode = self.combo_mode.currentText()
            record_mode = self.combo_record_mode.currentText()
            tracknet_ver = self.combo_tracknet_ver.currentText()

            self.rpcStreamingBadminton.start(content_mode=content_mode, tracknet_ver=tracknet_ver, record_mode=record_mode)
            self.btn_run.setText(f"Stop")
        else:
            self.rpcStreamingBadminton.stop()
            self.btn_run.setText("Run")
            self.trajectory_widget.render()
            self.ball_point_widget.reset()
        self.is_recording = not self.is_recording

    def sensing_callback_0(self, data:bytes):

        self.sensing_output_0.append(str(data))
        self.sensing_output_0.update()

        jdata = json.loads(data)

        points = [ (int(p['pos']['x']), int(p['pos']['y'])) for p in jdata['linear'] ]

        self.rpcCameraWidget.updateTrackNetPoint(0, points)

    def sensing_callback_1(self, data:bytes):
        self.sensing_output_1.append(str(data))
        self.sensing_output_1.update()

        jdata = json.loads(data)

        points = [ (int(p['pos']['x']), int(p['pos']['y'])) for p in jdata['linear'] ]

        self.rpcCameraWidget.updateTrackNetPoint(1, points)

    def sensing_callback_2(self, data:bytes):
        self.sensing_output_2.append(str(data))
        self.sensing_output_2.update()

        jdata = json.loads(data)

        points = [ (int(p['pos']['x']), int(p['pos']['y'])) for p in jdata['linear'] ]

        self.rpcCameraWidget.updateTrackNetPoint(2, points)

    def sensing_callback_3(self, data:bytes):
        self.sensing_output_3.append(str(data))
        self.sensing_output_3.update()

        jdata = json.loads(data)

        points = [ (int(p['pos']['x']), int(p['pos']['y'])) for p in jdata['linear'] ]

        self.rpcCameraWidget.updateTrackNetPoint(3, points)

    def content_callback_point(self, data:str):
        """處理球點資料"""
        self.content_output.append(f"[Point] {str(data)}")
        self.content_output.update()

        try:
            jdata = json.loads(data)
            points = [ Point.fromJson(d) for d in jdata['linear'] ]

            # 更新球點顯示
            if points:
                self.ball_point_widget.addBallPoints(points)
        except Exception as e:
            logging.error(f"處理球點資料時發生錯誤: {e}")

    def content_callback_event(self, data:str):
        """處理事件資料"""
        self.content_output.append(f"[Event] {str(data)}")
        self.content_output.update()

        try:
            event_data = json.loads(data)
            
            # 更新事件顯示
            self.ball_point_widget.processEventMessage(event_data)
            
        except Exception as e:
            logging.error(f"處理事件資料時發生錯誤: {e}")

    def content_callback_segment(self, data:str):
        """處理段資料"""
        self.content_output.append(f"[Segment] {str(data)}")
        self.content_output.update()

        try:
            segment_data = json.loads(data)
            
            # 更新段顯示
            self.ball_point_widget.processSegmentMessage(segment_data)
            
        except Exception as e:
            logging.error(f"處理段資料時發生錯誤: {e}")

    def getStackLayout(self):
        container = QWidget()
        container.setFixedSize(1500, 850)
        container_layout = QHBoxLayout()

        self.trajectory_widget = TrajectoryWidget()
        self.log_widget = self.getLogWidget()
        
        # 使用新的 BallPointWidget（支援事件和段）
        self.ball_point_widget = BallPointWidget()

        self.stack_list = [
            self.log_widget,
            self.rpcCameraWidget,
            self.trajectory_widget,
            self.ball_point_widget
        ]

        container_layout.addWidget(self.log_widget)
        container_layout.addWidget(self.rpcCameraWidget)
        container_layout.addWidget(self.trajectory_widget)
        container_layout.addWidget(self.ball_point_widget)

        self.stack_layout = container_layout

        container.setLayout(container_layout)
        return container

    def getLogWidget(self):
        container = QWidget()
        container_layout = QGridLayout()

        q0 = QLabel()
        q0.setText("Sensing_0")
        q0.setMinimumSize(100, 30)
        q1 = QLabel()
        q1.setText("Sensing_1")
        q1.setMinimumSize(100, 30)
        q2 = QLabel()
        q2.setText("Sensing_2")
        q2.setMinimumSize(100, 30)
        q3 = QLabel()
        q3.setText("Sensing_3")
        q3.setMinimumSize(100, 30)
        q4 = QLabel()
        q4.setText("Content")
        q4.setMinimumSize(100, 30)

        self.sensing_output_0 = QTextBrowser()
        self.sensing_output_0.setMinimumSize(450, 300)

        self.sensing_output_1 = QTextBrowser()
        self.sensing_output_1.setMinimumSize(450, 300)

        self.sensing_output_2 = QTextBrowser()
        self.sensing_output_2.setMinimumSize(450, 300)

        self.sensing_output_3 = QTextBrowser()
        self.sensing_output_3.setMinimumSize(450, 300)

        self.content_output = QTextBrowser()
        self.content_output.setMinimumSize(450, 750)

        container_layout.addWidget(q0, 0, 0, Qt.AlignmentFlag.AlignCenter)
        container_layout.addWidget(q1, 0, 1, Qt.AlignmentFlag.AlignCenter)
        container_layout.addWidget(q4, 0, 2, Qt.AlignmentFlag.AlignCenter)

        container_layout.addWidget(self.sensing_output_0, 1, 0, Qt.AlignmentFlag.AlignCenter)
        container_layout.addWidget(self.sensing_output_1, 1, 1, Qt.AlignmentFlag.AlignCenter)
        container_layout.addWidget(self.content_output, 1, 2, 3, 1, Qt.AlignmentFlag.AlignCenter)

        container_layout.addWidget(q2, 2, 0, Qt.AlignmentFlag.AlignCenter)
        container_layout.addWidget(q3, 2, 1, Qt.AlignmentFlag.AlignCenter)

        container_layout.addWidget(self.sensing_output_2, 3, 0, Qt.AlignmentFlag.AlignCenter)
        container_layout.addWidget(self.sensing_output_3, 3, 1, Qt.AlignmentFlag.AlignCenter)

        container.setLayout(container_layout)
        return container
    
    def changeTab(self, i):
        for idx, w in enumerate(self.stack_list):
            if i == idx:
                w.show()
            else:
                w.hide()

    def getControlBar(self):
        container = QWidget()
        container_layout = QVBoxLayout()

        self.tab_list = QListWidget()
        self.tab_list.insertItem(0, "Log")
        self.tab_list.insertItem(1, "Camera")
        self.tab_list.insertItem(2, "3D")
        self.tab_list.insertItem(3, "BallPoint (Event+Segment)")
        self.tab_list.currentRowChanged.connect(self.changeTab)

        self.btn_home = QPushButton()
        self.btn_home.setText('回首頁')
        self.btn_home.setFixedSize(QSize(160, 60))
        self.btn_home.setStyleSheet('font: 24px')
        self.btn_home.clicked.connect(self.backhome)

        mode_layout = QHBoxLayout()
        label_mode = QLabel("Content Mode:")
        self.combo_mode = QComboBox()
        self.combo_mode.addItems(["", "CES"])
        mode_layout.addWidget(label_mode, stretch=1)
        mode_layout.addWidget(self.combo_mode, stretch=2)

        record_mode_layout = QHBoxLayout()
        label_record_mode = QLabel("Record Mode:")
        self.combo_record_mode = QComboBox()
        self.combo_record_mode.addItems(["none", "h264_low", "h264_high"])
        self.combo_record_mode.setCurrentText("h264_high")
        record_mode_layout.addWidget(label_record_mode, stretch=1)
        record_mode_layout.addWidget(self.combo_record_mode, stretch=2)

        tracknet_ver_layout = QHBoxLayout()
        label_tracknet_ver = QLabel("TrackNet Ver.:")
        self.combo_tracknet_ver = QComboBox()
        self.combo_tracknet_ver.addItems(["tracknet_v2", "tracknet_1000"])
        self.combo_tracknet_ver.setCurrentText("tracknet_v2")
        tracknet_ver_layout.addWidget(label_tracknet_ver, stretch=1)
        tracknet_ver_layout.addWidget(self.combo_tracknet_ver, stretch=2)

        self.btn_run = QPushButton()
        self.btn_run.setText('Run')
        self.btn_run.setFixedSize(QSize(160, 60))
        self.btn_run.setStyleSheet('font: 24px')
        self.btn_run.clicked.connect(self.start)

        self.btn_debug_run = QPushButton()
        self.btn_debug_run.setText('Debug Run')
        self.btn_debug_run.setFixedSize(QSize(160, 60))
        self.btn_debug_run.setStyleSheet('font: 24px')
        self.btn_debug_run.clicked.connect(self.startTest)

        container_layout.addWidget(self.tab_list)
        container_layout.addWidget(self.btn_home)
        container_layout.addLayout(mode_layout)
        container_layout.addLayout(tracknet_ver_layout)
        container_layout.addLayout(record_mode_layout)
        container_layout.addWidget(self.btn_run)
        container_layout.addWidget(self.btn_debug_run)
        container.setLayout(container_layout)

        return container
