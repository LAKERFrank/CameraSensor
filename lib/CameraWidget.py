import os
import sys
import logging
import threading
import multiprocessing as mp
import json
import queue
import paho.mqtt.client as mqtt
import time
from datetime import datetime
import cv2
import numpy as np
from enum import Enum, auto
from turbojpeg import TurboJPEG
import shutil
import ast
from vidgear.gears import WriteGear

from PyQt5.QtWidgets import QGroupBox, QVBoxLayout, QComboBox, QWidget, QGridLayout, QLabel, QPushButton, QHBoxLayout, QSpinBox, QScrollArea, QDialog, QDoubleSpinBox
from PyQt5.QtCore import QSize, QThread, pyqtSignal, pyqtSlot, QTimer, QMetaMethod
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.Qt import *
from PyQt5.QtMultimedia import QSound

import gi
gi.require_version("Gst", "1.0")

from gi.repository import GLib, Gst, Tcam

DIRNAME = os.path.dirname(os.path.abspath(__file__))
ROOTDIR = os.path.dirname(DIRNAME)
ICONDIR = f"{DIRNAME}/icon"
AUX = '(1440, 1080)'
BUX = '(2048, 1536)'

from LayerApplication.UI.UISettings import *
from LayerApplication.UI.Services import SystemService, MsgContract

from lib.nodes import CameraReader
from lib.frame import Frame
from lib.message import *
from lib.common import insertById, loadConfig, saveConfig, setIntrinsicMtx, checkAUXorBUX

# bitrate control, replace None with:
# 2 mb/s : "2M"
# 500 kb/s : "500K"
CONSTANT_BITRATE = None

class FirstFrameState(Enum):
    NOT_READY = auto()
    KEEP = auto()
    DISCARD = auto()

class ClickLabel(QLabel):

    clicked = pyqtSignal(str)

    def __init__(self, imagename):
        super().__init__()
        self.imagename = imagename

    def mousePressEvent(self, event):
        self.clicked.emit(self.imagename)
        QLabel.mousePressEvent(self, event)

class CameraWidget(QWidget):
    # [connectable]
    # (camera idx, image)
    getImageSignal = pyqtSignal(int, QPixmap)

    def __init__(self, cfg, cameras:list, num_cameras):
        super().__init__()

        # initalize Service
        self.myService = None
        self.cameras = cameras
        self.broker = cfg["Project"]["mqtt_broker"]
        self.is_recording = False
        self.is_streaming = False
        self.record_done_cnt = 0
        self.snapshot_dir_path = os.path.join(ROOTDIR, "snapshot")
        os.makedirs(self.snapshot_dir_path, exist_ok=True)
        self.chessboard_path = f"{ROOTDIR}/Reader/{self.cameras[0].brand}/intrinsic_data/{self.cameras[0].hw_id}"
        logging.debug(self.chessboard_path)
        os.makedirs(self.chessboard_path, exist_ok=True)

        self.snapshot = []
        # self.image_size = QSize(480, 360)
        # self.image_size = QSize(360, 480)
        self.image_size = QSize(285, 380)

        self.preview_fps = 30

        # self.num_cameras = num_cameras
        self.num_cameras = len(self.cameras)
        self.selected_num_cameras = self.num_cameras
        self.idx_list = []

        self.initCameraInfo()

        self.blockSize = 10

        # setup UI
        self.setupUI()

        # for first frame
        self.first_frame_time = [0] * self.num_cameras
        self.first_frame_cnt = 0
        self.camera_start_timestamp = 0

        # get the serial numbers of existing camera
        Gst.init(sys.argv)
        self.source = Gst.ElementFactory.make("tcambin")
        self.serials = self.source.get_device_serials_backend()
        # logging.debug(f"serials from cameraSystem: {self.serials}")

        # find getImageSignal in metaobjs
        # WTF
        # https://stackoverflow.com/questions/8166571/how-to-find-if-a-signal-is-connected-to-anything
        self.getImageSignal_method = None
        oMetaObj = self.metaObject()
        for i in range (oMetaObj.methodCount()):
            oMetaMethod = oMetaObj.method(i)
            if not oMetaMethod.isValid():
                continue
            if oMetaMethod.methodType() == QMetaMethod.Signal and oMetaMethod.name() == 'getImageSignal':
                self.getImageSignal_method = oMetaMethod

    # [callable]
    # !! must be called at setupUI in your page
    def initWidget(self, num, image_size=QSize(480, 360), idx_list=[]):
        self.selected_num_cameras = num
        self.image_size = image_size
        if len(idx_list) == 0:
            self.idx_list = list(range(num))
        elif num == len(idx_list):
            self.idx_list = idx_list
        else:
            logging.error(f"Error initWidget num={num} != len={len(idx_list)}")
            return

        self.setHidden(False, self.idx_list)
        self.setHidden(True, [i for i in range(self.num_cameras) if i not in self.idx_list])

        # print(f'init widget called = {self.idx_list}')
        #initAllCameraFPS
        for i in range(len(self.preview_fps_label_list)):
            self.preview_fps_label_list[i].setText("preview FPS:0")

        tmp_list = []
        for i in self.idx_list:
            if self.cameras[i].hw_id == 'None':
                continue
            tmp_list.append(i)
        self.idx_list = tmp_list

        self.selected_num_cameras = len(self.idx_list)

    # [callable]
    # will be called automatically at showEvent
    def startStreaming(self, delta=2):
        if self.is_streaming:
            return
        self.is_streaming = True
        # check if all required cameras exist.
        all_camera_exist = True
        for i in self.idx_list:
            if (self.cameras[i].hw_id + "-v4l2") not in self.serials:
                all_camera_exist = False
                break

        # logging.debug(f'all_camera_exist = {all_camera_exist}')
        if not all_camera_exist:
            return

        for i in self.idx_list:
            self.camera_process_list[i].preview.defaultImage()

        self.initRecordInfo()

        self.first_frame_time = [0] * self.num_cameras
        self.first_frame_cnt = 0

        def sendStartMessage():
        #   啟動相機 streaming
            self.camera_start_timestamp = datetime.now().timestamp() + 0.5
            for i in self.idx_list:
                camera = self.cameras[i]

                camera.isStreaming = True
                msg = MsgContract(MsgContract.ID.CAMERA_STREAM)
                msg.value = camera
                msg.data = self.camera_start_timestamp
                self.myService.sendMessage(msg)

        QTimer.singleShot(delta * 1000, sendStartMessage)

    # [callable]
    # will be called automatically at hideEvent
    def stopStreaming(self, record_resolution=None):
        if not self.is_streaming:
            return
        self.is_streaming = False

        def sendStopMessage():
        #   tell all cameras to reinitialize(stop)
            for i in self.idx_list:
                msg = MsgContract(MsgContract.ID.CAMERA_SETTING)
                msg.value = self.cameras[i]
                msg.data = dict()
                msg.data['Reinit'] = True
                if record_resolution != None:
                    msg.data['RecordResolution'] = str(record_resolution)
                self.myService.sendMessage(msg)

                # pause all camera process
                for i in self.idx_list:
                    self.camera_process_list[i].pause()

        QTimer.singleShot(100, sendStopMessage)

    # [callable]
    def startRecording(self, idx_list=[]):
        """Start Recording Video

        Args:
            idx_list (list[int]): index of camera

        Returns:
            str: replay dir path
        """
        if self.is_recording:
            return
        self.is_recording = True
        self.record_done_cnt = 0

        self.first_frame_time = [0] * self.num_cameras
        self.first_frame_cnt = 0

        cur_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.replay_dir = os.path.join(ROOTDIR, "replay", cur_time)
        # make recorders start at specified timestamp
        start_timestamp = datetime.now().timestamp() + 10.5 * (1 / 120)

        indexes = idx_list if len(idx_list) > 0 else self.idx_list

        # fix for firstFrameSignal
        self.selected_num_cameras = len(indexes)

        for i in indexes:
            print(f"recording camera[{i}]")
            self.camera_process_list[i].startRecord(cur_time, start_timestamp)
        return os.path.realpath(self.replay_dir)

    # [callable]
    def stopRecording(self, idx_list=[]):
        if not self.is_recording:
            return
        self.is_recording = False

        indexes = idx_list if len(idx_list) > 0 else self.idx_list

        for i in indexes:
            self.camera_process_list[i].stopRecord()

        # fix for firstFrameSignal
        self.selected_num_cameras = len(self.idx_list)

    # [callable]
    def takeSnapshot(self):
        cur_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        for i in self.idx_list:
            pixmap = QPixmap(self.snapshot[i])
            filename = cur_time + '_' + str(i) + '.png'
            filepath = os.path.join(self.snapshot_dir_path, cur_time)
            if not os.path.exists(filepath):
                os.makedirs(filepath)
            filepath = os.path.join(filepath, filename)
            pixmap.save(filepath)
        logging.info(f"Take snapshot at: {cur_time}")

    # [callable]
    def setPreviewSize(self, image_size=QSize(480, 360)):
        self.image_size = image_size

    # [callable]
    def setHidden(self, is_hidden:bool, idx_list:list=[]):
        for i in idx_list:
            self.camera_label_list[i].setHidden(is_hidden)
            self.fps_container_list[i].setHidden(is_hidden)

    # [callable]
    def toggleFPS(self):
        for i in self.idx_list:
            self.fps_container_list[i].setHidden(not self.fps_container_list[i].isHidden())

    def showEvent(self, event):
        logging.debug(f"{self.__class__.__name__}: shown.")
        # self.startStreaming(1.25)

    def hideEvent(self, event):
        logging.debug(f"{self.__class__.__name__}: hided.")
        # self.stopStreaming()

    def setBackgroundService(self, service:SystemService):
        self.myService = service

    def initCameraInfo(self):
        p = mp.get_context("spawn").Pool(self.num_cameras)
        res = []
        # init all camera'(1440, 1080)'1s' fps and resolution

        for i in range(self.num_cameras):
            #get modelName(AUX or BUX)
            self.modelName = checkAUXorBUX(self.cameras[i].hw_id)
            if self.cameras[i].hw_id == 'None':
                continue
            config_file = f"{ROOTDIR}/Reader/{self.cameras[i].brand}/config/{self.cameras[i].hw_id}.cfg"
            cameraCfg = loadConfig(config_file)
            if(self.modelName == "AUX"):
                if self.num_cameras >= 4:
                    cameraCfg['Camera']['fps'] = '30'
                    cameraCfg['Camera']['RecordResolution'] = AUX
                elif self.num_cameras == 3:
                    cameraCfg['Camera']['fps'] = '60'
                    cameraCfg['Camera']['RecordResolution'] = AUX
                elif self.num_cameras == 2:
                    cameraCfg['Camera']['fps'] = '60'
                    cameraCfg['Camera']['RecordResolution'] = AUX
                else:
                    cameraCfg['Camera']['fps'] = '120'
                    cameraCfg['Camera']['RecordResolution'] = AUX
            elif(self.modelName == "BUX"):
                if self.num_cameras >= 4:
                    cameraCfg['Camera']['fps'] = '30'
                    cameraCfg['Camera']['RecordResolution'] = BUX
                elif self.num_cameras == 3:
                    cameraCfg['Camera']['fps'] = '60'
                    cameraCfg['Camera']['RecordResolution'] = BUX
                elif self.num_cameras == 2:
                    cameraCfg['Camera']['fps'] = '60'
                    cameraCfg['Camera']['RecordResolution'] = BUX
                else:
                    cameraCfg['Camera']['fps'] = '119'#BUX's FPS cannot reach 120
                    cameraCfg['Camera']['RecordResolution'] = BUX
            saveConfig(config_file, cameraCfg)

            image_path = f"{ROOTDIR}/Reader/{self.cameras[i].brand}/intrinsic_data/{self.cameras[i].hw_id}"
            os.makedirs(image_path, exist_ok=True)
            res.append(p.apply_async(setIntrinsicMtx, (config_file, image_path, cameraCfg['Camera']['RecordResolution'])))

        for i, r in enumerate(res):
            try:
                ret_v = r.get()
                if ret_v == 0:
                    logging.debug(f'Calculate intrinsic of camera[{i}] ok')
                elif ret_v == -2:
                    logging.debug(f'Calculate intrinsic of camera[{i}] < 4 pictures')
                else:
                    logging.debug(f'Calculate intrinsic of camera[{i}] error')
            except Exception as e:
                logging.error('Multiprocessing intrinsic error..')

        p.close()
        p.join()

    def initRecordInfo(self):
        fps_changed = False
        for i in range(self.num_cameras):
            config_file = f"{ROOTDIR}/Reader/{self.cameras[i].brand}/config/{self.cameras[i].hw_id}.cfg"
            cameraCfg = loadConfig(config_file)
            record_fps = int(cameraCfg['Camera']['fps'])
            record_resolution = ast.literal_eval(cameraCfg['Camera']['RecordResolution'])
            record_resolution_qsize = QSize(record_resolution[0], record_resolution[1]) #original
            # check if fps changed
            if record_fps != self.record_fps_list[i]:
                fps_changed = True
                self.record_fps_list[i] = record_fps

            self.camera_process_list[i].setRecordInfo(record_fps=record_fps, record_image_size=record_resolution_qsize)

        # if fps changed ask process to pause
        if fps_changed:
            for i in range(self.num_cameras):
                self.camera_process_list[i].pause()

    def setupUI(self):
        self.camera_block_layout = self.cameraPreview(number=self.num_cameras)
        self.setLayout(self.camera_block_layout)

    def addSingleCamera(self, container_layout, camera_id, row, col):
        self.snapshot.append(None)
        # camera
        camera_label = QLabel()
        camera_label.setText("Camera_Preview_" + str(camera_id))
        self.camera_label_list.append(camera_label)

        # FPS
        # add container to make hide and show running correctly
        fps_layout = QHBoxLayout()
        fps_container = QWidget()
        fps_container.resize(700, 50)
        preview_fps_label = QLabel()
        preview_fps_label.setText("preview FPS:0")
        preview_fps_label.setStyleSheet(UIStyleSheet.Check3DLabel)
        self.preview_fps_label_list.append(preview_fps_label)

        record_fps_label = QLabel()
        # record_fps_label.setText("record FPS:0")
        self.record_fps_label_list.append(record_fps_label)

        fps_layout.addWidget(preview_fps_label)
        fps_layout.addWidget(record_fps_label)
        fps_container.setLayout(fps_layout)
        self.fps_container_list.append(fps_container)

        # load record_fps from config file
        config_file = f"{ROOTDIR}/Reader/{self.cameras[camera_id].brand}/config/{self.cameras[camera_id].hw_id}.cfg"
        cameraCfg = loadConfig(config_file)
        record_fps = int(cameraCfg['Camera']['fps'])
        self.record_fps_list.append(record_fps)

        camera_process = CameraProcessThread(camera_id, self.broker, self.cameras[camera_id], self.preview_fps, record_fps, self.image_size)
        camera_process.firstFrameSignal.connect(self.checkFirstFrame)
        camera_process.preview.drawImageSignal.connect(self.updateCameraImage)
        camera_process.preview.drawFPSSignal.connect(self.updateCameraPreviewFPS)
        camera_process.recorder.drawFPSSignal.connect(self.updateCameraRecordFPS)
        camera_process.recorder.firstFrameSignal.connect(self.checkFirstFrame)
        camera_process.start()
        self.camera_process_list.append(camera_process)

        container_layout.addWidget(camera_label, row * 2, col, alignment=Qt.AlignBottom | Qt.AlignCenter) # 直、橫
        container_layout.addWidget(fps_container, row * 2 + 1, col, alignment=Qt.AlignCenter | Qt.AlignTop)

    def cameraPreview(self, number=2):
        container_layout = QGridLayout()

        self.camera_label_list = []
        self.fps_container_list = []
        self.preview_fps_label_list = []
        self.record_fps_label_list = []
        self.record_fps_list = []
        self.camera_process_list = []
        self.camera_process_list: list[CameraProcessThread] = []

        self.addSingleCamera(container_layout, 1 - 1, 0, 2)
        self.addSingleCamera(container_layout, 2 - 1, 0, 0)
        self.addSingleCamera(container_layout, 3 - 1, 1, 0)

        self.addSingleCamera(container_layout, 4 - 1, 1, 2)
        self.addSingleCamera(container_layout, 5 - 1, 0, 1)
        self.addSingleCamera(container_layout, 6 - 1, 1, 1)

        return container_layout

    # [connectable]
    @pyqtSlot(int, QPixmap)
    def trueDrawImage(self, camera_id: int, image: QPixmap):
        self.camera_label_list[camera_id].setPixmap(image.scaled(self.image_size))

    # this slot will be called by preview thread
    @pyqtSlot(int, QPixmap)
    def updateCameraImage(self, camera_id: int, image: QPixmap):
        self.snapshot[camera_id] = image
        # if getImageSignal is connected, transmit image to it.
        if self.isSignalConnected(self.getImageSignal_method):
            self.getImageSignal.emit(camera_id, image)
        else:
            self.trueDrawImage(camera_id, image)

    @pyqtSlot(int, str)
    def updateCameraPreviewFPS(self, camera_id: int, fps: str):
        self.preview_fps_label_list[camera_id].setText(fps)

    @pyqtSlot(int, str)
    def updateCameraRecordFPS(self, camera_id: int, fps: str):
        self.record_fps_label_list[camera_id].setText(fps)

    @pyqtSlot(int, float, bool)
    def checkFirstFrame(self, camera_id, timestamp, is_record):
        # logging.debug(f'camera[{camera_id}] check (cnt={self.first_frame_cnt}/num={self.num_cameras})')
        if is_record:
            target = self.camera_process_list[camera_id].recorder
        else:
            target = self.camera_process_list[camera_id]

        # discard old images from previous startup
        if timestamp < self.camera_start_timestamp:
            target.setFirstFrameState(FirstFrameState.DISCARD)
            return

        if self.first_frame_cnt == self.selected_num_cameras:
            latest_timestamp = max(self.first_frame_time)
            # print(f'[{camera_id}] {latest_timestamp} {timestamp}')
            if (latest_timestamp - timestamp) > (1 / (self.record_fps_list[camera_id] * 2)):
                self.first_frame_time[camera_id] = 0
                self.first_frame_cnt -= 1
                target.setFirstFrameState(FirstFrameState.DISCARD)
            else:
                target.setFirstFrameState(FirstFrameState.KEEP)
            return

        if self.first_frame_time[camera_id] == 0:
            self.first_frame_time[camera_id] = timestamp
            self.first_frame_cnt += 1

        target.setFirstFrameState(FirstFrameState.NOT_READY)


# receive images from mqtt and communicate with preview and recorder thread
class CameraProcessThread(QThread):
    firstFrameSignal = pyqtSignal(int, float, bool)

    def __init__(self, camera_id, mqtt_broker, camera:CameraReader, preview_fps, record_fps, image_size = QSize(800, 600)):
        super().__init__()

        self.camera_id = camera_id
        self.preview_fps = preview_fps
        self.record_fps = record_fps

        # for first frame check
        self.first_frame_state = FirstFrameState.NOT_READY
        self.pause_timestamp = 0

        # Event Queue
        self.waitEvent = threading.Event()
        self.messageQueue = []

        # 暫存收到的 image
        self.frameQueue = []
        self.image_size = image_size
        self.camera = camera

        # Mqtt client for receive raw image
        client = mqtt.Client()
        client.on_connect = self.onConnect
        client.on_message = self.onMessage
        client.connect(mqtt_broker)
        self.client = client

        self.jpeg = TurboJPEG('/usr/lib/x86_64-linux-gnu/libturbojpeg.so.0')

        self.preview = CameraPreviewThread(self.camera_id, self.preview_fps, self.image_size)
        self.recorder = CameraRecorder(self.camera_id, self.camera, self.record_fps, self.image_size)

    def onConnect(self, client, userdata, flag, rc):
        logging.info(f"{self.__class__.__name__}: Connected with result code: {rc}")
        self.client.subscribe(self.camera.output_topic)

    def onMessage(self, client, userdata, msg):
        data = json.loads(msg.payload)
        if 'id' in data and 'timestamp' in data and 'raw_data' in data:
            if data['timestamp'] < self.pause_timestamp:
                return
            frame = Frame(data['id'], data['timestamp'], data['raw_data'])
            insertById(self.frameQueue, frame)
            msg = MqttContract(MqttContract.ID.SUBSCRIBE, msg.topic, None)
            self.messageQueue.append(msg)
            self.waitEvent.set()

    def setRecordInfo(self, record_fps=None, record_image_size=None):
        if record_fps != None:
            self.record_fps = record_fps
        self.recorder.set_record_info(record_fps=record_fps, image_size=record_image_size)

    def pause(self):
        self.pause_timestamp = datetime.now().timestamp()
        self.messageQueue.append('[pause cameraprocess]')
        self.waitEvent.set()

    def startRecord(self, cur_time, start_timestamp):
        self.recorder.set_start_info(cur_time, start_timestamp)
        self.recorder.start()

    def stopRecord(self):
        self.recorder.stop()

    def setFirstFrameState(self, state):
        self.first_frame_state = state

    # check other threads' first frame time
    def checkFirstFrame(self, timestamp):
        self.first_frame_state = FirstFrameState.NOT_READY
        while True:
            self.firstFrameSignal.emit(self.camera_id, timestamp, False)
            if self.first_frame_state == FirstFrameState.NOT_READY:
                time.sleep(0.01)
                continue
            return self.first_frame_state == FirstFrameState.KEEP

    def setPreviewSize(self, preview_size):
        self.preview.setImageSize(preview_size)

    def getPreviewSize(self):
        return self.preview.getImageSize()

    def run(self):
        self.preview.start()
        first = True
        try:
            self.client.loop_start()
            while True:
                try:
                    if len(self.messageQueue) < 2:
                        self.waitEvent.wait()
                        self.waitEvent.clear()

                    # I don't know why queue here may be empty, so I wrap it with try
                    try:
                        msg = self.messageQueue.pop(0)
                    except:
                        continue

                    # use [first] flag to synchronize cameras
                    if msg == '[pause cameraprocess]':
                        first = True
                        self.frameQueue.clear()
                        self.messageQueue.clear()
                        self.preview.defaultImage()
                        continue

                    if msg.id == MqttContract.ID.STOP:
                        self.frameQueue.clear()
                        logging.debug(f"{self.__class__.__name__}: stop...")
                        break
                    elif msg.id == MqttContract.ID.SUBSCRIBE:
                        # decode image
                        frame = self.frameQueue.pop(0)
                        # logging.debug(f"{frame.fid}")
                        image = frame.coverToCV2ByTurboJPEG(self.jpeg)

                        # check if first frame is same time
                        if first:
                            logging.debug(f"camera[{self.camera_id}]'s fid[{frame.fid}] timestamp : {frame.timestamp}")
                            if not self.checkFirstFrame(frame.timestamp):
                                logging.debug(f"camera[{self.camera_id}]'s fid[{frame.fid}] => DISCARD")
                                continue
                            else:
                                logging.debug(f"camera[{self.camera_id}]'s fid[{frame.fid}] => KEEP")
                                first = False


                        self.preview.try_put_frame(image, frame.timestamp)
                        self.recorder.try_put_frame(image, frame.timestamp)

                except queue.Empty:
                    logging.warn(f"{self.__class__.__name__}: the message queue is empty.")
        finally:
            self.client.loop_stop()

class CameraPreviewThread(QThread):
    # to communicate with CameraSystem thread
    drawImageSignal = pyqtSignal(int, QPixmap)
    drawFPSSignal = pyqtSignal(int, str)
    def __init__(self, camera_id, preview_fps, image_size = QSize(800, 600)):
        super().__init__()

        self.camera_id = camera_id
        self.preview_fps = preview_fps
        self.last_preview_t = 0

        self.waitEvent = threading.Event()

        # 暫存收到的 image
        self.frameQueue = queue.Queue(maxsize=self.preview_fps * 2)
        self.image_size = image_size

        # 用來計算接收到的平均 FPS
        self.num_frame_calc = self.preview_fps // 2
        self.frame_time_queue = queue.Queue(maxsize=(self.num_frame_calc + 10))

    def try_put_frame(self, image, timestamp):
        try:
            self.frameQueue.put_nowait((image, timestamp))
            self.waitEvent.set()
        except queue.Full:
            self.frameQueue.queue.clear()
            logging.warning(f"preview[{self.camera_id}] frame queue is full.")

    def drawImage(self, image):
        index = self.camera_id
        # 這裡轉 image
        if(index == 0 or index == 2):
            image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        elif(index == 1 or index == 3):
            image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        height, width, channel = image.shape

        bytesPerLine = 3 * width
        qimage = QImage(image.data, width, height, bytesPerLine, \
                QImage.Format_RGB888).rgbSwapped()

        self.drawImageSignal.emit(self.camera_id, QPixmap(qimage))

    def getAvgFPS(self):
        now_time = time.time()
        self.frame_time_queue.put(now_time)

        if self.frame_time_queue.qsize() <= 1:
            return 0

        if self.frame_time_queue.qsize() < self.num_frame_calc:
            first_frame_time = self.frame_time_queue.queue[0]
            return int((self.frame_time_queue.qsize() - 1) / (now_time - first_frame_time))

        first_frame_time = self.frame_time_queue.get()
        return int((self.num_frame_calc - 1) / (now_time - first_frame_time))

    # set default image
    def defaultImage(self):
        self.drawImageSignal.emit(self.camera_id, QPixmap(f"{ICONDIR}/no_camera.png"))

    def setImageSize(self, image_size):
        self.image_size = image_size

    def getImageSize(self):
        return self.image_size

    def run(self):
        self.defaultImage()

        while True:
            if self.frameQueue.qsize() < 2:
                self.waitEvent.wait()
                self.waitEvent.clear()

            image, timestamp = self.frameQueue.get()
            if (timestamp - self.last_preview_t) > (1 / (self.preview_fps * 1.1)):
                self.drawImage(image)
                avg_fps = self.getAvgFPS()
                self.drawFPSSignal.emit(self.camera_id, f"preview FPS:{avg_fps}")
                self.last_preview_t = timestamp

class CameraRecorder(QThread):
    drawFPSSignal = pyqtSignal(int, str)
    firstFrameSignal = pyqtSignal(int, float, bool)

    def __init__(self, camera_id, camera:CameraReader, record_fps=None, image_size = QSize(800, 600)):
        super().__init__()

        self.camera_id = camera_id
        self.image_size = image_size
        self.camera = camera

        self.record_fps = record_fps
        self.is_recording = False

        self.record_dir_path = os.path.join(ROOTDIR, "replay")
        os.makedirs(self.record_dir_path, exist_ok=True)
        self.cur_time = None
        self.start_timestamp = None

        self.waitEvent = threading.Event()

        # for first frame check
        self.first_frame_state = FirstFrameState.NOT_READY

    def getAvgFPS(self):
        now_time = time.time()
        self.frame_time_queue.put(now_time)
        if self.frame_time_queue.qsize() <= 1:
            return 0

        if self.frame_time_queue.qsize() < self.num_frame_calc:
            first_frame_time = self.frame_time_queue.queue[0]
            return int((self.frame_time_queue.qsize() - 1) / (now_time - first_frame_time))

        first_frame_time = self.frame_time_queue.get()
        return int((self.num_frame_calc - 1) / (now_time - first_frame_time))

    def setFirstFrameState(self, state):
        self.first_frame_state = state

    # check other threads' first frame time
    def checkFirstFrame(self, timestamp):
        self.first_frame_state = FirstFrameState.NOT_READY
        while True:
            self.firstFrameSignal.emit(self.camera_id, timestamp, True)
            if self.first_frame_state == FirstFrameState.NOT_READY:
                time.sleep(0.01)
                continue
            return self.first_frame_state == FirstFrameState.KEEP

    def set_record_info(self, record_fps=None, image_size=None):
        if record_fps != None:
            self.record_fps = record_fps
        if image_size != None:
            self.image_size = image_size

    def set_start_info(self, cur_time, start_timestamp):
        self.cur_time = cur_time
        self.start_timestamp = start_timestamp

    def init_writer(self):
        # record information
        self.file_dir = os.path.join(self.record_dir_path, self.cur_time)
        os.makedirs(self.file_dir, exist_ok=True)

        self.file_name = os.path.join(self.file_dir, self.camera.name + ".mp4")

        # old cv2 writer
        # self.writer = cv2.VideoWriter(self.file_name, cv2.VideoWriter_fourcc(*'mp4v'), self.record_fps, \
        #     (self.image_size.width(), self.image_size.height()))

        output_params = {
            "-input_framerate": self.record_fps,
            "-disable_force_termination": True,
            "-vcodec": "h264_nvenc",

            # use cpu to encode
            # "-vcodec": "libx264"
            # "-preset": "fast",
            # "-tune": "zerolatency"
        }

        if CONSTANT_BITRATE:
            output_params["-rc"] = "cbr"
            output_params["-b:v"] = CONSTANT_BITRATE

        #                                          logging can display ffmpeg debug informations
        self.writer = WriteGear(output_filename=self.file_name, logging=False, **output_params)

        # 暫存收到的影像
        self.frameQueue = queue.Queue(maxsize=self.record_fps * 2)

        # 用來計算接收到的平均 FPS
        self.num_frame_calc = self.record_fps // 2
        self.frame_time_queue = queue.Queue(maxsize=(self.num_frame_calc + 10))

    def stop(self):
        logging.info(f"Stop recording camera-{self.camera.name}")
        self.is_recording = False
        self.waitEvent.set()

    def try_put_frame(self, image, timestamp):
        if not self.is_recording:
            return
        # reserve some tolerance on start time
        if timestamp > self.start_timestamp - ((1 / self.record_fps) / 2):
            try:
                self.frameQueue.put_nowait((image, timestamp))
                self.waitEvent.set()
            except queue.Full:
                self.frameQueue.queue.clear()
                logging.warning(f"recorder[{self.camera_id}] frame queue is full.")

    def saveConfig(self):
        main_config_file = f"{ROOTDIR}/config"
        camera_config_file = f"{ROOTDIR}/Reader/{self.camera.brand}/config/{self.camera.hw_id}.cfg"
        shutil.copy2(main_config_file, self.file_dir)
        shutil.copy2(camera_config_file, self.file_dir)

    def run(self):
        logging.info(f"Start recording camera-{self.camera.name}")
        self.init_writer()
        self.is_recording = True
        first = True

        try:
            while self.is_recording:
                if self.frameQueue.empty():
                    self.waitEvent.wait(timeout=1)
                    self.waitEvent.clear()
                if not self.is_recording:
                    break
                image, timestamp = self.frameQueue.get(timeout=1)
                if first:
                    # logging.debug(f"camera[{self.camera_id}]'s timestamp : {timestamp}")
                    if not self.checkFirstFrame(timestamp):
                        logging.debug(f"camera[{self.camera_id}]'s {timestamp} => DISCARD")
                        continue
                    else:
                        logging.debug(f"camera[{self.camera_id}]'s {timestamp} => KEEP")
                        first = False

                self.writer.write(image)

                avg_fps = self.getAvgFPS()
                self.drawFPSSignal.emit(self.camera_id, f"record FPS:{avg_fps}")
        except:
            pass
        finally:
            logging.info(f"Stop writing...{self.camera.name}")

        self.drawFPSSignal.emit(self.camera_id, f"record FPS:0")
        logging.info(f"Saving video... [{self.file_name}]")
        self.writer.close()
        self.saveConfig()
        logging.info(f"Saving completed... [{self.file_name}]")