import os
import logging
import cv2
import time
from pathlib import Path
from datetime import datetime

import numpy as np
from PyQt5.QtWidgets import QWidget, QGridLayout, QLabel, QHBoxLayout
from PyQt5.QtCore import QSize, QTimer
from PyQt5.QtGui import QImage

import gi

from LayerCamera.camera.Camera import Camera
from lib.nodes import CameraReader
gi.require_version("Gst", "1.0")

# DIRNAME = os.path.dirname(os.path.abspath(__file__))
# ICONDIR = f"{DIRNAME}/icon"
AUX = '(1440, 1080)'
BUX = '(2048, 1536)'

from LayerApplication.UI.UISettings import UIStyleSheet
from PyQt5.QtCore import Qt

from lib.common import loadConfig

from .CameraSystemC import recorder_module

from lib.common import ROOTDIR

class HPCameraWidget(QWidget):

    def __init__(self, cameras:list):
        super().__init__()

        # initalize Service
        self.cameras:list[CameraReader] = cameras
        self.is_recording = False
        self.is_streaming = False
        self.snapshot_dir_path = os.path.join(ROOTDIR, "snapshot")
        os.makedirs(self.snapshot_dir_path, exist_ok=True)

        self.snapshot = []

        # self.num_cameras = num_cameras
        self.num_cameras = len(self.cameras)
        self.idx_list:list[int] = []

        self.cameraList:list[Camera] = [None for _ in range(self.num_cameras)]
        self.metric_datas:list[recorder_module.MetricData] = [None for _ in range(self.num_cameras)]

        self.recorder = recorder_module.Recorder(ROOTDIR)

        self.label_show_resolution = False

        # setup UI
        self.__setupUI()

    def getAvailableCamera(self) -> "dict[str, str]":
        """Get all available camera on the system

        Returns:
            dict[str, str]: {serial: model}
        """
        return self.recorder.listAvailableCamera()

    def initWidget(self, num, image_size=QSize(380, 285), idx_list=[]):
        # reset hidden
        # FIXME: to be remove
        for i in self.idx_list:
            self.camera_label_list[i].setHidden(False)
            self.fps_container_list[i].setHidden(False)

    def init(self, num, image_size=QSize(380, 285), idx_list=[]):
        """Start camera

        Args:
            num (int): number of cameras
            image_size (QSize, optional): size of preview
            idx_list (list, optional): index list of camera
        """

        self.image_size_h = QSize(480, 360)
        self.image_size_v = QSize(285, 380)

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

        tmp_list = []
        available_cams = [item["serial"] for item in Camera.getAvailableCameras()]
        for i in self.idx_list:
            if self.cameras[i].hw_id == 'None':
                continue
            if self.cameras[i].hw_id not in available_cams:
                continue
        
            tmp_list.append(i)
        self.idx_list = tmp_list

        # winids = []
        # for i in range(len(self.camera_label_list)):
        #     winids.append()
        for i in range(len(self.camera_label_list)):
            self.camera_label_list[i].setFixedSize(self.image_size_h)

        t = time.time_ns()
        for idx in self.idx_list:
            self.cameraList[idx] = Camera(self.cameras[idx].hw_id)

            self.cameraList[idx].init("", 0, self.camera_winid_list[idx])
            self.cameraList[idx].start(t)
            self.metric_datas[idx] = self.cameraList[idx].getMetricData()

            if self.cameraList[idx].direction in [1, 3]:
                self.camera_label_list[idx].setFixedSize(self.image_size_v)

        self.is_streaming = True

    def stop(self):
        if not self.is_streaming:
            return

        self.is_streaming = False

        for idx in self.idx_list:
            if self.cameraList[idx] is None:
                continue
            self.cameraList[idx].release()
            self.cameraList[idx] = None

    def start(self):
        if self.is_streaming:
            return

        self.is_streaming = True

        self.init(self.num_cameras)

    # [callable]
    def startRecording(self, mode="h264_low"):
        """Start Recording Video

        Args:
            idx_list (list[int]): index of camera

        Returns:
            str: replay dir path
        """
        if self.is_recording:
            return
        self.is_recording = True

        # local
        dirname = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        for cam_idx in self.idx_list:
            self.cameraList[cam_idx].startRecording(False, dirname, cam_idx, mode)

        return Path(ROOTDIR) / "replay" / dirname

    # [callable]
    def stopRecording(self, idx_list=[]):
        if not self.is_recording:
            return
        self.is_recording = False

        for cam_idx in self.idx_list:
            self.cameraList[cam_idx].stopRecording()

    def getSnapshotQImage(self, cam_idx) -> QImage:
        if not self.is_streaming:
            return None

        im = self.getSnapshotArray(cam_idx)

        if im is None:
            return None

        return QImage(im.data, im.shape[1], im.shape[0], im.strides[0], QImage.Format_BGR888)

    def getSnapshotArray(self, cam_idx) -> np.ndarray:
        """Get camera snapshot image

        Args:
            cam_idx (int): camera index

        Returns:
            np.ndarray: RGB image
        """
        if not self.is_streaming:
            return None

        if self.cameraList[cam_idx] is None:
            # camera not exists
            return None
        else:
            return self.cameraList[cam_idx].getSnapshot()

    def takeSnapshot(self) -> "dict[str, np.ndarray]":
        if not self.is_streaming:
            return []

        ret = {}

        for cam_idx, c in enumerate(self.cameraList):
            if c is None:
                ret[cam_idx] = None
            else:
                ret[cam_idx] = c.getSnapshot()

        return ret

    def getCamIndexes(self):
        return self.idx_list

    # [callable]
    def setPreviewSize(self, image_size=QSize(285, 380)):
        pass

    # [callable]
    def setHidden(self, is_hidden:bool, idx_list:list=[]):
        for i in idx_list:
            self.camera_label_list[i].setHidden(is_hidden)
            self.fps_container_list[i].setHidden(is_hidden)

    def showEvent(self, event):
        logging.debug(f"{self.__class__.__name__}: shown.")

    def hideEvent(self, event):
        logging.debug(f"{self.__class__.__name__}: hided.")

    def __setupUI(self):
        self.camera_block_layout = self.__cameraPreview(number=self.num_cameras)
        self.setLayout(self.camera_block_layout)

    def showResolution(self, show=False):
        self.label_show_resolution = show

    def __addSingleCamera(self, container_layout, camera_id:int, row:int, col:int):
        self.snapshot.append(None)
        # camera
        camera_label = QLabel()
        # camera_label.setFixedSize()
        camera_label.setText("Camera_Preview_" + str(camera_id))
        self.camera_label_list.append(camera_label)
        self.camera_winid_list.append(camera_label.winId())

        # FPS
        # add container to make hide and show running correctly
        fps_layout = QHBoxLayout()
        fps_container = QWidget()
        fps_container.resize(700, 50)
        preview_fps_label = QLabel()
        preview_fps_label.setText("")
        preview_fps_label.setStyleSheet(UIStyleSheet.Check3DLabel)
        self.preview_fps_label_list.append(preview_fps_label)

        #record_fps_label = QLabel()
        #record_fps_label.setText("record FPS:0")
        #self.record_fps_label_list.append(record_fps_label)

        fps_layout.addWidget(preview_fps_label)
        #fps_layout.addWidget(record_fps_label)
        fps_container.setLayout(fps_layout)
        self.fps_container_list.append(fps_container)

        container_layout.addWidget(camera_label, row * 2, col, alignment=Qt.AlignBottom | Qt.AlignCenter) # 直、橫
        container_layout.addWidget(fps_container, row * 2 + 1, col, alignment=Qt.AlignCenter | Qt.AlignTop)

        def onTimer():
            label_text = ""
            if self.metric_datas[camera_id] is not None:
                if self.label_show_resolution:
                    label_text = f"({self.cameraList[camera_id].width}x{self.cameraList[camera_id].height}) @ "
                label_text += f"{self.metric_datas[camera_id].fps:.01f} fps"
            else:
                label_text += "0.0 fps"
            preview_fps_label.setText(label_text)

        t = QTimer()
        t.setInterval(500)
        t.timeout.connect(onTimer)
        t.start()
        self.metric_timers.append(t)

    def __cameraPreview(self, number=2):
        container_layout = QGridLayout()

        self.camera_label_list:list[QLabel] = []
        self.fps_container_list:list[QWidget] = []
        self.preview_fps_label_list:list[QLabel] = []
        self.record_fps_label_list:list[QLabel] = []
        self.camera_winid_list:list[int] = []
        self.metric_timers:list[QTimer] = []

        self.__addSingleCamera(container_layout, 1 - 1, 0, 2)
        self.__addSingleCamera(container_layout, 2 - 1, 0, 0)
        self.__addSingleCamera(container_layout, 3 - 1, 1, 0)

        self.__addSingleCamera(container_layout, 4 - 1, 1, 2)
        self.__addSingleCamera(container_layout, 5 - 1, 0, 1)
        self.__addSingleCamera(container_layout, 6 - 1, 1, 1)

        return container_layout

    #deprecated
    def checkAUXorBUX(self, serial:str) -> str:
        pass
    def startStreaming(self, delta=2):
        return
    def stopStreaming(self, record_resolution=None):
        return

    def resync(self):
        t = time.time_ns()

        for cam_idx in self.idx_list:
            self.cameraList[cam_idx].resync(t)
