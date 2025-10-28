import os
import asyncio
import logging
import json
import cv2
import time
from datetime import datetime

import numpy as np
from PyQt5.QtWidgets import QWidget, QGridLayout, QLabel, QPushButton, QVBoxLayout, QGroupBox
from PyQt5.QtCore import QSize, pyqtSignal
from PyQt5.QtGui import QImage
from pathlib import Path

from LayerCamera.camera.RpcCamera import RpcCamera

from lib.common import get_ip

import gi

from lib.nodes import CameraReader
gi.require_version("Gst", "1.0")
gi.require_version("GstVideo", "1.0")

from gi.repository import Gst
from gi.repository import GObject, Gst, GstVideo   


# DIRNAME = os.path.dirname(os.path.abspath(__file__))
# ICONDIR = f"{DIRNAME}/icon"
AUX = '(1440, 1080)'
BUX = '(2048, 1536)'

from LayerApplication.UI.UISettings import UIStyleSheet
from PyQt5.QtCore import Qt

from lib.common import loadConfig

from lib.common import ROOTDIR
from LayerApplication.utils.CameraExplorer import CameraExplorer

class RpcCameraWidget(QWidget):

    fpsUpdateSignal = pyqtSignal(int, bytes)

    def __init__(self, cameras:list, camera_explorer: CameraExplorer):
        super().__init__()

        # initalize Service
        self.cameras:'list[CameraReader]' = cameras
        self.is_recording = False
        self.is_streaming = False

        self.camera_explorer = camera_explorer

        self.image_size = QSize(285, 380)

        # self.num_cameras = num_cameras
        self.num_cameras = len(self.cameras)
        self.idx_list:'list[int]' = []

        self.label_show_resolution = False

        self.preivew_pipelines = [None for _ in range(self.num_cameras)]
        self.is_preview_playing = False

        # setup UI
        self.__setupUI()

        self.fpsUpdateSignal.connect(self.updateFpsLabel)

        self.kf:'list[tuple[float, float]]' = [(0.0, 0.0) for _ in range(self.num_cameras)]
        self.offsets = [0 for _ in range(self.num_cameras)]

    def _calculateOffsets(self, i):
        ref_timestamp, ref_dt = self.kf[self.idx_list[0]]

        timestamp, dt = self.kf[i]

        if ref_dt > 0:
            frame_diff = round((ref_timestamp - timestamp) / dt)
            aligned_timestamp = timestamp + frame_diff * dt

            offset = aligned_timestamp - ref_timestamp
            self.offsets[i] = offset * 1000 # ms

    def updateFpsLabel(self, index:int, payload:str):
        data = json.loads(payload)
        fps = data["fps"]
        rendered = data["frames_rendered"]

        self.kf[index] = (data['kf_timestamp'], data['kf_dt'])

        # calculate offset related to first camera
        self._calculateOffsets(index)

        if self.label_show_resolution:
            label_text = f"{self.cameraList[index].resolution} @ {fps:.01f} fps [{self.offsets[index]:+.3f} ms]"
        else:
            label_text = f"fps:{fps:.01f}, rendered:{rendered} [{self.offsets[index]:+.3f} ms]"
        self.preview_fps_label_list[index].setText(label_text)

    def initWidget(self, num, image_size=QSize(285, 380), idx_list=[]):
        for i in range(self.num_cameras):
            self.camera_layout_list[i].setHidden(False)

    def setupCameraMetrics(self):
        for idx in self.idx_list:
            name = self.cameraList[idx].device_name
            topic = f"/DATA/{name}/CameraLayer/Metrics"

            self.camera_explorer.mqttc.message_callback_add(topic, lambda a, b, msg, idx=idx: self.fpsUpdateSignal.emit(idx, msg.payload))
            self.camera_explorer.mqttc.subscribe(topic)

    def removeCameraMetrics(self):
        for idx in self.idx_list:
            name = self.cameraList[idx].device_name
            self.camera_explorer.mqttc.message_callback_remove(f"/DATA/{name}/CameraLayer/Metrics")

    async def _startCamera(self, num):
        self.cameraList:list[RpcCamera] = [None for _ in range(num)]
        for idx in self.idx_list:
            self.cameraList[idx] = RpcCamera(self.cameras[idx].camerasensor_hostname, self.camera_explorer.mqttc)

        tasks = [self.cameraList[idx].async_init() for idx in self.idx_list]
        await asyncio.gather(*tasks)

        t = time.time_ns()
        tasks = [self.cameraList[idx].async_start(t) for idx in self.idx_list]
        await asyncio.gather(*tasks)

        for idx in self.idx_list:
            if self.cameraList[idx].direction in [1, 3]:
                self.camera_label_list[idx].setFixedSize(self.image_size_v)

    def init(self, num, image_size=QSize(285, 380), idx_list=[]):
        """Start camera

        Args:
            num (int): number of cameras
            image_size (QSize, optional): size of preview
            idx_list (list, optional): index list of camera
        """

        self.image_size_h = QSize(400, 300)
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
        #initAllCameraFPS
        for i in range(len(self.preview_fps_label_list)):
            self.preview_fps_label_list[i].setText("FPS:0")

        rpc_camera_names = self.camera_explorer.getAvailableDeviceName()

        tmp_list = []
        for i in self.idx_list:
            if self.cameras[i].camerasensor_hostname == 'None':
                continue
            elif self.cameras[i].camerasensor_hostname not in rpc_camera_names:
                logging.warning(f"Camera {self.cameras[i].camerasensor_hostname} is offline")
                continue
            tmp_list.append(i)
        self.idx_list = tmp_list

        # winids = []
        # for i in range(len(self.camera_label_list)):
        #     winids.append()
        for i in range(len(self.camera_label_list)):
            self.camera_label_list[i].setFixedSize(self.image_size_h)

        self.visualize_tracknet = [[] for _ in range(num)]

        asyncio.run(self._startCamera(num))

        self.is_streaming = True

        self.setupCameraMetrics()

        self.__startDisplay()

    def draw_overlay(self, overlay, context, timestamp, duration, udata):
        cam_idx = udata

        if len(self.visualize_tracknet[cam_idx]) > 0:
            # Set color for the point (red)
            context.set_source_rgb(1.0, 0.0, 0.0)
    
            # Draw a point (small circle) at coordinates (x, y)
            for x, y in self.visualize_tracknet[cam_idx]:
                context.arc(x, y, 5, 0, 2 * 3.14159)  # Circle with radius 5
            context.fill()

    def __createDisplayPipeline(self, idx):

        if self.preivew_pipelines[idx] is not None:
            return

        Gst.init()  # init gstreamer

        pipeline = Gst.parse_launch("udpsrc name=bin"
            " ! capsfilter name=filter "
            " ! rtph264depay ! h264parse ! decodebin ! videoconvert "
            " ! cairooverlay name=overlay"
            " ! glimagesink name=sink sync=false ")

        # retrieve the bin element from the pipeline
        udpsrc = pipeline.get_by_name("bin")
        udpsrc.set_property("port", 9000+idx)

        caps = Gst.Caps.new_empty()

        structure = Gst.Structure.new_from_string("application/x-rtp")

        structure.set_value("media", "video")
        structure.set_value("clock-rate", 90000)
        structure.set_value("encoding-name", "H264")
        structure.set_value("payload", 96)

        caps.append_structure(structure)
        structure.free()

        filter = pipeline.get_by_name("filter")
        filter.set_property("caps", caps)


        sink = pipeline.get_by_name("sink")
        sink.set_window_handle(self.camera_winid_list[idx]) # pass the handle to the sink

        # Get the cairooverlay element
        overlay = pipeline.get_by_name("overlay")
        overlay.connect("draw", self.draw_overlay, idx)

        #pipeline.set_state(Gst.State.PLAYING)

        self.preivew_pipelines[idx] = pipeline

    def updateTrackNetPoint(self, cam_idx:int, coords:'list[(int, int)]'=[]):

        width, height = self.cameraList[cam_idx].resolution

        PREVIEW_WIDTH, PREVIEW_HEIGHT = 320, 240

        # rescale
        coords = [(int(x/width*PREVIEW_WIDTH), int(y/height*PREVIEW_HEIGHT)) for x, y in coords]

        # rotation
        if self.cameraList[cam_idx].direction == 1:
            coords = [(PREVIEW_WIDTH-y, x) for x, y in coords]
        elif self.cameraList[cam_idx].direction == 2:
            coords = [(PREVIEW_WIDTH-x, PREVIEW_HEIGHT-y) for x, y in coords]
        elif self.cameraList[cam_idx].direction == 3:
            coords = [(y, PREVIEW_HEIGHT-x) for x, y in coords]

        self.visualize_tracknet[cam_idx] = coords

    def __startDisplay(self):
        if self.is_preview_playing:
            return

        ip = get_ip()

        async def start(idx:int):
            await self.cameraList[idx].async_startUdp(ip, 9000+idx)
            self.preivew_pipelines[idx].set_state(Gst.State.PLAYING)

        async def start_all():
            tasks = []
            for idx in self.idx_list:
                if self.cameraList[idx] is not None:
                    tasks.append(start(idx))
            await asyncio.gather(*tasks)

        asyncio.run(start_all())

        self.is_preview_playing = True

    def __stopDisplay(self):
        if not self.is_preview_playing:
            return

        async def stop(idx:int):
            await self.cameraList[idx].async_stopUdp()
            self.preivew_pipelines[idx].set_state(Gst.State.PAUSED)

        async def stop_all():
            tasks = []
            for idx in self.idx_list:
                if self.cameraList[idx] is not None:
                    tasks.append(stop(idx))
            await asyncio.gather(*tasks)

        asyncio.run(stop_all())

        self.is_preview_playing = False

    def stop(self):
        if not self.is_streaming:
            return

        self.is_streaming = False

        async def do():
            tasks = []
            for idx in self.idx_list:
                camera = self.cameraList[idx]
                tasks.append(camera.async_release())
                self.cameraList[idx] = None
            await asyncio.gather(*tasks)

        asyncio.run(do())

    def start(self):
        if self.is_streaming:
            return

        self.is_streaming = True

        self.init(len(self.cameraList))

    def resync(self):
        t = time.time_ns()

        async def do():
            tasks = []
            # Setup camera recording
            for cam_idx in self.idx_list:
                tasks.append(self.cameraList[cam_idx].async_resync(t))
            await asyncio.gather(*tasks)

        asyncio.run(do())

    def startRecording(self, mode):
        if self.is_recording:
            return

        self.is_recording = True

        # local
        dirname = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        async def do():
            tasks = []
            # Setup camera recording
            for cam_idx in self.idx_list:
                tasks.append(
                    self.cameraList[cam_idx].async_startRecording(
                        image_buf=False, save_dirname=dirname,
                        cam_idx=cam_idx, mode=mode)
                )
            await asyncio.gather(*tasks)

        asyncio.run(do())

        return Path(ROOTDIR) / "replay" / dirname

    def stopRecording(self):
        if not self.is_recording:
            return

        self.is_recording = False

        async def do():
            tasks = []
            # Setup camera recording
            for cam_idx in self.idx_list:
                tasks.append(
                    self.cameraList[cam_idx].async_stopRecording()
                )
            await asyncio.gather(*tasks)

        asyncio.run(do())

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

            print(f"RpcCameraWidget.getSnapshotArray({cam_idx})")

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
        self.image_size = image_size

        for i in range(len(self.camera_label_list)):
            self.camera_label_list[i].setFixedSize(image_size)

    # [callable]
    def setHidden(self, is_hidden:bool, idx_list:list=[]):
        for i in idx_list:
            self.camera_layout_list[i].setHidden(is_hidden)

    def showEvent(self, event):
        logging.debug(f"{self.__class__.__name__}: shown.")
        #self.__startDisplay()

    def hideEvent(self, event):
        logging.debug(f"{self.__class__.__name__}: hided.")
        #self.__stopDisplay()

    def __setupUI(self):
        self.camera_block_layout = self.__cameraPreview(number=self.num_cameras)
        self.setLayout(self.camera_block_layout)

    def showResolution(self, show=False):
        self.label_show_resolution = show

    def __addSingleCamera(self, container_layout:QGridLayout, camera_id:int, row:int, col:int):

        widget = QGroupBox()
        widget.setTitle(f"Camera_{camera_id}")

        layout = QVBoxLayout()
        widget.setLayout(layout)

        # camera
        camera_label = QLabel()
        # camera_label.setFixedSize()
        camera_label.setText("Camera_Preview_" + str(camera_id))
        self.camera_label_list.append(camera_label)
        self.camera_winid_list.append(camera_label.winId())

        # FPS
        # add container to make hide and show running correctly
        preview_fps_label = QLabel()
        #preview_fps_label.setStyleSheet(UIStyleSheet.Check3DLabel)
        self.preview_fps_label_list.append(preview_fps_label)

        layout.addWidget(camera_label, alignment=Qt.AlignCenter)
        layout.addWidget(preview_fps_label, alignment=Qt.AlignCenter)

        self.camera_layout_list.append(widget)

        container_layout.addWidget(widget, row, col, alignment=Qt.AlignBottom | Qt.AlignCenter) # 直、橫

    def __cameraPreview(self, number=2):
        container_layout = QGridLayout()

        self.camera_label_list:'list[QLabel]' = []
        self.camera_winid_list:'list[int]' = []
        self.preview_fps_label_list:'list[QLabel]' = []
        self.camera_layout_list:'list[QWidget]' = []

        for i in range(self.num_cameras):
            self.__addSingleCamera(container_layout, i, int(i/4), int(i%4))

        for idx in range(self.num_cameras):
            self.__createDisplayPipeline(idx)

        button = QPushButton("Preview Playing")

        def on_button_click():
            if self.is_preview_playing:
                self.__stopDisplay()
                button.setText("Preview Stop")
            else:
                self.__startDisplay()
                button.setText("Preview Playing")

        button.clicked.connect(on_button_click)  # Connect click to function

        container_layout.addWidget(button, 10, 0, alignment=Qt.AlignLeft)

        return container_layout

    #deprecated
    def checkAUXorBUX(self, serial:str) -> str:
        pass
    def startStreaming(self, delta=2):
        return
    def stopStreaming(self, record_resolution=None):
        return
