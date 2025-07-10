import logging
import json
import cv2
import numpy as np

from PyQt5.QtWidgets import QGroupBox, QVBoxLayout, QComboBox, QWidget, QGridLayout, QLabel, QPushButton, QHBoxLayout, QSpinBox, QScrollArea, QDialog, QDoubleSpinBox, QProgressDialog
from PyQt5.QtCore import QSize, QThread, pyqtSignal, pyqtSlot, QTimer
from PyQt5.QtGui import QPixmap, QImage, QGuiApplication
from PyQt5.Qt import *
from PyQt5.QtMultimedia import QSound


from ..UISettings import *
from ..Services import SystemService, MsgContract

from lib.nodes import CameraReader
from lib.frame import Frame
from lib.message import *
from lib.common import insertById, is_local_camera, loadConfig, saveConfig, setIntrinsicMtx, checkAUXorBUX
from lib.Calibration import Calibration
from lib.camera import convert_sample_to_numpy

from lib.common import ROOTDIR, ICONDIR
from LayerApplication.utils.CameraExplorer import CameraExplorer
from LayerCamera.RpcCameraWidget import RpcCameraWidget

class ChooseCameraPage(QGroupBox):
    def __init__(self, cameras:'list[CameraReader]', camera_widget: RpcCameraWidget, camera_explorer: CameraExplorer):
        super().__init__()

        # self.setWindowTitle("選擇相機")
        # self.resize(800, 600)

        self.cameras = cameras
        self.camera_widget = camera_widget
        self.camera_explorer = camera_explorer

        self.from_page = 'CameraSystem'

        self.cfg_file = f"{ROOTDIR}/config"
        self.cfg = loadConfig(self.cfg_file)

        self.is_local = is_local_camera()

        # setupLogLevel(level=self.cfg['Project']['logging_level'], log_file="UI")
        if self.is_local:
            from LayerCamera.camera.Camera import Camera
            self.serials = [item["serial"] for item in Camera.getAvailableCameras()]
        else:
            camera_explorer.explore()
            self.serials = camera_explorer.getAvailableDeviceName()
        # num_exist_cam = len(self.serials)
        # for i in range(num_exist_cam, 4):
        #     self.serials.append('相機少於4')
        # # print(self.serials)
        self.serials.append('None')

        self.setupUI()
        #self.__updateImage()

    def hideEvent(self, event):
        logging.debug(f"{self.__class__.__name__}: hided.")
        # self.camera_widget.stopStreaming()
        # self.deleteUI()

    def showEvent(self, event):
        logging.debug(f"{self.__class__.__name__}: shown.")

        #self.ok_button.setEnabled(False)

        self.camera_explorer.explore()
        self.__updateImage()

        #QTimer.singleShot(3000, lambda: self.ok_button.setEnabled(True))

    def __updateImage(self):
        for i, camera in enumerate(self.cameras):
            if self.combo_boxes[i].currentText() == 'None':
                self.cams[i].setPixmap(self.no_cam_pixmap)
                continue

            image = self.camera_widget.getSnapshotArray(i)
            if image is None:
                self.cams[i].setPixmap(self.no_cam_pixmap)
                continue

            #undistort
            intrinsic = self.camera_widget.cameraList[i].getIntrinsic()
            direction = self.camera_widget.cameraList[i].direction
            if direction == 1:
                image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
            elif direction == 2:
                image = cv2.rotate(image, cv2.ROTATE_180)
            elif direction == 3:
                image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
            image = self.undistort(image, intrinsic)
            if direction == 1:
                image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
            elif direction == 2:
                image = cv2.rotate(image, cv2.ROTATE_180)
            elif direction == 3:
                image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)

            height, width, channel = image.shape
            bytesPerLine = 3*width
            qimage = QImage(image.data, width, height, bytesPerLine, QImage.Format.Format_BGR888)
            self.cams[i].setPixmap(QPixmap(qimage).scaled(240, 320))

    def setBackgroundService(self, service:SystemService):
        self.myService = service

    def setFromPage(self, page:str):
        self.from_page = page

    def undistort(self, image, intrinsic):
        # print(image.shape)
        mtx = np.array(intrinsic['ks'], np.float32)
        dist = np.array(intrinsic['dist'], np.float32)
        # newcameramtx = np.array(cfg['Other']['newcameramtx'])
        # print(mtx)
        # print(dist)
        # print(newcameramtx)
        # cameraCfg['Other']['ks'] = str(mtx.tolist())
        # cameraCfg['Other']['dist'] = str(dist.tolist())
        # cameraCfg['Other']['newcameramtx'] = str(newcameramtx.tolist())
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (image.shape[1], image.shape[0]), 0, (image.shape[1], image.shape[0]))
        # img = cv2.imread(images[i])
        # # undistort
        dst = cv2.undistort(image, mtx, dist, None, newcameramtx)

        # # crop the image
        x, y, w, h = roi
        dst = dst[y:y+h, x:x+w]

        img = cv2.resize(dst, (image.shape[1], image.shape[0]))
        return img

    def setupUI(self):
        layout_main = QVBoxLayout()

        self.ok_button = QPushButton("確定")
        self.ok_button.setFont(QFont('Arial', 20))
        self.ok_button.setStyleSheet(UIStyleSheet.SelectButton)
        self.ok_button.setFixedSize(QSize(180, 50))
        self.ok_button.clicked.connect(self.backhome)

        self.btn_setting = QPushButton("相機設定")
        self.btn_setting.setFont(QFont('Arial', 20))
        self.btn_setting.setFixedSize(QSize(180, 50))
        self.btn_setting.clicked.connect(self.setting)

        self.warning_text = QLabel()
        self.warning_text.setText(' ')
        # layout.addWidget(self.warning_text)

        option = self.getOptions()
        layout_main.addWidget(option)
        layout_main.addWidget(self.warning_text, alignment=Qt.AlignmentFlag.AlignCenter)
        # layout_main.addWidget(self.ok_button, alignment=Qt.AlignmentFlag.AlignCenter)

        self.setLayout(layout_main)

    def getOptions(self):
        self.combo_boxes = [0 for _ in range(len(self.cameras))]
        option_layout = QHBoxLayout()

        image_label = QLabel()
        pixmap = QPixmap(f"{ICONDIR}/court/std_court_with_num.png")  # Replace with the actual path
        pixmap = pixmap.scaledToHeight(500)
        image_label.setPixmap(pixmap)
        # option_layout.addWidget(image_label)

        container_control = QWidget()
        layout_control = QVBoxLayout()
        layout_control.addWidget(self.ok_button)
        # layout_control.addWidget(self.btn_setting)
        layout_control.addWidget(image_label)
        container_control.setLayout(layout_control)

        container_left = QWidget()
        layout_left = QGridLayout()
        #container_mid = QWidget()
        #layout_mid = QGridLayout()
        #container_right = QWidget()
        #layout_right = QGridLayout()

        no_cam_pic_label = QLabel()
        # self.no_cam_pixmap = QPixmap(f"{ICONDIR}/no_camera.png").scaled(480, 360)
        self.no_cam_pixmap = QPixmap(f"{ICONDIR}/no_camera.png").scaled(240, 320)
        no_cam_pic_label.setPixmap(self.no_cam_pixmap)
        self.cams = []
        for i in range(len(self.cameras)):
            cam_i = QLabel()
            cam_i.setPixmap(self.no_cam_pixmap)
            self.cams.append(cam_i)

            container = QGroupBox(f"Camera_{i}")
            layout = QVBoxLayout()
            container.setLayout(layout)

            layout.addWidget(self.getOptionWidget(i), alignment=Qt.AlignCenter)
            layout.addWidget(cam_i, alignment=Qt.AlignCenter)

            layout_left.addWidget(container, int(i/4), int(i%4), alignment=Qt.AlignHCenter)

        container_left.setLayout(layout_left)

        # option_widget = self.getOptionWidget()
        # option_layout.addWidget(option_widget)
        option_layout.addWidget(container_control)
        # option_layout.addWidget(image_label)
        option_layout.addWidget(container_left)
        #option_layout.addWidget(container_mid)
        #option_layout.addWidget(container_right)

        container_option = QWidget()
        container_option.setLayout(option_layout)
        return container_option


    def getOptionWidget(self, index):
        box_layout = QVBoxLayout()
        font = QFont()

        text_label = QLabel()
        text_label.setText(f'Camera {index + 1}:')
        font.setPixelSize(24)
        text_label.setFont(font)

        combo_box = QComboBox()
        combo_box.addItems(self.serials)
        font.setPixelSize(20)
        combo_box.setFont(font)
        combo_box.setFixedSize(285,35)
        combo_box.setStyleSheet(UIStyleSheet.CameraSettingCameraCombobox)
        # print(f'index = {index}, text = {self.cfg[f"CameraReader_{index}"]["hw_id"]}')
        
        current_id = self.cfg[f'CameraReader_{index}']['hw_id'] if self.is_local else self.cfg[f'CameraReader_{index}']['camerasensor_hostname']

        if current_id in self.serials:
            combo_box.setCurrentText(current_id)
        else:
            combo_box.setCurrentText("None")
        combo_box.currentIndexChanged.connect(lambda: self.comboboxChanged(index))
        self.combo_boxes[index] = combo_box

        # box_layout.addWidget(text_label, alignment=Qt.AlignmentFlag.AlignBottom)
        box_layout.addWidget(combo_box, alignment=Qt.AlignmentFlag.AlignTop)

        little_box = QWidget()
        little_box.setLayout(box_layout)
        return little_box

    def isDupSerials(self):
        st = set()
        for i in range(len(self.cameras)):
            if self.combo_boxes[i].currentText() == 'None':
                continue
            if self.combo_boxes[i].currentText() in st:
                return True
            st.add(self.combo_boxes[i].currentText())
        return False

    def comboboxChanged(self, index):
        print(f'change {index}')
        if self.isDupSerials():
            self.warning_text.setText("<font color=red size=20> 相機編號重複!</font>")
            return

        for i in range(len(self.cameras)):
            if self.combo_boxes[i] == 0:
                return

        # TODO: remove duplicate

        self.__updateConfig()

        self.camera_widget.stop()        
        self.camera_widget.start()

        QTimer.singleShot(1000, self.__updateImage)
        #self.__updateImage()

        if self.isDupSerials():
            self.warning_text.setText("<font color=red size=20> 相機編號重複!</font>")
            return

        self.warning_text.setText('')

    def __updateConfig(self):
        if self.is_local:
            for i in range(len(self.cameras)):
                self.cfg[f'CameraReader_{i}']['hw_id'] = self.combo_boxes[i].currentText()
                self.cameras[i].hw_id = self.combo_boxes[i].currentText()
        else:
            for i in range(len(self.cameras)):
                self.cfg[f'CameraReader_{i}']['camerasensor_hostname'] = self.combo_boxes[i].currentText()
                self.cameras[i].camerasensor_hostname = self.combo_boxes[i].currentText()

        # print(self.cfg)
        saveConfig(self.cfg_file, self.cfg)

        logging.info('config file saved')

    def backhome(self):
        """回首頁
        """
        msg = MsgContract(MsgContract.ID.PAGE_CHANGE, value='CameraSystem')
        self.myService.sendMessage(msg)

    def setting(self):
        """到相機設定頁面
        """
        msg = MsgContract(MsgContract.ID.PAGE_CHANGE, value='CameraSystem')
        msg.data = self.from_page
        self.myService.sendMessage(msg)
