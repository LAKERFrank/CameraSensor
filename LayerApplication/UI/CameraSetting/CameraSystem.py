import os
import logging
import time
from datetime import datetime
from enum import Enum, auto
import shutil

from PyQt5.QtWidgets import QGroupBox, QVBoxLayout, QComboBox, QWidget, QGridLayout, QLabel, QPushButton, QHBoxLayout, QSpinBox, QScrollArea, QDialog, QDoubleSpinBox, QProgressDialog
from PyQt5.QtCore import QSize, QThread, pyqtSignal, pyqtSlot, QTimer
from PyQt5.QtGui import QPixmap, QImage, QGuiApplication
from PyQt5.Qt import *
from PyQt5.QtMultimedia import QSound

from LayerCamera.RpcCameraWidget import RpcCameraWidget

AUX = "(1440, 1080)"
BUX = "(2048, 1536)"

from ..UISettings import *
from ..Services import SystemService, MsgContract

from lib.nodes import CameraReader
from lib.frame import Frame
from lib.message import *
from lib.common import insertById, loadConfig, saveConfig, setIntrinsicMtx, checkAUXorBUX
from lib.Calibration import Calibration

from lib.common import ROOTDIR, ICONDIR

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

class CameraSystem(QGroupBox):
    def __init__(self, broker_ip, cfg, cameras:'list[CameraReader]', camera_widget:RpcCameraWidget):
        super().__init__()

        self.from_page = 'Home'

        # initalize Service
        self.myService = None
        self.cameras = cameras
        self.broker = broker_ip
        self.cfg = cfg
        self.camera_widget = camera_widget
        self.isRecording = False
        self.record_done_cnt = 0
        self.snapshot_dir_path = os.path.join(ROOTDIR, "snapshot")
        os.makedirs(self.snapshot_dir_path, exist_ok=True)

        self.cur_camera = 0
        self.updateCamInfo()

        self.snapshot = []
        # self.image_size = QSize(480, 360)
        self.image_size = QSize(285, 380)

        # always 4 in NOL_Playground
        self.num_cameras = self.camera_widget.num_cameras
        # self.num_cameras = len(self.cameras)

        self.blockSize = 10


        # for first frame
        self.first_frame_time = [0] * self.num_cameras
        self.first_frame_cnt = 0

        self.NO_IMAGE = QPixmap(f"{ICONDIR}/no_camera.png")

    def updateCamInfo(self):
        cam = self.camera_widget.cameraList[self.cur_camera]

        if cam is not None:
            device_info = cam.getDeviceInfo()
            self.cam_brand = device_info['brand']
            self.cam_serial = device_info['serial']
            self.cam_model = device_info['model']

            self.chessboard_path = f"{ROOTDIR}/Reader/{self.cam_brand}/intrinsic_data/{self.cam_serial}"
            os.makedirs(self.chessboard_path, exist_ok=True)
        else:
            self.chessboard_path = ROOTDIR

    def showEvent(self, event):
        logging.debug(f"{self.__class__.__name__}: shown.")
        self.setupUI()
        self.camera_widget.showResolution(True)
        #self.checkMaxRecord()

        #self.btn_menu.setEnabled(False)
        #self.btn_setting.setEnabled(False)
        #self.btn_choose_camera.setEnabled(False)
        #self.btn_intrinsic.setEnabled(False)
        #self.btn_extrinsic.setEnabled(False)
        #self.btn_check3d.setEnabled(False)

        #self.camera_widget.startStreaming()

        #def enableButton():
        #    self.btn_menu.setEnabled(True)
        #    self.btn_setting.setEnabled(True)
        #    self.btn_choose_camera.setEnabled(True)
        #    self.btn_intrinsic.setEnabled(True)
        #    self.btn_extrinsic.setEnabled(True)
        #    self.btn_check3d.setEnabled(True)
        #    
        #QTimer.singleShot(3000, enableButton)

    def hideEvent(self, event):
        logging.debug(f"{self.__class__.__name__}: hided.")
        self.deleteUI()
        self.camera_widget.showResolution(False)

    def setBackgroundService(self, service:SystemService):
        self.myService = service

    def setFromPage(self, page):
        if page == None:
            return
        # print(f'page = {page}')
        self.from_page = page

    def setupUI(self):
        self.camera_widget.initWidget(self.num_cameras, self.image_size)
        # 取得控制按鈕
        self.control_panel = self.getControlPanelImage()

        # 內參回首頁按鈕
        self.btn_home = QPushButton()
        self.btn_home.setText('返回')#回選單
        self.btn_home.setFixedSize(QSize(180, 50))
        self.btn_home.setStyleSheet(UIStyleSheet.ReturnButton)
        self.btn_home.clicked.connect(self.showHome)

        # 內參區塊
        self.creatIntrinsicGroupBox()

        self.layout_main = QHBoxLayout()
        self.layout_main.addWidget(self.control_panel, 1, alignment=Qt.AlignHCenter)
        self.layout_main.addWidget(self.btn_home, alignment=Qt.AlignTop | Qt.AlignCenter)
        self.layout_main.addWidget(self.camera_widget, 5, alignment=Qt.AlignmentFlag.AlignVCenter)
        self.layout_main.addWidget(self.gbox_camera_intrinsic, alignment=Qt.AlignCenter)

        self.btn_home.hide()
        self.gbox_camera_intrinsic.hide()

        self.setLayout(self.layout_main)

    def deleteUI(self):
        # notice here!!
        self.layout_main.removeWidget(self.camera_widget)
        while self.layout_main.count():
            item = self.layout_main.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()
        self.layout_main.deleteLater()

    def getControlPanel(self):
        container = QWidget()
        container_layout = QVBoxLayout()

        # 回選單按鈕
        self.btn_menu = QPushButton()
        self.btn_menu.setText('回首頁')#返回
        self.btn_menu.setFixedSize(QSize(180, 50))
        # self.btn_menu.setStyleSheet('font: 24px')
        self.btn_menu.setStyleSheet(UIStyleSheet.ReturnButton)
        self.btn_menu.clicked.connect(self.backHome)
        # 設定相機參數按鈕
        self.btn_setting = QPushButton()
        self.btn_setting.setText('相機參數設定')
        self.btn_setting.setFixedSize(QSize(180, 50))
        self.btn_setting.setStyleSheet(UIStyleSheet.SelectButton)
        self.btn_setting.clicked.connect(self.settingCamera)
        # 校正內部參數按鈕
        self.btn_intrinsic = QPushButton()
        self.btn_intrinsic.setText('棋盤格法校正')
        self.btn_intrinsic.setFixedSize(QSize(180, 50))
        self.btn_intrinsic.setStyleSheet(UIStyleSheet.SelectButton)
        self.btn_intrinsic.clicked.connect(self.toggleIntrinsic)
        # 校正外部參數按鈕
        self.btn_extrinsic = QPushButton()
        self.btn_extrinsic.setText('世界座標設定')
        self.btn_extrinsic.setFixedSize(QSize(180, 50))
        self.btn_extrinsic.setStyleSheet(UIStyleSheet.SelectButton)
        self.btn_extrinsic.clicked.connect(self.extrinsic)
        # 錄影按鈕
        self.btn_record = QPushButton()
        self.btn_record.setText('錄影')
        self.btn_record.setFixedSize(QSize(180, 50))
        self.btn_record.clicked.connect(self.toggleRecord)
        self.btn_record.setStyleSheet(UIStyleSheet.SelectButton)
        # 拍照按鈕
        self.btn_snapshot = QPushButton()
        self.btn_snapshot.setText('拍照')
        self.btn_snapshot.setFixedSize(QSize(180, 50))
        self.btn_snapshot.setStyleSheet(UIStyleSheet.SelectButton)
        self.btn_snapshot.clicked.connect(self.toggleSnapshot)
        # 回放按鈕
        self.btn_replay = QPushButton()
        self.btn_replay.setText('查看回放')
        self.btn_replay.setFixedSize(QSize(180, 50))
        self.btn_replay.setStyleSheet(UIStyleSheet.SelectButton)
        self.btn_replay.clicked.connect(self.replay)
        # 3D驗證按鈕
        self.btn_check3d = QPushButton()
        self.btn_check3d.setText('3D座標驗證')
        self.btn_check3d.setFixedSize(QSize(180, 50))
        self.btn_check3d.setStyleSheet(UIStyleSheet.SelectButton)
        self.btn_check3d.clicked.connect(self.check3d)

        self.btn_choose_camera = QPushButton()
        self.btn_choose_camera.setText('相機位置')
        self.btn_choose_camera.setFixedSize(QSize(180, 50))
        self.btn_choose_camera.setStyleSheet(UIStyleSheet.SelectButton)
        self.btn_choose_camera.clicked.connect(self.choosecamera)

        image_label = QLabel()
        pixmap = QPixmap(f"{ICONDIR}/court/std_court_with_num.png")  # Replace with the actual path
        pixmap = pixmap.scaledToHeight(400)
        image_label.setPixmap(pixmap)

        container_layout.addWidget(self.btn_menu, alignment=Qt.AlignmentFlag.AlignCenter)
        container_layout.addWidget(self.btn_setting, alignment=Qt.AlignmentFlag.AlignCenter)
        container_layout.addWidget(self.btn_choose_camera, alignment=Qt.AlignmentFlag.AlignCenter)
        container_layout.addWidget(self.btn_intrinsic, alignment=Qt.AlignmentFlag.AlignCenter)
        container_layout.addWidget(self.btn_extrinsic, alignment=Qt.AlignmentFlag.AlignCenter)
        container_layout.addWidget(self.btn_record, alignment=Qt.AlignmentFlag.AlignCenter)
        # container_layout.addWidget(self.btn_snapshot)
        #container_layout.addWidget(self.btn_replay)
        container_layout.addWidget(self.btn_check3d, alignment=Qt.AlignmentFlag.AlignCenter)
        # container_layout.addWidget(image_label, alignment=Qt.AlignmentFlag.AlignCenter)
        container_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        container.setLayout(container_layout)
        return container
    
    def getControlPanelImage(self):
        container = QWidget()
        container.setFixedWidth(300)
        container_layout = QVBoxLayout()
        self.control_panel = self.getControlPanel()

        image_label = QLabel()
        pixmap = QPixmap(f"{ICONDIR}/court/std_court_with_num.png")  # Replace with the actual path
        pixmap = pixmap.scaledToHeight(400)
        image_label.setPixmap(pixmap)

        container_layout.addWidget(self.control_panel, 1, alignment=Qt.AlignHCenter)
        container_layout.addWidget(image_label, alignment=Qt.AlignmentFlag.AlignCenter)
        container.setLayout(container_layout)
        return container

    def toggleRecord(self):
        msg = MsgContract(MsgContract.ID.PAGE_CHANGE, value='RecordPage')
        self.myService.sendMessage(msg)

    def toggleSnapshot(self):
        self.camera_widget.takeSnapshot()

    def settingCamera(self):
        msg = MsgContract(MsgContract.ID.PAGE_CHANGE, value='CameraSettingPage')
        msg.data = self.camera_widget.idx_list
        self.myService.sendMessage(msg)

    def replay(self):
        msg = MsgContract(MsgContract.ID.PAGE_CHANGE, value='ReplayPage')
        self.myService.sendMessage(msg)

    def extrinsic(self):
        pixmaps = []
        original_sizes = []

        num_camera = self.camera_widget.num_cameras

        for i in range(num_camera):
            image = self.camera_widget.getSnapshotQImage(i)

            if image:
                pixmap = QPixmap(image)
            else:
                pixmap = self.NO_IMAGE

            pixmaps.append(pixmap)
            original_sizes.append(pixmap.size())

        data = dict()
        data['Pixmaps'] = pixmaps
        data['SelectedCameras'] = num_camera 
        data['Original_sizes'] = original_sizes

        msg = MsgContract(MsgContract.ID.PAGE_CHANGE, value='ExtrinsicPage', data=data)
        self.myService.sendMessage(msg)

    def check3d(self):
        pixmaps = []
        original_sizes = []

        num_camera = self.camera_widget.num_cameras

        for i in range(num_camera):
            image = self.camera_widget.getSnapshotQImage(i)

            if image:
                pixmap = QPixmap(image)
            else:
                pixmap = self.NO_IMAGE

            pixmaps.append(pixmap)
            original_sizes.append(pixmap.size())

        data = dict()
        data['Pixmaps'] = pixmaps
        data['SelectedCameras'] = num_camera 
        data['Original_sizes'] = original_sizes

        msg = MsgContract(MsgContract.ID.PAGE_CHANGE, value='Check3dPage', data=data)
        self.myService.sendMessage(msg)

    def choosecamera(self):
        msg = MsgContract(MsgContract.ID.PAGE_CHANGE, value='ChooseCameraPage')
        msg.data = 'CameraSystem'
        self.myService.sendMessage(msg)

    def backHome(self):
        # logging.debug(f'self.from_page ============================ {self.from_page}')
        if self.from_page == None:
            self.from_page = 'Home'
        msg = MsgContract(MsgContract.ID.PAGE_CHANGE, value=self.from_page)
        self.myService.sendMessage(msg)

    def toggleIntrinsic(self):
        self.intrinsic_cb.setCurrentIndex(0)
        self.control_panel.hide()
        self.btn_home.show()
        self.showIntrinsic()

    def onComboBoxChanged(self):
        self.showIntrinsic()

    #def checkMaxRecord(self):
    #    check_max = True
    #    for i in range(self.num_cameras):
    #        config_file = f"{ROOTDIR}/Reader/{self.cam_brand}/config/{self.cam_serial}.cfg"
    #        cameraCfg = loadConfig(config_file)
    #        RecordResolution = cameraCfg['Camera']['RecordResolution']
    #        if RecordResolution != AUX and RecordResolution != BUX:
    #            check_max = False
    #    if check_max:
    #        self.btn_intrinsic.setEnabled(True)
    #    else:
    #        self.btn_intrinsic.setEnabled(False)

    def showIntrinsic(self):
        self.cur_camera = int(self.intrinsic_cb.currentIndex())
        self.updateCamInfo()
        #self.checkMaxRecord()
        self.freshQscrollArea()

        # preview_size = QSize(800, 600)
        preview_size = QSize(480, 640)
        self.camera_widget.setPreviewSize(preview_size)
        self.camera_widget.setHidden(False, [self.cur_camera])
        self.camera_widget.setHidden(True, [i for i in range(self.num_cameras) if i != self.cur_camera])

        self.gbox_camera_intrinsic.show()

    def showHome(self):
        self.camera_widget.setPreviewSize(self.image_size)
        self.camera_widget.setHidden(False, [i for i in range(self.num_cameras)])
        self.gbox_camera_intrinsic.hide()
        self.btn_home.hide()
        self.control_panel.show()

    def creatIntrinsicGroupBox(self):
        self.gbox_camera_intrinsic = QGroupBox()
        self.gbox_camera_intrinsic.setFixedSize(680,880)

        self.timer = QTimer(self)
        self.count = 0
        self.timer.timeout.connect(self.playSound)

        self.corner_x = 9
        self.corner_y = 9
        self.interval = 1
        self.times = 1

        # 選擇要設定哪台相機內部參數
        self.gbox_intrinsic = QGroupBox()
        self.layout_intrinsic = QHBoxLayout()
        self.label_intrinsic = QLabel("選擇相機:")
        self.label_intrinsic.setFont(UIFont.SpinBox)
        self.intrinsic_cb = QComboBox()
        self.intrinsic_cb.setStyleSheet(UIStyleSheet.CameraSystemCameraCombobox)
        self.intrinsic_cb.addItems(["        相機 " + str(i+1) for i in range(self.num_cameras)])
        self.intrinsic_cb.setFixedSize(200,50)
        self.intrinsic_cb.setCurrentIndex(0)

        # disable cameras that are not exists
        for idx, cam in enumerate(self.camera_widget.cameraList):
            if cam is None:
                self.intrinsic_cb.model().item(idx).setEnabled(False)

        self.intrinsic_cb.currentIndexChanged.connect(self.onComboBoxChanged)
        # self.layout_intrinsic.addWidget(self.label_intrinsic)
        self.layout_intrinsic.addWidget(self.intrinsic_cb, alignment=Qt.AlignmentFlag.AlignRight)
        self.gbox_intrinsic.setLayout(self.layout_intrinsic)

        self.gbox_interval = QGroupBox()
        self.layout_interval = QHBoxLayout()
        self.gbox_interval.setFixedWidth(330)
        self.label_interval = QLabel("Interval second")
        # self.label_interval.setFont(UIFont.SpinBox)
        self.label_interval.setAlignment(Qt.AlignCenter)
        self.label_interval.setFixedSize(QSize(210, 50))
        self.label_interval.setStyleSheet(UIStyleSheet.CameraSystemGroup_1)
        self.spin_interval = QSpinBox()
        self.spin_interval.setFont(UIFont.SpinBox)
        self.spin_interval.setStyleSheet(UIStyleSheet.CameraSystemGroup_1)
        self.layout_interval.addWidget(self.label_interval)
        self.layout_interval.addWidget(self.spin_interval)
        self.gbox_interval.setLayout(self.layout_interval)


        self.gbox_times = QGroupBox()
        self.layout_times = QHBoxLayout()
        self.gbox_times.setFixedWidth(330)
        self.label_times = QLabel("Times")
        # self.label_times.setFont(UIFont.SpinBox)
        self.label_times.setAlignment(Qt.AlignCenter)
        self.label_times.setFixedSize(QSize(210, 50))
        self.label_times.setStyleSheet(UIStyleSheet.CameraSystemGroup_1)
        self.spin_times = QSpinBox()
        self.spin_times.setFont(UIFont.SpinBox)
        self.spin_times.setStyleSheet(UIStyleSheet.CameraSystemGroup_1)
        self.layout_times.addWidget(self.label_times)
        self.layout_times.addWidget(self.spin_times)
        self.gbox_times.setLayout(self.layout_times)

        self.gbox_chessboard_size = QGroupBox()
        self.layout_chessboard_size = QHBoxLayout()
        self.label_chessboard_size = QLabel("Chessboard size(N*N):")
        # self.label_chessboard_size.setFont(UIFont.SpinBox)
        self.label_chessboard_size.setAlignment(Qt.AlignCenter)
        self.label_chessboard_size.setFixedSize(QSize(310, 50))
        self.label_chessboard_size.setStyleSheet(UIStyleSheet.CameraSystemGroup_3)
        self.spin_chessboard_size = QSpinBox()
        self.spin_chessboard_size.setFont(UIFont.SpinBox)
        self.spin_chessboard_size.setValue(10)
        self.layout_chessboard_size.addWidget(self.label_chessboard_size)
        self.layout_chessboard_size.addWidget(self.spin_chessboard_size)
        self.gbox_chessboard_size.setLayout(self.layout_chessboard_size)

        self.gbox_block_size = QGroupBox()
        self.layout_block_size = QHBoxLayout()
        self.label_block_size = QLabel("Block size(mm):")
        # self.label_block_size.setFont(UIFont.SpinBox)
        self.label_block_size.setAlignment(Qt.AlignCenter)
        self.label_block_size.setFixedSize(QSize(310, 50))
        self.label_block_size.setStyleSheet(UIStyleSheet.CameraSystemGroup_3)
        self.spin_block_size = QDoubleSpinBox()
        self.spin_block_size.setFont(UIFont.SpinBox)
        self.spin_block_size.setValue(10.0)
        self.spin_block_size.setSingleStep(0.1)
        self.layout_block_size.addWidget(self.label_block_size)
        self.layout_block_size.addWidget(self.spin_block_size)
        self.gbox_block_size.setLayout(self.layout_block_size)

        # qscrollarea
        self.qscrollarea_image_preview = QScrollArea()
        self.qscrollarea_image_preview.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.qscrollarea_image_preview.setWidgetResizable(True)
        self.qscrollarea_image_preview.setMinimumHeight(150)
        self.widget_qscrollarea = QWidget()
        self.layout_qscrollarea = QHBoxLayout()
        self.layout_qscrollarea.setAlignment(Qt.AlignLeft)
        self.widget_qscrollarea.setLayout(self.layout_qscrollarea)
        self.qscrollarea_image_preview.setWidget(self.widget_qscrollarea)

        self.freshQscrollArea()

        self.gbox_take_and_calculate1 = QGroupBox()
        self.gbox_take_and_calculate2 = QGroupBox()
        self.gbox_take_and_calculate3 = QGroupBox()
        self.layout_take_and_calculate_1 = QHBoxLayout()
        self.layout_take_and_calculate_2 = QHBoxLayout()
        self.layout_take_and_calculate_3 = QHBoxLayout()
        self.btn_take = QPushButton("Snapshot Chessboard")
        self.btn_take.setStyleSheet(UIStyleSheet.CameraSystemGroup_1)#
        self.btn_take.setFixedSize(QSize(310, 50))
        self.label_intrinsic = QLabel("")
        # self.label_intrinsic.setFixedSize(560,190)
        self.label_intrinsic.setFixedSize(310,50)
        self.resetLabelIntrinsic()
        #self.btn_record = QPushButton("Video Chessboard")
        #self.btn_record.setStyleSheet(UIStyleSheet.CameraSystemGroup_2)#
        #self.btn_record.setFixedSize(QSize(310, 50))
        self.btn_calculate = QPushButton("Calculate")
        self.btn_calculate.setStyleSheet(UIStyleSheet.CameraSystemGroup_3)
        self.btn_calculate.setFixedSize(QSize(310, 50))
        self.btn_deleteChess = QPushButton("Clear")
        self.btn_deleteChess.setStyleSheet(UIStyleSheet.CameraSystemGroup_3)
        self.btn_deleteChess.setFixedSize(QSize(310, 50))
        self.layout_take_and_calculate_1.addWidget(self.btn_take)
        self.layout_take_and_calculate_1.addWidget(self.label_intrinsic)
        #self.layout_take_and_calculate_2.addWidget(self.btn_record, alignment=Qt.AlignmentFlag.AlignLeft)
        self.layout_take_and_calculate_3.addWidget(self.btn_calculate)
        self.layout_take_and_calculate_3.addWidget(self.btn_deleteChess)
        # wrapper1 = QVBoxLayout()
        # wrapper.addLayout(self.layout_take_and_calculate_1)
        # wrapper.addLayout(self.layout_take_and_calculate_2)
        self.gbox_take_and_calculate1.setLayout(self.layout_take_and_calculate_1)
        self.gbox_take_and_calculate2.setLayout(self.layout_take_and_calculate_2)
        self.gbox_take_and_calculate3.setLayout(self.layout_take_and_calculate_3)


        # Layout
        layout = QGridLayout()
        layout.setAlignment(Qt.AlignTop)
        layout.setContentsMargins(0,0,0,0)

        layout.addWidget(self.gbox_intrinsic,0,0,1,2)
        layout.addWidget(self.gbox_chessboard_size,1,0,1,2)
        layout.addWidget(self.gbox_block_size,2,0,1,2)
        layout.addWidget(self.gbox_interval,3,0)
        layout.addWidget(self.gbox_times,3,1)
        layout.addWidget(self.gbox_take_and_calculate1,4,0,1,2)
        layout.addWidget(self.qscrollarea_image_preview,5,0,4,2)
        layout.addWidget(self.gbox_take_and_calculate2,9,0,1,2)
        layout.addWidget(self.gbox_take_and_calculate3,10,0,1,2)
        # layout.addWidget(self.label_intrinsic,10,0,3,2)
        self.gbox_camera_intrinsic.setLayout(layout)
        #self.gbox_camera_intrinsic.setFont(UIFont.CONTENT)

        # Default Value
        self.spin_interval.setValue(self.interval)
        self.spin_times.setValue(self.times)

        # SIGNAL & SLOT
        self.btn_deleteChess.clicked.connect(self.deleteChessboard)
        self.spin_interval.valueChanged.connect(self.intervalValuechange)
        self.spin_times.valueChanged.connect(self.timesValuechange)
        self.btn_take.clicked.connect(self.takePicture)
        self.btn_calculate.clicked.connect(self.calculateIntrinsic)
        self.spin_chessboard_size.valueChanged.connect(self.cornerValuechange)
        self.spin_block_size.valueChanged.connect(self.blockValuechange)
        #self.btn_record.clicked.connect(self.toggleRecordVideoForIntrinsic)

        #######################
        self.chessboard_idx = 0
        # Chessboard Index Starts From Current Max Index+1
        for filename in os.listdir(self.chessboard_path):
            if filename.startswith('chessboard_') and filename.endswith('.jpg'):
                self.chessboard_idx = max(self.chessboard_idx,int(filename[len('chessboard_'):len('chessboard_')+4])+1)

    class CalibWorker(QThread):
        """Worker Thread for calibration
        """

        finish = pyqtSignal()
        progressLabel = pyqtSignal(str)
        progressValue = pyqtSignal(int)

        camera = None
        video_path = ""

        def __init__(self, camera:CameraReader, video_path, resolution):
            super().__init__()
            self.camera = camera
            self.video_path = video_path
            self.resolution = resolution

        def run(self):

            dst_path = f"{ROOTDIR}/Reader/{self.camera.brand}/intrinsic_data/{self.camera.hw_id}/chessboard_0.mp4"

            # TODO: file is not ready, so wait here
            time.sleep(3)
            shutil.copy(self.video_path, dst_path)

            config_path = f"{ROOTDIR}/Reader/Image_Source/config/{self.camera.hw_id}.cfg"
            folder_path = f"{ROOTDIR}/Reader/Image_Source/intrinsic_data/{self.camera.hw_id}/"
            cameraCfg = loadConfig(config_path)
            corner = int(cameraCfg['Other']['cornerValue']) - 1
            block = int(cameraCfg['Other']['blockValue'])

            
            self.serialnumber = self.camera.hw_id + "-v4l2"
            
            calib = Calibration(self.resolution, folder_path, corner, block)
            calib.setProgressFn(self.progressLabel, self.progressValue)

            # TODO: bind value for progress
            calib.processVideo()
            calib.savePickedImage()
            calib.clearUnusedData()

    def toggleRecordVideoForIntrinsic(self):
        if self.isRecording:
            self.btn_record.setText("Record Chessboard")
            self.length_indicator.stop()

            self.camera_widget.stopRecording([self.cur_camera])
            camera = self.cameras[self.cur_camera]
            video_path = os.path.join(self.replay_dir, f"CameraReader_{self.cur_camera}.mp4")

            self.p = QProgressDialog("", "Cancel", 0, 100, self)
            self.p.setWindowModality(Qt.WindowModal)
            self.p.setAutoClose(False)
            self.p.show()

            modelName = self.camera_widget.checkAUXorBUX(camera.hw_id)
            if modelName == "AUX":
                resolution = eval(AUX)
            elif modelName == "BUX":
                resolution = eval(BUX)

            self.t = self.CalibWorker(camera, video_path, resolution)
            self.t.start()

            self.t.finished.connect(self.p.deleteLater)
            self.t.finished.connect(self.freshQscrollArea)
            self.t.progressLabel.connect(self.p.setLabelText)
            self.t.progressValue.connect(self.p.setValue)

        else:
            def update_text():
                sec = round(datetime.now().timestamp() - self.record_start_timestamp)
                self.btn_record.setText(f"Recording...({sec})")
            self.length_indicator = QTimer(self)
            self.length_indicator.timeout.connect(update_text)
            self.length_indicator.start(500)
            self.record_start_timestamp = datetime.now().timestamp()

            self.replay_dir = self.camera_widget.startRecording([self.cur_camera])

        self.isRecording = not self.isRecording


    def freshQscrollArea(self):
        # init scrollarea picture
        time.sleep(0.2)
        for i in reversed(range(self.layout_qscrollarea.count())):
            self.layout_qscrollarea.itemAt(i).widget().setParent(None)
        for filename in os.listdir(self.chessboard_path):
            if filename.startswith('chessboard_') and filename.endswith('.jpg'): # Only show chessboard
                fullpath = os.path.join(self.chessboard_path, filename)
                pix = QPixmap(fullpath).scaled(150, 150, Qt.KeepAspectRatio)
                object = ClickLabel(fullpath)
                object.setPixmap(pix)
                object.clicked.connect(self.showImage)
                self.layout_qscrollarea.addWidget(object)

    def showImage(self, image_fullpath):
        new_label = QLabel()
        new_label.setPixmap(QPixmap(image_fullpath).scaled(800, 800, Qt.KeepAspectRatio))
        self.bigpicture = QDialog()
        btn_delete_image = QPushButton('delete')
        btn_delete_image.clicked.connect(lambda: self.deleteImage(image_fullpath))
        layout = QVBoxLayout()
        self.bigpicture.setLayout(layout)
        layout.addWidget(btn_delete_image)
        layout.addWidget(new_label)
        self.bigpicture.setWindowTitle('big picture')
        self.bigpicture.setAttribute(Qt.WA_DeleteOnClose)
        self.bigpicture.exec_()

    def deleteImage(self, image_fullpath):
        os.remove(image_fullpath)
        self.bigpicture.close()
        self.freshQscrollArea()

    def resetLabelIntrinsic(self):
        self.label_intrinsic.setFont(UIFont.Combobox)
        self.label_intrinsic.clear()

    def intervalValuechange(self):
        self.interval = self.spin_interval.value()

    def timesValuechange(self):
        self.times = self.spin_times.value()

    def cornerValuechange(self):
        self.corner_x = self.spin_chessboard_size.value() - 1
        self.corner_y = self.spin_chessboard_size.value() - 1

    def blockValuechange(self):
        self.blockSize = self.spin_block_size.value()

    def playSound(self):
        if (self.interval != 0):
            self.count += 1
            if (self.count % self.interval != 0):
                pass
            else:
                filepath = (os.path.join(self.chessboard_path,'chessboard_'+'{:0>4d}.jpg'.format(self.chessboard_idx)))
                pixmap = QPixmap(self.camera_widget.getSnapshotQImage(self.cur_camera))
                pixmap.save(filepath)
                self.label_intrinsic.setText("{}/{}".format(int(self.count/self.interval),self.times))
                self.label_intrinsic.repaint()
                self.chessboard_idx += 1
                self.freshQscrollArea()
            if (self.count >= self.interval * self.times):
                self.freshQscrollArea()
                self.count = 0
                self.timer.stop()

    def deleteChessboard(self):
        self.resetLabelIntrinsic()
        for filename in os.listdir(self.chessboard_path):
            fullpath = os.path.join(self.chessboard_path, filename)
            if filename.startswith('chessboard_') and filename.endswith('.jpg'):
                os.remove(fullpath)
        self.chessboard_idx = 0
        self.freshQscrollArea()

    def hideTakePictureBtn(self):
        self.btn_take.setEnabled(False)
        QTimer.singleShot(self.times * self.interval * 1000, lambda: self.btn_take.setDisabled(False))

    def takePicture(self):
        self.hideTakePictureBtn()
        #self.label_intrinsic.setFont(QFont('Times', 100))
        self.label_intrinsic.setText("0/{}".format(self.times))
        self.label_intrinsic.repaint()
        if self.times != 0:
            if self.interval != 0:
                self.timer.start(1000)
            else:
                for i in range(self.times):
                    filepath = (os.path.join(self.chessboard_path,'chessboard_'+'{:0>4d}.jpg'.format(self.chessboard_idx)))
                    pixmap = QPixmap(self.camera_widget.getSnapshotQImage(self.cur_camera))
                    pixmap.save(filepath)
                    self.label_intrinsic.setText("{}/{}".format(i+1,self.times))
                    self.label_intrinsic.repaint()
                    self.chessboard_idx += 1
                    self.freshQscrollArea()

    def calculateIntrinsic(self):
        cam = self.camera_widget.cameraList[self.cur_camera]

        resolution = cam.getCameraParameters()['RecordResolution']

        device_info = cam.getDeviceInfo()

        brand = device_info['brand']
        serial = device_info['serial']

        self.camera_widget.cameraList[self.cur_camera].getCameraParameters()

        config_file = f"{ROOTDIR}/Reader/{brand}/config/{serial}.cfg"
        image_path = f"{ROOTDIR}/Reader/{brand}/intrinsic_data/{serial}"

        # calculate
        ret = setIntrinsicMtx(config_file, image_path, resolution)

        if type(ret) is dict:
            cam.setIntrinsic(ks=ret['ks'], dist=ret['dist'], newcameramtx=ret['newcameramtx'])

        self.resetLabelIntrinsic()
