import logging
import ast
import multiprocessing as mp

from PyQt5.QtWidgets import QGroupBox, QWidget, QHBoxLayout, QVBoxLayout, QPushButton, QComboBox, QLabel, QSpinBox, QCheckBox
from PyQt5.QtCore import QSize, QTimer
from PyQt5 import QtWidgets
from ast import literal_eval as make_tuple

from ..UISettings import *
from ..Services import SystemService, MsgContract

from LayerCamera.RpcCameraWidget import RpcCameraWidget

class CameraSettingPage(QGroupBox):
    def __init__(self, cameras:list, camera_widget:RpcCameraWidget):
        super().__init__()

        self.cameras = cameras

        self.num_cameras = len(cameras)
        # self.image_size = QSize(640, 480)
        self.image_size = QSize(480, 640)

        self.camera_widget = camera_widget

        # self.selected_num_cameras = 0
        self.idx_list = []

        self.capture_formats = []
        self.items_fps = []
        self.items_resolution = []
        #get modelName(AUX or BUX)
        # self.modelName = checkAUXorBUX(self.serialnumber)
        # if(self.modelName == "AUX"):
        #     self.items_fps = ['30', '60', '120']
        #     self.items_resolution = ['(800, 600)', '(1024, 768)', '(1280, 720)', '(1440, 1080)']
        # elif(self.modelName == "BUX"):
        #     self.items_fps = ['30', '60', '119']
        #     self.items_resolution = ['(640, 480)', '(1024, 768)', '(2048, 1536)']            

    def hideEvent(self, event):
        logging.debug(f"{self.__class__.__name__}: hided.")
        # self.camera_widget.stopStreaming()
        self.deleteUI()

    def showEvent(self, event):
        logging.debug(f"{self.__class__.__name__}: shown.")
        self.setupUI()

        # load default settings from config file(camera 0)
        self.init_combo_flag = True
        self.camera_cb.clear()
        # items = [f'        相機 {i + 1}' for i in range(self.selected_num_cameras)]
        items = [f'        相機 {i + 1}' for i in self.idx_list]
        # self.camera_cb.setStyleSheet("font: 24px; border-radius : 15px; background-color : gray; border : 2px solid; border-color : black")
        self.camera_cb.addItems(items)
        self.camera_cb.setCurrentIndex(0)

        self.idx_list = self.camera_widget.idx_list

        if self.idx_list:
            self.onCameraChanged()
            self.setLayoutValue()

    def setBackgroundService(self, service:SystemService):
        self.myService = service

    def setupUI(self):
        # 控制按鈕
        self.control_bar = self.getControlBar()

        # 設定區域
        self.setting_area = self.getSettingArea()

        self.layout_main = QHBoxLayout()
        self.layout_main.addWidget(self.control_bar)
        self.layout_main.addWidget(self.camera_widget, alignment=Qt.AlignmentFlag.AlignVCenter)
        self.layout_main.addWidget(self.setting_area)

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

    def getControlBar(self):
        container = QWidget()
        container.setFixedWidth(300)
        container_layout = QVBoxLayout()

        # # 選擇要設定哪台相機的參數
        # self.camera_cb = QComboBox()
        # self.camera_cb.currentIndexChanged.connect(self.onComboBoxChanged)
        # self.camera_cb.setStyleSheet('font: 24px')
        # self.camera_cb.setFixedSize(180,50)

        # Home
        self.btn_home = QPushButton()
        self.btn_home.setText('返回')#回首頁
        self.btn_home.setFixedSize(QSize(180, 50))
        self.btn_home.setStyleSheet(UIStyleSheet.ReturnButton)
        self.btn_home.clicked.connect(self.backhome)

        # Apply
        self.btn_apply = QPushButton()
        self.btn_apply.setText('存檔')
        self.btn_apply.setFixedSize(QSize(180, 50))
        self.btn_apply.setStyleSheet(UIStyleSheet.SelectButton)
        self.btn_apply.clicked.connect(self.applySetting)

        # Apply resolution and fps to all cameras
        # self.btn_apply_res_fps = QPushButton()
        # self.btn_apply_res_fps.setText('套用至所有相機\n(解析度與FPS)')
        # self.btn_apply_res_fps.setFixedSize(QSize(200, 100))
        # self.btn_apply_res_fps.setStyleSheet('font: 24px')
        # self.btn_apply_res_fps.clicked.connect(self.applyAllResolutionFPS)

        # Reset
        self.btn_reset = QPushButton()
        self.btn_reset.setText('重置')
        self.btn_reset.setFixedSize(QSize(180, 50))
        self.btn_reset.setStyleSheet(UIStyleSheet.SelectButton)
        self.btn_reset.clicked.connect(self.resetSetting)

        self.btn_choose_camera = QPushButton()
        self.btn_choose_camera.setText('選擇相機')
        self.btn_choose_camera.setFixedSize(QSize(200, 60))
        self.btn_choose_camera.setStyleSheet(UIStyleSheet.SelectButton)
        self.btn_choose_camera.clicked.connect(self.choosecamera)

        container_layout.addWidget(self.btn_home, alignment=Qt.AlignCenter)
        # container_layout.addWidget(self.camera_cb, alignment=Qt.AlignCenter)
        container_layout.addWidget(self.btn_apply, alignment=Qt.AlignCenter)
        # container_layout.addWidget(self.btn_apply_res_fps)
        container_layout.addWidget(self.btn_reset, alignment=Qt.AlignCenter)
        # container_layout.addWidget(self.btn_choose_camera, alignment=Qt.AlignCenter)
        container_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        container.setLayout(container_layout)
        return container

    def getSettingArea(self):
        spin_max_value = 99999999
        spin_min_value = -99999999


        container = QWidget()
        container_layout = QVBoxLayout()
        container.setFixedWidth(800)

        # 選擇要設定哪台相機的參數
        self.camera_cb = QComboBox()
        self.camera_cb.currentIndexChanged.connect(self.onCameraChanged)
        # self.camera_cb.setStyleSheet('font: 24px')
        self.camera_cb.setFixedSize(200,50)
        # self.camera_cb.setFont(UIFont.SpinBox)
        self.camera_cb.setStyleSheet(UIStyleSheet.CameraSettingCameraCombobox)

        #initial button
        # self.btn_initial = QPushButton()
        # self.btn_initial.setText('Reset')
        # self.btn_initial.setFixedSize(QSize(180, 50))
        # self.btn_initial.setStyleSheet(UIStyleSheet.SelectButton)

        #Local variable Title
        self.label_local_title = QLabel("相機參數設定")
        self.label_local_title.setStyleSheet(UIStyleSheet.CameraStettingLocalTitle)
        self.label_local_title.setFixedSize(QSize(200, 50))
        self.label_local_title.setAlignment(Qt.AlignCenter)
        self.layout_local_title = QHBoxLayout()
        # self.layout_local_title.addStretch()
        self.layout_local_title.addWidget(self.label_local_title, alignment=Qt.AlignmentFlag.AlignLeft)
        self.layout_local_title.addWidget(self.camera_cb, alignment=Qt.AlignmentFlag.AlignRight)
        # self.layout_local_title.addWidget(self.btn_initial, alignment=Qt.AlignmentFlag.AlignRight)

        # Brightness
        self.label_brightness_value = QtWidgets.QLabel(None)
        self.label_brightness_value.setFixedSize(QSize(80, 40))
        self.label_brightness_value.setStyleSheet(UIStyleSheet.SliderLabel)
        self.label_brightness_value.setAlignment(Qt.AlignVCenter)
        def show_brightness():
            self.label_brightness_value.setText(str(self.slider_brightness.value()))

        self.label_brightness = QLabel("Brightness:")
        # self.label_brightness.setFont(UIFont.SpinBox)
        self.label_brightness.setStyleSheet(UIStyleSheet.CameraStettingLocalVariabl)
        # self.spin_brightness = QSpinBox()
        # self.spin_brightness.setFont(UIFont.SpinBox)
        # self.spin_brightness.setMinimum(spin_min_value)
        # self.spin_brightness.setMaximum(spin_max_value)
        # self.spin_brightness.setSingleStep(10)
        self.slider_brightness = QtWidgets.QSlider()
        self.slider_brightness.setRange(0, 200)   
        self.slider_brightness.setOrientation(1)   
        self.slider_brightness.setFixedSize(QSize(465, 20)) 
        self.slider_brightness.setStyleSheet(UIStyleSheet.Slider)
        self.slider_brightness.valueChanged.connect(show_brightness)
        self.layout_brightness = QHBoxLayout()
        self.layout_brightness.addWidget(self.label_brightness)
        # self.layout_brightness.addWidget(self.spin_brightness)
        self.layout_brightness.addWidget(self.slider_brightness)
        self.layout_brightness.addWidget(self.label_brightness_value)

        # slider = QtWidgets.QSlider()
        # # slider.setGeometry(20,40,100,30)
        # # slider.setStyleSheet("")
        # slider.setValue(50)
        # slider.setRange(0, 100)
        # slider.setOrientation(1)
        # slider.valueChanged.connect(show)

        # ExposureAuto
        # self.label_exposure_auto = QLabel("ExposureAuto:")
        # self.label_exposure_auto.setFont(UIFont.SpinBox)
        # self.chkbox_exposure_auto = QCheckBox()
        # self.chkbox_exposure_auto.clicked.connect(self.onExposureAutoClick)
        # self.layout_exposure_auto = QHBoxLayout()
        # self.layout_exposure_auto.addWidget(self.label_exposure_auto)
        # self.layout_exposure_auto.addWidget(self.chkbox_exposure_auto)

        # ExposureTimeAbs
        self.label_exposure_time_abs_value = QtWidgets.QLabel(None)
        self.label_exposure_time_abs_value.setFixedSize(QSize(80, 40))
        self.label_exposure_time_abs_value.setStyleSheet(UIStyleSheet.SliderLabel)
        self.label_exposure_time_abs_value.setAlignment(Qt.AlignVCenter)
        def show_exposure_time_abs():
            self.label_exposure_time_abs_value.setText(str(self.slider_exposure_time_abs.value()/1000))

        self.label_exposure_time_abs = QLabel("ExposureTime (ms):")
        # self.label_exposure_time_abs.setFont(UIFont.SpinBox)
        self.label_exposure_time_abs.setStyleSheet(UIStyleSheet.CameraStettingLocalVariabl)
        # self.spin_exposure_time_abs = QSpinBox()
        # self.spin_exposure_time_abs.setFont(UIFont.SpinBox)
        # self.spin_exposure_time_abs.setMinimum(spin_min_value)
        # self.spin_exposure_time_abs.setMaximum(spin_max_value)
        self.slider_exposure_time_abs = QtWidgets.QSlider()
        self.slider_exposure_time_abs.setRange(1, 10000) # 0.001 ms ~ 10 ms
        self.slider_exposure_time_abs.setOrientation(1)   
        self.slider_exposure_time_abs.setFixedSize(QSize(465, 20)) 
        self.slider_exposure_time_abs.setStyleSheet(UIStyleSheet.Slider)
        self.slider_exposure_time_abs.valueChanged.connect(show_exposure_time_abs)
        self.layout_exposure_time_abs = QHBoxLayout()
        self.layout_exposure_time_abs.addWidget(self.label_exposure_time_abs)
        # self.layout_exposure_time_abs.addWidget(self.spin_exposure_time_abs)
        self.layout_exposure_time_abs.addWidget(self.slider_exposure_time_abs)
        self.layout_exposure_time_abs.addWidget(self.label_exposure_time_abs_value)

        # GainAuto
        # self.label_gain_auto = QLabel("GainAuto:")
        # self.label_gain_auto.setFont(UIFont.SpinBox)
        # self.chkbox_gain_auto = QCheckBox()
        # self.chkbox_gain_auto.clicked.connect(self.onGainAutoClick)
        # self.layout_gain_auto = QHBoxLayout()
        # self.layout_gain_auto.addWidget(self.label_gain_auto)
        # self.layout_gain_auto.addWidget(self.chkbox_gain_auto)

        # Gain
        self.label_gain_value = QtWidgets.QLabel(None)
        self.label_gain_value.setFixedSize(QSize(80, 40))
        self.label_gain_value.setStyleSheet(UIStyleSheet.SliderLabel)
        self.label_gain_value.setAlignment(Qt.AlignVCenter)
        def show_gain():
            self.label_gain_value.setText(str(self.slider_gain.value() / 10))

        self.label_gain = QLabel("Gain (dB):")
        # self.label_gain.setFont(UIFont.SpinBox)
        self.label_gain.setStyleSheet(UIStyleSheet.CameraStettingLocalVariabl)
        # self.spin_gain = QSpinBox()
        # self.spin_gain.setFont(UIFont.SpinBox)
        # self.spin_gain.setMinimum(spin_min_value)
        # self.spin_gain.setMaximum(spin_max_value)
        # self.spin_gain.setSingleStep(10)
        self.slider_gain = QtWidgets.QSlider()
        self.slider_gain.setRange(0, 480)
        self.slider_gain.setOrientation(1)   
        self.slider_gain.setFixedSize(QSize(465, 20)) 
        self.slider_gain.setStyleSheet(UIStyleSheet.Slider)
        self.slider_gain.valueChanged.connect(show_gain)
        self.layout_gain = QHBoxLayout()
        self.layout_gain.addWidget(self.label_gain)       
        self.layout_gain = QHBoxLayout()
        self.layout_gain.addWidget(self.label_gain)
        # self.layout_gain.addWidget(self.spin_gain)
        self.layout_gain.addWidget(self.slider_gain)
        self.layout_gain.addWidget(self.label_gain_value)

        # BalanceWhiteAuto
        # self.label_balance_white_auto = QLabel("BalanceWhiteAuto:")
        # self.label_balance_white_auto.setFont(UIFont.SpinBox)
        # self.chkbox_balance_white_auto = QCheckBox()
        # self.chkbox_balance_white_auto.clicked.connect(self.onBalanceWhiteAutoClick)
        # self.layout_balance_white_auto = QHBoxLayout()
        # self.layout_balance_white_auto.addWidget(self.label_balance_white_auto)
        # self.layout_balance_white_auto.addWidget(self.chkbox_balance_white_auto)

        # BalanceRatioRed
        self.label_balance_ratio_red_value = QtWidgets.QLabel(None)
        self.label_balance_ratio_red_value.setFixedSize(QSize(80, 40))
        self.label_balance_ratio_red_value.setStyleSheet(UIStyleSheet.SliderLabel)
        self.label_balance_ratio_red_value.setAlignment(Qt.AlignVCenter)
        def show_balance_ratio_red():
            self.label_balance_ratio_red_value.setText(str(self.slider_balance_ratio_red.value()))

        self.label_balance_ratio_red = QLabel("BalanceRatioRed:")
        # self.label_balance_ratio_red.setFont(UIFont.SpinBox)
        self.label_balance_ratio_red.setStyleSheet(UIStyleSheet.CameraStettingLocalVariabl)
        # self.spin_balance_ratio_red = QSpinBox()
        # self.spin_balance_ratio_red.setFont(UIFont.SpinBox)
        # self.spin_balance_ratio_red.setMinimum(spin_min_value)
        # self.spin_balance_ratio_red.setMaximum(spin_max_value)
        self.slider_balance_ratio_red = QtWidgets.QSlider()
        self.slider_balance_ratio_red.setRange(0, 255)   
        self.slider_balance_ratio_red.setOrientation(1)   
        self.slider_balance_ratio_red.setFixedSize(QSize(465, 20)) 
        self.slider_balance_ratio_red.setStyleSheet(UIStyleSheet.Slider)
        self.slider_balance_ratio_red.valueChanged.connect(show_balance_ratio_red)
        self.layout_balance_ratio_red = QHBoxLayout()
        self.layout_balance_ratio_red.addWidget(self.label_balance_ratio_red)
        # self.layout_balance_ratio_red.addWidget(self.spin_balance_ratio_red)
        self.layout_balance_ratio_red.addWidget(self.slider_balance_ratio_red)
        self.layout_balance_ratio_red.addWidget(self.label_balance_ratio_red_value)

        # BalanceRatioBlue
        self.label_balance_ratio_blue_value = QtWidgets.QLabel(None)
        self.label_balance_ratio_blue_value.setFixedSize(QSize(80, 40))
        self.label_balance_ratio_blue_value.setStyleSheet(UIStyleSheet.SliderLabel)
        self.label_balance_ratio_blue_value.setAlignment(Qt.AlignVCenter)
        def show_balance_ratio_blue():
            self.label_balance_ratio_blue_value.setText(str(self.slider_balance_ratio_blue.value()))

        self.label_balance_ratio_blue = QLabel("BalanceRatioBlue:")
        # self.label_balance_ratio_blue.setFont(UIFont.SpinBox)
        self.label_balance_ratio_blue.setStyleSheet(UIStyleSheet.CameraStettingLocalVariabl)
        # self.spin_balance_ratio_blue = QSpinBox()
        # self.spin_balance_ratio_blue.setFont(UIFont.SpinBox)
        # self.spin_balance_ratio_blue.setMinimum(spin_min_value)
        # self.spin_balance_ratio_blue.setMaximum(spin_max_value)
        self.slider_balance_ratio_blue = QtWidgets.QSlider()
        self.slider_balance_ratio_blue.setRange(0, 255)   
        self.slider_balance_ratio_blue.setOrientation(1)   
        self.slider_balance_ratio_blue.setFixedSize(QSize(465, 20)) 
        self.slider_balance_ratio_blue.setStyleSheet(UIStyleSheet.Slider)
        self.slider_balance_ratio_blue.valueChanged.connect(show_balance_ratio_blue)
        self.layout_balance_ratio_blue = QHBoxLayout()
        self.layout_balance_ratio_blue.addWidget(self.label_balance_ratio_blue)
        self.layout_balance_ratio_blue = QHBoxLayout()
        self.layout_balance_ratio_blue.addWidget(self.label_balance_ratio_blue)
        # self.layout_balance_ratio_blue.addWidget(self.spin_balance_ratio_blue)
        self.layout_balance_ratio_blue.addWidget(self.slider_balance_ratio_blue)
        self.layout_balance_ratio_blue.addWidget(self.label_balance_ratio_blue_value)

        # BalanceRatioGreen
        self.label_balance_ratio_green_value = QtWidgets.QLabel(None)
        self.label_balance_ratio_green_value.setFixedSize(QSize(80, 40))
        self.label_balance_ratio_green_value.setStyleSheet(UIStyleSheet.SliderLabel)
        self.label_balance_ratio_green_value.setAlignment(Qt.AlignVCenter)
        def show_balance_ratio_green():
            self.label_balance_ratio_green_value.setText(str(self.slider_balance_ratio_green.value()))

        self.label_balance_ratio_green = QLabel("BalanceRatioGreen:")
        # self.label_balance_ratio_green.setFont(UIFont.SpinBox)
        self.label_balance_ratio_green.setStyleSheet(UIStyleSheet.CameraStettingLocalVariabl)
        # self.spin_balance_ratio_green = QSpinBox()
        # self.spin_balance_ratio_green.setFont(UIFont.SpinBox)
        # self.spin_balance_ratio_green.setMinimum(spin_min_value)
        # self.spin_balance_ratio_green.setMaximum(spin_max_value)
        self.slider_balance_ratio_green = QtWidgets.QSlider()
        self.slider_balance_ratio_green.setRange(0, 255)   
        self.slider_balance_ratio_green.setOrientation(1)   
        self.slider_balance_ratio_green.setFixedSize(QSize(465, 20)) 
        self.slider_balance_ratio_green.setStyleSheet(UIStyleSheet.Slider)
        self.slider_balance_ratio_green.valueChanged.connect(show_balance_ratio_green)
        self.layout_balance_ratio_green = QHBoxLayout()
        self.layout_balance_ratio_green.addWidget(self.label_balance_ratio_green)
        # self.layout_balance_ratio_green.addWidget(self.spin_balance_ratio_green)
        self.layout_balance_ratio_green.addWidget(self.slider_balance_ratio_green)
        self.layout_balance_ratio_green.addWidget(self.label_balance_ratio_green_value)

        # direction
        self.label_direction = QLabel("Direction:")
        self.label_direction.setStyleSheet(UIStyleSheet.CameraStettingLocalVariabl)
        self.combo_direction = QComboBox()
        self.combo_direction.addItems([str(i) for i in range(4)])
        self.combo_direction.setFont(UIFont.SpinBox)
        self.combo_direction.setStyleSheet(UIStyleSheet.CameraSettingCameraCombobox)
        #self.combo_direction.setFixedSize(QSize(512, 40))
        self.layout_direction = QHBoxLayout()
        self.layout_direction.addWidget(self.label_direction, 1)
        self.layout_direction.addWidget(self.combo_direction, 2)

        #Global variable Title
        self.label_global_title = QLabel("相機格式設定")
        self.label_global_title.setStyleSheet(UIStyleSheet.CameraStettingGlobalTitle)
        self.label_global_title.setFixedSize(QSize(200, 50))
        self.label_global_title.setAlignment(Qt.AlignCenter)
        self.layout_global_title = QHBoxLayout()
        self.layout_global_title.addWidget(self.label_global_title, alignment=Qt.AlignmentFlag.AlignLeft)

        # FPS
        self.label_fps = QLabel("FPS:")
        self.label_fps.setStyleSheet(UIStyleSheet.CameraStettingGlobalVariabl)
        self.combo_fps = QComboBox()
        # self.combo_fps.addItems(self.items_fps)
        # self.combo_fps.setFont(UIFont.SpinBox)
        self.combo_fps.setStyleSheet(UIStyleSheet.CameraSettingGlobalVariableCombobox)
        self.combo_fps.setFixedSize(QSize(512, 40))
        self.combo_fps.currentIndexChanged.connect(self.changeFPS)
        self.layout_fps = QHBoxLayout()
        self.layout_fps.addWidget(self.label_fps)
        self.layout_fps.addWidget(self.combo_fps)

        # RecordResolution
        self.label_record_resolution = QLabel("Resolution:")
        self.label_record_resolution.setStyleSheet(UIStyleSheet.CameraStettingGlobalVariabl)
        self.combo_record_resolution = QComboBox()
        # self.combo_record_resolution.addItems(self.items_resolution)
        # self.combo_record_resolution.setFont(UIFont.SpinBox)
        self.combo_record_resolution.setStyleSheet(UIStyleSheet.CameraSettingGlobalVariableCombobox)
        self.combo_record_resolution.setFixedSize(QSize(512, 40))
        self.combo_record_resolution.currentIndexChanged.connect(self.changeResolution)
        self.layout_record_resolution = QHBoxLayout()
        self.layout_record_resolution.addWidget(self.label_record_resolution)
        self.layout_record_resolution.addWidget(self.combo_record_resolution)

        # container_layout.addWidget(self.camera_cb, alignment=Qt.AlignCenter)
        container_layout.addLayout(self.layout_local_title)
        container_layout.addLayout(self.layout_brightness)
        container_layout.addLayout(self.layout_exposure_time_abs)
        # container_layout.addLayout(self.layout_exposure_auto)
        container_layout.addLayout(self.layout_gain)
        # container_layout.addLayout(self.layout_gain_auto)
        container_layout.addLayout(self.layout_balance_ratio_red)
        container_layout.addLayout(self.layout_balance_ratio_blue)
        container_layout.addLayout(self.layout_balance_ratio_green)
        container_layout.addLayout(self.layout_direction)
        # container_layout.addLayout(self.layout_balance_white_auto)
        container_layout.addLayout(self.layout_global_title)
        container_layout.addLayout(self.layout_fps)
        container_layout.addLayout(self.layout_record_resolution)
        container.setLayout(container_layout)
        container.setFixedHeight(800)
        return container

    def setSelectedCameras(self, num):
        self.selected_num_cameras = num
        # self.idx_list = num

    def setIdxList(self, idx_list):
        self.idx_list = idx_list

    def changeResolution(self, idx):
        self.combo_fps.clear()
        if self.combo_record_resolution.currentText() == "":
            return

        self.items_fps.clear()
        self.combo_fps.clear()

        for f in self.items_resolution[idx]['target_fps']:
            self.items_fps.append(f)
            self.combo_fps.addItem(str(f))

    def changeFPS(self):
        pass

    def setLayoutValue(self):
        # set button value (camera number)
        # self.btn_apply.setText(f'套用(相機 {self.camera_cb.currentText()})')
        self.btn_reset.setText(f'重置')

        p = self.camera_widget.cameraList[self.cam_idx].getCameraParameters()

        # self.spin_brightness.setValue(int(self.cameraCfg['Camera']['Brightness']))
        self.slider_brightness.setValue(int(p['BlackLevel']))
        # if self.cameraCfg['Camera']['ExposureAuto'] == 'On':
        #     self.chkbox_exposure_auto.setChecked(True)
        # else:
        #     self.chkbox_exposure_auto.setChecked(False)
        # self.spin_exposure_time_abs.setValue(int(self.cameraCfg['Camera']['ExposureTimeAbs']))
        self.slider_exposure_time_abs.setValue(int(p['ExposureTime']))
        # if self.cameraCfg['Camera']['GainAuto'] == 'On':
        #     self.chkbox_gain_auto.setChecked(True)
        # else:
        #     self.chkbox_gain_auto.setChecked(False)
        self.slider_gain.setValue(int(p['Gain'])*10)
        # if self.cameraCfg['Camera']['BalanceWhiteAuto'] == 'On':
        #     self.chkbox_balance_white_auto.setChecked(True)
        # else:
        #     self.chkbox_balance_white_auto.setChecked(False)
        self.slider_balance_ratio_red.setValue(int(p['BalanceWhiteRed']/0.015625))
        self.slider_balance_ratio_blue.setValue(int(p['BalanceWhiteBlue']/0.015625))
        self.slider_balance_ratio_green.setValue(int(p['BalanceWhiteGreen']/0.015625))

        self.combo_direction.setCurrentText(str(p["direction"]))

        self.fps_changed = False

        # set current resolution
        record_resolution = p['RecordResolution']
        record_skipping = p['skipping']
        for i, item in enumerate(self.items_resolution):
            print(record_resolution)
            if (item['width'], item['height']) == tuple(record_resolution) \
                and item['skipping'] == record_skipping:
                self.combo_record_resolution.setCurrentIndex(i)

        # set current fps
        record_fps = p['fps']
        for i, item in enumerate(self.items_fps):
            if str(item) == record_fps:
                self.combo_fps.setCurrentIndex(i)

    def onCameraChanged(self):
        if self.init_combo_flag:
            self.init_combo_flag = False
            return

        self.cam_idx = self.idx_list[int(self.camera_cb.currentIndex())]

        #     #clear combobox
        self.combo_fps.clear()
        self.combo_record_resolution.clear()

        self.capture_formats = self.camera_widget.cameraList[self.cam_idx].getCaptureFormats()

        self.items_resolution.clear()
        self.combo_record_resolution.clear()

        for f in self.capture_formats:
            self.items_resolution.append(f)
            self.combo_record_resolution.addItem(f"({f['width']}, {f['height']}) skiping={f['skipping']}")

        self.setLayoutValue()

        self.camera_widget.setHidden(False, [self.cam_idx])
        self.camera_widget.setHidden(True, [i for i in range(self.num_cameras) if i != self.cam_idx])

    # def onExposureAutoClick(self):
    #     if self.chkbox_exposure_auto.isChecked():
    #         self.spin_exposure_time_abs.setEnabled(False)
    #     else:
    #         self.spin_exposure_time_abs.setEnabled(True)

    # def onGainAutoClick(self):
    #     if self.chkbox_gain_auto.isChecked():
    #         self.spin_gain.setEnabled(False)
    #     else:
    #         self.spin_gain.setEnabled(True)

    # def onBalanceWhiteAutoClick(self):
    #     if self.chkbox_balance_white_auto.isChecked():
    #         self.spin_balance_ratio_red.setEnabled(False)
    #         self.spin_balance_ratio_blue.setEnabled(False)
    #     else:
    #         self.spin_balance_ratio_red.setEnabled(True)
    #         self.spin_balance_ratio_blue.setEnabled(True)

    def applySetting(self):
        cam_idx = self.idx_list[int(self.camera_cb.currentIndex())]

        # Set parameters
        parameters = {
            'BlackLevel'       : self.slider_brightness.value(),
            'ExposureTime'     : self.slider_exposure_time_abs.value(),
            'GainAuto'         : "Off",
            'Gain'             : self.slider_gain.value() / 10,
            'BalanceWhiteAuto' : "Off",
            'BalanceWhiteRed'  : self.slider_balance_ratio_red.value() * 0.015625,
            'BalanceWhiteBlue' : self.slider_balance_ratio_blue.value() * 0.015625,
            'BalanceWhiteGreen': self.slider_balance_ratio_green.value() * 0.015625,
        }

        self.camera_widget.cameraList[cam_idx].setCameraParameters(parameters)

        # Set resolution
        res = self.items_resolution[self.combo_record_resolution.currentIndex()]
        fps = self.items_fps[self.combo_fps.currentIndex()]
        direction = int(self.combo_direction.currentText())

        self.camera_widget.cameraList[cam_idx].setCaptureFormat(res["width"], res["height"], fps, res["skipping"], direction=direction)

        # apply resolution / fps for all camera
        #for cam_idx in self.idx_list:
        #    self.camera_widget.cameraList[cam_idx].setCaptureFormat(res['width'], res['height'], fps, res['skipping'], None)

    def resetSetting(self):
        self.setLayoutValue()

    def backhome(self):
        msg = MsgContract(MsgContract.ID.PAGE_CHANGE, value='CameraSystem')
        self.myService.sendMessage(msg)

    def choosecamera(self):
        msg = MsgContract(MsgContract.ID.PAGE_CHANGE, value='ChooseCameraPage')
        self.myService.sendMessage(msg)
