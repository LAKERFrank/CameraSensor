import sys
import os
import logging #
from pathlib import Path

from PyQt5.QtGui import QIcon, QPixmap, QFont, QImage
from PyQt5.QtWidgets import QApplication, QMainWindow, QGroupBox, QGridLayout, QWidget, QDialog, QMessageBox, QStatusBar, QLabel, QAction, QMenu, QVBoxLayout, QHBoxLayout, QComboBox, QPushButton
from PyQt5.QtCore import QSize, Qt


from LayerApplication.Rpc.RpcManager import RpcManager
from LayerApplication.Rpc.RpcStreamingBadminton import RpcStreamingBadminton
from LayerApplication.UI.CES_2024.Home import CESHomePage
from LayerApplication.UI.CameraSetting.RecordPage import RecordPage
from LayerApplication.UI.Debug.StreamingDemoPage import StreamingDemoPage
from lib.common import is_local_camera, saveConfig, loadConfig, setupLogLevel
from lib.nodes import setupCameras
from lib.message import *

from LayerCamera.RpcCameraWidget import RpcCameraWidget
#from LayerCamera.CameraWidget import CameraWidget

from LayerApplication.UI.UISettings import *
from LayerApplication.UI.HomePage import HomePage
from LayerApplication.UI.Services import SystemService
from LayerApplication.UI.CameraSetting.CameraSystem import CameraSystem
from LayerApplication.UI.CameraSetting.CameraSettingPage import CameraSettingPage
from LayerApplication.UI.CameraSetting.ExtrinsicPage import ExtrinsicPage
#from LayerApplication.UI.CameraSetting.autoExtrinsicPage import ExtrinsicPage
from LayerApplication.UI.CameraSetting.ReplayPage import ReplayPage
from LayerApplication.UI.CameraSetting.Check3dPage import Check3dPage
from LayerApplication.UI.CameraSetting.ChooseCameraPage import ChooseCameraPage
from LayerApplication.UI.CameraSetting.SettingPreviewPage import SettingPreviewPage
#from LayerApplication.UI.TestPage.TestPage import TestPage
#from LayerApplication.UI.Boom.BoomPage import BoomPage
#from LayerApplication.UI.MachineCalibration.VisualizePage import VisualizePage
#from LayerApplication.UI.MachineCalibration.CalibrationPage import CalibrationPage
from LayerApplication.UI.TrajectoryAnalyzing.TrajectoryAnalyzingPage import TrajectoryAnalyzingPage
from LayerApplication.UI.TrajectoryAnalyzing.WaitPage import WaitPage
from LayerApplication.UI.TrajectoryAnalyzing.Model3dPage import Model3dPage
#from LayerApplication.UI.NiceShot.NiceShot import NiceShot
#from LayerApplication.UI.NiceShot.NiceShotRecord import NiceShotRecord
#from LayerApplication.UI.NiceShot.Processing import Processing
#from LayerApplication.UI.pitcherUI.BaseballSettingPage import BaseballSettingPage
#from LayerApplication.UI.pitcherUI.ResultPage_keyframe import ResultPage_keyframe
#from LayerApplication.UI.pitcherUI.ResultPage_angleAnalyze import ResultPage_angleAnalyze
#from LayerApplication.UI.pitcherUI.ResultPage_powerChain import ResultPage_powerChain
#from LayerApplication.UI.pitcherUI.Result_angleDetail import Result_angleDetail
#from LayerApplication.UI.pitcherUI.ResultPage_3Dplot import ResultPage_3Dplot

from lib.common import ROOTDIR, ICONDIR

from LayerApplication.utils.Mqtt import MqttClient
from LayerApplication.utils.CameraExplorer import CameraExplorer

class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        # set desktop icon
        self.setWindowIcon(QIcon(f"{ICONDIR}/desktop.png"))

        # initialize
        self.pages = {}
        self.myService = None

        # loading project config
        cfg_file = f"{ROOTDIR}/config"
        cfg = loadConfig(cfg_file)

        # setup Logging Level
        setupLogLevel(level=cfg['Project']['logging_level'], log_file="UI")

        # setup background service
        service = SystemService(cfg)
        service.callback.connect(self.handleMessage)
        service.start()
        self.setBackgroundService(service)

        # setup Mqtt client
        broker_ip = cfg["Project"]["mqtt_broker"]
        broker_port = int(cfg["Project"]["mqtt_port"])
        self.mqtt_client = MqttClient(broker_ip, broker_port)
        self.camera_explorer = CameraExplorer(self.mqtt_client.getClient(), self.mqtt_client.getAppId())
        self.camera_explorer.explore()

        # setup UI
        self.setupUI(cfg)

    def closeEvent(self, event):
        self.hpCameraWidget.stop()
        if self.myService.isRunning():
            logging.debug("stop system service")
            self.myService.stop()
        logging.debug("closeEvent")

    def setupUI(self, cfg):
        self.layout_main = QGridLayout()
        mainWidget = QWidget()
        mainWidget.setLayout(self.layout_main)
        self.setCentralWidget(mainWidget)

        # setup Cameras Node of Project
        cameras = setupCameras(cfg)
        # start Camera Node
        #self.addNodes(cameras)

        #self.cameraWidget = CameraWidget(cfg, cameras, 4)
        #self.cameraWidget.setBackgroundService(self.myService)

        #self.hpCameraWidget = HPCameraWidget(cameras)
        #self.hpCameraWidget.init(6)

        if is_local_camera():
            from LayerCamera.HPCameraWidget import HPCameraWidget
            self.hpCameraWidget = HPCameraWidget(cameras)
            self.hpCameraWidget.init(8)
        else:
            self.rpcManager = RpcManager(self.camera_explorer.mqttc)
            self.rpcManager.setDeviceNamesFromConfig(cfg)
            self.rpcManager.open()
        
            self.streamingBadminton = RpcStreamingBadminton(self.rpcManager)

            self.hpCameraWidget = RpcCameraWidget(cameras, self.camera_explorer)
            self.hpCameraWidget.init(8)

        # Page :: Home
        self.home_page = HomePage()
        self.addPage("Home", self.home_page)
        # Page :: CameraSystem
        self.camera_system_page = CameraSystem(cfg["Project"]["mqtt_broker"], cfg, cameras, self.hpCameraWidget)
        self.addPage("CameraSystem", self.camera_system_page)
        # Page :: CameraSettingPage
        self.camera_setting_page = CameraSettingPage(cameras, self.hpCameraWidget)
        self.addPage("CameraSettingPage", self.camera_setting_page)
        # Page :: ExtrinsicPage
        self.extrinsic_page = ExtrinsicPage(self.hpCameraWidget)
        self.addPage("ExtrinsicPage", self.extrinsic_page)
        # Page :: ReplayPage
        #self.replay_page = ReplayPage(cfg)
        #self.addPage("ReplayPage", self.replay_page)
        # Page :: RecordPage
        self.record_page = RecordPage(self.hpCameraWidget)
        self.addPage("RecordPage", self.record_page)
        # Page :: Check3dPage
        self.check3d_page = Check3dPage(self.hpCameraWidget)
        self.addPage("Check3dPage", self.check3d_page)

        self.choose_camera_page = ChooseCameraPage(cameras, self.hpCameraWidget, self.camera_explorer)
        self.addPage("ChooseCameraPage", self.choose_camera_page)

        self.setting_preview_page = SettingPreviewPage(cameras, self.hpCameraWidget)
        self.addPage("SettingPreviewPage", self.setting_preview_page)
        #Page :: Boom
        #self.BoomPage = BoomPage(self.cameraWidget, cfg["Project"]["mqtt_broker"], cfg, cameras)
        #self.addPage("BoomPage", self.BoomPage)

        #self.machineCalibration = VisualizePage()
        #self.machineCalibration = CalibrationPage(self.cameraWidget)
        #self.addPage("MachineCalibration", self.machineCalibration)

        #self.test_page = TestPage(self.cameraWidget)
        #self.addPage("TestPage", self.test_page)

        # Page :: TrajectoryAnalyzing
        self.trajectory_analyzing_page = TrajectoryAnalyzingPage(self.hpCameraWidget)
        self.addPage("TrajectoryAnalyzingPage", self.trajectory_analyzing_page)
        # Page :: WaitPage
        self.wait_page = WaitPage(cfg)
        self.addPage("WaitPage", self.wait_page)
        # Page :: Model3dPage
        self.model3d_page = Model3dPage()
        self.addPage("Model3dPage", self.model3d_page)
        # Page :: NiceShot
        #self.niceshot_page = NiceShot()
        #self.addPage("NiceShot", self.niceshot_page)
        # Page :: NiceShotRecord
        #self.niceshot_record_page = NiceShotRecord(self.cameraWidget)
        #self.addPage("NiceShotRecord", self.niceshot_record_page)
        # Page :: Processing
        #self.processing_page = Processing(cfg)
        #self.addPage("Processing", self.processing_page)

        # page :: BaseballView
        #self.baseball_page = BaseballSettingPage(self.cameraWidget)
        #self.addPage("BaseballSettingPage", self.baseball_page)

        # Page :: ResultPage
        #self.reault_keyframe_page = ResultPage_keyframe()
        #self.addPage("ResultPage_keyframe", self.reault_keyframe_page)

        #self.resultPage_angleAnalyze_page = ResultPage_angleAnalyze()
        #self.addPage("ResultPage_angleAnalyze", self.resultPage_angleAnalyze_page)

        #self.resultPage_angleDetail_page = Result_angleDetail()
        #self.addPage("Result_angleDetail", self.resultPage_angleDetail_page)

        #self.resultPage_powerChain_page = ResultPage_powerChain()
        #self.addPage("ResultPage_powerChain", self.resultPage_powerChain_page)

        #self.resultPage_3Dplot_page = ResultPage_3Dplot()
        #self.addPage("ResultPage_3Dplot", self.resultPage_3Dplot_page)

        if not is_local_camera():
            self.mqttDemoPage = StreamingDemoPage(self.streamingBadminton, self.hpCameraWidget)
            self.addPage("MqttDemoPage", self.mqttDemoPage)

            # Page :: Home
            self.cesPage = CESHomePage(self.streamingBadminton)
            self.addPage("CESHomePage", self.cesPage)

        self.showPage("Home")

    def setBackgroundService(self, service:SystemService):
        if self.myService != None:
            self.closeService()
        self.myService = service

    def closeService(self):
        if self.myService.isRunning():
            logging.debug("stop system service")
            self.myService.stop()

    def addNodes(self, nodes):
        self.myService.addNodes(nodes)

    def addPage(self, name, page:QGroupBox):
        page.hide()
        page.setBackgroundService(self.myService)
        if name != "CESHomePage":
            page.setFixedSize(PAGE_SIZE)
        else:
            page.setFixedSize(QSize(1920, 1080))
        page.setStyleSheet("background-color: #DDD9C3")

        self.layout_main.addWidget(page, 0, 1, Qt.AlignCenter)
        if name in self.pages:
            del self.pages[name]
        self.pages[name] = page

    def showPage(self, show_name):
        logging.debug(f"{self.__class__.__name__}: showPage -> {show_name}")
        if show_name not in self.pages:
            logging.warning(f"Page {show_name} is not exist.")
            return False

        for name, page in self.pages.items():
            if show_name != name:
                page.hide()

        self.pages[show_name].show()
        self.setWindowTitle(show_name)

        return True

    def handleMessage(self, msg:MsgContract):
        logging.debug(f"{self.__class__.__name__}: handleMessage")
        if msg.id == MsgContract.ID.PAGE_CHANGE:
            page_name = msg.value
            if page_name == "CameraSettingPage":
                self.camera_setting_page.setIdxList(msg.data)
            if page_name == "ExtrinsicPage":
                self.extrinsic_page.setData(msg.data)
            if page_name == 'Check3dPage':
                self.check3d_page.setData(msg.data)
            if page_name == 'ChooseCameraPage':
                self.choose_camera_page.setFromPage(msg.data)
            if page_name == 'SettingPreviewPage':
                self.setting_preview_page.setFromPage(msg.data)
            if page_name == 'CameraSystem':
                self.camera_system_page.setFromPage(msg.data)
            self.showPage(page_name)
        elif msg.id == MsgContract.ID.TRACKNET_DONE:
            logging.debug(f"main page get [TRACKNET_DONE]")
            if msg.data == 'Processing':
                self.processing_page.startModel3D()
            else:
                self.wait_page.startModel3D()

if __name__ == '__main__':
    #updateLastExeDate()
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    window.move(70,55)
    sys.exit(app.exec_())