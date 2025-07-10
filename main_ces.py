import sys
import os
import logging #
from pathlib import Path

from PyQt5.QtGui import QIcon, QPixmap, QFont, QImage
from PyQt5.QtWidgets import QApplication, QMainWindow, QGroupBox, QGridLayout, QWidget, QDialog, QMessageBox, QStatusBar, QLabel, QAction, QMenu, QVBoxLayout, QHBoxLayout, QComboBox, QPushButton
from PyQt5.QtCore import QSize, Qt


from LayerApplication.Rpc.RpcManager import RpcManager
from LayerApplication.Rpc.RpcStreamingBadminton import RpcStreamingBadminton
from LayerApplication.utils.Mqtt import MqttClient
from LayerCamera.RpcCameraWidget import RpcCameraWidget
from lib.common import saveConfig, loadConfig, setupLogLevel
from lib.nodes import setupCameras
from lib.message import *

from LayerCamera.HPCameraWidget import HPCameraWidget
#from LayerCamera.CameraWidget import CameraWidget

from LayerApplication.UI.UISettings import *
from LayerApplication.UI.Services import SystemService
from LayerApplication.UI.CES_2024.Home import CESHomePage

from lib.common import ROOTDIR, ICONDIR

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

        # setup Mqtt client
        broker_ip = cfg["Project"]["mqtt_broker"]
        broker_port = int(cfg["Project"]["mqtt_port"])
        self.mqtt_client = MqttClient(broker_ip, broker_port)

        self.rpcManager = RpcManager(self.mqtt_client.getClient())
        self.rpcManager.setDeviceNamesFromConfig(cfg)
        self.rpcManager.open()
        
        self.streamingBadminton = RpcStreamingBadminton(self.rpcManager)

        # Page :: Home
        self.home_page = CESHomePage(self.streamingBadminton)
        self.addPage("CESHome", self.home_page)
        self.showPage("CESHome")

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
        page.setFixedSize(QSize(1920, 1080))

        self.layout_main.addWidget(page, 0, 1, Qt.AlignCenter)
        if name in self.pages:
            del self.pages[name]
        self.pages[name] = page

    def showPage(self, show_name):
        logging.debug(f"{self.__class__.__name__}: showPage -> {show_name}")
        if show_name not in self.pages:
            logging.warning(f"Page {name} is not exist.")
            return False

        for name, page in self.pages.items():
            if show_name != name:
                page.hide()

        self.pages[show_name].show()
        self.setWindowTitle(show_name)

        return True

    def handleMessage(self, msg:MsgContract):
        logging.debug(f"{self.__class__.__name__}: handleMessage")

if __name__ == '__main__':
    #updateLastExeDate()
    app = QApplication(sys.argv)
    window = MainWindow()
    window.showFullScreen()
    # window.move(70,55)
    sys.exit(app.exec_())