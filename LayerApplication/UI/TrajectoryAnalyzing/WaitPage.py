import os
import sys
import logging
import time
import subprocess
from PyQt5.QtWidgets import QGroupBox, QLabel, QGridLayout
from PyQt5.QtCore import Qt

from lib.common import ROOTDIR, REPLAYDIR

from ..Services import SystemService, MsgContract
from ..UISettings import *

from lib.nodes import setupOfflineTrackingNodes

class WaitPage(QGroupBox):
    def __init__(self, cfg):
        super().__init__()
        # initalize Service
        self.myService = None
        self.cfg = cfg

        # setup UI
        self.setupUI()
    def hideEvent(self, event):
        logging.debug(f"{self.__class__.__name__}: hided.")

    def showEvent(self, event):
        logging.debug(f"{self.__class__.__name__}: shown.")
        time.sleep(3)
        self.startTracking()

    def setBackgroundService(self, service:SystemService):
        self.myService = service

    def setupUI(self):
        # setup Layout
        layout_main = QGridLayout()
        title = QLabel()
        title.setText("Calculating Trajectories...")
        title.setStyleSheet("QLabel { background-color:transparent; color : blue; font: bold 80px 'Times'; }")
        layout_main.addWidget(title, 0, 0, Qt.AlignCenter | Qt.AlignTop)
        # set content
        self.setLayout(layout_main)

    def startTracking(self):
        logging.debug(f"TrackNet start...")
        self.weights = "no122_30.tar"
        replay_date = sorted(os.listdir(REPLAYDIR))[-1]
        self.replay_dir = os.path.join(REPLAYDIR, replay_date)
        nodes = setupOfflineTrackingNodes('CameraSystem', self.cfg, self.replay_dir, self.weights)
        self.myService.addNodes(nodes)

    def startModel3D(self):
        logging.debug(f"Model3D start...")
        config = os.path.abspath(os.path.join(self.replay_dir, "config"))
        output_csv = os.path.abspath(os.path.join(self.replay_dir, "Model3D.csv"))
        cmd = f"python3 {ROOTDIR}/LayerContent/Model3D_offline.py --config {config} --output_csv {output_csv} --atleast1hit"
        # logging.debug(cmd)
        subprocess.call(cmd, shell=True)
        self.show3dPage()

    def show3dPage(self):
        msg = MsgContract(MsgContract.ID.PAGE_CHANGE, value='Model3dPage')
        self.myService.sendMessage(msg)