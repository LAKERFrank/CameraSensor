import sys
import os
import logging
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *

DIRNAME = os.path.dirname(os.path.abspath(__file__))
sys.path.append(f"{DIRNAME}/../")

from UISettings import *
from Services import SystemService, MsgContract
from message import *
sys.path.append(f"{DIRNAME}/../lib")

class TestPage(QGroupBox):
    drawImageSignal = pyqtSignal(int, QPixmap)

    def __init__(self, camera_widget):
        super().__init__()

        self.camera_widget = camera_widget
        self.image_size = QSize(800, 600)

        self.is_recording = False

    def hideEvent(self, event):
        logging.debug(f"{self.__class__.__name__}: hided.")
        self.camera_widget.stopStreaming()
        self.deleteUI()

    def showEvent(self, event):
        logging.debug(f"{self.__class__.__name__}: shown.")
        self.setupUI()
        self.camera_widget.startStreaming()

    def setBackgroundService(self, service:SystemService):
        self.myService = service

    def setupUI(self):
        # main layout
        self.layout_main = QHBoxLayout()

        self.control_bar = self.getControlBar()

        self.camera_widget.initWidget(1, self.image_size, [4])

        # can turn off FPS display
        # self.camera_widget.toggleFPS()

        # connect signals
        self.camera_widget.getImageSignal.connect(self.receiveImage)
        self.drawImageSignal.connect(self.camera_widget.trueDrawImage)

        self.layout_main.addWidget(self.control_bar, 1, Qt.AlignCenter)
        self.layout_main.addWidget(self.camera_widget, 8, Qt.AlignCenter)

        self.setLayout(self.layout_main)

    def deleteUI(self):
        # disconnect signals
        try:
            self.camera_widget.getImageSignal.disconnect(self.receiveImage)
            self.drawImageSignal.disconnect(self.camera_widget.trueDrawImage)
        except TypeError:
            # I don't know why it will be ok when switching page,
            #  but it will throw error when closing the application.
            pass

        self.layout_main.removeWidget(self.camera_widget)
        while self.layout_main.count():
            item = self.layout_main.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()
        self.layout_main.deleteLater()

    def getControlBar(self):
        container = QWidget()
        container_layout = QVBoxLayout()

        self.btn_home = QPushButton()
        self.btn_home.setText('回首頁')
        self.btn_home.setFixedSize(QSize(160, 60))
        self.btn_home.setStyleSheet('font: 24px')
        self.btn_home.clicked.connect(self.backhome)

        self.btn_record = QPushButton()
        self.btn_record.setText('開始錄影')
        self.btn_record.setFixedSize(QSize(160, 60))
        self.btn_record.setStyleSheet('font: 24px')
        self.btn_record.clicked.connect(self.toggleRecord)

        container_layout.addWidget(self.btn_home)
        container_layout.addWidget(self.btn_record)
        container.setLayout(container_layout)
        return container

    @pyqtSlot(int, QPixmap)
    def receiveImage(self, camera_id: int, image: QPixmap):
        painter = QPainter(image)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setPen(Qt.white)
        painter.setFont(QFont("Arial", 36))
        painter.drawText(image.rect(), Qt.AlignCenter, str(camera_id))
        painter.end()

        self.drawImageSignal.emit(camera_id, image)

    def toggleRecord(self):
        if self.is_recording:
            self.is_recording = False
            self.btn_record.setText('開始錄影')
            self.btn_home.setEnabled(True)
            self.camera_widget.stopRecording()
        else:
            self.is_recording = True
            self.btn_record.setText('停止錄影')
            self.btn_home.setEnabled(False)
            self.camera_widget.startRecording()


    def backhome(self):
        msg = MsgContract(MsgContract.ID.PAGE_CHANGE, value='Home')
        self.myService.sendMessage(msg)
