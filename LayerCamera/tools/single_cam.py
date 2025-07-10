#!/usr/bin/env python3
import os
import signal
import sys
import shutil
import time
from datetime import datetime

from PyQt5.QtCore import QSize, Qt, QTimer, QElapsedTimer, pyqtSignal
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QGridLayout, QComboBox, QWidget, QSizePolicy, QLabel, QHBoxLayout, QLineEdit, QGroupBox

from LayerCamera.CameraSystemC import recorder_module
from lib.common import ROOTDIR

debug_dot_path = os.path.dirname(os.path.realpath(__file__)) + "/gst_debug"

# clear directory
if os.path.exists(debug_dot_path):
    shutil.rmtree(debug_dot_path)

os.makedirs(debug_dot_path, exist_ok=True)

os.environ["GST_DEBUG"] = "2"
os.environ["GST_DEBUG_DUMP_DOT_DIR"] = debug_dot_path

class UdpWidget(QGroupBox):
    def __init__(self, recorder:recorder_module.Recorder):
        super().__init__("Udp")

        self.isStreaming = False

        self.recorder = recorder

        self.mainLayout = QHBoxLayout()

        self.textHost = QLineEdit("127.0.0.1")

        self.textPort = QLineEdit("9000")

        self.btnToggle = QPushButton("start")
        self.btnToggle.clicked.connect(self.onToggle)

        self.setLayout(self.mainLayout)

        self.mainLayout.addWidget(self.textHost)
        self.mainLayout.addWidget(self.textPort)
        self.mainLayout.addWidget(self.btnToggle)

        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def reset(self):
        self.btnToggle.setText("start")
        self.isStreaming = False

    def onToggle(self):
        if self.isStreaming:
            self.recorder.disableUdp()
            self.btnToggle.setText("start")
            self.isStreaming = False
        else:
            host = self.textHost.text()
            port = int(self.textPort.text())
            self.recorder.enableUdp(host, port)
            self.btnToggle.setText("stop")
            self.isStreaming = True

class MetricWidget(QGroupBox):
    def __init__(self, recorder:recorder_module.Recorder):
        super().__init__("Metrics")

        self.recorder = recorder

        self.mainLayout = QHBoxLayout()

        self.labelFps = QLabel()

        self.mainLayout.addWidget(self.labelFps)

        self.setLayout(self.mainLayout)

        self.elapsedTimer = QElapsedTimer()
        self.timer = QTimer()
        self.timer.setInterval(500)
        self.timer.timeout.connect(self.onTimer)
        self.timer.start()

    def onTimer(self):

        metric = self.recorder.getMetricData()

        self.labelFps.setText(f"current fps: {metric.fps:.01f}, average fps: {metric.avg_fps:.01f}, "
                              f"rendered: {metric.frames_rendered}, dropped: {metric.frames_dropped}")

class RecordWidget(QGroupBox):
    def __init__(self, recorder:recorder_module.Recorder):
        super().__init__("Record")

        self.isRecording = False

        self.recorder = recorder

        self.mainLayout = QHBoxLayout()

        self.labelTime = QLabel()
        self.labelTime.setText("00:00:00.000")

        self.btnRecording = QPushButton()
        self.btnRecording.setText("Start Record")
        self.btnRecording.clicked.connect(self.toggleRecord)

        self.comboMode = QComboBox()
        self.comboMode.addItem("none")
        self.comboMode.addItem("h264_low")
        self.comboMode.addItem("h264_high")
        self.comboMode.addItem("lossless")

        self.mainLayout.addWidget(self.comboMode)
        self.mainLayout.addWidget(self.labelTime)
        self.mainLayout.addWidget(self.btnRecording)

        self.setLayout(self.mainLayout)

        self.elapsedTimer = QElapsedTimer()
        self.timer = QTimer()
        self.timer.setInterval(66)
        self.timer.timeout.connect(self.onTimer)

    def reset(self):
        self.btnRecording.setText("Start Record")
        self.isRecording = False
        self.timer.stop()

    def onTimer(self):
        elapsed = self.elapsedTimer.elapsed()
        msec = elapsed % 1000
        sec = elapsed // 1000 % 60
        min = elapsed // 60000 % 60
        hour = elapsed // 3600000

        self.labelTime.setText(f"{hour:02d}:{min:02d}:{sec:02d}.{msec:03d}")

    def toggleRecord(self):
        if self.isRecording:
            self.recorder.stopRecording()
            self.btnRecording.setText("Start Record")
            self.isRecording = False
            self.timer.stop()
        else:
            dtstr = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            savepath = f"{ROOTDIR}/replay/{dtstr}/CameraReader_0.mp4"

            mode = self.comboMode.currentText()

            self.recorder.startRecording(save_path=savepath, imgbuf=False, mode=mode)
            self.btnRecording.setText("Stop Record")
            self.isRecording = True
            self.elapsedTimer.restart()
            self.timer.start()

class DisplayWidget(QGroupBox):
    def __init__(self, recorder:recorder_module.Recorder):
        super().__init__("Display")

        self.recorder = recorder

        self.mainLayout = QGridLayout()

        self.btnDisplay = QPushButton("start")
        self.btnDisplay.clicked.connect(self.display)

        self.btnResync = QPushButton("resync")
        self.btnResync.clicked.connect(self.resync)

        self.preview = QWidget()
        self.preview.setMinimumSize(600, 600)
        self.preview.setSizePolicy(
            QSizePolicy.Expanding, QSizePolicy.Expanding
        )
        self.preview.setStyleSheet("background-color:gray")

        self.mainLayout.addWidget(self.btnDisplay, 0, 0, Qt.AlignCenter)
        self.mainLayout.addWidget(self.preview, 1, 0, Qt.AlignCenter)
        self.mainLayout.addWidget(self.btnResync, 2, 0, Qt.AlignCenter)

        self.setLayout(self.mainLayout)

        self.isDisplay = False

    def reset(self):
        self.btnDisplay.setText("start")
        self.isDisplay = False

    def resync(self):
        t = time.time_ns()
        self.recorder.resync(t)

    def display(self):
        if self.isDisplay:
            self.recorder.disableDisplay()
            self.btnDisplay.setText("start")
            self.isDisplay = False
        else:
            self.recorder.enableDisplay(self.preview.winId())
            self.btnDisplay.setText("stop")
            self.isDisplay = True

class FormatWidget(QGroupBox):
    def __init__(self, recorder:recorder_module.Recorder, reload:pyqtSignal):
        super().__init__("Display")

        self.singlaReload = reload

        self.recorder = recorder

        self.mainLayout = QHBoxLayout()

        self.comboResolution = QComboBox()
        self.comboResolution.setMinimumWidth(300)
        self.comboResolution.currentIndexChanged.connect(self.onResolutionChanged)

        self.comboFps = QComboBox()
        self.comboFps.setMinimumWidth(50)
        self.comboFps.currentIndexChanged.connect(self.onFpsChanged)

        self.comboDirection = QComboBox()
        self.comboDirection.setMinimumWidth(30)
        self.comboDirection.currentIndexChanged.connect(self.onDirectionChanged)
        self.comboDirection.addItems([str(i) for i in range(0, 4)])

        self.btnConfirmFormat = QPushButton()
        self.btnConfirmFormat.setText("Confirm")
        self.btnConfirmFormat.clicked.connect(self.onFormatConfirm)

        self.setLayout(self.mainLayout)

        self.mainLayout.addWidget(self.comboResolution)
        self.mainLayout.addWidget(self.comboFps)
        self.mainLayout.addWidget(self.comboDirection)
        self.mainLayout.addWidget(self.btnConfirmFormat)

    def initCaptureFormats(self):
        self.captureFormats = self.recorder.getCaptureFormats()

        self.cameraWidth = self.captureFormats[0]['width']
        self.cameraHeight = self.captureFormats[0]['height']
        self.cameraSkipping = self.captureFormats[0]['skipping']
        self.cameraFps = self.captureFormats[0]['target_fps'][0]

        self.comboResolution.clear()
        for item in self.captureFormats:
            self.comboResolution.addItem(f"({item['width']},{item['height']}) skipping={item['skipping']}", item)

        self.updateFpsOption()

    def updateFpsOption(self):
        data = self.comboResolution.itemData(self.comboResolution.currentIndex(), Qt.ItemDataRole.UserRole)

        self.comboFps.clear()

        if data is not None:
            for item in data["target_fps"]:
                self.comboFps.addItem(str(item), item)

    def onResolutionChanged(self, current_index):
        data = self.comboResolution.itemData(current_index, Qt.ItemDataRole.UserRole)

        if data is not None:
            self.cameraWidth = data['width']
            self.cameraHeight = data['height']
            self.cameraSkipping = data['skipping']

            self.updateFpsOption()

    def onFpsChanged(self, current_index):
        # update fps
        data = self.comboFps.itemData(current_index, Qt.ItemDataRole.UserRole)
        if data is not None:
            self.cameraFps = data

    def onDirectionChanged(self, current_index):
        self.cameraDirection = int(self.comboDirection.currentText())

    def onFormatConfirm(self):
        self.singlaReload.emit()

    def reset(self):
        self.comboResolution.clear()
        self.comboFps.clear()

# Subclass QMainWindow to customize your application's main window
class MainWindow(QMainWindow):

    reload = pyqtSignal()

    def __init__(self):
        super().__init__()

        self.recorder = recorder_module.Recorder(ROOTDIR)

        self.setWindowTitle("Debug Cam")

        self.main_frame = QWidget()
        self.main_layout = QGridLayout()
        self.main_frame.setLayout(self.main_layout)

        self.labelCamera = QLabel()
        self.labelCamera.setText("Camera:")

        self.comboCamera = QComboBox()
        self.comboCamera.addItem("", None)
        self.comboCamera.setMinimumWidth(50)
        self.comboCamera.currentIndexChanged.connect(self.onCameraChanged)

        for item in self.recorder.listAvailableCamera():
            self.comboCamera.addItem(f"{item['serial']} ({item['name']})", item["serial"])

        self.widgetFormat = FormatWidget(self.recorder, self.reload)
        self.widgetRecord = RecordWidget(self.recorder)
        self.widgetUdp = UdpWidget(self.recorder)
        self.widgetDisplay = DisplayWidget(self.recorder)
        self.widgetMetric = MetricWidget(self.recorder)

        self.main_layout.addWidget(self.labelCamera, 0, 0, 1, 1, Qt.AlignRight)
        self.main_layout.addWidget(self.comboCamera, 0, 1, 1, 1, Qt.AlignCenter)
        self.main_layout.addWidget(self.widgetFormat, 1, 0, 1, 2, Qt.AlignLeft)
        self.main_layout.addWidget(self.widgetRecord, 2, 0, 1, 2, Qt.AlignLeft)
        self.main_layout.addWidget(self.widgetUdp, 3, 0, 1, 2, Qt.AlignLeft)
        self.main_layout.addWidget(self.widgetMetric, 4, 0, 1, 2, Qt.AlignLeft)
        self.main_layout.addWidget(self.widgetDisplay, 5, 0, 1, 2, Qt.AlignLeft)

        self.setCentralWidget(self.main_frame)

        self.reload.connect(self.initCam)

    def reset(self, cam_changed=False):
        if cam_changed:
            self.widgetFormat.reset()
        self.widgetRecord.reset()
        self.widgetUdp.reset()
        self.widgetDisplay.reset()

    def onTest(self):
        self.recorder.enableUdp("127.0.0.1", 9000)

    def onCameraChanged(self, current_index):
        serial = self.comboCamera.itemData(current_index, Qt.ItemDataRole.UserRole)
        self.cameraSerial = serial
        self.initCam(True)

    def initCam(self, cam_changed=False):
        self.recorder.release()
        if self.cameraSerial is not None:
            d = self.widgetFormat.cameraDirection
            self.recorder.init(self.cameraSerial, 0, "", 0, d, clockoverlay=False, software_trigger=False)

            if cam_changed:
                self.widgetFormat.initCaptureFormats()

            w = self.widgetFormat.cameraWidth
            h = self.widgetFormat.cameraHeight
            f = self.widgetFormat.cameraFps
            s = self.widgetFormat.cameraSkipping

            self.recorder.setCaptureFormat(w, h, f, s)
            self.recorder.start(trigger_start=time.time_ns())
        else:
            self.reset(cam_changed)

if __name__ == "__main__":
    app = QApplication(sys.argv)

    signal.signal(signal.SIGTERM, lambda: app.exit())

    window = MainWindow()
    window.show()

    app.exec()

    # Generate dot png
    for f in os.listdir(debug_dot_path):
        if f[-4:] == ".dot":
            os.system(f"dot -Tpng \"{debug_dot_path}/{f}\" > \"{debug_dot_path}/{f[:-4]}.png\"")
            print(f"Generate {f[:-4]}.png")
