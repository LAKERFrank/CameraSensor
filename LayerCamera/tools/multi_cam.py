import signal
import sys
from datetime import datetime

from PyQt5.QtCore import QSize, Qt, QTimer, QElapsedTimer
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QGridLayout, QComboBox, QWidget, QSizePolicy, QLabel, QHBoxLayout, QStackedWidget, QListWidget

from LayerCamera.camera.Camera import Camera
from lib.common import ROOTDIR, loadConfig, saveConfig

class CameraList:
    def __init__(self, num:int):
        self.num = num
        self.cameras:'list[Camera|None]' = [None for _ in range(self.num)]
        self.widgets:'list[CameraWidget|None]' = [CameraWidget(i, None) for i in range(self.num)]

        self.availableCameras = {item["serial"]: item for item in Camera.getAvailableCameras()}

        for v in self.availableCameras.values():
            v["in_use"] = False

    def setPreviewSize(self, w:int, h:int):
        for w in self.widgets:
            w.preview.setMinimumSize(w, h)

    def startRecord(self, dir):
        for i, c in enumerate(self.cameras):
            if c is not None:
                c.startRecording(False, dir, i)

    def stopRecord(self):
        for i, c in enumerate(self.cameras):
            if c is not None:
                c.stopRecording()

    def open(self, cam_idx:int, serial:str):
        if self.cameras[cam_idx] is not None:
            self.close(cam_idx)
        self.cameras[cam_idx] = Camera(serial, "")
        self.widgets[cam_idx].setCamera(self.cameras[cam_idx])
        self.cameras[cam_idx].init(preview_win_id=self.widgets[cam_idx].preview.winId())
        self.availableCameras[serial]['in_use'] = True

    def close(self, cam_idx:int):
        if self.cameras[cam_idx] is not None:
            self.availableCameras[self.cameras[cam_idx].serial]['in_use'] = False
            self.cameras[cam_idx].release()
            self.cameras[cam_idx] = None

class CameraWidget(QWidget):
    def __init__(self, index:int, camera:Camera):
        super().__init__()
        self.camera = camera

        self.mainLayout = QGridLayout()
        self.setLayout(self.mainLayout)

        self.labelTop = QLabel()
        self.labelTop.setText(f"Camera {index}")

        self.preview = QWidget()
        self.preview.setMinimumSize(400, 300)
        self.preview.setSizePolicy(
            QSizePolicy.Expanding, QSizePolicy.Expanding
        )

        #self.labelFormat = QLabel()
        #self.labelFormat.setText("Capture Format (Resolution / FPS / Direction):")

        #self.widgetFormat = FormatWidget(self)

        self.mainLayout.addWidget(self.labelTop, 0, 0, 1, 2, Qt.AlignCenter)
        #self.mainLayout.addWidget(self.labelCamera, 1, 0, 1, 1, Qt.AlignRight)
        #self.mainLayout.addWidget(self.comboCamera, 1, 1, 1, 1, Qt.AlignCenter)
        #self.mainLayout.addWidget(self.labelFormat, 2, 0, 1, 1, Qt.AlignRight)
        #self.mainLayout.addWidget(self.widgetFormat, 2, 1, 1, 1, Qt.AlignCenter)
        self.mainLayout.addWidget(self.preview, 4, 0, 1, 2, Qt.AlignCenter)

    def setCamera(self, camera:Camera):
        self.camera = camera

    #def updateCameraList(self, items):
    #    self.comboCamera.clear()
    #    self.comboCamera.addItem("", None)
    #    for i, (serial, item) in enumerate(items):
    #        self.comboCamera.addItem(serial, serial)
    #        if item["in_use"]:
    #            self.comboCamera.model().item(i+1).setEnabled(False)

    #def onCameraChanged(self, current_index):
    #    serial = self.comboCamera.itemData(current_index, Qt.ItemDataRole.UserRole)
    #    self.cameraSerial = serial
    #    self.cameraDirection = 0
    #    self.initCam(True)

    #def initCam(self, cam_changed=False):
    #    if self.camera:
    #        self.camera.release()

    #    if self.cameraSerial is not None:
    #        self.camera = Camera(self.cameraSerial, "")
    #        self.camera.init("", 0, self.cameraDirection, self.preview.winId())

    #        if cam_changed:
    #            self.widgetFormat.initCaptureFormats()
    #def close(self):
    #    if self.camera is not None:
    #        self.camera.release()
    #        self.camera = None
    #        self.comboCamera.setCurrentIndex(0)

        #self.labelCamera.setVisible(show_list)
        #self.comboCamera.setVisible(show_list)
        #self.labelFormat.setVisible(show_format)
        #self.widgetFormat.setVisible(show_format)
        #self.preview.setVisible(show_preview)

class RecordWidget(QWidget):
    def __init__(self, cam_list:CameraList):
        super().__init__()
        
        self.isRecording = False

        self.camList = cam_list

        self.mainLayout = QHBoxLayout()

        self.labelTime = QLabel()
        self.labelTime.setText("00:00:00.000")

        self.btnRecording = QPushButton()
        self.btnRecording.setText("Start Record")
        self.btnRecording.clicked.connect(self.toggleRecord)

        self.mainLayout.addWidget(self.labelTime)
        self.mainLayout.addWidget(self.btnRecording)

        self.setLayout(self.mainLayout)

        self.elapsedTimer = QElapsedTimer()
        self.timer = QTimer()
        self.timer.setInterval(50)
        self.timer.timeout.connect(self.onTimer)

    def onTimer(self):
        elapsed = self.elapsedTimer.elapsed()
        msec = elapsed % 1000
        sec = elapsed // 1000 % 60
        min = elapsed // 60000 % 60
        hour = elapsed // 3600000

        self.labelTime.setText(f"{hour:02d}:{min:02d}:{sec:02d}.{msec:03d}")

    def toggleRecord(self):
        if self.isRecording:
            self.camList.stopRecord()
            self.btnRecording.setText("Start Record")
            self.isRecording = False
            self.timer.stop()
        else:
            dtstr = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            self.camList.startRecord(dtstr)
            self.btnRecording.setText("Stop Record")
            self.isRecording = True
            self.elapsedTimer.restart()
            self.timer.start()

class FormatWidget(QWidget):
    def __init__(self, cameraWidget:CameraWidget):
        super().__init__()

        self.cameraWidget = cameraWidget

        self.mainLayout = QHBoxLayout()
        self.setLayout(self.mainLayout)

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

        self.mainLayout.addWidget(self.comboResolution)
        self.mainLayout.addWidget(self.comboFps)
        self.mainLayout.addWidget(self.comboDirection)
        self.mainLayout.addWidget(self.btnConfirmFormat)

    def updateFpsOption(self):
        data = self.comboResolution.itemData(self.comboResolution.currentIndex(), Qt.ItemDataRole.UserRole)

        if data is not None:
            for item in data["target_fps"]:
                self.comboFps.addItem(str(item), item)

            # set current index
            p = self.cameraWidget.camera.getCameraParameters()
            for i, fps in enumerate(data["target_fps"]):
                if p["fps"] == str(fps):
                    self.comboFps.setCurrentIndex(i)
                    break

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
        self.cameraFps = data

    def onDirectionChanged(self, current_index):
        self.cameraDirection = int(self.comboDirection.currentText())

    def onFormatConfirm(self):
        if self.cameraWidget.camera is not None:
            self.cameraWidget.camera.setCaptureFormat(self.cameraWidth, self.cameraHeight, self.cameraFps, self.cameraSkipping)
            self.cameraWidget.initCam()

    def clear(self):
        self.comboFps.clear()
        self.comboResolution.clear()

    def initCaptureFormats(self):
        self.captureFormats = self.cameraWidget.camera.getCaptureFormats()

        for item in self.captureFormats:
            self.comboResolution.addItem(f"({item['width']},{item['height']}) skipping={item['skipping']}", item)

        # set current index
        p = self.cameraWidget.camera.getCameraParameters()
        for i, item in enumerate(self.captureFormats):
            if p["RecordResolution"] == (item['width'], item['height']) \
                and (p["skipping"] == "" or p["skipping"] == item["skipping"]):
                self.comboResolution.setCurrentIndex(i)
                break

        self.updateFpsOption()

class ChoosePage(QWidget):
    def __init__(self, cam_list:CameraList):
        super().__init__()

        self.camList = cam_list

        self.mainLayout = QGridLayout()
        self.setLayout(self.mainLayout)

        self.comboCameras:'list[QComboBox]' = []

        for i in range(6):
            combo = QComboBox()
            combo.addItem("", None)
            combo.setMinimumWidth(50)
            self.comboCameras.append(combo)

        self.mainLayout.addWidget(self.comboCameras[0], 0, 2, Qt.AlignCenter)
        self.mainLayout.addWidget(self.comboCameras[1], 0, 0, Qt.AlignCenter)
        self.mainLayout.addWidget(self.comboCameras[2], 2, 0, Qt.AlignCenter)
        self.mainLayout.addWidget(self.comboCameras[3], 2, 2, Qt.AlignCenter)
        self.mainLayout.addWidget(self.comboCameras[4], 0, 1, Qt.AlignCenter)
        self.mainLayout.addWidget(self.comboCameras[5], 2, 1, Qt.AlignCenter)

    def refreshCombo(self):
        for i, combo in enumerate(self.comboCameras):
            combo.disconnect()

            combo.clear()
            combo.addItem("", None)
            for j, item in enumerate(self.camList.availableCameras.values()):
                combo.addItem(f"{item['serial']} ({item['name']})", item["serial"])
                if item['in_use']:
                    combo.model().item(j+1).setEnabled(False)
            if self.camList.cameras[i] is not None:
                info = self.camList.cameras[i].getDeviceInfo()
                combo.setCurrentText(f"{info['serial']} ({info['model']})")

            combo.currentIndexChanged.connect(self.onCameraChanged(i))

    def onCameraChanged(self, i:int):
        def change(current_index):
            serial = self.comboCameras[i].itemData(current_index, Qt.ItemDataRole.UserRole)
            if serial is not None and serial != "":
                self.camList.open(i, serial)
            else:
                self.camList.close(i)
            self.refreshCombo()
        return change

    def showEvent(self, event):
        self.refreshCombo()
        self.mainLayout.addWidget(self.camList.widgets[0], 1, 2, Qt.AlignCenter)
        self.mainLayout.addWidget(self.camList.widgets[1], 1, 0, Qt.AlignCenter)
        self.mainLayout.addWidget(self.camList.widgets[2], 3, 0, Qt.AlignCenter)
        self.mainLayout.addWidget(self.camList.widgets[3], 3, 2, Qt.AlignCenter)
        self.mainLayout.addWidget(self.camList.widgets[4], 1, 1, Qt.AlignCenter)
        self.mainLayout.addWidget(self.camList.widgets[5], 3, 1, Qt.AlignCenter)

    def hideEvent(self, event):
        for w in self.camList.widgets:
            self.mainLayout.removeWidget(w)

class PreviewPage(QWidget):
    def __init__(self, cam_list:CameraList):
        super().__init__()
        self.camList = cam_list
        self.mainLayout = QGridLayout()
        self.setLayout(self.mainLayout)

    def showEvent(self, event):
        self.mainLayout.addWidget(self.camList.widgets[0], 1, 2, Qt.AlignCenter)
        self.mainLayout.addWidget(self.camList.widgets[1], 1, 0, Qt.AlignCenter)
        self.mainLayout.addWidget(self.camList.widgets[2], 2, 0, Qt.AlignCenter)
        self.mainLayout.addWidget(self.camList.widgets[3], 2, 2, Qt.AlignCenter)
        self.mainLayout.addWidget(self.camList.widgets[4], 1, 1, Qt.AlignCenter)
        self.mainLayout.addWidget(self.camList.widgets[5], 2, 1, Qt.AlignCenter)

    def hideEvent(self, event):
        for w in self.camList.widgets:
            self.mainLayout.removeWidget(w)

class RecordPage(QWidget):

    def __init__(self, cam_list:CameraList):
        super().__init__()
        self.camList = cam_list
        self.mainLayout = QGridLayout()
        self.setLayout(self.mainLayout)

        self.recordWidget = RecordWidget(self.camList)

        self.mainLayout.addWidget(self.recordWidget, 3, 0, 1, 3, Qt.AlignCenter)

    def showEvent(self, event):
        self.mainLayout.addWidget(self.camList.widgets[0], 1, 2, Qt.AlignCenter)
        self.mainLayout.addWidget(self.camList.widgets[1], 1, 0, Qt.AlignCenter)
        self.mainLayout.addWidget(self.camList.widgets[2], 2, 0, Qt.AlignCenter)
        self.mainLayout.addWidget(self.camList.widgets[3], 2, 2, Qt.AlignCenter)
        self.mainLayout.addWidget(self.camList.widgets[4], 1, 1, Qt.AlignCenter)
        self.mainLayout.addWidget(self.camList.widgets[5], 2, 1, Qt.AlignCenter)

    def hideEvent(self, event):
        for w in self.camList.widgets:
            self.mainLayout.removeWidget(w)

class SettingPage(QWidget):
    def __init__(self, cam_list:CameraList):
        super().__init__()

        self.camList = cam_list
        self.mainLayout = QGridLayout()
        self.setLayout(self.mainLayout)

        self.labelInfos = [QLabel() for _ in range(6)]
        self.btnConfig = [QPushButton() for _ in range(6)]
        
        for b in self.btnConfig:
            b.setText("setting")

        self.bars = [QHBoxLayout() for _ in range(6)]
        for i in range(6):
            self.bars[i].addWidget(self.labelInfos[i])
            self.bars[i].addWidget(self.btnConfig[i])

        self.mainLayout.addLayout(self.bars[0], 2, 2, Qt.AlignCenter)
        self.mainLayout.addLayout(self.bars[1], 2, 0, Qt.AlignCenter)
        self.mainLayout.addLayout(self.bars[2], 4, 0, Qt.AlignCenter)
        self.mainLayout.addLayout(self.bars[3], 4, 2, Qt.AlignCenter)
        self.mainLayout.addLayout(self.bars[4], 2, 1, Qt.AlignCenter)
        self.mainLayout.addLayout(self.bars[5], 4, 1, Qt.AlignCenter)

    def onSetting(self):
        self.window

    def refreshInfo(self):
        for i in range(6):
            if self.camList.cameras[i] is not None:
                p = self.camList.cameras[i].getCameraParameters()
                self.labelInfos[i].setText(f"{p['RecordResolution']}@{p['fps']} FPS")

    def showEvent(self, event):
        self.mainLayout.addWidget(self.camList.widgets[0], 1, 2, Qt.AlignCenter)
        self.mainLayout.addWidget(self.camList.widgets[1], 1, 0, Qt.AlignCenter)
        self.mainLayout.addWidget(self.camList.widgets[2], 3, 0, Qt.AlignCenter)
        self.mainLayout.addWidget(self.camList.widgets[3], 3, 2, Qt.AlignCenter)
        self.mainLayout.addWidget(self.camList.widgets[4], 1, 1, Qt.AlignCenter)
        self.mainLayout.addWidget(self.camList.widgets[5], 3, 1, Qt.AlignCenter)
        self.refreshInfo()

    def hideEvent(self, event):
        for w in self.camList.widgets:
            self.mainLayout.removeWidget(w)

class MainPage(QWidget):
    def __init__(self, cam_list:CameraList):
        self.camList = cam_list
        super().__init__()
        self.menuList = QListWidget()
        self.menuList.insertItem(0, "Preview")
        self.menuList.insertItem(1, "Select")
        self.menuList.insertItem(2, "Record")
        self.menuList.insertItem(3, "Setting")

        self.stack0 = PreviewPage(self.camList)
        self.stack1 = ChoosePage(self.camList)
        self.stack2 = RecordPage(self.camList)
        self.stack3 = SettingPage(self.camList)

        self.stack = QStackedWidget()
        self.stack.addWidget(self.stack0)
        self.stack.addWidget(self.stack1)
        self.stack.addWidget(self.stack2)
        self.stack.addWidget(self.stack3)

        hbox = QHBoxLayout()
        hbox.addWidget(self.menuList)
        hbox.addWidget(self.stack)

        self.setLayout(hbox)

        self.menuList.currentRowChanged.connect(self.display)

    def display(self, i):
        self.stack.setCurrentIndex(i)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.isRecording = False

        self.setWindowTitle("Debug Cam")

        self.main_frame = QWidget()
        self.main_layout = QGridLayout()
        self.main_frame.setLayout(self.main_layout)

        self.camList = CameraList(6)

        self.cameraPage = MainPage(self.camList)

        self.main_layout.addWidget(self.cameraPage)

        self.setCentralWidget(self.main_frame)


if __name__ == "__main__":
    app = QApplication(sys.argv)

    signal.signal(signal.SIGTERM, lambda: app.exit())

    window = MainWindow()
    window.show()

    app.exec()