import subprocess
import logging
import os
import sys
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from UISettings import *
from Services import SystemService, MsgContract
from message import *

DIRNAME = os.path.dirname(os.path.abspath(__file__))
sys.path.append(f"{DIRNAME}/../")
sys.path.append(f"{DIRNAME}/../lib")
cameranum = 4
class BaseballSettingPage(QGroupBox):
    drawImageSignal = pyqtSignal(int, QPixmap)
    def __init__(self, camera_widget):
        super().__init__()
        self.camera_widget = camera_widget
        self.image_size = QSize(1200, 800)

        self.is_recording = False
        # self.setupUI()
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
        self.layout_main = QGridLayout()
        self.layout_main.setColumnStretch(1, 2)
        # Page Title
        self.pagetitle = QLabel()
        self.pagetitle.setStyleSheet(UIStyleSheet.TitleText)
        line = QFrame()
        line.setFrameStyle(QFrame.HLine | QFrame.Sunken)

        self.control_panel = self.getControlPanel()
        # layout_main = QHBoxLayout()
        self.layout_main.addWidget(self.control_panel,0,0, alignment=Qt.AlignHCenter)

        self.camera_widget.initWidget(1, self.image_size, [cameranum])
        print(self.image_size)

        self.layout_main.addWidget(self.camera_widget, 0, 1, Qt.AlignCenter|Qt.AlignHCenter)

        self.setLayout(self.layout_main)
    def selection_change(self,index):
        # selected_index = index
        cameranum = 4 - index
        print(cameranum)
        self.camera_widget.stopStreaming()
        self.image_size = QSize(1200, 800)
        self.camera_widget.initWidget(1, self.image_size, [cameranum])
        print(self.image_size)
        self.camera_widget.startStreaming()

    def getControlPanel(self):
        container = QWidget()
        container_layout = QVBoxLayout()

        self.combo_box = QComboBox()
        self.combo_box.addItem("Camera4")
        self.combo_box.addItem("Camera3")
        self.combo_box.addItem("Camera2")
        self.combo_box.addItem("Camera1")
        self.combo_box.addItem("Camera0")


        self.combo_box.currentIndexChanged.connect(self.selection_change)
        # 設定相機
        self.btn_StartRecord = QPushButton()
        self.btn_StartRecord.setText('開始錄影')
        self.btn_StartRecord.setFixedSize(QSize(160, 60))
        self.btn_StartRecord.setStyleSheet('font: 24px')
        self.btn_StartRecord.clicked.connect(self.toggleRecord)
        # 分析按鈕
        self.btn_Analyze = QPushButton()
        self.btn_Analyze.setText('分析影片')
        self.btn_Analyze.setFixedSize(QSize(160, 60))
        self.btn_Analyze.setStyleSheet('font: 24px')
        self.btn_Analyze.clicked.connect(self.RunProgram)

        # 關鍵偵查看
        self.btn_Check3DPlot = QPushButton()
        self.btn_Check3DPlot.setText('查看3D骨架影片')
        self.btn_Check3DPlot.setFixedSize(QSize(160, 60))
        self.btn_Check3DPlot.clicked.connect(self.show3DPlotPage)
        self.btn_Check3DPlot.setStyleSheet('font: 20px')

        # 關鍵偵查看
        self.btn_CheckKeyFrame = QPushButton()
        self.btn_CheckKeyFrame.setText('查看關鍵幀')
        self.btn_CheckKeyFrame.setFixedSize(QSize(160, 60))
        self.btn_CheckKeyFrame.clicked.connect(self.ResultPage_keyframe)
        self.btn_CheckKeyFrame.setStyleSheet('font: 24px')
        # 查看分析角度
        self.btn_CheckAngle = QPushButton()
        self.btn_CheckAngle.setText('查看分析角度')
        self.btn_CheckAngle.setFixedSize(QSize(160, 60))
        self.btn_CheckAngle.clicked.connect(self.ResultPage_angleAnalyze)
        self.btn_CheckAngle.setStyleSheet('font: 24px')

        # 查看動力鏈
        self.btn_PowerChain = QPushButton()
        self.btn_PowerChain.setText('查看動力鏈')
        self.btn_PowerChain.setFixedSize(QSize(160, 60))
        self.btn_PowerChain.clicked.connect(self.ResultPage_powerChain)
        self.btn_PowerChain.setStyleSheet('font: 24px')

        #回首頁
        self.btn_Homepage = QPushButton()
        self.btn_Homepage.setText('回首頁')
        self.btn_Homepage.setFixedSize(QSize(160, 60))
        self.btn_Homepage.clicked.connect(self.showHomePage)
        self.btn_Homepage.setStyleSheet('font: 24px')

        container_layout.addWidget(self.combo_box)
        container_layout.addWidget(self.btn_StartRecord)
        container_layout.addWidget(self.btn_Analyze)
        container_layout.addWidget(self.btn_Check3DPlot)
        container_layout.addWidget(self.btn_CheckKeyFrame)
        container_layout.addWidget(self.btn_CheckAngle)
        container_layout.addWidget(self.btn_PowerChain)
        container_layout.addWidget(self.btn_Homepage)

        container.setLayout(container_layout)
        return container

    def showHomePage(self):
        msg = MsgContract(MsgContract.ID.PAGE_CHANGE, value='Home')
        self.myService.sendMessage(msg)

    def show3DPlotPage(self):
        msg = MsgContract(MsgContract.ID.PAGE_CHANGE, value='ResultPage_3Dplot')
        self.myService.sendMessage(msg)

    def ResultPage_keyframe(self):
        msg = MsgContract(MsgContract.ID.PAGE_CHANGE, value='ResultPage_keyframe')
        self.myService.sendMessage(msg)

    def ResultPage_angleAnalyze(self):
        msg = MsgContract(MsgContract.ID.PAGE_CHANGE, value='ResultPage_angleAnalyze')
        self.myService.sendMessage(msg)

    def ResultPage_powerChain(self):
        msg = MsgContract(MsgContract.ID.PAGE_CHANGE, value='ResultPage_powerChain')
        self.myService.sendMessage(msg)

    def RunProgram(self):
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Information)
            msg.setText("分析中")
            msg.setWindowTitle("訊息")
            msg.setStandardButtons(QMessageBox.Ok)
            msg.show()
            # msg.exec_()
            subprocess.call(["python3", "Pitcher/BaseballMain.py"])
            file = open('./Pitcher/resultData/log.txt', 'r')
            content = file.read()
            text = "投球姿勢有誤"
            file.close()
            if content == '0':
                subprocess.call(["python3", "Pitcher/3D_plot.py"])
                file = open('./Pitcher/resultData/3D_log.txt', 'r')
                content_3d = file.read()
                if content_3d == "0":
                    text = "分析完成"
                # text = "第一階段完成，進行第二階段中"
            if content == "1":
                text = "ERROR : BR - FP < 5"
            if content == '2':
                text = "ERROR : MER>BR"
            if content == "3":
                text = "ERROR : FP > BR"
            if content == '4':
                text = "未偵測到人體"
            msg.setWindowTitle("訊息")
            msg.setText(text)
            msg.exec_()


    def toggleRecord(self):
        if self.is_recording:
            self.is_recording = False
            self.btn_StartRecord.setText('開始錄影')
            self.btn_Homepage.setEnabled(True)
            self.btn_CheckKeyFrame.setEnabled(True)
            self.btn_CheckAngle.setEnabled(True)
            self.btn_PowerChain.setEnabled(True)
            self.btn_Analyze.setEnabled(True)
            self.btn_Check3DPlot.setEnabled(True)
            self.camera_widget.stopRecording()

            def find_newest_folder(directory):
                folders = [f for f in os.listdir(directory) if os.path.isdir(os.path.join(directory, f))]
                folders.sort(key=lambda x: os.path.getmtime(os.path.join(directory, x)), reverse=True)
                if folders:
                    return folders[0]
                else:
                    return None
            directory_path = "/home/nol/demo_barry/replay"
            newest_folder = find_newest_folder(directory_path)
            if newest_folder:
                newest_folder_path = os.path.join(directory_path, newest_folder)
                with open('./Pitcher/loc.txt', 'w', newline='') as logfile:
                    logfile.write(newest_folder_path)
                print("最新資料夾：", newest_folder_path)
            else:
                print("None newest folder")
        else:
            self.is_recording = True
            self.btn_StartRecord.setText('停止錄影')
            self.btn_Homepage.setEnabled(False)
            self.btn_CheckKeyFrame.setEnabled(False)
            self.btn_CheckAngle.setEnabled(False)
            self.btn_PowerChain.setEnabled(False)
            self.btn_Analyze.setEnabled(False)
            self.btn_Check3DPlot.setEnabled(False)
            self.camera_widget.startRecording()




    def deleteUI(self):
        self.layout_main.removeWidget(self.camera_widget)
        while self.layout_main.count():
            item = self.layout_main.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()
        self.layout_main.deleteLater()

    @pyqtSlot(int, QPixmap)
    def receiveImage(self, camera_id: int, image: QPixmap):
        painter = QPainter(image)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setPen(Qt.white)
        painter.setFont(QFont("Arial", 36))
        painter.drawText(image.rect(), Qt.AlignCenter, str(camera_id))
        painter.end()

        self.drawImageSignal.emit(camera_id, image)

