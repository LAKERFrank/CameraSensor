import os
import sys
import csv
import logging

from PyQt5.QtWidgets import QGroupBox, QVBoxLayout, QComboBox, QWidget, QHBoxLayout, QPushButton, QTableWidget, QGridLayout, QTableWidgetItem, QSizePolicy
from PyQt5.QtCore import QSize
from PyQt5.QtGui import QIcon

from lib.common import ROOTDIR
from ..Services import SystemService, MsgContract
#from ..TrajectoryAnalyzing.Trajectory3DVisualizeWidget import Trajectory3DVisualizeWidget
from .Trajectory3d import Trajectory3DVisualizeWidget
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# DIRNAME = os.path.dirname(os.path.abspath(__file__))
# ROOTDIR = os.path.dirname(DIRNAME)

REPLAYDIR = f"{ROOTDIR}/replay"
ICONDIR = f"{ROOTDIR}/UI/icon"

sys.path.append(f"{ROOTDIR}/lib")
from lib.point import Point
from lib.smooth import removeOuterPoint, detectEvent, smoothByEvent, detectBallTypeByEvent
from LayerContent.model3D_cut import cut_csv
from lib.trajectory_pred import placement_predict
class Model3dPage(QGroupBox):
    def __init__(self, dateList):
        super().__init__()

        # 用來放每次讀檔的軌跡，一段存一起
        # visibility = 1, event = 0 是球在飛行
        # event = 1 是擊球
        # event = 2 是發球
        # event = 3 是死球
        self.tracks = []
        self.ball_type = {
            '小球': 0,
            '挑球': 0,
            '推球': 0,
            '平球': 0,
            '切球': 0,
            '殺球': 0,
            '長球': 0,
            '例外': 0
        }

        # current points in video
        self.cur_points = 1
        self.max_points = 1

        # setup UI
        self.setFlat(True)  # 去除邊框
        self.setStyleSheet("QGroupBox { margin: 0px; padding: 0px; }")
        self.setupUI()

        self.widget_court.clearAll()
        # self.showTrajectory('2024-10-14_20-09-38')
        if(len(dateList)!=0):
            self.showMultipleTrajectories(dateList)
            # for date in dateList:
            #     self.showTrajectory(date)

    def hideEvent(self, event):
        self.widget_court.clearAll()
        logging.debug(f"{self.__class__.__name__}: hided.")

    def showEvent(self, event):
        #self.date_cb.clear()
        dates = [f for f in os.listdir(REPLAYDIR)]
        dates.sort()
        #self.date_cb.addItem('')
        #self.date_cb.addItems(dates)
        logging.debug(f"{self.__class__.__name__}: shown.")

    def setBackgroundService(self, service:SystemService):
        self.myService = service

    def setupUI(self):
        layout_main = QGridLayout()
        layout_main.setContentsMargins(0, 0, 0, 0)  # 設定邊距為 0
        layout_main.setSpacing(0)
        #self.control_bar = self.getControlBar()
        self.control_3d = self.getControl3d()
        #self.balltype_tb = self.getTable()
        #self.setLayout(QVBoxLayout())  # 可以使用 QVBoxLayout 來管理單一 widget
        #self.layout().addWidget(self.control_3d)

        #layout_main.addWidget(self.control_bar, 0, 0)
        layout_main.addWidget(self.control_3d, 0, 0)
        #layout_main.addWidget(self.balltype_tb, 1, 1)
        self.setLayout(layout_main)

    # def getControlBar(self):
    #     container = QWidget()
    #     container_layout = QHBoxLayout()
    #     '''
    #     self.btn_home = QPushButton()
    #     self.btn_home.setText('回首頁')
    #     self.btn_home.setFixedSize(QSize(120, 50))
    #     self.btn_home.setStyleSheet('font: 24px')
    #     self.btn_home.clicked.connect(self.backhome)
    #     '''
    #     # replay 日期選擇
    #     self.date_cb = QComboBox()
    #     self.date_cb.currentTextChanged.connect(self.date_cb_changed)
    #     self.date_cb.setFixedSize(1400,50)

    #     #container_layout.addWidget(self.btn_home)
    #     #container_layout.addWidget(self.date_cb)
    #     container.setLayout(container_layout)
    #     return container

    def getControl3d(self):
        #container = QWidget()
        #container_layout = QHBoxLayout()

        # 3D 軌跡圖
        self.widget_court = Trajectory3DVisualizeWidget()
        #self.widget_court.setMinimumSize(800, 800)
        #self.widget_court.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        #self.nextbtn = QPushButton(self)
        #self.nextbtn.clicked.connect(self.onClickNext)
        #self.nextbtn.setFixedSize(QSize(100,160))
        #self.nextbtn.setIcon(QIcon(f"{ICONDIR}/right_arrow.png"))
        #self.nextbtn.setIconSize(QSize(95,95))

        # change the court to show next shot
        
        #self.lastbtn = QPushButton(self)
        #self.lastbtn.clicked.connect(self.onClickLast)
        #self.lastbtn.setFixedSize(QSize(100,160))
        #self.lastbtn.setIcon(QIcon(f"{ICONDIR}/left_arrow.png"))
        #self.lastbtn.setIconSize(QSize(95,95))
        
        #container_layout.addWidget(self.lastbtn)
        #container_layout.addWidget(self.widget_court)
        #container_layout.addWidget(self.nextbtn)
        #container.setLayout(container_layout)
        return self.widget_court

    # def getTable(self):
    #     table = QTableWidget(len(self.ball_type), 2)
    #     i = 0
    #     for k in self.ball_type:
    #         table.setItem(i, 0, QTableWidgetItem(k))
    #         table.setItem(i, 1, QTableWidgetItem(str(0)))
    #         i += 1
    #     return table

    def showTrack(self):
        total_tracks = 0
        # self.widget_court.clearAll()
        for track in self.tracks:
            #print(f"Drawing track: {track}")
            for point in track:
                #print(f"Adding point at X: {point.x}, Y: {point.y}, Z: {point.z}, Event: {point.event}")
                if point.event == 1:
                    point.color = 'blue'
                if point.event == 2:
                    point.color = 'red'
                    #total_tracks += 1 #(這邊還沒做事件偵測，先都通時呈現不分段)
                if point.event == 3:
                    point.color = 'green'
                    #total_tracks += 1 #(理論上這段是雜訊)
                #if point.event == 0:
                    #point.color = 'green'
                    #total_tracks += 1
                self.widget_court.addPointByTrackID(point, total_tracks)
                #print("showTrack is called")
                # 確保畫面重繪
                #self.update()

    def readPoints(self, tracks_file):
        #print("Reading points from file...")
        points = []
        with open(tracks_file, 'r', newline='') as csvFile:
            rows = csv.DictReader(csvFile)
            eventFlag = False
            for row in rows:
                if int(float(row['Event'])) == 1:
                    eventFlag = True
                if int(float(row['Visibility'])) != 0 and eventFlag:
                    point = Point(fid=row['Frame'], timestamp=row['Timestamp'], visibility=row['Visibility'],
                                    x=row['X'], y=row['Y'], z=row['Z'],
                                    event=row['Event'])
                    points.append(point)
        self.tracks.append(points)
        #print(f"Tracks after reading points: {self.tracks}")

    # def showBallType(self, ball_type):
    #     if ball_type != []:
    #         for type in ball_type:
    #             self.ball_type[type] += 1
    #     for i in range(self.balltype_tb.rowCount()):
    #         type = self.balltype_tb.item(i, 0).text()
    #         count = self.ball_type[type]
    #         self.balltype_tb.setItem(i, 1, QTableWidgetItem(str(count)))

    # def date_cb_changed(self):
    #     if self.date_cb.currentText() == '':
    #         self.widget_court.clearAll()
    #         self.cur_points = 1
    #         self.max_points = 1
    #         for k in self.ball_type:
    #             self.ball_type[k] = 0
    #     else:
    #         self.cur_points = 1
    #         self.max_points = 1
    #         date = self.date_cb.currentText()
    #         #removeOuterPoint(date)
    #         # do something to cut mod to mod_1, mod_2, ...
    #         # cut_csv(date)
    #         files = [f for f in os.listdir(os.path.join(REPLAYDIR, date)) if f.startswith('Model3D_mod_')]
    #         self.max_points = len(files)
    #         self.showTrajectory(date)
    
    def showMultipleTrajectories(self, date_list):
        for date in date_list:
            self.showTrajectory(date)

    def showTrajectory(self, date):
        #print("showTrajectory is called")
        #self.widget_court.clearAll()
        for k in self.ball_type:
            self.ball_type[k] = 0
        self.tracks = []
        #detectEvent(date, self.cur_points)
        #smoothByEvent(date, self.cur_points)
        #ball_type = detectBallTypeByEvent(date, self.cur_points)
        #file = 'Model3D_smooth_gradient' + str(self.cur_points) + '.csv'
        file = 'Model3D_smooth_gradient' + '.csv'
        # file = 'Model3D_mod.csv'
        # file = 'Model3D_event_' + str(self.cur_points) + '.csv'
        track_file = os.path.join(REPLAYDIR, date, file)
        #track_file = os.path.join(date, file)

        #predict trajectory
        #placement_predict(date,file)
        # try:
        self.readPoints(tracks_file = track_file)
        #print(f"Tracks after reading points: {self.tracks}")
        self.showTrack()
        # self.widget_court.setShotText(self.cur_points)
        #self.update()
            #self.showBallType(ball_type = ball_type)
        # except:
        #     logging.warning("[Model3D_smooth] No Csv Files: {}".format(track_file))

    # def backhome(self):
    #     msg = MsgContract(MsgContract.ID.PAGE_CHANGE, value='TrackNetPage')
    #     self.myService.sendMessage(msg)

    # def onClickNext(self):
    #     self.cur_points += 1
    #     if self.cur_points > self.max_points:
    #         self.cur_points = 1
    #     #date = self.date_cb.currentText()
    #     date="2024-05-27_16-08-44"
    #     self.showTrajectory(date)

    # def onClickLast(self):
    #     self.cur_points -= 1
    #     if self.cur_points <= 0:
    #         self.cur_points = self.max_points
    #     #date = self.date_cb.currentText()
    #     date="2024-05-27_16-08-44"
    #     self.showTrajectory(date)

# def getTrackDate(date):
#     REPLAYDIR = f"/home/nol/demo/NOL_Playground/replay"
#     date =  REPLAYDIR + "/" + date
#     return date

if __name__ == "__main__":
    import sys
    from PyQt5.QtWidgets import QApplication
    
    print("Starting application...")  # 调试信息
    app = QApplication(sys.argv)
    window = Model3dPage()
    window.show()  # 显示主窗口
    window.adjustSize()  # 根據內容自適應視窗大小
    print("Application running...")
    sys.exit(app.exec_())
