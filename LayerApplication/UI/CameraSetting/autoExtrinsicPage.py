import os
import json
import logging
import numpy as np
import time
import re
import subprocess
import pickle
import math
from functools import partial

from PyQt5.QtWidgets import QGroupBox, QVBoxLayout, QWidget, QHBoxLayout, QPushButton, QLabel, QGridLayout, QCheckBox
from PyQt5.QtCore import QSize
from PyQt5.QtGui import QPixmap, QPainter
from PyQt5.Qt import *

from UISettings import *
from nodes import CameraReader
from Services import SystemService, MsgContract
from message import *
from common import *
from h2pose.H2Pose import H2Pose
from h2pose.Hfinder2 import Hfinder2
from auto.corner import corner
from auto.cornerDL import cornerDL
from auto.mapping import mapping
from auto.court_mapping import court_mapping
from auto.points_after_map import points_mapping

DIRNAME = os.path.dirname(os.path.abspath(__file__))
ROOTDIR = os.path.dirname(os.path.dirname(DIRNAME))
ICONDIR = f"{ROOTDIR}/UI/icon"

CORNER_SIZE = 15
STAR_SIZE = 20
CORNERS_ZOOM_X = [17, 47, 200, 350, 381]
CORNERS_ZOOM_Y = [20, 70, 305, 548, 784, 831]
CORNERS_TYPE = [8, 2, 2, 2, 2, 7,   3, 1, 1, 1, 1, 5,   3, 1, 5, 3, 1, 5,   3, 1, 1, 1, 1, 5,   9, 4, 4, 4, 4, 6]
CORNERS_2D_X = [14-CORNER_SIZE, 20, 44-CORNER_SIZE, 50, 197-CORNER_SIZE, 203, 347-CORNER_SIZE, 353, 378-CORNER_SIZE, 384]
CORNERS_2D_Y = [17-CORNER_SIZE, 23, 67-CORNER_SIZE, 73, 302-CORNER_SIZE, 308, 545-CORNER_SIZE, 551, 781-CORNER_SIZE, 787, 828-CORNER_SIZE, 834]
# CORNERS_3D_X = [-3.05, -3.01, -2.59, -2.55, -0.02, 0.02, 2.55, 2.59, 3.01, 3.05] # corner position in reality (unit: meter)
CORNERS_3D_X = [-1.159, -1.1438, -0.9842, -0.969, -0.0076, 0.0076, 0.969, 0.9842, 1.1438, 1.159]
# CORNERS_3D_Y = [ 6.7, 6.66, 5.94, 5.9, 2.02, 1.98, -1.98, -2.02, -5.9, -5.94, -6.66, -6.7]
CORNERS_3D_Y = [2.546, 2.5308, 2.2572, 2.242, 0.7676, 0.7524, -0.7524, -0.7676, -2.242, -2.2572, -2.5308, -2.546]
COURT_IMAGE = f"{ICONDIR}/court/std_court.png"
STAR_IMAGE = f"{ICONDIR}/court/star.png"
RESIZE_RATIO_W = 1440/800
RESIZE_RATIO_H = 1080/600

IMG_PATH = f"{ROOTDIR}/lib/auto/demo.jpg"  #img output
YOLO_TXT_PATH = f"{ROOTDIR}/lib/auto/yolo_output/exp/labels/demo.txt" #yolo output
COURT_PATH = f"{ROOTDIR}/lib/auto/court.pickle" #court output
WEIGHT_PATH = f"{ROOTDIR}/lib/auto/weight/model_weights_2023-12-02_20-23-43.h5" #weight
IMG_FOLDER = f"{ROOTDIR}/lib/auto/cropped_images" #cropped image file


def calculateExtrinsic(camera:CameraReader, court2D, court3D):
    config_file = f"{ROOTDIR}/Reader/{camera.brand}/config/{camera.hw_id}.cfg"
    cameraCfg = loadConfig(config_file)
    camera_ks = np.array(json.loads(cameraCfg['Other']['ks']))
    dist = np.array(json.loads(cameraCfg['Other']['dist']))
    nmtx = np.array(json.loads(cameraCfg['Other']['newcameramtx']))

    hf = Hfinder2(camera_ks=camera_ks, dist=dist, nmtx=nmtx, court2D=court2D, court3D=court3D)
    Hmtx = hf.getH()
    Kmtx = nmtx
    projection_mat = hf.getProjection_mat()
    extrinsic_mat = hf.getExtrinsic_mat()
    h2p = H2Pose(Kmtx, Hmtx)
    poses = h2p.getC2W()
    eye = h2p.getCamera().T
    eye[0][2] = abs(eye[0][2])
    # [TODO] Check if keys not in [Other] Section
    cameraCfg['Other']['poses'] = str(poses.tolist())
    cameraCfg['Other']['eye'] = str(eye.tolist())
    cameraCfg['Other']['hmtx'] = str(Hmtx.tolist())
    cameraCfg['Other']['projection_mat'] = str(projection_mat.tolist())
    cameraCfg['Other']['extrinsic_mat'] = str(extrinsic_mat.tolist())
    # msg = ID.SAVE_DEVICE_CONFIG
    logging.info('Poses:\n{}\n'.format(poses))
    logging.info('Eye:\n{}\n'.format(eye))
    logging.info('Hmtx:\n{}\n'.format(Hmtx))
    logging.info('projection_mat:\n{}\n'.format(projection_mat))
    logging.info('extrinsic_mat:\n{}\n'.format(extrinsic_mat))
    saveConfig(config_file, cameraCfg)

def getExtrinsic(camera:CameraReader):
    config_file = f"{ROOTDIR}/Reader/{camera.brand}/config/{camera.hw_id}.cfg"
    cameraCfg = loadConfig(config_file)
    ks = np.array(json.loads(cameraCfg['Other']['ks']))
    extrinsic_mat = np.array(json.loads(cameraCfg['Other']['extrinsic_mat']))
    return ks,extrinsic_mat
    
def getIntrinsic(camera:CameraReader):
    config_file = f"{ROOTDIR}/Reader/{camera.brand}/config/{camera.hw_id}.cfg"
    cameraCfg = loadConfig(config_file)
    ks = np.array(json.loads(cameraCfg['Other']['ks']))
    dist = np.array(json.loads(cameraCfg['Other']['dist']))
    newcameramtx = np.array(json.loads(cameraCfg['Other']['newcameramtx']))
    return ks,dist,newcameramtx



def QImageToCvMat(incomingImage):

    incomingImage = incomingImage.convertToFormat(QImage.Format_RGB32)
    width = incomingImage.width()
    height = incomingImage.height()
    ptr = incomingImage.bits()
    ptr.setsize(height * width * 4)
    arr = np.array(ptr, np.uint8).reshape((height, width, 4))
    return cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)

def CvMatToQPixmap(incomingImage):
    height, width, channel = incomingImage.shape
    bytesPerLine = 3 * width
    qImg = QImage(incomingImage.data, width, height, bytesPerLine, QImage.Format_RGB888)
    return QPixmap.fromImage(qImg)
        
def undistort_image(image, ks, dist, newcameramtx):
    h, w = image.shape[:2]
    
    undistorted_image = cv2.undistort(image,ks, dist, None, newcameramtx)
    return undistorted_image

class ToolWindow(QWidget):
    def __init__(self, on_add_callback, parent=None):
        super(ToolWindow, self).__init__(parent)
        self.on_add_callback = on_add_callback
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle('增加2d點')
        self.setGeometry(100, 100, 300, 100)
        layout = QVBoxLayout(self)

        label = QLabel('輸入index:', self)
        layout.addWidget(label)

        self.index_input = QLineEdit(self)
        layout.addWidget(self.index_input)

        add_button = QPushButton('確認', self)
        add_button.clicked.connect(self.on_confirm)
        layout.addWidget(add_button)

    def on_confirm(self):
        index = int(self.index_input.text())
        self.on_add_callback(index)  
        self.close()
        
class ExtrinsicPage(QGroupBox):
    def __init__(self, cameras:list, big_image_size = QSize(1000, 750)):
        super().__init__()

        self.big_image_size = big_image_size
        self.small_image_size = QSize(480, 360)
        self.cameras = cameras

        # two camera snapshot
        self.no_image = QPixmap(f"{ICONDIR}/no_camera.png").scaled(self.big_image_size)
        self.snapshots = [self.no_image, self.no_image]

        # setup UI
        self.setupUI()

        self.num_cameras = 4
        self.selected_num_cameras = 0

        self.original_sizes = []
        self.resize_ratio_x = 1
        self.resize_ratio_y = 1
        
        self.crop_x = 0  
        self.crop_y = 0  
        self.current_scale_factor = 1  
        
        self.index = 0
        
        #up(0)/down(1) court
        self.court_type = 0
        self.nonauto = 0
        
        self.dragFlag = 0

    def hideEvent(self, event):
        logging.debug(f"{self.__class__.__name__}: hided.")

    def showEvent(self, event):
        self.clearManualExtrinsic()
        for i in range(self.selected_num_cameras):
            self.snapshot_list[i].setPixmap(self.snapshots[i].scaled(self.small_image_size))
        
        for i in range(self.selected_num_cameras):
            self.snapshot_list[i].show()
        for i in range(self.selected_num_cameras, self.num_cameras):
            self.snapshot_list[i].hide()
        
        #self.gb_manual_extrinsic.hide()

        logging.debug(f"{self.__class__.__name__}: shown.")

    def setBackgroundService(self, service:SystemService):
        self.myService = service

    def setupUI(self):
        self.control_bar = self.getControlBar()
        self.snapshot_block = self.getSnapshot(number=len(self.cameras))
        self.gb_manual_extrinsic = self.setupManualExtrinsicPage()
        #self.gb_manual_extrinsic.hide()

        left_container = QWidget()
        left_layout = QVBoxLayout()
        left_layout.addWidget(self.control_bar, stretch=1, alignment=(Qt.AlignVCenter | Qt.AlignLeft))
        left_layout.addWidget(self.snapshot_block, stretch=9)
        left_container.setLayout(left_layout)

        layout_main = QHBoxLayout()
        layout_main.addWidget(left_container, stretch=7)
        layout_main.addWidget(self.gb_manual_extrinsic, stretch=3)

        self.setLayout(layout_main)

    def getControlBar(self):
        container = QWidget()
        container_layout = QHBoxLayout()

        self.btn_home = QPushButton()
        self.btn_home.setText('返回')#回首頁
        self.btn_home.setFixedSize(QSize(180, 50))
        self.btn_home.setStyleSheet(UIStyleSheet.ReturnButton)
        self.btn_home.clicked.connect(self.backhome)

        title_label = QLabel()
        title_label.setText("設定世界座標")
        title_label.setStyleSheet('font: 30px')
        
        #dropdown bar
        self.camera_selector = QComboBox()
        self.camera_selector.addItem("Camera0")
        self.camera_selector.addItem("Camera1")
        self.camera_selector.addItem("Camera2")
        self.camera_selector.addItem("Camera3")
        self.camera_selector.currentIndexChanged.connect(self.onCameraSelected)
        self.camera_selector.setStyleSheet('font: 24px')
        

        container_layout.addWidget(self.btn_home, alignment=(Qt.AlignVCenter | Qt.AlignLeft))
        container_layout.addWidget(title_label, alignment=(Qt.AlignVCenter | Qt.AlignLeft))
        container_layout.addWidget(self.camera_selector, alignment=(Qt.AlignVCenter | Qt.AlignLeft))
        container.setLayout(container_layout)
        return container

    def getSnapshot(self, number=2):
        container = QWidget()
        container_layout = QGridLayout()

        self.snapshot_list = []
        for i in range(number):
            tmp_snapshot = QLabel()
            self.snapshot_list.append(tmp_snapshot)
            y = int(i / 2)
            x = i % 2
            container_layout.addWidget(tmp_snapshot, y * 2, x, alignment=Qt.AlignCenter)

        container.setLayout(container_layout)
        return container

    def backhome(self):
        msg = MsgContract(MsgContract.ID.PAGE_CHANGE, value='CameraSystem')
        self.myService.sendMessage(msg)

    def setPixmap(self, images):
        self.snapshots = images

    def setData(self, data):
        self.snapshots = data['Pixmaps']
        self.selected_num_cameras = int(data['SelectedCameras'])
        self.original_sizes = data['Original_sizes']
        
    def world_to_image(self, point_3d, intrinsic, extrinsic):
      point_3d_homogeneous = np.append(point_3d, 1)
      point_camera = np.dot(extrinsic, point_3d_homogeneous)
      point_image_homogeneous = point_camera
      point_image_homogeneous = np.dot(intrinsic, point_camera)
      x = point_image_homogeneous[0] / point_image_homogeneous[2]
      y = point_image_homogeneous[1] / point_image_homogeneous[2]
      return [x, y]
                  
    def setupCheckExtrinsic(self):
        gb = QGroupBox()
        layout = QGridLayout()
        return gb
    
    def onCameraSelected(self, index):
        self.selected_camera_index = index
        self.selectedCameraEvent(index, None)

    def selectedCameraEvent(self, index, event):
        self.showSelectedCameraImage(index)
        """
        if index in [0, 1]:
            self.drawPointsOnImage(index)
        """
            
        
    def showSelectedCameraImage(self, index):
        self.camera = self.cameras[index]

        for i in range(self.selected_num_cameras):
          if i != index:  
            print('hide',i)
            self.snapshot_list[i].setHidden(True)
          else:
            
            qpixmap = self.snapshots[i]
            qimage = qpixmap.toImage()
            image = QImageToCvMat(qimage)
            ks, dist, newcameramtx = getIntrinsic(self.camera)
            undistorted_image = undistort_image(image, ks, dist, newcameramtx)
            undistorted_qpixmap = CvMatToQPixmap(undistorted_image)
            self.snapshot_list[i].setPixmap(undistorted_qpixmap.scaled(self.big_image_size))
            self.snapshot_list[i].setHidden(False)
            self.snapshot_list[i].mousePressEvent = partial(self.court2DPressEvent, index)
            self.original_pixmap = self.snapshot_list[index].pixmap().copy()
            self.current_scale_factor = 1
            self.last_crop_position = (0, 0)
            
            #pixmap.save(f'{IMG_PATH}', 'JPG')
            pixmap = self.snapshot_list[i].pixmap()
            pixmap.save(f'{IMG_PATH}', 'JPG')
            #self.snapshots[i].save(f'{IMG_PATH}', 'JPG') 
            
        
        
        
        self.gb_manual_extrinsic.show()
        self.resize_ratio_x = self.original_sizes[index].width() / self.big_image_size.width()
        self.resize_ratio_y = self.original_sizes[index].height() / self.big_image_size.height()
        self.index = index          
        print("press ", index)
       
    
    def drawPointsOnImage(self, index):
        ks, extrinsic_mat = getExtrinsic(self.camera)
        qpixmap = self.snapshot_list[index].pixmap()
        painter = QPainter()
        painter.begin(qpixmap)
        pen = QPen()
        pen.setWidth(3)  
        pen.setColor(QColor("red"))  
        painter.setPen(pen)
        
        point3ds = [[-1.159, 2.546], [-1.1438, 2.5308], [-0.9842, 2.5308], [-0.969, 2.5308], [-0.0076, 2.5308], [0.0076, 2.5308], [0.969, 2.5308], [0.9842, 2.5308], [1.159, 2.546], [1.1438, 2.5308], [-1.1438, 2.2572], [-1.1438, 2.242], [-0.9842, 2.2572], [-0.969, 2.2572], [-0.969, 2.242], [-0.9842, 2.242], [-0.0076, 2.2572], [0.0076, 2.2572], [0.0076, 2.242], [-0.0076, 2.242], [0.969, 2.2572], [0.9842, 2.2572], [0.9842, 2.242], [0.969, 2.242], [1.1438, 2.2572], [1.1438, 2.242], [-1.1438, 0.7676], [-1.1438, 0.7524], [-0.9842, 0.7676], [-0.969, 0.7676], [-0.969, 0.7524], [-0.9842, 0.7524], [-0.0076, 0.7676], [0.0076, 0.7676], [0.969, 0.7676], [0.9842, 0.7676], [0.9842, 0.7524], [0.969, 0.7524], [1.1438, 0.7676], [1.1438, 0.7524],[0.0, 0.0]]
        count = 0
        point3ds = [point + [0] for point in point3ds]
        for point3d in point3ds:
          point_2d = self.world_to_image(point3d, ks, extrinsic_mat)
          print(int(point_2d[0]),int(point_2d[1]))
          painter.drawPoint(int(point_2d[0]),int(point_2d[1]))
          count+=1
        
        painter.end()
        self.snapshot_list[index].setPixmap(qpixmap)
        
    def setupManualExtrinsicPage(self):
        gb = QGroupBox()
        layout = QGridLayout()

        # zoom image
        self.zoom_court = QLabel()
        self.zoom_court.setFixedSize(QSize(200, 200))
        self.zoom_court.setStyleSheet("border: 1px solid black;")
        self.zoom_court.mousePressEvent = self.zoomCourt3DPressEvent
        self.zoom_court_shown = False
        # hint image
        hint_image = QLabel()
        hint_image.setFixedSize(QSize(256, 64))
        hint_image.setPixmap(QPixmap(f"{ICONDIR}/court/hint_click_court.png").scaled(QSize(256, 64)))
        # corner coordinate
        self.ql_corner = QLabel()
        self.ql_corner.setStyleSheet(UIStyleSheet.ContentText)
        #checkbox
        self.checkbox1 = QCheckBox('下半場', self)
        self.checkbox1.clicked.connect(self.onCheckBox1Click)
        self.checkbox1.setStyleSheet('QCheckBox {spacing: 5px;font-size:25px;} QCheckBox::indicator {width:  33px;height: 33px;}')
        #checkbox
        self.checkbox2 = QCheckBox('手動調整', self)
        self.checkbox2.clicked.connect(self.onCheckBox2Click)
        self.checkbox2.setStyleSheet('QCheckBox {spacing: 5px;font-size:25px;} QCheckBox::indicator {width:  33px;height: 33px;}')
        #detect_button
        self.detect_button = QPushButton("自動偵測", self)
        self.detect_button.clicked.connect(self.detectFunction)
        self.detect_button.setFixedSize(UIComponentSize.ConfirmButton)
        self.detect_button.setStyleSheet('font: 24px')

        # set court image
        self.court_image = QLabel()
        self.court_image.setFixedSize(QSize(400, 852))
        self.court_image.mousePressEvent = self.court3DPressEvent
        self.corners_3D = []
        self.corners_2D = []
        self.drawIdxOnCourt()
        # clear Button
        btn_clear = QPushButton()
        btn_clear.setText("重設")
        btn_clear.clicked.connect(self.clearManualExtrinsic)
        btn_clear.setStyleSheet('font: 24px')
        # calculate button
        btn_calculate = QPushButton()
        btn_calculate.setText("確定")
        btn_calculate.clicked.connect(self.calculateExtrinsicMtx)
        btn_calculate.setStyleSheet('font: 24px')

        layout.addWidget(self.ql_corner, 0, 4, 1, 1, Qt.AlignCenter)
        #layout.addWidget(hint_image, 1, 4, 1, 2, Qt.AlignCenter)
        #layout.addWidget(self.zoom_court, 1, 4, 2, 2, Qt.AlignCenter)
        layout.addWidget(self.court_image, 0, 0, 4, 3, Qt.AlignCenter)
        layout.addWidget(self.checkbox1, 1, 4, 1, 2, Qt.AlignTop)
        layout.addWidget(self.checkbox2, 2, 4, 1, 2, Qt.AlignTop)
        layout.addWidget(self.detect_button, 3, 4, 1, 2, Qt.AlignTop)
        layout.addWidget(btn_clear, 5, 4, 1, 2, Qt.AlignTop)
        layout.addWidget(btn_calculate, 6, 4, 1, 2, Qt.AlignTop)
        
        gb.setLayout(layout)
        return gb

    def onCheckBox1Click(self):
        if self.checkbox1.isChecked():
            self.court_type = 1
            print('court_type = 1')
        else:
            self.court_type = 0
            print('court_type = 0')
    def onCheckBox2Click(self):
        if self.checkbox2.isChecked():
            self.nonauto = 1
            print('nonauto = 1')
        else:
            self.nonauto = 0
            print('nonauto = 0')
            
    def detectFunction(self):
        os.chdir(f'{ROOTDIR}/lib/auto/yolov7')
        subprocess.run(['python3', 'detect.py', '--weights', 'best.pt', '--source', f'{IMG_PATH}', '--save-txt', '--project', f'../yolo_output', '--name', 'exp', '--exist-ok'])
        
        MAPPING_TXT_PATH = YOLO_TXT_PATH[:-4] + "_mapping.txt"
        AFTER_MAPPING_TXT_PATH = YOLO_TXT_PATH[:-4] + "_mapping_after.txt"
        mapping(IMG_PATH, YOLO_TXT_PATH)
        court_mapping(IMG_PATH, MAPPING_TXT_PATH, COURT_PATH)
        points_mapping(IMG_PATH, MAPPING_TXT_PATH, COURT_PATH)
        intersection_path = cornerDL(IMG_PATH, AFTER_MAPPING_TXT_PATH, WEIGHT_PATH, IMG_FOLDER)
        
        with open(intersection_path, 'r') as fl:
            count = 0 
            for string in fl:
                points = re.findall(r'\((\d+), (\d+)\)', string)
                
                if count in [10, 11, 17, 18]:
                    for point in points:
                        x, y = map(int, point)
                        if [x, y] not in self.corners_2D:
                            self.corners_2D.append([x/1440*1000, y/1080*750])
                
                else:
                    for point in points:
                        x, y = map(int, point)
                        if [x, y] not in self.corners_2D:
                            self.corners_2D.append([x/1440*1000, y/1080*750])
                count += 1
        
        qpixmap = self.snapshot_list[self.index].pixmap()  
        painter = QPainter()
        painter.begin(qpixmap)
        for idx, point in enumerate(self.corners_2D):
            painter.drawPixmap(point[0], point[1], QPixmap(f"{ICONDIR}/court/point{idx+1}.png").scaled(QSize(CORNER_SIZE, CORNER_SIZE)))
        painter.end()
        self.snapshot_list[self.index].setPixmap(qpixmap)
    
        directories_to_clear = [f"{ROOTDIR}/lib/auto/yolo_output/exp/labels", f"{ROOTDIR}/lib/auto/yolo_output/cropped_images/"]

        for directory in directories_to_clear:
            for file in os.scandir(directory):
                os.remove(file.path)     
    
    def zoomCourt3DPressEvent(self, event):
        pass
        
    def drawIdxOnCourt(self):
        qpixmap = QPixmap(COURT_IMAGE).scaled(QSize(400, 852))
        painter = QPainter()
        painter.begin(qpixmap)

        # draw stars
        for x in CORNERS_ZOOM_X:
            for y in CORNERS_ZOOM_Y:
                painter.drawPixmap(x - STAR_SIZE // 2, y - STAR_SIZE // 2, \
                            QPixmap(STAR_IMAGE).scaled(QSize(STAR_SIZE, STAR_SIZE)))

        # draw points
        for idx, point in enumerate(self.corners_3D):
            point_x_idx = point[0]
            point_y_idx = point[1]
            painter.drawPixmap(CORNERS_2D_X[point_x_idx], CORNERS_2D_Y[point_y_idx],  QPixmap(f"{ICONDIR}/court/point{idx+1}.png").scaled(QSize(CORNER_SIZE, CORNER_SIZE)))
        painter.end()
        self.court_image.setPixmap(qpixmap)

    # Court 3D coordinate Manual setup
    def court3DPressEvent(self, event):
        if(self.nonauto == 0):
          if event.buttons() == Qt.LeftButton:
              # Read court map
              with open(COURT_PATH, "rb") as file:
                  court_data = pickle.load(file)
                  print('court_data')
                  print(court_data)

              if(self.court_type == 0): #upcourt
                self.corners_3D = []
                zero = [9,0],[8,1]
                one = [0,0],[1,1]
                two = [4,1],[5,1]
                three =  [4,2],[5,2],[5,3],[4,3]
                four =  [4,4],[5,4]
                five = [2,1],[3,1]
                six = [1,2],[1,3]
                seven = [2,2],[3,2],[3,3],[2,3]
                eight = [1,4],[1,5]
                nine = [2,4],[3,4],[3,5],[2,5]
                #ten =
                #eleven =
                twelve = [6,1],[7,1]
                thirdteen = [6,2],[7,2],[7,3],[6,3]
                fourteen = [8,2],[8,3]
                fifteen = [6,4],[7,4],[7,5],[6,5]
                sixteen = [8,4],[8,5]
                #seventeen =
                #eighteen =
                merged_list = []
                for i in range(len(court_data)):
                  for j in range(len(court_data[0])):
                      if(court_data[i][j]!=[0,0]):
                          if(i==0 and j == 4):
                              merged_list.extend(zero)
                          if(i==0 and j == 0):
                              merged_list.extend(one)
                          if(i==0 and j == 2):
                              merged_list.extend(two)
                          if(i==1 and j == 2):
                              merged_list.extend(three)
                          if(i==2 and j == 2):
                              merged_list.extend(four)
                          if(i==0 and j == 1):
                              merged_list.extend(five)
                          if(i==1 and j == 0):
                              merged_list.extend(six)
                          if(i==1 and j == 1):
                              merged_list.extend(seven)
                          if(i==2 and j == 0):
                              merged_list.extend(eight)
                          if(i==2 and j == 1):
                              merged_list.extend(nine)
                          if(i==0 and j == 3):
                              merged_list.extend(twelve)
                          if(i==1 and j == 3):
                              merged_list.extend(thirdteen)
                          if(i==1 and j == 4):
                              merged_list.extend(fourteen)
                          if(i==2 and j == 3):
                              merged_list.extend(fifteen)
                          if(i==2 and j == 4):
                              merged_list.extend(sixteen)
                      print(merged_list)
                self.corners_3D = merged_list
              elif(self.court_type == 1): #downcourt
                self.corners_3D = []
                zero = [0,11],[1,10]
                one = [9,11],[8,10]
                two = [5,10],[4,10]
                three =  [5,9],[4,9],[4,8],[5,8]
                four =  [5,7],[4,7]
                five = [7,10],[6,10]
                six = [8,9],[8,8]
                seven = [7,9],[6,9],[6,8],[7,8]
                eight = [8,7],[8,6]
                nine = [7,7],[6,7],[6,6],[7,6]
                #ten =
                #eleven =
                twelve = [3,10],[2,10]
                thirdteen = [3,9],[2,9],[2,8],[3,8]
                fourteen = [1,9],[1,8]
                fifteen = [3,7],[2,7],[2,6],[3,6]
                sixteen = [1,7],[1,6]
                #seventeen =
                #eighteen =
                merged_list = []
                for i in range(len(court_data)):
                  for j in range(len(court_data[0])):
                      if(court_data[i][j]!=[0,0]):
                          if(i==0 and j == 4):
                              merged_list.extend(zero)
                          if(i==0 and j == 0):
                              merged_list.extend(one)
                          if(i==0 and j == 2):
                              merged_list.extend(two)
                          if(i==1 and j == 2):
                              merged_list.extend(three)
                          if(i==2 and j == 2):
                              merged_list.extend(four)
                          if(i==0 and j == 1):
                              merged_list.extend(five)
                          if(i==1 and j == 0):
                              merged_list.extend(six)
                          if(i==1 and j == 1):
                              merged_list.extend(seven)
                          if(i==2 and j == 0):
                              merged_list.extend(eight)
                          if(i==2 and j == 1):
                              merged_list.extend(nine)
                          if(i==0 and j == 3):
                              merged_list.extend(twelve)
                          if(i==1 and j == 3):
                              merged_list.extend(thirdteen)
                          if(i==1 and j == 4):
                              merged_list.extend(fourteen)
                          if(i==2 and j == 3):
                              merged_list.extend(fifteen)
                          if(i==2 and j == 4):
                              merged_list.extend(sixteen)
                      print(merged_list)
                self.corners_3D = merged_list

        else:
            merged_list = self.corners_3D
            print(merged_list)
            if event.buttons() == Qt.RightButton:
              mouse_x = event.pos().x()
              mouse_y = event.pos().y()
              x_array = np.asarray(CORNERS_ZOOM_X)
              y_array = np.asarray(CORNERS_ZOOM_Y)
              self.zoom_x_idx = (np.abs(x_array-mouse_x)).argmin()
              self.zoom_y_idx = (np.abs(y_array-mouse_y)).argmin()
              corner_idx = self.zoom_x_idx*len(CORNERS_ZOOM_Y) + self.zoom_y_idx
              corner_type = CORNERS_TYPE[corner_idx]
              if(self.court_type == 0):
                zero = [9,0],[8,1]
                one = [0,0],[1,1]
                two = [4,1],[5,1]
                three =  [4,2],[5,2],[5,3],[4,3]
                four =  [4,4],[5,4]
                five = [2,1],[3,1]
                six = [1,2],[1,3]
                seven = [2,2],[3,2],[3,3],[2,3]
                eight = [1,4],[1,5]
                nine = [2,4],[3,4],[3,5],[2,5]
                twelve = [6,1],[7,1]
                thirdteen = [6,2],[7,2],[7,3],[6,3]
                fourteen = [8,2],[8,3]
                fifteen = [6,4],[7,4],[7,5],[6,5]
                sixteen = [8,4],[8,5]
                zoom_dict = {
                  (0, 0): one,
                  (1, 0): five,
                  (2, 0): two,
                  (3, 0): twelve,
                  (4, 0): zero,
                  (0, 1): six,
                  (1, 1): seven,
                  (2, 1): three,
                  (3, 1): thirdteen,
                  (4, 1): fourteen,
                  (0, 2): eight,
                  (1, 2): nine,
                  (2, 2): four,
                  (3, 2): fifteen,
                  (4, 2): sixteen,
                }
              else:
                zero = [0,11],[1,10]
                one = [9,11],[8,10]
                two = [5,10],[4,10]
                three =  [5,9],[4,9],[4,8],[5,8]
                four =  [5,7],[4,7]
                five = [7,10],[6,10]
                six = [8,9],[8,8]
                seven = [7,9],[6,9],[6,8],[7,8]
                eight = [8,7],[8,6]
                nine = [7,7],[6,7],[6,6],[7,6]
                twelve = [3,10],[2,10]
                thirdteen = [3,9],[2,9],[2,8],[3,8]
                fourteen = [1,9],[1,8]
                fifteen = [3,7],[2,7],[2,6],[3,6]
                sixteen = [1,7],[1,6]
                zoom_dict = {
                  (4, 5): one,
                  (3, 5): five,
                  (2, 5): two,
                  (1, 5): twelve,
                  (0, 5): zero,
                  (4, 4): six,
                  (3, 4): seven,
                  (2, 4): three,
                  (1, 4): thirdteen,
                  (0, 4): fourteen,
                  (4, 3): eight,
                  (3, 3): nine,
                  (2, 3): four,
                  (1, 3): fifteen,
                  (0, 3): sixteen,
                }
              key = (self.zoom_x_idx, self.zoom_y_idx)
              if key in zoom_dict:
                merged_list = [i for i in merged_list if i not in zoom_dict[key]]
              self.corners_3D = merged_list

        self.zoom_court_shown = False
        self.drawIdxOnCourt()

    def calculateExtrinsicMtx(self):
        if len(self.corners_2D) != len(self.corners_3D) or len(self.corners_3D) < 4:
            self.ql_corner.setText("<font color=red>????(???????4????)</font>")
            print(len(self.corners_2D) )
            print(len(self.corners_3D) )
            return
        

        court3D = []
        court2D = []
        court2D_tmp = self.corners_2D.copy()
        for point in court2D_tmp:
            point_x = point[0] * self.resize_ratio_x # * RESIZE_RATIO_W
            point_y = point[1] * self.resize_ratio_y # * RESIZE_RATIO_H
            court2D.append([point_x, point_y])
        for point_idx in self.corners_3D:
            actual_x_idx = point_idx[0]
            actual_y_idx = point_idx[1]
            point = [CORNERS_3D_X[actual_x_idx], CORNERS_3D_Y[actual_y_idx]]
            court3D.append(point)
        calculateExtrinsic(self.camera, court2D=court2D, court3D=court3D)
        self.ql_corner.setText("<font color=blue>設定成功</font>")
        self.clearManualExtrinsic()
        self.ql_corner.setText("<font color=blue>設定成功</font>")
        for i in range(self.selected_num_cameras):
            self.snapshot_list[i].setPixmap(self.snapshots[i].scaled(self.small_image_size))
            self.snapshot_list[i].setHidden(False)
        #self.gb_manual_extrinsic.hide()

    def clearManualExtrinsic(self):
        self.corners_3D = []
        self.corners_2D = []
        self.drawIdxOnCourt()
            
        for i in range(self.selected_num_cameras):
              self.snapshot_list[i].setPixmap(self.snapshots[i].scaled(self.big_image_size))
        self.dragFlag = 1
        for i in range(self.selected_num_cameras):
          if i != 0:  
            print('hide',i)
            self.snapshot_list[i].setHidden(True)
          else:
            self.snapshot_list[i].setHidden(False)
            self.snapshot_list[i].mousePressEvent = partial(self.court2DPressEvent, 0)
            self.original_pixmap = self.snapshot_list[self.index].pixmap().copy()
            self.current_scale_factor = 1
            self.last_crop_position = (0, 0)    
        
        
        
        self.ql_corner.setText("")
        self.zoom_court_shown = False
        self.zoom_court.clear()
    
    def show_tool_window(self, callback, x, y):
        self.tool_window = ToolWindow(callback, self)
        self.tool_window.show()
        self.pending_x, self.pending_y = x, y  

    def handle_new_corner(self, index):
        
        if 0 <= index < len(self.corners_2D):
            self.corners_2D.insert(index-1, (self.pending_x, self.pending_y))
        else:
            self.corners_2D.append((self.pending_x, self.pending_y))
    
    def court2DPressEvent(self, index, event):
        if(self.nonauto == 0):
            self.crop_x = 0
            self.crop_y = 0
            self.current_crop_image = self.original_pixmap.copy()
            if event.buttons() == Qt.RightButton:
                mouseX, mouseY = event.x(), event.y()

                
                scale_factor_increment = 2  
                self.current_scale_factor *= scale_factor_increment
                qpixmap_scaled = self.original_pixmap.scaled(self.current_scale_factor * self.original_pixmap.width(), 
                                                            self.current_scale_factor * self.original_pixmap.height())

                
                mouseX += self.last_crop_position[0]
                mouseY += self.last_crop_position[1]

                
                crop_width, crop_height = self.original_pixmap.width(), self.original_pixmap.height()
                crop_x = max(0, min(qpixmap_scaled.width() - crop_width, mouseX * scale_factor_increment - crop_width / 2))
                crop_y = max(0, min(qpixmap_scaled.height() - crop_height, mouseY * scale_factor_increment - crop_height / 2))
                self.crop_x = crop_x
                self.crop_y = crop_y
                self.last_crop_position = (crop_x, crop_y)  

                
                cropped_pixmap = qpixmap_scaled.copy(crop_x, crop_y, crop_width, crop_height)
                self.current_crop_image = cropped_pixmap.copy()
                
                painter = QPainter()
                painter.begin(cropped_pixmap)
                for idx, point in enumerate(self.corners_2D):
                    point_x = self.current_scale_factor * point[0] - crop_x
                    point_y = self.current_scale_factor * point[1] - crop_y
                    painter.drawPixmap(point_x, point_y, QPixmap(f"{ICONDIR}/court/point{idx+1}.png").scaled(QSize(CORNER_SIZE, CORNER_SIZE)))
                painter.end()

                
                self.snapshot_list[index].setPixmap(cropped_pixmap)
        
        if(self.nonauto == 1):
            self.crop_x = 0
            self.crop_y = 0
            self.current_crop_image = self.original_pixmap.copy()
            if event.buttons() == Qt.RightButton:
                mouseX = event.pos().x()
                mouseY = event.pos().y()
            
                closest_idx = None
                min_distance = float("inf")
                for idx, point in enumerate(self.corners_2D):
                    point_x = self.current_scale_factor * point[0] - self.last_crop_position[0]
                    point_y = self.current_scale_factor * point[1] - self.last_crop_position[1]
                    distance = math.sqrt((point_x - mouseX)**2 + (point_y - mouseY)**2)
                    if distance < min_distance:
                        min_distance = distance
                        closest_idx = idx
            
                
                if min_distance < 100:
                    del self.corners_2D[closest_idx]  
                    print(f"Point {closest_idx+1} removed")
                    
                qpixmap_copy = self.original_pixmap.copy()
                painter = QPainter()
                painter.begin(qpixmap_copy)
                for idx, point in enumerate(self.corners_2D):
                    point_x, point_y = point
                    painter.drawPixmap(point_x, point_y, QPixmap(f"{ICONDIR}/court/point{idx+1}.png").scaled(QSize(CORNER_SIZE, CORNER_SIZE)))
                painter.end()

                self.current_crop_image = qpixmap_copy.copy()
                self.snapshot_list[index].setPixmap(qpixmap_copy)
                
            elif event.buttons() == Qt.MiddleButton:
                self.current_scale_factor = 1  
                self.last_crop_position = (0, 0)  
                
                
                qpixmap_copy = self.original_pixmap.copy()
                painter = QPainter()
                painter.begin(qpixmap_copy)
                for idx, point in enumerate(self.corners_2D):
                    point_x, point_y = point
                    painter.drawPixmap(point_x, point_y, QPixmap(f"{ICONDIR}/court/point{idx+1}.png").scaled(QSize(CORNER_SIZE, CORNER_SIZE)))
                painter.end()

                self.current_crop_image = qpixmap_copy.copy()
                self.snapshot_list[index].setPixmap(qpixmap_copy)
                
            elif event.buttons() == Qt.LeftButton:
            
                mouseX = event.pos().x()
                mouseY = event.pos().y()
                
                #mouseX = (mouseX + self.crop_x) / self.current_scale_factor
                #mouseY = (mouseY + self.crop_y) / self.current_scale_factor
                print(mouseX,mouseY)

                self.show_tool_window(self.handle_new_corner, mouseX, mouseY)
                
                qpixmap_copy = self.original_pixmap.copy()
                painter = QPainter()
                painter.begin(qpixmap_copy)
                for idx, point in enumerate(self.corners_2D):
                    point_x, point_y = point
                    painter.drawPixmap(point_x, point_y, QPixmap(f"{ICONDIR}/court/point{idx+1}.png").scaled(QSize(CORNER_SIZE, CORNER_SIZE)))
                painter.end()

                self.current_crop_image = qpixmap_copy.copy()
                self.snapshot_list[index].setPixmap(qpixmap_copy)
                
                
                
                