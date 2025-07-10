import os
import sys
import logging
import json
import ast
import numpy as np
import cv2
from functools import partial

from PyQt5.QtWidgets import QGroupBox, QWidget, QHBoxLayout, QVBoxLayout, QPushButton, QComboBox, QLabel, QSpinBox, QCheckBox, QGridLayout
from PyQt5.QtCore import QSize
from PyQt5.QtGui import QPixmap, QPainter, QIcon, QMouseEvent

from LayerCamera.RpcCameraWidget import RpcCameraWidget

from ..UISettings import *
from ..Services import SystemService, MsgContract
from lib.common import loadConfig
from lib.point import Point

from lib.common import ROOTDIR, ICONDIR

STARSIZE = 10

from LayerContent.model3D import MultiCamTriang

# calculate one 3D point from list of 2D coordinates
# modified from Model3D_offline.py
def triangulation(coord_2d:list, camera_indices:'list[int]', camera_widget:RpcCameraWidget):

    # shape : (num_cam, mtx...)
    ks = []
    poses = []          # Only Used by Shao-Ping Method
    eye = []            # Only Used by Shao-Ping Method
    dist = []           # Our Method
    newcameramtx = []   # Our Method
    projection_mat = [] # Our Method

    # collect the specified camera configs
    for idx in camera_indices:

        intrinsic = camera_widget.cameraList[idx].getIntrinsic()
        extrinsic = camera_widget.cameraList[idx].getExtrinsic()

        ks.append(np.array(intrinsic["ks"], np.float32))
        dist.append(np.array(intrinsic["dist"], np.float32))
        newcameramtx.append(np.array(intrinsic["newcameramtx"], np.float32))
        poses.append(np.array(extrinsic["poses"], np.float32))
        eye.append(np.array(extrinsic["eye"], np.float32))
        projection_mat.append(np.array(extrinsic["projection_mat"], np.float32))

    ks = np.stack(ks, axis=0)
    poses = np.stack(poses, axis=0)
    eye = np.stack(eye, axis=0)
    dist = np.stack(dist, axis=0)
    newcameramtx = np.stack(newcameramtx, axis=0)
    projection_mat = np.stack(projection_mat, axis=0)

    points_2d = []
    # transform coordinates to Point type
    for coord in coord_2d:
        point = Point(x=coord[0], y=coord[1], z=0, visibility=1)
        points_2d.append(point)

    # Undistort
    undistort_points_2D = []
    for k in range(len(points_2d)):
        temp = cv2.undistortPoints(points_2d[k].toXY(),
                                    ks[k], #  k = cam_idx
                                    dist[k],
                                    None,
                                    newcameramtx[k]) # shape:(1,num_frame,2), num_frame=1
        temp = temp.reshape(-1,2) # shape:(num_frame,2), num_frame=1
        undistort_points_2D.append(temp)
    undistort_points_2D = np.stack(undistort_points_2D, axis=0) # shape:(num_cam,num_frame,2), num_frame=1

    multiCamTriang = MultiCamTriang(poses, eye, newcameramtx)

    # Triangluation
    if undistort_points_2D.shape[0] >= 2:

        multiCamTriang.setTrack2Ds(undistort_points_2D)
        multiCamTriang.setProjectionMats(projection_mat[range(len(points_2d))])
        track_3D = multiCamTriang.rain_calculate3D() # shape:(num_frame,3), num_frame=1

        point3d = Point(visibility=1,
                        x=track_3D[0][0],
                        y=track_3D[0][1],
                        z=track_3D[0][2],
                        color='white')

        return point3d

    # visibility = 0
    return Point()


class Check3dPage(QGroupBox):
    def __init__(self, cameraWidget: RpcCameraWidget, image_size = QSize(800, 600)):
        super().__init__()

        self.image_size = image_size

        self.cameraWidget = cameraWidget

        # two camera snapshot
        self.no_image = QPixmap(f"{ICONDIR}/no_camera.png").scaled(self.image_size)
        self.snapshots = [self.no_image, self.no_image]

        self.num_cameras = self.cameraWidget.num_cameras
        self.selected_num_cameras = 0

        self.original_sizes = []

        # setup UI
        self.setupUI()

    def hideEvent(self, event):
        logging.debug(f"{self.__class__.__name__}: hided.")

    def showEvent(self, event):
        self.initPageStatus()
        logging.debug(f"{self.__class__.__name__}: shown.")

    def setBackgroundService(self, service:SystemService):
        self.myService = service

    def setupUI(self):
        self.control_bar = self.getControlBarData()
        self.snapshot_block = self.getSnapshot(number=self.num_cameras)

        layout_main = QHBoxLayout()
        layout_main.addWidget(self.control_bar, stretch=2)
        layout_main.addWidget(self.snapshot_block, stretch=8)

        self.setLayout(layout_main)

    def getControlBar(self):
        container = QWidget()
        container_layout = QVBoxLayout()

        self.btn_home = QPushButton()
        self.btn_home.setText('返回')#回首頁
        self.btn_home.setFixedSize(QSize(180, 50))
        self.btn_home.setStyleSheet(UIStyleSheet.ReturnButton)
        self.btn_home.clicked.connect(self.backhome)

        title_label = QLabel()
        title_label.setText("請點選左網柱")
        title_label.setStyleSheet('font: 30px')

        self.btn_rechoose = QPushButton()
        self.btn_rechoose.setText('重置')
        self.btn_rechoose.setFixedSize(QSize(180, 50))
        self.btn_rechoose.setStyleSheet(UIStyleSheet.SelectButton)
        self.btn_rechoose.clicked.connect(self.initPageStatus)

        self.btn_calc3d = QPushButton()
        self.btn_calc3d.setText('計算座標')
        self.btn_calc3d.setFixedSize(QSize(180, 50))
        self.btn_calc3d.setStyleSheet(UIStyleSheet.SelectButton)
        self.btn_calc3d.clicked.connect(self.calc3d)

        self.label_3dcoord_txt = QLabel('3D座標:')
        self.label_3dcoord_txt.setFont(UIFont.SpinBox)
        self.label_3dcoord = QLabel("x =\ny =\nz =")
        self.label_3dcoord.setFont(UIFont.SpinBox)
        # self.btn_rechoose.setStyleSheet(UIStyleSheet.Check3DXYZ)

        container_layout.addWidget(self.btn_home, Qt.AlignCenter)
        container_layout.addWidget(self.btn_calc3d, Qt.AlignCenter)
        container_layout.addWidget(self.btn_rechoose, Qt.AlignCenter)
        container_layout.addWidget(title_label, Qt.AlignCenter)
        container_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        # container_layout.addWidget(self.label_3dcoord_txt, 4, 0, Qt.AlignCenter)
        # container_layout.addWidget(self.label_3dcoord, 5, 0, Qt.AlignCenter)
        container.setLayout(container_layout)
        return container
    
    def getControlBarData(self):
        container = QWidget()
        container.setFixedWidth(300)
        container_layout = QVBoxLayout()
        self.ControlBar = self.getControlBar()

        # title_label = QLabel()
        # title_label.setText("請點選左網柱")
        # title_label.setStyleSheet('font: 30px')
        self.label_3dcoord_txt = QLabel('3D座標:')
        # self.label_3dcoord_txt.setFont(UIFont.SpinBox)
        self.label_3dcoord_txt.setStyleSheet(UIStyleSheet.Check3DXYZ)
        self.label_3dcoord = QLabel("x =\ny =\nz =")
        self.label_3dcoord.setFont(UIFont.SpinBox)

        container_layout.addWidget(self.ControlBar, 1, alignment=Qt.AlignHCenter)
        # container_layout.addWidget(title_label, alignment=Qt.AlignmentFlag.AlignCenter)
        container_layout.addWidget(self.label_3dcoord_txt, alignment=Qt.AlignmentFlag.AlignCenter)
        container_layout.addWidget(self.label_3dcoord, alignment=Qt.AlignmentFlag.AlignCenter)
        container.setLayout(container_layout)
        return container

    def getSnapshot(self, number=2):
        container = QWidget()
        container_layout = QGridLayout()

        self.snapshot_list = []
        self.label_2dcoord_list = []
        self.container_2dcoord_list = []
        for i in range(number):

            # information block
            container_2dcoord = QWidget()
            container_2dcoord_layout = QHBoxLayout()

            label_2dcoord_txt = QLabel("2D座標:")
            label_2dcoord_txt.setStyleSheet(UIStyleSheet.Check3DLabel)
            label_2dcoord = QLabel("()")
            label_2dcoord.setStyleSheet(UIStyleSheet.Check3DLabel)
            self.label_2dcoord_list.append(label_2dcoord)

            container_2dcoord_layout.addWidget(label_2dcoord_txt)
            container_2dcoord_layout.addWidget(label_2dcoord)
            container_2dcoord.setLayout(container_2dcoord_layout)
            self.container_2dcoord_list.append(container_2dcoord)

            # snapshot image
            tmp_snapshot = QLabel()
            tmp_snapshot.mousePressEvent = partial(self.court2DPressEvent, i)
            self.snapshot_list.append(tmp_snapshot)
            container_layout.addWidget(tmp_snapshot, int(i/4)*2, int(i%4), alignment=Qt.AlignCenter)
            container_layout.addWidget(container_2dcoord, int(i/4)*2+1, int(i%4), alignment=Qt.AlignCenter)
        container.setLayout(container_layout)
        return container

    def initPageStatus(self):
        self.label_3dcoord.setText("x =\ny =\nz =")

        if self.selected_num_cameras == 1:
            self.image_size = QSize(1280, 720)
        elif self.selected_num_cameras == 2:
            self.image_size = QSize(750, 600)
        else:
            self.image_size = QSize(320, 240)

        for i in range(self.selected_num_cameras):
            self.snapshots[i] = self.snapshots[i].scaled(self.image_size)

        for i in range(self.selected_num_cameras):
            self.snapshot_list[i].setPixmap(self.snapshots[i])
            self.snapshot_list[i].mousePressEvent = partial(self.court2DPressEvent, i)
            self.snapshot_list[i].show()

            self.label_2dcoord_list[i].setText("()")
            self.container_2dcoord_list[i].show()

        for i in range(self.selected_num_cameras, self.num_cameras):
            self.snapshot_list[i].hide()
            self.container_2dcoord_list[i].hide()

    def backhome(self):
        msg = MsgContract(MsgContract.ID.PAGE_CHANGE, value='CameraSystem')
        self.myService.sendMessage(msg)

    def setPixmap(self, images):
        self.snapshots = images

    def setData(self, data):
        self.snapshots = data['Pixmaps']
        self.selected_num_cameras = int(data['SelectedCameras'])
        self.original_sizes = data['Original_sizes']

    def court2DPressEvent(self, index, event):
        if event.buttons() == Qt.LeftButton:

            if self.cameraWidget.cameraList[index] is None:
                return

            mouse_x = event.pos().x()
            mouse_y = event.pos().y()

            resize_ratio_x = self.original_sizes[index].width() / self.snapshots[index].width()
            resize_ratio_y = self.original_sizes[index].height() / self.snapshots[index].height()

            # 算回原始解析度
            # 用 index 左倒右倒
            direction = self.cameraWidget.cameraList[index].direction

            if direction == 1:
                original_x = round(mouse_y * resize_ratio_y)
                original_y = self.original_sizes[index].width()-round(mouse_x * resize_ratio_x)
            elif direction == 2:
                original_x = self.original_sizes[index].width()-round(mouse_x * resize_ratio_x)
                original_y = self.original_sizes[index].width()-round(mouse_y * resize_ratio_y)
            elif direction == 3:
                original_x = self.original_sizes[index].height()-round(mouse_y * resize_ratio_y)
                original_y = round(mouse_x * resize_ratio_x)
            else:
                original_x = round(mouse_x * resize_ratio_x)
                original_y = round(mouse_y * resize_ratio_y)

            self.label_2dcoord_list[index].setText(f"({original_x}, {original_y})")

            # paint star on image
            qpixmap = self.snapshots[index].copy()
            painter = QPainter()
            painter.begin(qpixmap)
            painter.drawPixmap(mouse_x - STARSIZE // 2, mouse_y - STARSIZE // 2, \
                    QPixmap(f"{ICONDIR}/court/star.png").scaled(QSize(STARSIZE, STARSIZE)))
            painter.end()
            self.snapshot_list[index].setPixmap(qpixmap)

    def calc3d(self):
        # check number of clicked images >= 2
        clicked_cnt = 0
        for i in range(self.selected_num_cameras):
            if self.label_2dcoord_list[i].text() != "()":
                clicked_cnt += 1

        if clicked_cnt < 2:
            self.label_3dcoord.setText("<font color=red>至少2個畫面</font>")
            return

        # transform from text to list
        coord_list = []
        camera_indices = []
        for i in range(self.selected_num_cameras):
            if self.label_2dcoord_list[i].text() != "()":
                coord_str = self.label_2dcoord_list[i].text()
                coord_tuple = ast.literal_eval(coord_str)
                coord_list.append([float(coord_tuple[0]), float(coord_tuple[1])])
                camera_indices.append(i)

        point3d = triangulation(coord_list, camera_indices, self.cameraWidget)

        if point3d.visibility == 0:
            self.label_3dcoord.setText("<font color=red>計算失敗</font>")
        else:
            self.label_3dcoord.setText(f"x = {point3d.x:.4f}\ny = {point3d.y:.4f}\nz = {point3d.z:.4f}")