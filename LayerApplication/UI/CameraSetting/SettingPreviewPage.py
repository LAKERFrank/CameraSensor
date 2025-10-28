import logging
import threading
import time
import cv2
import numpy as np
import shutil
import ast

from PyQt5.QtWidgets import QGroupBox, QVBoxLayout, QComboBox, QWidget, QGridLayout, QLabel, QPushButton, QHBoxLayout, QSpinBox, QScrollArea, QDialog, QDoubleSpinBox, QProgressDialog, QCheckBox
from PyQt5.QtCore import QSize, QThread, pyqtSignal, pyqtSlot, QTimer
from PyQt5.QtGui import QPixmap, QImage, QGuiApplication
from PyQt5.Qt import *
from PyQt5.QtMultimedia import QSound

from ..UISettings import *
from ..Services import SystemService, MsgContract

from lib.nodes import CameraReader
from lib.frame import Frame
from lib.message import *
from lib.common import insertById, loadConfig, saveConfig, setIntrinsicMtx, checkAUXorBUX
from lib.Calibration import Calibration
from lib.camera import convert_sample_to_numpy

from lib.common import ROOTDIR, ICONDIR

from LayerCamera.RpcCameraWidget import RpcCameraWidget

class ClickLabel(QLabel):

    clicked = pyqtSignal(int)

    def __init__(self, cam_idx):
        super().__init__()
        self.cam_idx = cam_idx

    def mousePressEvent(self, event):
        self.clicked.emit(self.cam_idx)
        QLabel.mousePressEvent(self, event)

class InteractiveLabel(QLabel):
    def __init__(self, pixmap, parent=None):
        super(InteractiveLabel, self).__init__(parent)
        self.original_pixmap = pixmap
        self.current_scale_factor = 1.0
        self.zoom_count = 0  
        self.setPixmap(self.original_pixmap)
        self.last_crop_position = (0, 0)

    def mousePressEvent(self, event):
        if event.buttons() == Qt.RightButton:
            self.handleZoom(event.pos())

    def handleZoom(self, mouse_pos):
        self.zoom_count += 1  
        if self.zoom_count == 3:
            
            self.current_scale_factor = 1.0
            self.zoom_count = 0  
            self.last_crop_position = (0, 0)
            cropped_pixmap = self.original_pixmap
        else:
            
            scale_factor_increment = 2
            self.current_scale_factor *= scale_factor_increment
            qpixmap_scaled = self.original_pixmap.scaled(self.current_scale_factor * self.original_pixmap.width(), 
                                                         self.current_scale_factor * self.original_pixmap.height())

            mouseX, mouseY = mouse_pos.x(), mouse_pos.y()
            mouseX += self.last_crop_position[0]
            mouseY += self.last_crop_position[1]

            crop_width, crop_height = self.original_pixmap.width(), self.original_pixmap.height()
            crop_x = max(0, min(qpixmap_scaled.width() - crop_width, mouseX * scale_factor_increment - crop_width / 2))
            crop_y = max(0, min(qpixmap_scaled.height() - crop_height, mouseY * scale_factor_increment - crop_height / 2))
            
            self.last_crop_position = (crop_x, crop_y)
            cropped_pixmap = qpixmap_scaled.copy(crop_x, crop_y, crop_width, crop_height)

        self.setPixmap(cropped_pixmap)

class SettingPreviewPage(QGroupBox):
    def __init__(self, cameras:'list[CameraReader]', camera_widget:RpcCameraWidget):
        super().__init__()

        # self.setWindowTitle("選擇相機")
        # self.resize(800, 600)

        self.cameras = cameras
        self.camera_widget = camera_widget

        self.from_page = 'CameraSystem'

        self.cfg_file = f"{ROOTDIR}/config"
        self.cfg = loadConfig(self.cfg_file)

        # setupLogLevel(level=self.cfg['Project']['logging_level'], log_file="UI")

        #self.serials = list(self.camera_widget.getAvailableCamera().keys())
        self.serials = []
        #num_exist_cam = len(self.serials)
        #for i in range(num_exist_cam, 4):
        #    self.serials.append('相機少於4')
        # # print(self.serials)
        self.serials.append('None')

        self.setupUI()

        self.stop_event = threading.Event()


    def hideEvent(self, event):
        logging.debug(f"{self.__class__.__name__}: hided.")
        self.stop_event.set()
        # self.camera_widget.stopStreaming()
        # self.deleteUI()

    def showEvent(self, event):
        logging.debug(f"{self.__class__.__name__}: shown.")
        self.stop_event.clear()
        self.t = threading.Thread(target=self.__updateDaemon, args=(self.stop_event,))
        self.t.setDaemon(True)
        self.t.start()

    def __updateDaemon(self, stop_event):
        while not stop_event.isSet():
            self.__updateImage()
            time.sleep(1)

    def __updateImage(self):
        for i, camera in enumerate(self.cameras):
            image = self.camera_widget.getSnapshotArray(i)
            if image is None:
                self.cams[i].setPixmap(self.no_cam_pixmap)
                continue

            #undistort
            intrinsic = self.camera_widget.cameraList[i].getIntrinsic()
            extrinsic = self.camera_widget.cameraList[i].getExtrinsic()

            direction = self.camera_widget.cameraList[i].direction

            # FIXME: direction
            if direction == 1:
                image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
            elif direction == 2:
                image = cv2.rotate(image, cv2.ROTATE_180)
            elif direction == 3:
                image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)

            distorted = self.checkbox_undistort.isChecked()

            if distorted:
                image = self.undistort(image, intrinsic)
            if self.checkbox_surface.isChecked():
                image = self.drawSurface(image, intrinsic, extrinsic, distorted)
            if self.checkbox_court.isChecked():
                image = self.drawCourt(image, intrinsic, extrinsic, distorted)

            if direction == 1:
                image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
            elif direction == 2:
                image = cv2.rotate(image, cv2.ROTATE_180)
            elif direction == 3:
                image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)

            height, width, channel = image.shape
            bytesPerLine = 3*width
            qimage = QImage(image.data, width, height, bytesPerLine, QImage.Format.Format_BGR888)

            if direction in [1, 3]:
                self.cams[i].setPixmap(QPixmap(qimage).scaled(240, 320))
            else:
                self.cams[i].setPixmap(QPixmap(qimage).scaled(320, 240))

    def setBackgroundService(self, service:SystemService):
        self.myService = service

    def setFromPage(self, page:str):
        print(f'page === {page}')
        if page == None:
            return
        self.from_page = page

    def undistort(self, image, intrinsic):
        # print(image.shape)
        mtx = np.array(intrinsic['ks'], np.float32)
        dist = np.array(intrinsic['dist'], np.float32)

        #h, w = image.shape[:2]

        #newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 0, (w, h))

        # # undistort
        dst = cv2.undistort(image, mtx, dist, None, mtx)

        # # crop the image
        #x, y, w, h = roi
        #dst = dst[y:y+h, x:x+w]

        #img = cv2.resize(dst, (image.shape[1], image.shape[0]))
        #return img
        return dst

    def world_to_image(self, point_3d, intrinsic, extrinsic):
        point_3d_homogeneous = np.append(point_3d, 1)
        point_camera = np.dot(extrinsic, point_3d_homogeneous)
        point_image_homogeneous = point_camera
        point_image_homogeneous = np.dot(intrinsic, point_camera)
        x = point_image_homogeneous[0] / point_image_homogeneous[2]
        y = point_image_homogeneous[1] / point_image_homogeneous[2]
        return [x, y]
    
    def draw_lines_on_image(self, image, points, color=(0, 255, 255), thickness=3):
        for i in range(len(points)):
            for j in range(i + 1, len(points)):
                start_point = points[i]
                end_point = points[j]
                image = cv2.line(image, (int(start_point[0]), int(start_point[1])), (int(end_point[0]), int(end_point[1])), color, thickness)
        return image

    def drawSurface(self, image, intrinsic, extrinsic, distorted:bool):
        ks = np.array(intrinsic['ks'], np.float32)
        dist = np.array(intrinsic['dist'], np.float32)
        extrinsic_mat = np.array(extrinsic['extrinsic_mat'], np.float32)

        # Define the range for x and y
        x = np.arange(3, -3.5, -0.5)  # From 10 to -10 (inclusive) with step -1
        y = np.arange(7, -7.5, -0.5)

        # Generate the 2D grid
        xx, yy, zz = np.meshgrid(x, y, 0)

        # Stack to get a list of (x, y) coordinates
        point3ds = np.array(np.column_stack((xx.ravel(), yy.ravel(), zz.ravel())), np.float32)

        if not distorted:
            rvec, _ = cv2.Rodrigues(extrinsic_mat[:, :3])
            tvec = extrinsic_mat[:, 3]
            point2ds, _ = cv2.projectPoints(point3ds, rvec, tvec, ks, dist)
            point2ds = point2ds.reshape(-1, 2)
        else:
            point2ds = np.array([self.world_to_image(p, ks, extrinsic_mat) for p in point3ds], np.float32)

        #for p in point2ds:
        #    image = cv2.circle(image, (int(p[0]), int(p[1])), 5, (255, 255, 255) ,-1)

        # draw grids
        for i in range(len(x)-1):
            for j in range(len(y)):
                image = self.draw_lines_on_image(image, [point2ds[len(x)*j+i], point2ds[len(x)*j+i+1]], color=(255, 255, 255), thickness=3)

        for i in range(len(x)):
            for j in range(len(y)-1):
                image = self.draw_lines_on_image(image, [point2ds[i+len(x)*j], point2ds[i+len(x)*(j+1)]], color=(255, 255, 255), thickness=3)

        return image

    def drawCourt(self, image, intrinsic, extrinsic, distorted:bool):
        ks = np.array(intrinsic['ks'], np.float32)
        dist = np.array(intrinsic['dist'], np.float32)
        extrinsic_mat = np.array(extrinsic['extrinsic_mat'], np.float32)

        if self.combo_court.currentText() == "Court":
            # draw points
            CORNERS_3D_X = [-3.03, -2.57, 0, 2.57, 3.03]
            CORNERS_3D_Y = [ 6.68, 5.92, 2, -2, -5.92, -6.68]
        else:
            CORNERS_3D_X = [-1.1514, -0.9766, 0, 0.9766, 1.1514]
            CORNERS_3D_Y = [ 2.5384, 2.2496, 0.76, -0.76, -2.2496, -2.5384]
        point3ds = np.array([[x, y, 0] for y in CORNERS_3D_Y for x in CORNERS_3D_X], np.float32)

        if not distorted:
            rvec, _ = cv2.Rodrigues(extrinsic_mat[:, :3])
            tvec = extrinsic_mat[:, 3]
            point2ds, _ = cv2.projectPoints(point3ds, rvec, tvec, ks, dist)
            point2ds = point2ds.reshape(-1, 2)
        else:
            point2ds = np.array([self.world_to_image(p, ks, extrinsic_mat) for p in point3ds], np.float32)

        # draw lines
        for i in range(len(CORNERS_3D_X)-1):
            for j in range(len(CORNERS_3D_Y)):
                image = self.draw_lines_on_image(image, [point2ds[len(CORNERS_3D_X)*j+i], point2ds[len(CORNERS_3D_X)*j+i+1]])

        for i in range(len(CORNERS_3D_X)):
            for j in range(len(CORNERS_3D_Y)-1):
                if len(CORNERS_3D_X)*j+i == 12:
                    continue
                image = self.draw_lines_on_image(image, [point2ds[i+len(CORNERS_3D_X)*j], point2ds[i+len(CORNERS_3D_X)*(j+1)]])

        # draw points
        for p in point2ds:
            image = cv2.circle(image, (int(p[0]), int(p[1])), 5, (0, 0, 255) ,-1)

        # draw text
        if distorted:
            y_plus_2ds = np.array(self.world_to_image([0, max(CORNERS_3D_Y), 0], ks, extrinsic_mat), np.float32)
            y_minus_2ds = np.array(self.world_to_image([0, min(CORNERS_3D_Y), 0], ks, extrinsic_mat), np.float32)

            # draw text
            cv2.putText(image, "+y", (int(y_plus_2ds[0])+10, int(y_plus_2ds[1])+10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(image, "-y", (int(y_minus_2ds[0])+10, int(y_minus_2ds[1])+10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        return image
        
    def callback(self, appsink, serial):
        # print('before')
        # serial = serial.split('-')[0]
        # print(f'call serial = {serial}')
        if serial in self.cur_images:
            # print(f'close {serial}')
            # self.closeDevice(serial)
            return
        sample = appsink.emit("pull-sample")

        t, image = convert_sample_to_numpy(sample)


        # logging.debug(f'image shape = {image.shape}')

        # cv2.imshow('img', image)

        image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)

        config_file = f"{ROOTDIR}/Reader/Image_Source/config/{serial}.cfg"
        # if not os.path.isfile(config_file):
        #     none_file = f"{ROOTDIR}/Reader/Image_Source/config/None.cfg"
        #     shutil.copy2(none_file, config_file)

        camera_cfg = loadConfig(config_file)

        res = ast.literal_eval(camera_cfg['Camera']['RecordResolution'])
        image = cv2.resize(image, (res[0], res[1]), interpolation=cv2.INTER_AREA)
        # print(camera_cfg)
        undist_image = self.undistort(image, camera_cfg)

        # print(undist_image.shape)

        point_image = self.drawPointsOnImage(undist_image, camera_cfg)

        self.cur_images[serial] = point_image
        # print('after')

    def setupUI(self):
        layout_main = QVBoxLayout()


        self.warning_text = QLabel()
        self.warning_text.setText(' ')
        # layout.addWidget(self.warning_text)

        option = self.getOptions()
        layout_main.addWidget(option)
        layout_main.addWidget(self.warning_text, alignment=Qt.AlignmentFlag.AlignCenter)
        # layout_main.addWidget(self.ok_button, alignment=Qt.AlignmentFlag.AlignCenter)

        self.setLayout(layout_main)

    def getOptions(self):
        container_control = QWidget()
        layout_control = QVBoxLayout()

        # 回選單按鈕
        self.btn_menu = QPushButton()
        self.btn_menu.setText('回首頁')#返回
        self.btn_menu.setFixedSize(QSize(180, 50))
        # self.btn_menu.setStyleSheet('font: 24px')
        self.btn_menu.setStyleSheet(UIStyleSheet.ReturnButton)
        self.btn_menu.clicked.connect(self.backMenu)

        #self.ok_button = QPushButton("確定")
        # self.ok_button.setFont(QFont('Arial', 20))
        #self.ok_button.setFixedSize(QSize(180, 50))
        #self.ok_button.setStyleSheet(UIStyleSheet.SelectButton)
        #self.ok_button.clicked.connect(self.confirm)

        self.checkbox_undistort = QCheckBox("Undistort Image")
        self.checkbox_undistort.setChecked(True)
        self.checkbox_court = QCheckBox("Show court")
        self.checkbox_court.setChecked(True)
        self.checkbox_surface = QCheckBox("Show surface")

        self.combo_court = QComboBox()
        self.combo_court.addItems(["Court", "Office"])

        #self.btn_setting = QPushButton("相機設定")
        # self.btn_setting.setFont(QFont('Arial', 20))
        #self.btn_setting.setFixedSize(QSize(180, 50))
        #self.btn_setting.setStyleSheet(UIStyleSheet.SelectButton)
        #self.btn_setting.clicked.connect(self.setting)

        layout_control.addWidget(self.btn_menu, alignment=Qt.AlignmentFlag.AlignCenter)
        #layout_control.addWidget(self.ok_button, alignment=Qt.AlignmentFlag.AlignCenter)
        layout_control.addWidget(self.checkbox_undistort, alignment=Qt.AlignmentFlag.AlignLeft)
        layout_control.addWidget(self.checkbox_court, alignment=Qt.AlignmentFlag.AlignLeft)
        layout_control.addWidget(self.checkbox_surface, alignment=Qt.AlignmentFlag.AlignLeft)
        layout_control.addWidget(self.combo_court, alignment=Qt.AlignmentFlag.AlignLeft)
        #layout_control.addWidget(self.btn_setting, alignment=Qt.AlignmentFlag.AlignCenter)
        layout_control.setAlignment(Qt.AlignmentFlag.AlignTop)
        container_control.setLayout(layout_control)

        self.combo_boxes = [0 for _ in range(len(self.cameras))]
        option_layout = QHBoxLayout()

        image_label = QLabel()
        pixmap = QPixmap(f"{ICONDIR}/court/std_court_with_num.png")  # Replace with the actual path
        pixmap = pixmap.scaledToHeight(500)
        image_label.setPixmap(pixmap)
        # option_layout.addWidget(image_label)

        container = QWidget()
        container.setFixedWidth(300)
        container_layout = QVBoxLayout()
        container_layout.addWidget(container_control, alignment=Qt.AlignHCenter)
        container_layout.addWidget(image_label, alignment=Qt.AlignmentFlag.AlignCenter)
        container.setLayout(container_layout)

        container_left = QWidget()
        layout_left = QGridLayout()

        no_cam_pic_label = QLabel()
        self.no_cam_pixmap = QPixmap(f"{ICONDIR}/no_camera.png").scaled(240, 320)
        no_cam_pic_label.setPixmap(self.no_cam_pixmap)
        self.cams = []
        for i in range(len(self.cameras)):
            cam_i = ClickLabel(i)
            cam_i.setPixmap(self.no_cam_pixmap)
            #cam_i.clicked.connect(self.showBigPicture)
            self.cams.append(cam_i)

            img_container = QGroupBox(f"Camera_{i}")
            layout = QVBoxLayout()
            img_container.setLayout(layout)

            layout.addWidget(self.getOptionWidget(i), alignment=Qt.AlignCenter)
            layout.addWidget(cam_i, alignment=Qt.AlignCenter)

            layout_left.addWidget(img_container, int(i/4), int(i%4), alignment=Qt.AlignCenter)

        container_left.setLayout(layout_left)

        # option_widget = self.getOptionWidget()
        # option_layout.addWidget(option_widget)
        option_layout.addWidget(container)
        # option_layout.addWidget(image_label)
        option_layout.addWidget(container_left)

        container_option = QWidget()
        container_option.setLayout(option_layout)
        return container_option


    def showBigPicture(self, cam_idx):
        new_label = QLabel()
        # serial = self.combo_boxes[cam_idx].currentText()

        #new_label.setPixmap(self.cur_images[self.combo_boxes[cam_idx].currentText()].scaled(1200, 900))
        
        image_pixmap = self.cur_images[self.combo_boxes[cam_idx].currentText()].scaled(1200, 900)

        new_label = InteractiveLabel(image_pixmap)
        
        self.bigpicture = QDialog()
        back_btn = QPushButton("返回")
        back_btn.setFont(QFont('Arial', 20))
        back_btn.setFixedSize(QSize(200, 60))

        back_btn.clicked.connect(self.bigpicture.accept)


        layout = QHBoxLayout()
        layout.addWidget(back_btn, alignment=Qt.AlignmentFlag.AlignTop)
        layout.addWidget(new_label)
        self.bigpicture.setLayout(layout)

        self.bigpicture.exec_()

    def getOptionWidget(self, index):
        box_layout = QVBoxLayout()
        font = QFont()

        text_label = QLabel()
        text_label.setText(f'Camera {index + 1}:')
        font.setPixelSize(24)
        text_label.setFont(font)

        combo_box = QComboBox()
        combo_box.addItems(self.serials)
        font.setPixelSize(20)
        combo_box.setFont(font)
        # print(f'index = {index}, text = {self.cfg[f"CameraReader_{index}"]["hw_id"]}')
        combo_box.currentIndexChanged.connect(self.comboboxChanged)
        combo_box.setCurrentText(self.cfg[f'CameraReader_{index}']['hw_id'])
        self.combo_boxes[index] = combo_box

        self.combo_boxes[index].setHidden(True)

        # box_layout.addWidget(text_label, alignment=Qt.AlignmentFlag.AlignBottom)
        box_layout.addWidget(combo_box, alignment=Qt.AlignmentFlag.AlignTop)

        little_box = QWidget()
        little_box.setLayout(box_layout)
        return little_box

    def isDupSerials(self):
        st = set()
        for i in range(len(self.cameras)):
            if self.combo_boxes[i].currentText() == 'None':
                continue
            if self.combo_boxes[i].currentText() in st:
                return True
            st.add(self.combo_boxes[i].currentText())
        return False

    def comboboxChanged(self):
        for i in range(len(self.cameras)):
            if self.combo_boxes[i] == 0:
                return

        for i in range(len(self.cameras)):
            if self.combo_boxes[i].currentText() == 'None' or self.cur_images.get(self.combo_boxes[i].currentText()) is None:
                self.cams[i].setPixmap(self.no_cam_pixmap)
                continue
            point_image = self.cur_images.get(self.combo_boxes[i].currentText())
            if(i == 0 or i == 2):
                point_image  = cv2.rotate(point_image, cv2.ROTATE_90_CLOCKWISE)
            if(i == 1 or i == 3):
                point_image  = cv2.rotate(point_image, cv2.ROTATE_90_COUNTERCLOCKWISE)
            height, width, channel = point_image.shape
            bytesPerLine = 3 * width
            qimage = QImage(point_image.data, width, height, bytesPerLine, \
                    QImage.Format_RGB888)

            self.cams[i].setPixmap(QPixmap(qimage).scaled(240, 320))

        if self.isDupSerials():
            self.warning_text.setText("<font color=red size=20> 相機編號重複!</font>")
            return

        self.warning_text.setText('')

    def confirm(self):
        """確認相機設定沒問題
        """
        if self.isDupSerials():
            self.warning_text.setText("<font color=red size=20> 相機編號重複!</font>")
            return

        # for i in range(len(self.cameras)):
        #     self.cfg[f'CameraReader_{i}']['hw_id'] = self.combo_boxes[i].currentText()
        #     self.cameras[i].hw_id = self.combo_boxes[i].currentText()

        # saveConfig(self.cfg_file, self.cfg)

        logging.info('config file saved')

        msg = MsgContract(MsgContract.ID.PAGE_CHANGE, value=self.from_page)
        self.myService.sendMessage(msg)

    def backMenu(self):
        """回首頁
        """

        msg = MsgContract(MsgContract.ID.PAGE_CHANGE, value='Home')
        self.myService.sendMessage(msg)

    def setting(self):
        """到相機設定頁面
        """

        msg = MsgContract(MsgContract.ID.PAGE_CHANGE, value='CameraSystem')
        msg.data = 'SettingPreviewPage'
        self.myService.sendMessage(msg)
