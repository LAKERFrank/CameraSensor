from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import numpy as np
import pandas as pd
import os
import sys
import cv2
import argparse, csv, json
from PIL import Image
import math

from OpenGL.GLU import *
from OpenGL.GL import *
from OpenGL.GLUT import *
from math import nan, sqrt, pi, sin, cos, tan
from ..UISettings import *

DIRNAME = os.path.dirname(os.path.abspath(__file__))
ROOTDIR = os.path.dirname(os.path.dirname(DIRNAME))
ICONDIR = f"{ROOTDIR}/UI/icon"

sys.path.append(f"{ROOTDIR}/Pseudo3D")
from generator import toRad, drawCourt, drawNet

np.set_printoptions(precision=14)
np.set_printoptions(suppress=True)

class Transform(object):
    def __init__(self):
        super(Transform, self).__init__()
        self.gt = False

        # Opengl param
        self.eye = np.zeros(3)
        self.obj = np.zeros(3)
        self.up = np.zeros(3)
        self.quadric = gluNewQuadric()

        # Curve param
        self._f = 0 # guess focal offset
        self.rad_z = 0
        self.rad_y = 0
        self.rad_x = 0

        # OpenGL Init camera pose parameters for gluLookAt() function
        init_pose = (
            np.array([0., -15.5885, 9.]),
            np.array([0., 0., 0.]),
            np.array([0. , 0.5 , 0.866])
        )
        self.setupCam(init_pose)

    def setupCam(self, pose):
        self.eye = pose[0]
        self.obj = pose[1]
        self.up = pose[2]
        # self.eye += np.array([0,0,5])
        # self.obj += np.array([0,0,5])

    def rotate(self, zAngle, yAngle, xAngle):
        rot_z = np.array([[cos(zAngle), -sin(zAngle), 0],
                        [sin(zAngle),  cos(zAngle), 0],
                        [0, 0, 1]])
        rot_y = np.array([[cos(yAngle), 0, sin(yAngle)],
                        [0, 1, 0],
                        [-sin(yAngle), 0, cos(yAngle)]])

        rot_x = np.array([[1, 0, 0],
                        [0, cos(xAngle), -sin(xAngle)],
                        [0, sin(xAngle), cos(xAngle)]])

        _eye = rot_x @ rot_y @ rot_z @ self.eye.reshape(-1,1)

        _up = rot_x @ rot_y @ rot_z @ self.up.reshape(-1,1)

        return _eye.reshape(-1), self.obj, _up.reshape(-1)

class Trajectory3DVisualizeWidget(QOpenGLWidget):
    def __init__(self, parent=None, fovy=40, height=1060, width=1920):
        QOpenGLWidget.__init__(self, parent)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update)
        self.timer.start(100)
        self.setFocusPolicy(Qt.StrongFocus)

        # initialize
        self.tracks = {}
        self.balltype = [1,1,1,1,0]
        # 0: all, 1: First shot, 2: Second shot, ...
        self.current_shot = 0

        # set opengl gluPerspective
        self.fovy = fovy
        self.height = height
        self.width = width

        self.x = 0
        self.y = 0
        self.ctl = False
        self.translate = 0
        self.translate_z = 0
        self.show_axis = False
        self.show_location = False

        # self.reset_btn = QPushButton('', self) # mode = 0
        # self.reset_btn.move(850, 600)
        # self.reset_btn.clicked.connect(self.onClickOrigin)
        # self.reset_btn.resize(70, 70)
        # size = QSize(62, 62)
        # self.reset_btn.setIcon(QIcon(f'{ICONDIR}/reset.png'))
        # self.reset_btn.setIconSize(size)

        # self.top_btn = QPushButton('', self) # mode = 1
        # self.top_btn.move(930, 600)
        # self.top_btn.clicked.connect(self.onClickTop)
        # self.top_btn.resize(70, 70)
        # self.top_btn.setIcon(QIcon(f'{ICONDIR}/top.png'))
        # self.top_btn.setIconSize(size)

        # self.side_btn = QPushButton('', self) # mode = 2
        # self.side_btn.move(1010, 600)
        # self.side_btn.clicked.connect(self.onClickSide)
        # self.side_btn.resize(70, 70)
        # self.side_btn.setIcon(QIcon(f'{ICONDIR}/side.png'))
        # self.side_btn.setIconSize(size)

        # self.show_axis_btn = QPushButton('', self)
        # self.show_axis_btn.move(1040, 720)
        # self.show_axis_btn.clicked.connect(self.onClickShowAxis)
        # self.show_axis_btn.resize(50, 50)
        # self.show_axis_btn.setIcon(QIcon(f'{ICONDIR}/xyz.png'))
        # self.show_axis_btn.setIconSize(size)

        # self.show_location_btn = QPushButton('', self)
        # self.show_location_btn.move(980, 720)
        # self.show_location_btn.clicked.connect(self.onClickShowLocation)
        # self.show_location_btn.resize(50, 50)
        # self.show_location_btn.setIcon(QIcon(f'{ICONDIR}/location.png'))
        # self.show_location_btn.setIconSize(size)

        # Checkboxes for choosing players
        self.A_checkbox = QCheckBox('Player A', self)
        self.A_checkbox.setChecked(True)
        self.A_checkbox.move(345, 650)
        self.A_checkbox.setStyleSheet(UIStyleSheet.TrajectoryABCheckBox)
        self.B_checkbox = QCheckBox('Player B', self)
        self.B_checkbox.setChecked(True)
        self.B_checkbox.move(740, 650)
        self.B_checkbox.setStyleSheet(UIStyleSheet.TrajectoryABCheckBox)

        self.ball = QLabel(self)
        self.ball.move(18, 8)
        self.ball.resize(100, 40)
        self.ball.setStyleSheet("color: white; background-color: #191919")
        # self.ball.setFont(QFont('BiauKai', 19))
        font = QFont()
        font.setPointSize(19)
        self.ball.setFont(font)


        self.tf = Transform()
        self.scaled = 1
        self.mode = 2 # 0:origin, 1:top, 2:side
        self.top_pt = 0
        self.side_pt = 0
        self.top_angle = [60, 0, -60, 0]
        self.side_angle = [-30, 0, 30, 0]
        # self.color_list = [
        #     [1,0,0],
        #     [0,1,0],
        #     [0,0,1],
        #     [1,1,0],
        #     [0,1,1],
        #     [1,0,1]
        # ]

        self.filtered_long_shots = []
        self.filtered_A_shots = []
        self.filtered_B_shots = []

        self.init_speed_img, self.init_speed_img_w, self.init_speed_img_h = self.LoadImgText(f'{ICONDIR}/init_speed.png')
        self.duration_img, self.duration_img_w, self.duration_img_h  = self.LoadImgText(f'{ICONDIR}/duration.png')
        self.ascent_speed_img, self.ascent_speed_img_w, self.ascent_speed_img_h  = self.LoadImgText(f'{ICONDIR}/ascent_speed.png')
        self.net_height_img, self.net_height_img_w, self.net_height_img_h = self.LoadImgText(f'{ICONDIR}/net_height.png')
        self.service_height_img, self.service_height_img_w, self.service_height_img_h = self.LoadImgText(f'{ICONDIR}/service_height.png')
        self.max_height_img, self.max_height_img_w, self.max_height_img_h = self.LoadImgText(f'{ICONDIR}/max_height.png')
        self.baseline_dis_img, self.baseline_dis_img_w, self.baseline_dis_img_h = self.LoadImgText(f'{ICONDIR}/baseline_distance.png')
        self.descent_angle_img, self.descent_angle_img_w, self.descent_angle_img_h  = self.LoadImgText(f'{ICONDIR}/descent_angle.png')

    def initializeGL(self):
        #print("initializeGL is called")
        glutInit()
        glutInitDisplayMode(GLUT_RGBA | GLUT_SINGLE)
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glClearColor(0.1, 0.1, 0.1, 1.)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        # Tell the Opengl, we are going to set PROJECTION function
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()  # always call this before gluPerspective
        # set intrinsic parameters (fovy, aspect, zNear, zFar)
        gluPerspective(self.fovy, self.width/self.height, 0.1, 100000.0)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

    def paintGL(self):
        #print("paintGL is called")
        #print(f"Tracks: {self.tracks}")
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

        _eye, _obj, _up = self.tf.rotate(self.tf.rad_z, self.tf.rad_y, self.tf.rad_x)
        #glPointSize(5.0)  # 設置點的大小為5像素
        #glBegin(GL_POINTS)
        #glVertex3f(0.0, 0.0, 0.0)
        #glEnd()
        if self.mode == 2:
            gluLookAt(_eye[0], _eye[1], _eye[2]+3,
                    _obj[0], _obj[1], _obj[2]+3,
                    _up[0], _up[1], _up[2])
            self.setRad(0, toRad(30), toRad(90))
        else:
            gluLookAt(_eye[0], _eye[1], _eye[2],
                    _obj[0], _obj[1], _obj[2],
                    _up[0], _up[1], _up[2])
        glScalef(self.scaled, self.scaled, self.scaled)
        translate_x = (abs(_eye[1])/(abs(_eye[0])+abs(_eye[1])))*self.translate
        translate_y = (abs(_eye[0])/(abs(_eye[0])+abs(_eye[1])))*self.translate
        glTranslatef(translate_x, translate_y, self.translate_z)

        if self.show_axis:
            line_len = 1
            grid_color = [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)]
            glLineWidth(3) # set line width thicker
            origin = [0.0, 0.0, 0.0]
            for i in range(3):
                tmp = [0.0, 0.0, 0.0]
                tmp[i] = line_len*1.02
                glColor3f(*grid_color[i])
                glBegin(GL_LINES)
                glVertex3f(*origin)
                glVertex3f(*tmp)
                glEnd()
            glColor4ub(255, 0, 0, 255)
            glRasterPos3f(1.1, 0.0, 0.1)
            glutBitmapCharacter(GLUT_BITMAP_HELVETICA_12, ord('x'))
            glColor4ub(0, 255, 0, 255)
            glRasterPos3f(0.0, 1.1, 0.1)
            glutBitmapCharacter(GLUT_BITMAP_HELVETICA_12, ord('y'))
            glColor4ub(0, 0, 255, 255)
            glRasterPos3f(0.0, 0.0, 1.7)
            glutBitmapCharacter(GLUT_BITMAP_HELVETICA_12, ord('z'))
            glLineWidth(1) # set line width back to default
        # Draw Model3D track
        total_tracks = len(self.tracks)
        if total_tracks > 0:
            for shot, track in self.tracks.items():
                if shot < total_tracks and self.balltype[shot - 1] == '長球':
                    if shot not in self.filtered_long_shots:
                        self.filtered_long_shots.append(shot)
            for shot, track in self.tracks.items():
                if shot in self.filtered_long_shots:
                    if track[0].y <= 0:
                        if shot not in self.filtered_A_shots:
                            self.filtered_A_shots.append(shot)
                    else:
                        if shot not in self.filtered_B_shots:
                            self.filtered_B_shots.append(shot)

            if self.mode == 2:
                self.drawAxis()
            for shot, track in self.tracks.items():
                if shot in self.filtered_long_shots:
                    if self.A_checkbox.isChecked() and self.B_checkbox.isChecked():
                        self.drawTrack(track, shot)
                        if self.mode == 2:
                            if shot == self.current_shot:
                                self.drawTrackDetails(track, shot)
                    elif self.A_checkbox.isChecked() and not self.B_checkbox.isChecked():
                        if shot in self.filtered_A_shots and shot < total_tracks:
                            self.drawTrack(track, shot)
                            if self.mode == 2:
                                if shot == self.current_shot:
                                    self.drawTrackDetails(track, shot)
                    elif self.B_checkbox.isChecked() and not self.A_checkbox.isChecked():
                        if shot in self.filtered_B_shots and shot < total_tracks:
                            self.drawTrack(track, shot)
                            if self.mode == 2:
                                if shot == self.current_shot:
                                    self.drawTrackDetails(track, shot)

        # Draw badminton court
        if self.mode == 2:
            glColor3f(3/255, 102/255, 71/255)
            glBegin(GL_LINES)
            glVertex3f(3, -6.7, 0)
            glVertex3f(3, 6.7, 0)
            glEnd()

            current_line_width = glGetFloatv(GL_LINE_WIDTH)
            glLineWidth(3.0)
            glColor3f(0.6, 0.6, 0)
            glBegin(GL_LINES)
            glVertex3f(3, 0, 0)
            glVertex3f(3, 0, 1.55)
            glEnd()
            glLineWidth(current_line_width)
        else:
            drawCourt()
            drawNet()
                                    

    def resizeGL(self, w, h):
        glViewport(0, 0, w, h)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(self.fovy, w / h, 0.1, 100000.0)
        # gluPerspective(40, w / h, 0.1, 100000.0)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

    def drawAxis(self):
        current_line_width = glGetFloatv(GL_LINE_WIDTH)
        glLineWidth(3.0)

        glColor3f(1, 1, 1)

        glBegin(GL_LINES)
        glVertex3f(3, 2.02, 0.1)
        glVertex3f(3, 2.02, -0.1)
        glEnd()

        glBegin(GL_LINES)
        glVertex3f(3, 5.94, 0.1)
        glVertex3f(3, 5.94, -0.1)
        glEnd()

        glBegin(GL_LINES)
        glVertex3f(3, 6.7, 0.1)
        glVertex3f(3, 6.7, -0.1)
        glEnd()

        glBegin(GL_LINES)
        glVertex3f(3, -2.02, 0.1)
        glVertex3f(3, -2.02, -0.1)
        glEnd()

        glBegin(GL_LINES)
        glVertex3f(3, -5.94, 0.1)
        glVertex3f(3, -5.94, -0.1)
        glEnd()

        glBegin(GL_LINES)
        glVertex3f(3, -6.7, 0.1)
        glVertex3f(3, -6.7, -0.1)
        glEnd()

        glLineWidth(current_line_width)

        glBegin(GL_LINES)
        glVertex3f(3, -6.95, 1.5)
        glVertex3f(3, -6.85, 1.5)
        glEnd()
        glRasterPos3f(3, -7.35, 1.42)
        text_1 = "1.5"
        for character in text_1:
            glutBitmapCharacter(GLUT_BITMAP_HELVETICA_18, ord(character))

        glBegin(GL_LINES)
        glVertex3f(3, -6.95, 3)
        glVertex3f(3, -6.85, 3)
        glEnd()
        glRasterPos3f(3, -7.35, 2.92)
        text_2 = "3.0"
        for character in text_2:
            glutBitmapCharacter(GLUT_BITMAP_HELVETICA_18, ord(character))

        glBegin(GL_LINES)
        glVertex3f(3, -6.95, 4.5)
        glVertex3f(3, -6.85, 4.5)
        glEnd()
        glRasterPos3f(3, -7.35, 4.42)
        text_3 = "4.5"
        for character in text_3:
            glutBitmapCharacter(GLUT_BITMAP_HELVETICA_18, ord(character))

        glBegin(GL_LINES)
        glVertex3f(3, -6.95, 6)
        glVertex3f(3, -6.85, 6)
        glEnd()
        glRasterPos3f(3, -7.3, 5.92)
        text_4 = "6.0"
        for character in text_4:
            glutBitmapCharacter(GLUT_BITMAP_HELVETICA_18, ord(character))


    def drawTrack(self, track, shot):
        pre_points = []
        MAX_TIME_DELAY = 0.5
        # color = shot % len(self.color_list)
        for point in track:
            #print(f"Drawing point at X: {point.x}, Y: {point.y}, Z: {point.z}")
            size = 0.02
            if point.color != 'white':
                self.sphere(point.fid, point.x, point.y, point.z, color=point.color, size = 0.06)
            else:
                if shot == self.current_shot:
                    self.sphere(point.fid, point.x, point.y, point.z, color='orange', size = 0.035)
                else:
                    self.sphere(point.fid, point.x, point.y, point.z, color=point.color, size = size)
            if len(pre_points) != 0:
                pre_point = pre_points.pop(0)
                if abs(pre_point.timestamp - point.timestamp) <= MAX_TIME_DELAY:
                    glBegin(GL_LINES)
                    if self.mode == 2:
                        glVertex3f(3, pre_point.y, pre_point.z)
                        glVertex3f(3, point.y, point.z)
                    else:
                        glVertex3f(pre_point.x, pre_point.y, pre_point.z)
                        glVertex3f(point.x, point.y, point.z)
                    glEnd()
            pre_points.append(point)

    def sphere(self, fid, x, y, z, color, size=0.02):
        if color == 'red':
            color = (255, 0, 0)
        if color == 'orange':
            color = (255, 165, 0)
        elif color == 'yellow':
            color = (255, 255, 0)
        elif color == 'green':
            color = (0, 255, 0)
        elif color == 'light blue':
            color = (173, 216, 230)
        elif color == 'cyan':
            color = (0, 255, 255)
        elif color == 'blue':
            color = (39, 142, 214)
        elif color == 'indigo':
            color = (75, 0, 130)
        elif color == 'purple':
            color = (128, 0, 128)
        elif color == 'violet':
            color = (238, 130, 238)
        elif color == 'magenta':
            color = (255, 0, 255)
        elif color == 'pink':
            color = (255, 192, 203)
        elif color == 'white':
            color = (255, 255, 255)
        elif color == 'gray' or color == 'grey':
            color = (128, 128, 128)
        elif color == 'black':
            color = (0, 0, 0)
        glColor3ub(color[0], color[1], color[2])
        if self.mode == 2:
            x = 3
        glTranslatef(x, y, z)
        gluSphere(self.tf.quadric, size, 32, 32)
        glTranslatef(-x, -y, -z)
        if self.show_location == True:
            glColor4ub(color[0], color[1], color[2], 255)
            glRasterPos3f(x+0.3 , y, z)
            self.DrawText('{}:({},{},{})'.format(fid, round(x,2), round(y,2), round(z,2)))

    def DrawText(self, string, str_type = GLUT_BITMAP_HELVETICA_12):
        for c in string:
            glutBitmapCharacter(str_type, ord(c))
    
    def LoadImgText(self, path):
        image = Image.open(path)
        width, height = image.size
        image_data = image.tobytes("raw", "RGBA", 0, -1)
        return image_data, width, height
    
    def DrawImgText(self, image, width, height, y1, y2, z1=7, z2=7.25):
        texture_id = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, texture_id)
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, image)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

        glColor3ub(255, 255, 255)  # Set color to white for texture
        glEnable(GL_TEXTURE_2D)
        glBindTexture(GL_TEXTURE_2D, texture_id)
        glBegin(GL_QUADS)
        # Define texture coordinates for each vertex
        glTexCoord2f(0, 0)
        glVertex3f(3, y1, z1) # 左下
        glTexCoord2f(1, 0)
        glVertex3f(3, y2, z1) # 右下
        glTexCoord2f(1, 1)
        glVertex3f(3, y2, z2) # 右上
        glTexCoord2f(0, 1)
        glVertex3f(3, y1, z2) # 左上
        glEnd()
        glDisable(GL_TEXTURE_2D)

    def ballType(self, balltype):
        self.balltype = balltype
    
    def ballInfo(self, ballinfo):
        self.ballinfo = ballinfo

    def shotDuration(self, duration):
        self.shot_duration = duration

    def drawTrackDetails(self, track, shot):
        if track:
            first_point = track[0]
            last_point = track[-1]
            highest_point = max(track, key=lambda point: point.z) 

            if first_point.y <= 0:
                player = 'A'
            else:
                player = 'B'

            # Speed
            init_speed = self.ballinfo[shot - 1]['Speed']
            if player == 'A':
                y1 = -5.9
                y2 = -4.7
                y = -6
            else:
                y1 = 4.95
                y2 = 6.15
                y = 4.85
            self.DrawImgText(self.init_speed_img, self.init_speed_img_w, self.init_speed_img_h, y1, y2)
            glColor3ub(0, 128, 255)
            glRasterPos3f(3, y, 6.7)
            self.DrawText(f'{init_speed:.2f} km/hr', GLUT_BITMAP_HELVETICA_18)
            

            #Duration
            duration = self.shot_duration.get(shot)
            if player== 'A':
                y1 = -5.95
                y2 = -4.75
                y = -5.7
            else:
                y1 = 4.9
                y2 = 6.1
                y = 5.15
            self.DrawImgText(self.duration_img, self.duration_img_w, self.duration_img_h, y1, y2, 6.3, 6.55)
            glColor3ub(0, 128, 255)
            glRasterPos3f(3, y, 6)
            self.DrawText(f'{duration:.2f} s', GLUT_BITMAP_HELVETICA_18)


            # 上升段(擊球後y座標7公尺)平均速度
            if player == 'A':
                speed_y_target = first_point.y + 7
            else:
                speed_y_target = first_point.y - 7
            speed_point1 = None
            speed_point2 = None
            if first_point.y > 0:
                for point in track:
                    if point.y >= speed_y_target:
                        speed_point1 = point
                    else:
                        speed_point2 = point
                        break
            else:
                for point in track:
                    if point.y <= speed_y_target:
                        speed_point1 = point
                    else:
                        speed_point2 = point
                        break
            if speed_point1 and speed_point2:
                t = (speed_y_target - speed_point1.y) / (speed_point2.y - speed_point1.y)
                speed_x_target = speed_point1.x + t * (speed_point2.x - speed_point1.x)
                speed_z_target = speed_point1.z + t * (speed_point2.z - speed_point1.z)
                speed_y_target = speed_y_target
                speed_time_target = speed_point1.timestamp + t * (speed_point2.timestamp - speed_point1.timestamp)
                dis = np.array([speed_x_target - first_point.x, speed_y_target - first_point.y, speed_z_target - first_point.z])
                ascent_speed = np.linalg.norm(dis) / (speed_time_target - first_point.timestamp) * 3.6

                if player == 'A':
                    y1 = -4
                    y2 = -2
                    y = -3.65
                else:
                    y1 = 2.15
                    y2 = 4.15
                    y = 2.5
                self.DrawImgText(self.ascent_speed_img, self.ascent_speed_img_w, self.ascent_speed_img_h, y1, y2)
                glColor3ub(0, 128, 255)
                glRasterPos3f(3, y, 6.7)
                self.DrawText(f'{ascent_speed:.2f} km/hr', GLUT_BITMAP_HELVETICA_18)
    

            # 下墜角度(最大高度點跟最後一點的連線的斜率(最後一點跟最大高度點切線的夾角))
            if player== 'A':
                v1 = (1, 0)
            else:
                v1 = (-1, 0)
            v2 = (last_point.y - highest_point.y, last_point.z - highest_point.z)
            dot_product = sum(a * b for a, b in zip(v1, v2))
            v1_length = math.sqrt(sum(a ** 2 for a in v1))
            v2_length = math.sqrt(sum(a ** 2 for a in v2))
            cos_theta = dot_product / (v1_length * v2_length)
            angle_rad = math.acos(cos_theta)
            angle_deg = math.degrees(angle_rad)
            glLineWidth(2.0)
            glColor3ub(102, 178, 255)
            glEnable(GL_LINE_STIPPLE)
            glLineStipple(5, 0xAAAA)
            glBegin(GL_LINES)
            glVertex3f(3, highest_point.y, highest_point.z)
            glVertex3f(3, last_point.y, last_point.z)                
            glEnd()
            glDisable(GL_LINE_STIPPLE)

            if player == 'A':
                y1 = -3.6
                y2 = -2.4
                y = -3.35
            else:
                y1 = 2.55
                y2 = 3.75
                y = 2.8
            self.DrawImgText(self.descent_angle_img, self.descent_angle_img_w, self.descent_angle_img_h, y1, y2, 6.3, 6.55)
            glColor3ub(102, 178, 255)
            glRasterPos3f(3, y, 6)
            self.DrawText(f'{angle_deg:.2f}°', GLUT_BITMAP_HELVETICA_18)


            # net height
            net_y_target = 0
            net_point1 = None
            net_point2 = None
            if first_point.y > 0:
                for point in track:
                    if point.y >= net_y_target:
                        net_point1 = point
                    else:
                        net_point2 = point
                        break
            else:
                for point in track:
                    if point.y <= net_y_target:
                        net_point1 = point
                    else:
                        net_point2 = point
                        break
            if net_point1 and net_point2:
                t = (net_y_target - net_point1.y) / (net_point2.y - net_point1.y)
                net_x_target = net_point1.x + t * (net_point2.x - net_point1.x)
                net_z_target = net_point1.z + t * (net_point2.z - net_point1.z)
                net_y_target = net_y_target
                glColor3ub(255, 102, 102)
                glEnable(GL_LINE_STIPPLE)
                glLineStipple(5, 0xAAAA)
                glBegin(GL_LINES)
                glVertex3f(3, net_y_target, 0)
                glVertex3f(3, net_y_target, net_z_target)
                glEnd()
                glDisable(GL_LINE_STIPPLE)

                self.DrawImgText(self.net_height_img, self.net_height_img_w, self.net_height_img_h, -0.6, 0.6)
                glColor3ub(255, 102, 102)
                glRasterPos3f(3, -0.35, 6.7)
                self.DrawText(f'{net_z_target:.2f} m', GLUT_BITMAP_HELVETICA_18)
    

            # serve line height
            sl_point1 = None
            sl_point2 = None
            if first_point.y > 0:
                sl_y_target = -2.02
                if track[-1].y <= sl_y_target:
                    for point in track:
                        if point.y >= sl_y_target:
                            sl_point1 = point
                        else:
                            sl_point2 = point
                            break
            else:
                sl_y_target = 2.02
                if track[-1].y >= sl_y_target:
                    for point in track:
                        if point.y <= sl_y_target:
                            sl_point1 = point
                        else:
                            sl_point2 = point
                            break    
            if sl_point1 and sl_point2:
                t = (sl_y_target - sl_point1.y) / (sl_point2.y - sl_point1.y)
                sl_x_target = sl_point1.x + t * (sl_point2.x - sl_point1.x)
                sl_z_target = sl_point1.z + t * (sl_point2.z - sl_point1.z)
                sl_y_target = sl_y_target
                glColor3ub(237, 161, 175)
                glEnable(GL_LINE_STIPPLE)
                glLineStipple(5, 0xAAAA)
                glBegin(GL_LINES)
                glVertex3f(3, sl_y_target, 0)
                glVertex3f(3, sl_y_target, sl_z_target)
                glEnd()
                glDisable(GL_LINE_STIPPLE)
                
                if player== 'A':
                    y1 = 1.1
                    y2 = 2.8
                    y = 1.6
                else:
                    y1 = -2.8
                    y2 = -1.1
                    y = -2.3
                self.DrawImgText(self.service_height_img, self.service_height_img_w, self.service_height_img_h, y1, y2)
                glColor3ub(237, 161, 175)
                glRasterPos3f(3, y, 6.7)
                self.DrawText(f'{sl_z_target:.2f} m', GLUT_BITMAP_HELVETICA_18)


            # max height
            glColor3ub(0, 255, 0)
            glEnable(GL_LINE_STIPPLE)
            glLineStipple(5, 0xAAAA)
            glBegin(GL_LINES)
            glVertex3f(3, highest_point.y, 0)
            glVertex3f(3, highest_point.y, highest_point.z)
            glEnd()
            glDisable(GL_LINE_STIPPLE)
            
            if player== 'A':
                y1 = 3.35
                y2 = 4.55
                y = 3.6
            else:
                y1 = -4.45
                y2 = -3.25
                y = -4.2
            self.DrawImgText(self.max_height_img, self.max_height_img_w, self.max_height_img_h, y1, y2)
            glColor3ub(0, 255, 0)
            glRasterPos3f(3, y, 6.7)
            self.DrawText(f'{highest_point.z:.2f} m', GLUT_BITMAP_HELVETICA_18)


            # baseline distance
            if last_point.y > 0:
                d = last_point.y - 6.7
            else:
                d = last_point.y - (-6.7)
            distance_to_baseline = abs(d)
            glColor3ub(238, 130, 238)
            glEnable(GL_LINE_STIPPLE)
            glLineStipple(5, 0xAAAA)
            glBegin(GL_LINES)
            if last_point.y > 0:
                glVertex3f(3, 6.7, last_point.z)
                glVertex3f(3, 6.7 + d, last_point.z)
            else:
                glVertex3f(3, -6.7, last_point.z)
                glVertex3f(3, -6.7 + d, last_point.z)                
            glEnd()
            glDisable(GL_LINE_STIPPLE)

            if last_point.y > 0:
                glBegin(GL_LINES)
                glVertex3f(3, 6.7, last_point.z + 0.15)
                glVertex3f(3, 6.7, last_point.z - 0.15)
                glEnd()
                glBegin(GL_LINES)
                glVertex3f(3, 6.7 + d, last_point.z + 0.15)
                glVertex3f(3, 6.7 + d, last_point.z - 0.15)
                glEnd()
            else:
                glBegin(GL_LINES)
                glVertex3f(3, -6.7, last_point.z + 0.15)
                glVertex3f(3, -6.7, last_point.z - 0.15)
                glEnd()
                glBegin(GL_LINES)
                glVertex3f(3, -6.7 + d, last_point.z + 0.15)
                glVertex3f(3, -6.7 + d, last_point.z - 0.15)
                glEnd()
            
            if player== 'A':
                y1 = 5.1
                y2 = 6.65
                y = 5.45
            else:
                y1 = -6.5
                y2 = -4.95
                y = -6.15
            self.DrawImgText(self.baseline_dis_img, self.baseline_dis_img_w, self.baseline_dis_img_h, y1, y2)
            glColor3ub(238, 130, 238)
            glRasterPos3f(3, y, 6.7)
            self.DrawText(f'{distance_to_baseline:.2f} m', GLUT_BITMAP_HELVETICA_18)
    
            glLineWidth(1.0)         
    
    def addPointByTrackID(self, point, id):
        if id not in self.tracks.keys():
            self.tracks[id] = []
        self.tracks[id].append(point)
        #print(f"Point added to track {id}: {point.x}, {point.y}, {point.z}")

    def getTracksByTrackID(self):
        return self.tracks

    def clearAll(self):
        self.tracks.clear()
        self.current_shot = 0
        # self.setRad(0, 0, 0)
        self.mode = 2
        self.scaled = 1

    def setShot(self, shot):
        if len(self.tracks) == 0:
            self.current_shot = 0
            return
        if shot < 0:
            shot = len(self.tracks) - 1
        if shot >= len(self.tracks):
            shot = 0
        self.current_shot = shot

    def getShot(self):
        return self.current_shot

    def setShotText(self, shot):
        if shot > 0:
            self.ball.setText(f"第 {shot} 拍")
        else:
            self.ball.setText(f"無高遠球")

    def onClickOrigin(self):
        self.mode = 0
        self.scaled = 1
        self.setRad(0, 0, 0)
        self.update
        self.translate = 0
        self.translate_z = 0

    def onClickTop(self):
        self.mode = 1
        self.scaled = 1
        self.top_pt = 0
        self.setRad(0, toRad(60), toRad(-90))
        self.update

    def onClickSide(self):
        self.mode = 2
        self.scaled = 1
        self.side_pt = 0
        # self.setRad(0, toRad(-28), toRad(-86)) # Slightly tilt, to see the trajectory pass above the net or not
        self.setRad(0, toRad(30), toRad(90))
        self.update

    def onClickShowAxis(self):
        if self.show_axis:
            self.show_axis = False
        else:
            self.show_axis = True
    def onClickShowLocation(self):
        if self.show_location:
            self.show_location = False
        else:
            self.show_location = True
    def mousePressEvent (self, event):      #click mouse
        self.x = event.x()
        self.y = event.y()
    def mouseMoveEvent(self, event):        #click and move mouse
        if event.buttons() and Qt.LeftButton:
            if not self.ctl:
                if int(event.x()) > int(self.x):
                    if self.mode == 0 or self.mode == 2: # origin
                        self.addRad(0, 0, toRad(-5))
                    elif self.mode == 1: # top
                        self.scaled = 1
                        self.top_pt -= 1
                        self.top_pt %= 4
                        y = self.top_angle[self.top_pt]
                        x = self.top_angle[(self.top_pt + 1) % 4]
                        self.setRad(toRad(x), toRad(y), self.tf.rad_z + toRad(-90))
                    self.x = event.x()
                    self.y = event.y()
                    self.update
                else:
                    if self.mode == 0 or self.mode == 2: # origin
                        self.addRad(0, 0, toRad(5))
                    elif self.mode == 1: # top
                        self.scaled = 1
                        self.top_pt += 1
                        self.top_pt %= 4
                        y = self.top_angle[self.top_pt]
                        x = self.top_angle[(self.top_pt + 1) % 4]
                        self.setRad(toRad(x), toRad(y), self.tf.rad_z + toRad(90))
                    self.x = event.x()
                    self.y = event.y()
                    self.update
            else:
                self.translate = (event.x()-self.x)/10
                self.translate_z = (event.y()-self.y)/10
    def wheelEvent(self, event):
        angle = event.angleDelta() / 8
        angleX = angle.x()
        angleY = angle.y()
        if angleY > 0:
            if self.scaled < 4:
                self.scaled += 0.1
                self.update
        else:
            if self.scaled > 0.5:
                self.scaled -= 0.1
                self.update
    # delete when finish
    def keyPressEvent(self, e):
        if e.key() == Qt.Key_C:
            print('exit...')
            os._exit(0)
        elif e.key() == Qt.Key_D:
            self.addRad(0, 0, toRad(5))
            self.update
        elif e.key() == Qt.Key_A:
            self.addRad(0, 0, toRad(-5))
            self.update
        elif e.key() == Qt.Key_W:
            self.setRad(0, toRad(60), toRad(-90))
            self.update
        elif e.key() == Qt.Key_S:
            self.setRad(0, toRad(-30), toRad(-90))
            self.update
        elif e.key() == Qt.Key_R:
            self.setRad(0, 0, 0)
            self.update
        elif e.key() == Qt.Key_Q:
            print(self.tf.rad_x)
            print(self.tf.rad_y)
            print(self.tf.rad_z)
            self.update
        elif e.key() == Qt.Key_K:
            if self.scaled < 2:
                self.scaled += 0.1
                self.update
        elif e.key() == Qt.Key_L:
            if self.scaled > 0.5:
                self.scaled -= 0.1
                self.update
        elif e.key() == Qt.Key_Control:
            self.ctl = True
        elif e.key() == Qt.Key_Z:
            current = self.getShot()
            self.setShot(current-1)
        elif e.key() == Qt.Key_X:
            current = self.getShot()
            self.setShot(current+1)

    def keyReleaseEvent(self, event):
        if event.key() == Qt.Key_Control:
            self.ctl = False

    def setRad(self, x, y, z):
        self.tf.rad_x = x
        self.tf.rad_x %= 2*pi
        self.tf.rad_y = y
        self.tf.rad_y %= 2*pi
        self.tf.rad_z = z
        self.tf.rad_z %= 2*pi

    def addRad(self, x, y, z):
        self.tf.rad_x += x
        self.tf.rad_x %= 2*pi
        self.tf.rad_y += y
        self.tf.rad_y %= 2*pi
        self.tf.rad_z += z
        self.tf.rad_z %= 2*pi
