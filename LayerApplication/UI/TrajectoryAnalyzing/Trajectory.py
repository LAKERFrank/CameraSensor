import sys

# Setting the Qt bindings for QtPy
import os
os.environ["QT_API"] = "pyqt5"

from qtpy import QtWidgets

import numpy as np

import pyvista as pv
from pyvistaqt import QtInteractor, MainWindow, BackgroundPlotter

from lib.point import Point

class TrajectoryWidget(QtWidgets.QFrame):
    def __init__(self):
        super().__init__()

        # create the frame
        vlayout = QtWidgets.QVBoxLayout()

        # add the pyvista interactor object
        self.plotter:pv.BasePlotter|QtInteractor = QtInteractor(auto_update=False)
        vlayout.addWidget(self.plotter.interactor)

        self.setLayout(vlayout)

        self.tracks:'dict[int, list[Point]]' = {}

        self.plotter.show_axes()

        self.meshes = []
        self.reset()

        self.plotter.set_background(color=(100, 100, 100))

    def setBackgroundGreen(self):
        self.plotter.set_background(color=(3, 102, 71))

    def set2DStyle(self):
        self.plotter.enable_parallel_projection()
        self.plotter.enable_2d_style()

    def render(self):
        # Create a MultiBlock dataset
        multi_blocks:'dict[tuple|str, pv.MultiBlock]' = {}

        for m, color in self.meshes:
            if color not in multi_blocks:
                multi_blocks[color] = pv.MultiBlock()

            multi_blocks[color].append(m)

        for c, mb in multi_blocks.items():
            self.plotter.add_mesh(mb, color=c)

    def reset(self):
        self.meshes.clear()
        self.tracks.clear()
        self.plotter.clear()
        self.drawCourt()
        self.drawNet()

    def addPointByTrackID(self, point:Point, id:int=0):
        if id not in self.tracks.keys():
            self.tracks[id] = []

        if len(self.tracks[id]) > 0:
            # draw line
            p1 = self.tracks[id][-1].toXYZ()
            p2 = point.toXYZ()
            line = pv.Line(resolution=1, pointa=p1, pointb=p2)
            self.meshes.append((line, 'white'))

        self.tracks[id].append(point)

        # draw point
        if point.color != 'white':
            sphere = pv.Sphere(center=point.toXYZ(), radius=0.1)
        else:
            sphere = pv.Sphere(center=point.toXYZ(), radius=0.03)

        self.meshes.append((sphere, point.color))

    def setCameraPosition(self, camera_position):
        self.plotter.camera_position = camera_position

    def setCameraZoom(self, zoom=1.0):
        self.plotter.camera.zoom(zoom)

    def drawCourt(self):
        BORDER=0.3
        court = {
            "CORNERS_3D_X": [-3.05, -3.01, -2.59, -2.55, -0.02, 0.02, 2.55, 2.59, 3.01, 3.05],
            "CORNERS_3D_Y": [ 6.7, 6.66, 5.94, 5.9, 2.02, 1.98, -1.98, -2.02, -5.9, -5.94, -6.66, -6.7]
        }

        # base plane
        box = pv.Box(bounds=(-(3.05+BORDER), 3.05+BORDER, -(6.7+BORDER), 6.7+BORDER, -0.11, -0.01))
        #plane = pv.Plane(i_resolution=1, j_resolution=1, i_size=(3.05+BORDER)*2, j_size=(6.7+BORDER)*2, center=(0, 0, -0.01))
        self.plotter.add_mesh(box, color=(3, 102, 71))

        lines = pv.MultiBlock()

        for x in [-3.03, -2.57, 2.57, 3.03]:
            plane = pv.Plane(i_resolution=1, j_resolution=1, i_size=0.04, j_size=6.7*2, center=(x, 0, 0))
            lines.append(plane)

        for y in [4.34, -4.34]:
            plane = pv.Plane(i_resolution=1, j_resolution=1, i_size=0.04, j_size=4.68, center=(0, y, 0))
            lines.append(plane)

        for y in [-6.68, -5.92, -2, 0, 2, 5.92, 6.68]:
            plane = pv.Plane(i_resolution=1, j_resolution=1, i_size=3.05*2, j_size=0.04, center=(0, y, 0))
            lines.append(plane)

        self.plotter.add_mesh(lines, color='white')

    def drawNet(self):
        pillars = pv.MultiBlock([
            pv.Cylinder(center=(3, 0, 1.55/2), direction=(0, 0, 1), radius=0.02, height=1.55),
            pv.Cylinder(center=(-3, 0, 1.55/2), direction=(0, 0, 1), radius=0.02, height=1.55),
        ])
        self.plotter.add_mesh(pillars, color=(153, 153, 0))

        cyl = pv.Cylinder(center=(0, 0, 1.53), direction=(1, 0, 0), radius=0.02, height=6)
        self.plotter.add_mesh(cyl, color=(255, 255, 255))

        plane = pv.Plane(i_resolution=1, j_resolution=1, i_size=0.72, j_size=6, direction=(0, 1, 0), center=(0, 0, 1.19))
        self.plotter.add_mesh(plane, color="black", opacity=0.3)
        
    def showEvent(self, e):
        #self.plotter.camera_position = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        return super().showEvent(e)

    def hideEvent(self, e):
        #self.plotter.close()
        return super().hideEvent(e)


class MyMainWindow(MainWindow):

    def __init__(self, parent=None, show=True):
        QtWidgets.QMainWindow.__init__(self, parent)

        # add the pyvista interactor object
        self.widget = TrajectoryWidget()

        self.setCentralWidget(self.widget)

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = MyMainWindow()
    window.show()
    sys.exit(app.exec_())