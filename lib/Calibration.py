import cv2
import os
import glob
import tqdm
import threading
import numpy as np
import math
import typing
import shutil
from PyQt5.QtCore import pyqtSignal
from matplotlib import pyplot as plt

class Calibration:

    def __init__(self, resolution:typing.Tuple[int, int], video_folder, corner:int, block:int) -> None:
        self.resolution = resolution
        self.videoFolder = video_folder
        self.corner = corner
        self.block = block
        self.dataset = None
        self.pickedImg = []
        self.setProgressFn()

    def setProgressFn(self, label:pyqtSignal=None, value:pyqtSignal=None):
        self.progressLabel = label
        self.progressValue = value

    def _setProgressLabel(self, msg:str):
        if self.progressLabel is not None:
            self.progressLabel.emit(msg)

    def _setProgressValue(self, value:int):
        if self.progressValue is not None:
            self.progressValue.emit(value)

    def processVideo(self):
        """讀取影片並挑選好圖片"""
        self._readVideo()
        self._setProgressLabel("Reading images ...")
        self._readImages()
        self._setProgressLabel("Picking images ...")
        self._pickImage()

    def _readVideo(self) -> None:
        """Convert video frame to images"""
        videos = glob.glob(os.path.join(self.videoFolder, "chessboard_*.mp4"))

        frames = 0

        t = tqdm.tqdm(desc="Capturing images from video")

        for file in videos:
            vidcap = cv2.VideoCapture(file)

            fps = math.ceil(vidcap.get(cv2.CAP_PROP_FPS))
            offset = 1 if fps <= 10 else math.floor(fps/10)

            basename = os.path.basename(file).rstrip(".mp4")
            img_folder = os.path.join(self.videoFolder, f"frames_{basename}")

            # create folder if not exists
            if not os.path.isdir(img_folder):
                os.mkdir(img_folder)

            img_count = 0

            ret, image = vidcap.read()
            while ret:
                img_path = os.path.join(img_folder, f"{img_count:d}.jpg")
                cv2.imwrite(img_path, image)
                ret, image = vidcap.read()
                img_count += 1
                t.update(1)
                frames += 1
                vidcap.set(cv2.CAP_PROP_POS_FRAMES, img_count*offset) #skip
                self._setProgressLabel(f"Step1: converting video ...{frames} frames")

    def _readImages(self):
        """Read images from folder frames_{video name}/*.jpg"""

        self._setProgressLabel("Step2: Finding chessboard ...")

        img_folders = glob.glob(os.path.join(self.videoFolder, "frames_*/"))

        image_dataset:list[ImageData] = []

        img_paths = []
        for eles in [glob.glob(os.path.join(f, "*.jpg")) for f in img_folders]:
            img_paths += eles

        img_total = len(img_paths)

        progressbar = tqdm.tqdm(total=img_total, desc='Finding chessboard')

        core_count = os.cpu_count()

        img_paths = np.array_split(np.array(img_paths), round(core_count/2))

        lock = threading.Lock()
        count = 0

        def createImageData(img_paths):
            nonlocal count, lock
            for filepath in img_paths:

                image_dataset.append(ImageData(self, filepath))
                progressbar.update(1)
                with lock:
                    self._setProgressValue(round(count/img_total*100))
                    count+= 1

        threads = [threading.Thread(target=createImageData, args=[data]) for data in img_paths]

        for i in range(len(threads)):
            threads[i].start()

        for i in range(len(threads)):
            threads[i].join()

        self.dataset = ImageDataset(self, image_dataset, self.resolution)

    def clearUnusedData(self):
        """清除掉產生的中繼圖檔"""
        videos = glob.glob(os.path.join(self.videoFolder, "chessboard_*.mp4"))

        for file in videos:
            basename = os.path.basename(file).rstrip(".mp4")
            img_folder = os.path.join(self.videoFolder, f"frames_{basename}")
            shutil.rmtree(img_folder)

    def _pickImage(self):
        """pick image for calibrating"""
        self.pickedImg = self.dataset.popByMovement(30)
        #self.pickedImg = self.dataset.popByCustomRule(50)

    def savePickedImage(self) -> None:
        for i, img_data in enumerate(self.pickedImg):
            shutil.copy(img_data.filepath, os.path.join(self.videoFolder, f"chessboard_{i:04d}.jpg"))

    def _calibreate(self, objpoints, imgpoints):
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, self.resolution, None, None)

        mean_err = self._meanReprojectionError(objpoints, imgpoints, mtx, dist, rvecs, tvecs)

        return (ret, mean_err, mtx, dist, rvecs, tvecs)

    def _meanReprojectionError(self, objpoints, imgpoints, mtx, dist, rvecs, tvecs):
        # Calculating Re-projection Error
        mean_error = 0
        for i in range(len(objpoints)):
            imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
            error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2)/len(imgpoints2)
            mean_error += error
        return mean_error/len(objpoints)

    def calibrateCamera(self):

        objpoints, imgpoints = [], []

        objp = np.zeros((self.corner*self.corner, 3), np.float32)
        objp[:, :2] = np.mgrid[0:self.corner, 0:self.corner].T.reshape(-1, 2)
        objp = objp * self.block

        objpoints = [objp for _ in range(len(self.pickedImg))]
        imgpoints = [d.corners for d in self.pickedImg]

        return self._calibreate(objpoints, imgpoints)

    def calibrateCameraDebug(self):

        objp = np.zeros((self.corner*self.corner, 3), np.float32)
        objp[:, :2] = np.mgrid[0:self.corner, 0:self.corner].T.reshape(-1, 2)
        objp = objp * self.block

        data_t = self.pickedImg

        objpoints = [objp for _ in range(len(data_t))]
        imgpoints = [d.corners for d in data_t]

        debugWindow = DebugWindow()
        rms_errs, avg_errs = [], []

        for i in tqdm.tqdm(range(len(data_t)), "Calibrating"):
            rms_err, avg_err, mtx, dist, rvecs, tvecs = self._calibreate(objpoints[:i+1], imgpoints[:i+1])
            rms_errs.append(rms_err)
            avg_errs.append(avg_err)
            debugWindow.graph2d.plotErr(rms_errs, avg_errs)
            debugWindow.graph2d.plotImg(data_t[i].getImage(), data_t[i].corners)
            debugWindow.graph2d.flush()

        debugWindow.graph3d.drawCamera()
        for i in range(0, len(data_t)):
            #err
            imgpoints2, _ = cv2.projectPoints(objp, rvecs[i], tvecs[i], mtx, dist)
            err = cv2.norm(data_t[i].corners, imgpoints2, cv2.NORM_L2)/len(imgpoints2)

            top = 0.5
            color = (np.clip(err/top, 0, 1), np.clip((1-err/top), 0, 1), 0)
            debugWindow.graph3d.drawChessboard(objp, rvecs[i], tvecs[i], color)
        debugWindow.graph3d.flush()

        return (rms_err, avg_err, mtx, dist, rvecs, tvecs)

class DebugWindow:
    def __init__(self):
        self.graph2d = self.Graph2D()
        self.graph3d = self.Graph3D()

    class Graph2D:
        def __init__(self):
            self.fig, self.axs = plt.subplots(1, 3)
            self.fig.set_tight_layout(True)
            self.axs[2].set_title("Coverage")
            self.axs[2].invert_yaxis()

        def plotErr(self, rms_errs, avg_errs):
            self.axs[0].clear()
            self.axs[0].set_title("Error of each calibration")
            self.axs[0].set_xlabel("Iteration")
            self.axs[0].set_ylabel("Error")
            self.axs[0].plot(list(range(1, len(rms_errs)+1)), rms_errs, 'bo-', label="rms")
            self.axs[0].plot(list(range(1, len(avg_errs)+1)), avg_errs, 'ro-', label="avg")
            self.axs[0].legend()

        def plotImg(self, img, corners):
            c = np.array(corners).reshape((-1, 2))
            self.axs[1].clear()
            self.axs[1].set_title("Current Image")
            self.axs[1].plot(c[:, 0], c[:, 1], 'o-')
            self.axs[1].imshow(img)
            self.axs[2].plot(c[:, 0], c[:, 1], 'go-')

        def flush(self):
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
            while not self.fig.waitforbuttonpress(): pass

    class Graph3D:
        def __init__(self):
            self.fig = plt.figure()
            self.ax:plt.Axes = self.fig.add_subplot(projection='3d')
        def clear(self):
            self.ax.clear()
        def drawChessboard(self, obj, rvec, tvec, color):
            rotM, _ = cv2.Rodrigues(rvec)
            rt_matrix = np.concatenate([rotM, tvec], axis=1) # 3*4
            obj = np.concatenate([obj, np.ones((len(obj), 1))], axis=1)
            obj_c = np.matmul(rt_matrix, obj.T).T
            for i in range(0, 49, 7):
                self.ax.plot(obj_c[[i, i+6], 0], obj_c[[i, i+6], 1], obj_c[[i, i+6], 2], color=color, linestyle='-')
            for i in range(0, 7):
                self.ax.plot(obj_c[[i, i+42], 0], obj_c[[i, i+42], 1], obj_c[[i, i+42], 2], color=color, linestyle='-')
        def drawChessboards(self, obj_list, rvecs, tvecs):
            for i in range(len(obj_list)):
                obj = np.array(obj_list[i])
                self.drawChessboard(obj, rvecs[i], tvecs[i], 'g' if i < (len(obj_list) - 1) else 'b')
        def drawCamera(self):
            cam = np.array([
                [0, 0, 0],
                [7, 7, 14],
                [7, -7, 14],
                [-7, -7, 14],
                [-7, 7, 14],
                [7, 7, 14],
                ])

            # draw camera
            self.ax.plot(cam[:, 0], cam[:, 1], cam[:, 2], 'r-')
            self.ax.plot(cam[[0, 2], 0], cam[[0, 2], 1], cam[[0, 2], 2], 'r-')
            self.ax.plot(cam[[0, 3], 0], cam[[0, 3], 1], cam[[0, 3], 2], 'r-')
            self.ax.plot(cam[[0, 4], 0], cam[[0, 4], 1], cam[[0, 4], 2], 'r-')

        def flush(self):
            self.ax.axis('scaled')
            self.fig.canvas.flush_events()
            self.fig.canvas.draw()
            while not self.fig.waitforbuttonpress(): pass

class ImageDataset:
    def __init__(self, calib:Calibration, data, resolution):
        self.data:list[ImageData] = data
        self.resolution = resolution
        self.calib = calib
    def _sortByFrame(self):
        self.data.sort(key=lambda d:f"{d.video_name}.{d.video_frame_id}")

    def popByMovement(self, n):
        if len(self.data) == 0:
            return []

        i32_max = np.iinfo(np.int32(10)).max

        movements = np.zeros(len(self.data))
        movements[0] = i32_max
        self._sortByFrame()

        # 計算每張frame的移動距離
        for i in range(1, len(self.data)):
            prev_c = self.data[i-1].getMeanCenter()
            curr_c = self.data[i].getMeanCenter()

            if prev_c is None or curr_c is None:
                movements[i] = i32_max
            else:
                movements[i] = np.linalg.norm(prev_c-curr_c)

        # 選擇移動距離短的前50%
        selected = np.array(self.data)[movements < np.median(movements)]

        print("selected", len(selected))
        set1 = self.popByGrid(self._chooseByArea(0, 30, selected), 10)
        set2 = self.popByGrid(self._chooseByArea(30, 70, selected), 10)
        set3 = self.popByGrid(self._chooseByArea(70, 100, selected), 10)

        print("set1", len(set1))
        print("set2", len(set2))
        print("set3", len(set3))
        return set1 + set2 + set3

    def _filterByCustomRule(self):
        def filter_condition(img_data:ImageData):
            if img_data.corners is None:
                return False
            #if img_data.area < 0.005*(self.calib.resolution[0]*self.calib.resolution[0]):
            #    return False
            if img_data.blurScore > 200:
                return False
            return True
        return list(filter(filter_condition, self.data))
    def popByCustomRule(self, n):
        if len(self.data) == 0:
            return []
        selected = self._filterByCustomRule()

        print("selected", len(selected))
        set1 = self.popByGrid(self._chooseByArea(0, 30, selected), round(n/3))
        set2 = self.popByGrid(self._chooseByArea(30, 70, selected), round(n/3))
        set3 = self.popByGrid(self._chooseByArea(70, 100, selected), round(n/3))
        print("set1", len(set1))
        print("set2", len(set2))
        print("set3", len(set3))
        return set1 + set2 + set3
    def popByGrid(self, data, n):
        MARGIN_RATIO = 0
        GRID_SIZE = 100
        # 計算點陣數量，跟扣到邊緣的 pixel offset
        # grids count of width and height
        w_grids = math.floor(self.resolution[0]*(1-MARGIN_RATIO)/GRID_SIZE)
        h_grids = math.floor(self.resolution[1]*(1-MARGIN_RATIO)/GRID_SIZE)
        # offset pixel of width and height
        w_offset = math.floor((self.resolution[0]-w_grids*GRID_SIZE)/2)
        h_offset = math.floor((self.resolution[1]-h_grids*GRID_SIZE)/2)

        # grids index -> pixel coordinate
        grids = np.mgrid[0:w_grids+1, 0:h_grids+1].T.reshape(-1, 2) * GRID_SIZE
        grids[:, 0] += w_offset
        grids[:, 1] += h_offset
        # Reference: https://www.eecs.umich.edu/courses/eecs380/HANDOUTS/PROJ2/InsidePoly.html
        def isPointInside(polygon, x, y):
            """判斷點是否落在多邊形內部

            Args:
                polygon (list): 多邊形的每一個座標點
                x (float): 點的x座標
                y (float): 點的y座標

            Returns:
                bool: 是否落在多邊形內
            """
            xp = polygon[:, 0]
            yp = polygon[:, 1]
            i, j = 0, 0
            c = False
            j = len(polygon) - 1
            for i in range(0, len(polygon)):
                if ((((yp[i] <= y) and (y < yp[j])) or
                 ((yp[j] <= y) and (y < yp[i]))) and
                (x < (xp[j] - xp[i]) * (y - yp[i]) / (yp[j] - yp[i]) + xp[i])):
                    c = not c
                j = i
            return c
        def getCoveredGridIndexes(polygon) -> set:
            """從多邊形取得附近的grids座標

            Args:
                polygon (list): 多邊形的每一個座標點

            Returns:
                set: grids 的 indexes
            """

            # 左上角座標
            left_top: tuple = (np.min(polygon[:, 0].flatten()), np.min(polygon[:, 1].flatten()))
            # 右下角座標
            bottom_right: tuple = (np.max(polygon[:, 0].flatten()), np.max(polygon[:, 1].flatten()))

            def between(lower_bound, upper_bound, value) -> int:
                """限制 value 在 lower_bound 與 upper_bound 之間
                """
                return int(min(upper_bound, max(lower_bound, value)))

            # 計算距離最近的 grid points 座標
            grid_left_top = (
                between(0, w_grids, math.floor((left_top[0]-w_offset)/GRID_SIZE)),
                between(0, h_grids, math.floor((left_top[1]-h_offset)/GRID_SIZE)))
            grid_bottom_right = (
                between(0, w_grids, math.ceil((bottom_right[0]-w_offset)/GRID_SIZE)),
                between(0, h_grids, math.ceil((bottom_right[1]-h_offset)/GRID_SIZE)))

            grid_width: int = grid_bottom_right[0]-grid_left_top[0]+1

            index_begin = (grid_left_top[1])*(w_grids+1)+grid_left_top[0]
            index_end = (grid_bottom_right[1])*(w_grids+1)+grid_bottom_right[0]

            scan_indexes = set([x+y for x, y in np.mgrid[0:grid_width, index_begin:index_end:w_grids+1][::-1].T.reshape(-1, 2)])

            covered_grid_indexes = set()

            for index in scan_indexes:
                x, y = grids[index]

                if isPointInside(polygon[[0, 1, 3, 2, 0]], x, y):
                    covered_grid_indexes.add(index)

            return covered_grid_indexes

        # for each image data
        for d in data:
            # 該四邊形覆蓋到的 grid index
            d.covered = getCoveredGridIndexes(d.getQuadrilateralCorner())

        grid_covered = set()
        chosen = []

        while d:=self._popMaxCovered(data):
            if len(d.covered) == 0:
                break
            if len(chosen) >= n:
                break

            chosen.append(d)

            # 這張圖片覆蓋到的 grid index 都丟進去集合中
            grid_covered.update(d.covered)

            # 用差集排除掉已經有覆蓋到的 grid index
            for d in data:
                d.covered.difference_update(grid_covered)

        #if len(chosen) < n:
        #    chosen += random.sample(data, n - len(chosen))
        return chosen

    def _popMaxCovered(self, data):
        max_value = 0
        max_ele = None
        for d in data:
            if len(d.covered) > max_value:
                max_value = len(d.covered)
                max_ele = d
        if max_ele:
            data.remove(max_ele)
        return max_ele

    def _chooseByArea(self, lower_percentile, upper_percentile, data=None):
        data = data if data is not None else self.data

        if len(data) == 0:
            return []

        areas = [d.area for d in data]
        low = np.percentile(areas, lower_percentile)
        high = np.percentile(areas, upper_percentile)

        return [d for d in data if low <= d.area <= high]

class ImageData:

    def __init__(self, calib:Calibration, file_path:str):
        self.calib = calib
        self.filepath = file_path
        self.video_frame_id = int(os.path.splitext(os.path.basename(file_path))[0])
        self.video_name = os.path.basename(file_path.rstrip(f"/{self.video_frame_id}.jpg")).lstrip("frames_")
        img = self.getImage()
        self.corners = self._findChessboardCorners(img)
        self.blurScore = self._calculateBlurScore(img)
        self.area = self._calculateArea()
        self.covered = set()

    def _calculateArea(self) -> float:
        p = self.getQuadrilateralCorner()

        if p is None:
            return 0.0

        return (0.5) * (
            abs(np.cross(p[2]-p[0], p[1] - p[0]))
            + abs(np.cross(p[2]-p[3], p[1] - p[3])))

    def getImage(self):
        img = cv2.imread(self.filepath)
        img = cv2.resize(img, self.calib.resolution, interpolation=cv2.INTER_AREA)
        return img

    def _findChessboardCorners(self, img):
        """尋找棋盤格座標"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 先降低解析度，運算得比較快
        scaled_gray = cv2.resize(gray, (800, 600), interpolation=cv2.INTER_AREA)

        ret, corners = cv2.findChessboardCorners(scaled_gray, (self.calib.corner, self.calib.corner))

        if not ret:
            return None

        # for cornersubpix
        #criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        #corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

        # recover to original scale
        return np.array(corners)*(self.calib.resolution[0]/800)

    def _calculateBlurScore(self, img) -> float:
        """計算圖片模糊分數 (分數越低越模糊)"""
        if self.corners is None:
            return 0.0

        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        x, y, w, h = cv2.boundingRect(self.corners)
        crop_gray = gray[y:y+h, x:x+w]
        # adjust color
        crop_gray = cv2.convertScaleAbs(crop_gray, alpha=128/crop_gray.mean())

        # 計算圖片是否模糊 (分數越低越模糊)
        return cv2.Laplacian(crop_gray, cv2.CV_64F).var()

    def getQuadrilateralCorner(self):
        """取四邊形的四個角落座標"""
        if self.corners is None:
            return None
        index = [0,
                 self.calib.corner-1,
                 self.calib.corner*(self.calib.corner-1)-1,
                 self.calib.corner*self.calib.corner-1
                ]
        return self.corners[index].reshape(4, 2)
    def getMeanCenter(self):
        q = self.getQuadrilateralCorner()
        if q is None:
            return None
        return np.average(q, axis=0)
