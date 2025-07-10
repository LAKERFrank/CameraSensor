'''
functions: common functions
'''
import os
import sys
import logging
import configparser
import cv2
import ast
import socket
from lib.Calibration import Calibration

import gi
gi.require_version("Gst", "1.0")

from gi.repository import GLib, Gst

from datetime import datetime
import numpy as np

DIRNAME = os.path.dirname(os.path.abspath(__file__))
ROOTDIR = os.path.dirname(DIRNAME)
ICONDIR = f"{ROOTDIR}/LayerApplication/UI/icon"
REPLAYDIR = f"{ROOTDIR}/replay"

# Hide debug log from PIL
logging.getLogger("PIL").setLevel(logging.WARNING)

def is_local_camera():
    return len(sys.argv) >= 2 and sys.argv[1] == "local"

def get_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.settimeout(0)
    try:
        # doesn't even have to be reachable
        s.connect(('10.254.254.254', 1))
        ip = s.getsockname()[0]
    except Exception:
        ip = '127.0.0.1'
    finally:
        s.close()
    return ip

def insertById(l, x):
    for i in range(len(l)):
        p = l[i]
        if p.fid > x.fid:
            l.insert(i, x)
            return
    l.append(x)

# save/load project_info into project_filename
def saveConfig(project_filename, project_info:configparser.ConfigParser):
    try:
        with open(project_filename, 'w') as configfile:
            project_info.write(configfile)
    except IOError as e:
        logging.error(e)
        sys.exit()

def loadConfig(cfg_file) -> configparser.ConfigParser:
    try:
        config = configparser.ConfigParser()
        config.optionxform = str
        with open(cfg_file) as f:
            config.read_file(f)
    except IOError as e:
        logging.error(e)
        sys.exit()
    return config

def loadNodeConfig(cfg_file, node_name):
    # loading configuartion file
    config = configparser.ConfigParser()
    try:
        settings = {}
        with open(cfg_file) as f:
            config.read_file(f)

        if config.has_section('Project'):
            for name, value in config.items('Project'):
                settings[name] = value
        if config.has_section(node_name):
            for name, value in config.items(node_name):
                settings[name] = value
        # setup Logging Level
        setupLogLevel(level=settings['logging_level'], log_file=node_name)
        return settings
    except IOError:
        logging.error("config file does not exist.")

def setupLogLevel(level, log_file):
    try:
        now = datetime.now()
        dt_string = now.strftime("%Y%m%d_%H%M%S")
        logPath = f"{ROOTDIR}/log/{dt_string}"
        if not os.path.isdir(logPath):
            os.makedirs(logPath)

        logFormatter = logging.Formatter("%(asctime)s %(levelname).1s %(lineno)03s: %(message)s")
        rootLogger = logging.getLogger()

        fileHandler = logging.FileHandler(f"{logPath}/{log_file}.log")
        fileHandler.setFormatter(logFormatter)
        rootLogger.addHandler(fileHandler)

        consoleHandler = logging.StreamHandler()
        consoleHandler.setFormatter(logFormatter)
        rootLogger.addHandler(consoleHandler)

        if level.lower() == "debug":
            rootLogger.setLevel(logging.DEBUG)
        elif level.lower() == "info":
            rootLogger.setLevel(logging.INFO)
        else:
            rootLogger.setLevel(logging.ERROR)
    except FileExistsError:
        logging.error("Can't open log directory")

def setIntrinsicMtxfromVideo(cfgPath, videoPath, size, debug=False):
    """
    Do camera calibration by video, and saves result to config file (cfgPath)

    Args:
        cfgPath (str): Path of config file
        videoPath (str): Directory Path of videos
        size (str): resolution of camera (ex: "(1920, 1080)")

    Returns:
        int: return 0 if successfully

    Reference: https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html
    """

    cameraCfg = loadConfig(cfgPath)
    resolution = ast.literal_eval(size)
    resolution = (resolution[0], resolution[1])
    # cornerValue: how many blocks on each border of squared chessboard 
    corner = int(cameraCfg['Other']['cornerValue']) - 1
    # block: If we know side length of each block, we can set blockValue
    #        (ex: set blockValue=15 for 15mm block)
    #        otherwise we could just set blockValue=1
    block = int(cameraCfg['Other']['blockValue'])
    logging.debug('chessboard_corner: {} * {}'.format(corner, corner))
    logging.debug('image_size: {} * {}'.format(*resolution))

    calib = Calibration(resolution, videoPath, corner, block)
    calib.processVideo()
    calib.savePickedImage()
    rms_err, avg_err, mtx, dist, rvecs, tvecs = calib.calibrateCamera()
    calib.clearUnusedData()
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, resolution, 0, resolution)
    logging.info('\ninstrinsic:\n{}\ndist:\n{}\nnewcameramtx:\n{}'.format(mtx, dist, newcameramtx))

    # Calculating Re-projection Error
    logging.debug(f"\n[CameraIntrinsic] Reprojection average error: {avg_err}")

    try:
        cameraCfg['Other']['ks'] = str(mtx.tolist())
        cameraCfg['Other']['dist'] = str(dist.tolist())
        cameraCfg['Other']['newcameramtx'] = str(newcameramtx.tolist())
        saveConfig(cfgPath, cameraCfg)
    except:
        logging.warning("Modify Camera Intrinsic Matrix Failed!")
        return -1
    else:
        logging.info("Modify Camera Intrinsic Matrix Successfully!")
        return 0

def setIntrinsicMtx(cfgPath:str, imgPath:str, resolution: 'tuple[int, int]'):
    """
    幫相機做棋盤格校正，再將校正結果儲存回 config file

    Args:
        cfgPath (str): Path of config file
        imgPath (str): Directory Path of images
        size (str): resolution of camera (ex: "(1920, 1080)")

    Returns:
        int: return 0 if successfully
    """

    cameraCfg = loadConfig(cfgPath)
    corner = int(cameraCfg['Other']['cornerValue']) - 1
    block = int(cameraCfg['Other']['blockValue'])
    logging.debug(f'chessboard_corner: {corner} * {corner}')
    logging.debug(f'image_size: {resolution[0]} * {resolution[1]}')

    objp = np.zeros((corner*corner,3), np.float32)
    objp[:,:2] = np.mgrid[0:corner, 0:corner].T.reshape(-1,2)
    objp = objp * block

    objpoints = [] # 3d points in real world space
    imgpoints = [] # 2d points in image plane.
    images = []
    for filename in os.listdir(imgPath):
        fullpath = os.path.join(imgPath, filename)
        if filename.startswith('chessboard_') and filename.endswith('.jpg'):
            images.append(fullpath)
    if len(images) < 4:
        logging.debug("Please Take 4 or more chessboards\n")
        return -2
    print('Start finding chessboard corners...')
    for idx, fname in enumerate(images):
        img = cv2.imread(fname)
        img = cv2.resize(img, (resolution[0], resolution[1]), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (corner, corner), None)
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)
        else:
            logging.debug('no points')
    img_size = (img.shape[1], img.shape[0])
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size,None,None)
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (img.shape[1],img.shape[0]), 0, (img.shape[1],img.shape[0]))
    logging.info('\ninstrinsic:\n{}\ndist:\n{}\nnewcameramtx:\n{}'.format(mtx, dist, newcameramtx))
    mean_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2)/len(imgpoints2)
        mean_error += error
    logging.debug( "\n[CameraIntrinsic] Reprojection average error: {}".format(mean_error/len(objpoints)) )
    try:
        cameraCfg['Other']['ks'] = str(mtx.tolist())
        cameraCfg['Other']['dist'] = str(dist.tolist())
        cameraCfg['Other']['newcameramtx'] = str(newcameramtx.tolist())
        saveConfig(cfgPath, cameraCfg)
    except:
        logging.warning("Modify Camera Intrinsic Matrix Failed!")
        return -1
    else:
        logging.info("Modify Camera Intrinsic Matrix Successfully!")
        return {
            'ks': mtx,
            'dist': dist,
            'newcameramtx': newcameramtx
        }

def loadCameraSetting(camera):
    config_file = f"{ROOTDIR}/Reader/{camera.brand}/config/{camera.hw_id}.cfg"
    return loadConfig(config_file)

def saveCameraSetting(camera, cfg):
    config_file = f"{ROOTDIR}/Reader/{camera.brand}/config/{camera.hw_id}.cfg"
    image_path = f"{ROOTDIR}/Reader/{camera.brand}/intrinsic_data/{camera.hw_id}"
    saveConfig(config_file, cfg)
    return setIntrinsicMtx(config_file, image_path, cfg['Camera']['RecordResolution'])

def checkAUXorBUX(serialnumber):
    Gst.init(sys.argv)
    sample_pipeline = Gst.parse_launch("tcambin name=source ! fakesink")

    source = sample_pipeline.get_by_name("source")

    (return_value, model,
        identifier, connection_type) = source.get_device_info(serialnumber)
    print("return_value: ",return_value)
    
    if(return_value):
        modelName = model[6:9]
    else:
        modelName = None

    return modelName #AUX or BUX or None