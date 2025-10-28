#!/usr/bin/env python3
import logging
import ast
import os
import cv2
import pathlib
import shutil
import argparse
import numpy as np
from tqdm import tqdm
from lib.common import ROOTDIR, setIntrinsicMtx, loadConfig

logging.getLogger().setLevel(logging.DEBUG)

parser = argparse.ArgumentParser()
parser.add_argument("serial")

args = parser.parse_args()

serial = args.serial

def calculateIntrinsic(serial:str):
    """
    幫相機做棋盤格校正
    """

    cfgPath = f"{ROOTDIR}/Reader/Image_Source/config/{serial}.cfg"
    imgPath = f"{ROOTDIR}/Reader/Image_Source/intrinsic_data/{serial}/"

    if not os.path.exists(cfgPath):
        print(f"Config file not exists ({cfgPath})")
        return
    if not os.path.exists(imgPath):
        print(f"Image folder not exists ({imgPath})")
        return

    cameraCfg = loadConfig(cfgPath)

    block = int(cameraCfg['Other']['blockValue'])

    logging.debug(f"chessboard block size: {block}mm")

    if cameraCfg['Other']['cornerValue'].count(",") > 0:
        c = tuple(ast.literal_eval(cameraCfg['Other']['cornerValue']))
        corner = (c[0] - 1, c[1] - 1)
        logging.debug(f'chessboard corner: {corner}')
        objp = np.zeros((corner[0]*corner[1],3), np.float32)
        objp[:,:2] = np.mgrid[0:corner[0], 0:corner[1]].T.reshape(-1,2)
        objp = objp * block
    else:
        c = int(cameraCfg['Other']['cornerValue']) - 1
        corner = (c, c)
        logging.debug(f'chessboard corner: {corner} * {corner}')
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

    if len(images) == 0:
        logging.debug("Please Take 4 or more chessboards\n")
        return

    print('Start finding chessboard corners...')
    shutil.rmtree(f"{imgPath}/debug", ignore_errors=True)
    os.makedirs(f"{imgPath}/debug", exist_ok=True)
    for fname in tqdm(sorted(images)):
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, corner, None)

        # debug
        debug_img = cv2.drawChessboardCorners(img, corner, corners, ret)
        cv2.imwrite(f"{imgPath}/debug/{pathlib.Path(fname).stem}.png", debug_img)

        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)
        else:
            logging.debug(f"No points detected on : {fname}")
    img_size = (img.shape[1], img.shape[0])
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size,None,None)
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (img.shape[1],img.shape[0]), 0, (img.shape[1],img.shape[0]))

    # calculate mean error
    mean_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2)/len(imgpoints2)
        mean_error += error
    logging.debug( "\n[CameraIntrinsic] Reprojection average error: {}".format(mean_error/len(objpoints)) )


    print(f"ks = {str(mtx.tolist())}")
    print(f"dist = {str(dist.tolist())}")
    print(f"newcameramtx = {str(newcameramtx.tolist())}")

if __name__ == "__main__":
    calculateIntrinsic(serial)
