import argparse
import logging
import threading
import os
import time

from datetime import datetime

from LayerSensing.TrackNetManager import TrackNetManager
from lib.common import ROOTDIR
from LayerApplication.utils.Mqtt import MqttClient
from lib.Rpc import RemoteProcedureCall
from LayerCamera.camera import RpcCamera
from LayerSensing.RpcSensing import RpcSensing
from LayerContent.Rpc_content import RpcContent


def main():
    broker_ip = '140.113.213.131'
    mqtt = MqttClient(broker_ip, 1884)

    args = parse_args()

    if len(args.camera_idxs) != len(args.camera_device):
        raise ValueError("The number of camera indices must match the number of camera devices.")

    tracknet_managers = {}
    for idx, device in zip(args.camera_idxs, args.camera_device):
        tracknet_managers[idx] = TrackNetManager(device, mqtt.mqttc, None)

    threads = []

    for idx in args.camera_idxs:
        videopath, metapath = None, None

        if os.path.exists(f"{ROOTDIR}/replay/{args.date}/CameraReader_{idx}_ball.csv"):
            videopath = f"{ROOTDIR}/replay/{args.date}/CameraReader_{idx}_ball.csv"
        elif os.path.exists(f"{ROOTDIR}/replay/{args.date}/TrackNet_{idx}.csv"):
            videopath = f"{ROOTDIR}/replay/{args.date}/TrackNet_{idx}.csv"

        if os.path.exists(f"{ROOTDIR}/replay/{args.date}/CameraReader_{idx}_meta.csv"):
            metapath = f"{ROOTDIR}/replay/{args.date}/CameraReader_{idx}_meta.csv"

        if videopath == None:
            print('No TrackNet file')
        else:
            print('v:', videopath)
            print('m:', metapath)
            tracknet_managers[idx].startDatafeeder(videopath, metapath)

    for idx in args.camera_idxs:
        tracknet_managers[idx].stopDatafeeder()

    time.sleep(1)


def parse_args():
    parser = argparse.ArgumentParser(description="Run TrackNet data feeder for multiple cameras.")
    parser.add_argument("--date", required=True, help="The date directory name (e.g., 2024-09-19_09-33-56).")
    parser.add_argument("--camera_idxs", required=True, nargs='+', type=int, help="List of camera indices (e.g., 1 2 3).")
    parser.add_argument("--camera_device", required=True, nargs='+', help="List of camera device corresponding to the indices (e.g., 39320296 39320299).")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    main()