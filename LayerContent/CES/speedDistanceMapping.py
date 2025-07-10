import argparse
import json
import logging
import os
import sys
import threading
import time
from datetime import datetime
from typing import Optional
import csv

import cv2
import numpy as np
import pandas as pd

DIRNAME = os.path.dirname(os.path.abspath(__file__))
ROOTDIR = os.path.dirname(DIRNAME)
ROOTDIR = os.path.dirname(ROOTDIR)
print(ROOTDIR)

sys.path.insert(0, ROOTDIR)
sys.path.insert(0, os.path.join(ROOTDIR, 'LayerApplication', 'ServeMachine', 'PC'))
sys.path.append(f"{ROOTDIR}/lib")
from point import Point
from calcDis import physics_predict3d_v2


def find_last_greater_than_dis(trajectories, dis):
    last_index = -1
    for i, row in enumerate(trajectories):
        if row[1] > dis:
            last_index = i
    return last_index

def simulate_trajectories(starting_point, v_base, dis, vy_range, fps, touch_ground_cut, alpha, cam):
    if dis > 0:
        dis = -dis
    results = []

    for vy in vy_range:
        v = [v_base[0], -vy / 3.6, v_base[2]]
        trajectories = physics_predict3d_v2(starting_point, v, fps, touch_ground_cut=touch_ground_cut, alpha = alpha)
        last_index = find_last_greater_than_dis(trajectories, dis)

        if last_index != -1:
            results.append({
                "Vy": vy,
                "Framecount": last_index,
                "Trajectory": trajectories[last_index].tolist()
            })
        else:
            results.append({
                "Vy": vy,
                "Framecount": -1,
                "Trajectory": []
            })

    results = pd.DataFrame(results)
    print(results)
    output_path = os.path.join(ROOTDIR, 'LayerContent', 'CES', 'speed_mapping')
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    results.to_csv(f'/{output_path}/{cam}_{-dis}.csv', index=False)

def main():
    parser = argparse.ArgumentParser(description="Simulate trajectories and generate speed mapping CSV.")
    parser.add_argument("--dis", type=float, required=True, help="The distance threshold (negative value).")
    parser.add_argument("--cam", type=int, required=True, help="The camera ID.")

    args = parser.parse_args()

    starting_point = [0.0, 0.0, 0.0, 0.0]
    v_base = [0.0, 0.0, 0.0]
    fps = 120.0
    touch_ground_cut = False
    alpha = 0.242
    dis = args.dis
    cam = args.cam

    vy_range = range(30, 301, 10)  # km/h

    simulate_trajectories(starting_point, v_base, dis, vy_range, fps, touch_ground_cut, alpha, cam)
   
if __name__ == "__main__":
    main()

