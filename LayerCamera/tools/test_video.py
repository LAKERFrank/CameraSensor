import cv2
import os
import argparse
import tqdm
import numpy as np
import matplotlib
import pandas as pd
from pathlib import Path
import matplotlib.colors as mcolors
from matplotlib import pyplot as plt

from lib.common import ROOTDIR

FPS = 120

from LayerCamera.kf import SimpleKalmanFilter

"""查看影片是否有掉楨

Usage:
    # 查看replay資料夾下最新的一部影片
    python3 test_video.py

    # 指定資料夾
    python3 test_video.py --dir /home/nol/camerasensor/replay/2024-12-18_21-03-30

"""

def read_csv(path:str, cam_idx: int):
    pts = []
    i = 0

    meta_csv = pd.read_csv(path[:-4]+"_meta.csv")

    progressbar = tqdm.tqdm(desc=path, total=len(meta_csv.index))

    for index, row in meta_csv.iterrows():
        try:
            # 忽略前10個frame (較不穩定)
            if index >= 10:
                pts.append({
                    "cam_idx": cam_idx,
                    "index": i,
                    "timestamp": row.timestamp,
                    "frame": "",
                })
        except Exception as e:
            print(e)
        progressbar.update()
        i += 1
    progressbar.close()
    return pts

def read_video(path:str, cam_idx: int):
    pts = []
    cap = cv2.VideoCapture(path)
    i = 0

    frame_save_dir = Path(path).with_suffix("")
    os.makedirs(frame_save_dir, exist_ok=True)

    meta_csv = None
    if os.path.exists(path[:-4]+"_meta.csv"):
        meta_csv = pd.read_csv(path[:-4]+"_meta.csv")

    progressbar = tqdm.tqdm(desc=path, total=cap.get(cv2.CAP_PROP_FRAME_COUNT))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_loc = frame_save_dir / Path(f"{i}.png")

        if args.sync and not frame_loc.exists():
            cv2.imwrite(str(frame_loc), frame)

        try:
            timestamp = cap.get(cv2.CAP_PROP_POS_MSEC)
            if meta_csv is not None:
                timestamp = meta_csv.iloc[i].timestamp * 1000
            # 忽略前10個frame (較不穩定)
            if i >= 10:
                pts.append({
                    "cam_idx": cam_idx,
                    "index": i,
                    "timestamp": timestamp / 1000,
                    "frame": frame_loc,
                })
        except Exception as e:
            print(e)
        progressbar.update()
        i += 1
    cap.release()
    progressbar.close()
    return pts

def statistics_3(data: dict, ax1:plt.Axes, ax2:plt.Axes):

    k = list(data.keys())[0]

    ax1.set_title(f"Timestamp Deviation Relative to Reference Camera (Cam {k})")
    ax1.set_xlabel("Timestamp (s)")
    ax1.set_ylabel("Deviation (ms)")

    ax2.set_title(f"Timestamp Deviation Relative to Reference Camera (Cam {k}) (Smoothed with KF)")
    ax2.set_xlabel("Timestamp (s)")
    ax2.set_ylabel("Deviation (ms)")


    points = []
    for d in data.values():
        points += d

    summary = dict([(i, []) for i in data.keys()])
    summary_t = dict([(i, []) for i in data.keys()])

    current_i = dict(zip(data.keys(), [0 for _ in range(len(data))]))
    current_i.pop(k)
    master_i = 0
    touched_i = dict(zip(data.keys(), [0 for _ in range(len(data))]))
    touched_i.pop(k)

    group = {}
    group_t = None
    stop = False

    # 初始對其
    for cam_i, frame_i in current_i.items():

        curr_t = data[cam_i][frame_i]["timestamp"]

        # skip previous
        while curr_t < (data[k][0]["timestamp"] - 1/(FPS*2)):
            current_i[cam_i] += 1
            frame_i = current_i[cam_i]
            curr_t = data[cam_i][frame_i]["timestamp"] 

    while True:
        if master_i >= len(data[k]):
            break

        group_t = data[k][master_i]["timestamp"]
        master_i += 1

        for cam_i, frame_i in current_i.items():
            # verify length
            if frame_i >= len(data[cam_i]):
                stop = True
                break

            curr_t = data[cam_i][frame_i]["timestamp"]

            if (touched_i[cam_i] > 0):
                touched_i[cam_i] -= 1

            # upper bound
            if curr_t > (group_t + 1/(FPS*2)):
                touched_i[cam_i] = 120*30 # 30sec
                # do nothing
                continue
            elif curr_t < (group_t - 1/(FPS*2)) and touched_i[cam_i] == 0:
                print("touched lower")
                while curr_t < (group_t + 1/(FPS*2)): # skip to next
                    print("skip_t", curr_t - group_t)
                    current_i[cam_i] += 1
                    curr_t = data[cam_i][current_i[cam_i]]["timestamp"]
                touched_i[cam_i] = 120*30
            else:
                # 符合時間條件
                group[cam_i] = curr_t
                current_i[cam_i] += 1

        for cam_i in list(group.keys()):
            offset = group[cam_i] - group_t
            summary[cam_i].append(offset*1000)
            summary_t[cam_i].append(group_t)
        summary[k].append(0)
        summary_t[k].append(group_t)

        group = {}
        group_t = None

        if stop:
            break


    first_timestamp = min([d[0]["timestamp"] for d in data.values()])

    #points.sort(key=lambda x: x["timestamp"])
    #first_timestamp = points[0]["timestamp"]

    #current_f = {}
    #current_t = None

    #i = 0
    #while i < len(points):
    #    current_t = points[i]['timestamp']
    #    if len(current_f) == 0 or \
    #        (abs(current_t - min([x["timestamp"] for x in current_f.values()])) < 1/(FPS*2)): # range在1/240秒以內，視為一組
    #        current_f[points[i]["cam_idx"]] = points[i]
    #        i += 1
    #    else:
    #        # calculate
    #        if k in current_f:
    #            main_timestamp = current_f[k]["timestamp"]
    #            for s in list(current_f.keys()):
    #                offset = current_f[s]["timestamp"] - main_timestamp
    #                summary[s].append(offset*1000)
    #                summary_t[s].append(main_timestamp)
    #        current_f = {}

    for s in list(data.keys()):
        ax1.plot(summary_t[s] - first_timestamp, summary[s], '.-', label=f"Cam {s}")

    #----- KF

    points = []

    for d in data.values():
        points += d

    summary = dict([(i, []) for i in data.keys()])
    summary_t = dict([(i, []) for i in data.keys()])

    points.sort(key=lambda x: x["timestamp"])
    first_timestamp = points[0]["timestamp"]

    current_f = {}
    current_t = None

    i = 0
    while i < len(points):
        current_t = points[i]['kf_timestamp']
        if len(current_f) == 0 or \
            (abs(current_t - min([x["kf_timestamp"] for x in current_f.values()])) < 1/(FPS*2)): # range在1/240秒以內，視為一組
            current_f[points[i]["cam_idx"]] = points[i]
            i += 1
        else:
            # calculate
            if k in current_f:
                main_timestamp = current_f[k]["kf_timestamp"]
                for s in list(current_f.keys()):
                    offset = current_f[s]["kf_timestamp"] - main_timestamp
                    summary[s].append(offset*1000)
                    summary_t[s].append(main_timestamp)
            current_f = {}

    for s in list(data.keys()):
        ax2.plot(summary_t[s] - first_timestamp, summary[s], '.-', label=f"Cam {s}")

    ax1.set_ylim(-5, 5)

    ax1.grid()
    ax2.grid()

    ax1.legend()
    ax2.legend()

def statistics_2(data: dict, ax:plt.Axes, ax2:plt.Axes):

    # graph 1 : Offset related to 1/fps sec
    k = list(data.keys())[0]
    start_ts = np.max([x[0]['timestamp'] for x in data.values()])
    avg_gap = np.mean([np.mean(np.diff([x['timestamp'] for x in d[:1000]])) for d in data.values()])
    standard_timestamps = [ avg_gap*i for i in range(max([len(x) for x in data.values()]))]

    for cam_idx, d in data.items():
        frame_timestamps = np.array([x['timestamp'] for x in d])

        # synchronize first frame
        i = 0
        while abs(frame_timestamps[i] - start_ts) > (avg_gap/2):
            i += 1
        frame_timestamps = frame_timestamps[i:]

        # plot
        offset = frame_timestamps - (standard_timestamps[:len(frame_timestamps)] + frame_timestamps[0])
        ax.plot(frame_timestamps-start_ts, offset*1000, '-', label=f"Cam {cam_idx}")

    ax.set_title(f"Offset related to {1/avg_gap:.3f} fps")
    ax.set_xlabel("Timestamp (sec)")
    ax.set_ylabel("Offset (ms)")
    ax.legend()

    # graph 2 : Offset related to Camera 0
    k = list(data.keys())[0]

    points = []
    for d in data.values():
        points += d

    summary = dict([(i, []) for i in data.keys()])
    summary_t = dict([(i, []) for i in data.keys()])

    points.sort(key=lambda x: x["timestamp"])
    first_timestamp = points[0]["timestamp"]

    current_f = {}
    current_t = None

    i = 0
    while i < len(points):
        current_t = points[i]['timestamp']
        if len(current_f) == 0 or \
            (abs(current_t - min([x["timestamp"] for x in current_f.values()])) < 1/(FPS*2)): # range在1/240秒以內，視為一組
            current_f[points[i]["cam_idx"]] = points[i]
            i += 1
        else:
            # calculate
            if k in current_f:
                main_timestamp = current_f[k]["timestamp"]
                for s in list(current_f.keys()):
                    offset = current_f[s]["timestamp"] - main_timestamp
                    summary[s].append(offset*1000)
                    summary_t[s].append(main_timestamp)
            current_f = {}

    for s in list(data.keys()):
        ax2.plot(summary_t[s] - first_timestamp, summary[s], '.', label=f"Cam {s}")
    ax2.set_title(f"Offset related to Cam {k}")
    ax2.set_xlabel("Timestamp (sec)")
    ax2.set_ylabel("Offset (ms)")
    ax2.legend()

def show_statistics(data:dict):
    #fig, axs = plt.subplots(2, 2)
    ax0 = plt.subplot(221)
    ax1 = plt.subplot(222, sharex=ax0, sharey=ax0)
    statistics_1(data, ax0, ax1)

    ax2 = plt.subplot(223)
    ax3 = plt.subplot(224, sharex=ax2, sharey=ax2)
    statistics_3(data, ax2, ax3)
    plt.tight_layout()
    plt.show()

def statistics_1(data: dict, ax1:plt.Axes, ax2:plt.Axes):

    ax1.set_title("Frame Interval Jitter")
    ax1.set_xlabel('Timestamp (s)')
    ax1.set_ylabel("Inter-frame Interval (ms)")

    ax2.set_title("Frame Interval Jitter (Smoothed with KF)")
    ax2.set_xlabel('Timestamp (s)')
    ax2.set_ylabel("Inter-frame Interval (ms)")

    ax1.set_prop_cycle(color=mcolors.TABLEAU_COLORS.values())
    ax2.set_prop_cycle(color=mcolors.TABLEAU_COLORS.values())

    offset = min([d[0]["timestamp"] for _, d in data.items()])

    for idx, d in data.items():
        x = np.array([row["timestamp"] for row in d[1:]])
        y = np.array([row["dt"] * 1000 for row in d[1:]])
        ax1.plot(x-offset, y, '.-', label=f"Cam {idx} $\mu={np.mean(y):.6f},\sigma={np.std(y):.6f}$")

        x = np.array([row["kf_timestamp"] for row in d[1:]])
        y = np.array([row["kf_dt"] * 1000 for row in d[1:]])
        ax2.plot(x-offset, y, '.-', label=f"Cam {idx} $\mu={np.mean(y):.6f},\sigma={np.std(y):.6f}$")

    ax1.grid()
    ax2.grid()

    ax1.legend()
    ax2.legend()

    order = sorted(range(len(data)), key=lambda x: np.mean([d["timestamp"] for d in data[x]]), reverse=True)

    # change legend order (ax1)
    handles, labels = ax1.get_legend_handles_labels()
    ax1.legend([handles[idx] for idx in order], [labels[idx] for idx in order])

    # change legend order (ax2)
    handles, labels = ax2.get_legend_handles_labels()
    ax2.legend([handles[idx] for idx in order], [labels[idx] for idx in order])

def sync_frame(data: dict):

    points = []
    for d in data.values():
        points += d

    sync_frames = []

    points.sort(key=lambda x: x["timestamp"])

    current_f = []

    i = 0
    while i < len(points):
        if len(current_f) == 0 or \
            (abs(current_f[0]['timestamp'] - points[i]['timestamp']) < 1/(FPS*2)):
            current_f.append(points[i])
            i += 1
        else:
            sync_frames.append(current_f)
            current_f = []

    save_dir = Path(root_dir) / "sync"

    os.makedirs(str(save_dir), exist_ok=True)

    def putText(img, text, pos):
        return cv2.putText(img, text,
                            pos, cv2.FONT_HERSHEY_SIMPLEX, 
                            1, (0, 0, 255), 2, cv2.LINE_AA)

    progressbar = tqdm.tqdm(desc="Generating sync frames ...", total=len(sync_frames))
    for i, fs in enumerate(sync_frames):
        progressbar.update()
        fs.sort(key=lambda x: x["cam_idx"])
        imgs = []
        for d in fs:
            image = cv2.imread(str(d["frame"]))
            image = putText(image, f"Camera_{d['cam_idx']}", (50, 50))
            image = putText(image, f"Frame Index: {d['index']}", (50, 90))
            image = putText(image, f"Timestamp: {d['timestamp']:.3f}", (50, 130))
            imgs.append(image)

        vis = np.concatenate(imgs, axis=1)
        cv2.imwrite(f"{save_dir}/{i}.png", vis)

parser = argparse.ArgumentParser()
parser.add_argument("--dir")
parser.add_argument("--sync-stat", action='store_true')
parser.add_argument("--sync", action='store_true')
args = parser.parse_args()

root_dir = ""

videos = {}

if args.dir:
    root_dir = args.dir
    for i in range(12):
        p = f"{args.dir}/CameraReader_{i}.mp4"
        m = f"{args.dir}/CameraReader_{i}_meta.csv"
        if os.path.exists(p) or os.path.exists(m):
            videos[i] = p
else:
    last = sorted(os.listdir(f"{ROOTDIR}/replay"), reverse=True)[0]
    root_dir = f"{ROOTDIR}/replay/{last}"
    for i in range(12):
        p = f"{ROOTDIR}/replay/{last}/CameraReader_{i}.mp4"
        m = f"{ROOTDIR}/replay/{last}/CameraReader_{i}_meta.csv"
        if os.path.exists(p) or os.path.exists(m):
            videos[i] = p

if len(videos) == 0:
    print(f"No video found at {root_dir}")
    exit(1)

data = {}

if args.sync:
    for idx, path in videos.items():
        data[idx] = read_video(path, idx)
    sync_frame(data)
else:
    for idx, path in videos.items():
        data[idx] = read_csv(path, idx)

        kf = SimpleKalmanFilter(data[idx][0]['timestamp'], 1/120)

        data[idx][0]['dt'] = 0
        data[idx][0]['kf_timestamp'] = data[idx][0]['timestamp']
        data[idx][0]['kf_dt'] = 0

        for j in range(1, len(data[idx])):
            measurement = data[idx][j]['timestamp']
            # cal dt
            data[idx][j]['dt'] = data[idx][j]['timestamp'] - data[idx][j-1]['timestamp']
            # kf
            data[idx][j]['kf_timestamp'] = kf.timestamp + kf.dt
            data[idx][j]['kf_dt'] = kf.dt
            kf.step(measurement)

    show_statistics(data)
