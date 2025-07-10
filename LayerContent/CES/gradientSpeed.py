from multiprocessing import current_process
import sys
import os
import numpy as np
import copy
import csv
from tqdm import tqdm

from sklearn import linear_model
from sklearn.cluster import DBSCAN
import shapely.geometry as geom
from skspatial.objects import Line, Point
from scipy.signal import savgol_filter
from scipy.optimize import minimize
import joblib

DIRNAME = os.path.dirname(os.path.abspath(__file__))
ROOTDIR = os.path.dirname(DIRNAME)
ROOTDIR = os.path.dirname(ROOTDIR)
REPLAYDIR = f"{ROOTDIR}/replay/"

#<<<<<<< Updated upstream
from LayerContent.CES.smooth import denoise, interpolatePoints
#=======
#from .smooth import denoise, interpolatePoints
#>>>>>>> Stashed changes

sys.path.insert(0, ROOTDIR)
sys.path.insert(0, os.path.join(ROOTDIR, 'lib'))
from lib.point import Point as P
from lib.writer import CSVWriter

sys.path.insert(0, ROOTDIR)
sys.path.insert(0, os.path.join(ROOTDIR, 'LayerApplication', 'ServeMachine', 'PC'))
from calcDis import physics_predict3d_v2

FPS = 120
TREND = FPS / 12
WINDOW_SIZE = 30


def loss_distance(V, original_trajectory, starting_point, fps, flight_time):
    total_loss = 0
    predicted_trajectory = physics_predict3d_v2(starting_point, V, fps, flight_time, touch_ground_cut=False, alpha=0.242)
    predicted_timestamps = predicted_trajectory[:, 3]
    for point in original_trajectory:
        if point.visibility == 1:
            original_timestamp = point.timestamp
            closest_idx = np.argmin(np.abs(predicted_timestamps - original_timestamp))
            if 0 <= closest_idx < len(predicted_trajectory):
                predicted_xyz = predicted_trajectory[closest_idx, :3]
                original_xyz = np.array([point.x, point.y, point.z])
                total_loss += np.linalg.norm(original_xyz - predicted_xyz)
    return total_loss


def optimize_velocity(original_trajectory, starting_point, fps, flight_time, initial_V, method='Nelder-Mead'):
    """Optimize the velocity V using scipy's minimize function."""
    # Nelder-Mead # BFGS
    result = minimize(
        loss_distance,  # The objective function to minimize
        initial_V,      # Initial guess for the velocity
        args=(original_trajectory, starting_point, fps, flight_time),  # Additional arguments for the loss function
        method=method   # Optimization method
    )
    
    if result.success:
        optimized_V = result.x
        print("Optimization successful:", optimized_V)
        return optimized_V
    else:
        print("Optimization failed:", result.message)
        return None



def gradient_speed(save_path, points, fps=120, flight_time=10):
    critical_frame_indices = [i for i, p in enumerate(points) if p.event != 0]
    
    # 條件 1: 事件數量正好為 3，且順序為 2 (serve) -> 1 (hit) -> 3 (dead)
    if (len(critical_frame_indices) == 3 and
        points[critical_frame_indices[0]].event == 2 and
        points[critical_frame_indices[1]].event == 1 and
        points[critical_frame_indices[2]].event == 3):
        print('===== event correct =====')
        start_idx = critical_frame_indices[1]  # hit
        hit_frame = int(points[start_idx].fid)
        end_idx = critical_frame_indices[-1]

    # 條件 2: 事件數量為 2，且事件為 2 (serve) -> 3 (dead)，中間有可見點，
    #         先找出區間中最大 Y 的點當 hit，再從 hit 後的區段找最小 Y 當 end_idx
    elif (len(critical_frame_indices) == 2 and
          points[critical_frame_indices[0]].event == 2 and
          points[critical_frame_indices[1]].event == 3):
        event_2_idx = critical_frame_indices[0]
        event_3_idx = critical_frame_indices[1]
        vis_indices = [i for i in range(event_2_idx + 1, event_3_idx)
                       if int(points[i].visibility) == 1]
        if vis_indices:
            start_idx = max(vis_indices, key=lambda i: float(points[i].y))
            hit_frame = int(points[start_idx].fid)
            vis_indices_after = [i for i in range(start_idx + 1, event_3_idx)
                                 if int(points[i].visibility) == 1]
            if vis_indices_after:
                end_idx = min(vis_indices_after, key=lambda i: float(points[i].y))
            else:
                end_idx = event_3_idx
            print(f'===== event warning : no hit event -> max Y -> Hit Frame: {hit_frame}, End Frame: {points[end_idx].fid} =====')
        else:
            print('===== event error: no visible point between serve and dead =====')
            return "Event Error"
    
    # 條件 3: event 數量大於 3，代表有多次擊球，取飛行方向由 Y 大到 Y 小且較長的軌跡段最為擊球段
    elif len(critical_frame_indices) > 3:
        print('===== event warning : Multiple hit detected =====', end=' ')
        event_2_indices = [i for i in critical_frame_indices if points[i].event == 2]
        event_1_indices = [i for i in critical_frame_indices if points[i].event == 1]
        event_3_index = next((i for i in critical_frame_indices if points[i].event == 3), None)
        sequences = []
        for i in range(len(event_1_indices)):
            current_idx = event_1_indices[i]
            next_idx = event_1_indices[i + 1] if i < len(event_1_indices)-1 else event_3_index
            if next_idx is None:
                continue
            current_y = float(points[current_idx].y)
            next_y = float(points[next_idx].y)
            if current_y - next_y > 0:
                sequences.append((current_idx, next_idx))
        if not sequences:
            print('===== event error: no valid sequence found =====')
            return "Event Error"
        longest_sequence = max(sequences, key=lambda seq: (seq[1] - seq[0], seq[0]))
        start_idx, end_idx = longest_sequence
        hit_frame = int(points[start_idx].fid)
        print(f'Using sequence: Start Frame={hit_frame}, End Frame={points[end_idx].fid} =====')
    
    # 不符合條件
    else:
        print('===== event error: invalid event sequence or count =====')
        return "Event Error"
    
   
    if end_idx < 0:
        end_idx = len(points) - 1
    
    if float(points[end_idx].y) > -0.5:
        print('===== dead ball error: dead point y > 0, ball did not cross the net =====')
        return "Ball did not cross the net"

    segment_points = points[start_idx:end_idx+1]
    denoised_points = denoise(segment_points)
    original_trajectory = denoised_points

    new_start_idx = 0
    for i in range(1, end_idx+1):
        if original_trajectory[i].visibility == 1:
            new_start_idx = i
            break
    gradient_start_idx = 0
    if original_trajectory[new_start_idx].y < original_trajectory[0].y:
        gradient_start_idx = new_start_idx
    gradient_start_frame = original_trajectory[gradient_start_idx].fid
    original_trajectory = original_trajectory[gradient_start_idx:end_idx+1]

    initial_V = np.array([0, -50, 0])
    starting_point = [original_trajectory[0].x, original_trajectory[0].y, original_trajectory[0].z,
                          original_trajectory[0].timestamp]
    
    optimized_V = optimize_velocity(original_trajectory, starting_point, fps, flight_time, initial_V)
    if optimized_V is None:
        print('===== gradient failure =====')
        return "Gradient Failure"
    
    smoothed_trajectory = physics_predict3d_v2(starting_point, optimized_V, fps, flight_time, alpha=0.242)

    new_points = []
    for p in points:
        if p.fid < gradient_start_frame:
            if p.event == 1:
                p.event = 0
            new_points.append(p)

    start_point = P(fid=str(gradient_start_frame), timestamp=str(smoothed_trajectory[0, 3]), visibility=1, x=smoothed_trajectory[0, 0], y=smoothed_trajectory[0, 1], z=smoothed_trajectory[0, 2], event=1)
    new_points.append(start_point)

    for i in range(1, len(smoothed_trajectory)):
        new_point = P(fid=str(gradient_start_frame + i), timestamp=str(smoothed_trajectory[i, 3]), visibility=1, x=smoothed_trajectory[i, 0], y=smoothed_trajectory[i, 1], z=smoothed_trajectory[i, 2], event=0)
        new_points.append(new_point)
    new_points[-1].event = 3

    output_path = os.path.join(save_path, 'Model3D_smooth_gradient.csv')
    csv3DWriter = CSVWriter(name='Model3D_smooth_gradient', filename=output_path)
    csv3DWriter.writePoints(new_points)
    
    return [start_point, optimized_V, new_points]
