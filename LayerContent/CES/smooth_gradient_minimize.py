import sys
import os
import numpy as np
import csv

from scipy.optimize import minimize

from lib.common import ROOTDIR
from lib.point import Point as P

from LayerContent.smooth import denoise, interpolatePoints

sys.path.insert(0, ROOTDIR)
sys.path.insert(0, os.path.join(ROOTDIR, 'LayerApplication', 'ServeMachine', 'PC'))
from calcDis import physics_predict3d_v2

REPLAYDIR = f"{ROOTDIR}/replay/"

FPS = 120
TREND = FPS / 12
WINDOW_SIZE = 30


def loss_distance(V, original_trajectory, starting_point, fps, flight_time):
    total_loss = 0
    predicted_trajectory = physics_predict3d_v2(starting_point, V, fps, flight_time, touch_ground_cut=False, alpha=0.242)
    predicted_timestamps = predicted_trajectory[:, 3]
    for i in range(len(original_trajectory)):
        if int(original_trajectory[i]['Visibility']) != 0:
            original_timestamp = float(original_trajectory[i]['Timestamp'])
            closest_idx = np.argmin(np.abs(predicted_timestamps - original_timestamp))
            if 0 <= closest_idx < len(predicted_trajectory):
                predicted_xyz = predicted_trajectory[closest_idx, :3]
                original_xyz = np.array([float(original_trajectory[i]['X']), float(original_trajectory[i]['Y']), float(original_trajectory[i]['Z'])])
                total_loss += np.linalg.norm(original_xyz - predicted_xyz)
    return total_loss


def optimize_velocity(original_trajectory, starting_point, fps, flight_time, initial_V, method='Nelder-Mead'):
    """Optimize the velocity V using scipy's minimize function."""
    result = minimize(
        loss_distance,  # The objective function to minimize
        initial_V,      # Initial guess for the velocity
        args=(original_trajectory, starting_point, fps, flight_time),  # Additional arguments for the loss function
        method=method,   # Optimization method
        options={'maxiter': 5000}
    )
    
    if result.success:
        optimized_V = result.x
        print("Optimization successful:", optimized_V)
        return optimized_V
    else:
        print("Optimization failed:", result.message)
        return None


def smoothByEvent_gradient_minimize(date, fps=120, flight_time=10):
    file = 'Model3D_event_1.csv'
    file_path = os.path.join(REPLAYDIR, date, file)
    output_file = 'Model3D_smooth_gradient.csv'
    output_path = os.path.join(REPLAYDIR, date, output_file)
    
    with open(file_path, 'r', newline='') as csvFile:
        rows = list(csv.DictReader(csvFile))
        critical_frame = [i for i, row in enumerate(rows) if int(row['Event']) != 0]
        end_idx = -1

        # 條件 1: event 數量為 3，且 event 順序為 2(serve) -> 1(hit) -> 3(dead)
        if len(critical_frame) == 3 and int(rows[critical_frame[0]]['Event']) == 2 and int(rows[critical_frame[1]]['Event']) == 1 and int(rows[critical_frame[2]]['Event']) == 3:
            print('===== event correct =====')
            start_idx = critical_frame[1]
            hit_frame = int(rows[start_idx]['Frame'])
            end_idx = critical_frame[-1]

        # 條件 2: event 數量為 2，且 event 為 2(serve) -> 3(dead)，中間有 visible 點，找到 Max Y 的 Frame 當成 hit
        elif (len(critical_frame) == 2 and int(rows[critical_frame[0]]['Event']) == 2 and int(rows[critical_frame[1]]['Event']) == 3):
            event_2_idx = critical_frame[0]
            event_3_idx = critical_frame[1]
            # 尋找 Event=2 到 Event=3 範圍內 Visibile 的點
            vis_indices = [i for i in range(event_2_idx + 1, event_3_idx) if rows[i]['Visibility'] == '1']
            if vis_indices:
                start_idx = max(vis_indices, key=lambda i: float(rows[i]['Y']))  # 找 Max Y 的 index
                hit_frame = int(rows[start_idx]['Frame'])
                critical_frame.insert(1, start_idx)
                # end_idx = critical_frame[-1]
                vis_indices = [i for i in range(start_idx + 1, event_3_idx+1) if rows[i]['Visibility'] == '1']
                if vis_indices:
                    end_idx = min(vis_indices, key=lambda i: float(rows[i]['Y']))
                    critical_frame[2] = end_idx
                    end_frame = int(rows[end_idx]['Frame'])
                else:
                    end_idx = critical_frame[-1]
                    end_frame = int(rows[end_idx]['Frame'])
                print(f'===== event warning : no hit event -> max Y -> Hit Frame: {hit_frame}, End Frame: {end_frame} =====')
            else:
                print('===== event error: no visible point between serve and dead =====')
                return 0
            
        # 條件 3: event 數量大於 3，代表有多次擊球，取飛行方向由 Y 大到 Y 小且較長的軌跡段最為擊球段
        elif len(critical_frame) > 3:
            print('===== event warning : Multiple hit detected -> ', end='')
            event_2_idx = [i for i in critical_frame if int(rows[i]['Event']) == 2]
            event_1_idx = [i for i in critical_frame if int(rows[i]['Event']) == 1]
            event_3_idx = next((i for i in critical_frame if int(rows[i]['Event']) == 3), None)
            sequences = []
            for i in range(len(event_1_idx)):
                current_idx = event_1_idx[i]
                next_idx = event_1_idx[i + 1] if i < len(event_1_idx)-1 else event_3_idx
                current_y = float(rows[current_idx]['Y'])
                next_y = float(rows[next_idx]['Y'])
                if current_y - next_y > 0:
                    sequences.append((current_idx, next_idx))
            longest_sequence = max(sequences, key=lambda seq: (seq[1] - seq[0], seq[0]), default=None)
            start_idx, end_idx = longest_sequence
            hit_frame = int(rows[start_idx]['Frame'])
            end_frame = int(rows[end_idx]['Frame'])
            critical_frame = [event_2_idx[0], start_idx, end_idx]
            print(f'Using sequence: Start Frame={hit_frame}, End Frame={end_frame} =====')

        # 不符合條件
        else:
            print('===== event error: invalid event sequence or count =====')
            return 0

        current_idx = start_idx
        # end_idx = critical_frame[-1] if len(critical_frame) >= 3 else len(rows) - 1
        if end_idx == -1:
            end_idx = len(rows) - 1

        points = []
        while current_idx <= end_idx:
            p = P(fid=rows[current_idx]['Frame'], timestamp=rows[current_idx]['Timestamp'], visibility=rows[current_idx]['Visibility'], x=rows[current_idx]['X'], y=rows[current_idx]['Y'], z=rows[current_idx]['Z'], event=rows[current_idx]['Event'])
            points.append(p)
            current_idx += 1
        points = denoise(points)
        # points = interpolatePoints(points)
        current_idx = start_idx
        for point in points:
            rows[current_idx]['Visibility'] = point.visibility
            rows[current_idx]['X'] = point.x
            rows[current_idx]['Y'] = point.y
            rows[current_idx]['Z'] = point.z
            current_idx += 1
        
        # original_trajectory = rows[start_idx:end_idx+1]
        new_start_idx = start_idx
        for i in range(start_idx+1, end_idx+1):
            if int(rows[i]['Visibility'] == 1):
                new_start_idx = i
                break
        gradient_start_idx = start_idx
        if float(rows[new_start_idx]['Y']) < float(rows[start_idx]['Y']):
             gradient_start_idx = new_start_idx
        original_trajectory = rows[gradient_start_idx:end_idx+1]
        vis_count = 0
        for i in range(len(original_trajectory)):
            if int(original_trajectory[i]['Visibility'] == 1):
                vis_count += 1
        if vis_count <= 2:
            print('===== Use 2D Mapping =====')
            return 0

        # Set initial velocity for gradient descent optimization [Vx, Vy, Vz]
        initial_V = np.array([0, -50, 0])
        
        # Set starting point
        starting_point = [float(original_trajectory[0]['X']), float(original_trajectory[0]['Y']), float(original_trajectory[0]['Z']), float(original_trajectory[0]['Timestamp'])]

        # Optimize the velocity
        optimized_V = optimize_velocity(original_trajectory, starting_point, fps, flight_time, initial_V)
        if optimized_V is None:
            print('===== gradient failure =====')
            return 0

        # Generate a smoothed trajectory using the optimized velocity
        smoothed_trajectory = physics_predict3d_v2(starting_point, optimized_V, fps, flight_time, alpha=0.242)
        
        new_rows = rows[0:start_idx]
        for row in new_rows:
            if row['Event'] == '1':
                row['Event'] = '0'
        new_rows.append(
            {
                'Frame': str(hit_frame),
                'Visibility': '1',
                'X': smoothed_trajectory[0, 0],
                'Y': smoothed_trajectory[0, 1],
                'Z': smoothed_trajectory[0, 2],
                'Event': '1',
                'Timestamp': str(smoothed_trajectory[0, 3])
            }
        )     
        # start_frame = int(rows[start_idx]['Frame'])
        tmp_frame = hit_frame + 1
        for i in range(1, len(smoothed_trajectory)):
            new_row = {
                'Frame': str(hit_frame + i),
                'Visibility': '1',
                'X': smoothed_trajectory[i, 0],
                'Y': smoothed_trajectory[i, 1],
                'Z': smoothed_trajectory[i, 2],
                'Event': '0',
                'Timestamp': str(smoothed_trajectory[i, 3])
            }
            new_rows.append(new_row)
            tmp_frame += 1
        new_rows[-1]['Event'] = 3

    with open(output_path, 'w', newline='') as outputfile:
        writer = csv.writer(outputfile)
        writer.writerow(['Frame', 'Visibility', 'X', 'Y', 'Z', 'Event', 'Timestamp'])
        for row in new_rows:
            writer.writerow([row['Frame'], row['Visibility'], row['X'], row['Y'], row['Z'], row['Event'], row['Timestamp']])
    return 1
