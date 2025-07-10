import argparse
from typing import Optional
import matplotlib.pyplot as plt
import math
import sys
import os
import numpy as np
import random

from scipy.integrate import solve_ivp
from scipy.optimize import minimize
from scipy.signal import find_peaks, peak_prominences
from scipy.ndimage import gaussian_filter

from lib.common import ROOTDIR
from lib.point import Point, load_vis_points_from_csv, sendPoints

REPLAYDIR = f"{ROOTDIR}/replay/"
PEAK_PROMINENCE = 0.1
INLIER_THRESHOLD = 0.3
INLIER_RATIO = 0.7

def physics_predict3d(starting_point, second_point, flight_time=10, touch_ground_cut=True, alpha=0.2151959552, g=9.81):
    # starting_point, second_point, shape: (4,) 4: XYZt
    fps = 1/(second_point[3] - starting_point[3])

    initial_velocity = (second_point[:3]-starting_point[:3]) * fps # shape: (3,) unit: m/s

    traj = solve_ivp(lambda t, y: bm_ball(t, y, alpha=alpha, g=g), [0, flight_time], np.concatenate((starting_point[:3], initial_velocity)), t_eval = np.arange(0, flight_time, 1/fps)) # traj.t traj.y

    xyz = np.swapaxes(traj.y[:3,:], 0, 1) # shape: (N points, 3)
    t = np.expand_dims(traj.t,axis=1) # shape: (N points, 1)
    trajectories = np.concatenate((xyz, t),axis=1) # shape: (N points, 4)

    # Cut the part under the ground
    if touch_ground_cut:
        for i in range(trajectories.shape[0]-1):
            if trajectories[i,2] >= 0 and trajectories[i+1,2] <= 0:
                trajectories = trajectories[:i+1,:]
                break
    # Add timestamp correctly
    trajectories[:,3] += (starting_point[3]) # shape: (N points, 4)

    return trajectories # shape: (N points, 4) , include input two Points

def physics_predict3d_v2(starting_point, v, fps, flight_time=10, touch_ground_cut=True, alpha=0.2151959552, g=9.81):

    initial_velocity = v

    traj = solve_ivp(lambda t, y: bm_ball(t, y, alpha=0.242, g=g), [0, flight_time], np.concatenate((starting_point[:3], initial_velocity)), t_eval = np.arange(0, flight_time, 1/fps)) # traj.t traj.y

    xyz = np.swapaxes(traj.y[:3,:], 0, 1) # shape: (N points, 3)
    t = np.expand_dims(traj.t,axis=1) # shape: (N points, 1)
    trajectories = np.concatenate((xyz, t),axis=1) # shape: (N points, 4)

    # Cut the part under the ground
    if touch_ground_cut:
        for i in range(trajectories.shape[0]-1):
            if trajectories[i,2] >= 0 and trajectories[i+1,2] <= 0:
                trajectories = trajectories[:i+1,:]
                break
    # Add timestamp correctly
    trajectories[:,3] += (starting_point[3]) # shape: (N points, 4)

    return trajectories # shape: (N points, 4) , include starting_point


def bm_ball(t,x,alpha=0.2151959552, g=9.81):
    # velocity
    v = math.sqrt(x[3]**2+x[4]**2+x[5]**2)
    # ordinary differential equations (3)
    xdot = [ x[3], x[4], x[5], -alpha*x[3]*v, -alpha*x[4]*v, -g-alpha*x[5]*v]
    return xdot

def loss_distance(V, original_trajectory, starting_point, fps, flight_time):
    total_loss = 0
    predicted_trajectory = physics_predict3d_v2(starting_point, V, fps, flight_time, touch_ground_cut=False, alpha=0.242)
    predicted_timestamps = predicted_trajectory[:, 3]
    predicted_xyz = predicted_trajectory[:, :3]
    for point in original_trajectory:
        if point.visibility == 1:

            # closest timestamp
            original_timestamp = point.timestamp
            closest_idx = np.argmin(np.abs(predicted_timestamps - original_timestamp))
            if 0 <= closest_idx < len(predicted_trajectory):
                predicted_xyz = predicted_trajectory[closest_idx, :3]
                original_xyz = np.array([point.x, point.y, point.z])
                distance = np.linalg.norm(original_xyz - predicted_xyz)
                total_loss += distance
            
            # # closest distance
            # original_xyz = np.array([point.x, point.y, point.z])
            # distances = np.linalg.norm(predicted_xyz - original_xyz, axis=1)
            # min_distance = np.min(distances)
            # total_loss += min_distance

    return total_loss

def loss_inliers_distance(V, original_trajectory, starting_point, fps, flight_time, inlier_threshold=INLIER_THRESHOLD):
    total_loss = 0
    inliers = 0
    predicted_trajectory = physics_predict3d_v2(starting_point, V, fps, flight_time, touch_ground_cut=False, alpha=0.242)
    predicted_timestamps = predicted_trajectory[:, 3]
    predicted_xyz = predicted_trajectory[:, :3]
    
    for point in original_trajectory:
        if point.visibility == 1:

            # closest timestamp
            original_timestamp = point.timestamp
            closest_idx = np.argmin(np.abs(predicted_timestamps - original_timestamp))
            if 0 <= closest_idx < len(predicted_trajectory):
                predicted_xyz = predicted_trajectory[closest_idx, :3]
                original_xyz = np.array([point.x, point.y, point.z])
                distance = np.linalg.norm(original_xyz - predicted_xyz)
                total_loss += distance
                if distance < inlier_threshold:  # Check if it's an inlier
                    inliers += 1
            
            # # closest distance
            # original_xyz = np.array([point.x, point.y, point.z])
            # distances = np.linalg.norm(predicted_xyz - original_xyz, axis=1)
            # min_distance = np.min(distances)
            # total_loss += min_distance
            # if min_distance < inlier_threshold:  # Check if it's an inlier
            #     inliers += 1

    return total_loss, inliers

def combined_loss(V, original_trajectory, starting_point, fps, flight_time, weight=1.0):
    loss, inliers = loss_inliers_distance(V, original_trajectory, starting_point, fps, flight_time)
    weighted_loss = loss - weight * inliers  # Combine loss and inliers with a weight
    return weighted_loss



# Nelder-Mead
def optimize_velocity(original_trajectory, starting_point, fps, flight_time, initial_V, method='Nelder-Mead', max_iter=None):
    """Optimize the velocity V using scipy's minimize function.""" 
    options = {'maxiter': max_iter} if max_iter is not None else {}
    
    result = minimize(
        loss_distance,  # The objective function to minimize
        initial_V,      # Initial guess for the velocity
        args=(original_trajectory, starting_point, fps, flight_time),  # Additional arguments for the loss function
        method=method,   # Optimization method
        options=options
    )
    
    if result.success:
        optimized_V = result.x
        loss = result.fun # The total loss at the optimal solution
        # DEBUG: PRINT
        # print(f"Optimization successful: {optimized_V}, Loss: {loss}")
        # print("Optimization successful:", optimized_V)
        # print("Loss:", loss)
        return optimized_V, loss
    else:
        print("Optimization failed:", result.message)
        return None, None

def optimize_velocity_2(original_trajectory, starting_point, fps, flight_time, initial_V, weight=1.0, method='Nelder-Mead'):
    """Optimize the velocity V to maximize inliers and minimize loss."""
    result = minimize(
        combined_loss,  # Use the new combined loss function
        initial_V,      # Initial guess for the velocity
        args=(original_trajectory, starting_point, fps, flight_time, weight),  # Pass the weight parameter
        method=method   # Optimization method
    )
    
    if result.success:
        optimized_V = result.x
        loss, inliers = loss_inliers_distance(optimized_V, original_trajectory, starting_point, fps, flight_time)
        print(f"Optimization successful: {optimized_V}, Inliers: {inliers}, Loss: {loss}")
        # print("Optimization successful:", optimized_V)
        # print("Inliers:", inliers)
        # print("Loss:", loss)
        return optimized_V, inliers, loss
    else:
        print("Optimization failed:", result.message)
        return None, None, None


    
def plot_trajectories(original_trajectory, predicted_trajectory):
    original_x = [p.x for p in original_trajectory if p.visibility == 1]
    original_y = [p.y for p in original_trajectory if p.visibility == 1]
    original_z = [p.z for p in original_trajectory if p.visibility == 1]

    predicted_x = predicted_trajectory[:, 0]
    predicted_y = predicted_trajectory[:, 1]
    predicted_z = predicted_trajectory[:, 2]

    fig = plt.figure(figsize=(10, 6))
    
    # 3D
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.scatter(original_x, original_y, original_z, c='r', label='Original Trajectory', marker='o')
    if original_x and original_y and original_z:
        ax1.scatter(original_x[0], original_y[0], original_z[0], c='g', label='Start Point', marker='^', s=100)
        ax1.text(original_x[0], original_y[0], original_z[0], 'Start', color='g')

        ax1.scatter(original_x[-1], original_y[-1], original_z[-1], c='purple', label='End Point', marker='v', s=100)
        ax1.text(original_x[-1], original_y[-1], original_z[-1], 'End', color='purple')
    ax1.plot(predicted_x, predicted_y, predicted_z, c='b', label='Predicted Trajectory')

    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Z (m)')
    ax1.set_xlim(-4, 4)
    ax1.set_ylim(-7, 7)
    ax1.set_zlim(0,7)
    ax1.set_title('Trajectory Comparison')
    ax1.legend()

    # (y,z) 2D
    ax2 = fig.add_subplot(122)
    ax2.scatter(original_y, original_z, c='r', label='Original Trajectory', marker='o')
    ax2.plot(predicted_y, predicted_z, c='b', label='Predicted Trajectory')
    if original_y and original_z:
        ax2.scatter(original_y[0], original_z[0], c='g', label='Start Point', marker='^', s=100)
        ax2.text(original_y[0], original_z[0], 'Start', color='g')

        ax2.scatter(original_y[-1], original_z[-1], c='purple', label='End Point', marker='v', s=100)
        ax2.text(original_y[-1], original_z[-1], 'End', color='purple')

    ax2.set_xlabel('Y (m)')
    ax2.set_ylabel('Z (m)')
    ax2.set_xlim(-7, 7)
    ax2.set_ylim(0, 7)
    ax2.set_title('YZ Plane Projection')
    ax2.legend()


    plt.tight_layout()
    plt.show()

def RMOutlier(points):
    print('=== remove ===')
    fid = [point.fid for point in points]
    point_x = [point.x for point in points]
    point_y = [point.y for point in points]
    point_z = [point.z for point in points]
    n = len(points)

    mid_point_x = sum(point_x) / n
    mid_point_y = sum(point_y) / n
    mid_point_z = sum(point_z) / n

    # find the radius being threshold
    distances = [np.sqrt((point_x[i] - mid_point_x)**2 + (point_y[i] - mid_point_y)**2 + (point_z[i] - mid_point_z)**2) for i in range(n)]
    sorted_distances = sorted(distances)
    # print(sorted_distances)
    radius_index = int(0.6 * n)
    radius = sorted_distances[radius_index - 1]
    # print(f"radius: {radius}")

    new_traj = []
    rm_points = []
    
    # remove outlier by checking if the distance between the point and the center is less than the radius * 2
    for i in range(n):
        # print(points[i].fid, ':', distances[i])
        if distances[i] <= radius * 5:
            new_traj.append(points[i])
        else:
            print(f'-------- remove Frame {points[i].fid} --------')
            rm_points.append(points[i])
    return new_traj, rm_points


def RMOutlier_motion(points):
    print('=== remove_motion ===')
    x = [p.x for p in points]
    y = [p.y for p in points]
    z = [p.z for p in points]
    vis = [p.visibility for p in points]
    fid = [p.fid for p in points]
    timestamp = [p.timestamp for p in points]

    distances = []
    avg_speeds = []
    frame_diffs = []

    last_visible_index = -1  # Tracks the most recent visible point index
    prev_avg_speed = 0       # Stores the previous average speed
    prev_frame_diff = 0      # Stores the previous frame difference
    prev_delta_y_direction = None  # Tracks the previous delta y direction

    speed_threshold = 30
    frame_threshold = 10

    for i in range(len(x)):
        if vis[i] == 1:  # current visible point
            if last_visible_index != -1:
                # Calculate the distance between the current point and the previous visible point
                distance = np.sqrt((x[i] - x[last_visible_index])**2 +
                                   (y[i] - y[last_visible_index])**2 +
                                   (z[i] - z[last_visible_index])**2)
                
                # Calculate the time difference between the two points
                time_diff = timestamp[i] - timestamp[last_visible_index]
                
                # Calculate the average speed (distance divided by time difference)
                avg_speed = distance / time_diff if time_diff > 0 else 0
                
                # Calculate the frame difference between the two points
                frame_diff = fid[i] - fid[last_visible_index]

                # Calculate delta y direction change
                delta_y_direction = (y[i] - y[last_visible_index]) > 0

                if (avg_speed > speed_threshold and frame_diff < frame_threshold):
                    if (prev_avg_speed > speed_threshold and prev_frame_diff < frame_threshold):
                        if (prev_delta_y_direction is not None and delta_y_direction != prev_delta_y_direction):
                            vis[last_visible_index] = 0  # Mark as outlier
                            points[last_visible_index].visibility = 0  # Modify original Point object
                            print(f'-------- remove motion Frame {fid[last_visible_index]} --------')
                
                distances.append(distance)
                avg_speeds.append(avg_speed)
                frame_diffs.append(frame_diff)

                prev_avg_speed = avg_speed
                prev_frame_diff = frame_diff
                prev_delta_y_direction = delta_y_direction
            else:
                distances.append(0)
                avg_speeds.append(0)
                frame_diffs.append(0)

            last_visible_index = i
        else:
            distances.append(0)
            avg_speeds.append(0)
            frame_diffs.append(0)
    
    new_traj = []
    rm_points = []
    for point in points:
        if point.visibility == 1:
            new_traj.append(point)
        else:
            rm_points.append(point)

    return new_traj, rm_points

def smooth_trajectory(traj, sigma=1.0):
    if len(traj) < 3:
        return traj
    
    xs = np.array([p.x for p in traj])
    ys = np.array([p.y for p in traj])
    zs = np.array([p.z for p in traj])

    xs_smooth = gaussian_filter(xs, sigma=sigma)
    ys_smooth = gaussian_filter(ys, sigma=sigma)
    zs_smooth = gaussian_filter(zs, sigma=sigma)

    for i in range(len(traj)):
        traj[i].x = xs_smooth[i]
        traj[i].y = ys_smooth[i]
        traj[i].z = zs_smooth[i]
    
    return traj


def findPeaks(points):
    y = [point.y for point in points]
    y = gaussian_filter(y, sigma=1.2)
    z = [point.z for point in points]
    z = gaussian_filter(z, sigma=1.2)
    # for p in zip(y,z):
    #     plt.scatter(p[0], p[1], c='none', edgecolors='black', s=8)
    fid = [point.fid for point in points]
    y_neg_peaks, y_neg_props = find_peaks(np.array(y) * -1, prominence=PEAK_PROMINENCE)
    y_neg_prominence = y_neg_props['prominences']
    y_pos_peaks, y_pos_props = find_peaks(np.array(y), prominence=PEAK_PROMINENCE)
    y_pos_prominence = y_pos_props['prominences']
    z_down_peaks, z_down_props = find_peaks(np.array(z) * -1, prominence=PEAK_PROMINENCE)
    z_down_prominence = z_down_props['prominences']
    all_peaks = [(p, y_neg_prominence[i], 'y_negative_peak') for i, p in enumerate(y_neg_peaks)] + \
                [(p, y_pos_prominence[i], 'y_positive_peak') for i, p in enumerate(y_pos_peaks)] + \
                [(p, z_down_prominence[i], 'z_down_peak') for i, p in enumerate(z_down_peaks)]
    all_peaks.sort(key=lambda x: fid[x[0]])
    if len(all_peaks) >= 1:
        max_peak = max(all_peaks, key=lambda x: x[1])
        print(f"Peak ID: {max_peak[0]}, Frame ID (FID): {fid[max_peak[0]]}")
        print(f"Prominence: {max_peak[1]}, Type: {max_peak[2]}")
        # max_fid = fid[max_peak[0]]
        return max_peak[0]
    return -1 


# closest timestamp
def count_inliers_time(trajectory, predicted_trajectory, inlier_threshold):
    inliers = 0
    predicted_timestamps = predicted_trajectory[:, 3]
    predicted_xyz = predicted_trajectory[:, :3]
    for point in trajectory:
        if point.visibility == 1:
            original_timestamp = point.timestamp
            closest_idx = np.argmin(np.abs(predicted_timestamps - original_timestamp))
            if 0 <= closest_idx < len(predicted_trajectory):
                predicted_xyz = predicted_trajectory[closest_idx, :3]
                original_xyz = np.array([point.x, point.y, point.z])
                distance = np.linalg.norm(original_xyz - predicted_xyz)
                # print(f"{point.fid}: {distance}, original_xyz: {original_xyz}, predicted_xyz: {predicted_xyz}")
                if distance < inlier_threshold:
                    inliers += 1
    return inliers

def count_inliers_distance(trajectory, predicted_trajectory, inlier_threshold):
    inliers = 0
    predicted_xyz = predicted_trajectory[:, :3]
    for point in trajectory:
        if point.visibility == 1:
            original_xyz = np.array([point.x, point.y, point.z])
            distances = np.linalg.norm(predicted_xyz - original_xyz, axis=1)
            min_distance = np.min(distances)
            min_index = np.argmin(distances)
            closest_predicted_xyz = predicted_xyz[min_index]
            # print(f"{point.fid}: {min_distance}, original_xyz: {original_xyz}, predicted_xyz: {closest_predicted_xyz}")
            if min_distance < inlier_threshold:  # Check if it's an inlier
                inliers += 1
    return inliers

def is_flight_trajectory(traj, fps, initial_V, loss_function, evaluation_method, inlier_threshold, inlier_ratio):
    starting_point = [traj[0].x, traj[0].y, traj[0].z, traj[0].timestamp]
    flight_time = traj[-1].timestamp - traj[0].timestamp

    # DEBUG: PRINT
    # print(f"{traj[0].fid} - {traj[-1].fid}")
    if loss_function == 1:
        optimized_V, loss = optimize_velocity(traj, starting_point, fps, flight_time, initial_V)
    elif loss_function == 2:
        optimized_V, inliers, loss = optimize_velocity_2(traj, starting_point, fps, flight_time, initial_V)

    x = [p.x for p in traj if p.visibility == 1]
    y = [p.y for p in traj if p.visibility == 1]
    z = [p.z for p in traj if p.visibility == 1]
    
    if optimized_V is not None:
        predicted_trajectory = physics_predict3d_v2(starting_point, optimized_V, fps, flight_time)

        if evaluation_method == 'vote':
            inliers = count_inliers_time(traj, predicted_trajectory, inlier_threshold)
            # inliers = count_inliers_distance(traj, predicted_trajectory, inlier_threshold)
            # DEBUG: PRINT
            # print('inliers:', inliers)
            # print('threshold:', len(traj) * inlier_ratio)

            if inliers >= len(traj) * inlier_ratio:
                return True, optimized_V, predicted_trajectory
            else:
                return False, optimized_V, predicted_trajectory
        elif evaluation_method == 'loss':
            if loss < 3:
                return True, optimized_V, predicted_trajectory
            else:
                return False, optimized_V, predicted_trajectory
    return False, None, None


def trajectory_segment(trajectory, fps=120, initial_V=np.array([0, 0, 0]), loss_function=1, evaluation_method='vote', inlier_threshold=INLIER_THRESHOLD, inlier_ratio=INLIER_RATIO):
    results = []

    # **Step 1: Split trajectory by timestamp gap**
    time_trajectory = []
    temp_traj = [trajectory[0]]
    for i in range(1, len(trajectory)):
        if trajectory[i].timestamp - trajectory[i-1].timestamp > 0.5:
            # print('Time Split')
            # print(trajectory[i-1].fid, trajectory[i].fid)
            if len(temp_traj) >= 5:
                time_trajectory.append(temp_traj)
            temp_traj = []
        temp_traj.append(trajectory[i])
    if len(temp_traj) >= 5:  # Last segment
        time_trajectory.append(temp_traj)

    # **Step 2: Remove outliers in each sub-segment**
    tmp_trajectory = []
    for sub_traj in time_trajectory:
        cleaned_traj, rm_points = RMOutlier(sub_traj)
        cleaned_traj, rm_points = RMOutlier_motion(cleaned_traj)
        if len(cleaned_traj) >= 5:
            tmp_trajectory.append(cleaned_traj)
    time_trajectory = tmp_trajectory

    # **Step 3: Smooth each sub-segment**
    # for sub_traj in time_trajectory:
    #     sub_traj = smooth_trajectory(sub_traj, sigma=1.2)    
    
    # **Step 4: Check for peak in each sub-segment**
    sub_trajectory = []
    for sub_traj in time_trajectory:
        peak_idx = findPeaks(sub_traj)
        if peak_idx != -1:
            if len(sub_traj[:peak_idx]) >= 5:
                sub_trajectory.append(sub_traj[:peak_idx])
            if len(sub_traj[peak_idx:]) >= 5:
                sub_trajectory.append(sub_traj[peak_idx:])
            # sub_trajectory = [sub_traj[:peak_idx], sub_traj[peak_idx:]]
        else:
            sub_trajectory.append(sub_traj)

                
    for traj in sub_trajectory:
        failed_segment = []
        isFly = False
        # Determine if the sub-segment is a flight trajectory
        while len(traj) > 1:
            isFly, optimized_V, pred_trajectory = is_flight_trajectory(traj, fps=fps, initial_V=initial_V, loss_function=loss_function, evaluation_method=evaluation_method, inlier_threshold=inlier_threshold, inlier_ratio=inlier_ratio)
            if isFly:
                if failed_segment:
                    results.append((failed_segment, False, None))
                    failed_segment = []
                results.append((traj, isFly, optimized_V))
                break
            else:
                print('... Skipping first 5 points ...')
                if len(traj) > 5:
                    failed_segment.extend(traj[:5])
                    traj = traj[5:]
                else:
                    failed_segment.extend(traj)
                    break
        if not isFly and failed_segment:
            results.append((failed_segment, False, None))
            failed_segment = []
        
    return results


def main(date):
    all_points = load_vis_points_from_csv(os.path.join(REPLAYDIR, date, 'Model3D_points.csv'))
    # all_points = all_points[1670:]
    # all_points = all_points[2100:]
    # all_points = all_points[639:659]
    
    segment_speed = []
    n = 20
    fps = 120
    initial_V = np.array([0, 0, 0])
    inlier_ratio = 0.8

    merged_segments = []
    prev_start, prev_end, prev_v, prev_points = None, None, None, []
    last_valid_time = None

    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    axs[0].set_xlabel('Y (m)')
    axs[0].set_ylabel('Z (m)')
    axs[0].set_xlim(-7, 7)
    axs[0].set_ylim(0, 7)

    axs[1].set_xlabel('Y (m)')
    axs[1].set_ylabel('Z (m)')
    axs[1].set_xlim(-7, 7)
    axs[1].set_ylim(0, 7)

    for i in range(int(len(all_points)/n)):
        print()
        print('========================================')
        c=[random.random(), random.random(), random.random()]

        trajectory = all_points[i*n:i*n+n]
        print(f"Sequence: {trajectory[0].fid} - {trajectory[-1].fid}")
        print('--------------------')

        y = [p.y for p in trajectory if p.visibility == 1]
        z = [p.z for p in trajectory if p.visibility == 1]
        axs[0].scatter(y, z, c=c, alpha=0.5, label='Original Trajectory', s=8)

        results = trajectory_segment(trajectory, fps=fps, initial_V=np.array([0, 0, 0]), loss_function=1, evaluation_method='vote', inlier_threshold=INLIER_THRESHOLD, inlier_ratio=INLIER_RATIO)
        for res in results:
            traj = res[0]
            isFly = res[1]
            v = res[2]
            if v is not None:
                speed = np.linalg.norm(v)
                print(f"--> {traj[0].fid} - {traj[-1].fid} : {isFly}, Speed: {v}, {speed}")
            elif v is None:
                speed = -1
                print(f"--> {traj[0].fid} - {traj[-1].fid} : {isFly}")

            traj_start_time = traj[0].timestamp
            if last_valid_time is not None and traj_start_time - last_valid_time > 2.0:
                print(">> 超過 2 秒無軌跡，重置")
                if prev_v is not None:
                    merged_segments.append((prev_start, prev_end, prev_v, prev_points))
                prev_start, prev_end, prev_v, prev_points = None, None, None, []  

            if isFly and speed >= 3:
                # **檢查是否合併**
                if (prev_v is not None and 
                    ((prev_v[1] >= 0 and v[1] >= 0) or (prev_v[1] < 0 and v[1] < 0)) and 
                    traj[0].timestamp - prev_points[-1].timestamp <= 1.0):
                    
                    print(">> 合併上一段")
                    prev_end = traj[-1].fid  
                    prev_points.extend(traj)  

                else:
                    # **存上一段，開啟新段落**
                    if prev_v is not None:
                        merged_segments.append((prev_start, prev_end, prev_v, prev_points))
                    prev_start, prev_end, prev_v, prev_points = traj[0].fid, traj[-1].fid, v, list(traj)

                last_valid_time = traj[-1].timestamp  # 更新最後有效軌跡timestamp 
            
            
            '''
            if isFly and speed >= 3:
                if prev_v is not None and ((prev_v[1] >= 0 and v[1] >= 0) or (prev_v[1] < 0 and v[1] < 0)):
                    # **與上一段方向相同，合併**
                    prev_end = traj[-1].fid  # 更新結束時間
                    prev_points.extend(traj)  # 合併軌跡點
                else:
                    # **存入上一個段落，開啟新段落**
                    if prev_v is not None:
                        merged_segments.append((prev_start, prev_end, prev_v, prev_points))
                    prev_start, prev_end, prev_v, prev_points = traj[0].fid, traj[-1].fid, v, list(traj)
            '''

            '''
            if isFly and speed >= 3:    
                traj_start_time = traj[0].timestamp

                # **判斷是否需要重置**
                if last_valid_time is not None and traj_start_time - last_valid_time > 2.0:
                    print(">> 超過 2 秒無有效軌跡，重置段落")
                    if prev_v is not None:
                        merged_segments.append((prev_start, prev_end, prev_v, prev_points))
                    prev_start, prev_end, prev_v, prev_points = None, None, None, []

                # **如果上一段的 y 方向相同，且時間間隔不超過 1 秒，則合併**
                if (prev_v is not None and 
                    ((prev_v[1] >= 0 and v[1] >= 0) or (prev_v[1] < 0 and v[1] < 0)) and 
                    traj[0].timestamp - prev_points[-1].timestamp <= 1.0):
                    
                    print(">> 合併與上一段")
                    prev_end = traj[-1].fid  
                    prev_points.extend(traj)  

                else:
                    # **存入上一個段落，開啟新段落**
                    if prev_v is not None:
                        merged_segments.append((prev_start, prev_end, prev_v, prev_points))
                    prev_start, prev_end, prev_v, prev_points = traj[0].fid, traj[-1].fid, v, list(traj)

                last_valid_time = traj[-1].timestamp  # 更新最後有效時間
            '''
            
            # DEBUG: DRAW, PLOT
            if v is not None:
                pred_trajectory = physics_predict3d_v2([traj[0].x, traj[0].y, traj[0].z, traj[0].timestamp], v, fps, (traj[-1].timestamp - traj[0].timestamp))
                pred_y = [p[1] for p in pred_trajectory]
                pred_z = [p[2] for p in pred_trajectory]
            
            if isFly == True:
                segment_speed.append((traj[0].fid, traj[-1].fid, v))
                axs[0].plot(pred_y, pred_z, c=c, alpha=0.7, label='Predicted Trajectory', linewidth=1)
                # for i in range(1, len(traj)):
                #     p = traj[i]
                #     axs[0].text(p.y, p.z, str(p.fid), fontsize=8, color='red', ha='left', va='bottom')
                # start point
                axs[0].scatter(traj[0].y, traj[0].z, c=c, s=10)
                if speed >= 3:
                    axs[0].text(traj[0].y, traj[0].z, str(traj[0].fid), fontsize=10, color='red', ha='left', va='bottom')
                else:
                    axs[0].text(traj[0].y, traj[0].z, str(traj[0].fid), fontsize=10, color='green', ha='left', va='bottom')
            elif isFly == False and v is not None:
                axs[0].plot(pred_y, pred_z, c='grey', alpha=0.7, label='Predicted Trajectory', linewidth=1)
                # start point
                axs[0].scatter(traj[0].y, traj[0].z, c=c, s=10)
                axs[0].text(traj[0].y, traj[0].z, str(traj[0].fid), fontsize=10, color='blue', ha='left', va='bottom')
            elif isFly == False and v is None:
                # axs[0].scatter(y, z, c=c, label='Original Trajectory', s=8)
                # start point
                axs[0].scatter(traj[0].y, traj[0].z, c=c, s=10)
                axs[0].text(traj[0].y, traj[0].z, str(traj[0].fid), fontsize=10, color='blue', ha='left', va='bottom')    

    # 最後的一段軌跡
    if prev_v is not None and prev_points:
        merged_segments.append((prev_start, prev_end, prev_v, prev_points))
    
    for s, e, v in segment_speed:
        speed = (v[0]**2 + v[1]**2 + v[2]**2) ** 0.5
        print(f"{s} - {e}: {speed}")

    for s, e, v, traj in merged_segments:
        print(f"{traj[0].fid} - {traj[-1].fid}")

    labels = []
    for idx, seg in enumerate(merged_segments):
        start_fid, end_fid, V, points = seg
        color = (random.random(), random.random(), random.random())

        y = [p.y for p in points]
        z = [p.z for p in points]

        labels.append(f"Traj {idx+1}: {start_fid} - {end_fid}")

        axs[1].plot(y, z, c=color, alpha=0.8, linewidth=2)   
        axs[1].scatter(y[0], z[0], c=color, s=40, edgecolors='black')
        axs[1].text(y[0], z[0], str(start_fid), fontsize=10, c=color, ha='left', va='bottom')   
        axs[1].scatter(y[-1], z[-1], c=color, s=40, edgecolors='black')
        axs[1].text(y[-1], z[-1], str(end_fid), fontsize=10, c=color, ha='left', va='bottom')   
    axs[1].text(1.05, 0.5, "\n".join(labels), transform=axs[1].transAxes, fontsize=10, verticalalignment='center')
        
    # plt.xlabel('Y (m)')
    # plt.ylabel('Z (m)')
    # plt.xlim(-7, 7)
    # plt.ylim(0, 7)
    # plt.show()
    plt.tight_layout()
    plt.show()
      

def parse_args() -> Optional[str]:
    # Args
    parser = argparse.ArgumentParser(description = 'Model3D')
    parser.add_argument('--date', type=str, help='default: replay/XX/')
    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = parse_args()
    # test()   
    main(args.date) 



# starting_point = [traj[0].x, traj[0].y, traj[0].z, traj[0].timestamp]
# flight_time = traj[-1].timestamp - traj[0].timestamp

# if loss_function == 1:
#     optimized_V, loss = optimize_velocity(traj, starting_point, fps, flight_time, initial_V)
# elif loss_function == 2:
#     optimized_V, inliers, loss = optimize_velocity_2(traj, starting_point, fps, flight_time, initial_V)

# x = [p.x for p in traj if p.visibility == 1]
# y = [p.y for p in traj if p.visibility == 1]
# z = [p.z for p in traj if p.visibility == 1]

# fly = False
# if optimized_V is not None:
#     predicted_trajectory = physics_predict3d_v2(starting_point, optimized_V, fps, flight_time)
#     pred_y = [p[1] for p in predicted_trajectory]
#     pred_z = [p[2] for p in predicted_trajectory]

#     if evaluation_method == 'vote':
#         inliers = count_inliers_time(traj, predicted_trajectory, INLIER_THRESHOLD)
#         inliers = count_inliers_distance(traj, predicted_trajectory,INLIER_THRESHOLD)
#         print('inliers:', inliers)
#         print('threshold:', len(traj) * inlier_ratio)

#         if inliers >= len(traj) * inlier_ratio:
#             print("fly")
#             fly = True
#         else:
#             print("non fly")
#     elif evaluation_method == 'loss':
#         if loss < 3:
#             print("fly")
#             fly = True
#         else:
#             print("non fly")
#     print('--------------------')

#     # Draw
#     plt.scatter(y, z, c=c, alpha=0.5, label='Original Trajectory', s=8)
#     if fly == True:
#         plt.plot(pred_y, pred_z, c=c, alpha=0.7, label='Predicted Trajectory', linewidth=1)
#         # for i in range(1, len(traj)):
#         #     p = traj[i]
#         #     plt.text(p.y, p.z, str(p.fid), fontsize=8, color='red', ha='left', va='bottom')

#         # start point
#         plt.scatter(traj[0].y, traj[0].z, c=c, s=10)
#         plt.text(traj[0].y, traj[0].z, str(traj[0].fid), fontsize=10, color='red', ha='left', va='bottom')
#     elif fly == False:
#         plt.plot(pred_y, pred_z, c='grey', alpha=0.7, label='Predicted Trajectory', linewidth=1)
#         # for i in range(1, len(traj)):
#         #     p = traj[i]
#         #     plt.text(p.y, p.z, str(p.fid), fontsize=8, color='blue', ha='left', va='bottom')
        
#         # start point
#         plt.scatter(traj[0].y, traj[0].z, c=c, s=10)
#         plt.text(traj[0].y, traj[0].z, str(traj[0].fid), fontsize=10, color='blue', ha='left', va='bottom')

# else:
#     print("non fly")
#     plt.scatter(y, z, c=c, label='Original Trajectory', s=8)
#     # for i in range(1, len(traj)):
#     #     p = traj[i]
#     #     plt.text(p.y, p.z, str(p.fid), fontsize=8, color='blue', ha='left', va='bottom')

#     # start point
#     plt.scatter(traj[0].y, traj[0].z, c=c, s=10)
#     plt.text(traj[0].y, traj[0].z, str(traj[0].fid), fontsize=10, color='blue', ha='left', va='bottom')     


# 2024-09-19_09-33-56
# 0single_rally
# 2025-01-16_15-18-59
def test(loss_method=1, method='vote'):
    all_points = load_vis_points_from_csv(os.path.join(REPLAYDIR, '0single_rally', 'rModel3D_points.csv'))
    
    
    fps = 120
    # flight_time = 10
    initial_V = np.array([0, 0, 0])
    n = 20
    inlier_ratio = 0.8
    for i in range(int(len(all_points)/n)):
        print('----------------------------------')
        traj = all_points[i*n:i*n+n]
        print(f"{traj[0].fid} - {traj[-1].fid}")

        traj, rm_points = RMOutlier(traj)

        c=[random.random(), random.random(), random.random()]
        for rm_point in rm_points:
            plt.scatter(rm_point.y, rm_point.z, c='none', edgecolors=c, alpha=0.7, marker='^', s=8)

        starting_point = [traj[0].x, traj[0].y, traj[0].z, traj[0].timestamp]
        flight_time = traj[-1].timestamp - traj[0].timestamp
        if loss_method == 1:
            optimized_V, loss = optimize_velocity(traj, starting_point, fps, flight_time, initial_V)
        elif loss_method == 2:
            optimized_V, inliers, loss = optimize_velocity_2(traj, starting_point, fps, flight_time, initial_V)
        x = [p.x for p in traj if p.visibility == 1]
        y = [p.y for p in traj if p.visibility == 1]
        z = [p.z for p in traj if p.visibility == 1]
        
        fly = False
        flight_time = traj[-1].timestamp - traj[0].timestamp
        if optimized_V is not None:
            predicted_trajectory = physics_predict3d_v2(starting_point, optimized_V, fps, flight_time)
            pred_y = [p[1] for p in predicted_trajectory]
            pred_z = [p[2] for p in predicted_trajectory]

            if method == 'vote':
                inliers = 0
                predicted_timestamps = predicted_trajectory[:, 3]
                predicted_xyz = predicted_trajectory[:, :3]
                for point in traj:
                    if point.visibility == 1:

                        # closest timestamp
                        original_timestamp = point.timestamp
                        closest_idx = np.argmin(np.abs(predicted_timestamps - original_timestamp))
                        if 0 <= closest_idx < len(predicted_trajectory):
                            predicted_xyz = predicted_trajectory[closest_idx, :3]
                            original_xyz = np.array([point.x, point.y, point.z])
                            distance = np.linalg.norm(original_xyz - predicted_xyz)
                            print(f"{point.fid}: {distance}, original_xyz: {original_xyz}, predicted_xyz: {predicted_xyz}")
                            if distance < INLIER_THRESHOLD:
                                inliers += 1
                        
                        # # closest distance
                        # original_xyz = np.array([point.x, point.y, point.z])
                        # distances = np.linalg.norm(predicted_xyz - original_xyz, axis=1)
                        # min_distance = np.min(distances)
                        # min_index = np.argmin(distances)
                        # closest_predicted_xyz = predicted_xyz[min_index]
                        # print(f"{point.fid}: {min_distance}, original_xyz: {original_xyz}, predicted_xyz: {closest_predicted_xyz}")
                        # if min_distance < INLIER_THRESHOLD:  # Check if it's an inlier
                        #     inliers += 1

                print('inliers:', inliers)
                print('threshold:', len(traj) * inlier_ratio)
                if inliers >= len(traj) * inlier_ratio:
                    print("fly")
                    fly = True
                else:
                    print("non fly")
            elif method == 'loss':
                if loss < 3:
                    print("fly")
                    fly = True
                else:
                    print("non fly")

            # Draw
            if fly == True:
                plt.scatter(y, z, c=c, alpha=0.5, label='Original Trajectory', s=8)
                plt.plot(pred_y, pred_z, c=c, alpha=0.7, label='Predicted Trajectory', linewidth=1)
                # for i in range(1, len(traj)):
                #     p = traj[i]
                #     plt.text(p.y, p.z, str(p.fid), fontsize=8, color='red', ha='left', va='bottom')

                # start point
                plt.scatter(traj[0].y, traj[0].z, c=c, s=10)
                plt.text(traj[0].y, traj[0].z, str(traj[0].fid), fontsize=10, color='red', ha='left', va='bottom')
            elif fly == False:
                plt.scatter(y, z, c=c, label='Original Trajectory', s=8)
                plt.plot(pred_y, pred_z, c='grey', alpha=0.7, label='Predicted Trajectory', linewidth=1)
                # for i in range(1, len(traj)):
                #     p = traj[i]
                #     plt.text(p.y, p.z, str(p.fid), fontsize=8, color='blue', ha='left', va='bottom')
                
                # start point
                plt.scatter(traj[0].y, traj[0].z, c=c, s=10)
                plt.text(traj[0].y, traj[0].z, str(traj[0].fid), fontsize=10, color='blue', ha='left', va='bottom')

        else:
            print("non fly")
            plt.scatter(y, z, c=c, label='Original Trajectory', s=8)
            # for i in range(1, len(traj)):
            #     p = traj[i]
            #     plt.text(p.y, p.z, str(p.fid), fontsize=8, color='blue', ha='left', va='bottom')

            # start point
            plt.scatter(traj[0].y, traj[0].z, c=c, s=10)
            plt.text(traj[0].y, traj[0].z, str(traj[0].fid), fontsize=10, color='blue', ha='left', va='bottom')     
    
    plt.xlabel('Y (m)')
    plt.ylabel('Z (m)')
    plt.xlim(-7, 7)
    plt.ylim(0, 7)
    plt.show()


    # original_trajectory = all_points[28:252]
    # # original_trajectory.reverse()
    # starting_idx = 251
    # second_idx = 252
    # starting_point = np.array([all_points[starting_idx].x, all_points[starting_idx].y, all_points[starting_idx].z, all_points[starting_idx].timestamp])
    # print('start:', starting_point)
    # second_point = np.array([all_points[second_idx].x, all_points[second_idx].y, all_points[second_idx].z, all_points[second_idx].timestamp])
    # print('second:', second_point)
    # traj = physics_predict3d_reverse(starting_point, second_point)
    # print(traj)
    # plot_trajectories(original_trajectory, traj)


    # fps = 120
    # final_point = [original_trajectory[0].x, original_trajectory[0].y, original_trajectory[0].z, original_trajectory[0].timestamp]
    # initial_V = np.array([0, 0, 0])
    # traj = physics_predict3d_v2_reverse(final_point, initial_V, fps)
    # print(traj)
    # plot_trajectories(original_trajectory, traj)

    # Second point
    # initial_second_point = second_point
    # optimized_second_point, _ = optimize_second_point_reverse(starting_point, original_trajectory, initial_second_point)
    # if optimized_second_point is not None:
    #     predicted_trajectory = physics_predict3d_reverse(starting_point, optimized_second_point, flight_time=10, touch_ground_cut=False, alpha=0.242)
    #     print(predicted_trajectory)
    #     plot_trajectories(original_trajectory, predicted_trajectory)

    
    # Speed
    # original_trajectory = all_points[28:252]
    # original_trajectory.reverse()
    # final_point = [original_trajectory[0].x, original_trajectory[0].y, original_trajectory[0].z,
    #                       original_trajectory[0].timestamp]
    # print('final_point:', final_point)
    # fps = 120
    # flight_time = 10
    # initial_V = np.array([0, 0, 0])

    # optimized_V, _ = optimize_velocity_reverse(original_trajectory, final_point, fps, flight_time, initial_V)
    # if optimized_V is not None:
    #     predicted_trajectory = physics_predict3d_v2_reverse(final_point, optimized_V, fps, flight_time, touch_ground_cut=False, alpha=0.242)
    #     print(predicted_trajectory)
    #     plot_trajectories(original_trajectory, predicted_trajectory)



    '''
    # original_trajectory = all_points[0:20] # 無轉折，Loss: 0.5087358225141696
    # original_trajectory = all_points[20:40] # 有轉折，Loss: 10.254627283910422
    # original_trajectory = all_points[240:260] # 有轉折，Loss: 7.18325887445171
    # original_trajectory = all_points[252:272] # 無轉折，Loss: 2.0963879783256982
    # original_trajectory = all_points[320:340] # 有轉折，Loss: 7.609487942479577
    # original_trajectory = all_points[325:345] # 無轉折，Loss: 1.5550962811627265
    # original_trajectory = all_points[325:345]
    # original_trajectory = all_points[28:252]
    original_trajectory = all_points[28:252]
    
    for p in original_trajectory:
        print(p.fid, p.x, p.y, p.z, p.timestamp)
    
    fps = 120
    flight_time = 10
    # flight_time = -0.1
    # flight_time = original_trajectory[-1].timestamp - original_trajectory[0].timestamp
    initial_V = np.array([0, 0, 0])
    
    first_point = np.array([original_trajectory[0].x, original_trajectory[0].y, original_trajectory[0].z])
    second_point = np.array([original_trajectory[1].x, original_trajectory[1].y, original_trajectory[1].z])

    # initial_V = (second_point - first_point) / (original_trajectory[1].timestamp - original_trajectory[0].timestamp)
    # print(initial_V)

    starting_point = [original_trajectory[0].x, original_trajectory[0].y, original_trajectory[0].z,
                          original_trajectory[0].timestamp]
    
    optimized_V, loss = optimize_velocity(original_trajectory, starting_point, fps, flight_time, initial_V)
    if optimized_V is not None:
        predicted_trajectory = physics_predict3d_v2(starting_point, optimized_V, fps, flight_time)
        plot_trajectories(original_trajectory, predicted_trajectory)
    '''



'''
        # Find y and z peak
        peak_idx = findPeaks(trajectory)
        if peak_idx != -1:
            if len(trajectory[:peak_idx]) > 1:
                sub_trajectory.append(trajectory[:peak_idx])
            if len(trajectory[peak_idx:]) > 1:
                sub_trajectory.append(trajectory[peak_idx:])
            # sub_trajectory = [trajectory[:peak_idx], trajectory[peak_idx:]]
        else:
            sub_trajectory = [trajectory]

        for traj in sub_trajectory:
            print('--------------------')
            print(f"{traj[0].fid} - {traj[-1].fid}")

            y = [p.y for p in traj if p.visibility == 1]
            z = [p.z for p in traj if p.visibility == 1]
            plt.scatter(y, z, c=c, alpha=0.5, label='Original Trajectory', s=8)

            isFly, optimized_V, pred_trajectory = is_flight_trajectory(traj, loss_function, evaluation_method, inlier_ratio=INLIER_RATIO)
            
            # Draw
            if pred_trajectory is not None:
                pred_y = [p[1] for p in pred_trajectory]
                pred_z = [p[2] for p in pred_trajectory]
            if isFly == True:
                plt.plot(pred_y, pred_z, c=c, alpha=0.7, label='Predicted Trajectory', linewidth=1)
                # for i in range(1, len(traj)):
                #     p = traj[i]
                #     plt.text(p.y, p.z, str(p.fid), fontsize=8, color='red', ha='left', va='bottom')
                # start point
                plt.scatter(traj[0].y, traj[0].z, c=c, s=10)
                plt.text(traj[0].y, traj[0].z, str(traj[0].fid), fontsize=10, color='red', ha='left', va='bottom')
            elif isFly == False and optimized_V is not None:
                plt.plot(pred_y, pred_z, c='grey', alpha=0.7, label='Predicted Trajectory', linewidth=1)
                # start point
                plt.scatter(traj[0].y, traj[0].z, c=c, s=10)
                plt.text(traj[0].y, traj[0].z, str(traj[0].fid), fontsize=10, color='blue', ha='left', va='bottom')
            elif isFly == False and optimized_V is None:
                print("non fly")
                plt.scatter(y, z, c=c, label='Original Trajectory', s=8)
                # start point
                plt.scatter(traj[0].y, traj[0].z, c=c, s=10)
                plt.text(traj[0].y, traj[0].z, str(traj[0].fid), fontsize=10, color='blue', ha='left', va='bottom')     

     
            # # TODO: 暴力法切斷軌跡
            # if fly == False:
            #     segment_length = 5
            #     for j in range(0, len(traj), segment_length):
            #         sub_traj = traj[j:j+segment_length]
                    
            #         starting_point = [sub_traj[0].x, sub_traj[0].y, sub_traj[0].z, sub_traj[0].timestamp]
            #         flight_time = sub_traj[-1].timestamp - sub_traj[0].timestamp

            #         optimized_V, loss = optimize_velocity(sub_traj, starting_point, fps, flight_time, initial_V)
            #         if optimized_V is not None:
        '''



def physics_predict3d_reverse(starting_point, second_point, flight_time=10, touch_ground_cut=True, alpha=0.2151959552, g=9.81):
    
    # fps = 1/(second_point[3] - starting_point[3])
    fps = 120.0

    initial_velocity = (second_point[:3]-starting_point[:3]) * fps # shape: (3,) unit: m/s

    traj = solve_ivp(lambda t, y: bm_ball(t, y, alpha=alpha, g=g), [flight_time, 0], np.concatenate((starting_point[:3], initial_velocity)), t_eval = np.arange(flight_time, 0, -1/fps)) # traj.t traj.y

    xyz = np.swapaxes(traj.y[:3,:], 0, 1) # shape: (N points, 3)
    t = np.expand_dims(traj.t,axis=1) # shape: (N points, 1)
    trajectories = np.concatenate((xyz, t),axis=1) # shape: (N points, 4)

    # Cut the part under the ground
    if touch_ground_cut:
        for i in range(trajectories.shape[0]-1):
            if trajectories[i,2] >= 0 and trajectories[i+1,2] <= 0:
                trajectories = trajectories[:i+1,:]
                break
    # Add timestamp correctly
    trajectories[:,3] += (starting_point[3]) # shape: (N points, 4)

    return trajectories

def loss_function_reverse(second_point, starting_point, real_trajectory, flight_time=10, alpha=0.2151959552, g=9.81, fps=120):
    second_point = np.array(second_point)  # Convert second_point to a numpy array
    predicted_trajectory = physics_predict3d_reverse(starting_point, second_point, flight_time, touch_ground_cut=False, alpha=alpha, g=g)
    
    # Compare predicted trajectory with real trajectory
    total_loss = 0
    real_timestamps = np.array([point.timestamp for point in real_trajectory])
    predicted_timestamps = predicted_trajectory[:, 3]
    
    # Find the closest predicted point for each real point based on timestamp
    for real_point in real_trajectory:
        original_timestamp = real_point.timestamp
        closest_idx = np.argmin(np.abs(predicted_timestamps - original_timestamp))
        if 0 <= closest_idx < len(predicted_trajectory):
            predicted_xyz = predicted_trajectory[closest_idx, :3]
            original_xyz = np.array([real_point.x, real_point.y, real_point.z])
            total_loss += np.linalg.norm(original_xyz - predicted_xyz)
    
    return total_loss

# Optimize the second point (using gradient descent or other optimization methods)
def optimize_second_point_reverse(starting_point, real_trajectory, initial_second_point, flight_time=10, alpha=0.2151959552, g=9.81, fps=120, method='Nelder-Mead'):
    # Minimize the loss function
    result = minimize(
        loss_function_reverse,  # The objective function to minimize
        initial_second_point,    # Initial guess for the second point
        args=(starting_point, real_trajectory, flight_time, alpha, g, fps),  # Additional arguments for the loss function
        method=method            # Optimization method
    )
    
    if result.success:
        optimized_second_point = result.x  # Optimal second point
        loss = result.fun                # The total loss at the optimal solution
        print("Optimization successful:", optimized_second_point)
        print("Loss:", loss)
        return optimized_second_point, loss
    else:
        print("Optimization failed:", result.message)
        return None, None



# Physics prediction (reverse)
def physics_predict3d_v2_reverse(final_point, v, fps, flight_time=10, touch_ground_cut=True, alpha=0.2151959552, g=9.81):
    initial_velocity = v  # Reverse velocity to simulate backwards

    traj = solve_ivp(lambda t, y: bm_ball(t, y, alpha=0.242, g=g), [flight_time, 0], np.concatenate((final_point[:3], initial_velocity)), t_eval = np.arange(flight_time, 0, -1/fps)) # traj.t traj.y

    xyz = np.swapaxes(traj.y[:3,:], 0, 1) # shape: (N points, 3)
    t = np.expand_dims(traj.t, axis=1) # shape: (N points, 1)
    trajectories = np.concatenate((xyz, t), axis=1) # shape: (N points, 4)

    # Cut the part under the ground
    if touch_ground_cut:
        for i in range(trajectories.shape[0]-1):
            if trajectories[i,2] >= 0 and trajectories[i+1,2] <= 0:
                trajectories = trajectories[:i+1,:]
                break
    # Add timestamp correctly
    trajectories[:,3] += (final_point[3]) # shape: (N points, 4)

    return trajectories # shape: (N points, 4) , including final_point

# Loss function to optimize velocity
def loss_distance_reverse(V, original_trajectory, final_point, fps, flight_time):
    total_loss = 0
    predicted_trajectory = physics_predict3d_v2_reverse(final_point, V, fps, flight_time, touch_ground_cut=False, alpha=0.242)
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

# Optimization function
def optimize_velocity_reverse(original_trajectory, final_point, fps, flight_time, initial_V, method='Nelder-Mead'):
    """Optimize the velocity V using scipy's minimize function in reverse direction.""" 
    result = minimize(
        loss_distance_reverse,  # The objective function to minimize
        initial_V,              # Initial guess for the velocity
        args=(original_trajectory, final_point, fps, flight_time),  # Additional arguments for the loss function
        method=method           # Optimization method
    )
    
    if result.success:
        optimized_V = result.x
        loss = result.fun # The total loss at the optimal solution
        print("Optimization successful:", optimized_V)
        print("Loss:", loss)
        return optimized_V, loss
    else:
        print("Optimization failed:", result.message)
        return None, None
