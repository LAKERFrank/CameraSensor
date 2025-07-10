# Lejun Shen, IACSS, 2017
# Measurement and Performance Evaluation of Lob Technique using Aerodynamic Model In Badminton Matches
import matplotlib.pyplot as plt
import math
import sys
import os
import numpy as np
from scipy.integrate import solve_ivp
import seaborn as sns
sns.set()

from lib.point import Point

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
    """空氣阻力軌跡預測公式

    Args:
        starting_point (tuple[float, float, float]): 起始點
        v (tuple[float, float, float]): 速度
        fps (int): 預測點數量(每秒)
        flight_time (int, optional): 飛行時間. Defaults to 10.
        touch_ground_cut (bool, optional): 是否停在落地點. Defaults to True.
        alpha (float, optional): 空氣阻力係數. Defaults to 0.2151959552.
        g (float, optional): 重力加速度. Defaults to 9.81.

    Returns:
        np.ndarray: 飛行軌跡
    """

    initial_velocity = v

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

    return trajectories # shape: (N points, 4) , include starting_point


def bm_ball(t,x,alpha=0.2151959552, g=9.81):
    # velocity
    v = math.sqrt(x[3]**2+x[4]**2+x[5]**2)
    # ordinary differential equations (3)
    xdot = [ x[3], x[4], x[5], -alpha*x[3]*v, -alpha*x[4]*v, -g-alpha*x[5]*v]
    return xdot

def get_correlated_dataset(n, dependency, mu, scale):
    latent = np.random.randn(n, 2)
    dependent = latent.dot(dependency)
    scaled = dependent * scale
    scaled_with_offset = scaled + mu
    # return x and y of the new, correlated dataset
    return scaled_with_offset[:, 0], scaled_with_offset[:, 1]

def PredictLandDis(start_height, datas, fps=60):
    # datas = np.array(initial_datas) # shape: (N datas, 2)
    # n = datas.shape[0]
    dis = []
    time = []
    p0 = np.array([0, 0, start_height, 0])

    for (v0, e0) in datas:
        initial_velocity = [0, v0 * math.cos(e0/180*math.pi), v0 * math.sin(e0/180*math.pi)]

        trajectories = physics_predict3d_v2(p0, initial_velocity, fps, alpha=0.235)

        # print(trajectories)

        heights = trajectories[:, 2]
        nearest_idx = np.abs(heights).argmin()

        sol = trajectories[nearest_idx, 1]
        dis.append(sol)
        time.append(trajectories[nearest_idx, 3])

    return dis, time

def PredictDis(start_height, end_height, datas, fps=60):
    # datas = np.array(initial_datas) # shape: (N datas, 2)
    # n = datas.shape[0]
    output = []
    
    p0 = np.array([0, 0, start_height, 0]) # (x, y, z, t)?

    for (v0, e0) in datas:
        initial_velocity = [0, v0 * math.cos(e0/180*math.pi), v0 * math.sin(e0/180*math.pi)]

        trajectories = physics_predict3d_v2(p0, initial_velocity, fps)

        # print(trajectories)

        heights = trajectories[:, 2]

        highest_idx = heights.argmax()
        nearest_idx = np.abs(heights - end_height).argmin()

        ans = []


        if start_height > end_height or highest_idx == nearest_idx: # 1 solution
            sol = trajectories[nearest_idx, 1]
            ans.append(sol)
        else: # 2 solutions
            left_idx = np.abs(heights[:highest_idx + 1] - end_height).argmin()
            right_idx = np.abs(heights[highest_idx:] - end_height).argmin() + highest_idx

            # print(f'left idx = {left_idx}, right_idx = {right_idx}')

            ans.append(trajectories[left_idx, 1])
            ans.append(trajectories[right_idx, 1])

        output.append(ans)

    return output


def testCalcDis():
    start_height = 1.557
    end_height = 0
    datas = [[13.56, 18.085], [12.75, 5]]

    dis, time = PredictLandDis(start_height, datas)

    print(dis)
    print(time)


if __name__ == '__main__':
    # test physics_predict3d function
    # p1 = np.array([5.6, 6.3, 2.9, 123.45])
    # p2 = np.array([5.4, 6.0, 3.1, 123.616])
    # a = physics_predict3d(p1, p2)
    # print(a)
    # sys.exit(1)

    testCalcDis()
    sys.exit(0)

    ###########################
    flight_time = 5
    fps = 120
    starting_point = [0, 0, 2.5, 0]
    datas = 1
    ###########################

    fig1 = plt.figure()
    ax1 = fig1.add_subplot(1,1,1)
    ax1.set_xlabel("Elevation(degree)")
    ax1.set_ylabel("Initial Velocity(km/hr)")

    fig2 = plt.figure()
    ax2 = fig2.add_subplot(1,1,1)
    ax2.set_title(f"Physics Model, Datas: {datas}")

    # Elevation & Speed Distribution
    corr = [[1,0],[0,1]]
    mu = 0, 75
    scale = 35, 20

    elevation, speed = get_correlated_dataset(datas, corr, mu, scale)

    elevation[elevation>85] = 80.0
    elevation[elevation<-80] = -80.0
    speed[speed<10] = 10.0
    # elevation = np.arange(-60,60,1) # -60 ~ 60
    # speed = np.arange(30, 150, 1) # 30 ~ 150


    # elevation = np.array([[]])
    speed = speed * 1000/3600 # km/hr -> m/s


    elevation = np.array([30])
    speed = np.array([20])
    print(f'elev = {elevation}')
    print(f'speed = {speed}')

    start_height = 2.5
    end_height = 4.32055

    cnt = 0
    for e,s in zip(elevation,speed):
        initial_velocity = [s * math.cos(e/180*math.pi), 0, s * math.sin(e/180*math.pi)]
        

        # traj = solve_ivp(bm_ball, [0, flight_time], starting_point + initial_velocity, t_eval = np.arange(0, flight_time, 1/fps)) # traj.t traj.y
        cnt += 1

        # xyz = np.swapaxes(traj.y[:3,:], 0, 1) # shape: (N points, 3)
        # t = np.expand_dims(traj.t,axis=1) # shape: (N points, 1)
        # trajectories = np.concatenate((xyz, t),axis=1) # shape: (N points, 4)
        # # Cut the part under the ground
        # for i in range(trajectories.shape[0]-1):
        #     if trajectories[i,2] >= 0 and trajectories[i+1,2] <= 0:
        #         trajectories = trajectories[:i+1,:]
        #         break

        trajectories = physics_predict3d_v2(starting_point, initial_velocity, fps)

        print(trajectories)

        heights = trajectories[:, 2]
        highest_idx = heights.argmax()
        print(f'highest idx = {highest_idx}')

        ans = []

        nearest_idx = np.abs(heights - end_height).argmin()

        if start_height > end_height or highest_idx == nearest_idx: # 1 solution
            sol = trajectories[nearest_idx, 0]
            ans.append(sol)
        else: # 2 solutions
            left_idx = np.abs(heights[:highest_idx + 1] - end_height).argmin()
            right_idx = np.abs(heights[highest_idx:] - end_height).argmin() + highest_idx

            print(f'left idx = {left_idx}, right_idx = {right_idx}')

            ans.append(trajectories[left_idx, 0])
            ans.append(trajectories[right_idx, 0])

        print(ans)

        # nearest_idx = np.abs(heights - end_height).argmin()
        # print(heights)
        # print(trajectories[nearest_idx])

        # ax1.scatter(e, s*3600/1000, color='red')
        ax2.plot(trajectories[:,0], trajectories[:,2], label=f"2D Physic-based trajectories FPS:{fps}, total: {cnt}",  marker='o', markersize=1)
        # for a in ans:
        #     ax2.plot(a[0], a[2],  marker='o', markersize=5)

    # plt.legend()
    plt.gca().set_aspect('equal', adjustable='box')
    plt.savefig('result.png')
    # plt.show()