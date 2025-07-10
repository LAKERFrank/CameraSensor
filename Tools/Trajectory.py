# Lejun Shen, IACSS, 2017
# Measurement and Performance Evaluation of Lob Technique using Aerodynamic Model In Badminton Matches
import matplotlib.pyplot as plt
import math
import sys
import os
import numpy as np
from scipy.integrate import solve_ivp
import seaborn as sns
from itertools import product
sns.set()
DIRNAME = os.path.dirname(os.path.abspath(__file__))
ROOTDIR = os.path.dirname(DIRNAME)
sys.path.append(f"{ROOTDIR}/lib")
#10-50 ~ v
Velocity = np.arange(10, 50, 1)
#0-24 ~ incline
Incline = np.arange(0, 24, 1)

def physics_predict3d(starting_point, height_limit, theta , Vz, Vxy, Dir, flight_time=5, touch_target_cut=True, alpha=0.21520, g=9.8):
    initial_velocity = [math.sin(theta)*Vxy*Dir[0], math.cos(theta)*Vxy*Dir[1], Vz]

    traj = solve_ivp(lambda t, y: bm_ball(t, y, alpha=alpha, g=g), [0, flight_time], np.concatenate((starting_point[:3], initial_velocity)), t_eval = np.arange(0, flight_time, 0.033)) # traj.t traj.y
    xyz = np.swapaxes(traj.y[:3,:], 0, 1) # shape: (N points, 3)
    t = np.expand_dims(traj.t,axis=1) # shape: (N points, 1)
    trajectories = np.concatenate((xyz, t),axis=1) # shape: (N points, 4)
    # Cut the part under the ground
    if touch_target_cut:
        for i in range(trajectories.shape[0]-1):
            if trajectories[i,2] >= height_limit and trajectories[i+1,2] <= height_limit:
                trajectories = trajectories[:i+1,:]
                break
    
    return trajectories # shape: (N points, 4) , include input two Points

def find_traj(starting_point, end_point):
    combs = [(math.sin(math.radians(i))*v,math.cos(math.radians(i))*v) for v,i in product(Velocity,Incline)]
    Dist = end_point[:2]-starting_point[:2]
    theta = math.atan(abs(Dist[0]/Dist[1]))
    Dir = Dist/abs(Dist)
    min_dist = 99
    for Vz,Vxy in combs:
        traj = physics_predict3d(starting_point, end_point[2], theta, Vz, Vxy, Dir)
        pred_end = traj[-1]
        dist = math.sqrt((pred_end[0]-end_point[0])**2 + (pred_end[1]-end_point[1])**2 + (pred_end[2]-end_point[2])**2)
        
        if(dist < min_dist):
            min_dist = dist
            min_traj = traj
            
    return min_traj

def bm_ball(t,x,alpha=0.21520, g=9.8):
    # velocity
    v = math.sqrt(x[3]**2+x[4]**2+x[5]**2)
    # ordinary differential equations (3)
    xdot = [ x[3], x[4], x[5], -alpha*x[3]*v, -alpha*x[4]*v, -g-alpha*x[5]*v]
    return xdot


if __name__ == '__main__':
    # # test physics_predict3d function
    # #===========case 1==========
    # p1 = np.array([1.166179244,	2.050772068, 1.491008632,  2.033333333])
    # p2 = np.array([2.129169792, -3.708546295, 0.095679186, 2.95])
    #===========case 2==========
    p1 = np.array([1.151755569, 2.128970498, 1.50247747])
    p2 = np.array([1.857840885, -3.662499565, 0.083455074])
    # a = physics_predict3d(p1, p2)
    # print(a)
    # print([x for x in product(Velocity,Incline)])

    print(find_traj(p1,p2))
    sys.exit(1)
    