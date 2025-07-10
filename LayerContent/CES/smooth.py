from multiprocessing import current_process
import sys
import os
import numpy as np
import copy
import csv
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import linear_model
from sklearn.cluster import DBSCAN
import shapely.geometry as geom
from skspatial.objects import Line, Point
from scipy.signal import savgol_filter
import joblib

DIRNAME = os.path.dirname(os.path.abspath(__file__))
ROOTDIR = os.path.dirname(DIRNAME)
ROOTDIR = os.path.dirname(ROOTDIR)
REPLAYDIR = f"{ROOTDIR}/replay/"

sys.path.insert(0, ROOTDIR)
sys.path.insert(0, os.path.join(ROOTDIR, 'lib'))
from lib.point import Point as P
from lib.writer import CSVWriter

FPS = 120
# TREND = FPS / 12
TREND = 5
WINDOW_SIZE = 30

def fitVertical2dPlane(points):
    # input : list of <Point>
    track = []
    for point in points:
        if point.visibility != 0:
            track.append([point.x, point.y, point.z, point.timestamp])
    track = np.array(track)

    M = track.shape[0]

    ret_2d = np.zeros((M,3)) # shape: (M,3) 3: XY,Z,t

    model = linear_model.LinearRegression() # 回歸線
    sample_weight = np.zeros(M)
    for i in range(M-1):
        sample_weight[i] = np.linalg.norm(np.array([track[i+1,0],track[i+1,1]]) - np.array([track[i,0],track[i,1]]))
    model.fit(track[:,0].reshape(-1,1), track[:,1], sample_weight=sample_weight)

    a = model.coef_[0] # 斜率
    b = model.intercept_ # 截距

    project_2d = []
    for i in range(M):
        v1 = np.array([track[i,0]-track[0,0], track[i,1]-track[0,1]])
        v2 = np.array([1,a])
        ret_2d[i,0] = abs(np.dot(v1,v2)/np.linalg.norm(v2)) # 投影至平面
        ret_2d[i,1] = track[i,2] # -track[0,2]
        ret_2d[i,2] = track[i,3]
        point = P(fid=points[i].fid,  visibility=points[i].visibility, x=ret_2d[i,0], y=ret_2d[i,1], timestamp=ret_2d[i,2], event=points[i].event)
        project_2d.append(point)

    return project_2d

def fit2d(points, deg=8, smooth_2d_x_accel=True):
    # input: list of <Point>
    X = []
    Y = []
    for point in points:
        if int(point.visibility) != 0:
            X.append(point.x)
            Y.append(point.y)
    X = np.copy(X)
    Y = np.copy(Y)
    sample_num = X.shape[0]
    curve_data=np.zeros((sample_num, 2))
    points_tmp=np.zeros((X.shape[0],2))
    points_new=np.zeros((X.shape[0],2))

    fitx = X
    fity = Y
    idx = np.isfinite(fitx) & np.isfinite(fity)
    f1, _, _, _, _ = np.polyfit(fitx[idx], fity[idx], deg, full=True)
    p1 = np.poly1d(f1)

    points_tmp[:,0]=fitx
    points_tmp[:,1]=fity
    dataX = np.linspace(np.nanmin(fitx[idx]), np.nanmax(fitx[idx]),X.shape[0])
    dataY = p1(dataX)
    curve_data[:,0]=dataX
    curve_data[:,1]=dataY

    coords = curve_data[:,:]
    line = geom.LineString(coords)
    for j in range(0,X.shape[0],1):
        point = geom.Point(points_tmp[j,0], points_tmp[j,1])
        if np.isnan(points_tmp[j,0]):
            points_new[j,0]=np.nan
            points_new[j,1]=np.nan
        else:
            point_on_line = line.interpolate(line.project(point))

            points_new[j,0]=(point_on_line.x)
            points_new[j,1]=(point_on_line.y)

    # 位移量平滑化
    if smooth_2d_x_accel:
        plot_x = []
        plot_y = []
        for j in range(0,X.shape[0],1):
            point = geom.Point(points_tmp[j,0], points_tmp[j,1])
            point_on_line = line.interpolate(line.project(point))
            if j == 0:
                tmp = point_on_line.x
                continue
            else:
                plot_x.append(j)
                plot_y.append(point_on_line.x-tmp)
                tmp = point_on_line.x
        # 多項式 fit
        param = np.polyfit(plot_x, plot_y, 3)
        z = np.polyval(param, plot_x)
        newX = []
        for j in range(0,X.shape[0],1):
            if j == 0:
                tmp_x = 0
            else:
                tmp_x += z[j-1]
            newX.append(tmp_x)
        newY = p1(newX)
        for j in range(0,X.shape[0],1):
            points_new[j,0]=(newX[j])
            points_new[j,1]=(newY[j])

    smooth_project = []
    base = points_new[0,0]
    for j in range(0,X.shape[0],1):
        point = P(fid=points[j].fid, timestamp=points[j].timestamp, visibility=points[j].visibility, x=points_new[j,0]-base, y=points_new[j,1], event=points[j].event)
        smooth_project.append(point)

    return smooth_project

def fit3d(points, deg=4):
    x = []
    y = []
    z = []
    for point in points:
        if point.visibility != 0:
            x.append(point.x)
            y.append(point.y)
            z.append(point.z)
    x = np.array(x)
    y = np.array(y)
    z = np.array(z)
    N = x.shape[0]

    x_new=np.zeros(N)
    y_new=np.zeros(N)
    z_new=np.zeros(N)
    curve_x=np.zeros(N)
    curve_y=np.zeros(N)
    curve_z=np.zeros(N)

    f_start=0
    f_end = x.shape[0]
    fitx1 = x[f_start:f_end]
    fity1 = y[f_start:f_end]
    fitz1 = z[f_start:f_end]
    idx = np.isfinite(fitx1) & np.isfinite(fity1) & np.isfinite(fitz1)

    yx= np.polyfit(fity1[idx], fitx1[idx], deg)
    pYX = np.poly1d(yx)
    yz= np.polyfit(fity1[idx], fitz1[idx], deg)
    pYZ = np.poly1d(yz)

    y_sorted_new = fity1

    # 將 y 位移量平滑化
    y_trans = []
    x_tmp = []
    i = 0
    for y in y_sorted_new:
        if i == 0:
            y_start = y
            last_y = y
        else:
            y_trans.append(y - last_y)
            last_y = y
            x_tmp.append(i-1)
        i += 1
    param = np.polyfit(x_tmp, y_trans, 3)
    polyXY = np.poly1d(param)
    y_trans_new = polyXY(x_tmp)

    ## 計算縮放比例，確保平滑後的y總位移量與原始y總位移量相同，處理後的y存在y_smooth_new
    origin_len = abs(fity1[0] - fity1[-1])
    new_len = 0
    for i in range(len(y_trans_new)):
        new_len += abs(y_trans_new[i])
    strech = origin_len / new_len

    y_smooth_new = []
    y_smooth_new.append(y_start)
    for i in range(len(y_trans_new)):
        y_smooth_new.append(y_smooth_new[i] + y_trans_new[i]*strech)

    x_pYX_new = pYX(y_smooth_new)
    z_pYZ_new = pYZ(y_smooth_new)

    x_new[f_start:f_end]=fitx1
    y_new[f_start:f_end]=fity1
    z_new[f_start:f_end]=fitz1
    curve_x[f_start:f_end]=x_pYX_new
    curve_y[f_start:f_end]=y_smooth_new
    curve_z[f_start:f_end]=z_pYZ_new

    new_points = []
    c = 0
    for i in range(len(points)):
        if points[i].visibility != 0:
            point = P(fid=points[i].fid, timestamp=points[i].timestamp, visibility=points[i].visibility, x=curve_x[c], y=curve_y[c], z=curve_z[c], event=points[i].event)
            c += 1
        else:
            point = P(fid=points[i].fid, timestamp=points[i].timestamp, visibility=points[i].visibility, x=points[i].x, y=points[i].y, z=points[i].z, event=points[i].event)
        new_points.append(point)
    return new_points

def denoise(points, deg=8):
    x = []
    y = []
    z = []
    tmp_points = copy.deepcopy(points)

    # remove duplicate point
    tmp_point = []
    for i in range(len(tmp_points)):
        if tmp_points[i].visibility == 1:
            for j in range(len(tmp_point)):
                if tmp_points[i].x == tmp_point[j].x and tmp_points[i].y == tmp_point[j].y and tmp_points[i].z == tmp_point[j].z and tmp_points[i].event == 0:
                    tmp_points[i].visibility = 0
                    break
            tmp_point.append(tmp_points[i])

    for point in tmp_points:
        if point.visibility != 0:
            x.append(point.x)
            y.append(point.y)
            z.append(point.z)
    x = np.array(x)
    y = np.array(y)
    z = np.array(z)

    f_start=0
    f_end = x.shape[0]
    fitx1 = x[f_start:f_end]
    fity1 = y[f_start:f_end]
    fitz1 = z[f_start:f_end]
    idx = np.isfinite(fitx1) & np.isfinite(fity1) & np.isfinite(fitz1)

    yx= np.polyfit(fity1[idx], fitx1[idx], 2)
    polyYX = np.poly1d(yx)
    yz= np.polyfit(fity1[idx], fitz1[idx], deg)
    polyYZ = np.poly1d(yz)

    array_yz = abs(z - polyYZ(y))
    max_accept_deviation = 0.8
    mask_yz = array_yz >= max_accept_deviation
    rows_to_del = np.asarray(tuple(te for te in np.where(mask_yz)[0]))

    for i in rows_to_del:
        tmp_points[i].visibility = 0

    array_yx = abs(x - polyYX(y))
    mask_yx = array_yx >= max_accept_deviation
    rows_to_del = np.asarray(tuple(te for te in np.where(mask_yx)[0]))

    for i in rows_to_del:
        tmp_points[i].visibility = 0

    return tmp_points

def interpolatePoints(points):
    x = []
    y = []
    z = []
    all_points = copy.deepcopy(points)
    for p in points:
        if p.visibility != 0:
            x.append(p.x)
            y.append(p.y)
            z.append(p.z)
    x = np.array(x)
    y = np.array(y)
    z = np.array(z)

    f_start=0
    f_end = x.shape[0]
    fitx1 = x[f_start:f_end]
    fity1 = y[f_start:f_end]
    fitz1 = z[f_start:f_end]
    idx = np.isfinite(fitx1) & np.isfinite(fity1) & np.isfinite(fitz1)

    # get yz poly
    yz= np.polyfit(fity1[idx], fitz1[idx], 4)
    polyYZ = np.poly1d(yz)

    # 用等距內插出缺失點
    # 插 xy, 2次, 直接插
    # 插 yz, 用 y 去算 polyYZ
    count_zero = 0
    start = end = -1
    for i in range(len(all_points)):
        if all_points[i].visibility == 0 and count_zero == 0:
            start = i
            count_zero += 1
        elif all_points[i].visibility == 0:
            count_zero += 1
        elif all_points[i].visibility != 0 and count_zero != 0:
            end = i-1
        if end != -1:
            x1 = all_points[start-1].x
            y1 = all_points[start-1].y
            x2 = all_points[end+1].x
            y2 = all_points[end+1].y
            x_len = (x2 - x1) / (count_zero + 1)
            y_len = (y2 - y1) / (count_zero + 1)
            for j in range(count_zero):
                print('interpolate ', 'Frame:', all_points[start + j].fid)
                all_points[start + j].x = all_points[start + j - 1].x + x_len
                all_points[start + j].y = all_points[start + j - 1].y + y_len
                all_points[start + j].z = polyYZ(all_points[start + j].y)
                all_points[start + j].visibility = 1
            start = end = -1
            count_zero = 0

    return all_points

def removeOuterPoint(date):
    file_path = os.path.join(REPLAYDIR, date, 'Model3D.csv')
    output_path = os.path.join(REPLAYDIR, date, 'Model3D_mod.csv')
    with open(file_path, 'r', newline='') as csvFile:
        rows = list(csv.DictReader(csvFile))
        with open(output_path, 'w', newline='') as outputfile:
            writer = csv.writer(outputfile)
            writer.writerow(['Frame', 'Visibility', 'X', 'Y', 'Z', 'Event', 'Timestamp'])
            for row in rows:
                x = float(row['X'])
                y = float(row['Y'])
                z = float(row['Z'])
                # if x > 4.5 or x < -4.5:
                #     row['Visibility'] = 0
                # if y > 8 or y < -8:
                #     row['Visibility'] = 0
                # if z > 10 or z < 0:
                #     row['Visibility'] = 0
                if x > 2.8 or x < -2.8:
                    row['Visibility'] = 0
                if y > 3 or y < -3:
                    row['Visibility'] = 0
                if z > 5 or z < 0:
                    row['Visibility'] = 0
                row['Frame'] = int(float(row['Frame']))
                # writer.writerow([row['Frame'], row['Visibility'], row['X'], row['Y'], row['Z'], 0, row['Timestamp']])

            # for idx in tqdm(range(0, len(rows) - WINDOW_SIZE + 1, 1)):
            #     window = np.array([[float(rows[idx + j]['X']), float(rows[idx + j]['Y']), float(rows[idx + j]['Z'])] for j in range(WINDOW_SIZE)])
            #     dbscan = DBSCAN(eps=0.3, min_samples=3, n_jobs=-1)
            #     dbscan.fit(window)
            #     labels = dbscan.labels_
            #     for j in range(WINDOW_SIZE):
            #         if labels[j] == -1:
            #             rows[idx + j]['Visibility'] = 0
            #             print('Del:')
            #             print(rows[idx + j]['Frame'])

            for row in rows:
                writer.writerow([row['Frame'], row['Visibility'], row['X'], row['Y'], row['Z'], 0, row['Timestamp']])

def removeOuterPoint_batch(save_path, points):
    output_path = os.path.join(save_path, 'Model3D_mod.csv')
    csv3DWriter = CSVWriter(name='Model3D_mod', filename=output_path)
    output_points = []
    for point in points:
        x = point.x
        y = point.y
        z = point.z
        if x > 4 or x < -4:
            point.visibility = 0
        if y > 7 or y < -7:
            point.visibility = 0
        if z > 5 or z < 0:
            point.visibility = 0
        if y > 0 and z < 1.5:
            point.visibility = 0
        if y < -4 and z > 2:
            point.visibility = 0
        # output_points.append(point)
        csv3DWriter.writePoints(point)
    csv3DWriter.csvfile.close()
    return points

def detectEvent(date, points=1):
    # file = 'Model3D_mod_' + str(points) + '.csv'
    file = 'Model3D_mod.csv'
    # file = 'Model3D_rm.csv'
    file_path = os.path.join(REPLAYDIR, date, file)
    output_file = 'Model3D_event_' + str(points) + '.csv'
    output_path = os.path.join(REPLAYDIR, date, output_file)
    serve = False
    serve_idx = -1
    with open(file_path, 'r', newline='') as csvFile:
        rows = list(csv.DictReader(csvFile))
        # detect serve
        for idx in range(len(rows)):
            if int(float(rows[idx]['Visibility'])) == 1 and float(rows[idx]['Y']) < 0:
                rows[idx]['Event'] = 2
                serve = True
                serve_idx = idx
                break
        # detect dead
        dead = len(rows)-1
        while(dead > 0):
            if int(float(rows[dead]['Visibility'])) == 1:
                rows[dead]['Event'] = 3
                break 
            else:
                dead -= 1
        # detect hit
        go_big = 0
        go_small = 0
        min_y = 9
        max_y = -9
        find_min = False
        find_max = False
        count_done = 0
        for idx in range(serve_idx, len(rows)):
            # if int(float(rows[idx]['Visibility'])) == 1:
            #     print()
            #     print('loop:', float(rows[idx]['Frame']))
            if find_max == True and int(float(rows[idx]['Visibility'])) == 1:
                if float(rows[idx]['Y']) > max_y:
                    max_y = float(rows[idx]['Y'])
                    max_idx = idx
                    count_done = 0
                else:
                    count_done += 1
                # print(idx, '(', float(rows[idx]['Frame']), float(rows[idx]['Y']), ')', ':', count_done)
                if count_done > TREND:
                    rows[max_idx]['Event'] = 1
                    # print('!!! HIT !!!', max_idx, float(rows[max_idx]['Frame']))
                    # print('max_y:', max_y, 'min_y:', min_y)
                    find_max = False
                    go_big = 0
                    go_small = count_done
                    max_y = -9
                    min_y = 9
                    count_done = 0
            elif find_min == True and int(float(rows[idx]['Visibility'])) == 1:
                if float(rows[idx]['Y']) < min_y:
                    min_y = float(rows[idx]['Y'])
                    min_idx = idx
                    count_done = 0
                else:
                    count_done += 1
                # print(idx, '(', float(rows[idx]['Frame']), float(rows[idx]['Y']), ')', ':', count_done)
                if count_done > TREND:
                    rows[min_idx]['Event'] = 1
                    # print('!!! HIT !!!', min_idx, float(rows[min_idx]['Frame']))
                    find_min = False
                    go_small = 0
                    go_big = 0
                    min_y = 9
                    max_y = -9
                    count_done = 0
            elif int(float(rows[idx]['Visibility'])) == 1 and serve == True:
                # print('!!!!!', int(float(rows[idx]['Frame'])))
                # print('check:', float(rows[idx]['Y']), 'max_y:', max_y, 'min_y:', min_y)
                if float(rows[idx]['Y']) > max_y:
                    # print('find max')
                    max_y = float(rows[idx]['Y'])
                    go_big += 1
                    # print('big:', go_big)
                    if go_big > TREND:
                        max_idx = idx
                        find_max = True
                        # print('---> find_max', max_idx, float(rows[max_idx]['Frame']), float(rows[max_idx]['Y']), 'max_y:', max_y)
                elif float(rows[idx]['Y']) < min_y:
                    # print('find min')
                    min_y = float(rows[idx]['Y'])
                    go_small += 1
                    # print('small:', go_small)
                    if go_small > TREND:
                        min_idx = idx
                        find_min = True
                        # print('---> find_min', min_idx, float(rows[min_idx]['Frame']), float(rows[min_idx]['Y']))
        # write back file
        with open(output_path, 'w', newline='') as outputfile:
            writer = csv.writer(outputfile)
            writer.writerow(['Frame', 'Visibility', 'X', 'Y', 'Z', 'Event', 'Timestamp'])
            for row in rows:
                writer.writerow([row['Frame'], row['Visibility'], row['X'], row['Y'], row['Z'], row['Event'], row['Timestamp']])

def detectEvent_batch(save_path, points):
    output_path = os.path.join(save_path, 'Model3D_event_1.csv')
    csv3DWriter = CSVWriter(name='Model3D_event_1', filename=output_path)
    output_points = []
    serve = False
    serve_idx = -1
    go_big = 0
    go_small = 0
    min_y = 9
    max_y = -9
    find_min = False
    find_max = False
    count_done = 0

    # detect serve
    for idx, point in enumerate(points):
        if point.visibility == 1 and point.y < 0:
            point.event = 2
            # print('### SERVE ###', point.fid)
            serve = True
            serve_idx = idx
            break
    
    # detect dead
    for idx in range(len(points)-1, -1, -1):
        if points[idx].visibility == 1:
            points[idx].event = 3
            # print('### DEAD ###', points[idx].fid)
            break
    
    # detect hit
    for idx in range(serve_idx, len(points)):
        if find_max == True and points[idx].visibility == 1:
            if points[idx].y > max_y:
                max_y = points[idx].y
                max_idx = idx
                # print('MAX:', points[max_idx].fid)
                count_done = 0
            else:
                count_done += 1
            # print(idx, '(', points[idx].fid, points[idx].timestamp, ')', ':', count_done)
            if count_done > TREND:
                points[max_idx].event = 1
                # print('### HIT ###', points[max_idx].fid)
                find_max = False
                go_big = 0
                go_small = 0
                max_y = -9
                min_y = 9
                count_done = 0
        elif find_min == True and points[idx].visibility == 1:
            if points[idx].y < min_y:
                min_y = points[idx].y
                min_idx = idx
                count_done = 0
            else:
                count_done += 1
            # print(idx, '(', points[idx].fid, points[idx].timestamp, ')', ':', count_done)
            if count_done > TREND:
                points[min_idx].event = 1
                # print('### HIT ###', points[max_idx].fid)
                find_min = False
                go_small = 0
                go_big = 0
                min_y = 9
                max_y = -9
                count_done = 0
        elif points[idx].visibility == 1 and serve == True:
            # print('!!!!!', points[idx].fid)
            if points[idx].y > max_y:
                max_y = points[idx].y
                go_big += 1
                # print('big:', go_big)
                if go_big > TREND:
                    max_idx = idx
                    find_max = True
                    # print('MAX:', points[max_idx].fid)
            elif points[idx].y < min_y:
                min_y = points[idx].y
                go_small += 1
                if go_small > TREND:
                    min_idx = idx
                    find_min = True
            
    # write back file
    csv3DWriter.writePoints(points)    

    return points

def smoothByEvent(date, points=1):
    file = 'Model3D_event_' + str(points) + '.csv'
    file_path = os.path.join(REPLAYDIR, date, file)
    output_file = 'Model3D_smooth_' + str(points) + '.csv'
    output_path = os.path.join(REPLAYDIR, date, output_file)
    with open(file_path, 'r', newline='') as csvFile:
        critical_frame = []
        rows = list(csv.DictReader(csvFile))
        for idx in range(len(rows)):
            if int(rows[idx]['Event']) != 0:
                critical_frame.append(idx)
        for idx in range(len(critical_frame) - 1):
            start_idx = critical_frame[idx]
            current_idx = start_idx
            end_idx = critical_frame[idx + 1]
            points = []
            while current_idx <= end_idx:
                p = P(fid=rows[current_idx]['Frame'], timestamp=rows[current_idx]['Timestamp'], visibility=rows[current_idx]['Visibility'], x=rows[current_idx]['X'], y=rows[current_idx]['Y'], z=rows[current_idx]['Z'], event=rows[current_idx]['Event'])
                points.append(p)
                current_idx += 1
            points = denoise(points)
            points = interpolatePoints(points)
            points = fit3d(points)
            current_idx = start_idx
            for point in points:
                rows[current_idx]['Visibility'] = point.visibility
                rows[current_idx]['X'] = point.x
                rows[current_idx]['Y'] = point.y
                rows[current_idx]['Z'] = point.z
                current_idx += 1
        # write back file
        with open(output_path, 'w', newline='') as outputfile:
            writer = csv.writer(outputfile)
            writer.writerow(['Frame', 'Visibility', 'X', 'Y', 'Z', 'Event', 'Timestamp'])
            for row in rows:
                writer.writerow([row['Frame'], row['Visibility'], row['X'], row['Y'], row['Z'], row['Event'], row['Timestamp']])

def detectBallTypeByEvent(date, points=1):
    # file = 'Model3D_smooth_' + str(points) + '.csv'
    file = 'Model3D_event_' + str(points) + '.csv'
    file_path = os.path.join(REPLAYDIR, date, file)
    ball_type = []
    with open(file_path, 'r', newline='') as csvFile:
        critical_frame = []
        rows = list(csv.DictReader(csvFile))
        for idx in range(len(rows)):
            if int(rows[idx]['Event']) != 0:
                critical_frame.append(idx)
        for idx in range(len(critical_frame) - 1):
            start_idx = critical_frame[idx]
            current_idx = start_idx
            end_idx = critical_frame[idx + 1]
            points = []
            while current_idx <= end_idx:
                p = P(fid=rows[current_idx]['Frame'], timestamp=rows[current_idx]['Timestamp'], visibility=rows[current_idx]['Visibility'], x=rows[current_idx]['X'], y=rows[current_idx]['Y'], z=rows[current_idx]['Z'], event=rows[current_idx]['Event'])
                points.append(p)
                current_idx += 1
            ball_type.append(detectBallType(points))
    return ball_type

def detectBallType(points):
    # 小球、挑球、推球、平球、殺球、切球、長球、沒過網
    # 前場: 網前 到 2m
    # 後場: 大於 4.5m
    # start_x, start_y, start_z, end_x, end_y, height, frame_num, label
    start_x, start_y, start_z = points[0].x, points[0].y, points[0].z
    end_x, end_y = points[-1].x, points[-1].y
    frame_num = len(points)
    height = 0
    for i in range(frame_num):
        if points[i].z > height:
            height = points[i].z
    # 沒過網
    if start_y * end_y > 0:
        # return '例外'
        return 'exception'
    # 殺球
    elif start_z < height + 0.1 and start_z > height - 0.1 and frame_num < 180 and abs(end_y) >= 4.5:
        # return '殺球'
        return 'smash'
    # 小球
    elif abs(start_y) < 2 and abs(end_y) < 2:
        # return '小球'
        return 'net' 
    # 切球
    elif abs(start_y) >= 2 and abs(end_y) < 4.5 and start_z > 2:
        # return '切球'
        return 'drop'
    # 挑球
    elif abs(end_y) >= 4.5 and start_z < 1:
        # return '挑球'
        return 'lift'
    # 推球
    elif abs(start_y) < 2 and abs(end_y) >= 4.5 and height < 3:
        # return '推球'
        return 'push'
    # 平球
    elif abs(start_y) < 4.5 and height < 3:
        # return '平球'
        return 'drive'
    # 長球
    elif (abs(start_y) >= 4.5 and abs(end_y) >= 4.5) and height >= 3:
        # return '長球'
        return 'clear'
    # 例外
    else:
        # return '例外'
        return 'exception'

def detectBallTypeRF(points):
    model = joblib.load(os.path.join(DIRNAME, 'RF_Rules_model'))
    start_x, start_y, start_z = points[0].x, points[0].y, points[0].z
    end_x, end_y = points[-1].x, points[-1].y
    frame_num = len(points)
    height = 0
    for i in range(frame_num):
        if points[i].z > height:
            height = points[i].z
    X = [[start_x, start_y, start_z, end_x, end_y, height, frame_num]]
    result = model.predict(X)
    if result[0] == 1:
        return '小球'
    elif result[0] == 2:
        return '切球'
    elif result[0] == 3:
        return '挑球'
    elif result[0] == 4:
        return '推球'
    elif result[0] == 5:
        return '平球'
    elif result[0] == 6:
        return '長球'
    elif result[0] == 7:
        return '殺球'


def removeMotionOutliers(date):
    file_path = os.path.join(REPLAYDIR, date, 'Model3D_mod.csv')
    output_path = os.path.join(REPLAYDIR, date, 'Model3D_rm.csv')
    df = pd.read_csv(file_path)
    df = df.fillna(0)

    x = df['X'].tolist()
    y = df['Y'].tolist()
    z = df['Z'].tolist()
    vis = df['Visibility'].tolist()
    fid = df['Frame'].tolist()
    timestamp = df['Timestamp'].tolist()

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

                # print('-------------------------------------------------')
                # print(i, y[i])
                # print(last_visible_index, y[last_visible_index])
                # print(f'From frame {fid[last_visible_index]} to frame {fid[i]}:')
                # print(f'Distance: {distance:.2f}, Avg Speed: {avg_speed:.2f}, Frame Diff: {frame_diff}')
                # print('delta y -' if x[i] - x[last_visible_index] < 0 else 'delta y +')

                if (avg_speed > speed_threshold and frame_diff < frame_threshold):
                    if (prev_avg_speed > speed_threshold and prev_frame_diff < frame_threshold):
                        if (prev_delta_y_direction is not None and delta_y_direction != prev_delta_y_direction):
                            vis[last_visible_index] = 0
                            print(f'-------- remove Frame {fid[last_visible_index]} --------')
                            # print('speed:', prev_avg_speed, avg_speed)
                            # print('frame diff:', prev_frame_diff, frame_diff)
                            # print('delta y:', prev_delta_y_direction, delta_y_direction)
                
                distances.append(distance)
                avg_speeds.append(avg_speed)
                frame_diffs.append(frame_diff)

                prev_avg_speed = avg_speed
                prev_frame_diff = frame_diff
                prev_delta_y_direction = delta_y_direction

            else:
                # first visible point
                distances.append(0)
                avg_speeds.append(0)
                frame_diffs.append(0)

            # Update the last visible point index
            last_visible_index = i

        else:
            # non-visible point
            distances.append(0)
            avg_speeds.append(0)
            frame_diffs.append(0)

    # write back file
    with open(output_path, 'w', newline='') as outputfile:
        writer = csv.writer(outputfile)
        writer.writerow(['Frame', 'Visibility', 'X', 'Y', 'Z', 'Event', 'Timestamp'])
        for i in range(len(fid)):
            writer.writerow([fid[i], vis[i], x[i], y[i], z[i], 0, timestamp[i]])


def removeMotionOutliers_batch(save_path, points):
    output_path = os.path.join(save_path, 'Model3D_rm.csv')
    csv3DWriter = CSVWriter(name='Model3D_rm', filename=output_path)
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
                            print(f'-------- remove Frame {fid[last_visible_index]} --------')
                
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

    # write back file
    csv3DWriter.writePoints(points) 
    
    return points
