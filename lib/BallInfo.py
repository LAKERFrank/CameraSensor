import os
import pandas as pd
import numpy as np

ROOTDIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPLAYDIR = f"{ROOTDIR}/replay"

def calcAngle(vec):
    v1_u = vec / np.linalg.norm(vec)
    vec_horizon = np.array([vec[0], vec[1], 0])
    v2_u = vec_horizon / np.linalg.norm(vec_horizon)
    ret = np.degrees(np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)))
    if vec[2] < 0:
        ret *= -1
    return ret

def getBallInfo(date, points=1, fps=120):
    trend_frames = fps // 12
    time_interval = 1 / 120

    input_file = 'Model3D_smooth_' + str(points) + '.csv'
    input_path = os.path.join(REPLAYDIR, date, input_file)

    df_in = pd.read_csv(input_path)

    critical_frames = df_in.index[df_in['Event'] != 0].tolist()

    ball_info = []
    for idx in range(len(critical_frames) - 1):
        start_idx = critical_frames[idx]
        trend_idx = min(start_idx + trend_frames - 1, critical_frames[idx + 1])
        end_idx = critical_frames[idx + 1]

        start_frame = np.array(df_in.iloc[start_idx][['X', 'Y', 'Z']])
        trend_frame = np.array(df_in.iloc[trend_idx][['X', 'Y', 'Z']])

        vec = trend_frame - start_frame
        dis = np.linalg.norm(vec)

        # km / h
        speed = dis / ((trend_idx - start_idx) * time_interval) * 3.6
        angle = calcAngle(vec)

        df_interval = df_in.iloc[start_idx:end_idx + 1]
        max_height = df_interval['Z'].max()

        min_at_positive = -1
        max_at_negative = -1

        try:
            min_at_positive = df_interval.loc[df_interval['Y'] > 0]['Y'].idxmin()
            max_at_negative = df_interval.loc[df_interval['Y'] < 0]['Y'].idxmax()
        except:
            pass

        mid_height = 0
        if min_at_positive != -1 and max_at_negative != -1:
            mid_height = (df_in.iloc[min_at_positive]['Z'] + df_in.iloc[max_at_negative]['Z']) / 2

        tmp_dict = dict()
        tmp_dict['Frame'] = int(df_in.iloc[start_idx]['Frame'])
        tmp_dict['Speed'] = speed
        tmp_dict['Angle'] = angle
        tmp_dict['MidHeight'] = mid_height
        tmp_dict['MaxHeight'] = max_height
        ball_info.append(tmp_dict)

    return ball_info

def writeBallInfo(csv_file, ball_info, ball_type=[]):
    if (len(ball_type) != 0 and len(ball_info) != len(ball_type)):
        print(f'len(ball_info) = {len(ball_info)} != len(ball_type) = {len(ball_type)}')
        return

    has_ball_type = True if len(ball_type) != 0 else False
    df = {}
    for i, p in enumerate(ball_info):
        df[i] = {}
        df[i]['Frame'] = ball_info[i]['Frame']
        if has_ball_type:
            df[i]['Type'] = ball_type[i]
        else:
            df[i]['Type'] = None
        df[i]['Speed'] = ball_info[i]['Speed']
        df[i]['Angle'] = ball_info[i]['Angle']
        df[i]['MidHeight'] = ball_info[i]['MidHeight']
        df[i]['MaxHeight'] = ball_info[i]['MaxHeight']
    COLUMNS = ['Frame', 'Type', 'Speed', 'Angle', 'MidHeight', 'MaxHeight']
    pd_df = pd.DataFrame.from_dict(df, orient='index', columns=COLUMNS)
    pd_df.to_csv(csv_file, encoding = 'utf-8',index = False)
