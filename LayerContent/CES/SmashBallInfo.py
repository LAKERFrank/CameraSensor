import os
import pandas as pd
import numpy as np
import sys

ROOTDIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPLAYDIR = f"{ROOTDIR}/LayerContent"

def runSmashAnalyze(date, gradient_is_success):
    result = []
    date_ = date.split('/')[-1]
    if not gradient_is_success:
        result.append({"Date": date_ , "Hit": False, "Speed": 0, "Height": 0 , 
                        "Landing_X": 0,"Landing_Y": 0,
                        "Landing_Z": 0, "Frame_cross": 0})
    else:
        points=1
        input_file = 'Model3D_event_' + str(points) + '.csv'

        # path_gradient = os.path.join(REPLAYDIR, date)
        # print(path_gradient)
        file_gradient = 'Model3D_smooth_gradient.csv'
        gradient_file_path = os.path.join(date, file_gradient)

        if os.path.exists(gradient_file_path):
            input_path = gradient_file_path
        else:
            input_path = os.path.join(date, input_file)
        
        all_trajectory_data = pd.read_csv(input_path)

        event_1_idx = all_trajectory_data.index[all_trajectory_data['Event'] == 1].tolist()
        event_3_idx = all_trajectory_data.index[all_trajectory_data['Event'] == 3].tolist()
        if len(event_1_idx) != 1 or len(event_3_idx) != 1:
            result.append({"Date": date_ , "Hit": False, "Speed": 0, "Height": 0 , 
                        "Landing_X": 0,"Landing_Y": 0,
                        "Landing_Z": 0, "Frame_cross": 0})
            return result
        
        start_idx = event_1_idx[0]
        end_idx = event_3_idx[0]
        if start_idx >= end_idx:
            result.append({"Date": date_ , "Hit": False, "Speed": 0, "Height": 0 , 
                        "Landing_X": 0,"Landing_Y": 0,
                        "Landing_Z": 0, "Frame_cross": 0})
            return result
        
        trajectory_data = all_trajectory_data.iloc[start_idx:end_idx + 1]
        trajectory_data = trajectory_data.reset_index(drop=True)

        length_frame = trajectory_data.shape[0] - 1
        flag_detect = False
        frame_cross = -1
        # for i in range(trajectory_data.shape[0]): # Analyze whether the ball passed the net backwards
        #     if trajectory_data["Y"][length_frame - i] < 0:
        #         flag_detect = True
        #     if trajectory_data["Y"][length_frame - i] > 0:
        #         if flag_detect == True:
        #             frame_cross = length_frame - i
        #             net_height = trajectory_data["Z"][frame_cross] 
        #             break
        last_y = trajectory_data["Y"][length_frame]
        if last_y < 0:
            for i in range(trajectory_data.shape[0]):
                if trajectory_data["Y"][length_frame - i] > 0:
                    frame_cross - length_frame - i
                    break
            if frame_cross == -1:
                frame_cross = 0
            net_height = trajectory_data["Z"][frame_cross]

        date_ = input_path.split('/')
        date_ = date_[-2]
        if frame_cross == -1: #not hit   
            speed = 0
            result.append({"Date": date_ , "Speed": 0, "Height": 0 , 
                            "Landing_X": 0,"Landing_Y": 0,
                            "Landing_Z": 0, "Frame_cross": 0, "Hit": False})
        else:
            trend_frames=2
            time_interval = 1 / 120

            # critical_frames = trajectory_data.index[trajectory_data['Event'] == 1]
            # print('Critical frames:', critical_frames)
        
            start_idx = trajectory_data.index[trajectory_data['Event'] == 1]
            start_idx = int(start_idx[0])
            end_idx = trajectory_data.index[trajectory_data['Event'] == 3]
            end_idx = int(end_idx[0])
            trend_idx = min(start_idx + trend_frames - 1, end_idx)

            start_frame = np.array(trajectory_data.iloc[start_idx][['X', 'Y', 'Z']])
            trend_frame = np.array(trajectory_data.iloc[trend_idx][['X', 'Y', 'Z']])

            vec = trend_frame - start_frame
            dis = np.linalg.norm(vec)

            # km / h
            speed = dis / ((trend_idx - start_idx) * time_interval) * 3.6
            time_diff = trajectory_data.iloc[trend_idx]['Timestamp'] - trajectory_data.iloc[start_idx]['Timestamp']
            speedt = dis / time_diff * 3.6

            landing_x = float(trajectory_data.iloc[end_idx]['X'])
            landing_y = float(trajectory_data.iloc[end_idx]['Y'])
            landing_z = float(trajectory_data.iloc[end_idx]['Z'])

            result.append({"Date": date_ , "Speed": speedt, "Height": net_height , 
                            "Landing_X": landing_x,"Landing_Y": landing_y,
                            "Landing_Z": landing_z, "Frame_cross": frame_cross, "Hit": True})

    return result   


def writeSmashBallInfo(date, ball_info):
    output_file = os.path.join(date, 'Model3D_smash_info.csv')
    print('Ball Info Output', output_file)
    df= {}
    for i in range(len(ball_info)):
        df[i] = {}
        df[i]['Date'] = ball_info[i]['Date']
        df[i]['Hit'] = ball_info[i]['Hit']
        df[i]['Speed'] = ball_info[i]['Speed']
        df[i]['Height'] = ball_info[i]['Height']
        df[i]['Landing_X'] = ball_info[i]['Landing_X']
        df[i]['Landing_Y'] = ball_info[i]['Landing_Y']
        df[i]['Landing_Z'] = ball_info[i]['Landing_Z']
        df[i]['Frame_cross'] = ball_info[i]['Frame_cross']
    COLUMNS = ['Date', 'Hit', 'Speed', 'Height', 'Landing_X', 'Landing_Y', 'Landing_Z', 'Frame_cross']
    pd_df = pd.DataFrame.from_dict(df, orient='index', columns=COLUMNS)
    pd_df.to_csv(output_file, encoding = 'utf-8',index = False)
