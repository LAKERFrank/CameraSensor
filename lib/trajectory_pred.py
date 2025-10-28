import os 
import pandas as pd
import numpy as np
import cv2

DIRNAME = os.path.dirname(os.path.abspath(__file__))
ROOTDIR = os.path.dirname(DIRNAME)
REPLAYDIR = f"{ROOTDIR}/replay"

def last_shot(csv):
    index = []
    element = []
    file_path = os.path.join(csv)
    model3D = pd.read_csv(file_path)
    for i,(x,y,z,event,time) in enumerate(zip(model3D['X'],model3D['Y'],model3D['Z'],model3D['Event'],model3D['Timestamp'])):
        index.append(i)
        element.append((x,y,z,event,time))
        
    finalball_start = 0
    finalball_end = 0
    for i in range(len(element)):
         # event element[i][3]
        if(element[i][3]==1):
            finalball_start = i
        if(element[i][3]==3):
            finalball_end = i
            print("最後一球frame區間：",finalball_start,"to", finalball_end)
    point = []
    for i in range(finalball_start,finalball_end+1):
        point.append(np.array([element[i][0],element[i][1],element[i][2],element[i][4]]))
    if(len(point)>0):
        return point
    else:
      return 0

def kalman_filter(point):
    v = dt =  0.05
    a = 0.5 * (dt**2)
    
    kalman = cv2.KalmanFilter(9, 3, 0)
    
    kalman.measurementMatrix = np.array([
            [1, 0, 0, v, 0, 0, a, 0, 0],
            [0, 1, 0, 0, v, 0, 0, a, 0],
            [0, 0, 1, 0, 0, v, 0, 0, a]
        ],np.float32)
    
    kalman.transitionMatrix = np.array([
            [1, 0, 0, v, 0, 0, a, 0, 0],
            [0, 1, 0, 0, v, 0, 0, a, 0],
            [0, 0, 1, 0, 0, v, 0, 0, a],
            [0, 0, 0, 1, 0, 0, v, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, v, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, v],
            [0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1]
        ],np.float32)
    
    kalman.processNoiseCov = np.array([
            [1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1]
        ],np.float32) * 0.007
    zx = []
    predicted = []
    for i in range(len(point)):
        
        if( point[i][1]!= 0 ):
            zx.append(np.array([[np.float32(point[i][0])],[np.float32(point[i][1])],[np.float32(point[i][2])]]))
    
    for i in range(1000):
        
        if(i>=len(zx)):
           
            kalman.correct(np.array([[np.float32(predicted[i-1][0])],[np.float32(predicted[i-1][1])],[np.float32(predicted[i-1][2])]]))
        else:
            kalman.correct(zx[i])
        tp = kalman.predict()
        
        predicted.append((tp[0],tp[1],tp[2]))
        if(tp[2]<0):
            break
    return predicted

def output_csv(point,predicted,csv):
    file_path = os.path.join(csv)
    model3D = pd.read_csv(file_path)
    print(model3D)
    
    mask = model3D["Event"] == 3
    final_ball_frame = model3D[mask]['Frame']
    
    mask2 = model3D["Frame"] > int(final_ball_frame)
    print(model3D[mask2])
    
    
    if(model3D[mask2].empty):
        print("No data")
        #加新資料在後面
          
        
        #find the end frame and edit its Event type
        model3D.mask(model3D.iloc[1:,5:6]==3, 0, inplace=True)
        
        
        #新增資料長度
        data_length = len(predicted) - len(point)
    
        Event = [0]*(data_length-1)
        Event.append(3)
        data = pd.DataFrame({'Frame':range(int(final_ball_frame) + 1,int(final_ball_frame) + data_length + 1),           
                         'Visibility':1,
                         'X':[i[0][0] for i in predicted[len(point):]],
                         'Y':[i[1][0] for i in predicted[len(point):]],
                         'Z':[i[2][0] for i in predicted[len(point):]],
                         'Event':Event,
                         'Timestamp':0})
        data = model3D.append(data)
        print(data)
        data.to_csv(file_path, index = False)
        
    else:
        print("Yes")
        #更改後面frame的xyz
        
        model3D.mask(model3D.iloc[1:,5:6]==3, 0, inplace=True)
        
        data_length = len(predicted) - len(point)
        Event = [0]*(data_length-1)
        Event.append(3)
        
        
        
        print( data_length )
        print(len([i[0][0] for i in predicted[len(point):]]))
        data = pd.DataFrame({'Frame':range(int(final_ball_frame) + 1,int(final_ball_frame) + data_length +1 ),           
                         'Visibility':1,
                         'X':[i[0][0] for i in predicted[len(point):]],
                         'Y':[i[1][0] for i in predicted[len(point):]],
                         'Z':[i[2][0] for i in predicted[len(point):]],
                         'Event':Event,
                         'Timestamp':0})
        print(data)
        data = model3D.append(data)
        print(data)
        data.to_csv(file_path, index = False)
        
def placement_predict(date,file):

    #file_path = 'Model3D_smooth_1_1008.csv'
    file_path = os.path.join(REPLAYDIR, date, file)
    last_shot_trajectory = last_shot(file_path)
    if(last_shot_trajectory == 0 ):
        pass
    else:
        predicted = kalman_filter(last_shot_trajectory)
        output_csv(last_shot_trajectory,predicted,file_path)
