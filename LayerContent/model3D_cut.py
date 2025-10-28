import os
import numpy as np
import pandas as pd

DIRNAME = os.path.dirname(os.path.abspath(__file__))
ROOTDIR = os.path.dirname(DIRNAME)
REPLAYDIR = f"{ROOTDIR}/replay"

def slope(x1,x2,y1,y2):
    slope = (y2 - y1)/(x2 - x1)
    return slope

def find_slope(element):
    i = 0
    slope_arr = []
    while(i<len(element)-30):
        x1 = i
        y1 = element[i]
        x2 = i+30
        y2 = element[i+30]
        slope_tmp = slope(x1,x2,y1,y2)
        slope_arr.append(abs(slope_tmp))
        i+=30
    return slope_arr

def traversal(add_arr,element):
    index = 0
    flag = 0
    cut_timestamp = []
    while(index<len(add_arr) - 8):
        y1 = add_arr[index]
        y2 = add_arr[index+8]
        ans = y2 - y1
        if(ans < 0.1 and ans > -0.1  and flag == 0):
            flag = 1
            cut_timestamp.append(element[index*30][3])
        elif(ans > 0.5 or ans < -0.5 and flag == 1 ):
            flag = 0
        index+=8
    return cut_timestamp

def output_csv(csv,cut_timestamp,date):
    model3D = pd.read_csv(csv)
    cut_tmp = 0
    count = 1
    cut_timestamp.append(model3D.iloc[-1]['Timestamp'])
    for timestamp in cut_timestamp:
        cut = [tmp for tmp in model3D[model3D['Timestamp'] == timestamp]['Frame']]
        model = model3D[cut_tmp:cut[0]]
        file_path = os.path.join(REPLAYDIR, date, "Model3D_mod_"+ str(count)+".csv")
        model.to_csv(file_path)
        cut_tmp = cut[0]
        count+=1

def cut_csv(date):
    index = []
    element = []
    file_path = os.path.join(REPLAYDIR, date, 'Model3D_mod.csv')
    model3D = pd.read_csv(file_path)
    for i,(x,y,z,time) in enumerate(zip(model3D['X'],model3D['Y'],model3D['Z'],model3D['Timestamp'])):
        index.append(i)
        element.append((x**2,y**2,z**2,time))
    X_slope = find_slope([i[0] for i in element])
    Y_slope = find_slope([i[1] for i in element])
    Z_slope = find_slope([i[2] for i in element])
    add_arr = [X_slope[i]+Y_slope[i]+Z_slope[i] for i in range(min(len(X_slope),len(Y_slope),len(Z_slope)))]
    cut_timestamp = traversal(add_arr,element)
    output_csv(file_path,cut_timestamp,date)