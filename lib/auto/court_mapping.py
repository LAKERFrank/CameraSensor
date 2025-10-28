import cv2 
import numpy as np
import math
import pickle

def distance_to_line(point, line_start, line_end):
    px, py = point
    ax, ay = line_start
    bx, by = line_end

    m = (by - ay) / (bx - ax)
    b = ay - m * ax

    distance = abs((m * px - py + b) / math.sqrt(m**2 + 1))
    return distance

def classify_points_by_distance(points, line_start, line_end):

    distances = [distance_to_line(point, line_start, line_end) for point in points]

    classified_points = {}
    for i, distance in enumerate(distances):
        found = False
        for key, value in classified_points.items():
            if abs(key - distance) < 20:
                classified_points[key].append(points[i])
                found = True
                break
        if not found:
            classified_points[distance] = [points[i]]

    
    for key, value in classified_points.items():
        classified_points[key] = sorted(value, key=lambda p: p[0])

    
    sorted_classifications = sorted(classified_points.items(), key=lambda x: x[0])
    top_3_classifications = sorted_classifications[:3]
    result = dict(top_3_classifications)

    return result

def cluster_button_corner(points_90):
    tolerance = 100
    classified_points = {}

    for point in points_90:
        x, y = point
        found = False

        for key in classified_points.keys():
            if abs(key - y) <= tolerance:
                classified_points[key].append(point)
                found = True
                break
        if not found:
            classified_points[y] = [point]

    last_y = points_90[-1][1]

    if len(points_90) % 2 != 0:
        classified_points[last_y] = [classified_points[last_y][-1]]

    keys_to_remove = []
    for key, value in classified_points.items():
        if len(value) == 1:
            keys_to_remove.append(key)

    for key in keys_to_remove:
        del classified_points[key]

    for category, point in classified_points.items():
        print(f"{category}: {point}")
    return classified_points

def find_mid(point1,point2):
    return ((point1[0]+point2[0])/2,(point1[1]+point2[1])/2)

def save_pickle(file_name,court_data):
    with open(file_name, "wb") as file:
        pickle.dump(court_data, file)
        print('court data saved'+file_name)

def court_mapping(img_file,txt_file,save_path):

    img = cv2.imread(img_file)
    dh, dw, _ = img.shape
    fl = open(txt_file, 'r')
    data = fl.readlines()
    fl.close()
    points_all = []
    points_least = []
    points_90 = []
    points_T = []
    points_cross = []
    points_all_data = []
    points = []


    count = 0
    for dt in data:


        label, x, y, w, h = map(float, dt.split(' '))

        data = label, x, y, w, h

        l = int((x - w / 2) * dw)
        r = int((x + w / 2) * dw)
        t = int((y - h / 2) * dh)
        b = int((y + h / 2) * dh)

        if l < 0:
            l = 0
        if r > dw - 1:
            r = dw - 1
        if t < 0:
            t = 0
        if b > dh - 1:
            b = dh - 1


        length = r - l
        width = b - t
        points.append(((int(l+length/2)*1,int(t+width/2)*1)))
        points_all_data.append(data)
        if(label==0):
            points_90.append(((int(l+length/2),int(t+width/2)*1)))
        if(label==1):
            points_T.append(((int(l+length/2),int(t+width/2)*1)))
        if(label==2):
            points_cross.append(((int(l+length/2),int(t+width/2)*1)))

    court = [[[0,0],[0,0],[0,0],[0,0],[0,0]],[[0,0],[0,0],[0,0],[0,0],[0,0]],[[0,0],[0,0],[0,0],[0,0],[0,0]]]
    button_corner_group = cluster_button_corner(points_90)

    for key,value in button_corner_group.items():
        if(len(value)==2):
            if(value[0][0] >value[1][0]):
                left_button = value[1]
                right_button = value[0]
            else:
                left_button = value[0]
                right_button = value[1]
    classified_points = classify_points_by_distance(points, left_button, right_button)

    count=0
    count_key = 0
    #90
    court[0][0] = left_button
    court[0][4] = right_button
    print(classified_points)
    mid_point = find_mid(court[0][0],court[0][4])
    for key,values in classified_points.items():
        """
        for point in values:
            if(point in points_90): #00 04
                pass
            elif(point in points_T):
                pass
            elif(point in points_cross):#11 12 13 21 23
                pass
            cv2.putText(img,str(count) , point, cv2.FONT_HERSHEY_SIMPLEX,0.5, (0, 255, 255), 1, cv2.LINE_AA)
            count+=1
        """
        print(len(values))
        print(values)
        print(mid_point)
        if(len(values) == 5):
            for i in range(len(values)):
                court[count_key][i] = values[i]
        else:
            for point in values:
                if(count_key == 0):
                    if(point in points_T):
                        print(abs((point[0]-mid_point[0])/100))
                        if( abs((point[0]-mid_point[0])/100) <=1 ):

                            court[0][2] = point
                            mid_point = point
                        else:
                            if(((point[0]-mid_point[0])/100)<-1):
                                court[0][1] = point

                            elif(((point[0]-mid_point[0])/100)>1):
                                court[0][3] = point
                if(count_key == 1):
                    if(point in points_cross):
                        if( abs((point[0]-mid_point[0])/100) <=1 ):
                            court[1][2] = point
                            mid_point = point
                        else:
                            if(((point[0]-mid_point[0])/100)<-1):
                                court[1][1] = point

                            elif(((point[0]-mid_point[0])/100)>1):
                                court[1][3] = point
                    if(point in points_T):
                        if(point[0]>mid_point[0]):
                            court[1][4] = point
                        elif(point[0]<mid_point[0]):
                            court[1][0] = point
                if(count_key == 2):
                    if(point in points_T):
                        if( abs((point[0]-mid_point[0])/100) <= 1 ):
                            court[2][2] = point
                            mid_point = point
                        else:
                            if(((point[0]-mid_point[0])/100)<-1):
                                court[2][0] = point

                            elif(((point[0]-mid_point[0])/100)>1):
                                court[2][4] = point

                    if(point in points_cross):
                        if(((point[0]-mid_point[0])/100)<-1):
                                court[2][1] = point
                        elif(((point[0]-mid_point[0])/100)>1):
                                court[2][3] = point

        count_key +=1
    print(court)
    save_pickle(save_path,court)

