import cv2
import math
import random
import numpy as np

def sort_objects(objects):
    sorted_objects = sorted(objects, key=lambda obj: (obj[1],obj[0]))
    return sorted_objects
def sort_objects_90(objects):
    sorted_objects = sorted(objects, key=lambda obj: (obj[0]))
    return sorted_objects

def cluster_point(points):
    tolerance = 10
    classified_points = {}
    classified_points.clear()
    # ??????
    for point in points:
        x, y = point
        found = False
        for key in classified_points.keys():
            if abs(key - y) <= tolerance:
                classified_points[key].append(point)
                found = True
                break

        if not found:
            classified_points[y] = [point]
    for category, points in classified_points.items():
        classified_points[category] = sorted(points, key=lambda p: p[0])

    for category, point in classified_points.items():
        print(f"?? {category}: {point}")
    return classified_points

def cluster_button_corner(points_90):
    tolerance = 100
    classified_points = {}
    classified_points.clear()
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

def separate_points_by_side(points, mid_point):
    left_points = []
    right_points = []

    mid_x, _ = mid_point

    for point in points:
        x, _ = point

        if x < mid_x:
            left_points.append(point)
        else:
            right_points.append(point)

    return left_points, right_points

def find_perpendicular_points(mid_point, point1, point2,point_all, distance_threshold):
    x0, y0 = mid_point
    x1, y1 = point1
    x2, y2 = point2
    print(mid_point)
    print(point1)
    print(point2)
    
    if x1 != x2:
        m = (y2 - y1) / (x2 - x1)
        perp_m = -1 / m
    else:
        perp_m = float('inf')

    perpendicular_points = []
    for point in point_all:
        x, y = point
        if abs(y - (perp_m * (x - x0) + y0)) <= distance_threshold:
            perpendicular_points.append(point)

    return perpendicular_points

def find_mid(points,points_T):
    x0,y0 = points[0]
    x1,y1 = points[1]
    mid_point = ((x1 + x0)/2,(y1 + y0)/2)
    min_distance = float('inf')
    closest_point = None

    for point in points_T:
        distance = math.dist(point, mid_point)
        if distance < min_distance:
            min_distance = distance
            closest_point = point

    return closest_point

def find_mid_button(points_T,button_corner_group):
    mid_button = []
    for category, point in button_corner_group.items():
        if(len(point)==2):
            mid_button.append(find_mid(point,points_T))
        else:
            print(False)
    return  mid_button

def find_data(points_all_data,target_point,dw,dh):
    for i in points_all_data:
        label, x, y, w, h = i
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
        point = int(l+length/2),int(t+width/2)*1
        if (point == target_point):
            return i

def remove_overlapping_boxes(boxes,labels):
    num_boxes = len(boxes)
    remove_indices = []

    for i in range(num_boxes):
        for j in range(i + 1, num_boxes):
            box1 = boxes[i]
            box2 = boxes[j]

            x1, y1 = box1
            x2, y2 = box2

            distance_x = abs(x1 - x2)
            distance_y = abs(y1 - y2)

            threshold = 10  
            if distance_x < threshold and distance_y < threshold:
                remove_indices.append(j)
    updated_boxes = [box for i, box in enumerate(boxes) if i not in remove_indices]
    updated_labels = [label for i, label in enumerate(labels) if i not in remove_indices]

    return updated_boxes

def mapping(img_path,yolo_txt_path):
    img = cv2.imread(img_path)
    dh, dw, _ = img.shape
    fl = open(yolo_txt_path, 'r')
    data = fl.readlines()
    fl.close()
    points_all = []
    points_least = []
    points_90 = []
    points_T = []
    points_cross = []
    points_all_data = []
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
        points_all.append(((int(l+length/2),int(t+width/2)*1)))
        points_all_data.append(data)
        points_all = remove_overlapping_boxes(points_all,points_all_data)
        if(label==0):
            points_90.append(((int(l+length/2),int(t+width/2)*1)))
        if(label==1):
            points_T.append(((int(l+length/2),int(t+width/2)*1)))
        if(label==2):
            points_cross.append(((int(l+length/2),int(t+width/2)*1)))


    count = 0
    points_90 = sort_objects_90(points_90)
    points_T = sort_objects(points_T)
    points_cross = sort_objects(points_cross)


    visited = []
    csv_data = []

    button_corner_group = cluster_button_corner(points_90)
    for _, points in button_corner_group.items():
        #print(point)
        for point in points:
            cv2.circle(img, point,3,(255, 255, 0),3)
            cv2.putText(img, str(count), point, cv2.FONT_HERSHEY_SIMPLEX,1, (255, 255, 0), 2, cv2.LINE_AA)

            target_data = find_data(points_all_data,point,dw,dh)
            csv_data.append(target_data)
            visited.append(point)
            count+=1


    #print(button_corner_group)
    mid_buttom = find_mid_button(points_T,button_corner_group)
    print(mid_buttom)
    for point in mid_buttom:
        #print(point)
        cv2.circle(img, point,3,(255, 255, 0),3)
        cv2.putText(img, str(count), point, cv2.FONT_HERSHEY_SIMPLEX,1, (255, 255, 0), 2, cv2.LINE_AA)
        target_data = find_data(points_all_data,point,dw,dh)
        csv_data.append(target_data)
        visited.append(point)
        count+=1


    perpendicular_points = []

    for category, points in button_corner_group.items():
            if(len(points)==2):
                perpendicular_points = find_perpendicular_points(mid_buttom[0], points[0], points[1],points_all,1500)

    perpendicular_points = sort_objects(perpendicular_points)

    for point in perpendicular_points:
        if(point not in visited):
            target_data = find_data(points_all_data,point,dw,dh)
            csv_data.append(target_data)
            visited.append(point)
            count+=1

    for point in points_all:
        if(point not in visited):
            points_least.append(point)

    left_points, right_points = separate_points_by_side(points_least,mid_buttom[0])

    left_points = sort_objects(left_points)
    right_points = sort_objects(right_points)

    left_points_cluster = cluster_point(left_points)
    right_points_cluster = cluster_point(right_points)

    for _,points in left_points_cluster.items():
        for point in points:
            if(point not in visited):
                target_data = find_data(points_all_data,point,dw,dh)
                csv_data.append(target_data)
                visited.append(point)
                count+=1
    for _,points in right_points_cluster.items():
        for point in points:
            if(point not in visited):
                target_data = find_data(points_all_data,point,dw,dh)
                csv_data.append(target_data)
                visited.append(point)
                count+=1

    print(len(csv_data))
    f= open(str(yolo_txt_path[:-4])+"_mapping.txt","w+")
    for data in csv_data:
        outputdata = str(int(data[0]))+' '+str(data[1])+' '+str(data[2])+' '+str(data[3])+' '+str(data[4])+'\n'
        f.write(outputdata)
    f.close()