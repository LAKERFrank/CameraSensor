import cv2
import numpy as np
import math
import random
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

def show_crop_img(img,l,t,length,width):
    l = int(l)
    t = int(t)
    length = int(length)
    width = int(width)
    roi = img[t:t+length, l:l+width]
    return roi
def auto_canny(image, sigma=0.33):
        # compute the median of the single channel pixel intensities
        v = np.median(image)

        # apply automatic Canny edge detection using the computed median
        lower = int(max(0, (1.0 - sigma) * v))
        upper = int(min(255, (1.0 + sigma) * v))
        edged = cv2.Canny(image, lower, upper)

        # return the edged image
        return edged

def find_mid_point(corner_coordinates,label_indices,center_point):
    distance_min = float("+inf")
    final_index = np.random.choice(label_indices) 
    for i in label_indices:
        distance = np.linalg.norm(np.array(corner_coordinates[i]) - np.array(center_point))
        if(distance < distance_min):
            distance_min = distance
            final_index = i
    return final_index  

def mapping_points_four(points,image_height,image_width):
    corners = [(0, 0), (0, image_height), (image_width, 0), (image_width, image_height)]
    labeled_points = []

    for point in points:
        distances = [np.linalg.norm(np.array(point) - np.array(corner)) for corner in corners]
        label = np.argmin(distances)
        labeled_points.append((point, label))

    labeled_points = sorted(labeled_points, key=lambda x: x[1])
    sorted_points = [point for point, _ in labeled_points]

    return sorted_points

def mapping_points_two(point):
    x, y = point
    return math.sqrt(x**2 + y**2)

def corner(img_path,mapping_txt_path):
    img = cv2.imread(img_path)
    dh, dw, _ = img.shape
    fl = open(mapping_txt_path, 'r')
    data = fl.readlines()
    fl.close()
    points = []
    final_points = []
    intersection_path = str(mapping_txt_path)[:-4]+"_intersection.txt"
    f= open(intersection_path,"w+")
    for dt in data:


        label, x, y, w, h = map(float, dt.split(' '))
        data_label = label

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




        roi_image = show_crop_img(img,l,t,length,width)
        roi_image_origin = roi_image.copy()
        roi_image_origin_2 = roi_image.copy()

        roi_image = cv2.resize(roi_image, (length*20, width*20), interpolation=cv2.INTER_NEAREST)
        x,y,c = roi_image.shape
        roi_image_2 = roi_image.copy()

        roi_image_gray = cv2.cvtColor(roi_image, cv2.COLOR_BGR2GRAY)

        roi_image_gray = cv2.medianBlur(roi_image_gray,61)
        ret, roi_image_Bi = cv2.threshold(roi_image_gray, 120,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        kernel = np.ones((3,3), np.uint8)
        roi_image_Bi = cv2.dilate(roi_image_Bi, kernel, iterations =11)
        roi_image_Bi = cv2.erode(roi_image_Bi, kernel, iterations = 6)


        edges = auto_canny(roi_image_Bi)
        edges = cv2.dilate(edges, kernel, iterations = 10)
        dest = cv2.cornerHarris(roi_image_Bi, 50, 3, 0.08)
        corner_coordinates = cv2.dilate(dest, kernel, iterations = 2)

        #DBSCAN algo
        threshold = 0.0001 * corner_coordinates.max()
        corner_coordinates = np.argwhere(corner_coordinates > threshold)
        corner_coordinates = corner_coordinates[:, [1, 0]]
        dbscan = DBSCAN(eps=10, min_samples=10)
        if(len(corner_coordinates) == 0):
            continue
        labels = dbscan.fit_predict(corner_coordinates)

        unique_labels, label_counts = np.unique(labels, return_counts=True)
        sorted_indices = np.argsort(label_counts)[::-1]

        if(data_label==2):
            num_clusters = 4
        elif(data_label==0 or data_label == 1):
            num_clusters = 2
        #num_clusters = 4
        selected_points = []
        for idx in sorted_indices[:num_clusters]:
            label = unique_labels[idx]
            label_indices = np.where(labels == label)[0]
            random_index = np.random.choice(label_indices)
            #sample_index = find_mid_point(corner_coordinates,label_indices,(x/2,y/2))#sample
            selected_point = corner_coordinates[random_index]
            selected_points.append(selected_point)
        #rulebased
        if(len(selected_points)==2):
            selected_points = sorted(selected_points, key=mapping_points_two)

        elif(len(selected_points)==4):

            selected_points = mapping_points_four(selected_points,x,y)


        final_txt = []

        #crop
        selected_points_originsize = []
        for enlarged_point in selected_points:
            original_x = int(enlarged_point[0] / 20)
            original_y = int(enlarged_point[1] / 20)
            selected_points_originsize.append((original_x,original_y))

        scale_x = dw / width
        scale_y = dh / length
        
        for point in selected_points_originsize:
            x_original = point[0] + l
            y_original = point[1] + t

            final_points.append((x_original,y_original))
            final_txt.append((x_original,y_original))

        for point in final_txt:
            f.write(str(point))
        f.write('\n')


    f.close()
    return intersection_path