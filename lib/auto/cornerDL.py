import glob
import os
import cv2
import re
import numpy as np
from tensorflow.keras.models import Sequential
from keras.layers import Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Conv2D, Flatten, Dense

def show_crop_img(img,l,t,length,width):
    l = int(l)
    t = int(t)
    length = int(length)
    width = int(width)
    roi = img[t:t+length, l:l+width]
    return roi

def get_number_from_filename(filename):
    match = re.search(r'(\d+).jpg$', filename)
    if match:
        return int(match.group(1))
    return 0

def preprocess(image_folder):
    file_list = glob.glob(os.path.join(image_folder, '*.jpg'))
    sorted_file_list = sorted(file_list, key=get_number_from_filename)
    images = []
    for img_path in sorted_file_list:
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = np.expand_dims(img, axis=-1)
        img = img / 255.0
        images.append(img)
    return images

def model_call(weight_path):
    input_img = Input(shape=(32, 32, 1))
    x = Conv2D(64, (5, 5), activation='relu', kernel_regularizer=l2(0.01))(input_img)

    out1 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(0.01))(x)
    out1 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(0.01))(out1)
    out1 = Add()([x, out1])

    out2 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(0.01))(out1)
    out2 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(0.01))(out2)
    out2 = Add()([out1, out2])

    x = Conv2D(32, (3, 3), activation='relu', kernel_regularizer=l2(0.01))(out2)
    x = Flatten()(x)
    x = Dense(128, activation='relu', kernel_regularizer=l2(0.01))(x)
    
    output = Dense(8)(x)

    model = Model(inputs=input_img, outputs=output)
    model.load_weights(weight_path)
    return model

def cornerDL(img_path,mapping_txt_path,weight_path,img_folder):
    img_origin = cv2.imread(img_path)
    dh, dw, _ = img_origin.shape
    fl = open(mapping_txt_path, 'r')
    data = fl.readlines()
    fl.close()
    points = []
    times = []
    final_points = []
    intersection_path = str(mapping_txt_path)[:-4]+"_intersection.txt"
    f= open(intersection_path,"w+")
    count_crop = 0
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

        times_x = length/32
        times_y = width/32
        times.append((times_x,times_y,l,t,label))
        #store images
        roi_image = show_crop_img(img_origin,l,t,length,width)
        roi_image = cv2.resize(roi_image, (32, 32), interpolation=cv2.INTER_NEAREST)
        if not os.path.exists(img_folder):
            os.makedirs(img_folder)
        cropped_path = os.path.join(img_folder, f"cropped_{count_crop}.jpg")
        count_crop+=1
        cv2.imwrite(cropped_path, roi_image)

    images = preprocess(img_folder)
    images = np.array(images)
    model = model_call(weight_path)
    pred = model.predict(images)
    points = []

    print(len(times))
    print(pred.shape)
    for i in pred:
        print(i)
    count = 0
    for coordinates in pred:
        point_tmp = []
        #print(coordinates)
        for i in range(0, len(coordinates), 2):
            x, y = coordinates[i], coordinates[i + 1]
            if x < 4 or y < 4:
                point = (0, 0)
            else:
                point = [int(x+0.5), int(y+0.5)]
                print(type(point[0]))
                point[0]*=times[count][0]
                point[1]*=times[count][1]
                point[0] = int(point[0] + times[count][2])
                point[1] = int(point[1] + times[count][3])
                point_tmp.append(tuple(point))
        if(times[count][4]==2): #if label = X
          if(len(point_tmp)!=4):
            point_tmp = []
        else:
          if(len(point_tmp)!=2 ):
            point_tmp = []
        points.append(point_tmp)
        count+=1
    for point in points:
        for i in point:
            f.write(str(i))
        f.write('\n')
    f.close()
    return intersection_path