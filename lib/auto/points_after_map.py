import cv2
import pickle
import os

DIRNAME = os.path.dirname(os.path.abspath(__file__))
ROOTDIR = os.path.dirname(os.path.dirname(DIRNAME))
IMG_FOLDER = f"{ROOTDIR}/lib/auto/"+"cropped_images"  #img output

def show_crop_img(img,l,t,length,width):
    l = int(l)
    t = int(t)
    length = int(length)
    width = int(width)
    roi = img[t:t+length, l:l+width]
    return roi

def points_mapping(img_path,yolo_txt_path,court_path):
    img = cv2.imread(img_path)
    dh, dw, _ = img.shape
    fl = open(yolo_txt_path, 'r')
    data = fl.readlines()
    fl.close()
    points_all_data = []
    points = []
    points_after_mapping = []



    count = 0
    for dt in data:


        label, x, y, w, h = map(float, dt.split(' '))

        data = label, x, y, w, h
        #print(data)
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

    with open(court_path, "rb") as file:
        court_data = pickle.load(file)

    for i in range(len(court_data)):
        for j in range(len(court_data[i])):
            for k in range(len(points)):
                if(court_data[i][j] == points[k]):
                    points_after_mapping.append(points_all_data[k])



    print(len(points_after_mapping))
    print(str(yolo_txt_path[:-4])+"_after.txt")
    f= open(str(yolo_txt_path[:-4])+"_after.txt","w+")
    for data in points_after_mapping:
        outputdata = str(int(data[0]))+' '+str(data[1])+' '+str(data[2])+' '+str(data[3])+' '+str(data[4])+'\n'
        f.write(outputdata)
    f.close()