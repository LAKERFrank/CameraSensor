import cv2
import mediapipe as mp
import csv
import numpy as np
import matplotlib.pyplot as plt
import sys
# 清空資料夾
import shutil
import os
import subprocess
def clearResultData():
    # 要清空的資料夾路徑
    folder_path = "./Pitcher/resultData"

    # 刪除資料夾內所有檔案和子資料夾
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f"Failed to delete {file_path}. Reason: {e}")
    print("Clean Data success")

clearResultData()



def find_mp4_files():
    file = open('./Pitcher/loc.txt', 'r')
    directory = file.read()
    mp4_files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f)) and f.endswith(".mp4")]
    if mp4_files:
        print("mp4 file:", directory + '/' + mp4_files[0])
        return directory + "/" + mp4_files[0]
    else:
        return None

# 設定輸入與輸出影片的路徑
# input_video_path = find_mp4_files()
input_video_path = "Pitcher/inputVideo/video6.mp4"
output_video_path = "./Pitcher/resultData/output.mp4"
# 設定視訊來源
cap = cv2.VideoCapture(input_video_path)
# 設定mp物件
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
# mp_holistic = mp.solutions.holistic
mp_holistic = mp.solutions.pose

# 取得影片的寬、高與 FPS
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_interval = 1.0 / fps
print("FPS:", fps, "間隔:", frame_interval)
# 設定輸出影片的編碼格式
fourcc = cv2.VideoWriter_fourcc(*"mp4v")

# 建立輸出影片的物件
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))  # fps暫時先用20
PTime = 0

# 定義軌跡的顏色和粗細
color = (0, 255, 0)  # 綠色
thickness = 2
none_num = 0 #用來看是否有偵測到人骨架
# 投球的軌跡
hand_position = []
# 開始進行視訊捕捉與影像分析
with mp_holistic.Pose(
    min_detection_confidence=0.5,
        min_tracking_confidence=0.6) as holistic:
    if not cap.isOpened():
        print("Cannot open camera")
        exit()
    # 建立 CSV 檔案的物件
    # current_dir = os.getcwd()
    # print(current_dir)
    with open('./Pitcher/resultData/keypoint.csv','w',newline= '') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['frame','Keypoint id','x','y','z'])
        with open('./Pitcher/resultData/keypoint_3D.csv','w',newline='') as csvfile_3D:
            writer_3D = csv.writer(csvfile_3D)
            writer_3D.writerow(['frame', 'Keypoint id', 'x', 'y', 'z'])
            while True:
            # 讀取每一幀的影像
                ret, img = cap.read()
                if not ret:
                    print("Cannot receive frame")
                    break
            # 轉換色彩空間並進行媒體流程的處理
            #     img.flags.writeable = False
            #     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                # img.flags.writeable = True
                # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                results = holistic.process(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

                keypoints = []  # 儲存人體關鍵點座標資訊
                keypoints_3D = []  # 儲存關鍵點3D座標
            # 如果有偵測到人體關鍵點
                if results.pose_world_landmarks is not None:
                    for landmark in results.pose_world_landmarks.landmark:
                        keypoints_3D.append([int(cap.get(cv2.CAP_PROP_POS_FRAMES)),
                                          landmark.visibility,
                                         landmark.x ,
                                         landmark.y ,
                                         landmark.z ])
                    for i, landmark in enumerate(keypoints_3D):
                        writer_3D.writerow([landmark[0], i, landmark[2], landmark[3], landmark[4]])
                if results.pose_world_landmarks is None:
                    none_num += 1
                    if none_num > 5:
                        with open('./Pitcher/resultData/log.txt', 'w', newline='') as logfile:
                            logfile.write("4")
                        print("ERROR :未偵測到人體")
                        sys.exit(1)

            # 如果有偵測到人體關鍵點
                if results.pose_landmarks is not None:
                    for landmark in results.pose_landmarks.landmark:
                        h, w, c = img.shape
                        cx, cy = int(landmark.x * w), int(landmark.y * h)
                        keypoints.append([int(cap.get(cv2.CAP_PROP_POS_FRAMES)),landmark.visibility, cx, cy, landmark.z])
                        # cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)

                    for i,landmark in enumerate(keypoints):
                        if i == 16: # 右手拿球
                            hand_position.append([landmark[2], landmark[3]])
                        #如果第三个参数为False，您将获得连接所有点的折线，而不是闭合形状。
                            cv2.polylines(img, [np.array(hand_position)], False, color, thickness)
                        if i not in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 17, 18, 19, 20, 21, 22]:
                            writer.writerow([landmark[0], i, landmark[2], landmark[3], landmark[4]])

                # 繪製關鍵點座標資訊到影像上
                mp_drawing.draw_landmarks(img, results.pose_landmarks,
                                      mp_holistic.POSE_CONNECTIONS,
                                      landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

                cv2.putText(img, str(int(cap.get(cv2.CAP_PROP_POS_FRAMES))), (60, 60), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 3, cv2.LINE_AA)
                out.write(img)

cap.release()
out.release()
cv2.destroyAllWindows()
cap = cv2.VideoCapture(output_video_path)
peak_lag = h
peak_lag_frame = 0

foot_27 = 0
foot_28 = 0
foot_plant = 0
foot_plant_frame = 0

max_shoulder_external_rotation_frame = 0
max_shoulder_external_rotation = 0

ball_release = 0
ball_release_frame = 0
wrist_16 = np.array([0,0])
last_wrist_location = np.array([0,0])
base = np.array([0, 0])
max_acc = 0
if not cap.isOpened():
    print("Cannot open camera")
    exit()
with open('./Pitcher/resultData/keypoint.csv', 'r', newline='') as csvfile:
    reader = csv.reader(csvfile)
    for i, row in enumerate(reader):
        if i > 1:
            if int(row[1]) == 25:  # 找出抬腿姿勢的frame
                if peak_lag > float(row[3]):
                    peak_lag = float(row[3])
                    peak_lag_frame = int(float(row[0]))
            if int(row[1]) == 27:
                foot_27 = float(row[2])
            if int(row[1]) == 28:
                foot_28 = float(row[2])
            if foot_plant < foot_27 - foot_28 and int(row[1]) == 30:
                foot_plant = foot_27 - foot_28
                foot_plant_frame = int(float(row[0])) #find foot_plant frame
                foot_27 = 0
                foot_28 = 0
            if int(row[1]) == 16:
                last_wrist_location = wrist_16
                wrist_16 = np.array([float(row[2]),float(row[3])])
            if int(row[1]) == 11:
                base = np.array([float(row[2]),float(row[3])])
            if max_acc < np.linalg.norm(wrist_16 - last_wrist_location) and int(row[1]) == 30 and int(row[0]) > peak_lag_frame + 2:
                if wrist_16[0] > base[0] and wrist_16[1] < base[1]:
                    max_acc = np.linalg.norm(wrist_16 - last_wrist_location)
                    ball_release_frame = int(float(row[0])) + 1
                    print(int(row[0]),"&",wrist_16[0],"&",base)
with open('./Pitcher/resultData/keypoint_3D.csv', 'r', newline='') as csvfile3d:
    reader = csv.reader(csvfile3d)
    next(reader)  # 跳過第一行
    stand = np.array([0,0,0])

    shoulder = np.array([0,0,0])
    elbow = np.array([0,0,0])
    wrist = np.array([0,0,0])
    hip = np.array([0,0,0])
    frame = 0
    # # Elbow Flexion
    frame_list = []
    elbow_flexion_angle_list = []
    for a, row in enumerate(reader):
        if int(row[1]) == 16:
            wrist = np.array([float(row[2]) * 100, -float(row[3]) * 100, -float(row[4]) * 100])
        if int(row[1]) == 12:
            shoulder = np.array([float(row[2]) * 100, -float(row[3]) * 100, -float(row[4]) * 100])
        if int(row[1]) == 14:
            elbow = np.array([float(row[2]) * 100, -float(row[3]) * 100, -float(row[4]) * 100])
        if not np.array_equal(wrist, stand) and not np.array_equal(elbow, stand) and not np.array_equal(shoulder,stand):
            # 計算肩膀和手肘的向量，手肘到手腕的向量
            upper_arm = elbow - shoulder
            forearm = wrist - elbow
            # 計算這兩個向量的內積
            dot_product = np.dot(upper_arm, forearm)
            # 計算這兩個向量的長度
            upper_arm_length = np.linalg.norm(upper_arm)
            forearm_length = np.linalg.norm(forearm)
            # 計算這兩個向量間的cos𝜃
            cosine_angle = dot_product / (upper_arm_length * forearm_length)
            # 利用反餘弦，求出𝜃
            angle = np.arccos(cosine_angle)
            # 將angle轉乘degree
            angle_degree = np.degrees(angle)
            flexion_degree = angle_degree
            frame_list.append(int(row[0]))
            elbow_flexion_angle_list.append(flexion_degree)
            # print(int(row[0]),"Elbow Flexsion:",flexion_degree)
            shoulder = np.array([0, 0, 0])
            elbow = np.array([0, 0, 0])
            wrist = np.array([0, 0, 0])
            hip = np.array([0, 0, 0])

    # shoulder abduction
    shoulder_abduction_angle_list = []
    csvfile3d.seek(0)
    next(reader)
    for k, row in enumerate(reader):
        if int(row[1]) == 24:
            hip = np.array([float(row[2]) * 100, -float(row[3]) * 100, -float(row[4]) * 100])
        if int(row[1]) == 12:
            shoulder = np.array([float(row[2]) * 100, -float(row[3]) * 100, -float(row[4]) * 100])
        if int(row[1]) == 14:
            elbow = np.array([float(row[2]) * 100, -float(row[3]) * 100, -float(row[4]) * 100])
        if int(row[1]) == 11:
            left_shoulder = np.array([float(row[2]) * 100, -float(row[3]) * 100, -float(row[4]) * 100])
        if not np.array_equal(hip, stand) and not np.array_equal(elbow, stand) and not np.array_equal(shoulder,stand):
            # 計算向量
            shoulder_vector = elbow - shoulder
            mid_shoulder = (shoulder + left_shoulder)/ 2
            mid_vector = -mid_shoulder
            # 計算內積
            dot_product = np.dot(shoulder_vector, mid_vector)
            # 計算兩向量長度
            shoulder_vector_length = np.linalg.norm(shoulder_vector)
            hip_vector_length = np.linalg.norm(mid_vector)
            # 計算cos theta
            cosine_angle = dot_product / (shoulder_vector_length * hip_vector_length)
            # 算出弦度，在轉換成角度
            angle = np.arccos(cosine_angle)
            angle_degrees = np.degrees(angle)

            abduction_angle = angle_degrees
            shoulder_abduction_angle_list.append(abduction_angle)
            # print(int(row[0]), "abduction_angle:", abduction_angle)
            shoulder = np.array([0, 0, 0])
            elbow = np.array([0, 0, 0])
            wrist = np.array([0, 0, 0])
            hip = np.array([0, 0, 0])
    # shoulder horizontal abduction
    shoulder_horizontal_abduction_angle_list = []
    csvfile3d.seek(0)
    next(reader)
    left_shoulder = np.array([0,0,0])
    for l, row in enumerate(reader):
        if int(row[1]) == 11:
            left_shoulder = np.array([float(row[2]) * 100, -float(row[3]) * 100, -float(row[4]) * 100])
        if int(row[1]) == 12:
            shoulder = np.array([float(row[2]) * 100, -float(row[3]) * 100, -float(row[4]) * 100])
        if int(row[1]) == 14:
            elbow = np.array([float(row[2]) * 100, -float(row[3]) * 100, -float(row[4]) * 100])
        if not np.array_equal(left_shoulder, stand) and not np.array_equal(elbow, stand) and not np.array_equal(shoulder, stand):
            # 計算左肩往右肩的向量
            # shoulder_vec = [left_shoulder[0] - shoulder[0], 0, left_shoulder[2] - shoulder[2]]
            shoulder_vec = [shoulder[0] - left_shoulder[0], 0, shoulder[2] - left_shoulder[2]]
            # 計算右肩往右肘的向量
            arm_vec = [elbow[0] - shoulder[0], 0, elbow[2] - shoulder[2]]
            # 做內積
            dot_product = np.dot(shoulder_vec, arm_vec)
            # 算出 向量長度
            shoulder_vec_length = np.linalg.norm(shoulder_vec)
            arm_vec_length = np.linalg.norm(arm_vec)
            # 計算cos theta
            cosine_angle = dot_product / (shoulder_vec_length * arm_vec_length)
            # 算出弦度，再轉角度
            angle = np.arccos(cosine_angle)
            angle_degrees = np.degrees(angle)
            # 將兩向量做外積，然後根據y軸的正負值來判斷 角度的正負值
            PorN = np.cross(shoulder_vec , arm_vec)[1]  # np.cross(x,y) 相反的話正負號，會改變
            if PorN > 0 :
                angle_degrees *= -1
            shoulder_horizontal_abduction_angle_list.append(angle_degrees)
            # print(int(row[0]), "shoulder_horizontal_abduction:", angle_degrees)

            shoulder = np.array([0, 0, 0])
            elbow = np.array([0, 0, 0])
            left_shoulder = np.array([0, 0, 0])

    csvfile3d.seek(0)
    next(reader)
    # Shoulder External Rotation
    shoulder_external_rotation_angle_list = []
    for i, row in enumerate(reader):
        if int(row[1]) == 16:
            wrist = np.array([float(row[2]) * 100, -float(row[3]) * 100, -float(row[4]) * 100])
        if int(row[1]) == 12:
            shoulder = np.array([float(row[2]) * 100, -float(row[3]) * 100, -float(row[4]) * 100])
        if int(row[1]) == 14:
            elbow = np.array([float(row[2]) * 100, -float(row[3]) * 100, -float(row[4]) * 100])
        if int(row[1]) == 24:
            hip = np.array([float(row[2]) * 100, -float(row[3]) * 100, -float(row[4]) * 100])
        if int(row[1]) == 11:
            left_shoulder = np.array([float(row[2]) * 100, -float(row[3]) * 100, -float(row[4]) * 100])

        if not np.array_equal(wrist, stand) and not np.array_equal(elbow, stand) and not np.array_equal(shoulder, stand) and int(row[1]) == 30:
            # 計算向量
            shoulder_vector = elbow - shoulder
            mid_vector = hip - shoulder
            # mid_shoulder = (shoulder + left_shoulder) / 2
            # mid_vector = -mid_shoulder
            # 計算內積
            dot_product = np.dot(shoulder_vector, mid_vector)
            # 計算兩向量長度
            shoulder_vector_length = np.linalg.norm(shoulder_vector)
            hip_vector_length = np.linalg.norm(mid_vector)
            # 計算cos theta
            cosine_angle = dot_product / (shoulder_vector_length * hip_vector_length)
            # 算出弦度，在轉換成角度
            angle = np.arccos(cosine_angle)
            vertical_abduction_angle_degrees = np.degrees(angle)


            # print("frame:", int(row[0]),"angle:",vertical_abduction_angle_degrees)
            if vertical_abduction_angle_degrees != 90:
                rotation_axis = -np.cross(mid_vector, shoulder_vector)
                rotation_axis /= np.linalg.norm(rotation_axis)
                # print(rotation_axis)
                rotation_angle = (angle - np.pi/2.0)
                # print(np.degrees(rotation_angle))
                # rotation_angle = angle - 90
                rotation_matrix = np.array([[np.cos(rotation_angle) + rotation_axis[0] ** 2 * (1 - np.cos(rotation_angle)),
                                     rotation_axis[0] * rotation_axis[1] * (1 - np.cos(rotation_angle)) - rotation_axis[
                                         2] * np.sin(rotation_angle),
                                     rotation_axis[0] * rotation_axis[2] * (1 - np.cos(rotation_angle)) + rotation_axis[
                                         1] * np.sin(rotation_angle)],
                                    [rotation_axis[1] * rotation_axis[0] * (1 - np.cos(rotation_angle)) + rotation_axis[
                                        2] * np.sin(rotation_angle),
                                     np.cos(rotation_angle) + rotation_axis[1] ** 2 * (1 - np.cos(rotation_angle)),
                                     rotation_axis[1] * rotation_axis[2] * (1 - np.cos(rotation_angle)) - rotation_axis[
                                         0] * np.sin(rotation_angle)],
                                    [rotation_axis[2] * rotation_axis[0] * (1 - np.cos(rotation_angle)) - rotation_axis[
                                        1] * np.sin(rotation_angle),
                                     rotation_axis[2] * rotation_axis[1] * (1 - np.cos(rotation_angle)) + rotation_axis[
                                         0] * np.sin(rotation_angle),
                                     np.cos(rotation_angle) + rotation_axis[2] ** 2 * (1 - np.cos(rotation_angle))]])
                new_right_elbow = shoulder + np.matmul(rotation_matrix, shoulder_vector)

                wrist_vector = wrist - shoulder
                new_right_wrist = shoulder + np.matmul(rotation_matrix, wrist_vector)
                # print("frame elbow:", int(row[0]), elbow, ",new:", new_right_elbow)
                # print("frame wrist:", int(row[0]), wrist,",new:",new_right_wrist)

                shoulder_vector_new = new_right_elbow - shoulder
                # 計算內積
                dot_product_new = np.dot(shoulder_vector_new, mid_vector)
                # 計算兩向量長度
                shoulder_vector_length_new = np.linalg.norm(shoulder_vector_new)
                hip_vector_length_new = np.linalg.norm(mid_vector)
                # 計算cos theta
                cosine_angle_new = dot_product_new / (shoulder_vector_length_new * hip_vector_length_new)
                angle_new = np.arccos(cosine_angle_new)
                # print("frame:", int(row[0]), "new angle:", np.degrees(angle_new))

        ####
            elbow_wrist_vec = new_right_wrist - new_right_elbow
            elbow_shoulder_vec = new_right_elbow - shoulder

            elbow_wrist_vec_pure = wrist - elbow
            elbow_shoulder_vec_pure = elbow - shoulder

            shoulder_elbow_vector = new_right_elbow - shoulder
            shoulder_hip_vector = hip - shoulder
            # 往背前的法向量
            plane_a_vector = -np.cross(elbow_shoulder_vec, elbow_wrist_vec)
            # plane_a_vector_pure = -np.cross(elbow_shoulder_vec_pure, elbow_wrist_vec_pure)
            # print(int(row[0]), "frame a", plane_a_vector/np.linalg.norm(plane_a_vector))
            # 往胸前的法向量
            plane_b_vector = np.cross(shoulder_elbow_vector, shoulder_hip_vector)
            # print(int(row[0]), "frame b", plane_b_vector/np.linalg.norm(plane_b_vector))
            dot_product = np.dot(plane_a_vector, plane_b_vector)
            plane_a_length = np.linalg.norm(plane_a_vector)
            plane_b_length = np.linalg.norm(plane_b_vector)
            cosine_angle = dot_product / (plane_a_length * plane_b_length)
            angle = np.arccos(cosine_angle)
            angle_degree = np.degrees(angle)
            # print("frame pure:", int(row[0]), "=", angle_degree, "Y：", plane_a_vector[1], "pure Y:", plane_a_vector_pure[1])
            if plane_a_vector[1] < 0:
                angle_degree = -angle_degree
            if angle_degree + 90 < 180:
                angle_degree += 90
            # print("frame pure1:", int(row[0]), "=", angle_degree)
            # if wrist[1] > elbow[1]:
            #     angle_degree = 180 - angle_degree
            #     angle_degree += 90
            # else:
            #     angle_degree = angle_degree - 90

            # print("frame:", int(row[0]), "=", angle_degree)
            if max_shoulder_external_rotation < angle_degree and wrist[1] > shoulder[1] \
                    and int(row[0]) > foot_plant_frame:
                max_shoulder_external_rotation = angle_degree
                max_shoulder_external_rotation_frame = int(row[0])
            shoulder_external_rotation_angle_list.append(angle_degree)
            # if int(row[0]) > max_shoulder_external_rotation_frame and shoulder[1] > wrist[1] and wrist[0] > shoulder[0]\
            #          and shoulder[0] > left_shoulder[0] and ball_release_frame == 0:
            #     ball_release_frame = int(row[0])

            shoulder = np.array([0, 0, 0])
            elbow = np.array([0, 0, 0])
            wrist = np.array([0, 0, 0])

    csvfile3d.seek(0)
    next(reader)
    # Pelvis Rotation
    right_hip = np.array([0, 0, 0])
    left_hip = np.array([0, 0, 0])

    pelvis_rotation_angle_list = []
    for i, row in enumerate(reader):
        if int(row[1]) == 23:
            left_hip = np.array([float(row[2]) * 100, -float(row[3]) * 100, -float(row[4]) * 100])
        if int(row[1]) == 24:
            right_hip = np.array([float(row[2]) * 100, -float(row[3]) * 100, -float(row[4]) * 100])
        if int(row[1]) == 30:
            hip_vector = [right_hip[0] - left_hip[0], 0, right_hip[2]-left_hip[2]]
            ps_x_vector = [1, 0, 0]
            # 取兩向量的夾角，正Ｘ軸的方向 以及 原點往right_hip的方向 的夾角。
            dot_product = np.dot(hip_vector, ps_x_vector)
            right_hip_vector_length = np.linalg.norm(hip_vector)
            ps_x_vector_length = np.linalg.norm(ps_x_vector)
            cosine_angle = dot_product/right_hip_vector_length * ps_x_vector_length
            angle = np.arccos(cosine_angle)
            angle_degree = np.degrees(angle)
            pelvis_rotation_angle_list.append(angle_degree)
            # print(int(row[0]), "Pelvis Rotation = ", angle_degree)

    csvfile3d.seek(0)
    next(reader)
    #Torso Rotation
    Torso_rotation_angle_list = []
    for i, row in enumerate(reader):
        if int(row[1]) == 11:
            left_shoulder = np.array([float(row[2]) * 100, -float(row[3]) * 100, -float(row[4]) * 100])
        if int(row[1]) == 12:
            shoulder = np.array([float(row[2]) * 100, -float(row[3]) * 100, -float(row[4]) * 100])
        if int(row[1]) == 30:
            # right_shoulder_vector = [shoulder[0], 0, shoulder[2]]
            # left_shoulder_vector = [left_shoulder[0], 0, left_shoulder[2]]
            shoulder_vector = [shoulder[0] - left_shoulder[0], 0, shoulder[2] - left_shoulder[2]]
            ps_x_vector = [1, 0, 0]
            # 取兩向量的夾角，正Ｘ軸的方向 以及 原點往right_hip的方向 的夾角。
            dot_product = np.dot(shoulder_vector, ps_x_vector)
            right_shoulder_vector_length = np.linalg.norm(shoulder_vector)
            ps_x_vector_length = np.linalg.norm(ps_x_vector)
            cosine_angle = dot_product / right_shoulder_vector_length * ps_x_vector_length
            angle = np.arccos(cosine_angle)
            angle_degree = np.degrees(angle)
            Torso_rotation_angle_list.append(angle_degree)
            # print(int(row[0]), "Torso Rotation = ", angle_degree)
    # 將各角度存入csv中

def catch_keyframeimage():

    while True:
        ret, img= cap.read()
        if not ret:
            print("Cannot receive frame")
            break
        img.flags.writeable = False
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img.flags.writeable = True
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        if cap.get(cv2.CAP_PROP_POS_FRAMES) == peak_lag_frame:
        # cv2.circle(img, (cx, cy), 5, (0, 255, 0), cv2.FILLED)
            cv2.imwrite('./Pitcher/resultData/peak_lag.jpg', img)
            print("Peak_lag success")
        # break
        if cap.get(cv2.CAP_PROP_POS_FRAMES) == foot_plant_frame:
            cv2.imwrite('./Pitcher/resultData/foot_plant.jpg', img)
            print("foot_plant success")
        if cap.get(cv2.CAP_PROP_POS_FRAMES) == max_shoulder_external_rotation_frame:
            cv2.imwrite('./Pitcher/resultData/max_shoulder_external_rotation.jpg', img)
            print("max_shoulder_external_rotation success")

        if cap.get(cv2.CAP_PROP_POS_FRAMES) == ball_release_frame:
            cv2.imwrite('./Pitcher/resultData/Ball_release.jpg', img)
            print("Ball_release success")

    cap.release()

def check_keyframe():
    if max_shoulder_external_rotation_frame > ball_release_frame:
        with open('./Pitcher/resultData/log.txt', 'w', newline='') as logfile:
            logfile.write("2")
        print("ERROR : MER>BR")
        catch_keyframeimage()
        sys.exit(1)
    if ball_release_frame - foot_plant_frame < 5 :
        with open('./Pitcher/resultData/log.txt', 'w', newline='') as logfile:
            logfile.write("1")
        print("ERROR : BR - FP < 5")
        catch_keyframeimage()
        sys.exit(1)
    if ball_release_frame < foot_plant_frame:
        with open('./Pitcher/resultData/log.txt', 'w', newline='') as logfile:
            logfile.write("3")
        print("ERROR : FP > BR")
        catch_keyframeimage()
        sys.exit(1)


check_keyframe()


with open('./Pitcher/resultData/Angle_list.csv', 'w', newline='') as angle_list:
    angle_list = csv.writer(angle_list)
    angle_list.writerow(['frame', 'Elbow Flexion', 'shoulder External Rotation', 'shoulder Abduction', 'shoulder Horizontal Abduction',
                         "Pelvis Rotation", "Torso Rotation"])
    main_frame_list = []
    main_elbow_flexion_angle_list = []
    main_shoulder_abduction_angle_list = []
    main_shoulder_external_rotation_angle_list = []
    main_shoulder_horizontal_abduction_angle_list = []
    main_pelvis_angle_list = []
    main_torso_angle_list = []
    main_pelvis_rotation_angle_list = []
    main_torso_rotation_angle_list = []
    main_elbow_extension_angle_list = []
    main_shoulder_internal_rotation_angle_list = []
    for i in range(len(frame_list)):
        angle_list.writerow([frame_list[i], elbow_flexion_angle_list[i], shoulder_external_rotation_angle_list[i],
                             shoulder_abduction_angle_list[i], shoulder_horizontal_abduction_angle_list[i],
                            pelvis_rotation_angle_list[i], Torso_rotation_angle_list[i]])
        if ball_release_frame + 1 >= frame_list[i] >= foot_plant_frame - 1:
            # angle_list.writerow([frame_list[i],elbow_flexion_angle_list[i],shoulder_external_rotation_angle_list[i],
            #                  shoulder_abduction_angle_list[i],shoulder_horizontal_abduction_angle_list[i]])
            main_frame_list.append(frame_list[i])
            main_elbow_flexion_angle_list.append(elbow_flexion_angle_list[i])
            main_shoulder_abduction_angle_list.append(shoulder_abduction_angle_list[i])
            main_shoulder_external_rotation_angle_list.append(shoulder_external_rotation_angle_list[i])
            main_shoulder_horizontal_abduction_angle_list.append(shoulder_horizontal_abduction_angle_list[i])
            main_pelvis_angle_list.append(pelvis_rotation_angle_list[i])
            main_torso_angle_list.append(Torso_rotation_angle_list[i])

        if ball_release_frame + 1 >= frame_list[i] >= foot_plant_frame - 1:
            main_pelvis_rotation_angle_list.append((pelvis_rotation_angle_list[i-1] - pelvis_rotation_angle_list[i])/ frame_interval)
            main_torso_rotation_angle_list.append((Torso_rotation_angle_list[i-1] - Torso_rotation_angle_list[i]) / frame_interval)
            main_elbow_extension_angle_list.append((elbow_flexion_angle_list[i] - elbow_flexion_angle_list[i-1])/ frame_interval)
            main_shoulder_internal_rotation_angle_list.append((shoulder_external_rotation_angle_list[i-1]- shoulder_external_rotation_angle_list[i])/frame_interval)



# intercept = np.polyfit(main_frame_list, main_elbow_flexion_angle_list, ball_release_frame - foot_plant_frame)
# y_fit = np.polyval(intercept,main_frame_list)
# plt.plot(main_frame_list,y_fit,label="elbow_flexion_s")
from scipy.interpolate import make_interp_spline
fig1 = plt.figure(figsize=(12, 8))
ax1 = fig1.add_subplot(111)

x_new = np.linspace(min(main_frame_list), max(main_frame_list), 200)
y_smooth = make_interp_spline(main_frame_list, main_elbow_flexion_angle_list)(x_new)
ax1.plot(x_new, y_smooth, label="elbow_flexion")
y_smooth = make_interp_spline(main_frame_list, main_shoulder_abduction_angle_list)(x_new)
ax1.plot(x_new, y_smooth, label="shoulder_abduction")
y_smooth = make_interp_spline(main_frame_list, main_shoulder_external_rotation_angle_list)(x_new)
ax1.plot(x_new, y_smooth, label="shoulder_rotation")
y_smooth = make_interp_spline(main_frame_list, main_shoulder_horizontal_abduction_angle_list)(x_new)
ax1.plot(x_new, y_smooth, label="shoulder_horizontal_abduction")

# ax1.plot(main_frame_list,main_elbow_flexion_angle_list,label="elbow_flexion")
# ax1.plot(main_frame_list,main_shoulder_abduction_angle_list,label = "shoulder_abduction")
# ax1.plot(main_frame_list,main_shoulder_external_rotation_angle_list,label = "shoulder_rotation")
# ax1.plot(main_frame_list,main_shoulder_horizontal_abduction_angle_list,label = "shoulder_horizontal_abduction")

total1 = main_elbow_flexion_angle_list + main_shoulder_abduction_angle_list + main_shoulder_external_rotation_angle_list\
        + main_shoulder_horizontal_abduction_angle_list

ax1.axvline(x= foot_plant_frame,color = "red")
ax1.annotate("FP",xy=(foot_plant_frame,0),xytext=(foot_plant_frame -0.1,max(total1)+ 30))
ax1.axvline(x= max_shoulder_external_rotation_frame,color = "yellow")
ax1.annotate("MER",xy=(max_shoulder_external_rotation_frame,0),xytext=(max_shoulder_external_rotation_frame -0.2,max(total1)+ 30))
ax1.axvline(x= ball_release_frame,color = "black")
ax1.annotate("BR",xy=(ball_release_frame,0),xytext=(ball_release_frame -0.1,max(total1)+ 30))
ax1.legend()
x_pos = [foot_plant_frame, (foot_plant_frame + ball_release_frame)/2, ball_release_frame]
x_label = ['0%', '50%', '100%']
plt.xticks(x_pos, x_label)
plt.xlim(foot_plant_frame -1, ball_release_frame + 1)


plt.ylim(min(total1) - 30, max(total1) + 30)
plt.savefig("./Pitcher/resultData/line_Chart.jpg")
# plt.show()

fig2 = plt.figure(figsize=(12,8))
ax2 = fig2.add_subplot(111)

x_new = np.linspace(min(main_frame_list),max(main_frame_list), 200)

y_smooth = make_interp_spline(main_frame_list, main_pelvis_rotation_angle_list)(x_new)
ax2.plot(x_new, y_smooth, color="purple", label="pelvis_rotation")
ax2.annotate("", xy=(x_new[np.argmax(y_smooth)], max(y_smooth)), xytext=(x_new[np.argmax(y_smooth)], max(y_smooth) + 100), arrowprops=dict(facecolor='purple', shrink=0.05),)

y_smooth = make_interp_spline(main_frame_list, main_torso_rotation_angle_list)(x_new)
ax2.plot(x_new, y_smooth, color="red", label="torso_rotation")
ax2.annotate("", xy=(x_new[np.argmax(y_smooth)], max(y_smooth)), xytext=(x_new[np.argmax(y_smooth)], max(y_smooth) + 100), arrowprops=dict(facecolor='red', shrink=0.05),)

y_smooth = make_interp_spline(main_frame_list,main_elbow_extension_angle_list)(x_new)
ax2.plot(x_new,y_smooth, color="green", label="elbow_extension")
ax2.annotate("", xy=(x_new[np.argmax(y_smooth)], max(y_smooth)), xytext=(x_new[np.argmax(y_smooth)], max(y_smooth) + 100), arrowprops=dict(facecolor='green', shrink=0.05),)

y_smooth = make_interp_spline(main_frame_list,main_shoulder_internal_rotation_angle_list)(x_new)
ax2.plot(x_new, y_smooth,color="blue", label="shoulder_internal_rotation")
ax2.annotate("", xy=(x_new[np.argmax(y_smooth)], max(y_smooth)), xytext=(x_new[np.argmax(y_smooth)], max(y_smooth) + 100), arrowprops=dict(facecolor='blue', shrink=0.05),)

total2 = main_pelvis_rotation_angle_list + main_torso_rotation_angle_list + main_elbow_extension_angle_list + main_shoulder_internal_rotation_angle_list

x_pos = [foot_plant_frame, (foot_plant_frame + ball_release_frame)/2, ball_release_frame]
x_label = ['0%', '50%', '100%']
plt.xticks(x_pos, x_label)
plt.xlim(foot_plant_frame - 1, ball_release_frame + 1)
plt.ylim(min(total2) - 500, max(total2) + 500)


ax2.axvline(x= foot_plant_frame,color = "red")
ax2.annotate("FP",xy=(foot_plant_frame,0),xytext=(foot_plant_frame -0.1,max(total2)+ 500))
ax2.axvline(x= max_shoulder_external_rotation_frame,color = "yellow")
ax2.annotate("MER",xy=(max_shoulder_external_rotation_frame,0),xytext=(max_shoulder_external_rotation_frame -0.2,max(total2)+ 500))
ax2.axvline(x= ball_release_frame,color = "black")
ax2.annotate("BR",xy=(ball_release_frame,0),xytext=(ball_release_frame -0.1,max(total2)+ 500))


ax2.legend()
plt.savefig("./Pitcher/resultData/Kinematic_Sequence_Chart.jpg")
# plt.show()

fig3 = plt.figure(figsize=(12,8))
ax3 = fig3.add_subplot(111)
x_new = np.linspace(min(main_frame_list),max(main_frame_list), 200)
y_smooth = make_interp_spline(main_frame_list, main_pelvis_angle_list)(x_new)
ax3.plot(x_new, y_smooth, color="purple", label="pelvis_rotation_degree")

y_smooth = make_interp_spline(main_frame_list, main_torso_angle_list)(x_new)
ax3.plot(x_new, y_smooth, color="red", label="torso_rotation_degree")

y_smooth = make_interp_spline(main_frame_list, main_elbow_flexion_angle_list)(x_new)
ax3.plot(x_new, y_smooth, color="green", label="elbow_flexion_degree")

y_smooth = make_interp_spline(main_frame_list, main_shoulder_external_rotation_angle_list)(x_new)
ax3.plot(x_new, y_smooth, color="blue", label="shoulder_external_degree")

total3 = main_pelvis_angle_list + main_torso_angle_list + main_shoulder_external_rotation_angle_list + main_elbow_flexion_angle_list

x_pos = [foot_plant_frame, (foot_plant_frame + ball_release_frame)/2, ball_release_frame]
x_label = ['0%', '50%', '100%']
plt.xticks(x_pos, x_label)
plt.xlim(foot_plant_frame - 1, ball_release_frame + 1)
plt.ylim(min(total3) - 50, max(total3) + 50)

ax3.axvline(x= foot_plant_frame,color = "red")
ax3.annotate("FP",xy=(foot_plant_frame,0),xytext=(foot_plant_frame -0.1,max(total3)+ 50))
ax3.axvline(x= max_shoulder_external_rotation_frame,color = "yellow")
ax3.annotate("MER",xy=(max_shoulder_external_rotation_frame,0),xytext=(max_shoulder_external_rotation_frame -0.2,max(total3)+ 50))
ax3.axvline(x= ball_release_frame,color = "black")
ax3.annotate("BR",xy=(ball_release_frame,0),xytext=(ball_release_frame -0.1,max(total3)+ 50))

ax3.legend()
# plt.show()
plt.savefig("./Pitcher/resultData/Kinematic_degree_Chart.jpg")

# while True:
#     ret, img= cap.read()
#     if not ret:
#         print("Cannot receive frame")
#         break
#     img.flags.writeable = False
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     img.flags.writeable = True
#     img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
#
#     if cap.get(cv2.CAP_PROP_POS_FRAMES) == peak_lag_frame:
#         # cv2.circle(img, (cx, cy), 5, (0, 255, 0), cv2.FILLED)
#         cv2.imwrite('./Pitcher/resultData/peak_lag.jpg', img)
#         print("Peak_lag success")
#         # break
#     if cap.get(cv2.CAP_PROP_POS_FRAMES) == foot_plant_frame:
#         cv2.imwrite('./Pitcher/resultData/foot_plant.jpg', img)
#         print("foot_plant success")
#     if cap.get(cv2.CAP_PROP_POS_FRAMES) == max_shoulder_external_rotation_frame:
#         cv2.imwrite('./Pitcher/resultData/max_shoulder_external_rotation.jpg', img)
#         print("max_shoulder_external_rotation success")
#     # if ball_release_frame == 0:
#     #     ball_release_frame = max_shoulder_external_rotation_frame + 2
#     if cap.get(cv2.CAP_PROP_POS_FRAMES) == ball_release_frame:
#         cv2.imwrite('./Pitcher/resultData/Ball_release.jpg', img)
#         print("Ball_release success")
# cap.release()
catch_keyframeimage()
subprocess.call(["python3", "Pitcher/tran_3D_location.py"])
with open('./Pitcher/resultData/log.txt', 'w', newline='') as logfile:
    logfile.write("0")