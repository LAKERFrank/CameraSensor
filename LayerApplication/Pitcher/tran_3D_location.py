import numpy as np
import csv
import cv2

object_points_3D = np.empty((0, 12, 3), dtype=np.float32)
object_points_2D = np.empty((0, 12, 2), dtype=np.float32)

camera_maxtric = np.array([[1080.0, 0.0, 720.0],
                            [0.0, 1080.0, 540.0],
                            [0.0, 0.0, 1.0]], dtype=np.float32)
# dist_coeffs = np.array([[-0.5357608528825365, 0.42420392423835374, -0.0020378974867134004, -0.004261218299961821, -0.2375760856910901]], dtype=np.float32)
dist_coeffs = np.zeros((4,1))

max_frame = 0
with open('./Pitcher/resultData/keypoint.csv', 'r', newline='') as csvfile:
    reader = csv.reader(csvfile)
    next(reader)
    object_points = np.empty((0, 2), dtype=np.float32)
    save_index = 0
    for row in reader:
        frame_index = int(row[0])
        max_frame = frame_index
        if save_index != int(row[0]):
            save_index = int(row[0])
            object_points = np.empty((0, 2), dtype=np.float32)
        # if row[1] == str(11) or row[1] == str(12) or row[1] == str(13) or row[1] == str(14) or row[1] == str(15) or row[1] == str(16) or row[1] == str(23) \
        #         or row[1] == str(24) or row[1] == str(25) or row[1] == str(26) or row[1] == str(27) or row[1] == str(28) or row[1] == str(31) or row[1] == str(32):
        # if row[1] == str(11) or row[1] == str(12) or row[1] == str(15) or row[1] == str(16) or row[1] == str(23) \
        #         or row[1] == str(24) or row[1] == str(27) or row[1] == str(28):
        if 10 < int(row[1]) < 17 or 22 < int(row[1]) < 29:
            arr = np.array([float(row[2]), float(row[3])], dtype=np.float32)
            object_points = np.append(object_points, [arr], axis=0)
            # if row[1] == str(26):
            if row[1] == str(28):
                object_points_2D = np.append(object_points_2D, [object_points], axis=0)
                # print(object_points)
                # print("=======")
    print("object_points_2D.shape", object_points_2D.shape)
    print("max_Frame:", max_frame)
# 上面是把所以2D點都儲存起來。(像素座標）
# 11,12,13,14,15,16,23,24,25,26 共這10個骨架點


with open('./Pitcher/resultData/keypoint_3D.csv', 'r', newline='') as csvfile:
    reader2 = csv.reader(csvfile)

    cur_index = 0
    object_points = np.empty((0, 3), dtype=np.float32)
    save_index = 0
    next(reader2)
    for row in reader2:
        if save_index != int(row[0]):
            save_index = int(row[0])
            object_points = np.empty((0, 3), dtype=np.float32)

        cur_index = row[0]
        # if row[1] == str(11) or row[1] == str(12) or row[1] == str(13) or row[1] == str(14) or row[1] == str(15) or row[1] == str(16) or row[1] == str(23) \
        #         or row[1] == str(24) or row[1] == str(25) or row[1] == str(26) or row[1] == str(27) or row[1] == str(28) or row[1] == str(31) or row[1] == str(32):
        # if row[1] == str(11) or row[1] == str(12) or row[1] == str(23) or row[1] == str(24) or row[1] == str(27) \
        #         or row[1] == str(28) or row[1] == str(15) or row[1] == str(16):
        if 10 < int(row[1]) < 17 or 22 < int(row[1]) < 29:
            arr = np.array([float(row[2]), float(row[3]), float(row[4])], dtype=np.float32)
            object_points = np.append(object_points, [arr], axis=0)
            # if row[1] == str(26):
            if row[1] == str(28):
                object_points_3D = np.append(object_points_3D, [object_points], axis=0)
                # print(object_points)
                # print("=======")
    print("object_points_3D.shape:",object_points_3D.shape)
# 上面也勢將所有3D座標的骨架點儲存起來（mediapipe的世界座標）
# 11,12,13,14,15,16,23,24,25,26 有這些骨架點

# 用來存放旋轉向量(rvec)以及移動向量(tvec)
rvec_list = np.empty((0, 3, 1), dtype=np.float32)
tvec_list = np.empty((0, 3, 1), dtype=np.float32)

with open('./Pitcher/resultData/trans_vec.csv', 'w', newline='') as tran_vec:
    writer = csv.writer(tran_vec)
    writer.writerow(['frame', 'Keypoint id', 'rvec', 'tvec'])
    for i in range(max_frame):
        if i < max_frame:
            retval, rvec, tvec = cv2.solvePnP(object_points_3D[i], object_points_2D[i], camera_maxtric, dist_coeffs,flags = cv2.SOLVEPNP_ITERATIVE | cv2.SOLVEPNP_UPNP)
        # rotation_maxtrix, _ = cv2.Rodrigues(rvec)
        # object_points_camera = np.dot(rotation_maxtrix, object_points_3D[i].T) + tvec
        # print("====",object_points_camera)
        # print(rvec)
        # print(tvec)

            writer.writerow([i+1, rvec, tvec])

        # print(i+1 ,"rvec_list:", rvec)
        # print(i+1 ,"tvec_list:",tvec)
            rvec_list = np.append(rvec_list, [rvec], axis=0)
            tvec_list = np.append(tvec_list, [tvec], axis=0)
print("rvec_list.shape:", rvec_list.shape)
print("tvec_list.shape:", tvec_list.shape)
# 上面是將每個frame的rvec和tvec都儲存到其ｌｉｓｔ中

with open('./Pitcher/resultData/keypoint_trans.csv', 'w', newline='') as csvfile_tran:
    writer = csv.writer(csvfile_tran)
    writer.writerow(['frame', 'Keypoint id', 'x', 'y', 'z'])
    with open('./Pitcher/resultData/keypoint_3D.csv', 'r', newline='') as csvfile:
        reader2 = csv.reader(csvfile)
        next(reader2)

        for row in reader2:
            if int(row[0]) > 0:
                matrix1 = np.array([[float(row[2])], [float(row[3])], [float(row[4])]])
                # print(rvec_list[int(row[0])-2][0])
                rotation_maxtrix, _ = cv2.Rodrigues(rvec_list[int(row[0])-1])

                # rotation_maxtrix, _ = cv2.Rodrigues(rvec_list[5])
                object_points_camera = np.dot(rotation_maxtrix, matrix1) + tvec_list[int(row[0]) - 2]
                # object_points_camera = np.dot(rotation_maxtrix, matrix1) + tvec_list[5]

                writer.writerow([row[0], row[1], float(object_points_camera[0]), float(object_points_camera[1]), float(object_points_camera[2])])

                # print(object_points_camera)
                # print("====")
