import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import csv
import os
import shutil

# Set up video output
# fps = 118
# fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# out = cv2.VideoWriter('./Pitcher/resultData/output_3D.mp4', fourcc, fps, (800, 800))

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')
frames = []
# csv_file = open("./resultData/keypoint_3D.csv")
csv_file = open("./Pitcher/resultData/keypoint_trans.csv")
csv_reader = csv.reader(csv_file)
next(csv_reader)  # 跳過第一行
# header = next(csv_reader)
max_value = float('-inf')
for row in csv_reader:
    try:
        value = float(row[0])
        if value > max_value:
            max_value = value
    except ValueError:
        # Handle invalid values here
        pass

print('Max value:', max_value)

csv_file.seek(0)
next(csv_reader)
for frame in range(2, int(max_value)+1):
    x = []
    y = []
    z = []
    for row in csv_reader:
        if int(row[0]) == frame:
            x.append(float(row[2]) * 100)
            y.append(-float(row[3]) * 100)  # reverse Y
            z.append(-float(row[4]) * 100)  # reverse Z

    ax.clear()
    ax.scatter3D(x, y, z, color='green')

    connection = [[11, 12], [12, 24], [24, 23], [11, 23], [24, 26], [26, 28], [23, 25], [25, 27], [15, 13],
                  [13, 11],
                  [12, 14], [14, 16]]
    for c in connection:
        x1 = [x[c[0]], x[c[1]]]
        y1 = [y[c[0]], y[c[1]]]
        z1 = [z[c[0]], z[c[1]]]
        ax.plot(x1, y1, z1, color='blue')

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_xlim3d(-200, 200)
    ax.set_ylim3d(-200, 200)
    ax.set_zlim3d(-200, 200)
    ax.view_init(elev=100, azim=-90)
    plt.title(frame)

    # if frame == 54:
    #     plt.show()
    # Draw the plot
    fig.canvas.draw()
    # Convert the plot to an image
    img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img = img.reshape((800, 800, 3))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    frames.append(img)
    # Write the image to the video file
    # out.write(img)

    # Rewind the CSV file to the beginning
    csv_file.seek(0)
    next(csv_reader)

# Release the video writer and close the plot


plt.close()
csv_file.close()
fps = 30
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('./Pitcher/resultData/output_3D.mp4', fourcc, fps, (800, 800))
for frame in frames:
    out.write(frame)
out.release()

with open('./Pitcher/resultData/3D_log.txt', 'w', newline='') as logfile:
    logfile.write("0")
    print("3D_plot success")

#複製檔案到影片的目錄內
def copy_folder_contents(source_folder, destination_folder):
    # 创建目标文件夹
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    # 遍历源文件夹中的所有文件和子文件夹
    for item in os.listdir(source_folder):
        item_path = os.path.join(source_folder, item)
        destination_path = os.path.join(destination_folder, item)

        if os.path.isfile(item_path):
            # 如果是文件，则直接复制到目标文件夹
            shutil.copy2(item_path, destination_path)
        elif os.path.isdir(item_path):
            # 如果是文件夹，则递归调用自身进行复制
            copy_folder_contents(item_path, destination_path)
def find_file_location():
    file = open('./Pitcher/loc.txt', 'r')
    directory = file.read()
    return directory + "/"

# 指定源文件夹和目标文件夹的路径
source_folder = "./Pitcher/resultData/"
destination_folder = find_file_location()

copy_folder_contents(source_folder, destination_folder)