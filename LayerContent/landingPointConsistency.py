########################################
# ITR＿EOSL_R300
# author: Rachel Liu
# badmiton: consistency calculation
#########################################
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime


def calcLandingPoint(X,Y):
    # 將X和Y轉換為numpy數組
    X = np.array(X)
    Y = np.array(Y)
    # print(X, Y)

    # 計算中心點
    center_x = np.mean(X)
    center_y = np.mean(Y)
    # print("mean: ", X, Y)

    # 計算每個點與中心點的距離
    distances = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
    # print("distance: ", distances)
    # print(distances)
    # 計算平均距離
    # mean_distance = np.mean(distances)

    # 找出最大距離 (作為半徑)
    radius = np.max(distances)

    # 計算標準差
    # std_distance = np.std(distances)

    # 使用平均距離和標準差來計算一致性
    #consistency = 100 - (std_distance / mean_distance) * 100

    #平均距離/最大包覆圓半徑
    # consistency =  (mean_distance) / (radius)
    # print(consistency)
    #consistency = 100 - (std_distance / mean_distance) * 100

    # 保證在0到100之間
    # consistency = np.round(consistency*100,2)  


    # 繪製散佈圖與中心點
    # if(isPltFlg):
    #     fig, ax = plt.subplots()
    #     plt.scatter(X, Y, label='Points')
    #     ax.set_xlim(0, 350)
    #     ax.set_ylim(0, 350)
    #     # 繪製最大包覆圓
    #     circle = plt.Circle((center_x, center_y), radius, color='blue', fill=False, label='Bounding Circle')
    #     plt.gca().add_patch(circle)


    #     plt.scatter(center_x, center_y, color='red', label='Center Point')
    #     plt.legend()
    #     plt.xlabel('X')
    #     plt.ylabel('Y')
    #     plt.title('Scatter Plot with Center Point-Consistency={0}%\n std={1}, mean={2}'.format(consistency, np.round(std_distance,2), np.round(mean_distance,2)))
    #     plt.grid(True)
    #     plt.show()
    #     fig.savefig('test'+str(datetime.now().strftime("%Y%m%d_%H%M%S"))+'.png')
    # return center_x, center_y, consistency, radius
    return radius
if __name__=='__main__':
    # 給定的X和Y座標
    # X = [25, 180, 180, 325, 150]
    # Y =  [125,20, 75, 275,275]
    # X = [218.0232517794178, 180, 325, 150]
    # Y =  [25.34680024010295, 75, 275,275]
    X = [2180.232517794178]
    Y =  [2534.680024010295]

    #X = [125, 138, 135, 120, 140]
    #Y =  [141,126, 125, 136,120]
    # center_x, center_y, consistency, radius = Calc_Consistency(X,Y)
    radius = calcLandingPoint(X,Y)
    # print("中心點 (x, y):", (center_x, center_y))
    # print("一致性:", consistency)
    print(f'最大包覆圓半徑: {radius:.10f}')