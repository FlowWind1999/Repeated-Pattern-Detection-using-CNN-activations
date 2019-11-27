import math
import time
import os
import threading

import cv2
import numpy as np
from sklearn import linear_model
from matplotlib import pyplot as plt


min_ang = 0.01
nMaxThread = threading.Semaphore(4)  # 这里设置需要开启几条线程
def rotate(img_path):
    with nMaxThread:
        o_pic = cv2.imread(img_path, 1)
        row = o_pic.shape[0]
        col = o_pic.shape[1]

        pic = cv2.Canny(o_pic, 80, 150)
        print("pic shape is ", pic.shape)

        ang = 360
        # 旋转多次
        while math.fabs(ang) > min_ang:
            points_x = []
            points_y = []
            for r in range(row):
                for c in range(col):
                    if pic[r][c]:
                        points_x.append([c])
                        points_y.append([r])

            print("points size", len(points_x))
            reg = linear_model.LinearRegression()
            reg.fit(points_x, points_y)
            ang = math.atan(reg.coef_) / math.pi * 180
            # if ang < 0:
            #    ang = ang + 360
            print("reg coef_", reg.coef_)
            print('ang is ', ang)

            centre_x = np.sum(points_x) / len(points_x)
            centre_y = np.sum(points_y) / len(points_y)
            # rot = cv2.getRotationMatrix2D((int(pic.shape[0] * 0.5), int(pic.shape[1] * 0.5)), ang, 1)
            rot = cv2.getRotationMatrix2D((int(centre_x), int(centre_y)), ang, 1)
            if math.fabs(ang) > 71:
                pic = pic.transpose(1, 0)
                o_pic = o_pic.transpose(1, 0, 2)
                row, col = col, row
            else:
                pic = cv2.warpAffine(pic, rot, (o_pic.shape[1], o_pic.shape[0]))
                o_pic = cv2.warpAffine(o_pic, rot, (o_pic.shape[1], o_pic.shape[0]))

        cv2.imwrite("./rot/" + os.path.basename(img_path), o_pic, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
        #cv2.imshow(os.path.basename(img_path), o_pic)
        #cv2.waitKey(0)

if __name__ == '__main__':
    start = time.time()
    #img_path = './seg/000038.png'
    #rotate(img_path)

    thds = []
    filelist = os.listdir("./seg")
    for file in filelist:
        img_path = "./seg/"+file

        thd = threading.Thread(target=rotate, args=(img_path,))
        thd.start()
        thds.append(thd)

        #rotate(img_path)
    for thd in thds:
        thd.join()

    end = time.time()
    print("total used time is {} s".format(end-start))