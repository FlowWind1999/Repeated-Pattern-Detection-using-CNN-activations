import math
import time
import os
from scipy import stats
import threading

import cv2
import numpy as np
from matplotlib import pyplot as plt

def draw_func(img_path):
    print(img_path)
    o_pic = cv2.imread(img_path, 1)
    col = o_pic.shape[1]
    y = []
    for i in range(col):
        count = np.sum(o_pic[:, i, 2])
        if count:
            y.append(count)
    deta = sorted(y)[int(len(y) * 0.6)]
    deta = np.array([deta]*len(y))
    # print(y)
    plt.figure()
    x = np.arange(len(y))
    y = np.array(y)
    plt.plot(x, y)
    plt.plot(x, deta)
    plt.savefig("./func/" + os.path.basename(img_path))
    # plt.show()

'''利用重心偏移判断法'''
def f_method(img_path):
    print(img_path)
    o_pic = cv2.imread(img_path, 1)
    row = o_pic.shape[0]
    col = o_pic.shape[1]
    y = []
    for i in range(col):
        count = np.sum(o_pic[:, i, 2])
        if count:
            y.append(count)

    # deta = sorted(y)[int(len(y) * 0.6)]
    deta = max(y)*0.75
    len_y = len(y)
    s = weight = 0
    o_c = len_y / 2
    for i in range(len_y):
        if y[i] > deta:
            s += i*y[i]
            weight += y[i]
    p_c = s/weight
    print(p_c, o_c)
    if abs(p_c - o_c)/o_c > 0.05:
        print(os.path.basename(img_path)+" is breaken!!!")

'''利用峰值间距判断法'''
def peak_method(img_path):
    print(img_path)
    o_pic = cv2.imread(img_path, 1)
    col = o_pic.shape[1]
    y = []
    for i in range(col):
        count = np.sum(o_pic[:, i, 2])
        if count:
            y.append(count)

    deta = sorted(y)[int(len(y) * 0.6)]
    len_y = len(y)

    peaks = [i for i in range(1, len_y-1)
             if y[i] > y[i-1] and y[i] > y[i+1] and y[i] >= deta]
    gap = [peaks[i]-peaks[i-1] for i in range(1, len(peaks))]
    # print(peaks)
    # print(gap)
    gap = [i for i in gap if i > sum(gap)/len(gap)/1.5]
    # print(gap)
    gap = np.array(gap)
    # var = np.var(gap)  //方差
    var = max(gap)
    # most_num = stats.mode(gap)[0][0] 众数
    mid = np.median(gap)
    if var > 1.75 * mid:
        print(os.path.basename(img_path)+" is breaken!!!")


if __name__ == '__main__':
    start = time.time()

    #img_path = './rot/111111.png'
    #f_method(img_path)

    filelist = os.listdir("./rot")
    for file in filelist:
        img_path = "./rot/" + file
        # draw_func(img_path)
        # f_method(img_path)
        peak_method(img_path)
    end = time.time()
    print("total used time is {} s".format(end - start))