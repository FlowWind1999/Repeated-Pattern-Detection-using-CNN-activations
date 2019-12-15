import math
import time
import os
from scipy import stats
import threading

# import cv2
from scipy import interpolate
from cv2 import cv2
import numpy as np
from matplotlib import pyplot as plt
sigma = 5
fi = 0.65
global TP
global FP
global FN

TP = 0
FP = 0
FN = 0

def draw_func(img_path):
    print(img_path)
    o_pic = cv2.imread(img_path, 1)
    col = o_pic.shape[1]
    y = []
    for i in range(col):
        cnt = np.sum(o_pic[:, i, 2])
        if cnt:
            y.append(cnt)
    len_y = len(y)

    # 平滑处理
    # x = [x for x in range(len_y)]
    # f = interpolate.interp1d(x, y, kind='cubic')
    # y = f(x)
    deta = sorted(y)[int(len_y * fi)]  # 取上60%分位点
    peaks = [i for i in range(len_y)
            if y[i] == max(y[max(0, i-sigma):min(i+sigma, len_y)]) and y[i] >= deta]
    '''
    peaks= []
    for i in range(len_y):
        l = max(0, i-sigma)
        r = min(len_y-1, i+sigma)
        if y[i] == max(y[l:r]) and y[i] >= deta:
            peaks.append(i)
    '''
    deta = [deta]*len_y
    # print(y)
    plt.figure()
    x = np.arange(len(y))
    y = np.array(y)
    plt.plot(x, y)
    plt.plot(x, deta)
    plt.scatter(peaks, [y[i] for i in peaks], s=40, color='red')
    plt.savefig("./func/" + os.path.basename(img_path))
    # plt.show()

'''利用重心偏移判断法  这个不能用'''
def f_method(img_path):
    print(img_path)
    o_pic = cv2.imread(img_path, 1)
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
    global TP
    global FP
    global FN
    print(img_path)
    o_pic = cv2.imread(img_path, 1)
    col = o_pic.shape[1]
    y = []
    for i in range(col):
        cnt = np.sum(o_pic[:, i, 2])
        if cnt:
            y.append(cnt)
    len_y = len(y)
    deta = sorted(y)[int(len_y * fi)]   # 取上60%分位点
    peaks = [i for i in range(len_y)
             if y[i] == max(y[max(0, i-sigma):min(i+sigma, len_y)]) and y[i] >= deta]
    gap = [peaks[i]-peaks[i-1] for i in range(1, len(peaks))]
    # gap = [i for i in gap if i > sum(gap)/(len(gap)*1.5)]  # 过滤掉较近的峰值点
    # print(gap)
    # gap = np.array(gap)
    var = max(gap)
    mid = np.median(gap)   # 中位数 用中位数估计正常间距
    '''
    # 最大间距超出正常间距过多
    if var > 1.75 * mid:
        print(os.path.basename(img_path)+" is breaken !!!")
        if len(os.path.basename(img_path)) > 10:  # 判断正确
            TP += 1
        else:
            FN += 1
    else:
        if len(os.path.basename(img_path)) > 10:  # 判断错误，但是是正样本
            FP += 1
    '''
    # 最大间距超出正常间距过多
    if var > 1.75 * mid:
        print(os.path.basename(img_path) + " is breaken !!!")
        if len(os.path.basename(img_path)) == 10:
            FN += 1
    else:
        if len(os.path.basename(img_path)) == 10:  # 判断正确，是正样本
            TP += 1
        else:
            FP += 1


if __name__ == '__main__':
    start = time.time()

    #img_path = './rot/111111.png'
    #f_method(img_path)

    filelist = os.listdir("./rot")
    for file in filelist:
        img_path = "./rot/" + file
        draw_func(img_path)
        peak_method(img_path)
    end = time.time()

    pre = TP/(TP+FP)
    recall = TP/(TP+FN)
    F1score = 2*pre*recall/(pre+recall)
    print("precision : ", pre)
    print("recall : ", recall)
    print("F1score : ", F1score)
    print("total used time is {} s".format(end - start))