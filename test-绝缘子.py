import os
import pickle
import time
import multiprocessing

import torch
from torchvision import models, transforms
from AlexNetConvLayers import alexnet_conv_layers

import numpy as np

import matplotlib.pyplot as plt
from PIL import Image
from scipy.ndimage.filters import maximum_filter, gaussian_filter
from scipy.stats import multivariate_normal
from skimage.feature import peak_local_max

from utils import custom_plot

# parameters
sigma_l = []
alfa_l = [5, 7, 15, 15, 15]
fi_prctile = 80
delta = 0.65

subsample_pairs = 10
peaks_max = 10000

preprocess_transform = transforms.Compose([transforms.ToTensor()])

dev = torch.device("cpu")
model = alexnet_conv_layers()
model.to(dev)

def load_image(img_path):
    image = Image.open(img_path).convert('RGB')
    return preprocess_transform(image).unsqueeze(0).to(dev)

def detection(img_path):
    start=time.time()

    image = load_image(img_path)
    # print(image.size(2))

    # conv features computation
    conv_feats = model(image)
    # print("computed features!")
    # peaks extraction
    peaks = []
    for li, l in enumerate(conv_feats):
        peaks.append([])
        maps = l.squeeze().detach().cpu().numpy()
        sigma_l.append((image.size(2) / maps.shape[1]) / 2)
        # print(sigma_l)
        # #visualization
        '''
        for fi, fmap in enumerate(maps[:5]):
             plt.subplot(1, 2, 1)
             plt.imshow(fmap)
             plt.subplot(1, 2, 2)
             tmp_max = maximum_filter(fmap, 1)
             max_coords = peak_local_max(tmp_max, 1)
             plt.imshow(peak_local_max(tmp_max, 1, indices=False))
             plt.waitforbuttonpress()
        '''
        print("maps", maps.shape)
        for fi, fmap in enumerate(maps):
            fmap = np.array(Image.fromarray(fmap).resize((image.size(3), image.size(2))))  # length and width

            # tmp_max = maximum_filter(fmap, 1)
            # max_coords = peak_local_max(tmp_max, 5)

            # plt.subplot(1, 2, 1)
            # plt.imshow(fmap)

            fmap = gaussian_filter(fmap, sigma=10)  # 高斯模糊
            tmp_max = maximum_filter(fmap, 1)
            max_coords = peak_local_max(tmp_max, 5)  # 得到peak点的坐标值

            # plt.subplot(1, 2, 2)
            # plt.imshow(fmap)
            # plt.waitforbuttonpress()

            peaks[li].append(max_coords[np.random.permutation(max_coords.shape[0])[:peaks_max]])

    print("peaks extraction!")
    # compute displacement set and voting space
    if not os.path.exists("./V"):
        os.mkdir("./V")
    pickefile = "./V/V_" + os.path.basename(img_path) + ".pkl"
    if os.path.exists(pickefile):
        with open(pickefile, 'rb') as f:
            V = pickle.load(f)
    else:
        quant_r, quant_c = np.mgrid[0:image.size(2):1, 0:image.size(3):1]
        V = np.zeros(quant_r.shape)
        quant_rc = np.empty(quant_r.shape + (2,), dtype=np.float32)
        quant_rc[:, :, 0] = quant_r
        quant_rc[:, :, 1] = quant_c
        disps = []
        for li, p in enumerate(peaks):
            disps.append([])
            for fi, p2 in enumerate(p):
                # pairs_inds = np.asarray([(i, j) for i in range(p2.shape[0])
                # for j in range(p2.shape[0]) if i != j and j > i])
                pairs_inds = np.asarray(np.meshgrid(np.arange(p2.shape[0]), np.arange(p2.shape[0])),
                                        dtype=np.uint8).T.reshape(-1, 2)
                pairs_inds = pairs_inds[pairs_inds[:, 0] > pairs_inds[:, 1]]
                if pairs_inds.shape[0] > 0:
                    tmp_disps = np.abs(p2[pairs_inds[:, 0]] - p2[pairs_inds[:, 1]])
                else:
                    tmp_disps = np.asarray([[]])
                if tmp_disps.size == 0:
                    continue
                tmp_disps = tmp_disps[np.random.permutation(tmp_disps.shape[0])[:subsample_pairs]]  # 抽了10个样点

                # disps[li].append(tmp_disps)
                # tmp_disps è Dfl
                for ij, dij in enumerate(tmp_disps):
                    # 这一条可能有问题
                    tmp_Vfiij = multivariate_normal.pdf(quant_rc, mean=dij,
                                                        cov=np.asarray([[sigma_l[li], 0], [0, sigma_l[li]]],
                                                        dtype=np.float32))

                    # 这一条可能有问题
                    tmp_Vfiij /= tmp_disps.shape[0]
                    V += tmp_Vfiij

        with open(pickefile, 'wb') as handle:
            pickle.dump(V, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print("compute displacement set and voting space")

    # find best step
    starting_ind = 10
    # TODO qualcosa per pesare di più gli step più piccoli
    #  dstar = np.asarray(((V[:, 0] / np.arange(0, V.shape[0], 1))[starting_ind:].argmax() + starting_ind
    #                    , (V[0, :] / np.arange(0, V.shape[1], 1))[starting_ind:].argmax() + starting_ind))

    dstar = np.asarray((V[starting_ind:, 0].argmax() + starting_ind
                        , V[0, starting_ind:].argmax() + starting_ind))  

    # compute consistent votes to compute fi
    fi_acc = []
    for li, p in enumerate(peaks):
        for fi, p2 in enumerate(p):
            pairs_inds = np.asarray(np.meshgrid(np.arange(p2.shape[0]), np.arange(p2.shape[0])),
                                    dtype=np.uint8).T.reshape(-1, 2)
            pairs_inds = pairs_inds[pairs_inds[:, 0] > pairs_inds[:, 1]]
            if pairs_inds.shape[0] > 0:
                tmp_disps = np.abs(p2[pairs_inds[:, 0]] - p2[pairs_inds[:, 1]])
            else:
                fi_acc.append(0)
                continue
            tmp_disps = tmp_disps[np.random.permutation(tmp_disps.shape[0])[:subsample_pairs]]

            fi_acc.append(len([1 for dij in tmp_disps if (np.linalg.norm(dij - dstar)) < 3 * alfa_l[li]]))
            # 算的是不超过阈值的率

    # is this correct??
    param_fi = np.percentile(fi_acc, fi_prctile)  # 取第80%个的数值

    # find weights for filters
    disps_star = []
    weights = []
    for li, p in enumerate(peaks):
        disps_star.append([])
        weights.append([])
        for fi, p2 in enumerate(p):
            # pairs_inds = np.asarray([(i, j) for i in range(p2.shape[0])
            # for j in range(p2.shape[0]) if i != j and j > i])
            pairs_inds = np.asarray(np.meshgrid(np.arange(p2.shape[0]), np.arange(p2.shape[0])),
                                    dtype=np.uint8).T.reshape(-1, 2)
            pairs_inds = pairs_inds[pairs_inds[:, 0] > pairs_inds[:, 1]]
            if pairs_inds.shape[0] > 0:
                tmp_disps = np.abs(p2[pairs_inds[:, 0]] - p2[pairs_inds[:, 1]])
            else:
                tmp_disps = np.asarray([[]])
            weights[li].append(0)    # 这里加数了
            if tmp_disps.size == 0:
                continue
            tmp_disps = tmp_disps[np.random.permutation(tmp_disps.shape[0])[:subsample_pairs]]
            # disps_star[li].append(tmp_disps)
            # tmp_disps è Dfl

            for ij, dij in enumerate(tmp_disps):
                tmp_diff = np.linalg.norm(dij - dstar, ord=2)  # 求L2范数
                if tmp_diff < 3 * alfa_l[li]:  # 根据阈值选d
                    # φ è 80esimo percentile, bisogna sommare i pesi per calcolare per ogni filtro
                    wijfl = np.exp(-(tmp_diff ** 2) /
                                   (2 * (alfa_l[li] ** 2))) / (tmp_disps.shape[0] + param_fi)
                    weights[li][-1] += wijfl
    print("find weights for filters")

    # find filters with weights higher than threshold
    selected_filters = []  # 从一个卷积层里面选出可利用特征的filter
    for li, w in enumerate(weights):
        tmp_weight_thr = delta * max(w)
        selected_filters.append([fi for fi, w2 in enumerate(w) if w2 > tmp_weight_thr])

    # accumulate origin coordinates loss
    acc_origin = []
    acc_origin_weights = []
    for li, w in enumerate(weights):
        for fi in selected_filters[li]:
            p2 = peaks[li][fi]
            pairs_inds = np.asarray(np.meshgrid(np.arange(p2.shape[0]), np.arange(p2.shape[0])),
                                    dtype=np.uint8).T.reshape(-1, 2)
            pairs_inds = pairs_inds[pairs_inds[:, 0] > pairs_inds[:, 1]]
            if pairs_inds.shape[0] > 0:
                tmp_disps = np.abs(p2[pairs_inds[:, 0]] - p2[pairs_inds[:, 1]])
            else:
                fi_acc.append(0)
                continue
            cons_disps = [dij for ij, dij in enumerate(tmp_disps)
                          if (np.linalg.norm(dij - dstar)) < 3 * alfa_l[li]]
            cons_disps_weights = [
                np.exp(-(np.linalg.norm(dij - dstar) ** 2) /
                       (2 * (alfa_l[li] ** 2))) / (tmp_disps.shape[0] + param_fi)
                for dij in cons_disps]
            acc_origin.extend(cons_disps)
            acc_origin_weights.extend(cons_disps_weights)

    o_r = np.linspace(-dstar[0], dstar[0], 10)  # 等差数列
    o_c = np.linspace(-dstar[1], dstar[1], 10)
    min_rc = (-1, -1)
    min_val = np.inf
    for r in o_r:
        for c in o_c:
            tmp_orig = np.asarray([r, c])
            # 这里的mod应该是一个从新定义的映射    是一个二元的映射
            tmp_val = [np.linalg.norm(np.mod((dij - tmp_orig), dstar) - (dstar / 2)) * acc_origin_weights[ij]
                       for ij, dij in enumerate(acc_origin)]
            tmp_val = np.sum(tmp_val)
            if tmp_val < min_val:
                min_val = tmp_val
                min_rc = (r, c)  # 关键要找出这个offset

    boxes = []
    tmp_img = np.array(Image.open(img_path))

    flag = [False, 0, False]
    # 用RANSRC（随机抽样一致）？
    for ri in range(100):
        flag[0] = False
        flag[1] = 0
        min_r = min_rc[0] + (dstar[0] * ri) - (dstar[1] / 2)
        if min_r + dstar[0] < tmp_img.shape[0] and min_r > 0:
            for ci in range(100):
                min_c = min_rc[1] + (dstar[1] * ci) - dstar[0] / 2
                if min_c + dstar[1] < tmp_img.shape[1] and min_c > 0:
                    sums=np.sum(tmp_img[int(min_r):int(min_r+dstar[0]), int(min_c):int(min_c+dstar[1]), 0] > 0)
                    if sums > 0.5 * dstar[0] * dstar[1]:
                        flag[0] = True
                        tmp_box = np.asarray([min_c, min_r, dstar[1], dstar[0]])
                        boxes.append(tmp_box)
                    else:
                        if flag[0] == True:
                            flag[1] = flag[1]+1
        if(flag[1] > 1):
            flag[2] = True

    name = os.path.basename(img_path)
    end = time.time()
    print("{} is breaken".format(name))
    print("{} used time is {} min".format(name, (end - start) / 60.0))
    custom_plot(tmp_img, box=boxes)


if __name__ == '__main__':
    img_path = "./rot/000000.png"
    detection(img_path)
