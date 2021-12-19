import math

import numpy as np
import scipy
from operator import itemgetter
import cv2


def sub2ind(array_shape, rows, cols):
    return rows * array_shape[1] + cols


def ind2sub(array_shape, ind):
    rows = (ind.astype('int') / array_shape[1])
    cols = (ind.astype('int') % array_shape[1])
    return rows, cols


def ismember(A, B):
    tf = [np.sum(a == B) for a in A]
    index = np.zeros((len(tf),), dtype='int')
    for i in range(len(index)):
        if tf[i]:
            index[i] = np.where(B == A[i])[0]
        else:
            index[i] = -1
    return tf, index


def image_meshgen(height, width):
    x, y = np.meshgrid(np.arange(height), np.arange(width))
    vertex = np.array([y.flatten(), x.flatten()]).transpose()
    n, m = vertex.shape
    temp = np.arange(height - 1)
    for i in range(width - 2):
        temp = np.hstack((temp, np.arange(height - 1) + (i + 1) * height))
    face = np.vstack((np.vstack((temp, np.vstack((temp + height, temp + 1)))).transpose(),
                      np.vstack((temp + 1, np.vstack((temp + height, temp + height + 1)))).transpose()))
    vertex = np.hstack((vertex, np.ones((n, 1))))
    return vertex, face


def image_free_boundary(height, width):
    return np.vstack((np.concatenate((np.arange(0, height * width - height, width),
                                      np.arange(height * width - height, height * width - 1),
                                      np.flip(np.arange(2 * height - 1, height * width, width)),
                                      np.flip(np.arange(1, height)))),
                      np.concatenate((np.arange(height, height * width, width),
                                      np.arange(height * width - height + 1, height * width),
                                      np.flip(np.arange(height - 1, height * width - height, width)),
                                      np.flip(np.arange(0, height - 1)))))).transpose()


def vertex_search(XY, vertex):
    k = len(XY)
    index = np.zeros(shape=(k,))
    for i in range(k):
        index[i] = \
        min(enumerate(np.sqrt((vertex[0, :] - XY[i, 0]) ** 2 + (vertex[1, :] - XY[i, 1]) ** 2)), key=itemgetter(1))[0]
    return index


def close_curve_division(B, pt):
    for i in range(1, len(B) - 1):
        index1 = np.where(B[i::, 0] == B[i - 1, 1])
        tempa = B[i + index1[0], :]
        tempb = B[i, :]
        B[i, :] = tempa
        B[i + index1[0], :] = tempb
    n = len(pt)
    Edge = []
    _, location = ismember(pt, B[:, 0])
    sort_location = np.sort(location)
    index = np.argsort(location)
    for i in range(n - 1):
        Edge.append(B[np.arange(sort_location[i], sort_location[i + 1] + 1), 0])
    if sort_location[0] == 0:
        Edge.append(np.array([*B[sort_location[-1]::, 0], *[B[0, 0]]]))
    else:
        Edge.append(np.array([*B[sort_location[-1]::, 0], *B[np.arange(sort_location[0]), 0]]))
    Edge = np.squeeze(Edge)
    temp = []
    for i in index:
        temp.append(Edge[i])
    Edge = np.squeeze(temp)
    return Edge


def ResizeWithAspectRatio(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    return cv2.resize(image, dim, interpolation=inter)


def calculate_ssim(img1, img2, radius, neighborhood, window_size, alpha, beta, gamma):
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    if img1.max() > 1:
        img1 = img1/255
    if img2.max() > 1:
        img2 = img2/255
    kernel = cv2.getGaussianKernel(neighborhood, radius)
    window = np.outer(kernel, kernel.transpose())
    mu1 = cv2.filter2D(img1, -1, window)[window_size:-window_size, window_size:-window_size]
    mu2 = cv2.filter2D(img2, -1, window)[window_size:-window_size, window_size:-window_size]
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[window_size:-window_size, window_size:-window_size] - mu1_sq
    sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[window_size:-window_size, window_size:-window_size] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[window_size:-window_size, window_size:-window_size] - mu1_mu2
    # ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))
    luminance = (2 * mu1_mu2 + c1) / (mu1_sq + mu2_sq + c1)
    contrast = (2 * sigma1_sq * sigma2_sq + c2) / (sigma1_sq + sigma2_sq + c2)
    structure = (2 * sigma12 + c2) / (2 * sigma1_sq * sigma2_sq + c2)
    ssim_map = (luminance ** alpha) * (contrast ** beta) * (structure ** gamma)
    return ssim_map.mean()


def my_ssim(img1, img2, radius=1.5, neighborhood=11, window_size=5, alpha=1, beta=1, gamma=1):
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return calculate_ssim(img1, img2, radius, neighborhood, window_size, alpha, beta, gamma)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(calculate_ssim(img1, img2, radius, neighborhood, window_size, alpha, beta, gamma))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return calculate_ssim(np.squeeze(img1), np.squeeze(img2), radius, neighborhood, window_size, alpha, beta, gamma)
    else:
        raise ValueError('Wrong input image dimensions.')
