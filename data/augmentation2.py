# coding=utf-8
import cv2
import numpy as np
from numpy import random
import math


class RandomRotation(object):
    def __call__(self, image, keypoints):
        h, w, c = image.shape
        center = (w / 2, h / 2)
        if random.randint(2):
            degree = random.randint(-30, 30)
            w_r = int(w * math.fabs(math.cos(math.radians(degree))) +
                      h * math.fabs(math.sin(math.radians(degree))))
            h_r = int(w * math.fabs(math.sin(math.radians(degree))) +
                      h * math.fabs(math.cos(math.radians(degree))))
            m = cv2.getRotationMatrix2D(center, degree, 1)  # 旋转矩阵
            m[0][2] += (w_r - w) / 2
            m[1][2] += (h_r - h) / 2
            image = cv2.warpAffine(image, m, (w_r, h_r))
            # 计算旋转后的关节点的位置
            kps = np.reshape(keypoints, (-1, 3))[:, 0:2]
            v = np.reshape(keypoints, (-1, 3))[:, 2]
            kps = np.hstack((kps, np.ones(14, dtype=int)[:, np.newaxis]))
            kps = np.dot(kps, m.T).astype('int')
            keypoints = np.hstack((kps, v[:, np.newaxis]))
            print(keypoints)
            print('--------RR---------')
        return image, keypoints


class RandomMirror(object):  # 实现图片的水平反转flip
    def __call__(self, image, keypoints):
        h, w, c = image.shape

        if random.randint(2):
            image = image[:, ::-1].copy()
            keypoints = keypoints.copy()
            keypoints[:, 0] = w - keypoints[:, 0]
            print(keypoints)
            print('--------RM---------')
        return image, keypoints


class Make_padding(object):
    def __call__(self, image):
        h, w, c = image.shape
        if h > w:
            image = np.concatenate((image, np.zeros((h, h - w, 3), dtype=image.dtype)), axis=1)
        elif w > h:
            image = np.concatenate((image, np.zeros((w - h, w, 3), dtype=image.dtype)), axis=0)
        return image


class Resize(object):
    def __init__(self, size=256):
        self.size = size
        self.mp = Make_padding()

    def __call__(self, image, keypoints):
        image = self.mp(image)
        image = cv2.resize(image, (self.size, self.size))
        return image, keypoints


class KeypointTransform(object):
    def __call__(self, image, keypoints):
        h, w, c = image.shape
        if h > w:
            scale = 256.0 / h
            m = np.array([[scale, 0, 0], [0, scale, 0]])  # 缩放变换矩阵
            keypoints = np.reshape(keypoints.copy(), (-1, 3))
            v = keypoints[:, 2]
            #             keypoints = np.dot(keypoints, m.T)
            keypoints = np.hstack((np.dot(keypoints, m.T), v[:, np.newaxis]))
        else:
            scale = 256.0 / w
            m = np.array([[scale, 0, 0], [0, scale, 0]])  # 缩放变换矩阵
            keypoints = np.reshape(keypoints.copy(), (-1, 3))
            v = keypoints[:, 2]
            keypoints = np.hstack((np.dot(keypoints, m.T), v[:, np.newaxis]))
        return image, keypoints.astype('int')


def makeGaussian(height, width, sigma=5, center=None):
    """
    Make a square gaussian kernel.
    :param height: 边长
    :param width: 边长
    :param sigma: 分布的幅度,标准差
    :param center: 高斯核中心
    :return: heatmap 带有高斯核
    """
    x = np.arange(0, width, 1, float)
    y = np.arange(0, height, 1, float)[:, np.newaxis]
    if center is None:
        x0 = width // 2
        y0 = height // 2
    else:
        x0 = center[0]
        y0 = center[1]
    return np.exp(-4 * np.log(2) * ((x - x0) ** 2 + (y - y0) ** 2) / sigma ** 2)


class GenerateHeatMap(object):
    def __call__(self, image, keypoints):
        hm = np.zeros((14, 256, 256), dtype=np.float32)
        for i, kp in enumerate(keypoints):
            if kp[2] == 1:
                hm[i] = makeGaussian(256, 256, sigma=5, center=(kp[0], kp[1]))
        return image, hm


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, iamge, keypoints):
        for t in self.transforms:
            image, keypoints = t(iamge, keypoints)
        return image, keypoints


class HPEAugmentation2(object):
    def __init__(self):
        self.augment = Compose([
            RandomMirror(),
            RandomRotation()
        ])

    def __call__(self, image, keypoints):
        return self.augment(image, keypoints)
