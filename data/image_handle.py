# coding:utf8
from skimage import transform
import numpy as np


def trans_coordinate(hm_position, keypoints):
    """
    实现从原图到256*256分辨率的坐标转换,包括标注点(关节点)
    :param hm_position: 人框的位置,两个对角坐标点
    :param keypoints: 原关节标注点
    :return: 字典: 缩放率 scale
    info 人体位置信息和关节点信息组成的numpy数组维度(16,3)
    ((原图人体左上角坐标, 是否站立), (原图人体右下角坐标, 0), (人体右下角位置), 其他14个关节点)
    """
    span_x = hm_position[2] - hm_position[0]
    span_y = hm_position[3] - hm_position[1]
    isupright = True if span_y - span_x >= 0 else False
    keypoints_res = [hm_position[2], hm_position[3], 0] + keypoints
    keypoints_res = np.reshape(keypoints_res, (15, 3))

    if isupright:
        scale_ratio = 256.0 / span_y
        keypoints_res = keypoints_res - [hm_position[0], hm_position[1], 0]
        keypoints_res = keypoints_res * [scale_ratio, scale_ratio, 1]
        keypoints_res = np.vstack(
            ([[hm_position[0], hm_position[1], 1], [hm_position[2], hm_position[3], 0]], keypoints_res))
        keypoints_res = keypoints_res.astype(np.int32)
        return {'scale': span_y, 'info': keypoints_res}
    else:
        scale_ratio = 256.0 / span_x
        keypoints_res = keypoints_res - [hm_position[0], hm_position[1], 0]
        keypoints_res = keypoints_res * [scale_ratio, scale_ratio, 1]
        keypoints_res = np.vstack(
            ([[hm_position[0], hm_position[1], 0], [hm_position[2], hm_position[3], 0]], keypoints_res))
        keypoints_res = keypoints_res.astype(np.int32)
        return {'scale': span_x, 'info': keypoints_res}


def annotations_handle(img_name, human_positions, keypoints):
    """
    生成剪切以及按比例缩放的关键点
    :param img_name: 图片名字
    :param human_positions: 人框位置
    :param keypoints: 原关节标注点
    :return: List [{'scale': scale_ratio, 'info': keypoints_res, 'img_id': img_name}]
    """
    processed_list = []
    for k, v in human_positions.items():
        # 例(k, v) (u'human1', [185, 161, 418, 936])
        dict_temp = trans_coordinate(v, keypoints[k])
        dict_temp.update({'img_id': img_name})
        processed_list.append(dict_temp)
    # print(processed_list)
    return processed_list


def val_annotations_handle(img_name, human_positions, keypoints):
    """
    验证集,生成剪切以及按比例缩放的关键点
    :param img_name: 图片名字
    :param human_positions: 人框位置
    :param keypoints: 原关节标注点
    :return: List [{'scale': scale_ratio, 'info': keypoints_res, 'img_id': img_name, 'human': human}]
    """
    processed_list = []
    for k, v in human_positions.items():
        # 例(k, v) (u'human1', [185, 161, 418, 936])
        dict_temp = trans_coordinate(v, keypoints[k])
        dict_temp.update({'img_id': img_name, 'human': k})
        processed_list.append(dict_temp)
    return processed_list


def crop_and_scale(img, offset, isupright):
    """
    剪切以及按比例缩放,生成输入数据
    :param img: 原图,三维矩阵
    :param offset: 距离右上角的偏差(偏右或偏下)
    :param isupright: 是否是直立
    :return: 裁剪好的图像256*256,有可能是多张,与原图中人个数有关,维度:(256,256,3)
    """
    if isupright:
        func_off = offset[0]
        # print (func_off)
        return np.concatenate(
            (transform.resize(img, (256, func_off), mode='reflect'), np.zeros((256, 256 - func_off, 3))), axis=1)
    else:
        func_off = offset[1]
        # print (func_off)
        return np.concatenate(
            (transform.resize(img, (func_off, 256), mode='reflect'), np.zeros((256 - func_off, 256, 3))), axis=0)


def generate_part_label(keypoints, height=256, width=256, radius=10):
    """
    生成part detection subnet 标签数据
    :param keypoints: 新的标注点
    :param height: 生成的高度
    :param width: 生成的宽度
    :param radius: binary区域半径
    :return: labels
    """
    heatmap_res = None
    x = np.arange(0, width, dtype=np.uint32)
    y = np.arange(0, height, dtype=np.uint32)[:, np.newaxis]
    for kp in keypoints:
        if kp[2] == 1:
            if heatmap_res is None:
                heatmap_res = ((x - kp[0]) ** 2 + (y - kp[1]) ** 2) <= radius ** 2
            else:
                heatmap_res = np.vstack((heatmap_res, (((x - kp[0]) ** 2 + (y - kp[1]) ** 2) <= radius ** 2)))
        else:
            if heatmap_res is None:
                heatmap_res = np.zeros((height, width), dtype=np.uint8)
            else:
                heatmap_res = np.vstack((heatmap_res, np.zeros((height, width), dtype=np.uint8)))
    heatmap_res = heatmap_res.astype(np.uint8)

    return np.reshape(heatmap_res, (-1, height, width))


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


def generate_regression_hm(keypoint, height=256, width=256, sigma=5):
    """
    生成回归子网络的label
    :param keypoint: 关节点
    :param height: 边长
    :param width: 边长
    :param sigma: 分布的幅度,标准差
    :return: label
    """
    hm = np.zeros((14, height, width), dtype=np.float32)
    for i, kp in enumerate(keypoint):
        if kp[2] == 1:
            hm[i] = makeGaussian(height, width, sigma=sigma, center=(kp[0], kp[1]))
    return hm
