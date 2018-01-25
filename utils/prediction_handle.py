# coding:utf8
import numpy as np
from config import opt
import torch.nn.functional as F


def val_input_convert(predictions_in):
    """ 格式转换,转换成eval脚本能使用的prediction,处理成eval适配的数据格式
    :param predictions_in:
    :return: 例如:{"image_ids": ['img1','img2',...], "annos": {'img1':{"human3": [254, 203, 1, ...],"human2": ...}}}
    """
    predictions = dict()
    predictions['image_ids'] = []
    predictions['annos'] = dict()
    for pred in predictions_in:
        if pred[0] in predictions['image_ids']:
            predictions['annos'][pred[0]]['keypoint_annos'].update(pred[1])
        else:
            predictions['image_ids'].append(pred[0])
            predictions['annos'][pred[0]] = dict()
            predictions['annos'][pred[0]]['keypoint_annos'] = pred[1]
    return predictions


def get_prediction_keypoints(processed_info, preds):
    """ 计算一个batch 预测出来的骨骼点坐标
    :param processed_info: List [{'scale': scale_ratio, 'info': keypoints_res, 'img_id': img_name, 'human': human}]
    :param preds: 预测出来的热图,14张叠加
    :return: dict ["image name", {"human3": [254, 203, 1, ...],"huma2": ...}]
    """
    predictions = dict()
    # print(processed_info['img_id'][:2])
    for i, pred in enumerate(preds):
        if processed_info['img_id'][i] not in predictions.keys():
            predictions[processed_info['img_id'][i]] = {}
            kp_annos = get_keypoint_coordinate(processed_info['info'][i][0][2], pred, opt.threshold)
            new_kp_annos = convert_coordinate(kp_annos, processed_info['info'][i][0], processed_info['scale'][i])
            predictions[processed_info['img_id'][i]][processed_info['human'][i]] = new_kp_annos
        else:
            kp_annos = get_keypoint_coordinate(processed_info['info'][i][0][2], pred, opt.threshold)
            new_kp_annos = convert_coordinate(kp_annos, processed_info['info'][i][0], processed_info['scale'][i])
            predictions[processed_info['img_id'][i]][processed_info['human'][i]] = new_kp_annos
    return predictions.items()


def get_pred_kps(processed_info, preds):
    """ 计算一个batch 预测出来的骨骼点坐标
    :param processed_info: List [{'scale': scale_ratio, 'info': keypoints_res, 'img_id': img_name, 'human': human}]
    :param preds: 预测出来的热图,14张叠加
    :return: dict ["image name", {"human3": [254, 203, 1, ...],"huma2": ...}]
    """
    # print(len(processed_info))5
    # print(processed_info)
    # print(len(preds))8
    predictions = dict()
    # print(processed_info['img_id'][:2])
    # print(processed_info)
    preds = F.upsample(preds, scale_factor=4, mode='bilinear').data.numpy()
    for i, pred in enumerate(preds):
        coord = processed_info['coords'][i].numpy()
        span_x = coord[2] - coord[0]
        span_y = coord[3] - coord[1]
        isupright, scale = (True, span_y) if span_y >= span_x else (False, span_x)

        if processed_info['image_id'][i] not in predictions.keys():
            predictions[processed_info['image_id'][i]] = {}
            kp_annos = get_keypoint_coordinate(isupright, pred, opt.threshold)
        else:
            kp_annos = get_keypoint_coordinate(isupright, pred, opt.threshold)

        new_kp_annos = convert_coordinate(kp_annos, coord, scale)
        predictions[processed_info['image_id'][i]][processed_info['human'][i]] = new_kp_annos
    # print("--", list(predictions.items()))
    return list(predictions.items())


def get_keypoint_coordinate(isupright, pred, threshold=0.0):
    """
    实现函数trans_coordinate()的逆过程,坐标转换
    :param threshold:
    :param isupright: 是否站立
    :param pred: 预测的热图
    :return: 返回预测的缩放后的骨骼点
    """
    kps = []
    if isupright:
        for p in pred:
            if np.max(p) > threshold:
                x = np.argmax(p) % 256
                y = np.argmax(p) / 256
                kps.append([x, y, 1])
            else:
                kps.append([0, 0, 0])
    else:
        for p in pred:
            if np.max(p) > threshold:
                x = np.argmax(p) % 256
                y = np.argmax(p) / 256
                kps.append([x, y, 1])
            else:
                kps.append([0, 0, 0])
    return kps


def convert_coordinate(keypoints, human_position, scale):
    """predicted keypoints 2 original keypoints
    :param scale: 缩放比
    :param keypoints: 关节点
    :param human_position: 原图中人框坐标
    :return: 返回新的keypoints
    """
    kps = np.reshape(keypoints, (-1, 3))
    kps = (kps * [scale / 256.0, scale / 256.0, 1]).astype(np.int16)
    kps = kps + [human_position[0], human_position[1], 0]
    return kps.reshape(-1).tolist()


if __name__ == '__main__':
    a = [('sdfg', {'h1': [1, 4, 5], 'h2': [23, 7]}), ('wer', {'h1': [2, 43, 5]}), ('sdfg', {'h3': [23, 435]})]
    print(val_input_convert(a))
