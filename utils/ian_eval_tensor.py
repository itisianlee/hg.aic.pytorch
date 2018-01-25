# coding:utf8
import numpy as np
from config import opt
import torch.nn.functional as F


def compute_batch_oks(preds, annos):
    oks_all = np.zeros(0)
    # 8, 14, 64, 64]
    preds = F.upsample(preds, scale_factor=4, mode='bilinear').data.numpy()
    for i, pred in enumerate(preds):
        coord = annos['coords'][i].numpy()
        kps = annos['keypoints'][i].numpy()
        span_x = coord[2] - coord[0]
        span_y = coord[3] - coord[1]
        isupright, scale = (True, span_y) if span_y >= span_x else (False, span_x)
        _kps = get_keypoint_coordinate(isupright, pred, opt.threshold)
        pred_kps = convert_coordinate(_kps, coord, scale)
        oks_all = np.append(oks_all, compute_oks(pred_kps, kps, opt.delta, span_x * span_y))
    return oks_all


def compute_oks(pred_kps, kps, delta, scale):
    """Compute oks matrix (size gtN*pN)."""
    anno_keypoints = np.reshape(kps, (14, 3))
    visible = anno_keypoints[:, 2] == 1
    predict_keypoints = np.reshape(pred_kps, (14, 3))
    dis = np.sum((anno_keypoints[visible, :2] - predict_keypoints[visible, :2]) ** 2, axis=1)
    oks = np.mean(np.exp(-dis / 2 / delta[visible] ** 2 / (scale + 1)))
    return oks


def get_keypoint_coordinate(isupright, pred, threshold=0.0):
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
    kps = np.reshape(keypoints, (-1, 3))
    kps = (kps * [scale / 256.0, scale / 256.0, 1]).astype(np.int16)
    kps = kps + [human_position[0], human_position[1], 0]
    return kps.reshape(-1).tolist()
