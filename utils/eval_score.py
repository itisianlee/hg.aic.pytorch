# coding=utf-8
import json
import numpy as np


def load_annotations(anno_file):
    """Convert annotation JSON file."""

    annotations = dict()
    annotations['image_ids'] = set([])
    annotations['annos'] = dict()
    annotations['delta'] = 2 * np.array([0.01388152, 0.01515228, 0.01057665, 0.01417709, \
                                         0.01497891, 0.01402144, 0.03909642, 0.03686941, 0.01981803, \
                                         0.03843971, 0.03412318, 0.02415081, 0.01291456, 0.01236173])
    try:
        # print (anno_file)
        annos = json.load(open(anno_file, 'r'))
    except Exception:
        print('Annotation file does not exist or is an invalid JSON file.')

    for anno in annos:
        annotations['image_ids'].add(anno['image_id'])
        annotations['annos'][anno['image_id']] = dict()
        annotations['annos'][anno['image_id']]['human_annos'] = anno['human_annotations']
        annotations['annos'][anno['image_id']]['keypoint_annos'] = anno['keypoint_annotations']

    return annotations


def old_compute_oks(anno, predict, delta):
    """Compute oks matrix (size gtN*pN)."""

    anno_count = len(anno['keypoint_annos'].keys())
    predict_count = len(predict.keys())
    oks = np.zeros((anno_count, predict_count))
    # for every human keypoint annotation
    for i in range(anno_count):
        anno_key = list(anno['keypoint_annos'].keys())[i]
        anno_keypoints = np.reshape(anno['keypoint_annos'][anno_key], (14, 3))
        visible = anno_keypoints[:, 2] == 1
        bbox = anno['human_annos'][anno_key]
        scale = np.float32((bbox[3] - bbox[1]) * (bbox[2] - bbox[0]))
        if np.sum(visible) == 0:
            for j in range(predict_count):
                oks[i, j] = 0
        else:
            # for every predicted human
            for j in range(predict_count):
                predict_key = list(predict.keys())[j]
                predict_keypoints = np.reshape(predict[predict_key], (14, 3))
                dis = np.sum((anno_keypoints[visible, :2] \
                              - predict_keypoints[visible, :2]) ** 2, axis=1)
                oks[i, j] = np.mean(np.exp(-dis / 2 / delta[visible] ** 2 / scale))
    return oks


def compute_oks(anno, predict, delta):
    """Compute oks matrix (size gtN*pN)."""

    anno_count = len(anno['keypoint_annos'].keys())
    predict_count = len(predict.keys())
    oks = np.zeros((anno_count, predict_count))
    if predict_count == 0:
        return oks.T

    # for every human keypoint annotation
    for i in range(anno_count):
        anno_key = list(anno['keypoint_annos'].keys())[i]
        anno_keypoints = np.reshape(anno['keypoint_annos'][anno_key], (14, 3))
        visible = anno_keypoints[:, 2] == 1
        bbox = anno['human_annos'][anno_key]
        scale = np.float32((bbox[3] - bbox[1]) * (bbox[2] - bbox[0]))  # 框的面积
        if np.sum(visible) == 0:
            for j in range(predict_count):
                oks[i, j] = 0
        else:
            # for every predicted human
            for j in range(predict_count):
                predict_key = list(predict.keys())[j]
                predict_keypoints = np.reshape(predict[predict_key], (14, 3))
                dis = np.sum((anno_keypoints[visible, :2] - predict_keypoints[visible, :2]) ** 2, axis=1)
                oks[i, j] = np.mean(np.exp(-dis / 2 / delta[visible] ** 2 / (scale + 1)))
    return oks


def keypoint_eval(predictions, annotations):
    """Evaluate predicted_file and return mAP."""

    oks_all = np.zeros((0))
    oks_num = 0

    prediction_id_set = set(predictions['image_ids'])
    # for every annotation in our test/validation set
    for image_id in annotations['image_ids']:
        # if the image in the predictions, then compute oks
        # print(image_id)
        if image_id in prediction_id_set:
            oks = compute_oks(anno=annotations['annos'][image_id], \
                              predict=predictions['annos'][image_id]['keypoint_annos'], \
                              delta=annotations['delta'])
            # view pairs with max OKSs as match ones, add to oks_all
            oks_all = np.concatenate((oks_all, np.max(oks, axis=1)), axis=0)
            # accumulate total num by max(gtN,pN)
            oks_num += np.max(oks.shape)
        else:
            # otherwise report warning
            # number of humen in ground truth annotations
            gt_n = len(annotations['annos'][image_id]['human_annos'].keys())
            # fill 0 in oks scores
            oks_all = np.concatenate((oks_all, np.zeros((gt_n))), axis=0)
            # accumulate total num by ground truth number
            oks_num += gt_n

    # compute mAP by APs under different oks thresholds
    average_precision = []
    for threshold in np.linspace(0.5, 0.95, 10):
        average_precision.append(np.sum(oks_all > threshold) / np.float32(oks_num))

    return np.mean(average_precision)


def val_input_convert(anno):
    """ 格式转换,转换成eval脚本能使用的prediction
    :return: 例如:{"image_ids": ['img1','img2',...], "annos": {'img1':{"human3": [254, 203, 1, ...],"huma2": ...}}}
    """
    predictions = dict()
    predictions['image_ids'] = []
    predictions['annos'] = dict()
    for pred in anno:
        predictions['image_ids'].append(pred[0])
        predictions['annos'][pred[0]] = dict()
        predictions['annos'][pred[0]]['keypoint_annos'] = pred[1]
    return predictions


if __name__ == '__main__':
    path = '/home/bnrc2/ai_challenge/ian/Pytorch_Human_Pose_Estimation/interim_data/val_dataset' \
           '/keypoint_validation_annotations_20170911.json'
    anno = load_annotations(path)
    # print keypoint_eval(anno, anno)
    print(anno)
