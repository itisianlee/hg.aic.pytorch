# coding:utf8
from torch.utils.data import Dataset
import pandas as pd
import os
import pickle
from skimage import io
from tqdm import tqdm
from . import image_handle
from config import opt
from utils import Helper
import numpy as np


class HumanPoseDetectionDataset(Dataset):
    """Human Pose Detection Dataset"""

    def __init__(self, annotations_file, img_dir, transform=None):
        """
        :param annotations_file: 标注信息的路径
        :param img_dir: 图片保存路径
        :param transform(callable, optional): 数据增强
        """
        self.processed_annotations = []
        self.annotations_file = annotations_file
        self.img_dir = img_dir
        self.transform = transform
        self.img_list = Helper().img_list  # 有问题的图片list

        self.gen_intermediate_file()

    def gen_intermediate_file(self):
        if os.path.exists(opt.interim_data_path + 'train_processed_dataset.pkl'):
            self.processed_annotations = pickle.load(open(opt.interim_data_path + 'train_processed_dataset.pkl', 'r'))
            # print(self.processed_annotations[:2])
        else:
            anno = pd.read_json(self.annotations_file + 'keypoint_train_annotations_20170909.json')
            for i in tqdm(xrange(anno.shape[0])):
                self.processed_annotations.extend(image_handle.annotations_handle(anno.image_id[i],
                                                                                  anno.human_annotations[i],
                                                                                  anno.keypoint_annotations[i]))
            # 过滤掉有问题的图片,后续可能会手工对有问题的图片处理
            # print(self.processed_annotations[:2])
            self.processed_annotations = filter(lambda x: x['img_id'] not in self.img_list, self.processed_annotations)
            pickle.dump(self.processed_annotations,
                        open(opt.interim_data_path + 'train_processed_dataset.pkl', 'w'))
            del anno

    def __getitem__(self, idx):
        img_name = self.processed_annotations[idx]['img_id']
        isupright = self.processed_annotations[idx]['info'][0][2]
        offset = self.processed_annotations[idx]['info'][2]
        lx, ly = self.processed_annotations[idx]['info'][0][:2]
        rx, ry = self.processed_annotations[idx]['info'][1][:2]
        img = io.imread(self.img_dir + img_name + '.jpg')
        img = img[ly:ry, lx: rx, :]
        processed_img = image_handle.crop_and_scale(img, offset, isupright)
        processed_img = np.transpose(processed_img, (2, 0, 1))
        label = image_handle.generate_part_label(self.processed_annotations[idx]['info'][3:])

        if self.transform is not None:
            processed_img, label = self.transform(processed_img, label)
        return processed_img.astype(np.float), label.astype(np.float)

    def __len__(self):
        return len(self.processed_annotations)


class HumanPoseValDataset(Dataset):
    """探测网络验证集"""

    def __init__(self, annotations_file, img_dir):
        self.annotations_file = annotations_file
        self.img_dir = img_dir
        self.problem_imgs = []  # 有问题的图片
        self.processed_annotations = []

        self.official_data = pd.read_json(self.annotations_file + 'keypoint_validation_annotations_20170911.json')
        self.process_problem_data()
        self.gen_intermediate_file()

    def __len__(self):
        return len(self.processed_annotations)

    def __getitem__(self, idx):
        img_name = self.processed_annotations[idx]['img_id']
        isupright = self.processed_annotations[idx]['info'][0][2]
        offset = self.processed_annotations[idx]['info'][2]
        lx, ly = self.processed_annotations[idx]['info'][0][:2]
        rx, ry = self.processed_annotations[idx]['info'][1][:2]
        img = io.imread(self.img_dir + img_name + '.jpg')
        img = img[ly:ry, lx: rx, :]
        processed_img = image_handle.crop_and_scale(img, offset, isupright)
        processed_img = np.transpose(processed_img, (2, 0, 1))
        label = image_handle.generate_part_label(self.processed_annotations[idx]['info'][3:])
        return processed_img.astype(np.float), label.astype(np.float)

    def process_problem_data(self):
        for i in xrange(self.official_data.shape[0]):
            for k, v in self.official_data.human_annotations[i].items():
                if v[0] >= v[2] or v[1] >= v[3]:
                    self.problem_imgs.append(self.official_data.image_id[i])

    def gen_intermediate_file(self):
        if os.path.exists(opt.interim_data_path + 'val_processed_dataset.pkl'):
            self.processed_annotations = pickle.load(open(opt.interim_data_path + 'val_processed_dataset.pkl', 'r'))
        else:
            for i in tqdm(xrange(self.official_data.shape[0])):
                if self.official_data.image_id[i] in self.problem_imgs:
                    continue
                else:
                    self.processed_annotations.extend(
                        image_handle.val_annotations_handle(
                            self.official_data.image_id[i],
                            self.official_data.human_annotations[i],
                            self.official_data.keypoint_annotations[i]
                        )
                    )
            pickle.dump(self.processed_annotations,
                        open(opt.interim_data_path + 'val_processed_dataset.pkl', 'w'))
            del self.official_data


# 以下是回归网络的数据集------------------------------------
class HumanPoseRegressionDataset(Dataset):
    """回归子网络数据集"""

    def __len__(self):
        return len(self.processed_annotations)

    def __init__(self, annotations_file, img_dir, transform=None):
        """
        :param annotations_file: 标注信息的路径
        :param img_dir: 图片保存路径
        :param transform(callable, optional): Optional transform to be applied on a sample.
        """
        self.annotations_file = annotations_file
        self.img_dir = img_dir
        self.transform = transform
        self.img_list = Helper().img_list  # 有问题的图片list

        # self.processed_annotations = pickle.load(open(self.annotations_file + 'train_processed_dataset.pkl', 'r'))
        self.processed_annotations = []
        self.gen_intermediate_file()

    def __getitem__(self, idx):
        img_name = self.processed_annotations[idx]['img_id']
        isupright = self.processed_annotations[idx]['info'][0][2]
        offset = self.processed_annotations[idx]['info'][2]
        lx, ly = self.processed_annotations[idx]['info'][0][:2]
        rx, ry = self.processed_annotations[idx]['info'][1][:2]
        img = io.imread(self.img_dir + img_name + '.jpg')
        img = img[ly:ry, lx: rx, :]
        processed_img = image_handle.crop_and_scale(img, offset, isupright)
        processed_img = np.transpose(processed_img, (2, 0, 1))

        label = image_handle.generate_regression_hm(self.processed_annotations[idx]['info'][3:])

        if self.transform is not None:
            processed_img, label = self.transform(processed_img, label)

        return processed_img.astype(np.float), label.astype(np.float)

    def gen_intermediate_file(self):
        if os.path.exists(opt.interim_data_path + 'train_processed_dataset.pkl'):
            self.processed_annotations = pickle.load(open(opt.interim_data_path + 'train_processed_dataset.pkl', 'r'))
            # print(self.processed_annotations[:2])
        else:
            anno = pd.read_json(self.annotations_file + 'keypoint_train_annotations_20170909.json')
            for i in tqdm(xrange(anno.shape[0])):
                self.processed_annotations.extend(image_handle.annotations_handle(anno.image_id[i],
                                                                                  anno.human_annotations[i],
                                                                                  anno.keypoint_annotations[i]))
            # 过滤掉有问题的图片,后续可能会手工对有问题的图片处理
            # print(self.processed_annotations[:2])
            self.processed_annotations = filter(lambda x: x['img_id'] not in self.img_list, self.processed_annotations)
            pickle.dump(self.processed_annotations,
                        open(opt.interim_data_path + 'train_processed_dataset.pkl', 'w'))
            del anno


class RegressionValDataset(Dataset):
    """回归网络验证集"""

    def __init__(self, annotations_file, img_dir):
        self.annotations_file = annotations_file
        self.img_dir = img_dir
        self.problem_imgs = []  # 有问题的图片

        # self.processed_annotations = pickle.load(open(opt.interim_data_path + 'val_processed_dataset.pkl', 'r'))
        self.processed_annotations = []

        self.official_data = pd.read_json(self.annotations_file + 'keypoint_validation_annotations_20170911.json')
        self.process_problem_data()
        self.gen_intermediate_file()

    def __len__(self):
        return len(self.processed_annotations)

    def __getitem__(self, idx):
        img_name = self.processed_annotations[idx]['img_id']
        isupright = self.processed_annotations[idx]['info'][0][2]
        offset = self.processed_annotations[idx]['info'][2]
        lx, ly = self.processed_annotations[idx]['info'][0][:2]
        rx, ry = self.processed_annotations[idx]['info'][1][:2]
        img = io.imread(self.img_dir + img_name + '.jpg')
        img = img[ly:ry, lx: rx, :]
        processed_img = image_handle.crop_and_scale(img, offset, isupright)
        processed_img = np.transpose(processed_img, (2, 0, 1))
        return processed_img.astype(np.float), self.processed_annotations[idx]

    def process_problem_data(self):
        for i in xrange(self.official_data.shape[0]):
            for k, v in self.official_data.human_annotations[i].items():
                if v[0] >= v[2] or v[1] >= v[3]:
                    self.problem_imgs.append(self.official_data.image_id[i])

    def gen_intermediate_file(self):
        if os.path.exists(opt.interim_data_path + 'val_processed_dataset.pkl'):
            self.processed_annotations = pickle.load(open(opt.interim_data_path + 'val_processed_dataset.pkl', 'r'))
        else:
            for i in tqdm(xrange(self.official_data.shape[0])):
                if self.official_data.image_id[i] in self.problem_imgs:
                    continue
                else:
                    self.processed_annotations.extend(
                        image_handle.val_annotations_handle(
                            self.official_data.image_id[i],
                            self.official_data.human_annotations[i],
                            self.official_data.keypoint_annotations[i]
                        )
                    )
            pickle.dump(self.processed_annotations,
                        open(opt.interim_data_path + 'val_processed_dataset.pkl', 'w'))
            del self.official_data
