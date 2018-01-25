# coding:utf8
from torch.utils.data import Dataset
import os
import pickle
import numpy as np
import cv2
from config import opt
from .augmentation import GenerateHeatMap, Resize, SubtractMeans
from tqdm import tqdm
import pandas as pd
import random


def gen_intermediate_file(img_dir, phase='train', transform=None):
    _pkl_file = opt.interim_data_path + '{}_preprocessed.pkl'.format(phase)
    pro_annos = pickle.load(open(_pkl_file, 'rb'))
    for i in range(2):
        if i == 1:
            for anno in tqdm(pro_annos):
                # print(self.img_dir + anno['image_id'] + '.jpg')
                image = cv2.imread(img_dir + anno['image_id'] + '.jpg')
                lx, ly = anno['coords'][:2]
                rx, ry = anno['coords'][2:]
                img = image[ly:ry, lx:rx, :]
                kps = np.array(anno['keypoints']).reshape(-1, 3) - [lx, ly, 0]
                if transform is not None:
                    img, kps = transform(img, kps)
                img, label = generatehm(img, kps, H=64, W=64, sigma=3)
                img = img[:, :, (2, 1, 0)]
                img = np.transpose(img, (2, 0, 1))
                return img, label, anno


class hgDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, phase='train'):
        self.anno_file = annotations_file
        self.img_dir = img_dir
        self.transform = transform
        self.phase = phase
        self.pro_annos = []
        self.gen_intermediate_file()
        self.generatehm = GenerateHeatMap()

    def gen_intermediate_file(self):
        _pkl_file = opt.interim_data_path + '{}_preprocessed.pkl'.format(self.phase)
        if os.path.exists(_pkl_file):
            self.pro_annos = pickle.load(open(_pkl_file, 'rb'))
        else:
            anno = pd.read_json(self.anno_file)
            for i in tqdm(range(anno.shape[0]), ncols=20):
                img_np = cv2.imread(self.img_dir + anno.image_id[i] + '.jpg')
                h, w = np.shape(img_np)[:2]
                for k, v in anno.human_annotations[i].items():
                    self.pro_annos.append({'image_id': anno.image_id[i],
                                           'human': k,
                                           'coords': np.array(v),
                                           'height_width': (h, w),
                                           'keypoints': np.array(anno.keypoint_annotations[i][k])})
            del anno
            with open(_pkl_file, 'wb') as f:
                pickle.dump(self.pro_annos, f)

    def __getitem__(self, idx):
        anno = self.pro_annos[idx]
        # print(self.img_dir + anno['image_id'] + '.jpg')
        image = cv2.imread(self.img_dir + anno['image_id'] + '.jpg')
        lx, ly = anno['coords'][:2]
        rx, ry = anno['coords'][2:]
        img = image[ly:ry, lx:rx, :]
        kps = np.array(anno['keypoints']).reshape(-1, 3) - [lx, ly, 0]
        if self.transform is not None:
            img, kps = self.transform(img, kps)
        img, label = self.generatehm(img, kps, H=64, W=64, sigma=3)
        img = img[:, :, (2, 1, 0)]
        img = np.transpose(img, (2, 0, 1))
        return img, label, anno

    def __len__(self):
        return len(self.pro_annos)


class hgValDataset(Dataset):
    def __init__(self, annotations_file, img_dir, num=None, phase='val'):
        self.anno_file = annotations_file
        self.img_dir = img_dir
        assert isinstance(num, int) or num is None  # 抽样数据量,num应该是整数
        self.num = num

        self.mean = opt.val_mean if phase == 'val' else opt.train_mean
        self.phase = phase

        self.pro_annos = []
        self.gen_intermediate_file()

        self.resize = Resize(mean=self.mean)
        self.submean = SubtractMeans(mean=self.mean)

    def gen_intermediate_file(self):
        _pkl_file = opt.interim_data_path + '{}_preprocessed.pkl'.format(self.phase)
        if os.path.exists(_pkl_file):
            self.pro_annos = pickle.load(open(_pkl_file, 'rb'))
            if self.num is not None:
                self.pro_annos = random.sample(self.pro_annos, self.num)
        else:
            anno = pd.read_json(self.anno_file)
            for i in tqdm(range(anno.shape[0]), ncols=20):
                img_np = cv2.imread(self.img_dir + anno.image_id[i] + '.jpg')
                h, w = np.shape(img_np)[:2]
                for k, v in anno.human_annotations[i].items():
                    self.pro_annos.append({'image_id': anno.image_id[i],
                                           'human': k,
                                           'coords': np.array(v),
                                           'height_width': (h, w),
                                           'keypoints': np.array(anno.keypoint_annotations[i][k])})
            del anno
            with open(_pkl_file, 'wb') as f:
                pickle.dump(self.pro_annos, f)

    def __getitem__(self, idx):
        anno = self.pro_annos[idx]
        image = cv2.imread(self.img_dir + anno['image_id'] + '.jpg')
        lx, ly = anno['coords'][:2]
        rx, ry = anno['coords'][2:]
        img = image[ly:ry, lx:rx, :]
        img, _ = self.resize(img, None)
        img, _ = self.submean(img, None)
        img = img[:, :, (2, 1, 0)]
        img = np.transpose(img, (2, 0, 1))
        return img, anno

    def __len__(self):
        return len(self.pro_annos)


class EvalDataset(Dataset):
    def __init__(self, annotations_file, test_img_dir, phase='val'):
        self.anno_file = annotations_file
        self.img_dir = test_img_dir
        self.phase = phase

        self.pro_annos = []
        self.gen_intermediate_file()
        if phase is 'val':
            self.mean = opt.val_mean
        elif phase is 'train':
            self.mean = opt.train_mean
        else:
            self.mean = opt.test_mean
        self.resize = Resize(mean=self.mean)
        self.submean = SubtractMeans(mean=self.mean)

    def gen_intermediate_file(self):
        _pkl_file = opt.interim_data_path + 'eval_{}30000_preprocessed.pkl'.format(self.phase)
        if os.path.exists(_pkl_file):
            self.pro_annos = pickle.load(open(_pkl_file, 'rb'))
        else:
            with open(self.anno_file, 'rb') as f:
                anno = pickle.load(f)
            for i in tqdm(range(len(anno)), ncols=50):
                img_np = cv2.imread(self.img_dir + anno[i]['image_id'] + '.jpg')
                h, w = np.shape(img_np)[:2]
                for k, v in anno[i]['human_annotations'].items():
                    coords = np.array(v).reshape(-1, 2)
                    offset = (coords[1] - coords[0]) * 0.15  # 沿着长和宽扩大30%
                    coords = v + np.concatenate((-offset, offset))
                    coords = coords.astype("int")
                    coords[np.where(coords < 0)] = 0
                    if coords[2] > w:
                        coords[2] = w
                    if coords[3] > h:
                        coords[3] = h
                    self.pro_annos.append({'image_id': anno[i]['image_id'],
                                           'human': k,
                                           'coords': coords,
                                           'height_width': (h, w)})
            with open(_pkl_file, 'wb') as f:
                pickle.dump(self.pro_annos, f)

    def __getitem__(self, idx):
        anno = self.pro_annos[idx]
        image = cv2.imread(self.img_dir + anno['image_id'] + '.jpg')
        lx, ly = anno['coords'][:2]
        rx, ry = anno['coords'][2:]
        img = image[ly:ry, lx:rx, :]
        kps = None
        img, kps = self.resize(img, kps)
        img, kps = self.submean(img, kps)
        img = img[:, :, (2, 1, 0)]
        img = np.transpose(img, (2, 0, 1))
        return img, anno

    def __len__(self):
        return len(self.pro_annos)
