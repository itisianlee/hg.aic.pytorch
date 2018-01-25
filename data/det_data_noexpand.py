# coding:utf8
from torch.utils.data import Dataset
import pandas as pd
import os
import pickle
from tqdm import tqdm
import numpy as np
import cv2
from config import opt
from .augmentation import GenerateLabel
from utils.helper import Helper


class HPEDetDataset_NE(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, phase='train'):
        self.anno_file = annotations_file
        self.img_dir = img_dir
        self.transform = transform
        self.phase = phase
        self.helper = Helper()

        self.pro_annos = []
        self.gen_intermediate_file()
        self.generatelabel = GenerateLabel()

    def __getitem__(self, idx):
        anno = self.pro_annos[idx]
        image = cv2.imread(self.img_dir + anno['image_id'] + '.jpg')
        lx, ly = anno['coords'][:2]
        rx, ry = anno['coords'][2:]
        img = image[ly:ry, lx:rx, :]
        kps = np.array(anno['keypoints']).reshape(-1, 3) - [lx, ly, 0]
        if self.transform is not None:
            img, kps = self.transform(img, kps)
        img, label = self.generatelabel(img, kps)
        img = img[:, :, (2, 1, 0)]
        img = np.transpose(img, (2, 0, 1))

        return img, label

    def __len__(self):
        return len(self.pro_annos)

    def gen_intermediate_file(self):
        _pkl_file = opt.interim_data_path + '{}_preprocessed.pkl'.format(self.phase)
        if os.path.exists(_pkl_file):
            self.pro_annos = pickle.load(open(_pkl_file, 'rb'))
            # if self.phase is 'val':
            #     self.pro_annos = self.pro_annos[:5000]
        else:
            anno = pd.read_json(self.anno_file)
            for i in tqdm(range(anno.shape[0])):
                img_np = cv2.imread(self.img_dir + anno.image_id[i] + '.jpg')
                h, w = np.shape(img_np)[:2]
                for k, v in anno.human_annotations[i].items():
                    self.pro_annos.append({'image_id': anno.image_id[i],
                                           'human': k,
                                           'coords': v,
                                           'height_width': (h, w),
                                           'keypoints': anno.keypoint_annotations[i][k]})
            self.pro_annos = list(filter(lambda x: x['image_id'] not in self.helper.img_list, self.pro_annos))
            del anno
            with open(_pkl_file, 'wb') as f:
                pickle.dump(self.pro_annos, f)
