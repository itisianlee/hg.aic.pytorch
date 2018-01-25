# coding:utf8
from torch.utils.data import Dataset
import os
import pickle
import numpy as np
import cv2
from config import opt
from .augmentation import Resize, SubtractMeans
from tqdm import tqdm


class HPEPoseTestDataset_NE(Dataset):
    def __init__(self, annotations_file, test_img_dir):
        self.anno_file = annotations_file
        self.img_dir = test_img_dir

        self.pro_annos = []
        self.gen_intermediate_file()

        self.resize = Resize(mean=opt.test_mean)
        self.submean = SubtractMeans(mean=opt.test_mean)

    def gen_intermediate_file(self):
        _pkl_file = opt.interim_data_path + 'test_preprocessed.pkl'
        if os.path.exists(_pkl_file):
            self.pro_annos = pickle.load(open(_pkl_file, 'rb'))
        else:
            with open(self.anno_file, 'rb') as f:
                anno = pickle.load(f)
            for i in tqdm(range(len(anno))):
                img_np = cv2.imread(self.img_dir + anno[i]['image_id'] + '.jpg')
                h, w = np.shape(img_np)[:2]
                for k, v in anno[i]['human_annotations'].items():
                    self.pro_annos.append({'image_id': anno[i]['image_id'],
                                           'human': k,
                                           'coords': v,
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
