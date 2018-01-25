# coding:utf8
import time
import numpy as np


class Config:
    def __init__(self):
        self.demo = False  # 是否是在少量数据集上运行
        self.reuse = False
        self.resume = False
        self.device = 0
        self.progress_file = 'train_progress.json'
        self.title = "AIC"
        self.progress = {
            'best_path': '',
            'best_mAP': None,
            'count': 0,
            'lr': 2.5e-4,
            'epoch': 0,
        }

        self.train_mean = (108, 115, 122)
        self.val_mean = (103, 111, 119)  # [ 102.72338076,  111.23301759,  119.31230327]
        self.test_mean = (114, 122, 129)

        self.batch_size = 8  # batch
        self.val_bs = 8  # 验证集batch size
        self.epoch = 20  # 轮数
        self.start_epoch = 0
        self.start_count = 0
        self.lr = 2.5e-4  # 学习率
        self.min_lr = 1e-6
        self.lr_decay = 0.99  #
        self.shuffle = True
        self.augument = False  # 是否使用数据增强
        self.check_every = 5000  # 每多少个batch查看一下mAP,并修改学习率
        self.is_train = True  # 是否是训练阶段
        self.plot_every = 100  # 每10个batch, 更新visdom

        self.root_dir = './'
        # self.root_dir = '/home/hadoop/deeplearning/pytorch/pytorch_ai_challenger_HPE/'
        self.dataset_root_dir = '/media/bnrc2/_backup/ai/'

        self.img_dir = '/media/bnrc2/_backup/ai/ai_challenger_keypoint_train_20170902/keypoint_train_images_20170902/'
        self.annotations_file = '/home/bnrc2/ai_challenge/ian/hg.aic.pytorch/official/' \
                                'keypoint_train_annotations_newclear.json'

        self.val_anno_file = '/home/bnrc2/ai_challenge/ian/hg.aic.pytorch/official/' \
                             'keypoint_validation_annotations_newclear.json'
        self.val_img_dir = '/media/bnrc2/_backup/ai/ai_challenger_keypoint_validation_20170911' \
                           '/keypoint_validation_images_20170911/'
        self.test_anno_file = '/media/bnrc2/_backup/ai/ai_challenger_keypoint_test_a_20170923/test_anno.pkl'
        self.test_img_dir = '/media/bnrc2/_backup/ai/ai_challenger_keypoint_test_a_20170923/' \
                            'keypoint_test_a_images_20170923/'
        # self.val_anno_file = '/home/bnrc2/ai_challenge/ian/Pytorch_Human_Pose_Estimation/interim_data/val_dataset/'
        # self.val_img_dir = '/home/bnrc2/ai_challenge/ian/Pytorch_Human_Pose_Estimation/interim_data/val_dataset' \
        #                    '/val_imgs/'

        self.interim_data_path = self.root_dir + 'interim_data/'  # 训练过程中产生的中间数据(暂存)
        self.model_path = self.root_dir + 'models/'
        self.checkpoints = self.root_dir + 'checkpoints/'
        self.logs_path = self.root_dir + 'logs/'
        self.model_id = 0  # 选择的模型编号,
        self.model = ['Part_detection_subnet_model', 'Regression_subnet', 'Part_detection_subnet101',
                      'Regression_subnet101', 'HourglassNet', 'AIC-HGNet']  # 模型

        self.threshold = 0.0

        self.num_workers = 4  # 多线程加载所需要的线程数目
        self.pin_memory = False  # 数据从CPU->pin_memory—>GPU加速

        self.env = time.strftime('%m%d_%H%M%S')  # Visdom env
        # 1/右肩，2/右肘，3/右腕，4/左肩，5/左肘，6/左腕，7/右髋，8/右膝，9/右踝，10/左髋，11/左膝，12/左踝，13/头顶，14/脖子
        self.part_id = {1: 'r_shoulder', 2: 'r_elbow', 3: 'r_wrist', 4: 'l_shoulder', 5: 'l_elbow', 6: 'l_wrist',
                        7: 'r_hip', 8: 'r_knee', 9: 'r_ankle', 10: 'l_hip', 11: 'l_knee', 12: 'l_ankle', 13: 'head',
                        14: 'neck'}
        self.delta = 2 * np.array([0.01388152, 0.01515228, 0.01057665, 0.01417709, \
                                   0.01497891, 0.01402144, 0.03909642, 0.03686941, 0.01981803, \
                                   0.03843971, 0.03412318, 0.02415081, 0.01291456, 0.01236173])

    def parse(self, kwargs):
        """
        根据字典kwargs 更新 config参数
        """
        for k, v in kwargs.items():
            if not hasattr(self, k):
                raise Exception("opt has not attribute <%s>" % k)
            setattr(self, k, v)

    def config_info_print(self):
        print("Train&Val Batch size: ".rjust(30, ' '), self.batch_size, self.val_bs)
        print("Epochs: ".rjust(30, ' '), self.epoch)
        print("GPU device: ".rjust(30, ' '), self.device)
        print("Beginning Learning Rate: ".rjust(30, ' '), self.lr)
        print("Check&Plot every: ".rjust(30, ' '), self.check_every, self.plot_every)
        print("Train Data Dir: ".rjust(30, ' '), self.annotations_file)
        print("Val Data Dir: ".rjust(30, ' '), self.val_anno_file)
        if self.demo:
            print(" ".rjust(30, ' '), "NOTE: Using demo data!")
        else:
            print(" ".rjust(30, ' '), "NOTE: Not using demo data!")


opt = Config()
