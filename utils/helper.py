# coding:utf8
import pickle
from config import opt


class Helper(object):
    """
    实现一个helper类,用于对数据的预处理和清洗
    """

    def __init__(self):
        self.img_list = [u'97fb2cb75320cb0094681715dbb5aa7a13b27fb7',
                         u'bc4f5b8d1daef85d542a4dee4843c045d9511e9f',
                         u'9b4694e434c41e7aab3b921b7397293d68311a61',
                         u'ff585c6fbdb0672e0173afa08e689a98af22aa82',
                         u'c298cd4a867943e79e03ba92b52204e19c0cbe16',
                         u'46d05fc2173ae77970ba1eb33107092021753654',
                         u'9b1d18cae697da8160beed40834ee3b1bfb7386e',
                         u'd39b35193ef4d7f587ffe0cb4de9d2203b59fa63',
                         u'aa8e4ce1f69018eaeebac4fa8714a3c00ee85cba',
                         u'fd0a492dd5aa8d9bd73758f323bbbd1d88e2b03b',
                         u'b125b9cda788d1a02a11131f7aa1b0f835e13cbf',
                         u'b125b9cda788d1a02a11131f7aa1b0f835e13cbf',
                         u'e622c27c0760ed757e7f60b0fac37595ec538506',
                         u'f94c9f8d14f1c432e38b282012a570e9504c1239',
                         u'ba8ca016b0000b99f44176cb5c2636a951796621',
                         u'3f8957d7948790c29886f29e27e3809d8acd3ccc',
                         u'6805fee7416a6bc5003d291dc06956d5a5b06dc3',
                         u'daae63a17f06df617d7681d78968dc856685c1d4',
                         u'aee27632b9990f08cc39da6c8ce595544de96d16',
                         u'f3f1402e49251ddbc079ae208fa80ae6036eda94',
                         u'3aef0e2d3a64f2d45b2f0b2b4d38d202aff098cf',
                         u'89269460a718cd1902c525084d0ba2424ad1a348']

    def del_problem_imgs(self):
        new_anno = pickle.load(open(opt.annotations_file + 'processed_dataset.pkl', 'r'))
        for i, na in enumerate(new_anno):
            if na[0] in self.img_list:
                new_anno.pop(i)
        pickle.dump(new_anno, open(opt.annotations_file + 'processed_dataset.pkl', 'w'))
