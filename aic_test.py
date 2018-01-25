# coding:utf8
from config import opt
import models
from data import HPEPoseTestDataset
import torch
from torch.utils import data
from torch.autograd import Variable
from utils.prediction_handle import get_pred_kps
from tqdm import tqdm
from torch import nn
import json


def aic_test():
    opt.model_id = 4
    model = getattr(models, opt.model[opt.model_id])(num_stacks=6)
    model = model.cuda()
    # with open('checkpoints/AIC-HGNet_progress.json', 'r') as f:
    #     progress = json.load(f)
    # model.load_state_dict(torch.load(progress['best_path']))
    best_path = 'checkpoints/AIC-HGNet_0.567476429117.model'
    model.load_state_dict(torch.load(best_path))
    # model = nn.DataParallel(model, device_ids=[0, 1])

    model.eval()

    opt.test_anno_file = '/media/bnrc2/_backup/ai/ai_challenger_keypoint_test_b_20171120/test_b_0.4.pkl'
    opt.test_img_dir = '/media/bnrc2/_backup/ai/ai_challenger_keypoint_test_b_20171120/keypoint_test_b_images_20171120/'
    dataset = HPEPoseTestDataset(opt.test_anno_file, opt.test_img_dir)

    print(len(dataset))
    dataloader = data.DataLoader(dataset, batch_size=opt.val_bs, num_workers=opt.num_workers)
    print("proposessing data begin...")
    pred_list = []
    for processed_img, processed_info in tqdm(dataloader, ncols=50):
        processed_img = processed_img.float()
        processed_img = Variable(processed_img.cuda())
        pred_list += get_pred_kps(processed_info, model(processed_img)[-1].cpu())
    print("proposessing data end...")
    pred_list_file = opt.interim_data_path + 'pred_test_list.pkl'
    with open(pred_list_file, 'wb') as f:
        import pickle
        pickle.dump(pred_list, f)
    submit = get_keypoints(pred_list)
    with open('12_03_submit.json', 'w') as f:
        json.dump(submit, f)


def get_keypoints(pred_list):
    predictions = dict()
    for pred in pred_list:
        if pred[0] in list(predictions.keys()):
            predictions[pred[0]]['keypoint_annotations'].update(pred[1])
        else:
            predictions[pred[0]] = {
                'image_id': pred[0],
                'keypoint_annotations': pred[1]
            }
    return list(predictions.values())


if __name__ == '__main__':
    aic_test()
