# coding:utf8
from config import opt
import models
from data import EvalDataset
import torch
from torch.utils import data
from torch.autograd import Variable
from utils.prediction_handle import get_pred_kps, val_input_convert
from utils import eval_score
from tqdm import tqdm
import json


def eval_train_val():
    opt.model_id = 4
    opt.val_bs = 8
    model = getattr(models, opt.model[opt.model_id])(num_stacks=4)
    torch.cuda.set_device(1)
    model = model.cuda(1)
    with open('checkpoints/AIC-HGNet_progress.json', 'r') as f:
        progress = json.load(f)
    model.load_state_dict(torch.load(progress['best_path']))
    # best_path = 'checkpoints/AIC-HGNet_0.567476429117.model'
    # model.load_state_dict(torch.load(best_path))

    val_anno_path = 'official/keypoint_validation_annotations_newclear.json'
    annotations = eval_score.load_annotations(val_anno_path)
    val_anno_file = '/media/bnrc2/_backup/ai/ai_challenger_keypoint_test_a_20170923/val10000_anno-newclear_thr4.5.pkl'
    # val_anno_file = '/home/bnrc2/mu/mxnet-ssd/22338-test-b-data.json'
    # val_anno_file = '/media/bnrc2/_backup/ai/mu/abiao_liang/res24_anno.pkl'
    print(val_anno_file)
    dataset = EvalDataset(val_anno_file, opt.val_img_dir)
    # dataset = EvalDataset(val_anno_file, '/media/bnrc2/_backup/ai/mu/abiao_liang/', 'test')

    dataloader = data.DataLoader(dataset, batch_size=opt.val_bs, num_workers=opt.num_workers)
    # TODO: model.eval()
    model.eval()
    print("proposessing data begin...")
    pred_list = []
    for processed_img, processed_info in tqdm(dataloader, ncols=50):
        processed_img = processed_img.float()
        processed_img = Variable(processed_img.cuda())
        pred_list += get_pred_kps(processed_info, model(processed_img)[-1].cpu())
    print("proposessing data end...")

    predictions = val_input_convert(pred_list)
    with open('./res12-03_keypoints.json', 'w') as f:
        json.dump(predictions, f)
    model.train()
    mAP_value = eval_score.keypoint_eval(predictions, annotations)
    print('mAP_value:', mAP_value)


if __name__ == '__main__':
    eval_train_val()
