# coding:utf8
from data import HPEDetDataset_NE, HPEAugmentation, HPEPoseDataset, HPEPoseValDataset, HPEBaseTransform
import torch
from torch.utils import data
from torch.autograd import Variable
import numpy as np
from utils.prediction_handle import get_pred_kps, val_input_convert


def part1_val(model, writer, count, opt):
    """局部探测网络验证"""
    model.eval()
    if opt.demo:
        opt.val_anno_file = opt.root_dir + 'demo_data/keypoint_validation_annotations_20170911.json'
    dataset = HPEDetDataset_NE(opt.val_anno_file, opt.val_img_dir, HPEBaseTransform(opt.val_mean), phase='val')
    dataloader = data.DataLoader(dataset,
                                 batch_size=opt.batch_size,
                                 num_workers=opt.num_workers,
                                 pin_memory=opt.pin_memory,
                                 )
    loss_sum = 0
    for processed_img, label in dataloader:
        processed_img, label = Variable(processed_img.float().cuda()), Variable(label.float().cuda())
        detection_result = model(processed_img)
        val_loss = detection_loss_func(detection_result, label, opt)
        loss_sum += val_loss.data[0]
    model.train()
    return loss_sum / np.ceil(len(dataset) / opt.batch_size)


def part2_val(model, writer, count, opt):
    """回归子网络验证"""
    model.eval()
    if opt.demo:
        pass
    dataset = HPEPoseValDataset(opt.val_anno_file, opt.val_img_dir)
    dataloader = data.DataLoader(dataset, batch_size=opt.val_bs, num_workers=opt.num_workers)

    pred_list = []
    for processed_img, processed_info in dataloader:
        processed_img = processed_img.float()
        processed_img = Variable(processed_img.cuda())
        pred_list += get_pred_kps(processed_info, model(processed_img).cpu().data.numpy())

    predictions = val_input_convert(pred_list)

    model.train()
    return predictions


def detection_loss_func(detection_result, label, opt):
    s = torch.sum(
        torch.mul(label, torch.log(detection_result)) + torch.mul((1 - label),
                                                                  torch.log((1 - detection_result))))
    return torch.div(-s, 14.0 * opt.batch_size)
