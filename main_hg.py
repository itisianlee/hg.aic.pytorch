# coding:utf8
from os.path import join
from config import opt
import models
from data import HPEBaseTransform
from data import hgDataset, hgValDataset
import torch
import fire
from torch import nn
from torch.utils import data
from torch.autograd import Variable
from tensorboardX import SummaryWriter
import os
import numpy as np
from tqdm import tqdm
from utils.ian_eval import compute_batch_oks, compute_batch_oks_gpu
from utils.logger import Logger
import json
import time


def main(**kwargs):
    writer = SummaryWriter(opt.logs_path, comment=opt.env)  # tensorboard
    opt.title = 'AIC-HGNet'

    use_gpu = torch.cuda.is_available()
    opt.model_id = 4
    opt.parse(kwargs)
    model = getattr(models, opt.model[opt.model_id])(num_stacks=4)
    if use_gpu:
        torch.cuda.set_device(opt.device)
        model = model.cuda()
    if opt.resume:
        logger = Logger(join(opt.logs_path, '{}_log.txt'.format(opt.title)), title=opt.title, resume=True)  # 生成日志文件
        with open(opt.checkpoints + '{}_progress.json'.format(opt.title), 'r') as f:
            opt.progress = json.load(f)
        model.load(opt.progress['best_path'], opt.device)
        opt.lr = opt.progress['lr']
    else:
        with open(opt.checkpoints + '{}_progress.json'.format(opt.title), 'w') as f:
            json.dump(opt.progress, f)
        opt.progress['lr'] = opt.lr
        logger = Logger(join(opt.logs_path, '{}_log.txt'.format(opt.title)), title=opt.title)
        logger.set_names(['Epoch', '--Time', '--TrainLoss', '--TrainmAP', '--ValmAP'])

    # 是否使用少量数据跑 Demo
    if opt.demo:
        opt.img_dir = '/home/bnrc2/ai_challenge/ian/hg.aic.pytorch/demo_data/train_images/'
        opt.annotations_file = '/home/bnrc2/ai_challenge/ian/hg.aic.pytorch/demo_data/' \
                               'keypoint_train_annotations_20170909.json'
        opt.val_img_dir = '/home/bnrc2/ai_challenge/ian/hg.aic.pytorch/demo_data/validation_images/'
        opt.val_anno_file = '/home/bnrc2/ai_challenge/ian/hg.aic.pytorch/demo_data/' \
                            'keypoint_validation_annotations_20170911.json'
    opt.config_info_print()
    # 数据集
    trainset = hgDataset(opt.annotations_file, opt.img_dir, HPEBaseTransform(opt.train_mean, hm_side=64.0))
    valset = hgValDataset(opt.val_anno_file, opt.val_img_dir)
    valloader = data.DataLoader(valset, batch_size=opt.val_bs, num_workers=opt.num_workers)

    optimizer = model.get_optimizer()
    criterion = nn.MSELoss()

    for epoch in range(opt.progress['epoch'], opt.epoch):
        if epoch in [5, 10, 15, 20]:
            opt.lr *= 0.5
            print(opt.lr)
            model.get_optimizer(opt.lr)
        if epoch in range(5, 200, 5):
            model_path = 'checkpoints/{}_epoch_{}'.format(opt.title, epoch) + '.model'
            torch.save(model.state_dict(), model_path)
        opt.progress['epoch'] = epoch
        opt.progress['count'] = train(model, optimizer, criterion, trainset, valloader, logger, writer)


def train(model, optimizer, criterion, trainset, valloader, logger, writer):
    model.train()
    trainloader = data.DataLoader(trainset, batch_size=opt.batch_size, shuffle=True, num_workers=4)
    stride = int(len(trainset) / opt.batch_size)
    epoch = opt.progress['epoch']
    count = opt.progress['count'] + 1
    start_count = count % stride
    print('bs:', opt.batch_size)
    for i, (img, label, anno) in tqdm(enumerate(trainloader, start_count), ncols=30):
        img, label = Variable(img.float().cuda()), Variable(label.float().cuda())
        optimizer.zero_grad()
        output = model(img)
        # score_map = output[-1].cpu()
        score_map = output[-1]
        loss = criterion(output[0], label)
        for j in range(1, len(output)):
            loss += criterion(output[j], label)
        loss.backward()  # 反向传播
        optimizer.step()  # 参数更新
        if count % opt.plot_every == 0:
            oks_all = compute_batch_oks_gpu(score_map, anno)
            average_precision = []
            for threshold in np.linspace(0.5, 0.95, 10):
                average_precision.append(np.sum(oks_all > threshold) / np.float32(opt.batch_size))
            trainmAP = np.mean(average_precision)
            logger.append([epoch, time.strftime('%m/%d-%H:%M:%S'), (loss.data[0] * 10000), trainmAP, '-'])
            writer.add_scalar('hg_loss', loss.data[0], count)
            writer.add_scalar('train_mAP', trainmAP, count)
        if count % opt.check_every == 0 and count != start_count:
            valmAP = val(model, valloader)
            logger.append([epoch, time.strftime('%m/%d-%H:%M:%S'), (loss.data[0] * 10000), '-', valmAP])
            writer.add_scalar('mAP_value', valmAP, count)
            if opt.progress['best_mAP'] is None or valmAP > opt.progress['best_mAP']:
                opt.progress['best_mAP'] = valmAP
                opt.progress['count'] = count
                opt.progress['lr'] = opt.lr
                if opt.progress['best_path'] is not '':
                    os.system('rm {}'.format(opt.progress['best_path']))  # 删除mAP值低的模型
                best_path = model.save(opt.title + '_' + str(valmAP))
                opt.progress['best_path'] = best_path
                with open(opt.checkpoints + '{}_progress.json'.format(opt.title), 'w') as f:
                    json.dump(opt.progress, f)
                    # tra_loader = data.DataLoader(trainset, batch_size=opt.batch_size, shuffle=True, num_workers=4)
        count += 1
        if i >= stride:
            break
    return count


def val(model, loader):
    """回归子网络验证"""
    model.eval()
    set_len = len(loader.dataset)
    oks_all = np.zeros(0)
    for img, anno in loader:
        img = img.float().cuda()
        img = Variable(img)
        oks_all = np.concatenate((oks_all, compute_batch_oks_gpu(model(img)[-1], anno)), axis=0)
    average_precision = []
    for threshold in np.linspace(0.5, 0.95, 10):
        average_precision.append(np.sum(oks_all > threshold) / np.float32(set_len))
    model.train()
    return np.mean(average_precision)


def exp_lr_scheduler(optimizer, epoch, init_lr=0.001, decay_rate=0.95, lr_decay_epoch=2):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""

    lr = init_lr * (0.95 ** (epoch // lr_decay_epoch))

    if epoch % lr_decay_epoch == 0:
        print('LR is set to {}'.format(lr))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return optimizer


def detection_loss_func(detection_result, label):
    s = torch.sum(
        torch.mul(label, torch.log(detection_result)) + torch.mul((1 - label),
                                                                  torch.log((1 - detection_result))))
    return torch.div(-s, 14.0 * opt.batch_size)


def l2_loss_func(regressiong_result, label):
    s = torch.sum(torch.pow((regressiong_result - label), 2))
    return torch.div(s, 14.0 * opt.batch_size)


if __name__ == "__main__":
    fire.Fire()
