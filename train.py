# -*- coding: UTF-8 -*-
from __future__ import print_function
import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from models.DarkNet import DarkNet
import torch.nn.init as init
import utils
from datasets import ImageData
from tqdm import tqdm
from PIL import Image, ImageDraw
import torch.nn.functional as F
from torchvision import transforms

class Denormalize(object):
    def __init__(self, mean, std, inplace=False):
        self.mean = mean
        self.demean = [-m/s for m, s in zip(mean, std)]
        self.std = std
        self.destd = [1/s for s in std]
        self.inplace = inplace

    def __call__(self, tensor):

        tensor = transforms.Normalize(tensor, self.demean, self.destd)
        print(tensor)
        # clamp to get rid of numerical errors
        return torch.clamp(tensor, 0.0, 1.0)

def denormalize(x, mean, std):
    # 3, H, W, B
    ten = x.clone().permute(1, 2, 3, 0)
    for t, m, s in zip(ten, mean, std):
        t.mul_(s).add_(m)
    # B, 3, H, W
    return torch.clamp(ten, 0, 1).permute(3, 0, 1, 2)

parser = argparse.ArgumentParser(description='train hed model')
parser.add_argument('--batchSize', type=int, default=1, help='with batchSize=1 equivalent to instance normalization.')
parser.add_argument('--niter', type=int, default=300, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate, default=0.0002')
parser.add_argument('--lr_decay', type=float, help='learning rate decay', default=0.1)
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--outf', default='checkpoints/', help='folder to output images and model checkpoints')
parser.add_argument("--positives", "-p",
                    type=str,
                    help='String Value - The path to the folder containing only positive examples.',
                    )
parser.add_argument("--negatives", "-n",
                    type=str,
                    help='String Value - The path to the folder containing only negative examples i.e. healthy skin.',
                    )

opt = parser.parse_args()
print(opt)

device = torch.device('cuda')

try:
    os.makedirs(opt.outf)
except OSError:
    pass

lr_decay_epoch = {100, 200}
lr_decay = opt.lr_decay
mean=[0.485, 0.456, 0.406]
std=[0.229, 0.224, 0.225]

mask = [0, 1, 2]
anchors = [10,13,  16,30,  33,23,  30,61,  62,45,  59,119,  116,90,  156,198,  373,326]
mask1 = [6, 7, 8]
mask2 = [3, 4, 5]
mask3 = [0, 1, 2]
anchors = [(anchors[i], anchors[i+1]) for i in range(0, len(anchors),2)]
anchors1 = [anchors[i] for i in mask1] # 52x52
anchors2 = [anchors[i] for i in mask2] # 26x26
anchors3 = [anchors[i] for i in mask3] # 13x13
anchors = [anchors1, anchors2, anchors3]


IMAGE_X = 416
IMAGE_Y = 416

print (anchors)

##########   DATASET   ###########
transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean,std)
            ])


train_files_n, val_files_n, test_files_n = utils.get_directory(opt.positives, 0.1, 0.1)
train_files_p, val_files_p, test_files_p = utils.get_directory(opt.negatives, 0.1, 0.1)

train_labels_n = len(train_files_n) * [0]
val_labels_n = len(val_files_n) * [0]
test_labels_n = len(test_files_n) * [0]
train_labels_p = len(train_files_p) * [1]
val_labels_p = len(val_files_p) * [1]
test_labels_p = len(test_files_p) * [1]

batch_size = 2
num_workers = 4

train_dataset_p = ImageData(train_files_p, train_labels_p,
                            transform=utils.get_train_transform_isic((IMAGE_X, IMAGE_Y)))
val_dataset_p = ImageData(val_files_p, val_labels_p,
                          transform=utils.get_train_transform_isic((IMAGE_X, IMAGE_Y)))
test_dataset_p = ImageData(test_files_p, test_labels_p,
                           transform=utils.get_test_transform_isic((IMAGE_X, IMAGE_Y)))
train_data_loader_p = torch.utils.data.DataLoader(train_dataset_p, batch_size=batch_size, shuffle=True,
                                                  num_workers=num_workers)
val_data_loader_p = torch.utils.data.DataLoader(val_dataset_p, batch_size=batch_size, shuffle=False,
                                                num_workers=num_workers)
test_data_loader_p = torch.utils.data.DataLoader(test_dataset_p, batch_size=1, shuffle=False,
                                                 num_workers=num_workers)

train_dataset_n = ImageData(train_files_n, train_labels_n,
                            transform=utils.get_train_transform_isic((IMAGE_X, IMAGE_Y)))
val_dataset_n = ImageData(val_files_n, val_labels_n, transform=utils.get_train_transform_isic((IMAGE_X, IMAGE_Y)))
test_dataset_n = ImageData(test_files_n, test_labels_n,
                           transform=utils.get_train_transform_isic((IMAGE_X, IMAGE_Y)))
train_data_loader_n = torch.utils.data.DataLoader(train_dataset_n, batch_size=batch_size, shuffle=True,
                                                  num_workers=num_workers)
val_data_loader_n = torch.utils.data.DataLoader(val_dataset_n, batch_size=batch_size, shuffle=False,
                                                num_workers=num_workers)
test_data_loader_n = torch.utils.data.DataLoader(test_dataset_n, batch_size=1, shuffle=False,
                                                 num_workers=num_workers)

net = DarkNet(anchors, 1)

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        init.xavier_uniform_(m.weight.data)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

net.apply(weights_init)
#net.load_state_dict(torch.load('checkpoints/darknet_50.pth'))
net.to(device)

lr = opt.lr
optimizer = torch.optim.Adam(net.parameters(),lr=lr, betas=(opt.beta1, 0.999))

########### Training   ###########
loss_step = 0
step = 0
net.train()


for epoch in range(1,opt.niter+1):

    total = min(len(train_data_loader_p), len(train_data_loader_n))
    p_bar = tqdm(zip(train_data_loader_p, train_data_loader_n), total=total,
                 desc=f"Training epoch {epoch}")
    for i, ((images_isic, labels_isic, file_isic), (images_sd, labels_sd, file_sd)) in enumerate(p_bar):
        images = torch.cat([images_isic, images_sd])
        one_hot_targets_isic = torch.from_numpy(np.eye(2)[labels_isic.reshape(-1)])
        one_hot_targets_sd = torch.from_numpy(np.eye(2)[labels_sd.reshape(-1)])
        gt_classes = torch.concat([one_hot_targets_isic, one_hot_targets_sd])
        gt_classes = gt_classes[:,None,:]
        gt_boxes_isic = torch.from_numpy(np.repeat(np.array([[[0.5,0.5,1,1]]]), len(images_isic), axis=0))
        gt_boxes_sd = torch.from_numpy(np.repeat(np.array([[[0.0, 0.0, 0, 0]]]), len(images_isic), axis=0))
        gt_boxes = torch.cat([gt_boxes_isic, gt_boxes_sd])


        images = images.to(device)
        gt_boxes = gt_boxes.to(device)
        gt_classes = gt_classes.to(device)
        # print ('boxes', gt_boxes)
        # print ('classes', gt_classes)
        loss_13, loss_26, loss_52 = net(images, gt_boxes, gt_classes)
        loss = loss_13 + loss_26 + loss_52
        loss.backward()
        optimizer.step()

        loss_step += loss.data.sum()
        step += 1
        loss_show = loss_step / float(step)

        if(i % 10 == 0):
            print('[%d/%d][%d/%d] loss_show: %.4f, Loss_13: %.4f, Loss_26: %.4f, Loss_52: %.4f, lr= %.g'
                      % (epoch, opt.niter, i, total,
                         loss_show, loss_13.data.sum(), loss_26.data.sum(), loss_52.data.sum(), lr))
            loss_step = 0
            step = 0
        # if(i % 1000 == 0):
        #     vutils.save_image(final_output_F.data,
        #                'tmp/samples_i_%d_%03d.png' % (epoch, i),
        #                normalize=True)

    net.eval()
    total = min(len(val_data_loader_p), len(val_data_loader_n))
    de_transform = Denormalize(mean=(0.5,0.5,0.5), std=(0.5, 0.5, 0.5))
    p_bar = tqdm(zip(val_data_loader_p, val_data_loader_n), total=total,
                 desc=f"Validating epoch {epoch}")
    filename = 0
    for i, ((images_isic, labels_isic, file_isic), (images_sd, labels_sd, file_sd)) in enumerate(p_bar):
        images = torch.cat([images_isic, images_sd])
        images = images.to(device)
        detections = net(images, training = False)
        images = denormalize(images, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        for img, d in zip(images, detections):

            d = d.detach().cpu().numpy()
            ind = np.argmax(d[:,4])
            d = d[ind]
            x_min = d[0]-d[2]//2
            x_max = d[0]+d[2]//2
            y_min = d[1]-d[2]//2
            y_max = d[1]+d[2]//2

            img = img.detach().cpu().numpy()
            img = np.swapaxes(img, 0, 2)

            img = Image.fromarray(np.uint8(img*256))
            draw = ImageDraw.Draw(img)
            draw.rectangle(((x_min, x_max), (y_min, y_max)), outline="blue", width=5)
            img.save("see/"+str(filename)+".jpg")
            filename += 1



    if epoch in lr_decay_epoch:
                lr *= lr_decay
                optimizer = torch.optim.Adam(net.parameters(),lr=lr, betas=(opt.beta1, 0.999))
    if(epoch % 10 == 0):
        torch.save(net.state_dict(), '%s/darknet_%d.pth' % (opt.outf, epoch))
