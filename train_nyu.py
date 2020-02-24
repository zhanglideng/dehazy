# -*- coding: utf-8 -*-
import sys

sys.path.append('/home/aistudio/external-libraries')

import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import numpy as np
from utils.loss import *
from utils.print_time import *
from utils.save_log_to_excel import *
from dataloader_nyu import EdDataSet
from Res_ED_model import *
import time
import xlwt
from utils.ms_ssim import *
import os

ednet_LR = 0.004  # 学习率
Atnet_LR = 0.0004  # 学习率
EPOCH = 20  # 轮次
BATCH_SIZE = 2  # 批大小
excel_train_line = 1  # train_excel写入的行的下标
excel_val_line = 1  # val_excel写入的行的下标
alpha = 1  # 损失函数的权重
accumulation_steps = 2  # 梯度积累的次数，类似于batch-size=64
itr_to_lr = 10000 // BATCH_SIZE  # 训练10000次后损失下降50%
itr_to_excel = 64 // BATCH_SIZE  # 训练64次后保存相关数据到excel
# 由于loss数量过多，建议使用分步训练以降低显存占用。
loss_num = 12  # loss的数量。重建损失2个,At网络5个,去雾损失（正向反向）,中间特征约束。
weight_At = [1, 1, 1, 1, 1]
weight_ed = [1, 1, 1, 1, 0.01]
weight_recon = [1, 1]
train_haze_path = '/input/data/nyu/train/'  # 去雾训练集的路径
val_haze_path = '/input/data/nyu/val/'  # 去雾验证集的路径
gt_path = '/input/data/nyu/gth/'
t_path = '/input/data/nyu/depth/'
save_path = './result_nyu_' + time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()) + '/'
save_model_ed_name = save_path + 'ed_model.pt'  # 保存模型的路径
save_model_At_name = save_path + 'At_model.pt'  # 保存模型的路径
excel_save = save_path + 'result.xls'  # 保存excel的路径
mid_save_ed_path = './mid_model/ednet_model.pt'  # 保存的中间模型，用于下一步ntire数据的训练。
mid_save_At_path = './mid_model/Atnet_model.pt'
# 初始化excel
f, sheet_train, sheet_val = init_excel()
# 加载模型
ednet_path = './pre_model/ednet_model.pt'
Atnet_path = './pre_model/Atnet_model.pt'
ednet = torch.load(ednet_path)
Atnet = torch.load(Atnet_path)
ednet = ednet.cuda()
Atnet = Atnet.cuda()
# print(ednet)
# print(Atnet)
for param in ednet.decoder.parameters():
    param.requires_grad = False

# 数据转换模式
transform = transforms.Compose([transforms.ToTensor()])
# 读取训练集数据
train_path_list = [train_haze_path, gt_path, d_path]
train_data = nyu_DataSet(transform, train_path_list)
train_data_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

# 读取验证集数据
val_path_list = [val_haze_path, gt_path, d_path]
val_data = nyu_DataSet(transform, val_path_list)
val_data_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

# 定义优化器
ednet_optim = torch.optim.Adam(ednet.parameters(), lr=ednet_LR, weight_decay=1e-5)
Atnet_optim = torch.optim.Adam(Atnet.parameters(), lr=Atnet_LR, weight_decay=1e-5)

min_loss = 999999999
min_epoch = 0
itr = 0  # 记录一共训练了多少个batch
start_time = time.time()

# 开始训练
print("\nstart to train!")
for epoch in range(EPOCH):
    index = 0
    loss = 0
    loss_excel = [0] * loss_num
    for haze, gt, A_haze, A_gt, t_haze, t_gt in train_data_loader:
        index += 1
        itr += 1
        J, A, t = Atnet(haze)
        output, gt_scene, I = ednet(gt, A_gt, t_gt)
        dehaze, hazy_scene, I = ednet(haze, A, t)
        # 分批计算loss，以防现存溢出。
        loss_image = [gt, A_haze, t_haze, J, A, t]
        loss, temp_loss = loss_At_function(loss_image, weight_At)
        loss_excel = [loss_excel[i] + temp_loss[i] for i in range(len(loss_excel))]
        loss = loss / accumulation_steps
        loss.backward()

        loss_image = [output, gt, dehaze, gt_scene, hazy_scene]
        loss, temp_loss = loss_ed_function(loss_image, weight_ed)
        loss_excel = [loss_excel[i + 5] + temp_loss[i] for i in range(len(loss_excel))]
        loss = loss / accumulation_steps
        loss.backward()

        loss_image = [haze, I]
        loss, temp_loss = loss_recon_function(loss_image, weight_recon)
        loss_excel = [loss_excel[i + 10] + temp_loss[i] for i in range(len(loss_excel))]
        loss = loss / accumulation_steps
        loss.backward()

        # 3. update parameters of net
        if ((index + 1) % accumulation_steps) == 0:
            # optimizer the net
            optimizer.step()  # update parameters of net
            optimizer.zero_grad()  # reset gradient
        if np.mod(index, itr_to_excel) == 0:
            loss_excel = [loss_excel[i] / itr_to_excel for i in range(len(loss_excel))]
            print('epoch %d, %03d/%d, loss=%.5f' % (epoch + 1, index, len(train_data_loader), sum(loss_excel)))
            print_time(start_time, index, EPOCH, len(train_data_loader), epoch)
            excel_train_line = write_excel(sheet=sheet_train,
                                           data_type='train',
                                           line=excel_train_line,
                                           epoch=epoch,
                                           itr=itr,
                                           loss=loss_excel)
            f.save(excel_save)
            loss_excel = [0] * loss_num
    optimizer.step()
    optimizer.zero_grad()
    loss_excel = [0] * loss_num
    with torch.no_grad():
        for haze, gt, A_haze, A_gt, t_haze, t_gt in val_data_loader:
            J, A, t = Atnet(haze)
            output, gt_scene, I = ednet(gt, A_gt, t_gt)
            dehaze, hazy_scene, I = ednet(haze, A_haze, t_haze)

            loss_image = [gt, A_haze, t_haze, J, A, t]
            loss, temp_loss = loss_At_function(loss_image, weight_At)
            loss_excel = [loss_excel[i] + temp_loss[i] for i in range(len(loss_excel))]

            loss_image = [output, gt, dehaze, gt_scene, hazy_scene]
            loss, temp_loss = loss_ed_function(loss_image, weight_ed)
            loss_excel = [loss_excel[i + 5] + temp_loss[i] for i in range(len(loss_excel))]

            loss_image = [haze, I]
            loss, temp_loss = loss_recon_function(loss_image, weight_recon)
            loss_excel = [loss_excel[i + 10] + temp_loss[i] for i in range(len(loss_excel))]

    val_epoch_loss = sum(loss_excel)
    loss_excel = [loss_excel[i] / len(val_data_loader) for i in range(len(loss_excel))]
    print('val_epoch_loss = %.5f' % val_epoch_loss)
    excel_val_line = write_excel(sheet=sheet_val,
                                 data_type='val',
                                 line=excel_val_line,
                                 epoch=epoch,
                                 itr=False,
                                 loss=loss_excel)
    f.save(excel_save)
    if val_epoch_loss < min_loss:
        min_loss = val_epoch_loss
        min_epoch = epoch
        torch.save(ednet, save_model_ed_name)
        torch.save(Atnet, save_model_At_name)
        torch.save(ednet, mid_save_ed_path)
        torch.save(Atnet, mid_save_At_path)
        print('saving the epoch %d model with %.5f' % (epoch + 1, min_loss))
print('Train is Done!')
