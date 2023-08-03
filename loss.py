# #!/usr/bin/env python
# # -*- encoding: utf-8 -*-
#
# 文件 : loss.py
# 说明 : 损失函数定义
# 时间 : 2021/04/27
# 作者 : 张继洲

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
import math


# 设备参数
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

#可简化
class Loss(nn.Module):
    def __init__(self, args):
        super(Loss, self).__init__()
        self.args = args
        self.losses = []
        for loss_parts in args.loss.split('+'):
            weight, loss_type = loss_parts.split('*')
            if loss_type == 'L1':
                loss_function = nn.L1Loss().to(device)
            elif loss_type == 'L2':
                loss_function = nn.MSELoss().to(device)
            elif loss_type == 'FPM':
                loss_function = FPM_loss().to(device)
            elif loss_type == 'Noise':
                loss_function = noise_loss().to(device)
            elif loss_type == 'TV':
                loss_function = TV_loss().to(device)
            elif loss_type == 'TVA':
                loss_function = TV_loss_part().to(device)
            elif loss_type == 'TVP':
                loss_function = TV_loss_part().to(device)

            # if torch.cuda.is_available():
            #     loss_function = nn.DataParallel(loss_function, device_ids=list(range(args.ngpu)))

            self.losses.append({
                'type': loss_type,
                'weight': float(weight),
                'function': loss_function,
                'temp': [],
                'value': []})

        self.losses.append({
            'type': 'Sum',
            'weight': 1,
            'function': None,
            'temp': [],
            'value': []})

    def forward(self, est_inten, gtr_inten, est_field):
        # 计算损失
        loss_sum = 0
        for i, l in enumerate(self.losses):
            if l['weight'] != 0:
                if l['type'] == 'Sum':
                    continue
                elif (l['type'] == 'Noise') or (l['type'] == 'TV'):
                    loss = l['function'](est_field)
                elif l['type'] == 'TVA':
                    loss = l['function'](torch.abs(est_field))
                elif l['type'] == 'TVP':
                    loss = l['function'](torch.angle(est_field))
                else:
                    loss = l['function'](est_inten, gtr_inten)
                effective_loss = l['weight'] * loss
                loss_sum += effective_loss
                l['temp'].append(loss.detach().item())
        self.losses[-1]['temp'].append(loss_sum.detach().item())

        return loss_sum

    def ave_loss(self, n_iter):
        for i, l in enumerate(self.losses):
            l['value'].append( sum(l['temp']) / n_iter )
            l['temp'].clear()

    def plot_loss(self, epoch):
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
        plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
        matplotlib.use('Agg')
        axis = np.linspace(1, epoch, epoch)
        for i, l in enumerate(self.losses):
            label = '{} Loss'.format(l['type'])
            fig = plt.figure()
            plt.title(label)
            plt.plot(axis, np.array(l['value']), label=label)
            plt.legend()
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.grid(True)
            plt.savefig('{}/Loss_{}_plot.pdf'.format(self.args.result_dir, l['type']))
            plt.close(fig)

    def save_data(self, epoch):
        data = np.zeros(shape=(epoch, len(self.losses)))
        label_column = []
        label_row = ['epoch_{:0>4d}'.format(i) for i in range(epoch)]
        for i, l in enumerate(self.losses):
            label = '{} Loss'.format(l['type'])
            label_column.append(label)
            data[:,i] = np.array(l['value'])
        writer = pd.ExcelWriter(self.args.result_dir + '/Loss_data.xlsx')                   # 创建excel表格
        data_df = pd.DataFrame(data, columns=label_column, index=label_row)                 # 数据帧
        data_df.to_excel(writer,'Loss_data',float_format='%.4f')    # float_format 控制精度，将data_df写到表格第一页中
        writer.save()                                               # 保存



class FPM_loss(nn.Module):
    def __init__(self):
        super(FPM_loss, self).__init__()

    def forward(self, est_intens, imaged_intens):

        sum_loss = torch.sum(torch.sum(torch.abs(est_intens-imaged_intens),2),1) / torch.sum(torch.sum(imaged_intens,2),1)
        self.loss = torch.mean(sum_loss)

        return self.loss


class noise_loss(nn.Module):
    def __init__(self):
        super(noise_loss, self).__init__()

        self.filter = torch.nn.Conv2d(in_channels=1,
                                      out_channels=1,
                                      kernel_size=3,
                                      padding=1,
                                      padding_mode='zeros',
                                      bias=False)

        kernel = torch.from_numpy(np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=np.double))
        kernel = kernel.reshape((1, 1, 3, 3))
        self.filter.weight.data = kernel
        self.filter.weight.requires_grad = False


    def forward(self, est_field):

        H, W = est_field.shape

        # est_inten = torch.square(torch.abs(est_field)).unsqueeze(0).unsqueeze(0)
        est_ampli = torch.abs(est_field).unsqueeze(0).unsqueeze(0)
        est_phase = torch.angle(est_field).unsqueeze(0).unsqueeze(0)

        est_ampli_filtered = self.filter(est_ampli)
        est_phase_filtered = self.filter(est_phase)
        # sigma_inten = torch.mean(est_inten_filtered)
        sigma_ampli = torch.sum(torch.sum(torch.sum(torch.abs(est_ampli_filtered).squeeze(0), dim=2), dim=1), dim=0)
        sigma_ampli = sigma_ampli * torch.sqrt(torch.tensor(0.5 * math.pi)) / (6 * (W-2) * (H-2))
        # sigma_phase = torch.mean(est_phase_filtered)
        sigma_phase = torch.sum(torch.sum(torch.sum(torch.abs(est_phase_filtered).squeeze(0), dim=2), dim=1), dim=0)
        sigma_phase = sigma_phase * torch.sqrt(torch.tensor(0.5 * math.pi)) / (6 * (W-2) * (H-2))

        return 0.5*sigma_ampli+0.5*sigma_phase


class TV_loss(nn.Module):
    def __init__(self):
        super(TV_loss, self).__init__()
        # self.weight_inten = 1
        # self.weight_phase = 1
        self.weight_inten = 0.6
        self.weight_phase = 0.4

    def forward(self, est_field):
        est_inten = torch.abs(est_field)
        est_phase = torch.angle(est_field)
        hei, wid = est_inten.size()
        hei_tv_inten = torch.pow((est_inten[1:,:]-est_inten[:-1,:]),2).sum()
        wid_tv_inten = torch.pow((est_inten[:,1:]-est_inten[:,:-1]),2).sum()
        hei_tv_phase = torch.pow((est_phase[1:,:]-est_phase[:-1,:]),2).sum()
        wid_tv_phase = torch.pow((est_phase[:,1:]-est_phase[:,:-1]),2).sum()
        cnt_hei = (hei-1)*wid
        cnt_wid = hei*(wid-1)

        part_i = hei_tv_inten/cnt_hei + wid_tv_inten/cnt_wid
        part_p = hei_tv_phase/cnt_hei + wid_tv_phase/cnt_hei
        result = self.weight_inten * part_i + self.weight_phase * part_p

        return result



class TV_loss_part(nn.Module):
    def __init__(self):
        super(TV_loss_part, self).__init__()

    def forward(self, x):
        hei, wid = x.size()
        hei_tv = torch.pow((x[1:,:]-x[:-1,:]),2).sum()
        wid_tv = torch.pow((x[:,1:]-x[:,:-1]),2).sum()
        cnt_hei = (hei-1)*wid
        cnt_wid = hei*(wid-1)

        result = hei_tv/cnt_hei + wid_tv/cnt_wid

        return result

# class noise_loss(nn.Module):
#     def __init__(self):
#         super(noise_loss, self).__init__()
#         self.pch_size = 8
#
#
#     def im2patch(self, im, pch_size, stride=1):
#
#         if isinstance(pch_size, tuple):
#             pch_H, pch_W = pch_size
#         elif isinstance(pch_size, int):
#             pch_H = pch_W = pch_size
#         else:
#             sys.exit('The input of pch_size must be a integer or a int tuple!')
#
#         if isinstance(stride, tuple):
#             stride_H, stride_W = stride
#         elif isinstance(stride, int):
#             stride_H = stride_W = stride
#         else:
#             sys.exit('The input of stride must be a integer or a int tuple!')
#
#         C, H, W = im.shape
#         num_H = len(range(0, H - pch_H + 1, stride_H))
#         num_W = len(range(0, W - pch_W + 1, stride_W))
#         num_pch = num_H * num_W
#         pch = torch.zeros((C, pch_H * pch_W, num_pch), dtype=torch.double)
#         kk = 0
#         for ii in range(pch_H):
#             for jj in range(pch_W):
#                 temp = im[:, ii:H - pch_H + ii + 1:stride_H, jj:W - pch_W + jj + 1:stride_W]
#                 pch[:, kk, :] = temp.reshape((C, num_pch))
#                 kk += 1
#
#         return pch.reshape((C, pch_H, pch_W, num_pch))
#
#
#     def forward(self, est_field, pch_size=8):
#         est_field = est_field.transpose((2, 0, 1))
#
#         # image to patch
#         pch = self.im2patch(est_field, pch_size, 3)  # C x pch_size x pch_size x num_pch tensor
#         num_pch = pch.shape[3]
#         pch = pch.reshape((-1, num_pch))  # d x num_pch matrix
#         d = pch.shape[0]
#
#         mu = pch.mean(axis=1, keepdims=True)  # d x 1
#         X = pch - mu
#         sigma_X = torch.matmul(X, X.t()) / num_pch
#         sig_value, _ = torch.linalg.eigh(sigma_X)
#         sig_value.sort()
#
#         for ii in range(-1, -d-1, -1):
#             tau = torch.mean(sig_value[:ii])
#             if torch.sum(sig_value[:ii]>tau) == torch.sum(sig_value[:ii] < tau):
#                 return torch.sqrt(tau)
#

