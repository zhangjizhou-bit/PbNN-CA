# #!/usr/bin/env python
# # -*- encoding: utf-8 -*-
#
# 文件 : tienn.py
# 说明 : 神经网络求解复振幅
# 时间 : 2021/03/29
# 作者 : 张继洲


import torch
import numpy as np
from tqdm import tqdm
import os
import time
from scipy.io import loadmat, savemat

from loss import Loss
from utility import Metrics, MeasureWeights
from model import FPM_estimator, FPM_estimator_zheng


os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# 设备参数
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

class FPM_reconstruction(object):

    def __init__(self, args):
        super(FPM_reconstruction, self).__init__()

        self.args = args
        os.makedirs(self.args.result_dir)


    def train_model(self):  # 训练模型
        self.model = FPM_estimator(self.args).to(device)
        # self.model = FPM_estimator_zheng(self.args).to(device)

        if self.args.optimizer == "Adam":
            optimizer_func = torch.optim.Adam
        elif self.args.optimizer == "SGD":
            optimizer_func = torch.optim.SGD

        base_params = [self.model.est_field,]
        params = [{"params": base_params},]
        if self.args.channel_estimator == 'CA':
            cam_params = list(filter(lambda p: id(p), self.model.CAM.parameters()))
            params.append({"params": cam_params, "lr": self.args.lr_cam})
        if self.args.pupil_estimator == 'CA':
            pupil_params = list(filter(lambda p: id(p), self.model.PE.parameters()))
            params.append({"params": pupil_params, "lr": self.args.lr_pupil})
        elif self.args.pupil_estimator == 'layer':
            pupil_params = list(filter(lambda p: id(p), self.model.PE.parameters()))
            params.append({"params": pupil_params, "lr": 1e-3})

        optimizer = optimizer_func(params, lr=self.args.lr)

        # params = filter(lambda p: p.requires_grad, self.model.parameters())
        # optimizer = optimizer_func(params, lr=self.args.lr)

        self.loss_function = Loss(self.args)

        n_iter = 1
        start_time = time.time()
        # 开始逐轮训练
        for epoch in tqdm(range(1, self.args.epochs + 1), ncols=100, desc='training:'):
            # 前向传播
            est_intens_tensor = self.model()
            # 计算损失
            loss_sum = self.loss_function(est_intens_tensor, self.args.imaged_intens_tensor, self.model.est_field)
            # 后向传播
            optimizer.zero_grad()
            loss_sum.backward(retain_graph=True)
            # 更新模型
            optimizer.step()
            # scheduler.step()
            self.loss_function.ave_loss(n_iter)

        end_time = time.time()
        self.time_consume = end_time - start_time


    def eval_model(self):
        self.loss_function.plot_loss(self.args.epochs)
        self.loss_function.save_data(self.args.epochs)

        self.model.eval()
        _ = self.model()
        est_field = self.model.est_field.detach().cpu().numpy()

        self.args.est_ampli = np.abs(est_field)
        self.args.est_phase = np.angle(est_field)

        if self.args.channel_estimator != 'None':
            size = int(np.sqrt(self.model.channel_weights.size(0)))
            self.args.channel_weights = self.model.channel_weights.detach().cpu().numpy()[:, 0, 0].reshape((size,size))
        if self.args.pupil_estimator != 'None':
            pupil_estimate = self.model.pupil.detach().cpu().numpy()
            self.args.pupil_phase_estimate = np.angle(pupil_estimate)
            if self.args.pupil_estimator == 'CA':
                self.args.phase_coeff = self.model.PE.phase_coeff.detach().cpu().numpy()
