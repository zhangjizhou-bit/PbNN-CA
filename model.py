# #!/usr/bin/env python
# # -*- encoding: utf-8 -*-
#
# 文件 : model.py
# 说明 : 网络结构
# 时间 : 2021/03/29
# 作者 : 张继洲

import numpy as np
import torch
from torch import nn
from scipy.io import loadmat, savemat

# 设备参数
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class FPM_estimator(nn.Module):
    def __init__(self, args):
        super(FPM_estimator, self).__init__()
        # --------system parameters-----------------------------------------------------
        pixel_size = args.pixel_size
        magnification = args.magnification
        NA_obj = args.NA_obj
        wavelength = args.wavelength
        LED_spacing = args.LED_spacing
        illumination_distance = args.illumination_distance
        LED_num_side = args.LED_num_side
        upsample_ratio = args.upsample_ratio

        # --------calculate intermediate parameters-----------------------------------------------
        uniform_px = pixel_size / magnification
        LED_num_center = (LED_num_side - 1) / 2
        LED_x = (np.arange(LED_num_side, dtype=np.float32) - LED_num_center) * LED_spacing
        LED_y = (np.arange(LED_num_side, dtype=np.float32) - LED_num_center) * LED_spacing
        LED_x, LED_y = np.meshgrid(LED_x, LED_y)
        LED_num = LED_num_side * LED_num_side
        LED_x = LED_x.reshape((LED_num, 1))
        LED_y = LED_y.reshape((LED_num, 1))
        distances = np.sqrt(np.square(LED_x) + np.square(LED_y) + np.square(illumination_distance))
        sin_theta_y = LED_y / distances
        sin_theta_x = LED_x / distances
        kiy = sin_theta_y / wavelength
        kix = sin_theta_x / wavelength

        # --------calculate coordinate parameters-----------------------------------------------
        Hei = args.Hei
        Wid = args.Wid
        hei = Hei // upsample_ratio
        wid = Wid // upsample_ratio
        deltaFy = 1 / (hei * uniform_px)
        deltaFx = 1 / (wid * uniform_px)
        deltaF = np.min([deltaFx, deltaFy])
        pupil_radius = np.round(NA_obj / wavelength / deltaF)
        y = np.arange(-np.fix(wid/2), np.ceil(wid/2))
        x = np.arange(-np.fix(hei/2), np.ceil(hei/2))
        x, y = np.meshgrid(x, y)
        radius = np.sqrt(x*x + y*y)
        pupil = np.single(radius <= pupil_radius)
        shift_fy = kiy / deltaF
        shift_fx = kix / deltaF
        center_y = np.ceil((Hei-1)/2)
        center_x = np.ceil((Wid-1)/2)
        LED_fy = shift_fy + center_y
        LED_fx = shift_fx + center_x

        self.hei = torch.tensor(hei)
        self.wid = torch.tensor(wid)
        self.LED_num = torch.tensor(LED_num)
        self.upsample_ratio = torch.tensor(upsample_ratio)
        self.LED_fy = torch.from_numpy(LED_fy)
        self.LED_fx = torch.from_numpy(LED_fx)
        self.pupil_flat = torch.from_numpy(pupil).to(device)

        # --------object field-----------------------------------------------------
        self.args = args
        if args.mode == "simulator":
            self.est_field = args.gtr_field_tensor
            self.pupil_real = torch.from_numpy(args.pupil_real).to(device)
            self.illumination_distance = illumination_distance
            self.distances = distances
        elif args.mode == "estimator":
            imaged_intens0 = args.imaged_intens_tensor[(LED_num - 1) // 2, :, :].unsqueeze(0).unsqueeze(0)
            imaged_intens0_interp = nn.functional.interpolate(imaged_intens0, size=(args.Hei, args.Wid), mode='bilinear',
                                                              align_corners=False).squeeze(0).squeeze(0)
            self.est_field = nn.Parameter(imaged_intens0_interp * torch.exp(1j * torch.zeros_like(imaged_intens0_interp)))

            if self.args.channel_estimator == 'CA':
                self.CAM = ChannelAttention(LED_num).to(device)
            if self.args.pupil_estimator == 'CA':
                zernike_pupils_tensor = torch.from_numpy(args.zernike_pupils).to(device)
                self.PE = PupilEstimate_CA(zernike_pupils_tensor).to(device)
            elif self.args.pupil_estimator == 'CA_com':
                zernike_pupils_tensor = torch.from_numpy(args.zernike_pupils).to(device)
                self.PE = PupilEstimate_CA_complex(zernike_pupils_tensor).to(device)
            elif self.args.pupil_estimator == 'layer':
                zernike_pupils_tensor = torch.from_numpy(args.zernike_pupils).to(device)
                self.PE = PupilEstimate_layer(zernike_pupils_tensor).to(device)

    def forward(self):
        if self.args.mode == "simulator":
            self.pupil = self.pupil_real
        elif self.args.mode == "estimator":
            if self.args.pupil_estimator != 'None':
                self.pupil = self.PE()
            else:
                self.pupil = self.pupil_flat
        # --------estimate intensity images-----------------------------------------------
        gtr_spec = torch.fft.fftshift(torch.fft.fft2(self.est_field)).unsqueeze(0)
        gtr_spec_selected = torch.zeros([self.LED_num, self.hei, self.wid], dtype=torch.complex64).to(device)
        for i in range(self.LED_num):
            top = (torch.round(self.LED_fy[i] - torch.fix(self.hei / 2))).int()
            left = (torch.round(self.LED_fx[i] - torch.fix(self.wid / 2))).int()
            gtr_spec_selected[i, :, :] = gtr_spec[0, top:top + self.hei, left:left + self.wid]

        abbr_spectrums = gtr_spec_selected * self.pupil / self.upsample_ratio ** 2
        object_LR = torch.fft.ifft2(torch.fft.ifftshift(abbr_spectrums, dim=(1, 2)))
        est_intens = torch.square(torch.abs(object_LR))
        # est_intens = torch.abs(object_LR)

        if self.args.mode == "estimator":
            if self.args.channel_estimator == 'CA':
                self.channel_weights = self.CAM()
                est_intens = est_intens * self.channel_weights
            elif self.args.channel_estimator == 'cos4':
                calibrate_weights = loadmat('cos4_weights.mat')['intensity_weights']
                self.channel_weights = torch.from_numpy(calibrate_weights).to(device)
                est_intens = est_intens * self.channel_weights

        return est_intens



class FPM_estimator_zheng(nn.Module):
    def __init__(self, args):
        super(FPM_estimator_zheng, self).__init__()
        # --------system parameters-----------------------------------------------------
        pixel_size = args.pixel_size
        magnification = args.magnification
        NA_obj = args.NA_obj
        wavelength = args.wavelength
        LED_spacing = args.LED_spacing
        illumination_distance = args.illumination_distance
        LED_num_side = args.LED_num_side
        upsample_ratio = args.upsample_ratio

        # --------calculate intermediate parameters-----------------------------------------------
        uniform_px = pixel_size / magnification
        LED_num_center = (LED_num_side - 1) / 2
        LED_x = (np.arange(LED_num_side, dtype=np.float32) - LED_num_center) * LED_spacing
        LED_y = (np.arange(LED_num_side, dtype=np.float32) - LED_num_center) * LED_spacing
        LED_x, LED_y = np.meshgrid(LED_x, LED_y)
        LED_num = LED_num_side * LED_num_side
        LED_x = LED_x.reshape((LED_num, 1))
        LED_y = LED_y.reshape((LED_num, 1))
        LED_x = LED_x + 130
        LED_y = LED_y + 300
        distances = np.sqrt(np.square(LED_x) + np.square(LED_y) + np.square(illumination_distance))
        sin_theta_y = LED_y / distances
        sin_theta_x = LED_x / distances
        kiy = sin_theta_y / wavelength
        kix = sin_theta_x / wavelength

        # --------calculate coordinate parameters-----------------------------------------------
        Hei = args.Hei
        Wid = args.Wid
        hei = Hei // upsample_ratio
        wid = Wid // upsample_ratio
        deltaFy = 1 / (hei * uniform_px)
        deltaFx = 1 / (wid * uniform_px)
        deltaF = np.min([deltaFx, deltaFy])
        pupil_radius = np.round(NA_obj / wavelength / deltaF)
        y = np.arange(-np.fix(wid/2), np.ceil(wid/2))
        x = np.arange(-np.fix(hei/2), np.ceil(hei/2))
        x, y = np.meshgrid(x, y)
        radius = np.sqrt(x*x + y*y)
        pupil = np.single(radius <= pupil_radius)
        shift_fy = kiy / deltaF
        shift_fx = kix / deltaF
        center_y = np.ceil((Hei-1)/2)
        center_x = np.ceil((Wid-1)/2)
        LED_fy = shift_fy + center_y
        LED_fx = shift_fx + center_x

        self.hei = torch.tensor(hei)
        self.wid = torch.tensor(wid)
        self.LED_num = torch.tensor(LED_num)
        self.upsample_ratio = torch.tensor(upsample_ratio)
        self.LED_fy = torch.from_numpy(LED_fy)
        self.LED_fx = torch.from_numpy(LED_fx)
        self.pupil_flat = torch.from_numpy(pupil).to(device)

        # --------object field-----------------------------------------------------
        self.args = args
        if args.mode == "simulator":
            self.est_field = args.gtr_field_tensor
            self.pupil_real = torch.from_numpy(args.pupil_real).to(device)
            self.illumination_distance = illumination_distance
            self.distances = distances
        elif args.mode == "estimator":
            imaged_intens0 = args.imaged_intens_tensor[(LED_num - 1) // 2, :, :].unsqueeze(0).unsqueeze(0)
            imaged_intens0_interp = nn.functional.interpolate(imaged_intens0, size=(args.Hei, args.Wid), mode='bilinear',
                                                              align_corners=False).squeeze(0).squeeze(0)
            self.est_field = nn.Parameter(imaged_intens0_interp * torch.exp(1j * torch.zeros_like(imaged_intens0_interp)))

            if self.args.channel_estimator == 'CA':
                self.CAM = ChannelAttention(LED_num).to(device)
            if self.args.pupil_estimator == 'CA':
                zernike_pupils_tensor = torch.from_numpy(args.zernike_pupils).to(device)
                self.PE = PupilEstimate_CA(zernike_pupils_tensor).to(device)
            elif self.args.pupil_estimator == 'CA_com':
                zernike_pupils_tensor = torch.from_numpy(args.zernike_pupils).to(device)
                self.PE = PupilEstimate_CA_complex(zernike_pupils_tensor).to(device)
            elif self.args.pupil_estimator == 'layer':
                zernike_pupils_tensor = torch.from_numpy(args.zernike_pupils).to(device)
                self.PE = PupilEstimate_layer(zernike_pupils_tensor).to(device)

    def forward(self):
        if self.args.mode == "simulator":
            self.pupil = self.pupil_real
        elif self.args.mode == "estimator":
            if self.args.pupil_estimator != 'None':
                self.pupil = self.PE()
            else:
                self.pupil = self.pupil_flat
        # --------estimate intensity images-----------------------------------------------
        gtr_spec = torch.fft.fftshift(torch.fft.fft2(self.est_field)).unsqueeze(0)
        gtr_spec_selected = torch.zeros([self.LED_num, self.hei, self.wid], dtype=torch.complex64).to(device)
        for i in range(self.LED_num):
            top = (torch.round(self.LED_fy[i] - torch.fix(self.hei / 2))).int()
            left = (torch.round(self.LED_fx[i] - torch.fix(self.wid / 2))).int()
            gtr_spec_selected[i, :, :] = gtr_spec[0, top:top + self.hei, left:left + self.wid]

        abbr_spectrums = gtr_spec_selected * self.pupil / self.upsample_ratio ** 2
        object_LR = torch.fft.ifft2(torch.fft.ifftshift(abbr_spectrums, dim=(1, 2)))
        est_intens = torch.square(torch.abs(object_LR))

        if self.args.mode == "estimator":
            if self.args.channel_estimator == 'CA':
                self.channel_weights = self.CAM()
                est_intens = est_intens * self.channel_weights
            elif self.args.channel_estimator == 'cos4':
                calibrate_weights = loadmat('cos4_weights.mat')['intensity_weights']
                self.channel_weights = torch.from_numpy(calibrate_weights).to(device)
                est_intens = est_intens * self.channel_weights

        return est_intens




class PupilEstimate_CA(nn.Module):
    def __init__(self, zernike_pupils):
        super(PupilEstimate_CA, self).__init__()

        coeff = np.zeros((15, 1, 1), dtype=np.float32)
        self.phase_coeff = nn.Parameter(torch.from_numpy(coeff).to(device))
        # self.act = nn.Sigmoid()
        self.act = nn.Tanh()
        self.zernike_pupils = zernike_pupils

    def forward(self):
        pupil_ampli = self.zernike_pupils[0, :, :]
        pupil_phase = torch.sum(self.zernike_pupils * self.act(self.phase_coeff), dim=0)
        pupil = pupil_ampli * torch.exp(1j * pupil_phase)

        return pupil


class PupilEstimate_CA_complex(nn.Module):
    def __init__(self, zernike_pupils):
        super(PupilEstimate_CA_complex, self).__init__()

        coeff = np.zeros((15, 1, 1), dtype=np.float32)
        self.phase_coeff = nn.Parameter(torch.from_numpy(coeff).to(device))
        coeff[0,0,0] = 1
        self.ampli_coeff = nn.Parameter(torch.from_numpy(coeff).to(device))
        self.act = nn.Tanh()
        self.zernike_pupils = zernike_pupils

    def forward(self):
        pupil_ampli = torch.sum(self.zernike_pupils * self.act(self.ampli_coeff), dim=0)
        pupil_phase = torch.sum(self.zernike_pupils * self.act(self.phase_coeff), dim=0)
        pupil = pupil_ampli * torch.exp(1j * pupil_phase)

        return pupil


class PupilEstimate_layer(nn.Module):
    def __init__(self, zernike_pupils):
        super(PupilEstimate_layer, self).__init__()

        self.pupil_ampli = zernike_pupils[0, :, :].to(device)
        self.pupil_phase = nn.Parameter(torch.zeros_like(zernike_pupils[0, :, :]).to(device))

    def forward(self):
        pupil = self.pupil_ampli * torch.exp(1j * self.pupil_phase)
        return pupil


class ChannelAttention(nn.Module):
    def __init__(self, LED_num):
        super(ChannelAttention, self).__init__()

        self.trainable_weights = nn.Parameter(torch.ones(LED_num, 1, 1))
        # self.act = nn.Sigmoid()
        self.act = nn.Tanh()

    def forward(self):
        out = self.act(self.trainable_weights)
        # out = self.trainable_weights
        return out