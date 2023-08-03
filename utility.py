# #!/usr/bin/env python
# # -*- encoding: utf-8 -*-
#
# 文件 : utility.py
# 说明 : 各种辅助函数定义
# 时间 : 2021/04/27
# 作者 : 张继洲


import numpy as np
import torch
from skimage import metrics
import matplotlib
import matplotlib.pyplot as plt
import cv2
from scipy.io import loadmat, savemat
from model import FPM_estimator
from parameters import args

# 设备参数
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def MAE(I1, I2):
    mae = np.mean(abs(I1 - I2))
    return mae

def PSNR(I1, I2):
    err = metrics.mean_squared_error(I1, I2)
    data_range = np.max(I1) - np.min(I1)
    return 10 * np.log10((data_range ** 2) / err)

def Metrics(args):

    est_ampli = args.est_ampli
    est_phase = args.est_phase

    if hasattr(args,'gtr_ampli'):
        gtr_ampli = args.gtr_ampli
        gtr_phase = args.gtr_phase

        est_ampli_mean = np.mean(np.mean(est_ampli, axis=0, keepdims=True), axis=1, keepdims=True)
        est_ampli_std = np.std(np.std(est_ampli, axis=0, keepdims=True), axis=1, keepdims=True)
        est_phase_mean = np.mean(np.mean(est_phase, axis=0, keepdims=True), axis=1, keepdims=True)
        est_phase_std = np.std(np.std(est_phase, axis=0, keepdims=True), axis=1, keepdims=True)
        gtr_ampli_mean = np.mean(np.mean(gtr_ampli, axis=0, keepdims=True), axis=1, keepdims=True)
        gtr_ampli_std = np.std(np.std(gtr_ampli, axis=0, keepdims=True), axis=1, keepdims=True)
        gtr_phase_mean = np.mean(np.mean(gtr_phase, axis=0, keepdims=True), axis=1, keepdims=True)
        gtr_phase_std = np.std(np.std(gtr_phase, axis=0, keepdims=True), axis=1, keepdims=True)

        est_ampli_norm = (est_ampli-est_ampli_mean) / est_ampli_std * gtr_ampli_std + gtr_ampli_mean
        est_phase_norm = (est_phase-est_phase_mean) / est_phase_std * gtr_phase_std + gtr_phase_mean
        args.est_ampli_norm = est_ampli_norm
        args.est_phase_norm = est_phase_norm

        mae_ampli = MAE(gtr_ampli, est_ampli_norm)
        psnr_ampli = metrics.peak_signal_noise_ratio(gtr_ampli, est_ampli_norm)
        ssim_ampli = metrics.structural_similarity(gtr_ampli, est_ampli_norm)

        mae_phase = MAE(gtr_phase, est_phase_norm)
        psnr_phase = metrics.peak_signal_noise_ratio(gtr_phase, est_phase_norm)
        ssim_phase = metrics.structural_similarity(gtr_phase, est_phase_norm)

        print('--------------------------------------')
        print('mae_ampli:{:.3e}'.format(mae_ampli))
        print('psnr_ampli:{:.2f}'.format(psnr_ampli))
        print('ssim_ampli:{:.4f}'.format(ssim_ampli))
        print('--------------------------------------')
        print('mae_phase:{:.3e}'.format(mae_phase))
        print('psnr_phase:{:.2f}'.format(psnr_phase))
        print('ssim_phase:{:.4f}'.format(ssim_phase))
        print('--------------------------------------')

        args.mae_ampli = mae_ampli
        args.psnr_ampli = psnr_ampli
        args.ssim_ampli = ssim_ampli
        args.mae_phase = mae_phase
        args.psnr_phase = psnr_phase
        args.ssim_phase = ssim_phase

    else:
        args.est_ampli_norm = (est_ampli-np.min(est_ampli))/(np.max(est_ampli)-np.min(est_ampli))
        args.est_phase_norm = (est_phase-np.min(est_phase))/(np.max(est_phase)-np.min(est_phase))


    mae_pupil_phase = 0
    psnr_pupil_phase = 0
    ssim_pupil_phase = 0
    if hasattr(args,'pupil_phase_estimate') and hasattr(args,'pupil_phase'):
        if args.pupil_phase.shape == args.pupil_phase_estimate.shape:
            est_pupil_phase_mean = np.mean(np.mean(args.pupil_phase_estimate, axis=0, keepdims=True), axis=1, keepdims=True)
            est_pupil_phase_std = np.std(np.std(args.pupil_phase_estimate, axis=0, keepdims=True), axis=1, keepdims=True)
            gtr_pupil_phase_mean = np.mean(np.mean(args.pupil_phase, axis=0, keepdims=True), axis=1, keepdims=True)
            gtr_pupil_phase_std = np.std(np.std(args.pupil_phase, axis=0, keepdims=True), axis=1, keepdims=True)

            est_pupil_phase_norm = (args.pupil_phase_estimate-est_pupil_phase_mean) / est_pupil_phase_std * gtr_pupil_phase_std + gtr_pupil_phase_mean

            args.est_pupil_phase_norm = est_pupil_phase_norm

            mae_pupil_phase = MAE(args.pupil_phase, est_pupil_phase_norm)
            psnr_pupil_phase = PSNR(args.pupil_phase, est_pupil_phase_norm) # 取值范围问题，单独计算
            ssim_pupil_phase = metrics.structural_similarity(args.pupil_phase, est_pupil_phase_norm)

    args.mae_pupil_phase = mae_pupil_phase
    args.psnr_pupil_phase = psnr_pupil_phase
    args.ssim_pupil_phase = ssim_pupil_phase

    if hasattr(args,'channel_weights') and hasattr(args,'intensity_weights'):
        temp = args.intensity_weights[:,0,0].reshape((13,13))
        mae_LED = MAE(temp, args.channel_weights)
        psnr_LED = metrics.peak_signal_noise_ratio(temp, args.channel_weights)
        ssim_LED = metrics.structural_similarity(temp, args.channel_weights)
    else:
        mae_LED = 0
        psnr_LED = 0
        ssim_LED = 0

    args.mae_LED = mae_LED
    args.psnr_LED = psnr_LED
    args.ssim_LED = ssim_LED



def MeasureWeights(est_intens_tensor, imaged_intens_tensor):
    est_intens = est_intens_tensor.detach().cpu().numpy()
    imaged_intens = imaged_intens_tensor.detach().cpu().numpy()

    weights1 = np.mean(np.abs(est_intens-imaged_intens)/imaged_intens, axis=(1,2))
    weights2 = np.mean(np.abs(est_intens-imaged_intens), axis=(1,2))/np.mean(imaged_intens, axis=(1,2))

    return weights1



def ShowResults(args):

    matplotlib.use('TkAgg')
    fig = plt.figure(figsize=(16, 4))

    if hasattr(args,'gtr_ampli'):
        ax = fig.add_subplot(2, 4, 1)
        ax.imshow(args.gtr_ampli, cmap='gray', clim=(0, 1))
        ax.set_title('amplitude (GT)')
        ax = fig.add_subplot(2, 4, 2)
        ax.imshow(args.gtr_phase, cmap='gray', clim=(0, 1))
        ax.set_title('phase (GT)')
        ax = fig.add_subplot(2, 4, 3)
        temp = args.intensity_weights[:,0,0].reshape((13,13))
        ax.imshow(temp, cmap='viridis', clim=(0, 1))
        ax.set_title('channel weights (GT)')
        ax = fig.add_subplot(2, 4, 4)
        pupil_phase_real = args.pupil_phase
        ax.imshow(pupil_phase_real, cmap='gray', clim=(-1, 2))
        ax.set_title('pupil phase (GT)')

        ax = fig.add_subplot(2, 4, 5)
        ax.imshow(args.est_ampli_norm, cmap='gray', clim=(0, 1))
        ax.set_title('amplitude')
        ax = fig.add_subplot(2, 4, 6)
        ax.imshow(args.est_phase_norm, cmap='gray', clim=(0, 1))
        ax.set_title('phase')
        if args.channel_estimator != 'None':
            ax = fig.add_subplot(2, 4, 7)
            ax.imshow(args.channel_weights, cmap='viridis', clim=(0, 1))
            ax.set_title('channel weights')
        if args.pupil_estimator != 'None':
            pupil_phase_est = args.pupil_phase_estimate
            ax = fig.add_subplot(2, 4, 8)
            ax.imshow(pupil_phase_est, cmap='gray', clim=(-1, 2))
            ax.set_title('pupil phase')
    else:
        ax = fig.add_subplot(1, 4, 1)
        # ax.imshow(args.est_inten, cmap='gray', clim=(0, 1))
        ax.imshow(args.est_ampli_norm, cmap='gray', clim=(0, 1))
        ax.set_title('amplitude')
        ax = fig.add_subplot(1, 4, 2)
        ax.imshow(args.est_phase_norm, cmap='gray', clim=(0, 1))
        ax.set_title('phase')
        if args.channel_estimator != 'None':
            ax = fig.add_subplot(1, 4, 3)
            ax.imshow(args.channel_weights, cmap='gray', clim=(0, np.max(args.channel_weights)))
            ax.set_title('channel weights')
        if args.pupil_estimator != 'None':
            # pupil_ampli_est = args.pupil_ampli_estimate
            # ax = fig.add_subplot(1, 5, 4)
            # ax.imshow(pupil_ampli_est, cmap='gray', clim=(np.min(pupil_ampli_est), np.max(pupil_ampli_est)))
            # ax.set_title('pupil_amplitude')
            pupil_phase_est = args.pupil_phase_estimate
            ax = fig.add_subplot(1, 4, 4)
            ax.imshow(pupil_phase_est, cmap='gray', clim=(np.min(pupil_phase_est), np.max(pupil_phase_est)))
            ax.set_title('pupil_phase')

    plt.show()



def generate_cos4_zuo(args):
    gtr_ampli = np.array(cv2.imread('Image/Baboon.bmp', 0).astype(np.float64)) / 255
    gtr_phase = np.array(cv2.imread('Image/Aerial.bmp', 0).astype(np.float64)) / 255
    gtr_field = np.sqrt(gtr_ampli) * np.exp(1j * gtr_phase)
    gtr_field_tensor = torch.from_numpy(gtr_field).to(device)
    args.Hei = gtr_ampli.shape[0]
    args.Wid = gtr_ampli.shape[1]
    args.gtr_ampli = gtr_ampli
    args.gtr_phase = gtr_phase
    args.gtr_field_tensor = gtr_field_tensor

    # simulate pupil aberration
    zernike_pupils = loadmat('generate_zernike_pupils/zernike_pupils_zuo_128.mat')['zernike_pupils'].astype(np.float32).transpose((2, 0, 1))
    pupil_ampli = zernike_pupils[0, :, :]
    args.zernike_coeff = np.zeros((15, 1, 1), dtype=np.float32)
    pupil_phase = np.sum(zernike_pupils * args.zernike_coeff, axis=0)
    pupil = pupil_ampli * np.exp(1j * pupil_phase)
    args.zernike_pupils = zernike_pupils
    args.pupil_real = pupil
    args.pupil_ampli = pupil_ampli
    args.pupil_phase = pupil_phase

    # simulate low resolution intensity
    args.mode = "simulator"
    fpm_estimator = FPM_estimator(args).to(device)
    fpm_estimator.requires_grad = False
    imaged_intens_tensor = fpm_estimator()
    args.ideal_intens_tensor = imaged_intens_tensor

    # simulate intensity weights
    intensity_weights = generate_inten_vara(fpm_estimator, args.channel_variaty)
    savemat('cos4_weights_zuo.mat', {'intensity_weights':intensity_weights})



def simulate_data(args):
    gtr_ampli = np.array(cv2.imread(args.ampli_dir, 0).astype(np.float64)) / 255
    gtr_phase = np.array(cv2.imread(args.phase_dir, 0).astype(np.float64)) / 255
    # gtr_phase = np.array(np.zeros_like(gtr_ampli)) / 255
    gtr_field = np.sqrt(gtr_ampli) * np.exp(1j * gtr_phase)
    gtr_field_tensor = torch.from_numpy(gtr_field).to(device)
    args.Hei = gtr_ampli.shape[0]
    args.Wid = gtr_ampli.shape[1]
    args.gtr_ampli = gtr_ampli
    args.gtr_phase = gtr_phase
    args.gtr_field_tensor = gtr_field_tensor

    # simulate pupil aberration
    zernike_pupils = loadmat(args.zernike_dir)['zernike_pupils'].astype(np.float32).transpose((2, 0, 1))
    pupil_ampli = zernike_pupils[0, :, :]
    pupil_phase = np.sum(zernike_pupils * args.zernike_coeff, axis=0)
    pupil = pupil_ampli * np.exp(1j * pupil_phase)
    args.zernike_pupils = zernike_pupils
    args.pupil_real = pupil
    args.pupil_ampli = pupil_ampli
    args.pupil_phase = pupil_phase

    # simulate low resolution intensity
    fpm_estimator = FPM_estimator(args).to(device)
    fpm_estimator.requires_grad = False
    imaged_intens_tensor = fpm_estimator()
    args.ideal_intens_tensor = imaged_intens_tensor

    # simulate intensity weights
    intensity_weights = generate_inten_vara(fpm_estimator, args.channel_variaty)
    # savemat('cos4_weights.mat', {'intensity_weights':intensity_weights})
    intensity_weights_tensor = torch.from_numpy(intensity_weights).to(device)
    imaged_intens_tensor = imaged_intens_tensor * intensity_weights_tensor

    # torch.manual_seed(0)
    # simulate noise
    imaged_intens_tensor = imaged_intens_tensor + args.noise_level * torch.randn(imaged_intens_tensor.shape).to(device)

    args.intensity_weights = intensity_weights
    args.imaged_intens_tensor = imaged_intens_tensor
    args.imaged_intens = imaged_intens_tensor.cpu().numpy()

    # savemat('imaged_intens_0.mat', {'imaged_intens_0': args.imaged_intens})



def get_star(angle):
    star = np.ones((256, 256), dtype=np.float32)
    center = (128, 128)
    axes = (128, 128)
    for i in range(0, 360, 2*angle):
        cv2.ellipse(star, center, axes, angle, i, i+angle, 0, -1, cv2.LINE_AA)
    return star


def simulate_data_star(args):
    star = get_star(10)
    gtr_ampli = star
    gtr_phase = np.zeros_like(star, dtype=np.float32)
    # gtr_phase = np.array(np.zeros_like(gtr_ampli)) / 255
    gtr_field = np.sqrt(gtr_ampli) * np.exp(1j * gtr_phase)
    gtr_field_tensor = torch.from_numpy(gtr_field).to(device)
    args.Hei = gtr_ampli.shape[0]
    args.Wid = gtr_ampli.shape[1]
    args.gtr_ampli = gtr_ampli
    args.gtr_phase = gtr_phase
    args.gtr_field_tensor = gtr_field_tensor

    # simulate pupil aberration
    zernike_pupils = loadmat(args.zernike_dir)['zernike_pupils'].astype(np.float32).transpose((2, 0, 1))
    pupil_ampli = zernike_pupils[0, :, :]
    pupil_phase = np.sum(zernike_pupils * args.zernike_coeff, axis=0)
    pupil = pupil_ampli * np.exp(1j * pupil_phase)
    args.zernike_pupils = zernike_pupils
    args.pupil_real = pupil
    args.pupil_ampli = pupil_ampli
    args.pupil_phase = pupil_phase

    # simulate low resolution intensity
    fpm_estimator = FPM_estimator(args).to(device)
    fpm_estimator.requires_grad = False
    imaged_intens_tensor = fpm_estimator()
    args.ideal_intens_tensor = imaged_intens_tensor

    # simulate intensity weights
    intensity_weights = generate_inten_vara(fpm_estimator, args.channel_variaty)
    # savemat('cos4_weights.mat', {'intensity_weights':intensity_weights})
    intensity_weights_tensor = torch.from_numpy(intensity_weights).to(device)
    imaged_intens_tensor = imaged_intens_tensor * intensity_weights_tensor

    # torch.manual_seed(0)
    # simulate noise
    imaged_intens_tensor = imaged_intens_tensor + args.noise_level * torch.randn(imaged_intens_tensor.shape).to(device)

    args.intensity_weights = intensity_weights
    args.imaged_intens_tensor = imaged_intens_tensor
    args.imaged_intens = imaged_intens_tensor.cpu().numpy()

    # savemat('imaged_intens_0.mat', {'imaged_intens_0': args.imaged_intens})


def generate_inten_vara(model, type="None"):
    LED_num = model.LED_num
    illumination_distance = model.illumination_distance
    distances = model.distances

    np.random.seed(0)
    if type == "None":
        intensity_weights = np.ones((LED_num, 1, 1), np.float32)
    elif type == "cos4":
        intensity_weights = ((illumination_distance / distances) ** 4).reshape((LED_num, 1, 1))
    elif args.channel_variaty == "random":
        intensity_weights = (np.random.rand(LED_num, 1, 1) * args.variaty_level + 1 - args.variaty_level).astype(np.float32)
        # intensity_weights = np.random.rand(LED_num, 1, 1).astype(np.float32)
    elif args.channel_variaty == "cos4rand":
        intensity_weights = ((illumination_distance / distances) ** 4).reshape((LED_num, 1, 1)) * \
                            (np.random.rand(LED_num, 1, 1) * args.variaty_level + 1 - args.variaty_level).astype(np.float32)
        intensity_weights = intensity_weights / np.max(intensity_weights)

    return intensity_weights



def load_real_data(args):
    imaged_intens = loadmat(args.raw_dir)['process_intens'].transpose(2, 0, 1).astype(np.float32)
    if imaged_intens.shape[1] > 64:
        imaged_intens = imaged_intens[:, 0:64, 0:64]
    imaged_intens -= 8/65535
    imaged_intens[imaged_intens<0] = 0
    imaged_intens_tensor = torch.from_numpy(imaged_intens).to(device)
    imaged_intens_tensor.requires_grad = False
    args.Hei = imaged_intens_tensor.shape[1] * args.upsample_ratio
    args.Wid = imaged_intens_tensor.shape[2] * args.upsample_ratio

    args.imaged_intens = imaged_intens
    args.imaged_intens_tensor = imaged_intens_tensor

    zernike_pupils = loadmat(args.zernike_dir)['zernike_pupils'].astype(np.float32).transpose((2, 0, 1))
    args.zernike_pupils = zernike_pupils



def load_real_data2(args):
    imaged_intens = np.square(loadmat(args.raw_dir)['RAW'].transpose(2, 0, 1).astype(np.float32))
    imaged_intens = imaged_intens[:, 32:96, 32:96]
    imaged_intens_tensor = torch.from_numpy(imaged_intens).to(device)
    imaged_intens_tensor.requires_grad = False
    args.Hei = imaged_intens_tensor.shape[1] * args.upsample_ratio
    args.Wid = imaged_intens_tensor.shape[2] * args.upsample_ratio

    args.imaged_intens = imaged_intens
    args.imaged_intens_tensor = imaged_intens_tensor

    zernike_pupils = loadmat(args.zernike_dir)['zernike_pupils'].astype(np.float32).transpose((2, 0, 1))
    args.zernike_pupils = zernike_pupils



def load_real_data3(args):
    imaged_intens = loadmat(args.raw_dir)['imlow_HDR'].transpose(2, 0, 1).astype(np.float32)
    index = loadmat('mat_zheng/index.mat')['I'].squeeze(0) - 1
    imaged_intens = imaged_intens[index, :128, :128]
    imaged_intens_tensor = torch.from_numpy(imaged_intens).to(device)
    imaged_intens_tensor.requires_grad = False
    args.Hei = imaged_intens_tensor.shape[1] * args.upsample_ratio
    args.Wid = imaged_intens_tensor.shape[2] * args.upsample_ratio

    args.imaged_intens = imaged_intens
    args.imaged_intens_tensor = imaged_intens_tensor

    zernike_pupils = loadmat(args.zernike_dir)['zernike_pupils'].astype(np.float32).transpose((2, 0, 1))
    args.zernike_pupils = zernike_pupils



def load_real_data4(args):
    imaged_intens = loadmat(args.raw_dir)['RAW'].transpose(2, 0, 1).astype(np.float32)
    imaged_intens_tensor = torch.from_numpy(imaged_intens).to(device)
    imaged_intens_tensor.requires_grad = False
    args.Hei = imaged_intens_tensor.shape[1] * args.upsample_ratio
    args.Wid = imaged_intens_tensor.shape[2] * args.upsample_ratio

    args.imaged_intens = imaged_intens
    args.imaged_intens_tensor = imaged_intens_tensor

    zernike_pupils = loadmat(args.zernike_dir)['zernike_pupils'].astype(np.float32).transpose((2, 0, 1))
    args.zernike_pupils = zernike_pupils


def test_FPM_imaging():
    simulate_data(args)
    imaged_intens = args.imaged_intens_tensor.detach().cpu().numpy()
    pupil = args.pupil_tensor.detach().cpu().numpy()

    matplotlib.use('TkAgg')
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(2, 5, 1)
    ax.imshow(imaged_intens[0, :, :], cmap='gray')
    ax = fig.add_subplot(2, 5, 2)
    ax.imshow(imaged_intens[1, :, :], cmap='gray')
    ax = fig.add_subplot(2, 5, 3)
    ax.imshow(imaged_intens[2, :, :], cmap='gray')
    ax = fig.add_subplot(2, 5, 4)
    ax.imshow(imaged_intens[3, :, :], cmap='gray')
    ax = fig.add_subplot(2, 5, 5)
    ax.imshow(imaged_intens[84, :, :], cmap='gray')

    ax = fig.add_subplot(2, 5, 6)
    ax.imshow(pupil.real[:, :], cmap='gray')
    # ax = fig.add_subplot(2, 5, 7)
    # ax.imshow(pupils_high[1,:,:], cmap='gray')
    # ax = fig.add_subplot(2, 5, 8)
    # ax.imshow(pupils_high[2,:,:], cmap='gray')
    # ax = fig.add_subplot(2, 5, 9)
    # ax.imshow(pupils_high[3,:,:], cmap='gray')
    # ax = fig.add_subplot(2, 5, 10)
    # ax.imshow(pupils_high[84,:,:], cmap='gray')
    plt.show()
