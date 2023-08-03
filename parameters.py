#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import argparse
import time
import numpy as np

parser = argparse.ArgumentParser(description='FPM with deep learning')
args = parser.parse_args()


#----------------program parameters------------------------------------------
args.ampli_dir = 'Image/Baboon_256.bmp'
args.phase_dir = 'Image/Aerial_256.bmp'
# args.raw_dir = 'mat_data_add/new_cell_135.mat'
args.raw_dir = 'mat_data/mat_cell_clean.mat'
args.zernike_dir = 'generate_zernike_pupils/zernike_pupils_256.mat'
# args.zernike_dir = 'generate_zernike_pupils/zernike_pupils_512.mat'
args.optimizer = "Adam"

args.mode = "simulator"

if args.mode == "simulator":
    # args.lr = 1e-2
    # args.lr_pupil = 2e-3
    # args.lr_cam = 0.4
    # args.epochs = 500
    # args.loss = "1*L1+5*L2+1e-4*FPM+5e-5*TV"
    # args.loss = "1*L1+5*L2"
    # args.loss = "1*L1+5*L2"
    args.lr = 1e-2
    args.lr_pupil = 2e-3
    args.lr_cam = 0.1
    args.epochs = 250
    args.loss = "1*L1+5*L2+5e-4*FPM+5e-4*TV"
elif args.mode == "real_data":
    args.lr = 2e-2
    args.lr_pupil = 5e-3
    args.lr_cam = 0.1
    args.epochs = 200
    args.loss = "1*L1+20*L2+1e-3*FPM+4e-2*TV"


args.result_dir = time.strftime('./results/%Y_%m_%d_%H_%M_%S',time.localtime(time.time()))   # 结果保存位置

#----------------system errors------------------------------------------
# args.noise_level = 5e-5
args.noise_level = 0
# args.zernike_ampli_coeff[0, 0, 0] = 1
args.zernike_coeff = np.zeros((15, 1, 1), dtype=np.float32)
args.zernike_coeff[4, 0, 0] = -0.5
args.zernike_coeff[5, 0, 0] = -0.2
args.zernike_coeff[6, 0, 0] = 0.1
args.zernike_coeff[7, 0, 0] = 0.7
args.zernike_coeff[8, 0, 0] = -0.2
args.zernike_coeff[9, 0, 0] = 0.1
args.zernike_coeff[11, 0, 0] = 0.4
args.zernike_coeff[12, 0, 0] = 0.6
args.zernike_coeff[14, 0, 0] = -0.3
# np.random.seed(3)
# ratio = 0.5
# args.zernike_coeff = 2*ratio*np.random.rand(15, 1, 1).astype(np.float32) - ratio
args.channel_variaty = "cos4rand"  # None, cos4, random, cos4rand
args.variaty_level = 0.2
args.channel_estimator = "CA"  # None, CA, cos4
args.pupil_estimator = "CA"  # None, CA, layer
# args.epochs = 500

#----------------system parameters------------------------------------------
args.pixel_size = 6.5  # actual size of sensor pixel
args.magnification = 4  # magnification of the objective
args.NA_obj = 0.13  # numerical aperture of the objective
args.wavelength = 0.505  # wavelength of light used for simulated illumination
args.LED_spacing = 8128  # distance between LEDs in the array
args.illumination_distance = 9.8e4  # distance from the LED matrix to the sample
args.LED_num_side = 13  # LED num in one side
args.upsample_ratio = 4  # upsample ratio of reconstruction


