#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import argparse
import time
import numpy as np

parser = argparse.ArgumentParser(description='FPM with deep learning')
args3 = parser.parse_args()


#----------------program parameters------------------------------------------
args3.ampli_dir = 'Image/Baboon_256.bmp'
args3.phase_dir = 'Image/Aerial_256.bmp'
args3.raw_dir = 'mat_zheng/HE_green.mat'
args3.zernike_dir = 'generate_zernike_pupils/zernike_pupils_zheng_128.mat'
# args3.zernike_dir = 'generate_zernike_pupils/zernike_pupils_512.mat'
args3.optimizer = "Adam"

args3.mode = "real_data"

if args3.mode == "real_data":
    args3.lr = 1e-3
    args3.lr_pupil = 1e-4
    args3.lr_cam = 0.1
    args3.epochs = 100
    # args3.loss = "1*L1+5*L2+1e-3*FPM+5e-3*TV"
    args3.loss = "1*L1"


args3.result_dir = time.strftime('./results/%Y_%m_%d_%H_%M_%S',time.localtime(time.time()))   # 结果保存位置

#----------------system errors------------------------------------------
# args3.noise_level = 5e-5
args3.noise_level = 0
# args3.zernike_ampli_coeff[0, 0, 0] = 1
args3.zernike_coeff = np.zeros((15, 1, 1), dtype=np.float32)
args3.zernike_coeff[4, 0, 0] = -0.5
# args3.zernike_coeff[5, 0, 0] = -0.2
# args3.zernike_coeff[6, 0, 0] = 0.1
# args3.zernike_coeff[7, 0, 0] = 0.7
# args3.zernike_coeff[8, 0, 0] = -0.2
# args3.zernike_coeff[9, 0, 0] = 0.1
# args3.zernike_coeff[11, 0, 0] = 0.4
# args3.zernike_coeff[12, 0, 0] = 0.6
# args3.zernike_coeff[14, 0, 0] = -0.3
# np.random.seed(3)
# ratio = 0.5
# args3.zernike_coeff = 2*ratio*np.random.rand(15, 1, 1).astype(np.float32) - ratio
args3.channel_variaty = "None"  # None, cos4, random, cos4rand
args3.variaty_level = 0.2
args3.channel_estimator = "None"  # None, CA, cos4
args3.pupil_estimator = "None"  # None, CA, layer
# args3.epochs = 500

#----------------system parameters------------------------------------------
args3.pixel_size = 1.845  # actual size of sensor pixel
args3.magnification = 4  # magnification of the objective
args3.NA_obj = 0.1  # numerical aperture of the objective
args3.wavelength = 0.532  # wavelength of light used for simulated illumination
args3.LED_spacing = 4000  # distance between LEDs in the array
args3.illumination_distance = 9.088e4  # distance from the LED matrix to the sample
args3.LED_num_side = 15  # LED num in one side
args3.upsample_ratio = 4  # upsample ratio of reconstruction


