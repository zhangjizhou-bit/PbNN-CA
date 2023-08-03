#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import argparse
import time
import numpy as np

parser = argparse.ArgumentParser(description='FPM with deep learning')
args2 = parser.parse_args()


#----------------program parameters------------------------------------------
args2.raw_dir = 'D:/Research/Simulation/FPM/code_adap/Raw_data.mat'
# args2.zernike_dir = 'generate_zernike_pupils/zernike_pupils_zuo_128.mat'
args2.zernike_dir = 'generate_zernike_pupils/zernike_pupils_zuo_64.mat'
args2.optimizer = "Adam"

args2.mode = "real_data"

args2.lr = 1e-2
args2.lr_pupil = 2e-3
args2.lr_cam = 1e-4
args2.epochs = 200
# args2.loss = "1*L1+10*L2+1e-3*FPM+1e-4*TV"
# args2.loss = "1*L1+1e1*L2+1e-3*FPM+1e-5*TV"
# args2.loss = "1*L1+1e1*L2+1e-3*FPM"
args2.loss = "1*L1+2e-3*FPM"
# args2.loss = "1*L1+1e-3*FPM+1e-3*TV"

args2.result_dir = time.strftime('./results/%Y_%m_%d_%H_%M_%S',time.localtime(time.time()))   # 结果保存位置

#----------------system errors------------------------------------------
args2.channel_variaty = "cos4"  # None, cos4, random, cos4rand
args2.variaty_level = 0.2
args2.channel_estimator = "CA"  # None, CA, cos4
args2.pupil_estimator = "CA"  # None, CA, layer
#----------------system parameters------------------------------------------
args2.pixel_size = 6.5  # actual size of sensor pixel
args2.magnification = 4  # magnification of the objective
args2.NA_obj = 0.1  # numerical aperture of the objective
args2.wavelength = 0.626  # wavelength of light used for simulated illumination
args2.LED_spacing = 2500  # distance between LEDs in the array
args2.illumination_distance = 87.5e3  # distance from the LED matrix to the sample
args2.LED_num_side = 21  # LED num in one side
args2.upsample_ratio = 4  # upsample ratio of reconstruction


