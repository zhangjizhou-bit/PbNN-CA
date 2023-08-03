#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import argparse
import time
import numpy as np

parser = argparse.ArgumentParser(description='FPM with deep learning')
args4 = parser.parse_args()


#----------------program parameters------------------------------------------
args4.raw_dir = 'mat_tian_121/Raw_data.mat'
args4.zernike_dir = 'generate_zernike_pupils/zernike_pupils_tian_128.mat'
args4.optimizer = "Adam"

args4.mode = "real_data"

args4.lr = 2e-2
args4.lr_pupil = 1e-4
args4.lr_cam = 5e-1
args4.epochs = 300
args4.loss = "1*L1+3e1*L2+5e-4*FPM+2e-1*TV"



# args4.loss = "1*L1+10*L2+1e-3*FPM+1e-4*TV"
# args4.loss = "1*L1+1e1*L2+1e-3*FPM"
# args4.loss = "1*L1+1e1*L2"
# args4.loss = "1*L1"

args4.result_dir = time.strftime('./results/%Y_%m_%d_%H_%M_%S',time.localtime(time.time()))   # 结果保存位置

#----------------system errors------------------------------------------
args4.channel_variaty = "cos4"  # None, cos4, random, cos4rand
args4.variaty_level = 0.2
args4.channel_estimator = "CA"  # None, CA, cos4
args4.pupil_estimator = "CA"  # None, CA, layer
#----------------system parameters------------------------------------------
args4.pixel_size = 6.5  # actual size of sensor pixel
args4.magnification = 8.1458  # magnification of the objective
args4.NA_obj = 0.2  # numerical aperture of the objective
args4.wavelength = 0.514  # wavelength of light used for simulated illumination
args4.LED_spacing = 4000  # distance between LEDs in the array
args4.illumination_distance = 67.5e3  # distance from the LED matrix to the sample
args4.LED_num_side = 11  # LED num in one side
args4.upsample_ratio = 4  # upsample ratio of reconstruction


