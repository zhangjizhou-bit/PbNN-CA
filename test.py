import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import os
from train import FPM_reconstruction
from utility import *
from parameters import args
import time
import argparse
import pickle
from matplotlib.patches import ConnectionPatch



if __name__ == '__main__':
    args.zernike_ampli_coeff = np.zeros((21, 1, 1), dtype=np.float32)
    args.zernike_ampli_coeff[0, 0, 0] = 1
    # args.zernike_ampli_coeff[0, 0, 0] = 0.7
    # args.zernike_ampli_coeff[4, 0, 0] = 0.3
    args.zernike_phase_coeff = np.zeros((21, 1, 1), dtype=np.float32)
    # args.zernike_phase_coeff[1, 0, 0] = 0.25
    # args.zernike_phase_coeff[2, 0, 0] = 0.25
    args.channel_variaty = "cos4rand"  # None, cos4, random, cos4rand
    args.channel_estimator = "None"  # None, CA, cos4
    args.pupil_estimator = "CA"  # None, CA, layer
    args.mode = "simulator"
    simulate_data(args)
    args.mode = "estimator"
    fpm_reconstruction = FPM_reconstruction(args)
    fpm_reconstruction.train_model()
    fpm_reconstruction.eval_model()
    Metrics(args)
    ShowResults(args)


