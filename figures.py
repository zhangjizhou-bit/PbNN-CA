import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import os
from train import FPM_reconstruction
from utility import *
from parameters import args
from parameters_zuo import args2
import time
import argparse
import pickle
from matplotlib.patches import ConnectionPatch
from matplotlib.transforms import Affine2D
import matplotlib.lines as mlines

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# 设备参数
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def figure_3_1_a():
    matplotlib.use('TkAgg')
    plt.rcParams['font.sans-serif'] = ['Times New Roman']  # 用来正常显示中文标签
    plt.rcParams['font.size'] = 12

    intensity_weights = np.ones((13, 13))
    fig = plt.figure(figsize=(2, 2))
    plt.subplots_adjust(top=0.95, bottom=0.13, left=0.14, right=0.95, hspace=0, wspace=0)
    ax = plt.gca()
    ax.imshow(intensity_weights, cmap='viridis',clim=(0, 1))
    # ax.set_title('Intensity weights')
    # ax.set_ylim((0, 1))
    plt.savefig('figures/figure_3_1_a.png', dpi=600)
    plt.savefig('figures/figure_3_1_a.svg', dpi=600)
    plt.show()
    plt.close(fig)


def figure_3_1_b():
    matplotlib.use('TkAgg')
    plt.rcParams['font.sans-serif'] = ['Times New Roman']  # 用来正常显示中文标签
    plt.rcParams['font.size'] = 12

    fig = plt.figure(figsize=(2, 2))
    plt.subplots_adjust(top=0.95, bottom=0.13, left=0.14, right=0.95, hspace=0, wspace=0)
    ax = plt.gca()
    zernike_pupils = loadmat(args.zernike_dir)['zernike_pupils'].astype(np.float32).transpose((2, 0, 1))
    pupil0 = zernike_pupils[0,:,:]
    ax.imshow(pupil0, cmap='viridis',clim=(-1, 1))
    # ax.set_title('Pupil amplitude')
    plt.savefig('figures/figure_3_1_b.png', dpi=600)
    plt.savefig('figures/figure_3_1_b.svg', dpi=600)

    pupil00 = np.zeros_like(pupil0)
    ax.imshow(pupil00, cmap='viridis', clim=(-1, 1))
    # ax.set_title('Pupil phase')
    plt.savefig('figures/figure_3_1_c.png', dpi=600)
    plt.savefig('figures/figure_3_1_c.svg', dpi=600)
    plt.show()
    plt.close(fig)


def ShowResult_3_1(args):
    matplotlib.use('TkAgg')
    plt.rcParams['font.sans-serif'] = ['Times New Roman']  # 用来正常显示中文标签
    plt.rcParams['font.size'] = 12

    cv2.imwrite('figures/figure_3_1_d.png', args.est_ampli_norm * 255)
    cv2.imwrite('figures/figure_3_1_e.png', args.est_phase_norm * 255)

    fig = plt.figure(figsize=(2, 2))
    plt.subplots_adjust(top=0.95, bottom=0.13, left=0.14, right=0.95, hspace=0, wspace=0)
    ax = plt.gca()
    ax.imshow(args.pupil_phase_estimate, cmap='viridis', clim=(-1, 1.5))
    # ax.set_title('Pupil phase estimate',fontsize=10)
    plt.savefig('figures/figure_3_1_g.png', dpi=600)
    plt.savefig('figures/figure_3_1_g.svg', dpi=600)
    plt.show()
    plt.close(fig)
    pupil_phase_norm = args.pupil_phase_estimate
    pupil_phase_norm = ((pupil_phase_norm - np.min(pupil_phase_norm))/(np.max(pupil_phase_norm)-np.min(pupil_phase_norm))*255).astype(np.uint8)
    pupil_phase_norm = cv2.applyColorMap(pupil_phase_norm, cv2.COLORMAP_VIRIDIS)
    cv2.imwrite('figures/pupil_phase_estimate.png', pupil_phase_norm)

    fig = plt.figure(figsize=(2, 2))
    plt.subplots_adjust(top=0.95, bottom=0.13, left=0.14, right=0.95, hspace=0, wspace=0)
    ax = plt.gca()
    ax.imshow(args.channel_weights, cmap='viridis', clim=(0, 1))
    # ax.set_title('Intensity weights')
    plt.savefig('figures/figure_3_1_h.png', dpi=600)
    plt.savefig('figures/figure_3_1_h.svg', dpi=600)
    plt.show()
    plt.close(fig)


def figure_3_1():

    figure_3_1_a()
    figure_3_1_b()
    args.variaty_level = 0.2
    args.channel_estimator = "CA"  # None, CA, cos4
    args.pupil_estimator = "CA"  # None, CA, layer
    if args.mode == "simulator":
        simulate_data(args)
    elif args.mode == "real_data":
        load_real_data(args)

    args.mode = "estimator"
    fpm_reconstruction = FPM_reconstruction(args)
    fpm_reconstruction.train_model()
    fpm_reconstruction.eval_model()

    Metrics(args)
    ShowResult_3_1(args)


def figure_3_2():
    matplotlib.use('TkAgg')
    plt.rcParams['font.sans-serif'] = ['Times New Roman']  # 用来正常显示中文标签
    plt.rcParams['font.size'] = 12
    plt.rc('axes', axisbelow=True)
    fig = plt.figure(figsize=(2, 2))
    plt.subplots_adjust(top=0.95, bottom=0.13, left=0.14, right=0.95, hspace=0, wspace=0)

    zernike_coeff = np.zeros((15), dtype=np.float32)
    zernike_coeff[4] = -0.5
    zernike_coeff[5] = -0.2
    zernike_coeff[6] = 0.1
    zernike_coeff[7] = 0.7
    zernike_coeff[8] = -0.2
    zernike_coeff[9] = 0.1
    zernike_coeff[11] = 0.4
    zernike_coeff[12] = 0.6
    zernike_coeff[14] = -0.3

    plt.bar(range(len(zernike_coeff)), zernike_coeff)
    ax = plt.gca()
    ax.grid(True)
    # ax.set_title('Pupil phase weights', fontsize=14)
    ax.set_ylim([-1, 1])
    ax.set_yticks([-1, -0.5, 0, 0.5, 1])
    ax.set_yticklabels(('-1', '', '0', '', '1'))
    ax.set_xticks([0, 5, 10, 15])
    ax.set_xticklabels(('0', '5', '10', '15'))
    # plt.grid(axis="y")
    plt.savefig('figures/figure_3_2_a.png', dpi=600)
    plt.savefig('figures/figure_3_2_a.svg', dpi=600)

    plt.show()
    plt.close(fig)


def figure_3_3():

    args.mode = "simulator"
    edge = 5

    args.channel_variaty = "None"  # None, cos4, random
    args.zernike_ampli_coeff = np.zeros((21, 1, 1), dtype=np.float32)
    args.zernike_ampli_coeff[0, 0, 0] = 1
    args.zernike_phase_coeff = np.zeros((21, 1, 1), dtype=np.float32)
    simulate_data(args)
    intens1 = args.ideal_intens_tensor.detach().cpu().numpy()
    # load_real_data(args)
    # intens1 = args.imaged_intens_tensor.detach().cpu().numpy()
    c, h, w = intens1.shape
    merge1 = np.ones((h*13+edge*14, w*13+edge*14), dtype=np.float)
    idxs = np.array([70,71,72,83,84,85,96,97,98])
    for yy in range(13):
        y = edge + yy * (h+edge)
        for xx in range(13):
            x = edge + xx * (h+edge)
            idx = yy*13+xx
            if idx in idxs:
                merge1[y:y+h, x:x+w] = intens1[idx, :,:]
            else:
                merge1[y:y+h, x:x+w] = 50*intens1[idx, :,:]

    cv2.imwrite('figures/figure_3_3_a.png', merge1 * 255)

    args.channel_variaty = "cos4rand"  # None, cos4, random, cos4rand
    simulate_data(args)
    intensity_weights = args.intensity_weights
    weights = args.intensity_weights.reshape((13,13))
    weights_large = cv2.resize(weights, (0,0), None, 10, 10, cv2.INTER_NEAREST)
    cv2.imwrite('figures/figure_3_3_c.png', weights_large * 255)


def simulate_all_noise(args):
    args.channel_variaty = "cos4rand"  # None, cos4, random, cos4rand
    args.zernike_coeff = np.zeros((15, 1, 1), dtype=np.float32)
    args.zernike_coeff[4, 0, 0] = 0.1
    args.zernike_coeff[5, 0, 0] = -0.5
    args.zernike_coeff[6, 0, 0] = 0.4
    args.zernike_coeff[7, 0, 0] = 0.6
    args.zernike_coeff[8, 0, 0] = -0.1
    args.zernike_coeff[9, 0, 0] = -0.4
    args.zernike_coeff[12, 0, 0] = 0.5
    args.zernike_coeff[14, 0, 0] = -0.5
    for noise_level in np.arange(0, 2.1e-4, 1e-5):
        args.noise_level = noise_level
        args.result_dir = time.strftime('./results/%Y_%m_%d_%H_%M_%S', time.localtime(time.time()))
        args.mode = "simulator"
        simulate_data(args)
        file_name = 'experiment1_results/args_{:1.1e}'.format(args.noise_level)
        f = open(file_name, 'wb')
        pickle.dump(args, f)
        f.close()
        mat_name = 'experiment1_results/imaged_intens_{:1.1e}.mat'.format(args.noise_level)
        savemat(mat_name, {'imaged_intens': args.imaged_intens})


def eval_all_matlab_results(args):
    simulate_data(args)
    solvers = ['AS', 'GS', 'EPRY', 'GS_cos4']
    for solver in solvers:
        for noise_level in np.arange(0, 2.1e-4, 1e-5):
            file_name = 'experiment1_results/{}_{:1.1e}.mat'.format(solver, noise_level)
            data = loadmat(file_name, struct_as_record=False)
            args.est_ampli = data['est_ampli']
            args.est_phase = data['est_phase']
            args.pupil_ampli_estimate = data['pupil_ampli_estimate']
            args.pupil_phase_estimate = data['pupil_phase_estimate']
            Metrics(args)
            data['est_ampli_norm'] = args.est_ampli_norm
            data['est_phase_norm'] = args.est_phase_norm
            data['mae_ampli'] = args.mae_ampli
            data['psnr_ampli'] = args.psnr_ampli
            data['ssim_ampli'] = args.ssim_ampli
            data['mae_phase'] = args.mae_phase
            data['psnr_phase'] = args.psnr_phase
            data['ssim_phase'] = args.ssim_phase

            data['mae_pupil_phase'] = args.mae_pupil_phase
            data['psnr_pupil_phase'] = args.psnr_pupil_phase
            data['ssim_pupil_phase'] = args.ssim_pupil_phase
            savemat(file_name, data)


def eval_all_NN_results():
    estimators = ['None', 'layer', 'cos4', 'CA']
    # estimators = ['layer']
    for estimator in estimators:
        for noise_level in np.arange(0, 2.1e-4, 1e-5):
            file_name = 'experiment1_results/args_{:1.1e}'.format(noise_level)
            f = open(file_name, 'rb')
            args = pickle.load(f)
            f.close()
            args.mode = "estimator"
            args.result_dir = time.strftime('./results/%Y_%m_%d_%H_%M_%S', time.localtime(time.time()))
            if estimator == 'None':
                args.pupil_estimator = 'None'
                args.channel_estimator = 'None'
            elif estimator == 'layer':
                args.pupil_estimator = 'layer'
                args.channel_estimator = 'None'
                args.epochs = 500
            elif estimator == 'cos4':
                args.pupil_estimator = 'None'
                args.channel_estimator = 'cos4'
            elif estimator == 'CA':
                args.pupil_estimator = 'CA'
                args.channel_estimator = 'CA'
            else:
                args.pupil_estimator = estimator
            fpm_reconstruction = FPM_reconstruction(args)
            fpm_reconstruction.train_model()
            fpm_reconstruction.eval_model()
            Metrics(args)
            data = {}
            data['est_ampli'] = args.est_ampli
            data['est_phase'] = args.est_phase
            data['est_ampli_norm'] = args.est_ampli_norm
            data['est_phase_norm'] = args.est_phase_norm
            if hasattr(args, 'pupil_ampli_estimate'):
                data['pupil_ampli_estimate'] = args.pupil_ampli_estimate
                data['pupil_phase_estimate'] = args.pupil_phase_estimate
            else:
                data['pupil_ampli_estimate'] = 0
                data['pupil_phase_estimate'] = 0

            data['mae_ampli'] = args.mae_ampli
            data['psnr_ampli'] = args.psnr_ampli
            data['ssim_ampli'] = args.ssim_ampli
            data['mae_phase'] = args.mae_phase
            data['psnr_phase'] = args.psnr_phase
            data['ssim_phase'] = args.ssim_phase

            data['mae_pupil_phase'] = args.mae_pupil_phase
            data['psnr_pupil_phase'] = args.psnr_pupil_phase
            data['ssim_pupil_phase'] = args.ssim_pupil_phase
            save_name = 'experiment1_results/PbNN-{}_{:1.1e}.mat'.format(estimator, noise_level)
            savemat(save_name, data)


def figure_4_4():
    mae_ampli = np.zeros((8,21))
    psnr_ampli = np.zeros((8,21))
    ssim_ampli = np.zeros((8,21))
    mae_phase = np.zeros((8,21))
    psnr_phase = np.zeros((8,21))
    ssim_phase = np.zeros((8,21))
    mae_pupil_ampli = np.zeros((8,21))
    psnr_pupil_ampli = np.zeros((8,21))
    ssim_pupil_ampli = np.zeros((8,21))
    mae_pupil_phase = np.zeros((8,21))
    psnr_pupil_phase = np.zeros((8,21))
    ssim_pupil_phase = np.zeros((8,21))
    color_idxs = [0, 9, 1, 2, 7, 4, 3]
    # solvers = ['GS', 'GS_cos4', 'AS', 'PbNN-None', 'PbNN-cos4', 'PbNN-IE']
    solvers = ['GS', 'GS_cos4', 'AS', 'EPRY', 'PbNN-None', 'PbNN-layer', 'PbNN-CA']
    legends = ['AP', 'AP-cos4', 'AS', 'EPRY', 'PbNN', 'PbNN-layer', 'PbNN-CA']
    noise_levels = np.arange(0, 2.1e-4, 1e-5)
    for solver_idx in range(len(solvers)):
        solver = solvers[solver_idx]
        for noise_idx in range(len(noise_levels)):
            noise_level = noise_levels[noise_idx]
            file_name = 'experiment1_results/{}_{:1.1e}.mat'.format(solver, noise_level)
            data = loadmat(file_name, struct_as_record=False)
            mae_ampli[solver_idx, noise_idx] = data['mae_ampli']
            psnr_ampli[solver_idx, noise_idx] = data['psnr_ampli']
            ssim_ampli[solver_idx, noise_idx] = data['ssim_ampli']
            mae_phase[solver_idx, noise_idx] = data['mae_phase']
            psnr_phase[solver_idx, noise_idx] = data['psnr_phase']
            ssim_phase[solver_idx, noise_idx] = data['ssim_phase']
            mae_pupil_phase[solver_idx, noise_idx] = data['mae_pupil_phase']
            psnr_pupil_phase[solver_idx, noise_idx] = data['psnr_pupil_phase']
            ssim_pupil_phase[solver_idx, noise_idx] = data['ssim_pupil_phase']


    plt.rcParams['font.sans-serif'] = ['Times New Roman']  # 用来正常显示中文标签
    plt.rcParams['font.size'] = 12
    plt.rc('axes', axisbelow=True)
    # plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    grid = plt.GridSpec(2, 3, hspace=0.45, wspace=0.45)

    matplotlib.use('TkAgg')
    fig = plt.figure(figsize=(8, 5.4))
    # fig.set_dpi(72)
    plt.subplots_adjust(top=0.85, bottom=0.10, left=0.1, right=0.99)

    ax11 = fig.add_subplot(grid[0, 0])
    for solver_idx in range(len(solvers)):
        ax11.plot(noise_levels, mae_ampli[solver_idx, :], linewidth=1.5, marker='o', markersize=2.5, color=colors[color_idxs[solver_idx]])
    ax11.set_xlabel('Noise level ($\\times\mathregular{10^{-4}}$)',labelpad = 0)
    ax11.set_xticks(np.arange(0, 2.1e-4, 5e-5))
    ax11.set_xticklabels(['0', '0.5', '1.0', '1.5', '2.0'])
    ax11.set_ylim([0.02, 0.16])
    ax11.set_ylabel('MAE of amplitude')
    ax11.grid(True)

    ax12 = fig.add_subplot(grid[0, 1])
    for solver_idx in range(len(solvers)):
        ax12.plot(noise_levels, psnr_ampli[solver_idx, :], linewidth=1.5, marker='o', markersize=2.5, color=colors[color_idxs[solver_idx]])
    ax12.set_xlabel('Noise level ($\\times\mathregular{10^{-4}}$)',labelpad = 0)
    ax12.set_xticks(np.arange(0, 2.1e-4, 5e-5))
    ax12.set_xticklabels(['0', '0.5', '1.0', '1.5', '2.0'])
    ax12.legend(legends, bbox_to_anchor=(0.5, 1.1), loc='lower center', ncol=4)
    ax12.set_ylim([14, 31])
    ax12.set_ylabel('PSNR of amplitude')
    ax12.grid(True)

    ax13 = fig.add_subplot(grid[0, 2])
    for solver_idx in range(len(solvers)):
        ax13.plot(noise_levels, ssim_ampli[solver_idx, :], linewidth=1.5, marker='o', markersize=2.5, color=colors[color_idxs[solver_idx]])
    ax13.set_xlabel('Noise level ($\\times\mathregular{10^{-4}}$)',labelpad = 0)
    ax13.set_xticks(np.arange(0, 2.1e-4, 5e-5))
    ax13.set_xticklabels(['0', '0.5', '1.0', '1.5', '2.0'])
    ax13.set_ylim([0.25, 1.05])
    ax13.set_ylabel('SSIM of amplitude')
    ax13.grid(True)

    ax21 = fig.add_subplot(grid[1, 0])
    for solver_idx in range(len(solvers)):
        ax21.plot(noise_levels, mae_phase[solver_idx, :], linewidth=1.5, marker='o', markersize=2.5, color=colors[color_idxs[solver_idx]])
    ax21.set_xlabel('Noise level ($\\times\mathregular{10^{-4}}$)',labelpad = 0)
    ax21.set_xticks(np.arange(0, 2.1e-4, 5e-5))
    ax21.set_xticklabels(['0', '0.5', '1.0', '1.5', '2.0'])
    ax21.set_ylim([0.02, 0.16])
    ax21.set_ylabel('MAE of phase')
    ax21.grid(True)

    ax22 = fig.add_subplot(grid[1, 1])
    for solver_idx in range(len(solvers)):
        ax22.plot(noise_levels, psnr_phase[solver_idx, :], linewidth=1.5, marker='o', markersize=2.5, color=colors[color_idxs[solver_idx]])
    ax22.set_xlabel('Noise level ($\\times\mathregular{10^{-4}}$)',labelpad = 0)
    ax22.set_xticks(np.arange(0, 2.1e-4, 5e-5))
    ax22.set_xticklabels(['0', '0.5', '1.0', '1.5', '2.0'])
    ax22.set_ylim([14, 31])
    ax22.set_ylabel('PSNR of phase')
    ax22.grid(True)

    ax23 = fig.add_subplot(grid[1, 2])
    for solver_idx in range(len(solvers)):
        ax23.plot(noise_levels, ssim_phase[solver_idx, :], linewidth=1.5, marker='o', markersize=2.5, color=colors[color_idxs[solver_idx]])
    ax23.set_xlabel('Noise level ($\\times\mathregular{10^{-4}}$)',labelpad = 0)
    ax23.set_xticks(np.arange(0, 2.1e-4, 5e-5))
    ax23.set_xticklabels(['0', '0.5', '1.0', '1.5', '2.0'])
    ax23.set_ylim([0.25, 1.05])
    ax23.set_ylabel('SSIM of phase')
    ax23.grid(True)

    fig.text(0.010, 0.85, '(a)', fontsize=14, bbox=dict(facecolor="k", alpha=0.1))
    fig.text(0.355, 0.85, '(b)', fontsize=14, bbox=dict(facecolor="k", alpha=0.1))
    fig.text(0.685, 0.85, '(c)', fontsize=14, bbox=dict(facecolor="k", alpha=0.1))
    fig.text(0.010, 0.38, '(e)', fontsize=14, bbox=dict(facecolor="k", alpha=0.1))
    fig.text(0.355, 0.38, '(f)', fontsize=14, bbox=dict(facecolor="k", alpha=0.1))
    fig.text(0.685, 0.38, '(g)', fontsize=14, bbox=dict(facecolor="k", alpha=0.1))

    plt.savefig('figures/figure_4_4.png', dpi=600)
    plt.savefig('figures/figure_4_4.pdf', dpi=600)

    plt.show()
    plt.close(fig)



def figure_4_5():
    ampli_results = np.zeros((256, 256, 8))
    phase_results = np.zeros((256, 256, 8))
    mae_ampli = np.zeros((8,1))
    psnr_ampli = np.zeros((8,1))
    ssim_ampli = np.zeros((8,1))
    mae_phase = np.zeros((8,1))
    psnr_phase = np.zeros((8,1))
    ssim_phase = np.zeros((8,1))
    solvers = ['GS', 'GS_cos4', 'AS', 'EPRY', 'PbNN-None', 'PbNN-layer', 'PbNN-CA']
    legends = ['AP', 'AP-cos4', 'AS', 'EPRY', 'PbNN', 'PbNN-layer', 'PbNN-CA']
    for solver_idx in range(len(solvers)):
        solver = solvers[solver_idx]
        noise_level = 5e-5
        file_name = 'experiment1_results/{}_{:1.1e}.mat'.format(solver, noise_level)
        data = loadmat(file_name, struct_as_record=False)
        ampli_results[:,:,solver_idx] = data['est_ampli_norm']
        phase_results[:,:,solver_idx] = data['est_phase_norm']
        mae_ampli[solver_idx, 0] = data['mae_ampli']
        psnr_ampli[solver_idx, 0] = data['psnr_ampli']
        ssim_ampli[solver_idx, 0] = data['ssim_ampli']
        mae_phase[solver_idx, 0] = data['mae_phase']
        psnr_phase[solver_idx, 0] = data['psnr_phase']
        ssim_phase[solver_idx, 0] = data['ssim_phase']

    matplotlib.use('TkAgg')
    plt.rcParams['font.sans-serif'] = ['Times New Roman']  # 用来正常显示中文标签
    plt.rcParams['font.size'] = 12
    # plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

    fig1, axeses = plt.subplots(nrows=4, ncols=7, figsize=(10, 5.2),
                                gridspec_kw={"width_ratios": [1, 1, 1, 1, 1, 1, 1],
                                             "height_ratios": [1, 1, 0.2, 1]})
    plt.subplots_adjust(top=0.96, bottom=0, left=0.03, right=0.99, hspace=0.1, wspace=0.1)

    for row in range(axeses.shape[0]):
        for col in range(axeses.shape[1]):
            axeses[row, col].set_axis_off()

    for idx in range(7):
        axeses[0, idx].imshow(ampli_results[:,:,idx], cmap='gray',clim=(0.0, 1))
        axeses[0, idx].text(128, -25, legends[idx], verticalalignment='center', horizontalalignment='center')
        axeses[1, idx].imshow(phase_results[:,:,idx], cmap='gray',clim=(0.0, 1))
        if idx not in [3,5,6]:
            str = '{:0.3f} / {:2.2f} / {:0.3f}'.format(mae_ampli[idx,0], psnr_ampli[idx,0], ssim_ampli[idx,0])
            axeses[0, idx].text(128, 280, str, verticalalignment='center', horizontalalignment='center',fontsize=11)
            str = '{:0.3f} / {:2.2f} / {:0.3f}'.format(mae_phase[idx,0], psnr_phase[idx,0], ssim_phase[idx,0])
            axeses[1, idx].text(128, 280, str, verticalalignment='center', horizontalalignment='center',fontsize=11)
    axeses[0, 0].text(-30, 128, 'Amplitude', rotation=90, verticalalignment='center', horizontalalignment='center')
    axeses[1, 0].text(-30, 128, 'Phase', rotation=90, verticalalignment='center', horizontalalignment='center')

    idx = 3
    axeses[0, idx].text(38, 280, '{:0.3f}'.format(mae_ampli[idx, 0]), verticalalignment='center', horizontalalignment='center', fontsize=11, color='green')
    axeses[0, idx].text(84, 280, '/', verticalalignment='center', horizontalalignment='center', fontsize=11, color='black')
    axeses[0, idx].text(128, 280, '{:2.2f}'.format(psnr_ampli[idx, 0]), verticalalignment='center', horizontalalignment='center', fontsize=11, color='green')
    axeses[0, idx].text(172, 280, '/', verticalalignment='center', horizontalalignment='center', fontsize=11, color='black')
    axeses[0, idx].text(218, 280, '{:0.3f}'.format(ssim_ampli[idx, 0]), verticalalignment='center', horizontalalignment='center', fontsize=11, color='black')

    axeses[1, idx].text(38, 280, '{:0.3f}'.format(mae_phase[idx, 0]), verticalalignment='center', horizontalalignment='center', fontsize=11, color='green')
    axeses[1, idx].text(84, 280, '/', verticalalignment='center', horizontalalignment='center', fontsize=11, color='black')
    axeses[1, idx].text(128, 280, '{:2.2f}'.format(psnr_phase[idx, 0]), verticalalignment='center', horizontalalignment='center', fontsize=11, color='green')
    axeses[1, idx].text(172, 280, '/', verticalalignment='center', horizontalalignment='center', fontsize=11, color='black')
    axeses[1, idx].text(218, 280, '{:0.3f}'.format(ssim_phase[idx, 0]), verticalalignment='center', horizontalalignment='center', fontsize=11, color='black')

    idx = 5
    axeses[0, idx].text(38, 280, '{:0.3f}'.format(mae_ampli[idx, 0]), verticalalignment='center', horizontalalignment='center', fontsize=11, color='green')
    axeses[0, idx].text(84, 280, '/', verticalalignment='center', horizontalalignment='center', fontsize=11, color='black')
    axeses[0, idx].text(128, 280, '{:2.2f}'.format(psnr_ampli[idx, 0]), verticalalignment='center', horizontalalignment='center', fontsize=11, color='black')
    axeses[0, idx].text(172, 280, '/', verticalalignment='center', horizontalalignment='center', fontsize=11, color='black')
    axeses[0, idx].text(218, 280, '{:0.3f}'.format(ssim_ampli[idx, 0]), verticalalignment='center', horizontalalignment='center', fontsize=11, color='green')

    axeses[1, idx].text(38, 280, '{:0.3f}'.format(mae_phase[idx, 0]), verticalalignment='center', horizontalalignment='center', fontsize=11, color='black')
    axeses[1, idx].text(84, 280, '/', verticalalignment='center', horizontalalignment='center', fontsize=11, color='black')
    axeses[1, idx].text(128, 280, '{:2.2f}'.format(psnr_phase[idx, 0]), verticalalignment='center', horizontalalignment='center', fontsize=11, color='black')
    axeses[1, idx].text(172, 280, '/', verticalalignment='center', horizontalalignment='center', fontsize=11, color='black')
    axeses[1, idx].text(218, 280, '{:0.3f}'.format(ssim_phase[idx, 0]), verticalalignment='center', horizontalalignment='center', fontsize=11, color='green')

    idx = 6
    axeses[0, idx].text(38, 280, '{:0.3f}'.format(mae_ampli[idx, 0]), verticalalignment='center', horizontalalignment='center', fontsize=11, color='red')
    axeses[0, idx].text(84, 280, '/', verticalalignment='center', horizontalalignment='center', fontsize=11, color='black')
    axeses[0, idx].text(128, 280, '{:2.2f}'.format(psnr_ampli[idx, 0]), verticalalignment='center', horizontalalignment='center', fontsize=11, color='red')
    axeses[0, idx].text(172, 280, '/', verticalalignment='center', horizontalalignment='center', fontsize=11, color='black')
    axeses[0, idx].text(218, 280, '{:0.3f}'.format(ssim_ampli[idx, 0]), verticalalignment='center', horizontalalignment='center', fontsize=11, color='red')

    axeses[1, idx].text(38, 280, '{:0.3f}'.format(mae_phase[idx, 0]), verticalalignment='center', horizontalalignment='center', fontsize=11, color='red')
    axeses[1, idx].text(84, 280, '/', verticalalignment='center', horizontalalignment='center', fontsize=11, color='black')
    axeses[1, idx].text(128, 280, '{:2.2f}'.format(psnr_phase[idx, 0]), verticalalignment='center', horizontalalignment='center', fontsize=11, color='red')
    axeses[1, idx].text(172, 280, '/', verticalalignment='center', horizontalalignment='center', fontsize=11, color='black')
    axeses[1, idx].text(218, 280, '{:0.3f}'.format(ssim_phase[idx, 0]), verticalalignment='center', horizontalalignment='center', fontsize=11, color='red')

    con = ConnectionPatch(xyA=(0,0.5), xyB=(1,0.5), coordsA="data", coordsB="data", axesA=axeses[2, 0], axesB=axeses[2, 6], color="gray")
    axeses[2, 0].add_artist(con)


    gtr_ampli = np.array(cv2.imread(args.ampli_dir, 0).astype(np.float64)) / 255
    gtr_phase = np.array(cv2.imread(args.phase_dir, 0).astype(np.float64)) / 255

    wid = axeses[0, 0].get_position().x1 - axeses[0, 0].get_position().x0
    hei = axeses[0, 0].get_position().y1 - axeses[0, 0].get_position().y0
    ax1 = fig1.add_axes([0.1, 0.02, wid, hei])
    ax1.set_axis_off()
    ax2 = fig1.add_axes([0.28, 0.02, wid, hei])
    ax2.set_axis_off()

    a1 = ax1.imshow(gtr_ampli, cmap='gray',clim=(0.0, 1))
    ax1.text(128, -25, 'Ground truth amplitude', verticalalignment='center', horizontalalignment='center')
    a2 = ax2.imshow(gtr_phase, cmap='gray',clim=(0.0, 1))
    ax2.text(128, -25, 'Ground truth phase', verticalalignment='center', horizontalalignment='center')

    cax1 = fig1.add_axes([ax1.get_position().x1 + 0.01, ax1.get_position().y0, 0.01, hei])
    cax1.tick_params(labelsize=11)
    cb1 = plt.colorbar(a1, cax=cax1)
    cb1.set_ticks([0, 1])

    cax2 = fig1.add_axes([ax2.get_position().x1 + 0.01, ax2.get_position().y0, 0.01, hei])
    cax2.tick_params(labelsize=11)
    cb2 = plt.colorbar(a2, cax=cax2)
    cb2.set_ticks([0, 1])


    args.mode = "simulator"
    args.channel_variaty = "cos4rand"  # None, cos4, random, cos4rand
    args.variaty_level = 0.2
    simulate_data(args)
    ax3 = fig1.add_axes([0.72-wid, 0.02, wid, hei])
    ax3.set_axis_off()
    ax4 = fig1.add_axes([0.9-wid, 0.02, wid, hei])
    ax4.set_axis_off()
    # 画pupil相位
    pupil_phase_real = args.pupil_phase
    a3 = ax3.imshow(pupil_phase_real, cmap='viridis', clim=(-1, 2))
    ax3.text(32, -6.5, 'Pupil function', verticalalignment='center', horizontalalignment='center')
    cax3 = fig1.add_axes([ax3.get_position().x1 + 0.01, ax3.get_position().y0, 0.01, hei])
    cax3.tick_params(labelsize=11)
    cb3 = plt.colorbar(a3, cax=cax3)
    cb3.set_ticks([-1, 0, 1, 2])
    # 画LED强度误差
    intensity_weights = args.intensity_weights.reshape((13,13))
    a4 = ax4.imshow(intensity_weights, cmap='viridis', clim=(0, 1))
    ax4.text(6.5, -1.5, 'LED intensity', verticalalignment='center', horizontalalignment='center')
    cax4 = fig1.add_axes([ax4.get_position().x1 + 0.01, ax4.get_position().y0, 0.01, hei])
    cax4.tick_params(labelsize=11)
    cb4 = plt.colorbar(a4, cax=cax4)
    cb4.set_ticks([0, 1])

    plt.savefig('figures/figure_4_5.pdf', dpi=600)
    plt.savefig('figures/figure_4_5.jpg', dpi=600)
    plt.show()
    plt.close(fig1)


def simulate_all_noise2(args):
    args.channel_variaty = "None"  # None, cos4, random, cos4rand
    # args.channel_variaty = "cos4rand"  # None, cos4, random, cos4rand
    # args.variaty_level = 0.2
    args.channel_estimator = "None"
    noise_level = 5e-5
    args.noise_level = noise_level
    args.result_dir = time.strftime('./results/%Y_%m_%d_%H_%M_%S', time.localtime(time.time()))
    args.mode = "simulator"
    simulate_data(args)
    file_name = 'experiment2_results/args_{:1.1e}'.format(args.noise_level)
    f = open(file_name, 'wb')
    pickle.dump(args, f)
    f.close()
    mat_name = 'experiment2_results/imaged_intens_{:1.1e}.mat'.format(args.noise_level)
    savemat(mat_name, {'imaged_intens': args.imaged_intens})


def eval_all_matlab_results2(args):
    args.mode = "simulator"
    simulate_data(args)
    solvers = ['EPRY']
    for solver in solvers:
        noise_level = 5e-5
        file_name = 'experiment2_results/{}_{:1.1e}.mat'.format(solver, noise_level)
        data = loadmat(file_name, struct_as_record=False)
        args.est_ampli = data['est_ampli']
        args.est_phase = data['est_phase']
        args.pupil_phase_estimate = data['pupil_phase_estimate']
        Metrics(args)
        data['est_ampli_norm'] = args.est_ampli_norm
        data['est_phase_norm'] = args.est_phase_norm
        data['est_pupil_phase_norm'] = args.est_pupil_phase_norm
        data['mae_ampli'] = args.mae_ampli
        data['psnr_ampli'] = args.psnr_ampli
        data['ssim_ampli'] = args.ssim_ampli
        data['mae_phase'] = args.mae_phase
        data['psnr_phase'] = args.psnr_phase
        data['ssim_phase'] = args.ssim_phase

        data['mae_pupil_phase'] = args.mae_pupil_phase
        data['psnr_pupil_phase'] = args.psnr_pupil_phase
        data['ssim_pupil_phase'] = args.ssim_pupil_phase
        savemat(file_name, data)


def eval_all_NN_results2():
    estimators = ['layer', 'CA']
    # estimators = ['IE', 'PA']
    for estimator in estimators:
        noise_level = 5e-5
        file_name = 'experiment2_results/args_{:1.1e}'.format(noise_level)
        f = open(file_name, 'rb')
        args = pickle.load(f)
        f.close()
        args.mode = "estimator"
        args.result_dir = time.strftime('./results/%Y_%m_%d_%H_%M_%S', time.localtime(time.time()))
        if estimator == 'PA':
            args.pupil_estimator = 'CA'
            args.channel_estimator = 'None'
        elif estimator == 'layer':
            args.pupil_estimator = 'layer'
            args.channel_estimator = 'CA'
        elif estimator == 'CA':
            args.pupil_estimator = 'CA'
            args.channel_estimator = 'CA'
        fpm_reconstruction = FPM_reconstruction(args)
        fpm_reconstruction.train_model()
        fpm_reconstruction.eval_model()
        Metrics(args)
        data = {}
        data['est_ampli'] = args.est_ampli
        data['est_phase'] = args.est_phase
        data['est_ampli_norm'] = args.est_ampli_norm
        data['est_phase_norm'] = args.est_phase_norm
        if hasattr(args, 'pupil_phase_estimate'):
            data['pupil_phase_estimate'] = args.pupil_phase_estimate

            data['est_pupil_phase_norm'] = args.est_pupil_phase_norm
        else:
            data['pupil_phase_estimate'] = 0

        data['mae_ampli'] = args.mae_ampli
        data['psnr_ampli'] = args.psnr_ampli
        data['ssim_ampli'] = args.ssim_ampli
        data['mae_phase'] = args.mae_phase
        data['psnr_phase'] = args.psnr_phase
        data['ssim_phase'] = args.ssim_phase

        data['mae_pupil_phase'] = args.mae_pupil_phase
        data['psnr_pupil_phase'] = args.psnr_pupil_phase
        data['ssim_pupil_phase'] = args.ssim_pupil_phase
        save_name = 'experiment2_results/PbNN-{}_{:1.1e}.mat'.format(estimator, noise_level)
        savemat(save_name, data)


def figure_4_1():
    pupil_phase_results = np.zeros((64, 64, 4))
    phase_coeff_results = np.zeros((15, 4))
    mae_pupil_phase = np.zeros((4,1))
    psnr_pupil_phase = np.zeros((4,1))
    ssim_pupil_phase = np.zeros((4,1))
    solvers = ['EPRY', 'PbNN-layer', 'PbNN-CA']
    legends = ['EPRY', 'PbNN-layer', 'PbNN-CA']
    for solver_idx in range(len(solvers)):
        solver = solvers[solver_idx]
        noise_level = 5e-5
        file_name = 'experiment2_results/{}_{:1.1e}.mat'.format(solver, noise_level)
        data = loadmat(file_name, struct_as_record=False)
        pupil_phase_results[:,:,solver_idx] = data['pupil_phase_estimate']
        # pupil_phase_results[:,:,solver_idx] = data['est_pupil_phase_norm']
        phase_coeff_results[:,solver_idx] = data['phase_coeff'][:,0]
        mae_pupil_phase[solver_idx, 0] = data['mae_pupil_phase']
        psnr_pupil_phase[solver_idx, 0] = data['psnr_pupil_phase']
        ssim_pupil_phase[solver_idx, 0] = data['ssim_pupil_phase']

    matplotlib.use('TkAgg')
    plt.rcParams['font.sans-serif'] = ['Times New Roman']  # 用来正常显示中文标签
    plt.rcParams['font.size'] = 12
    plt.rc('axes', axisbelow=True)
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']

    fig1, axeses = plt.subplots(nrows=2, ncols=4, figsize=(8, 3.6),
                                gridspec_kw={"width_ratios": [1, 1, 1, 1],
                                             "height_ratios": [1, 1]})
    plt.subplots_adjust(top=0.94, bottom=0.08, left=0.06, right=0.94, hspace=0.4, wspace=0.5)

    for col in range(axeses.shape[1]):
        axeses[0, col].set_axis_off()


    file_name = 'experiment2_results/args_5.0e-05'
    f = open(file_name, 'rb')
    args = pickle.load(f)
    f.close()

    for idx in range(4):
        if idx == 0:
            ax1 = axeses[0, idx].imshow(pupil_phase_results[:,:,idx], cmap='viridis',clim=(0, 3))
        elif idx == 3:
            ax1 = axeses[0, idx].imshow(args.pupil_phase, cmap='viridis',clim=(-1, 1.5))
        else:
            ax1 = axeses[0, idx].imshow(pupil_phase_results[:,:,idx], cmap='viridis',clim=(-1, 1.5))
        cax1 = fig1.add_axes([axeses[0, idx].get_position().x1 + 0.01, axeses[0, idx].get_position().y0, 0.01,
                            axeses[0, idx].get_position().y1 - axeses[0, idx].get_position().y0])
        plt.colorbar(ax1, cax=cax1)
        cax1.tick_params(labelsize=10)  # 设置色标刻度字体大小。
        axeses[1, idx].grid(True)
        if idx == 3:
            axeses[1, idx].bar(range(15), args.zernike_coeff[:,0,0])
        else:
            axeses[1, idx].bar(range(15), phase_coeff_results[:,idx])
        axeses[1, idx].set_ylim([-1, 1])
        axeses[1, idx].set_yticks([-1, -0.5, 0, 0.5, 1])
        axeses[1, idx].set_yticklabels(('-1', '', '0', '', '1'), fontsize=10)
        axeses[1, idx].set_xticks([0, 5, 10, 15])
        axeses[1, idx].set_xticklabels(('0', '5', '10', '15'), fontsize=10)


    axeses[0, 0].text(32, -7, 'EPRY', verticalalignment='center', horizontalalignment='center')
    axeses[0, 1].text(32, -7, 'PbNN-layer', verticalalignment='center', horizontalalignment='center')
    axeses[0, 2].text(32, -7, 'PbNN-CA', verticalalignment='center', horizontalalignment='center')
    axeses[0, 3].text(32, -7, 'Ground truth', verticalalignment='center', horizontalalignment='center')

    axeses[0, 0].text(-17, 32, 'Pupil phase', rotation=90, verticalalignment='center', horizontalalignment='center')
    axeses[1, 0].text(-5, 0, 'Coefficients', rotation=90, verticalalignment='center', horizontalalignment='center')

    axeses[0, 0].text(6, 72, '{:0.3f}'.format(mae_pupil_phase[0, 0]), verticalalignment='center', horizontalalignment='center', fontsize=11, color='black')
    axeses[0, 0].text(19, 72, '/', verticalalignment='center', horizontalalignment='center', fontsize=11, color='black')
    axeses[0, 0].text(32, 72, '{:2.2f}'.format(psnr_pupil_phase[0, 0]), verticalalignment='center', horizontalalignment='center', fontsize=11, color='black')
    axeses[0, 0].text(45, 72, '/', verticalalignment='center', horizontalalignment='center', fontsize=11, color='black')
    axeses[0, 0].text(59, 72, '{:0.3f}'.format(ssim_pupil_phase[0, 0]), verticalalignment='center', horizontalalignment='center', fontsize=11, color='black')

    axeses[0, 1].text(6, 72, '{:0.3f}'.format(mae_pupil_phase[1, 0]), verticalalignment='center', horizontalalignment='center', fontsize=11, color='green')
    axeses[0, 1].text(19, 72, '/', verticalalignment='center', horizontalalignment='center', fontsize=11, color='black')
    axeses[0, 1].text(32, 72, '{:2.2f}'.format(psnr_pupil_phase[1, 0]), verticalalignment='center', horizontalalignment='center', fontsize=11, color='green')
    axeses[0, 1].text(45, 72, '/', verticalalignment='center', horizontalalignment='center', fontsize=11, color='black')
    axeses[0, 1].text(58, 72, '{:0.3f}'.format(ssim_pupil_phase[1, 0]), verticalalignment='center', horizontalalignment='center', fontsize=11, color='green')

    axeses[0, 2].text(6, 72, '{:0.3f}'.format(mae_pupil_phase[2, 0]), verticalalignment='center', horizontalalignment='center', fontsize=11, color='red')
    axeses[0, 2].text(19, 72, '/', verticalalignment='center', horizontalalignment='center', fontsize=11, color='black')
    axeses[0, 2].text(32, 72, '{:2.2f}'.format(psnr_pupil_phase[2, 0]), verticalalignment='center', horizontalalignment='center', fontsize=11, color='red')
    axeses[0, 2].text(45, 72, '/', verticalalignment='center', horizontalalignment='center', fontsize=11, color='black')
    axeses[0, 2].text(58, 72, '{:0.3f}'.format(ssim_pupil_phase[2, 0]), verticalalignment='center', horizontalalignment='center', fontsize=11, color='red')

    plt.savefig('figures/figure_4_1.pdf', dpi=600)
    plt.savefig('figures/figure_4_1.jpg', dpi=600)
    plt.show()
    plt.close(fig1)


def simulate_all_noise3(args):
    args.zernike_coeff = np.zeros((15, 1, 1), dtype=np.float32)
    # args.zernike_coeff[5, 0, 0] = -0.5
    # args.zernike_coeff[6, 0, 0] = 0.4
    # args.zernike_coeff[7, 0, 0] = 0.6
    # args.zernike_coeff[9, 0, 0] = -0.4
    # args.zernike_coeff[12, 0, 0] = 0.5
    # args.zernike_coeff[14, 0, 0] = -0.5
    # np.random.seed(3)
    # ratio = 0.2
    # args.zernike_coeff = 2*ratio*np.random.rand(15, 1, 1).astype(np.float32) - ratio
    args.channel_variaty = "cos4rand"  # None, cos4, random, cos4rand
    args.noise_level = 5e-5
    # args.noise_level = 0

    for variaty_level in np.arange(0, 0.51, 0.05):
    # variaty_level = 0.25
        args.variaty_level = variaty_level
        args.result_dir = time.strftime('./results/%Y_%m_%d_%H_%M_%S', time.localtime(time.time()))
        args.mode = "simulator"
        simulate_data(args)
        file_name = 'experiment3_results/args_{:0.2f}'.format(args.variaty_level)
        f = open(file_name, 'wb')
        pickle.dump(args, f)
        f.close()
        mat_name = 'experiment3_results/imaged_intens_{:0.2f}.mat'.format(args.variaty_level)
        savemat(mat_name, {'imaged_intens': args.imaged_intens})


def eval_all_matlab_results3(args):
    simulate_data(args)
    solvers = ['AS', 'GS', 'GS_cos4']
    # solvers = ['GS_cos4']
    for solver in solvers:
        for variaty_level in np.arange(0, 0.51, 0.05):
            file_name = 'experiment3_results/{}_{:0.2f}.mat'.format(solver, variaty_level)
            data = loadmat(file_name, struct_as_record=False)
            args.est_ampli = data['est_ampli']
            args.est_phase = data['est_phase']
            Metrics(args)
            data['est_ampli_norm'] = args.est_ampli_norm
            data['est_phase_norm'] = args.est_phase_norm
            data['mae_ampli'] = args.mae_ampli
            data['psnr_ampli'] = args.psnr_ampli
            data['ssim_ampli'] = args.ssim_ampli
            data['mae_phase'] = args.mae_phase
            data['psnr_phase'] = args.psnr_phase
            data['ssim_phase'] = args.ssim_phase
            savemat(file_name, data)


def eval_all_NN_results3():
    estimators = ['None', 'cos4', 'IE', 'CA']
    for estimator in estimators:
        for variaty_level in np.arange(0, 0.51, 0.05):
        # variaty_level = 0.2

            file_name = 'experiment3_results/args_{:0.2f}'.format(variaty_level)
            f = open(file_name, 'rb')
            args = pickle.load(f)
            f.close()
            args.mode = "estimator"
            args.result_dir = time.strftime('./results/%Y_%m_%d_%H_%M_%S', time.localtime(time.time()))
            if estimator == 'IE':
                args.pupil_estimator = 'None'
                args.channel_estimator = 'CA'
            elif estimator == 'CA':
                args.pupil_estimator = 'CA'
                args.channel_estimator = 'CA'
            elif estimator == 'cos4':
                args.pupil_estimator = 'None'
                args.channel_estimator = 'cos4'
            elif estimator == 'None':
                args.pupil_estimator = 'None'
                args.channel_estimator = 'None'
            elif estimator == 'PA':
                args.pupil_estimator = 'CA'
                args.channel_estimator = 'None'
            fpm_reconstruction = FPM_reconstruction(args)
            fpm_reconstruction.train_model()
            fpm_reconstruction.eval_model()
            Metrics(args)
            data = {}
            data['est_ampli'] = args.est_ampli
            data['est_phase'] = args.est_phase
            data['est_ampli_norm'] = args.est_ampli_norm
            data['est_phase_norm'] = args.est_phase_norm
            if args.channel_estimator != 'None':
                data['channel_weights'] = args.channel_weights

            data['mae_ampli'] = args.mae_ampli
            data['psnr_ampli'] = args.psnr_ampli
            data['ssim_ampli'] = args.ssim_ampli
            data['mae_phase'] = args.mae_phase
            data['psnr_phase'] = args.psnr_phase
            data['ssim_phase'] = args.ssim_phase
            data['mae_LED'] = args.mae_LED
            data['psnr_LED'] = args.psnr_LED
            data['ssim_LED'] = args.ssim_LED

            save_name = 'experiment3_results/PbNN-{}_{:0.2f}.mat'.format(estimator, variaty_level)
            savemat(save_name, data)


def figure_4_2():
    LED_results = np.zeros((13, 13, 5))
    mae_ampli = np.zeros((7,11))
    psnr_ampli = np.zeros((7,11))
    ssim_ampli = np.zeros((7,11))
    mae_phase = np.zeros((7,11))
    psnr_phase = np.zeros((7,11))
    ssim_phase = np.zeros((7,11))
    mae_LED = np.zeros((7,11))
    psnr_LED = np.zeros((7,11))
    ssim_LED = np.zeros((7,11))
    # solvers = ['GS', 'PbNN-cos4', 'PbNN-IE', 'PbNN-CA']
    # legends = ['AP', 'PbNN-cos4', 'PbNN-IE', 'PbNN-CA']

    # legends = ['AP', 'AS', 'EPRY', 'PbNN', 'PbNN-IE', 'PbNN-PA', 'PbNN-layer', 'PbNN-CA']
    color_idxs = [0, 9, 3, 7]
    solvers = ['GS', 'GS_cos4', 'PbNN-None', 'PbNN-CA']
    legends = ['AP', 'AP-cos4', 'PbNN', 'PbNN-CA']
    variaty_levels = np.arange(0, 0.51, 0.05)
    for solver_idx in range(len(solvers)):
        solver = solvers[solver_idx]
        for variaty_idx in range(len(variaty_levels)):
            variaty_level = variaty_levels[variaty_idx]
            file_name = 'experiment3_results/{}_{:0.2f}.mat'.format(solver, variaty_level)
            data = loadmat(file_name, struct_as_record=False)
            mae_ampli[solver_idx, variaty_idx] = data['mae_ampli']
            psnr_ampli[solver_idx, variaty_idx] = data['psnr_ampli']
            ssim_ampli[solver_idx, variaty_idx] = data['ssim_ampli']
            mae_phase[solver_idx, variaty_idx] = data['mae_phase']
            psnr_phase[solver_idx, variaty_idx] = data['psnr_phase']
            ssim_phase[solver_idx, variaty_idx] = data['ssim_phase']

    plt.rcParams['font.sans-serif'] = ['Times New Roman']  # 用来正常显示中文标签
    plt.rcParams['font.size'] = 12
    plt.rc('axes', axisbelow=True)
    # plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    temp = colors[3]
    colors[3] = colors[7]
    colors[7] = temp
    grid = plt.GridSpec(2, 3, hspace=0.45, wspace=0.45)

    matplotlib.use('TkAgg')
    fig = plt.figure(figsize=(8, 5))
    # fig.set_dpi(72)
    plt.subplots_adjust(top=0.89, bottom=0.10, left=0.1, right=0.99)

    ax11 = fig.add_subplot(grid[0, 0])
    for solver_idx in range(len(solvers)):
        ax11.plot(variaty_levels, mae_ampli[solver_idx, :], linewidth=1.5, marker='o', markersize=2.5, color=colors[color_idxs[solver_idx]])
    ax11.set_xlabel('Fluctuation level')
    ax11.set_xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5])
    # ax11.set_xticklabels(['0', '10%', '20%', '30%', '40%', '50%'])
    ax11.set_ylim([0.03, 0.05])
    ax11.set_yticks([0.03, 0.035, 0.04, 0.045, 0.05])
    ax11.set_yticklabels(['0.03', '', '0.04', '', '0.05'])
    ax11.set_ylabel('MAE of amplitude')
    ax11.grid(True)

    ax12 = fig.add_subplot(grid[0, 1])
    for solver_idx in range(len(solvers)):
        ax12.plot(variaty_levels, psnr_ampli[solver_idx, :], linewidth=1.5, marker='o', markersize=2.5, color=colors[color_idxs[solver_idx]])
    ax12.set_xlabel('Fluctuation level')
    ax12.set_xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5])
    # ax12.set_xticklabels(['0', '10%', '20%', '30%', '40%', '50%'])
    ax12.legend(legends, bbox_to_anchor=(0.5, 1.1), loc='lower center', ncol=4)
    ax12.set_ylim([24, 27])
    ax12.set_ylabel('PSNR of amplitude')
    ax12.grid(True)

    ax13 = fig.add_subplot(grid[0, 2])
    for solver_idx in range(len(solvers)):
        ax13.plot(variaty_levels, ssim_ampli[solver_idx, :], linewidth=1.5, marker='o', markersize=2.5, color=colors[color_idxs[solver_idx]])
    ax13.set_xlabel('Fluctuation level')
    ax13.set_xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5])
    # ax13.set_xticklabels(['0', '10%', '20%', '30%', '40%', '50%'])
    ax13.set_ylim([0.8, 1])
    ax13.set_yticks([0.8, 0.85, 0.9, 0.95, 1])
    ax13.set_yticklabels(['0.8', '', '0.9', '', '1.0'])
    ax13.set_ylabel('SSIM of amplitude')
    ax13.grid(True)

    ax21 = fig.add_subplot(grid[1, 0])
    for solver_idx in range(len(solvers)):
        ax21.plot(variaty_levels, mae_phase[solver_idx, :], linewidth=1.5, marker='o', markersize=2.5, color=colors[color_idxs[solver_idx]])
    ax21.set_xlabel('Fluctuation level')
    ax21.set_xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5])
    # ax21.set_xticklabels(['0', '10%', '20%', '30%', '40%', '50%'])
    ax21.set_ylim([0.02, 0.08])
    ax21.set_ylabel('MAE of phase')
    ax21.grid(True)

    ax22 = fig.add_subplot(grid[1, 1])
    for solver_idx in range(len(solvers)):
        ax22.plot(variaty_levels, psnr_phase[solver_idx, :], linewidth=1.5, marker='o', markersize=2.5, color=colors[color_idxs[solver_idx]])
    ax22.set_xlabel('Fluctuation level')
    ax22.set_xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5])
    # ax22.set_xticklabels(['0', '10%', '20%', '30%', '40%', '50%'])
    ax22.set_ylim([20, 32])
    ax22.set_yticks([20, 24, 28, 32])
    ax22.set_ylabel('PSNR of phase')
    ax22.grid(True)

    ax23 = fig.add_subplot(grid[1, 2])
    for solver_idx in range(len(solvers)):
        ax23.plot(variaty_levels, ssim_phase[solver_idx, :], linewidth=1.5, marker='o', markersize=2.5, color=colors[color_idxs[solver_idx]])
    ax23.set_xlabel('Fluctuation level')
    ax23.set_xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5])
    # ax23.set_xticklabels(['0', '10%', '20%', '30%', '40%', '50%'])
    ax23.set_ylim([0.7, 1])
    ax23.set_ylabel('SSIM of phase')
    ax23.grid(True)

    fig.text(0.010, 0.89, '(a)', fontsize=14, bbox=dict(facecolor="k", alpha=0.1))
    fig.text(0.355, 0.89, '(b)', fontsize=14, bbox=dict(facecolor="k", alpha=0.1))
    fig.text(0.685, 0.89, '(c)', fontsize=14, bbox=dict(facecolor="k", alpha=0.1))
    fig.text(0.010, 0.41, '(e)', fontsize=14, bbox=dict(facecolor="k", alpha=0.1))
    fig.text(0.355, 0.41, '(f)', fontsize=14, bbox=dict(facecolor="k", alpha=0.1))
    fig.text(0.685, 0.41, '(g)', fontsize=14, bbox=dict(facecolor="k", alpha=0.1))


    plt.savefig('figures/figure_4_2.pdf', dpi=600)
    plt.savefig('figures/figure_4_2.jpg', dpi=600)
    plt.show()
    plt.close(fig)


def figure_4_3():
    ampli_results = np.zeros((256, 256, 4))
    phase_results = np.zeros((256, 256, 4))
    LED_results = np.zeros((13, 13, 4))
    mae_ampli = np.zeros((4,1))
    psnr_ampli = np.zeros((4,1))
    ssim_ampli = np.zeros((4,1))
    mae_phase = np.zeros((4,1))
    psnr_phase = np.zeros((4,1))
    ssim_phase = np.zeros((4,1))
    mae_LED = np.zeros((4,1))
    psnr_LED = np.zeros((4,1))
    ssim_LED = np.zeros((4,1))
    # solvers = ['GS', 'PbNN-cos4', 'PbNN-IE', 'PbNN-CA']
    # legends = ['AP', 'PbNN-cos4', 'PbNN-IE', 'PbNN-CA']
    solvers = ['PbNN-cos4', 'PbNN-CA']
    legends = ['PbNN-cos4', 'PbNN-CA']
    for solver_idx in range(len(solvers)):
        solver = solvers[solver_idx]
        variaty_level = 0.2

        file_name = 'experiment3_results/{}_{:0.2f}.mat'.format(solver, variaty_level)
        data = loadmat(file_name, struct_as_record=False)
        ampli_results[:,:,solver_idx] = data['est_ampli_norm']
        phase_results[:,:,solver_idx] = data['est_phase_norm']
        if solver != 'PbNN-None':
            LED_results[:,:,solver_idx] = data['channel_weights']
        mae_ampli[solver_idx, 0] = data['mae_ampli']
        psnr_ampli[solver_idx, 0] = data['psnr_ampli']
        ssim_ampli[solver_idx, 0] = data['ssim_ampli']
        mae_phase[solver_idx, 0] = data['mae_phase']
        psnr_phase[solver_idx, 0] = data['psnr_phase']
        ssim_phase[solver_idx, 0] = data['ssim_phase']
        mae_LED[solver_idx, 0] = data['mae_LED']
        psnr_LED[solver_idx, 0] = data['psnr_LED']
        ssim_LED[solver_idx, 0] = data['ssim_LED']

    matplotlib.use('TkAgg')
    plt.rcParams['font.sans-serif'] = ['Times New Roman']  # 用来正常显示中文标签
    plt.rcParams['font.size'] = 13
    plt.rc('axes', axisbelow=True)
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']

    fig1, axeses = plt.subplots(nrows=1, ncols=3, figsize=(5, 1.8),
                                gridspec_kw={"width_ratios": [1, 1, 1],
                                             "height_ratios": [1]})
    plt.subplots_adjust(top=0.95, bottom=0.05, left=0.02, right=0.9, hspace=0.2, wspace=0.15)

    for col in range(axeses.shape[0]):
        axeses[col].set_axis_off()

    for idx in range(3):
        axeses[idx].imshow(LED_results[:,:,idx], cmap='viridis',clim=(0, 1))

    file_name = 'experiment3_results/args_0.20'
    f = open(file_name, 'rb')
    args = pickle.load(f)
    f.close()

    IW_square = args.intensity_weights.reshape((13,13))
    ax1 = axeses[2].imshow(IW_square, cmap='viridis',clim=(0, 1))
    cax1 = fig1.add_axes([axeses[2].get_position().x1 + 0.01, axeses[2].get_position().y0, 0.01,
                        axeses[2].get_position().y1 - axeses[2].get_position().y0])
    plt.colorbar(ax1, cax=cax1)
    cax1.tick_params(labelsize=10)  # 设置色标刻度字体大小。

    # axeses[0, 0].text(128, -25, 'PbNN', verticalalignment='center', horizontalalignment='center')
    axeses[0].text(6.5, -1.8, 'PbNN-cos4', verticalalignment='center', horizontalalignment='center')
    axeses[1].text(6.5, -1.8, 'PbNN-CA', verticalalignment='center', horizontalalignment='center')
    axeses[2].text(6.5, -1.8, 'Ground truth', verticalalignment='center', horizontalalignment='center')

    # axeses[0, 0].text(-25, 128, 'Amplitude', rotation=90, verticalalignment='center', horizontalalignment='center')
    # axeses[1, 0].text(-25, 128, 'Phase', rotation=90, verticalalignment='center', horizontalalignment='center')
    # axeses[0].text(-1.8, 6.5, 'LED intensity', rotation=90, verticalalignment='center', horizontalalignment='center')

    # idx = 1
    # str = '{:0.3f} / {:2.2f} / {:0.3f}'.format(mae_LED[idx,0], psnr_LED[idx,0], ssim_LED[idx,0])
    # axeses[idx].text(6.5, 13.8, str, verticalalignment='center', horizontalalignment='center',fontsize=11)
    idx = 0
    axeses[idx].text(2, 13.8, '{:0.3f}'.format(mae_LED[idx, 0]), verticalalignment='center', horizontalalignment='center', fontsize=11, color='green')
    axeses[idx].text(4.3, 13.8, '/', verticalalignment='center', horizontalalignment='center', fontsize=11, color='black')
    axeses[idx].text(6.5, 13.8, '{:2.2f}'.format(psnr_LED[idx, 0]), verticalalignment='center', horizontalalignment='center', fontsize=11, color='green')
    axeses[idx].text(8.7, 13.8, '/', verticalalignment='center', horizontalalignment='center', fontsize=11, color='black')
    axeses[idx].text(11, 13.8, '{:0.3f}'.format(ssim_LED[idx, 0]), verticalalignment='center', horizontalalignment='center', fontsize=11, color='green')

    idx = 1
    axeses[idx].text(2, 13.8, '{:0.3f}'.format(mae_LED[idx, 0]), verticalalignment='center', horizontalalignment='center', fontsize=11, color='red')
    axeses[idx].text(4.3, 13.8, '/', verticalalignment='center', horizontalalignment='center', fontsize=11, color='black')
    axeses[idx].text(6.5, 13.8, '{:2.2f}'.format(psnr_LED[idx, 0]), verticalalignment='center', horizontalalignment='center', fontsize=11, color='red')
    axeses[idx].text(8.7, 13.8, '/', verticalalignment='center', horizontalalignment='center', fontsize=11, color='black')
    axeses[idx].text(11, 13.8, '{:0.3f}'.format(ssim_LED[idx, 0]), verticalalignment='center', horizontalalignment='center', fontsize=11, color='red')

    plt.savefig('figures/figure_4_3.pdf', dpi=600)
    plt.savefig('figures/figure_4_3.jpg', dpi=600)
    plt.show()
    plt.close(fig1)


def simulate_all_noise4(args):
    args.mode = "real_data"
    # args.lr = 5e-2
    # args.lr_pupil = 5e-3
    # args.lr_cam = 0.1
    # args.epochs = 50
    # args.loss = "1*L1+5*L2+1e-3*FPM+5e-3*TV"
    args.lr = 2e-2
    args.lr_pupil = 5e-3
    args.lr_cam = 0.1
    args.epochs = 200
    args.loss = "1*L1+20*L2+1e-3*FPM+4e-2*TV"
    args.result_dir = time.strftime('./results/%Y_%m_%d_%H_%M_%S', time.localtime(time.time()))
    load_real_data(args)
    file_name = 'experiment4_results/args'
    f = open(file_name, 'wb')
    pickle.dump(args, f)
    f.close()
    mat_name = 'experiment4_results/imaged_intens.mat'
    savemat(mat_name, {'imaged_intens': args.imaged_intens})


def eval_all_NN_results4():
    estimators = ['None', 'layer', 'CA']
    # estimators = ['None']
    for estimator in estimators:
        file_name = 'experiment4_results/args'
        f = open(file_name, 'rb')
        args = pickle.load(f)
        f.close()
        args.mode = "estimator"
        args.result_dir = time.strftime('./results/%Y_%m_%d_%H_%M_%S', time.localtime(time.time()))
        if estimator == 'layer':
            args.lr = 5e-2
            args.lr_pupil = 1e-1
            args.lr_cam = 0.1
            args.epochs = 50
            args.loss = "1*L1+5*L2+1e-4*FPM+1e-3*TV"
            args.pupil_estimator = 'layer'
            args.channel_estimator = 'None'
        elif estimator == 'None':
            args.lr = 5e-2
            args.lr_pupil = 5e-3
            args.lr_cam = 0.1
            args.epochs = 50
            args.loss = "1*L1+5*L2+1e-4*FPM+1e-3*TV"
            args.pupil_estimator = 'None'
            args.channel_estimator = 'None'
        elif estimator == 'PA':
            args.pupil_estimator = 'CA'
            args.channel_estimator = 'None'
        fpm_reconstruction = FPM_reconstruction(args)
        fpm_reconstruction.train_model()
        fpm_reconstruction.eval_model()
        Metrics(args)
        data = {}
        data['est_ampli'] = args.est_ampli
        data['est_phase'] = args.est_phase
        data['est_ampli_norm'] = args.est_ampli_norm
        data['est_phase_norm'] = args.est_phase_norm
        if args.channel_estimator != 'None':
            data['channel_weights'] = args.channel_weights

        if hasattr(args, 'pupil_phase_estimate'):
            data['pupil_phase_estimate'] = args.pupil_phase_estimate

        save_name = 'experiment4_results/PbNN-{}.mat'.format(estimator)
        savemat(save_name, data)


def figure_4_6():
    calibrate_weights = loadmat('cos4_weights.mat')['intensity_weights']
    zernike_pupils = loadmat('generate_zernike_pupils/zernike_pupils_256.mat')['zernike_pupils'].astype(np.float32)
    pupil0 = zernike_pupils[:, :, 0:1]
    LED_results = np.ones((13, 13, 7))
    ampli_results = np.zeros((256, 256, 7))
    phase_results = np.zeros((256, 256, 7))
    # pupil_ampli_results = np.zeros((64, 64, 7))
    pupil_ampli_results = np.tile(pupil0, [1, 1, 7])
    pupil_phase_results = np.zeros((64, 64, 7))
    solvers = ['PbNN-CA', 'PbNN-layer', 'PbNN-None', 'EPRY', 'AS', 'GS_cos4', 'GS']
    legends = ['PbNN-CA', 'PbNN-layer', 'PbNN', 'EPRY', 'AS', 'AP-cos4', 'AP']
    for solver_idx in range(len(solvers)):
        solver = solvers[solver_idx]
        file_name = 'experiment4_results/{}.mat'.format(solver)
        data = loadmat(file_name, struct_as_record=False)
        ampli_results[:,:,solver_idx] = data['est_ampli']
        phase_results[:,:,solver_idx] = data['est_phase']
        if solver == 'EPRY' or solver == 'PbNN-layer' or solver == 'PbNN-CA':
            pupil_phase_results[:,:,solver_idx] = data['pupil_phase_estimate']
        if solver == 'PbNN-CA' or solver == 'PbNN-IE':
            LED_results[:,:,solver_idx] = data['channel_weights']
        if solver == 'GS_cos4':
            LED_results[:,:,solver_idx] = calibrate_weights.reshape((13,13))

    matplotlib.use('TkAgg')
    plt.rcParams['font.sans-serif'] = ['Times New Roman']  # 用来正常显示中文标签
    plt.rcParams['font.size'] = 12
    plt.rc('axes', axisbelow=True)
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']


    fig = plt.figure(figsize=(8, 8))

    solvers_idx = [3, 1, 0]
    max_phase = 0.5
    min_phase = -0.5

    # epry结果
    ax00 = fig.add_axes([0.045, 0.8, 0.14, 0.14])
    ax00.set_axis_off()
    ax00.imshow(ampli_results[:,:,3], cmap='gray',clim=(np.min(ampli_results[:, :, 3]), np.max(ampli_results[:, :, 3])))
    ax00.text(128, -24, 'EPRY', verticalalignment='center', horizontalalignment='center')
    ax00.text(-24, 128, 'Amplitude', rotation = 90, verticalalignment='center', horizontalalignment='center')

    ax10 = fig.add_axes([0.045, 0.65, 0.14, 0.14])
    ax10.set_axis_off()
    ax10.imshow(phase_results[:,:,3], cmap='gray',clim=(min_phase, max_phase))
    ax10.text(-24, 128, 'Phase', rotation = 90, verticalalignment='center', horizontalalignment='center')

    ax20 = fig.add_axes([0.045, 0.5, 0.14, 0.14])
    ax20.set_axis_off()
    ax20.imshow(pupil_phase_results[:,:,3], cmap='viridis',clim=(-1, 1))
    ax20.text(-6, 32, 'Pupil phase', rotation = 90, verticalalignment='center', horizontalalignment='center')

    # PbNN-layer结果
    ax01 = fig.add_axes([0.20, 0.8, 0.14, 0.14])
    ax01.set_axis_off()
    ax01.imshow(ampli_results[:,:,1], cmap='gray',clim=(np.min(ampli_results[:, :, 1]), np.max(ampli_results[:, :, 1])))
    ax01.text(128, -24, 'PbNN-layer', verticalalignment='center', horizontalalignment='center')

    ax11 = fig.add_axes([0.20, 0.65, 0.14, 0.14])
    ax11.set_axis_off()
    ax11.imshow(phase_results[:,:,1], cmap='gray',clim=(min_phase, max_phase))

    ax21 = fig.add_axes([0.20, 0.5, 0.14, 0.14])
    ax21.set_axis_off()
    ax21.imshow(pupil_phase_results[:,:,1], cmap='viridis',clim=(-1, 1))

    # PbNN-CA结果
    ax02 = fig.add_axes([0.355, 0.8, 0.14, 0.14])
    ax02.set_axis_off()
    ax02.imshow(ampli_results[:,:,0], cmap='gray',clim=(np.min(ampli_results[:, :, 0]), np.max(ampli_results[:, :, 0])))
    ax02.text(128, -24, 'PbNN-CA', verticalalignment='center', horizontalalignment='center')

    ax12 = fig.add_axes([0.355, 0.65, 0.14, 0.14])
    ax12.set_axis_off()
    ax12.imshow(phase_results[:,:,0], cmap='gray',clim=(min_phase, max_phase))

    ax22 = fig.add_axes([0.355, 0.5, 0.14, 0.14])
    ax22.set_axis_off()
    ax22.imshow(pupil_phase_results[:,:,0], cmap='viridis',clim=(-1, 1))

    # colorbar
    ax_img0 = ax02.get_images()[0]
    cax0 = fig.add_axes([ax02.get_position().x1 + 0.01, ax02.get_position().y0 + 0.01, 0.008, ax02.get_position().y1 - ax02.get_position().y0 - 0.02])
    cbar0 = plt.colorbar(ax_img0, cax=cax0)
    cbar0.set_ticks([np.min(ampli_results[:, :, 1]), np.max(ampli_results[:, :, 1])])
    cbar0.set_ticklabels( ('0', '1'))
    cax0.tick_params(labelsize=11)  # 设置色标刻度字体大小。

    ax_img1 = ax12.get_images()[0]
    cax1 = fig.add_axes([ax12.get_position().x1 + 0.01, ax12.get_position().y0 + 0.01, 0.008, ax12.get_position().y1 - ax12.get_position().y0 - 0.02])
    cbar1 = plt.colorbar(ax_img1, cax=cax1)
    cax1.tick_params(labelsize=11)  # 设置色标刻度字体大小。

    ax_img2 = ax22.get_images()[0]
    cax2 = fig.add_axes([ax22.get_position().x1 + 0.01, ax22.get_position().y0 + 0.01, 0.008, ax22.get_position().y1 - ax22.get_position().y0 - 0.02])
    cbar2 = plt.colorbar(ax_img2, cax=cax2)
    cax2.tick_params(labelsize=11)  # 设置色标刻度字体大小。

    ax00.add_patch(plt.Rectangle((-65, -65), 1020, 885, fill=False, edgecolor='orange', clip_on=False, linewidth=1.5, linestyle='--'))


    # AP-cos4结果
    ax03 = fig.add_axes([0.63, 0.8, 0.14, 0.14])
    ax03.set_axis_off()
    ax03.imshow(ampli_results[:,:,5], cmap='gray',clim=(np.min(ampli_results[:, :, 5]), np.max(ampli_results[:, :, 5])))
    ax03.text(128, -24, 'AP-cos4', verticalalignment='center', horizontalalignment='center')
    ax03.text(-24, 128, 'Amplitude', rotation = 90, verticalalignment='center', horizontalalignment='center')

    ax13 = fig.add_axes([0.63, 0.65, 0.14, 0.14])
    ax13.set_axis_off()
    ax13.imshow(phase_results[:,:,5], cmap='gray',clim=(min_phase, max_phase))
    ax13.text(-24, 128, 'Phase', rotation = 90, verticalalignment='center', horizontalalignment='center')

    ax23 = fig.add_axes([0.63, 0.5, 0.14, 0.14])
    ax23.set_axis_off()
    ax23.imshow(LED_results[:,:,5], cmap='viridis',clim=(0, 1))
    ax23.text(-1.5, 6.5, 'LED intensity', rotation = 90, verticalalignment='center', horizontalalignment='center')

    # PbNN-CA结果
    ax04 = fig.add_axes([0.785, 0.8, 0.14, 0.14])
    ax04.set_axis_off()
    ax04.imshow(ampli_results[:,:,0], cmap='gray',clim=(np.min(ampli_results[:, :, 0]), np.max(ampli_results[:, :, 0])))
    ax04.text(128, -24, 'PbNN-CA', verticalalignment='center', horizontalalignment='center')
    ax04.add_patch(plt.Rectangle((ax04.get_position().x0 + 160, ax04.get_position().y0 + 180), 106, 76, fill=True, facecolor='white', clip_on=False, alpha=0.8))
    ax04.text(210, 210, '20um', verticalalignment='center', horizontalalignment='center')
    ax04.add_patch(plt.Rectangle((ax04.get_position().x0 + 190, ax04.get_position().y0 + 230), 49.2, 10, fill=True, facecolor='black', clip_on=False, linewidth=0))

    ax14 = fig.add_axes([0.785, 0.65, 0.14, 0.14])
    ax14.set_axis_off()
    ax14.imshow(phase_results[:,:,0], cmap='gray',clim=(min_phase, max_phase))

    ax24 = fig.add_axes([0.785, 0.5, 0.14, 0.14])
    ax24.set_axis_off()
    ax24.imshow(LED_results[:,:,0], cmap='viridis',clim=(0, 1))

    ax_img3 = ax24.get_images()[0]
    cax3 = fig.add_axes([ax24.get_position().x1 + 0.01, ax24.get_position().y0 + 0.01, 0.008, ax24.get_position().y1 - ax24.get_position().y0 - 0.02])
    cbar3 = plt.colorbar(ax_img3, cax=cax3)
    cbar3.set_ticks([0, 1])
    cbar3.set_ticklabels( ('0', '1'))
    cax3.tick_params(labelsize=11)  # 设置色标刻度字体大小。

    ax03.add_patch(plt.Rectangle((-65, -65), 700, 885, fill=False, edgecolor='green', clip_on=False, linewidth=1.5, linestyle='--'))

    # AP结果
    ax31 = fig.add_axes([0.27, 0.29, 0.14, 0.14])
    ax31.set_axis_off()
    ax31.imshow(ampli_results[:, :, 6], cmap='gray',
                clim=(np.min(ampli_results[:, :, 6]), np.max(ampli_results[:, :, 6])))
    ax31.text(128, -24, 'AP', verticalalignment='center', horizontalalignment='center')
    ax31.text(-24, 128, 'Amplitude', rotation = 90, verticalalignment='center', horizontalalignment='center')

    ax41 = fig.add_axes([0.27, 0.14, 0.14, 0.14])
    ax41.set_axis_off()
    ax41.imshow(phase_results[:, :, 6], cmap='gray', clim=(min_phase, max_phase))
    ax41.text(-24, 128, 'Phase', rotation = 90, verticalalignment='center', horizontalalignment='center')

    # AS结果
    ax32 = fig.add_axes([0.425, 0.29, 0.14, 0.14])
    ax32.set_axis_off()
    ax32.imshow(ampli_results[:, :, 4], cmap='gray',
                clim=(np.min(ampli_results[:, :, 4]), np.max(ampli_results[:, :, 4])))
    ax32.text(128, -24, 'AS', verticalalignment='center', horizontalalignment='center')

    ax42 = fig.add_axes([0.425, 0.14, 0.14, 0.14])
    ax42.set_axis_off()
    ax42.imshow(phase_results[:, :, 4], cmap='gray', clim=(min_phase, max_phase))

    # PbNN结果
    ax33 = fig.add_axes([0.58, 0.29, 0.14, 0.14])
    ax33.set_axis_off()
    ax33.imshow(ampli_results[:, :,2], cmap='gray',
                clim=(np.min(ampli_results[:, :, 2]), np.max(ampli_results[:, :, 2])))
    ax33.text(128, -24, 'PbNN', verticalalignment='center', horizontalalignment='center')

    ax43 = fig.add_axes([0.58, 0.14, 0.14, 0.14])
    ax43.set_axis_off()
    ax43.imshow(phase_results[:, :, 2], cmap='gray', clim=(min_phase, max_phase))

    ax31.add_patch(plt.Rectangle((-65, -65), 910, 615, fill=False, edgecolor='blue', clip_on=False, linewidth=1.5, linestyle='--'))


    plt.savefig('figures/figure_4_6.pdf', dpi=600)
    plt.savefig('figures/figure_4_6.jpg', dpi=600)
    plt.show()
    plt.close(fig)


def simulate_all_noise5(args):
    args.channel_variaty = "cos4rand"  # None, cos4, random, cos4rand
    args.zernike_coeff = np.zeros((15, 1, 1), dtype=np.float32)
    args.zernike_coeff[4, 0, 0] = 0.1
    args.zernike_coeff[5, 0, 0] = -0.5
    args.zernike_coeff[6, 0, 0] = 0.4
    args.zernike_coeff[7, 0, 0] = 0.6
    args.zernike_coeff[8, 0, 0] = -0.1
    args.zernike_coeff[9, 0, 0] = -0.4
    args.zernike_coeff[12, 0, 0] = 0.5
    args.zernike_coeff[14, 0, 0] = -0.5
    args.noise_level = 5e-5
    args.result_dir = time.strftime('./results/%Y_%m_%d_%H_%M_%S', time.localtime(time.time()))
    args.mode = "simulator"
    simulate_data_star(args)
    file_name = 'experiment5_results/args_{:1.1e}'.format(args.noise_level)
    f = open(file_name, 'wb')
    pickle.dump(args, f)
    f.close()
    mat_name = 'experiment5_results/imaged_intens_{:1.1e}.mat'.format(args.noise_level)
    savemat(mat_name, {'imaged_intens': args.imaged_intens})


def eval_all_matlab_results5(args):
    args.mode = "simulator"
    simulate_data(args)
    solvers = ['AS', 'GS', 'EPRY', 'GS_cos4']
    for solver in solvers:
        noise_level = 5e-5
        file_name = 'experiment5_results/{}_{:1.1e}.mat'.format(solver, noise_level)
        data = loadmat(file_name, struct_as_record=False)
        args.est_ampli = data['est_ampli']
        args.est_phase = data['est_phase']
        args.pupil_ampli_estimate = data['pupil_ampli_estimate']
        args.pupil_phase_estimate = data['pupil_phase_estimate']
        Metrics(args)
        data['est_ampli_norm'] = args.est_ampli_norm
        data['est_phase_norm'] = args.est_phase_norm
        data['mae_ampli'] = args.mae_ampli
        data['psnr_ampli'] = args.psnr_ampli
        data['ssim_ampli'] = args.ssim_ampli
        data['mae_phase'] = args.mae_phase
        data['psnr_phase'] = args.psnr_phase
        data['ssim_phase'] = args.ssim_phase

        data['mae_pupil_phase'] = args.mae_pupil_phase
        data['psnr_pupil_phase'] = args.psnr_pupil_phase
        data['ssim_pupil_phase'] = args.ssim_pupil_phase
        savemat(file_name, data)

def eval_all_NN_results5():
    estimators = ['None', 'layer', 'CA']
    for estimator in estimators:
        noise_level = 5e-5
        file_name = 'experiment5_results/args_{:1.1e}'.format(noise_level)
        f = open(file_name, 'rb')
        args = pickle.load(f)
        f.close()
        args.mode = "estimator"
        args.result_dir = time.strftime('./results/%Y_%m_%d_%H_%M_%S', time.localtime(time.time()))
        if estimator == 'None':
            args.pupil_estimator = 'None'
            args.channel_estimator = 'None'
            args.lr = 1e-2
            args.lr_pupil = 2e-3
            args.lr_cam = 0.1
            args.epochs = 200
            args.loss = "1*L1+5*L2+5e-4*FPM+5e-4*TV"
        elif estimator == 'layer':
            args.pupil_estimator = 'layer'
            args.channel_estimator = 'None'
            args.lr = 1e-2
            args.lr_pupil = 2e-3
            args.lr_cam = 0.1
            args.epochs = 250
            args.loss = "1*L1+5*L2+5e-4*FPM+5e-4*TV"
        elif estimator == 'CA':
            args.pupil_estimator = 'CA'
            args.channel_estimator = 'CA'
            args.lr = 1e-2
            args.lr_pupil = 2e-3
            args.lr_cam = 0.1
            args.epochs = 250
            args.loss = "1*L1+5*L2+5e-4*FPM+5e-4*TV"
        else:
            args.pupil_estimator = estimator
        fpm_reconstruction = FPM_reconstruction(args)
        fpm_reconstruction.train_model()
        fpm_reconstruction.eval_model()
        Metrics(args)
        data = {}
        data['est_ampli'] = args.est_ampli
        data['est_phase'] = args.est_phase
        data['est_ampli_norm'] = args.est_ampli_norm
        data['est_phase_norm'] = args.est_phase_norm
        if hasattr(args, 'pupil_ampli_estimate'):
            data['pupil_ampli_estimate'] = args.pupil_ampli_estimate
            data['pupil_phase_estimate'] = args.pupil_phase_estimate
        else:
            data['pupil_ampli_estimate'] = 0
            data['pupil_phase_estimate'] = 0

        data['mae_ampli'] = args.mae_ampli
        data['psnr_ampli'] = args.psnr_ampli
        data['ssim_ampli'] = args.ssim_ampli
        data['mae_phase'] = args.mae_phase
        data['psnr_phase'] = args.psnr_phase
        data['ssim_phase'] = args.ssim_phase

        data['mae_pupil_phase'] = args.mae_pupil_phase
        data['psnr_pupil_phase'] = args.psnr_pupil_phase
        data['ssim_pupil_phase'] = args.ssim_pupil_phase
        save_name = 'experiment5_results/PbNN-{}_{:1.1e}.mat'.format(estimator, noise_level)
        savemat(save_name, data)


def get_line(img, center, radius):
    num = 360
    line = np.zeros((1, num))
    for i in range(num):
        angle = i/180*np.pi
        y = center[1] + radius * np.sin(angle)
        x = center[0] + radius * np.cos(angle)
        line[0, i] = cv2.remap(img, np.array([[y]], np.float32), np.array([[x]], np.float32), cv2.INTER_LINEAR)
    return line


def figure_4_7():
    ampli_results = np.zeros((256, 256, 8))
    phase_results = np.zeros((256, 256, 8))
    mae_ampli = np.zeros((8,1))
    psnr_ampli = np.zeros((8,1))
    ssim_ampli = np.zeros((8,1))
    mae_phase = np.zeros((8,1))
    psnr_phase = np.zeros((8,1))
    ssim_phase = np.zeros((8,1))
    lines = np.zeros((8,360))
    solvers = ['GS', 'GS_cos4', 'AS', 'EPRY', 'PbNN-None', 'PbNN-layer', 'PbNN-CA']
    legends = ['AP', 'AP-cos4', 'AS', 'EPRY', 'PbNN', 'PbNN-layer', 'PbNN-CA']
    for solver_idx in range(len(solvers)):
        solver = solvers[solver_idx]
        noise_level = 5e-5
        file_name = 'experiment5_results/{}_{:1.1e}.mat'.format(solver, noise_level)
        data = loadmat(file_name, struct_as_record=False)
        ampli_results[:,:,solver_idx] = data['est_ampli_norm']
        phase_results[:,:,solver_idx] = data['est_phase_norm']
        mae_ampli[solver_idx, 0] = data['mae_ampli']
        psnr_ampli[solver_idx, 0] = data['psnr_ampli']
        ssim_ampli[solver_idx, 0] = data['ssim_ampli']
        mae_phase[solver_idx, 0] = data['mae_phase']
        psnr_phase[solver_idx, 0] = data['psnr_phase']
        ssim_phase[solver_idx, 0] = data['ssim_phase']

    matplotlib.use('TkAgg')
    plt.rcParams['font.sans-serif'] = ['Times New Roman']  # 用来正常显示中文标签
    plt.rcParams['font.size'] = 12
    # plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

    fig1, axeses = plt.subplots(nrows=5, ncols=4, figsize=(6, 6.5),
                                gridspec_kw={"width_ratios": [1, 1, 1, 1],
                                             "height_ratios": [1, 1, 0.3, 1, 1]})
    plt.subplots_adjust(top=0.96, bottom=0.04, left=0.09, right=0.99, hspace=0.15, wspace=0.12)

    for col in range(axeses.shape[1]):
        axeses[0, col].set_axis_off()
        axeses[2, col].set_axis_off()
        axeses[3, col].set_axis_off()

    xx = np.arange(360)
    for idx in range(4):
        axeses[0, idx].imshow(ampli_results[:,:,idx], cmap='gray',clim=(0.0, 1))
        axeses[0, idx].text(128, -25, legends[idx], verticalalignment='center', horizontalalignment='center')
        axeses[0, idx].add_patch(plt.Circle((128, 128), 15, color='yellow', fill=False, linewidth=2, linestyle=':'))
        lines[idx, :] = get_line(ampli_results[:,:,idx], (128, 128), 15)
        axeses[1, idx].grid(True)
        axeses[1, idx].plot(xx, lines[idx, :])
        axeses[1, idx].set_ylim([-0.1, 1.2])
        axeses[1, idx].set_yticks([0, 0.25, 0.5, 0.75, 1])
        axeses[1, idx].set_xticks([0, 90, 180, 270, 360])
        axeses[1, idx].set_xticklabels(['0', '', '$\pi$', '', '2$\pi$'], fontsize=11)
        if idx == 0:
            axeses[1, idx].set_yticklabels(['0', '', '0.5', '', '1.0'], fontsize=11)
        else:
            axeses[1, idx].set_yticklabels(['', '', '', '', ''])
    axeses[0, 0].text(-80, 128, 'Amplitude', rotation=90, verticalalignment='center', horizontalalignment='center')
    axeses[1, 0].text(-150, 0.5, 'Line plot', rotation=90, verticalalignment='center', horizontalalignment='center')

    for idx in range(3):
        axeses[3, idx].imshow(ampli_results[:,:,idx+4], cmap='gray',clim=(0.0, 1))
        axeses[3, idx].text(128, -25, legends[idx+4], verticalalignment='center', horizontalalignment='center')
        axeses[3, idx].add_patch(plt.Circle((128, 128), 15, color='yellow', fill=False, linewidth=2, linestyle=':'))
        # axeses[1, idx].imshow(phase_results[:,:,idx], cmap='gray',clim=(0.0, 1))
        lines[idx+4, :] = get_line(ampli_results[:,:,idx+4], (128, 128), 15)
        axeses[4, idx].grid(True)
        axeses[4, idx].plot(xx, lines[idx+4, :])
        axeses[4, idx].set_ylim([-0.1, 1.2])
        axeses[4, idx].set_yticks([0, 0.25, 0.5, 0.75, 1])
        axeses[4, idx].set_xticks([0, 90, 180, 270, 360])
        axeses[4, idx].set_xticklabels(['0', '', '$\pi$', '', '2$\pi$'], fontsize=11)
        if idx == 0:
            axeses[4, idx].set_yticklabels(['0', '', '0.5', '', '1.0'], fontsize=11)
        else:
            axeses[4, idx].set_yticklabels(['', '', '', '', ''])
    axeses[3, 0].text(-80, 128, 'Amplitude', rotation=90, verticalalignment='center', horizontalalignment='center')
    axeses[4, 0].text(-150, 0.5, 'Line plot', rotation=90, verticalalignment='center', horizontalalignment='center')

    star = get_star(10)
    axeses[3, 3].imshow(star, cmap='gray', clim=(0.0, 1))
    axeses[3, 3].text(128, -25, 'Ground truth', verticalalignment='center', horizontalalignment='center')
    # axeses[1, idx].imshow(phase_results[:,:,idx], cmap='gray',clim=(0.0, 1))
    lines[7, :] = get_line(star, (128, 128), 15)
    axeses[4, 3].grid(True)
    axeses[4, 3].plot(xx, lines[7, :])
    axeses[4, 3].set_ylim([-0.1, 1.2])
    axeses[4, 3].set_yticks([0, 0.25, 0.5, 0.75, 1])
    axeses[4, 3].set_xticks([0, 90, 180, 270, 360])
    axeses[4, 3].set_xticklabels(['0', '', '$\pi$', '', '2$\pi$'], fontsize=11)
    axeses[4, 3].set_yticklabels(['', '', '', '', ''])

    axeses[3, 0].add_patch(plt.Rectangle((-100, -70), 1220, 4, fill=True, facecolor='gray', clip_on=False, linewidth=0))
    axeses[3, 3].add_patch(plt.Circle((128, 128), 15, color='yellow', fill=False, linewidth=2, linestyle=':'))

    plt.savefig('figures/figure_4_7.pdf', dpi=600)
    plt.savefig('figures/figure_4_7.jpg', dpi=600)
    plt.show()
    plt.close(fig1)


# def figure_4_7():
#     ampli_results = np.zeros((256, 256, 8))
#     phase_results = np.zeros((256, 256, 8))
#     mae_ampli = np.zeros((8,1))
#     psnr_ampli = np.zeros((8,1))
#     ssim_ampli = np.zeros((8,1))
#     mae_phase = np.zeros((8,1))
#     psnr_phase = np.zeros((8,1))
#     ssim_phase = np.zeros((8,1))
#     lines = np.zeros((8,360))
#     solvers = ['GS', 'GS_cos4', 'AS', 'EPRY', 'PbNN-None', 'PbNN-layer', 'PbNN-CA']
#     legends = ['AP', 'AP-cos4', 'AS', 'EPRY', 'PbNN', 'PbNN-layer', 'PbNN-CA']
#     for solver_idx in range(len(solvers)):
#         solver = solvers[solver_idx]
#         noise_level = 5e-5
#         file_name = 'experiment5_results/{}_{:1.1e}.mat'.format(solver, noise_level)
#         data = loadmat(file_name, struct_as_record=False)
#         ampli_results[:,:,solver_idx] = data['est_ampli_norm']
#         phase_results[:,:,solver_idx] = data['est_phase_norm']
#         mae_ampli[solver_idx, 0] = data['mae_ampli']
#         psnr_ampli[solver_idx, 0] = data['psnr_ampli']
#         ssim_ampli[solver_idx, 0] = data['ssim_ampli']
#         mae_phase[solver_idx, 0] = data['mae_phase']
#         psnr_phase[solver_idx, 0] = data['psnr_phase']
#         ssim_phase[solver_idx, 0] = data['ssim_phase']
#
#     matplotlib.use('TkAgg')
#     plt.rcParams['font.sans-serif'] = ['Times New Roman']  # 用来正常显示中文标签
#     plt.rcParams['font.size'] = 12
#     # plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
#
#     fig1, axeses = plt.subplots(nrows=2, ncols=7, figsize=(10, 3.2),
#                                 gridspec_kw={"width_ratios": [1, 1, 1, 1, 1, 1, 1],
#                                              "height_ratios": [1, 1]})
#     plt.subplots_adjust(top=0.92, bottom=0.08, left=0.055, right=0.99, hspace=0.15, wspace=0.12)
#
#     for col in range(axeses.shape[1]):
#         axeses[0, col].set_axis_off()
#
#     xx = np.arange(360)
#     for idx in range(7):
#         axeses[0, idx].imshow(ampli_results[:,:,idx], cmap='gray',clim=(0.0, 1))
#         axeses[0, idx].text(128, -25, legends[idx], verticalalignment='center', horizontalalignment='center')
#         # axeses[1, idx].imshow(phase_results[:,:,idx], cmap='gray',clim=(0.0, 1))
#         lines[idx, :] = get_line(ampli_results[:,:,idx], (128, 128), 15)
#         axeses[1, idx].grid(True)
#         axeses[1, idx].plot(xx, lines[idx, :])
#         axeses[1, idx].set_ylim([-0.1, 1.2])
#         axeses[1, idx].set_yticks([0, 0.25, 0.5, 0.75, 1])
#         axeses[1, idx].set_xticks([0, 90, 180, 270, 360])
#         axeses[1, idx].set_xticklabels(['0', '', '$\pi$', '', '2$\pi$'], fontsize=11)
#         if idx == 0:
#             axeses[1, idx].set_yticklabels(['0', '', '0.5', '', '1.0'], fontsize=11)
#         else:
#             axeses[1, idx].set_yticklabels(['', '', '', '', ''])
#     axeses[0, 0].text(-80, 128, 'Amplitude', rotation=90, verticalalignment='center', horizontalalignment='center')
#     axeses[1, 0].text(-150, 0.5, 'Line plot', rotation=90, verticalalignment='center', horizontalalignment='center')
#
#
#
#     plt.savefig('figures/figure_4_7.pdf', dpi=600)
#     plt.savefig('figures/figure_4_7.jpg', dpi=600)
#     plt.show()
#     plt.close(fig1)


        #
# def simulate_all_noise5(args2):
#     args2.mode = "real_data"
#     args2.lr = 5e-2
#     args2.lr_pupil = 2e-3
#     args2.lr_cam = 1e-3
#     args2.epochs = 500
#     args2.loss = "1*L1+5*L2+1e-3*FPM+5e-3*TV"
#     args2.result_dir = time.strftime('./results/%Y_%m_%d_%H_%M_%S', time.localtime(time.time()))
#     load_real_data2(args2)
#     file_name = 'experiment5_results/args2'
#     f = open(file_name, 'wb')
#     pickle.dump(args2, f)
#     f.close()
#     mat_name = 'experiment5_results/imaged_intens.mat'
#     savemat(mat_name, {'imaged_intens': args2.imaged_intens})
#
#
# def eval_all_NN_results5():
#     # estimators = ['None', 'layer', 'CA']
#     estimators = ['None']
#     for estimator in estimators:
#         file_name = 'experiment5_results/args2'
#         f = open(file_name, 'rb')
#         args = pickle.load(f)
#         f.close()
#         args.mode = "estimator"
#         args.result_dir = time.strftime('./results/%Y_%m_%d_%H_%M_%S', time.localtime(time.time()))
#         if estimator == 'layer':
#             args.pupil_estimator = 'layer'
#             args.channel_estimator = 'None'
#         elif estimator == 'CA':
#             args.pupil_estimator = 'CA'
#             args.channel_estimator = 'CA'
#         elif estimator == 'None':
#             args.lr = 5e-2
#             args.lr_pupil = 5e-3
#             args.lr_cam = 0.1
#             args.epochs = 50
#             args.loss = "1*L1+5*L2+1e-4*FPM+1e-3*TV"
#             args.pupil_estimator = 'None'
#             args.channel_estimator = 'None'
#         fpm_reconstruction = FPM_reconstruction(args)
#         fpm_reconstruction.train_model()
#         fpm_reconstruction.eval_model()
#         Metrics(args)
#         data = {}
#         data['est_ampli'] = args.est_ampli
#         data['est_phase'] = args.est_phase
#         data['est_ampli_norm'] = args.est_ampli_norm
#         data['est_phase_norm'] = args.est_phase_norm
#         if args.channel_estimator != 'None':
#             data['channel_weights'] = args.channel_weights
#
#         if hasattr(args, 'pupil_phase_estimate'):
#             data['pupil_phase_estimate'] = args.pupil_phase_estimate
#
#         save_name = 'experiment5_results/PbNN-{}.mat'.format(estimator)
#         savemat(save_name, data)
#

if __name__ == '__main__':
    # figure_3_1()
    # figure_3_2()
    # figure_3_3()

    # simulate_all_noise(args)
    # 运行matlab脚本process1.m
    # eval_all_matlab_results(args)
    # eval_all_NN_results()
    # figure_4_4()
    # figure_4_5()

    # simulate_all_noise2(args)
    # 运行matlab脚本process2.m
    # eval_all_matlab_results2(args)
    # eval_all_NN_results2()
    # 运行matlab脚本calculate_coeffs.m
    figure_4_1()

    # simulate_all_noise3(args)
    # 运行matlab脚本process3.m
    # eval_all_matlab_results3(args)
    # eval_all_NN_results3()
    # figure_4_2()
    # figure_4_3()

    # simulate_all_noise4(args)
    # 运行matlab脚本process4.m
    # eval_all_NN_results4()
    # figure_4_6()


    # simulate_all_noise5(args)
    # 运行matlab脚本process5.m
    # eval_all_matlab_results5(args)
    # eval_all_NN_results5()
    # figure_4_7()