% load('../experiment4_results/imaged_intens.mat');
load('E:/Dataset/FPM_DLSR_data_FPNN/FPM_raw_sim_20210907/0000.mat');
imaged_intens = sim_inten;
load('../cos4_weights.mat');
varables.solver = 'EPRY';
if varables.solver == "GS_cos4"
    imaged_intens = imaged_intens ./intensity_weights;
end
varables.imaged_intens = permute(imaged_intens,[2 3 1]);
varables.cx = 1280;
varables.cy = 1080;
varables.patch_cx = 1280;
varables.patch_cy = 1180;
% varables.half_width = 32;
varables.loop_num = 1;
% varables.hide = 0;
varables = fpm_reconstruction(varables);
est_ampli = varables.FPM_ampli;
est_phase = varables.FPM_phase;
pupil_ampli_estimate = abs(varables.pupil_fun);
pupil_phase_estimate = angle(varables.pupil_fun);

figure;
subplot(2,2,1),imshow(est_ampli,[])
subplot(2,2,2),imshow(est_phase,[])
subplot(2,2,3),imshow(pupil_ampli_estimate,[-1,1])
subplot(2,2,4),imshow(pupil_phase_estimate,[-1,1])
