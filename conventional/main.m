clear;
% varables.data_dir = 'E:\Dataset\FPM_DLSR_data_FPNN\2018-05-21-18-18\G\';
varables.data_dir = 'E:\Dataset\MISR\2019-06-18-21-35-blood_cell\Raw\Raw_0006\G\';
% varables.cx = 1280;
% varables.cy = 1080;
% varables.patch_cx = 1180;
% varables.patch_cy = 880;
varables.cx = 136;
varables.cy = 116;
varables.patch_cx = 156;
varables.patch_cy = 136;
varables.half_width = 64;
% varables = import_color( varables );
varables.thre = 140/65535;
varables = import_data(varables);

varables.loop_num = 20;
varables.solver = 'EPRY';

% varables.solver = 'GS_global_Poisson';
varables = fpm_reconstruction(varables);
% varables = spectrum_prop(varables);

% varables.tol = 1e-3;
% varables = aft_process(varables);
% figure(2);
% subplot(2,2,1),imshow(varables.FPM_inten,[]);
% subplot(2,2,2),imshow(varables.FPM_phase,[]);
% subplot(2,2,3),imshow(varables.out_inten,[]);
% subplot(2,2,4),imshow(varables.out_phase,[]);

% varables = wavelet_fusion(varables);
% figure(3);
% subplot(1,3,1),imshow(varables.LR_color);
% subplot(1,3,2),imshow(varables.HR_color_fusion);
% subplot(1,3,3),imshow(varables.HR_color);
