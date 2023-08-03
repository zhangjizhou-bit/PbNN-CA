clear;
clc;

real_parameters;
HEI = 256;
WID = 256;

hei = HEI/downsample_ratio;
wid = WID/downsample_ratio;
deltaFy = 1/(hei*uniform_px);        % the value (in spatial frequency units) of each pixel in the fourier domain
deltaFx = 1/(wid*uniform_px);        % the value (in spatial frequency units) of each pixel in the fourier domain
deltaF = min([deltaFx, deltaFy]);
pupil_radius = round(NA_obj/wavelength/deltaF);             % f0/��fu in pixels
dd = sqrt(LED_x.^2 + LED_y.^2 + illumination_distance^2);	% distance from each LED to sample
sin_thetay = LED_y./dd;
sin_thetax = LED_x./dd;
kiy = sin_thetay/wavelength;	% y component of wave vector
kix = sin_thetax/wavelength;	% x component of wave vector
center_y = ceil((HEI+1)/2);
center_x = ceil((WID+1)/2);
LED_fy = kiy/deltaF + center_y;	% y coordinate in the fourier domain
LED_fx = kix/deltaF + center_x;	% x coordinate in the fourier domain

[x, y] = meshgrid((-fix(wid/2):ceil(wid/2)-1),(-fix(hei/2):ceil(hei/2)-1));
[~,R] = cart2pol(x,y);
pupil = ~(R>pupil_radius);

solvers = {'EPRY', 'PbNN-layer', 'PbNN-CA'};
% solvers = {'EPRY'};
for idx = 1:length(solvers)
    solver = solvers{idx};
%     for noise_level = 1e-5:1e-5:2e-4
    noise_level = 5e-5;
    
    fprintf('noise_level: %1.1e\n',noise_level);
    mat_name = ['../experiment2_results/',solver,'_',num2str(noise_level,'%1.1e'),'.mat'];
    load(mat_name);
%     [DC, ampli_coeff] = ZernikeCalc(1:1:21, est_pupil_ampli_norm, pupil,'STANDARD');
    [DC, phase_coeff] = ZernikeCalc(1:1:15, pupil_phase_estimate, pupil,'STANDARD');
%     [DC, phase_coeff] = ZernikeCalc(1:1:15, est_pupil_phase_norm, pupil,'STANDARD');
    clear pupil_phase_estimate;
%     save(mat_name, 'ampli_coeff', 'phase_coeff', '-append');
    save(mat_name, 'phase_coeff', '-append');
end