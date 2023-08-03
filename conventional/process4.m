clear;
clc;
solvers = {'GS', 'GS_cos4', 'AS', 'EPRY'};
load('../cos4_weights.mat');
for idx = 1:length(solvers)
    solver = solvers{idx};
    
    load('../experiment4_results/imaged_intens.mat');
    if solver == "GS_cos4"
        imaged_intens = imaged_intens ./intensity_weights;
    end
    varables.imaged_intens = permute(imaged_intens,[2 3 1]);
    varables.cx = 1280;
    varables.cy = 1080;
    varables.patch_cx = 1280;
    varables.patch_cy = 1180;
    varables.half_width = 32;
    varables.loop_num = 50;
    varables.solver = solver;
    varables.hide = 1;
    varables = fpm_reconstruction(varables);
    est_ampli = varables.FPM_ampli;
    est_phase = varables.FPM_phase;
    pupil_ampli_estimate = abs(varables.pupil_fun);
    pupil_phase_estimate = angle(varables.pupil_fun);
    save(['../experiment4_results/',varables.solver,'.mat'],...
    'est_ampli', 'est_phase', 'pupil_ampli_estimate', 'pupil_phase_estimate');
    clear args;
    clear varables;
        
end