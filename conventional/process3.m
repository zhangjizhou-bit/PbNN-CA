clear;
clc;
solvers = {'AS', 'GS', 'GS_cos4'};
load('../cos4_weights.mat');
for idx = 1:length(solvers)
    solver = solvers{idx};
    for variaty_level = 0:0.05:0.5
        fprintf('variaty_level: %0.2f\n',variaty_level);    
        load(['../experiment3_results/imaged_intens_',num2str(variaty_level,'%0.2f'),'.mat']);
        if solver == "GS_cos4"
            imaged_intens = imaged_intens ./intensity_weights;
        end
        varables.imaged_intens = permute(imaged_intens,[2 3 1]);
        varables.cx = 1280;
        varables.cy = 1080;
        varables.patch_cx = 1280;
        varables.patch_cy = 1080;
        varables.half_width = 32;
        varables.loop_num = 100;
        varables.solver = solver;
        varables.hide = 1;
        varables = fpm_reconstruction(varables);
        est_ampli = varables.FPM_ampli;
        est_phase = varables.FPM_phase;
        pupil_ampli_estimate = abs(varables.pupil_fun);
        pupil_phase_estimate = angle(varables.pupil_fun);
        save(['../experiment3_results/',varables.solver,'_',num2str(variaty_level,'%0.2f'),'.mat'],...
        'est_ampli', 'est_phase', 'pupil_ampli_estimate', 'pupil_phase_estimate');
        clear args;
        clear varables;
    end
end