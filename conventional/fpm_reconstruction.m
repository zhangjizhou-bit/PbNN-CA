function varables = fpm_reconstruction(varables)
%-------------------------------------------------------------------
% Use two image and a series of setup parameters to generate a simulate FPM dataset. 
% Caution that the unit in the space domain is (um) and the unit in the Fourier domain
% is (cycles/um) not (rad/um).
%
% Zhou @ 2016.09.17
%---------------structure of input-----------------------------------------
% intensity_dir:        the directory of the intensity image
% phase_dir:            the directory of the phase image
%---------------parameters of in and out-----------------------------------
cx = varables.cx;	cy = varables.cy;
patch_cx = varables.patch_cx;
patch_cy = varables.patch_cy;
loop_num = varables.loop_num;
imaged_intens = varables.imaged_intens;
[hei, wid, ~] = size(imaged_intens);
%---------------parameters of microscope setup----------------------------
detected_parameters;
%---------------parameters calculated based above----------------------------
Hei = upsample_ratio*hei;
Wid = upsample_ratio*wid;
deltaFy = 1/(hei*uniform_px);        % the value (in spatial frequency units) of each pixel in the fourier domain
deltaFx = 1/(wid*uniform_px);        % the value (in spatial frequency units) of each pixel in the fourier domain
deltaF = min([deltaFx, deltaFy]);
pupil_radius = round(NA_obj/wavelength/deltaF);             % f0/��fu in pixels
LED_x = LED_x - (patch_cx - cx)*uniform_px;
LED_y = LED_y - (patch_cy - cy)*uniform_px;
dd = sqrt(LED_x.^2 + LED_y.^2 + illumination_distance^2);	% distance from each LED to sample
sin_thetay = LED_y./dd;
sin_thetax = LED_x./dd;
kiy = -sin_thetay/wavelength;	% y component of wave vector
kix = -sin_thetax/wavelength;	% x component of wave vector
center_y = ceil((Hei+1)/2);
center_x = ceil((Wid+1)/2);
LED_fy = kiy/deltaF + center_y;	% y coordinate in the fourier domain
LED_fx = kix/deltaF + center_x;	% x coordinate in the fourier domain

[x, y] = meshgrid((-fix(wid/2):ceil(wid/2)-1),(-fix(hei/2):ceil(hei/2)-1));
[~,R] = cart2pol(x,y);
pupil = double(~(R>pupil_radius));
pupil_fun = 1;

varables.F = F;
varables.Ft = Ft;
varables.hei = hei;
varables.wid = wid;
varables.Hei = Hei;
varables.Wid = Wid;
varables.idx = idx;
varables.LED_fx = LED_fx;
varables.LED_fy = LED_fy;
varables.LED_num = LED_num;
varables.pupil = pupil;
varables.pupil_fun = pupil_fun;
varables.deltaF = deltaF;
varables.wavelength = wavelength;
%---------------preprocess----------------------------
% varables = preprocess_wavelet( varables );
% varables = preprocess( varables );
% varables.process_amplis = sqrt(varables.process_intens);
varables.process_amplis = sqrt(varables.imaged_intens);
%---------------initialization----------------------------
FPM_ampli0 = imresize(varables.process_amplis(:,:,(LED_num+1)/2), [Hei, Wid], 'bilinear');
scale = upsample_ratio.^2;
FPM_ampli0 = FPM_ampli0/scale;
FPM_phase0 = zeros(Hei, Wid, 'double');
FPM_image0 = FPM_ampli0.*exp(1i*FPM_phase0);
FPM_inten0 = FPM_ampli0.^2;
FPM_spectrum0 = F(FPM_image0);
FPM_spectrum = FPM_spectrum0;

varables.FPM_spectrum = FPM_spectrum;
varables.scale = scale;
%---------------reconstruction----------------------------
if ( ~isfield(varables, 'hide') )
    f1 = figure('numbertitle','off','units','normalized','outerposition',[0 0 1 1]);
end

for loop_count = 1:loop_num
    varables.loop_count = loop_count;
    if strcmp(varables.solver, 'GS')
        varables = solver_GS(varables);
    elseif strcmp(varables.solver, 'GS_cos4')
        varables = solver_GS_cos4(varables);
    elseif strcmp(varables.solver, 'GS_global')
        varables = solver_GS_global(varables);
    elseif strcmp(varables.solver, 'GS_global_Poisson')
        varables = solver_GS_global_Poisson(varables);
    elseif strcmp(varables.solver, 'EPRY')
        varables = solver_EPRY(varables);
    elseif strcmp(varables.solver, 'AS')
        varables = solver_AS(varables);
    elseif strcmp(varables.solver, 'AS_global')
        varables = solver_AS_global(varables);
    elseif strcmp(varables.solver, 'AS_global_Poisson')
        varables = solver_AS_global_Poisson(varables);
    end
    
    FPM_spectrum = varables.FPM_spectrum;
    FPM_object = Ft(FPM_spectrum);
    FPM_ampli = abs(FPM_object);
    FPM_inten = FPM_ampli.^2;
    FPM_phase = -angle(FPM_object);
    
    if ( ~isfield(varables, 'hide') )
        fprintf('Iteration: %03d\n',loop_count);            
        set(f1,'name',sprintf('Iteration: %03d\n',loop_count));
%         set(f1,'name',sprintf('Iteration: %03d\tAlpha:%1.4f\n',loop_count,varables.alpha));
        subplot(3,2,1);
        imagesc(log10(abs(FPM_spectrum))); colormap(gray); axis image;
        title('recovered Fourier magnitude');
        subplot(3,2,2);
        imshow(FPM_inten0,[]); axis image;
        title('original intensity');
        subplot(3,2,3);
        imshow(FPM_inten,[]); axis image;
        title('spatial intensity');
        subplot(3,2,4);
        imshow(FPM_phase,[]); axis image;
        title('spatial phase');
        subplot(3,2,5);
        imshow(abs(varables.pupil_fun),[]); axis image;
        title('spatial intensity');
        subplot(3,2,6);
        imshow(angle(varables.pupil_fun),[]); axis image;
        title('spatial phase');
        drawnow;
    end
    if ( isfield(varables, 'alpha') && varables.alpha == 0 )
        break;
    end
end

varables.FPM_inten = FPM_inten;
varables.FPM_ampli = FPM_ampli;
varables.FPM_phase = FPM_phase;
varables.FPM_spectrum = FPM_spectrum;

end



