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

zernike_pupils = [];
for ii = 1:21
%     pupil_phase = ZernikeCalc(ii, 1, pupil);
%     zernike_pupil = single(pupil.*exp(1i*pupil_phase));
    zernike_pupil = ZernikeCalc(ii, 1, pupil,'STANDARD');
    zernike_pupils = cat(3, zernike_pupils, zernike_pupil);
    
    imwrite(mat2gray(zernike_pupil), ['zernike_coeff_',num2str(ii,'%02d'),'.png']);
end

save('zernike_pupils_256.mat', 'zernike_pupils', '-v6');
