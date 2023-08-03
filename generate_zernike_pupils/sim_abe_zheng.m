real_parameters_zheng;

hei = 128;
wid = 128;
deltaFy = 1/(hei*uniform_px);        % the value (in spatial frequency units) of each pixel in the fourier domain
deltaFx = 1/(wid*uniform_px);        % the value (in spatial frequency units) of each pixel in the fourier domain
deltaF = min([deltaFx, deltaFy]);
pupil_radius = round(NA_obj/wavelength/deltaF);             % f0/��fu in pixels


[x, y] = meshgrid((-fix(wid/2):ceil(wid/2)-1),(-fix(hei/2):ceil(hei/2)-1));
[~,R] = cart2pol(x,y);
pupil = ~(R>pupil_radius);

zernike_pupils = [];
for ii = 1:15
%     pupil_phase = ZernikeCalc(ii, 1, pupil);
%     zernike_pupil = single(pupil.*exp(1i*pupil_phase));
    zernike_pupil = ZernikeCalc(ii, 1, pupil,'STANDARD');
    zernike_pupils = cat(3, zernike_pupils, zernike_pupil);
    
    imwrite(mat2gray(zernike_pupil), ['zernike_coeff_',num2str(ii,'%02d'),'.png']);
end

save('zernike_pupils_zheng_128.mat', 'zernike_pupils', '-v6');
