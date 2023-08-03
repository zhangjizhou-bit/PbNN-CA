function varables = spectrum_prop(varables)
% Fresnel Transfer Function (TF) Propagator
% Computional Fourier Optics Chapter 5.1
deltaF = varables.deltaF;
FPM_spectrum = varables.FPM_spectrum;
half_width = varables.half_width;
wavelength = varables.wavelength;
[Hei, Wid] = size(FPM_spectrum);
ratio = 2*half_width/Hei;
Center_y = ceil((Hei+1)/2);
Center_x = ceil((Wid+1)/2);
kx = ((1:Wid)-Center_x)*deltaF*ratio;
ky = ((1:Hei)-Center_y)*deltaF*ratio;
[kx,ky] = meshgrid(kx,ky);
k = 2*pi/wavelength;
temp = kx.^2 + ky.^2;
% distance = -0.1:0.01:0.1;
distance = (-5:1:5)*20*wavelength;
% img_arr = [];
counter = 1;
delete('prop_images/*.jpg');
for z = distance
    phase_prop = exp(-1i*pi*wavelength*z*temp)*exp(1i*k*z);
    FPM_spectrum_prop = FPM_spectrum.*phase_prop;
    FPM_image_prop = ifft2(ifftshift(FPM_spectrum_prop));
    
    img = abs(FPM_image_prop);
%     img_arr(:,:,counter) = img;
    imwrite(mat2gray(img),['prop_images/prop_',num2str(z-min(distance),'%03.4f'),'.jpg']);
    counter = counter + 1;
end

end
