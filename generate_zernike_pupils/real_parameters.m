%---------------------ALL DIMENSIONS ARE IN MICRONS------------------------
%---------------------parameters of the microscope------------------------
image_px = 6.5;                 % the actual pixel size of sensor
magnification = 4;              % the magnification of the objective
uniform_px = image_px/magnification;	% the uniform pixel size
wavelength = 0.505;            % wavelength of light used for simulated illumination
LED_spacing = 8128;             % distance between LEDs in the array
illumination_distance = 98000;	% distance from the LED matrix to the object
NA_obj = 0.13;                  % Numerical aperture of simulated imaging system
downsample_ratio = 4;           % downsample ratio of imaging
%---------------------parameters of the LED matrix------------------------
illumination_layers = 7;        % no of layers of a spiral square matrix of LEDs used to illuminate the setup
N = 2*illumination_layers - 1;	% side of square of illumination matrix
LED_x = double( ((1:N)-illumination_layers)*LED_spacing );    % the x position of LEDs
LED_y = double( ((1:N)-illumination_layers)*LED_spacing );    % the y position of LEDs
[LED_x, LED_y] = meshgrid(LED_x, LED_y);            % the coordinate matrix
LED_num = N*N;
LED_x = reshape(LED_x',[1,LED_num]);
LED_y = reshape(LED_y',[1,LED_num]);

F = @(x) fftshift(fft2(x));
Ft = @(x) ifft2(ifftshift(x));

