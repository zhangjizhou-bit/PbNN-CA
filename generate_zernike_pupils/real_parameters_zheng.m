%---------------------ALL DIMENSIONS ARE IN MICRONS------------------------
%---------------------parameters of the microscope------------------------
image_px = 1.845;                 % the actual pixel size of sensor
magnification = 4;              % the magnification of the objective
uniform_px = image_px/magnification;	% the uniform pixel size
wavelength = 0.532;            % wavelength of light used for simulated illumination
LED_spacing = 4000;             % distance between LEDs in the array
illumination_distance = 90880;	% distance from the LED matrix to the object
NA_obj = 0.1;                  % Numerical aperture of simulated imaging system


