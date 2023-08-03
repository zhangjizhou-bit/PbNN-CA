function varables = import_data(varables)

data_dir = varables.data_dir;
patch_cx = varables.patch_cx;
patch_cy = varables.patch_cy;
half_width = varables.half_width;

list = dir(data_dir);
num = size(list,1)-2;
imaged_intens = zeros(2*half_width, 2*half_width, num);
for i = 1:num
    temp = double(imread([data_dir, list(i+2).name]));
    temp = temp(patch_cy-half_width+1:patch_cy+half_width, patch_cx-half_width+1:patch_cx+half_width);
    imaged_intens(:,:,i) = temp/65535;
end

imaged_intens = imaged_intens - varables.thre;
imaged_intens(imaged_intens<0) = 0;

varables.imaged_intens = imaged_intens;

end
