function varables = aft_process(varables)

tol = varables.tol;
[high, low] = find_lim(varables.FPM_inten, tol);
out_inten = varables.FPM_inten;
out_inten(out_inten<low)=low;
out_inten(out_inten>high)=high;
out_inten = 0.9*(out_inten-low)/(high-low)+0.05;
varables.out_inten = out_inten;
varables.out_ampli = sqrt(varables.out_inten);

[high, low] = find_lim(varables.FPM_phase, tol);
out_phase = varables.FPM_phase;
out_phase(out_phase<low)=low;
out_phase(out_phase>high)=high;
out_phase = (out_phase-low)/(high-low);
varables.out_phase = out_phase;

end

function [high, low] = find_lim(img, tol)
pixel_num = size(img,1)*size(img,2);
[N,EDGES] = histcounts(img,1000);
N_sum = cumsum(N);
N_sum = N_sum/pixel_num;
for i = 1:1:length(N)
    if ( N_sum(i) > tol )
        low = EDGES(i);
        break;
    end
end
for i = length(N):-1:1
    if ( N_sum(i) < 1-tol )
        high = EDGES(i);
        break;
    end
end
end