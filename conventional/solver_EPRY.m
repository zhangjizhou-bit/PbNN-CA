function varas = solver_EPRY(varas)

wid = varas.wid;
hei = varas.hei;
LED_fy = varas.LED_fy;
LED_fx = varas.LED_fx;
idx = varas.idx;
pupil = varas.pupil;
pupil_fun = varas.pupil_fun;
FPM_spectrum = varas.FPM_spectrum;
scale = varas.scale;

for j = 1:varas.LED_num
    top = round(LED_fy(idx(j)) - fix(hei/2));
    left = round(LED_fx(idx(j)) - fix(wid/2));
    subspectrum = FPM_spectrum(top:top+hei-1,left:left+wid-1);
    abbr_subspectrum = subspectrum .* pupil .* pupil_fun;
    object_LR = varas.Ft(abbr_subspectrum);
    object_LR_new = varas.process_amplis(:,:,idx(j)).*(object_LR./abs(object_LR)); % update the LR field
    subspecturm_new = varas.F(object_LR_new);
    
    FPM_spectrum(top:top+hei-1,left:left+wid-1) = FPM_spectrum(top:top+hei-1,left:left+wid-1) + conj(pupil.*pupil_fun)./(max(max(abs(pupil.*pupil_fun).^2))).*(subspecturm_new - abbr_subspectrum);
    pupil_fun = pupil_fun + conj(FPM_spectrum(top:top+hei-1,left:left+wid-1))./(max(max(abs(FPM_spectrum(top:top+hei-1,left:left+wid-1)).^2))).*(subspecturm_new - abbr_subspectrum); 
        
%     figure(1);
%     imagesc(log10(abs(FPM_spectrum))); colormap(gray); axis image;
%     drawnow;
end

varas.pupil_fun = pupil_fun .* pupil;
varas.FPM_spectrum = FPM_spectrum;

end
