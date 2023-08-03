function varas = solver_GS_global(varas)

if (~isfield(varas, 'global_weights'))
    global_weights = zeros(varas.Hei, varas.Wid);
    for j = 1:varas.LED_num
        top = round(varas.LED_fy(j) - fix(varas.hei/2));
        left = round(varas.LED_fx(j) - fix(varas.wid/2));
        global_weights(top:top+varas.hei-1,left:left+varas.wid-1) = ...
            global_weights(top:top+varas.hei-1,left:left+varas.wid-1) + varas.pupil;
    end
    global_weights(global_weights==0)=1;
    varas.global_weights = 1./global_weights;
end
subspectrums_new = zeros(varas.hei, varas.wid, varas.LED_num);
for j = 1:varas.LED_num
    top = round(varas.LED_fy(j) - fix(varas.hei/2));
    left = round(varas.LED_fx(j) - fix(varas.wid/2));
    subspectrum = varas.FPM_spectrum(top:top+varas.hei-1,left:left+varas.wid-1);
    abbr_subspectrum = subspectrum.*varas.pupil;
    object_LR = varas.Ft(abbr_subspectrum);
    object_LR_new = varas.process_amplis(:,:,j).*(object_LR./abs(object_LR)); % update the LR field
    abbr_subspectrum_corrected = varas.F(object_LR_new);

    W = abs(varas.pupil)./max(max(abs(varas.pupil)));
    invP = conj(varas.pupil)./((abs(varas.pupil)).^2+eps.^2);
    subspectrum_new = (W.*abbr_subspectrum_corrected + (1-W).*(abbr_subspectrum_corrected)).*invP;
    subspectrum_new(varas.pupil==0) = 0; % update the subspectrum
    subspectrums_new(:,:,j) = subspectrum_new;
end
FPM_spectrum = zeros(varas.Hei, varas.Wid);
for j = 1:varas.LED_num
    top = round(varas.LED_fy(j) - fix(varas.hei/2));
    left = round(varas.LED_fx(j) - fix(varas.wid/2));
    FPM_spectrum(top:top+varas.hei-1,left:left+varas.wid-1) = ...
        FPM_spectrum(top:top+varas.hei-1,left:left+varas.wid-1) + subspectrums_new(:,:,j);
end
varas.FPM_spectrum = varas.global_weights.*FPM_spectrum;

end
