function varas = solver_GS(varas)

for j = 1:varas.LED_num
    top = round(varas.LED_fy(varas.idx(j)) - fix(varas.hei/2));
    left = round(varas.LED_fx(varas.idx(j)) - fix(varas.wid/2));
    subspectrum = varas.FPM_spectrum(top:top+varas.hei-1,left:left+varas.wid-1);
    abbr_subspectrum = subspectrum.*varas.pupil;
    object_LR = varas.Ft(abbr_subspectrum);
    object_LR_new = varas.process_amplis(:,:,varas.idx(j)).*(object_LR./abs(object_LR)); % update the LR field
    abbr_subspectrum_corrected = varas.F(object_LR_new);

    W = abs(varas.pupil)./max(max(abs(varas.pupil)));
    invP = conj(varas.pupil)./((abs(varas.pupil)).^2+eps.^2);
    subspectrum_new = (W.*abbr_subspectrum_corrected + (1-W).*(abbr_subspectrum_corrected)).*invP;
    subspectrum_new(varas.pupil==0) = subspectrum(varas.pupil==0); % update the subspectrum

    varas.FPM_spectrum(top:top+varas.hei-1,left:left+varas.wid-1) = subspectrum_new;
    
%     figure(1);
%     imagesc(log10(abs(varas.FPM_spectrum))); colormap(gray); axis image;
%     drawnow;
end

end
