function varas = solver_AS(varas)

if ( varas.loop_count ==1 )% Calcute the step-size for the next iteration
    varas.alpha = 1;
    varas.err_bef = inf;
else
    varas = calc_stepsize(varas);
end
for j = 1:varas.LED_num
    top = round(varas.LED_fy(varas.idx(j)) - fix(varas.hei/2));
    left = round(varas.LED_fx(varas.idx(j)) - fix(varas.wid/2));
    subspectrum = varas.FPM_spectrum(top:top+varas.hei-1,left:left+varas.wid-1);
    abbr_subspectrum = subspectrum.*varas.pupil;
    object_LR = varas.Ft(abbr_subspectrum);
    object_LR_new = varas.process_amplis(:,:,varas.idx(j)).*(object_LR./abs(object_LR)); % update the LR field
    abbr_subspectrum_corrected = varas.F(object_LR_new);

    W = varas.alpha*abs(varas.pupil)./max(max(abs(varas.pupil)));
    invP = conj(varas.pupil)./((abs(varas.pupil)).^2+eps.^2);
    subspectrum_new = (W.*abbr_subspectrum_corrected + (1-W).*(abbr_subspectrum_corrected)).*invP;
    subspectrum_new(varas.pupil==0) = subspectrum(varas.pupil==0); % update the subspectrum

    varas.FPM_spectrum(top:top+varas.hei-1,left:left+varas.wid-1) = subspectrum_new;          
end

end


function varas = calc_stepsize(varas)

err_now = 0;
for j = 1:varas.LED_num
    top = round(varas.LED_fy(j) - fix(varas.hei/2));
    left = round(varas.LED_fx(j) - fix(varas.wid/2));
    subspectrum = varas.FPM_spectrum(top:top+varas.hei-1,left:left+varas.wid-1);
    abbr_subspectrum = subspectrum.*varas.pupil;
    object_LR = varas.Ft(abbr_subspectrum);
    err_now = err_now + sum(sum((abs(object_LR)-varas.process_amplis(:,:,j)).^2));
end
if((varas.err_bef-err_now)/varas.err_bef<0.001)
    varas.alpha = varas.alpha/2;% Reduce the stepsize when no sufficient progress is made
    if(varas.alpha<0.001)% Stop the iteration when Alpha is less than 0.001(convergenced)
        varas.alpha = 0;
    end
end
varas.err_bef = err_now;

end
