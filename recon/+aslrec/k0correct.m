function kdata = k0correct(kdata, klocs, options)
% function kdata = k0correct(kdata, klocs, options)
% 
% use redundant data in the center of k-space to remove fluctuations in the
% signals.
% 
% the data dimensions are assumed to be 4D: 
% Npoints x nechoes x Nframes x Ncoils
% options:
%   1 - scale each FID so that the mean of the navigator 
%       (magnitude and phase)is the same in each frame
%
%   2 - shift the phase of each FID so that the mean of the
%       navigator phase is constant in each frame

fprintf('\nExecuting k0 correction %d \n', options);

[Ndat, Nviews, Nframes, Ncoils] = size(kdata);

% compute the radial position of the trajectory (distance to center)
% and identify the redundant k0 points
% assume all the views have a set of navigator points in the same position
% so we just use the first view
klocs = squeeze(klocs(:,1,:));

R = sqrt(sum(klocs.^2, 2));
k0inds = find(R<1e-5);

if length(k0inds) < 1
    length(k0inds)
    fprintf('WARNING:less than 1 navigator points. Skipping the correction \n');
    return 
end


for c=1:Ncoils
    for f=1:Nframes

        % averaging the navigator data from all the views of each frame as the
        % reference for that frame and that coil
        tmp = kdata(k0inds,:,f,c);
        baseline = mean(tmp(:));

        if options==2
            baseline = 1*exp(i*angle(baseline));
        end

        for v=1:Nviews
        
            tmp = kdata(:,v,f,c);
            
            vm = mean(tmp(k0inds));
            
            if options==2
                vm = 1*exp(i*angle(vm));
            end

            tmp2 = tmp * baseline / vm  ;
%{
            [v f c]
            plot(abs(tmp)); hold on
            plot(abs(tmp2)); hold off
            baseline
            vm 
            pause
%}
            kdata(:,v,f,c) = tmp2;
            
        end
    end
end


return