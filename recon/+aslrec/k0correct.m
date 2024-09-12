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

[Ndat, Nviews, Nframes, Ncoils] = size(kdata);

% compute the radial position of the trajectory (distance to center)
% and identify the redundant k0 points
% assume all the views have the navigator points in the same position

klocs = squeeze(klocs(:,1,:));
R = sqrt(sum(klocs.^2, 2));
k0inds = find(R==0);

if length(k0inds) < 10
    fprintf('WARNING:less than 10 navigator points. Skipping the correction ');
    return(kdata)
end


for c=1:Ncoils
    for f=1:Nframes

        % use the first view of each frame as the reference
        tmp = kdata(:,1,f,c);
        baseline = mean(tmp(k0inds));
    
        if options==2
            baseline = 1*exp(i*angle(baseline));
        end

        for v=1:Nviews
        
            tmp = kdata(:,v,f,c);
            
            vm = mean(tmp(k0inds));
            
            if options==2
                vm = 1*exp(i*angle(vm));
            end

            tmp = tmp /(vm) * baseline;

            
            kdata(:,v,f,c) = tmp;
            
        end
    end
end
