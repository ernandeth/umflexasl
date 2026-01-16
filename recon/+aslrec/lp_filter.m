function kdata = lp_filter(kdata, klocs, cutoff)
% function kdata = lp_filter(kdata, klocs, cutoff)
% 
% Simple Low pass filter k-space data. (Fermi)
%
% zero-out the k-space data at radial positions (3D) further than
% Kmax*cutoff.  Uses a Fermi filter to reduce ripple.
% 
% the data dimensions are assumed to be 4D: 
% Npoints x nechoes x Nframes x Ncoils

[Ndat, Nviews, Nframes, Ncoils] = size(kdata);

% compute the radial position of the trajectory (distance to center)
% and identify the redundant k0 points
% assume all the views have a set of navigator points in the same position
% so we just use the first view
klocs = squeeze(klocs(:,1,:));

R = sqrt(sum(klocs.^2, 2));
% the weights in k-space as a function of radial position.
F = R/max(R);
F(F>cutoff) = 0;
F(F<=cutoff) = 1;

width=0.01;
F = 1 ./ (1 + exp( (R - cutoff) ./ width ));

fprintf('\nExecuting LP filter in k-space- cutoff %d \n', cutoff);


for c=1:Ncoils
    for f=1:Nframes

        for v=1:Nviews
        
            tmp = kdata(:,v,f,c) .* F;
            
            kdata(:,v,f,c) = tmp;
            
        end
    end
end


return