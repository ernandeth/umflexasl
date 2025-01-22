function [kdata, klocs] = aqdel(kdata, klocs, ndel)
% function [kdata, klocs] = aqdel(kdata, klocs, ndel)
% 
% shift the kspace trajectory in time to account for gradient deays
% may clip the data if needed
%
% the data dimensions are assumed to be 4D: 
% Npoints x nechoes x Nframes x Ncoils

[Ndat, Nviews, Nframes, Ncoils] = size(kdata);

% compute the radial position of the trajectory (distance to center)
% and identify the redundant k0 points
% assume all the views have the navigator points in the same position


fprintf('\nshifting trajectory by %d \n', ndel);

if ndel>0
    klocs = klocs(ndel+1:end, :,:);
    kdata = kdata(1:end-ndel,:,:,:);
end
if ndel<0
    klocs = klocs(1:end+ndel, :,:);
    kdata = kdata(-ndel+1:end,:,:,:);
end


return