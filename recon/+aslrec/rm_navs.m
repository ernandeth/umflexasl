function [kdata klocs] = rm_navs(kdata, klocs)
% 
% function [kdata klocs] = rm_navs(kdata, klocs)
%
% find and remove redundant data in the center of k-space 
fprintf('Removing Nav points\n');

[Ndat, Nviews, Nframes, Ncoils] = size(kdata);

% compute the radial position of the trajectory (distance to center)
% and identify the redundant k0 points
% assume all the views have a set of navigator points in the same position
% so we just use the first view
klocs_tmp = squeeze(klocs(:,1,:));

R = sqrt(sum(klocs_tmp.^2, 2));
k0inds = find(R<1e-5);
% keep the ends 
k0inds = k0inds(2:end-1);


if length(k0inds) < 10
    fprintf('WARNING:less than 10 navigator points. Skipping the correction \n');
    return 
end


klocs(k0inds,:,:) = [];
kdata(k0inds, :,:,:,:) = [];

return