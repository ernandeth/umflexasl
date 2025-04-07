function kdata = hp_filter(kdata, klocs, order, amplitude)
% function kdata = hp_filter(kdata, klocs, order, amplitude)
% 
% Multiply the data by a filter kernel that is defined by the radial
% potision in k-space.  this is like the ramp filter in filtered
% backprojection.  We willl include options for filter order (not just
% ramp - determined by parameter 'order')
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
F = F.^order;
F = amplitude*F/max(F) + 1;

fprintf('\nExecuting radial filter in k-space- order %d \n', order);


for c=1:Ncoils
    for f=1:Nframes

        for v=1:Nviews
        
            tmp = kdata(:,v,f,c) .* F;
            
            kdata(:,v,f,c) = tmp;
            
        end
    end
end


return