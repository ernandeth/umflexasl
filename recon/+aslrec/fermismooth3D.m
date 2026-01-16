function out = fermismooth3D(in, cutoff, width)
% function out = fermismooth3D(in, cutoff, width)
%
% use a Fermi filter in k-space to smoothout data
%
% inputs
%   in: input 3D data
%   cutoff:   the fraction of the k-space radius where cut-off will be
%       applied. eg-  if 0.5, then we only keep the bottom half og the freuqnecies
%       in k-space
%   width: the fraction of the transition band for the filter window
% 
%   output:  out - the filtered data
%

% create the fermi window in 3D
sz = size(in);
[x y z] = meshgrid(linspace(-1,1,sz(1)), linspace(-1,1,sz(2)), linspace(-1,1,sz(3)));
R = sqrt(x.^2 + y.^2 + z.^2);
F = 1 ./ (1 + exp( (R - cutoff) ./ width ));

% filter the k-space data:
tmp = fftshift(fftn(in));
tmp = tmp .*F;

% back to image space
out = (ifftn(tmp));



return