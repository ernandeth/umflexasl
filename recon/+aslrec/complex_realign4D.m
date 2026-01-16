function out = complex_realign4D(in)
% function out = complex_realign4D(in)
%
% use matlab built in functions to realign a series of 3D images using affine
% transformations, when the 3D images are complex.
% - useful for motion correction in segmented data

re = real(in);
im = imag(in);

ref = abs(mean(in,4));

[optimizer, metric] = imregconfig('monomodal');
optimizer.GradientMagnitudeTolerance = 1e-5;
optimizer.MaximumIterations = 500;


out = zeros(size(in));
rotmats = zeros(4,4,size(in,4));

parfor n=1:size(in, 4)

    fprintf('\nrealigning ... %d -th frame', n);
    rtmp = squeeze(re(:,:,:,n));
    itmp = squeeze(im(:,:,:,n));

    tmp = abs(squeeze(im(:,:,:,n)));

    % calculate the affine transformation matrix
    tform= imregtform(ref, tmp, 'rigid', optimizer, metric);
    
    % Create an output view with the same size/limits as the input
    % The output image will be clipped if the transformation causes content to go outside these limits
    OutView = affineOutputView(size(ref), tform, "BoundsStyle", "SameAsInput");


    rout = imwarp(rtmp, tform, "OutputView", OutView);
    iout = imwarp(itmp, tform, "OutputView", OutView);

    out(:,:,:,n) = rout + 1i*iout;
    rotmats(:,:,n) = tform.A;
end

save movements.mat rotmats

return