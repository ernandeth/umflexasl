function [out] = despike_data(raw,threshold)
% function [out] = despike_data(raw,threshold )
%
% replace data points in a time series that deviate more than a specified 
% number of std_devs from the mean, by the mean of their neighbors
%
ndat=size(raw,1);
nviews = size(raw,2);
nframes = size(raw,3);
ncoils = size(raw,4);

% rearrange the data :
% we want :  rows = time,   cols = kspace
raw = permute(raw, [1,2,4,3]);
raw =reshape(raw, ndat*nviews*ncoils, nframes);
out = zeros(size(raw));

% for ASL, we need to spilt the data:
r1 = raw(1:2:end,:);
r2 = raw(2:2:end,:);

% Now call the despiker for each half
out1 = actual_despiker(r1, threshold);
out2 = actual_despiker(r2, threshold);

% put it all back together
out(1:2:end,:) = out1;
out(2:2:end,:) = out2;

% rearrange to original shape
out =reshape(out, ndat,nviews,ncoils, nframes);
out = permute(out, [1,2,4,3]);

return

%%
function out = actual_despiker(in, spike_threshold)
out =in;
for col=1:size(in,2)
    % identify spikes at each k-space location:
    % first we detrend the data, then we identify the spikes
    kpoint = detrend(in(:,col));

    means = mean(kpoint);
    stds2 = std(kpoint);
    thresh = abs(means) + abs(spike_threshold* stds2);
    badrows = find(abs(kpoint) > thresh);

    % now replace the spikes by the mean of their neighbors
    for r=1:length(badrows)
        % replace spikes by a mean of its neighbors (in time) -
        % (linear interpolation)
        % make sure we're not overwriting the field map (the first row)!
        row = badrows(r);
        if row == 1
            out(row, col) = in(row+1, col);
        elseif row == size(in,1)
            out(row, col) = in(row-1, col);
        else
            out(row, col) = 0.5*(in(row-1, col) + in(row+1, col)) ;
        end
    end
end

return