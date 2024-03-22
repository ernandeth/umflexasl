function [im,smap] = recon3dflex(varargin)
% function im = recon3dflex(varargin)
%
% Part of asl3dflex project by Luis Hernandez-Garcia and David Frey
% @ University of Michigan 2022
%
% Description: Function to reconstruct images from *3dflex ASL sequence
%   using Model-based SENSE recon with Pipe & Menon Density compensation
%
%
% Notes:
%   - this function does not write images, use writenii to write the
%       outputs to file
%   - default values in help message may not be up to date - check defaults
%       structure under the function header
%
% Path dependencies:
%   - matlab default path
%       - can be restored by typing 'restoredefaultpath'
%   - umasl
%       - github: fmrifrey/umasl
%       - umasl/matlab/ and subdirectories must be in current path
%   - mirt (matlab version)
%       - github: JeffFessler/mirt
%       - mirt setup must have successfully ran
%
% Variable input arguments (type 'help varargin' for usage info):
%   - 'pfile'
%       - search string for pfile
%       - string containing search path for desired pfile
%       - will only be used if either 'raw' or 'info' is left blank
%       - type 'help readpfile' for more information
%       - default is 'P*.7' (uses first pfile from directory to read in
%           raw data and info)
%   - 'niter'
%       - maximum number of iterations for model-based recon
%       - integer describing number of iterations
%       - if 0 is passed, conjugate phase recon will be used
%       - if a value less than 0 is passed, NUFFT recon will be used
%       - default is 0
%   - 'echoes'
%       - echoes in data to recon
%       - integer array containing specific frames to recon
%       - if 'all' is passed, all echoes will be used in recon
%       - default is 'all'
%   - 'shots'
%       - shots in data to recon
%       - integer array containing specific shots to recon
%       - if 'all' is passed, all shots will be used in recon
%       - default is 'all'
%   - 'frames'
%       - frames in timeseries to recon
%       - integer array containing specific frames to recon
%       - if 'all' is passed, all frames will be reconned
%       - default is 'all'
%   'coils'
%       - coils in data to recon
%       - integer array containing specific coils to recon
%       - if 'all' is passed, all coils will be used in recon
%       - default is 'all'
%   - 'resfac'
%       - resolution (dimension) upsampling factor
%       - float/double describing factor
%       - passing a value > 1 will result in higher output image dimension
%       - default is 1
%   - 'zoomfac'
%       - field of view zoom factor
%       - float/double describing factor
%       - passing a value > 1 will result in a smaller output image fov
%       - default is 1
%   - 'smap'
%       - coil sensitivity map for multi-coil datasets
%       - complex double/float array of dim x ncoils representing
%           sensitivity for each coil, or 'espirit' to estimate
%       - default is empty
%   - 'ccfac'
%       - coil compression factor
%       - float from 0 to 1 describing factor of reduction in # of coils
%       - default is 1
%   - 'tag'
%       - output tag
%       - string to append to image names
%       - default is empty
%
% Function output:
%   - im:
%       - output timeseries image (coil combined)
%       - complex array of image dimension
%

% Assign defaults
defaults = struct( ...
    'pfile', './P*.7', ...
    'ccfac', 0.25, ...
    'resfac', 1, ...
    'zoomfac', 1, ...
    'ndel', 0, ...
    'frames', 'all', ...
    'echoes', 'all', ...
    'shots', 'all', ...
    'llorder', 0, ...
    'niter', 0, ...
    'smap', [], ...
    'save', 1, ...
    'tag', '' ...
    );

% Parse through arguments
args = vararginparser(defaults, varargin{:});

% Read in pfile
pfile = dir(args.pfile);
pfile = [pfile(1).folder,'/',pfile(1).name];
[raw,phdr] = readpfile(pfile); % raw = [ndat x nframes*nshots+2 x nechoes x 1 x ncoils]

% Read in kviews file
kviews = dir('./kviews*.txt');
kviews = [kviews(1).folder,'/',kviews(1).name];
kviews = load(kviews);

% Read in ktraj file
ktraj = dir('./ktraj*.txt');
ktraj = [ktraj(1).folder,'/',ktraj(1).name];
ktraj = load(ktraj);

% Save important header info
ndat = phdr.rdb.frame_size;
nframes = phdr.rdb.user1;
nshots = phdr.rdb.user2; 
nechoes = phdr.rdb.user3;
ncoils = phdr.rdb.dab(2) - phdr.rdb.dab(1) + 1;
dim = [phdr.image.dim_X,phdr.image.dim_X];
if phdr.rdb.user4 > 0 % if image is 3d
    dim = [dim, phdr.image.dim_X];
end
fov = phdr.image.dfov/10 * ones(size(dim));
tr = phdr.image.tr*1e-3;
te = phdr.image.te*1e-3;

% Reshape raw to make things a little easier
raw = reshape(raw(:,1:nshots*nframes,:,:,:), ...
    ndat, nshots, nframes, nechoes, ncoils); % raw = [ndat x nshots x nframes x nechoes x ncoils]
raw = permute(raw, [1,4,2,3,5]); % raw = [ndat x nechoes x nshots x nframes x ncoils]

% Loop through views and get transformed kspace locations
klocs = zeros(ndat,3,nechoes,nshots,nframes); % klocs = [ndat x 3 x nechoes x nshots x nframes]
for framen = 1:nframes
    for shotn = 1:nshots
        for echon = 1:nechoes
            % Get transformation matrix for current view
            matidx = (framen-1)*nshots*nechoes + (shotn-1)*nechoes + echon;
            T = reshape(kviews(matidx,end-8:end)',3,3)';

            % Append kspace locations and data
            klocs(:,:,echon,shotn,framen) = ktraj*T';
        end
    end
end

% Set frames to recon
if strcmpi(args.frames,'all')
    args.frames = 1:nframes;
end
nframes = length(args.frames);

% Set shots to recon
if strcmpi(args.shots,'all')
    args.shots = 1:nshots;
end
nshots = length(args.shots);

% Set echoes to recon
if strcmpi(args.echoes,'all')
    args.echoes = 1:nechoes;
end
nechoes = length(args.echoes);

% Remove unwanted data
klocs = klocs(:,:,args.echoes,args.shots,args.frames);
raw = raw(:,args.echoes,args.shots,args.frames,:);

% Apply sampling delay
raw = interp1(1:ndat,raw,(1:ndat)+args.ndel,'PCHIP',0);

% Reorder data for look-locker sequences
if args.llorder
    klocs = permute(klocs, [1,2,4,3,5]); % klocs = [ndat x 3 x nshots x nechoes x nframes]
    klocs = reshape(klocs, [ndat, 3, nshots, 1, nechoes*nframes]); % klocs = [ndat x 3 x nshots x 1 x nechoes*nframes]
    raw = permute(raw, [1,3,2,4,5]); % raw = [ndat x nshots x nechoes x nframes x ncoils]
    raw = reshape(raw, [ndat, nshots, 1, nechoes*nframes, ncoils]); % raw = [ndat x nshots x 1 x nechoes*nframes x ncoils]
    nframes = nechoes*nframes;
    nechoes = nshots;
    nshots = 1;
end

% Vectorize the data and locations
klocs = reshape(permute(klocs,[1,3,4,2,5]),ndat*nechoes*nshots,3,nframes);
kdata = reshape(raw,ndat*nechoes*nshots,nframes,ncoils);

% Recon using CG/CP SENSE
[im,smap] = recon_cgsense(klocs(:,:,1), kdata(:,args.frames,:), ...
    round(args.resfac*dim), fov/args.zoomfac, ...
    'compfrac', args.ccfac, ...
    'smap', args.smap, ...
    'niter', args.niter);

% Write out image
if args.save
    writenii(sprintf('im_mag%s',args.tag),abs(im));
    writenii(sprintf('im_ang%s',args.tag),angle(im));

    % Write out sensitivity map
    writenii(sprintf('smap_mag%s',args.tag),abs(smap));
    writenii(sprintf('smap_ang%s',args.tag),angle(smap));
end

end