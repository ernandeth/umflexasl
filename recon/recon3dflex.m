function x = recon3dflex(varargin)
% Function for performing regularized CG-SENSE NUFFT reconstruction for umvsasl data
%
% by David Frey and Luis Hernandez-Garcia
% 
% Usage:
% Specify a data directory using the 'pfile' argument, or simply run this
%   function from the directory.
% The data directory must include the following:
%   - raw data pfile: (P*.7)
%   - kviews file (kviews*.txt)
%   - ktraj file (ktraj*.txt)
%
% Required paths:
%   - MIRT (git@github.com:JeffFessler/mirt.git)
%
% Arguments:
%   - pfile: pfile name search string, leave empty to use first P*.7 file
%       in current working directory
%   - smap: sensitivity map (must be [image size x ncoils]), leave empty
%       to compress coils
%   - niter: number of iterations for CG reconstruction
%   - coilwise: recon individual coil images - necessary for creating sense
%                   maps for subsequent CGSENSE recon
%               1 - rearrange data for coil-wise recon of 1st frame,
%               2 - rearrange data for coilwise recon from ALL frames,
%               when each frame has different parts of k-space
%               (an in MRF type data)
%   - resfac: image space resolution upsampling factor
%   - ccfac: coil compression factor (i.e. ccfac = 4 will compressed data
%       to 1/4 of the channels)
%   - frames: frame indicies to reconstruct (default is all frames,
%       reconned sequentially)
%   - k0correct: scale each of the FIDs (views) according to the mean signal
%       in the navigator segment (redundant points collected in the center of k space)
%               0: default is to do nothing.
%               1: use the complex mean of the navigator segment
%               2: use only the phase of the segments.
%   - mrf_mode: 0: default - all frames have the same k-space locations
%               1: Fingerprinting mode (MRF)means that each temporal frame will 
%                   have a different set of rotation matrices.
%               2: combine rotation matrices from all even and odd frames,
%                   the output is only two frames.
%   - selectviews: which echoes to use for recon and discard the rest.  
%       This is a vector specifying the views that will be used.
%       eg - if  Nshots=4 aand ETL=12, 
%       but you only want the first 10 echoes, 
%       choose this vector: [1:10, 13:22, 25:34 , 37:46] 
%   - aqdel (N) : adjust acquisition delay by shifting the readout N points 
%       in time (it will result in losing aqdel data points per view)
%   - despike (N-std) : replace kspace data in time series by average of
%       neighboring frames if they deviate more than a specified threshold.
%       Does the odds and the eves time frames separately (for ASL)
%   - lp_filter (cutoff) ; use a low pass filter .  zeros-out the data
%       at locations greater than k_cutoff. 'cutoff' defines fraction of
%       Kmax.
%   - hp_filter (amplitude) ; use a high pass filter .  filter is a ramp 
%       function of radial distance .  amplitude defines the response:
%               out(r) = in(r) + amplitude*in(r)*ramp(r)
%   - nonavs (0/1) : remove navigator data from k=0 ?
%   - Reg (0) : Thikonov regularization weight. Default is 0 

    % check that mirt is set up
    aslrec.check4mirt();

    % set defaults
    defaults.pfile = [];
    defaults.smap = [];
    defaults.niter = 0;
    defaults.coilwise = 0;
    defaults.resfac = 1;
    defaults.ccfac = 1;
    defaults.frames = [];
    defaults.k0correct = 0;
    defaults.mrf_mode = 0;
    defaults.selectviews = [];
    defaults.aqdel = 0;
    defaults.despike = [];
    defaults.nonavs = 0;
    defaults.hp_filter = 0;
    defaults.lp_filter = 0;
    defaults.Reg = 0;

    % parse input parameters
    args = vararg_pair(defaults,varargin);

    % get data from pfile
    if (args.mrf_mode > 0) || (args.coilwise == 2)
        read_mrf_data_flag = 1;
    else
        read_mrf_data_flag = 0;
    end
    [kdata,klocs,N,fov] = aslrec.read_data(args.pfile , read_mrf_data_flag);

    % go to single precision to save memory!
    kdata = single(kdata);
    klocs = single(klocs);

    N = ceil(N*args.resfac); % upsample N (image matrix size)
    
    % cut off first 50 pts of acquisition (sometimes gets corrupted)
    kdata(1:50,:,:,:) = [];
    klocs(1:50,:,:) = [];
    
    % get sizes
    ndat = size(kdata,1);   % number of data per view.
    nviews = size(kdata,2);  % number of view per frame
    nframes = size(kdata,3); % number of frames
    ncoils = size(kdata,4); % number of coils
    framesize = ndat*nviews;  % number of data per frame.

    % coil mapping: rearrange for coil-wise reconstruction of frame 1 (for making SENSE maps)
    % if (args.coilwise==1) 
    %     kdata = permute(kdata(:,:,1,:),[1,2,4,3]);
    % end

    if isempty(args.frames)
        args.frames = 1:nframes; % default - use all frames
    end

    % choosing only the selected view for reconstruction and discarding the
    % rest of them.
    if ~isempty(args.selectviews)

        sv = args.selectviews(:)';

        if (args.mrf_mode) || (args.coilwise==2)
            tmp = repmat(sv, nframes,1);
            tmp = tmp + nviews*[0:nframes-1]';
            tmp = tmp';
            klocs = klocs(:, tmp(:), :);

        else
            klocs = klocs(:, sv, :);
        end
        kdata = kdata(:, sv, :, :);
        nviews = length(sv);

    end


    % scale echoes using navigator data
    if args.k0correct>0 && args.coilwise==0
        kdata = aslrec.k0correct(kdata, klocs, args.k0correct);
    end
    
    % remove navigator data before reconstruction
    if args.nonavs>0
        [kdata klocs] = aslrec.rm_navs(kdata, klocs);
        ndat = size(kdata,1);   % number of data per view.
        
    end

    % run a high pass filter (FBP) with speficified order
    if args.hp_filter>0
        kdata  = aslrec.hp_filter(kdata, klocs, 1, args.hp_filter);
    end

    % run a Low pass Fermi filter (FBP) with speficified cut-off fraction
    if args.lp_filter>0
        kdata  = aslrec.lp_filter(kdata, klocs, args.lp_filter);
    end

    % Remove spikes from data
    if ~isempty(args.despike)
        kdata = aslrec.despike_data(kdata,3);
    end

    % fix acquisition delays (time shits)
    if args.aqdel ~= 0
        [kdata klocs] = aslrec.aqdel(kdata, klocs, args.aqdel);
    end

    % do coil compression
    if ncoils >1

        if (args.ccfac==1) && isempty(args.smap) && (args.coilwise==0)

            ncoils = 1;  % why do this???
            
            fprintf('compressing data to %d coils...\n', ncoils);
            [kdata, sing, Vr] = ir_mri_coil_compress(kdata,'ncoil',ncoils);

            % try doing sum of squares of the coils... BAD idea!
            % fprintf('sum of squares to %d coils...\n', ncoils);
            % kdata = sqrt(sum(kdata `.^2,4));
            
        elseif (args.ccfac > 1)
            ncoils = ceil(ncoils/args.ccfac);
            fprintf('compressing data to %d coils...\n', ncoils);
            % kdata = ir_mri_coil_compress(kdata,'ncoil',ncoils);
            % args.smap = ir_mri_coil_compress(args.smap,'ncoil',ncoils);
            
            % compress the data
            [kdata, sing, Vr] = ir_mri_coil_compress(kdata,'ncoil',ncoils);

            if ~isempty(args.smap) && size(args.smap,4)>ncoils
                fprintf('compressing sense maps down to %d coils to match data compression...', ncoils);
                % compress the sense maps
                kc = fftc(args.smap, 1:3);  % make smaps in kspace

                idim = size(kc);  % input dimensions -before compression
                n_in = idim(end);
                kc = reshape(kc, [], n_in); % [*N n_in]
                cckc = kc * Vr;             % compres smaps in  k-smaps

                odim = idim;
                odim(end) = ncoils;  % dimensions after compression
                cckc = reshape(cckc,odim);

                args.smap = ifftc(cckc, 1:3);
            end
    
        elseif (size(args.smap,4) < ncoils) && (args.coilwise==0)
            ncoils = size(args.smap,4);
            warning('compressing data down to %d coils to match SENSE map...', ncoils);
            fprintf('compressing data down to %d coils to match SENSE map...', ncoils);
            kdata = ir_mri_coil_compress(kdata,'ncoil',ncoils);
        end


    end

    switch(args.coilwise)
        case 1
            % Prepare data for making coil images from the first frame (after compressing the data, if needed)
            % rearrange for coil-wise reconstruction of frame 1 (for making SENSE maps)
            % pretend the coil images are the frames and it's only one coil
            kdata = permute(kdata(:,:,1,:),[1,2,4,3]);
            args.frames = 1:size(kdata,3);
            ncoils =1;
        case 2
            % Prepare data for making coil images by concatenating all the frames
            % into one
            kdata = reshape(kdata, ndat, nviews*nframes, ncoils);

            % update sizes
            ndat = size(kdata,1);   % number of data per view.
            nviews = size(kdata,2);  % number of view per frame - now it's nviews*nframes!
            nframes = size(kdata,3); % number of frames is now the number of coils
            ncoils = size(kdata,4); % number of coils should be 1
            framesize = ndat*nviews;  % number of data per frame.
            args.frames = 1:nframes;

            % scale echoes using navigator data
            if args.k0correct>0
                kdata = aslrec.k0correct(kdata, klocs, args.k0correct);
            end
    
        otherwise
            % do nothing
    end

    % mrf_mode 2, means that the frames even and odd frames will be
    % aggregates into two single frames
    if args.mrf_mode==2

        % rearrange the kdata and their locations
        odds = zeros(ndat,nviews*nframes/2, 1, ncoils);
        evens = zeros(ndat,nviews*nframes/2, 1, ncoils);
        
        oddlocs = zeros(ndat,nviews*nframes/2, 3);
        evenlocs = zeros(ndat,nviews*nframes/2, 3);
        
        for f=1:nframes/2
            vx = [1:nviews] + (f-1)*nviews; 

            % split the data into odd and even frames
            odds(:,vx,1,:) = kdata(:,:,2*f-1,:);
            evens(:,vx,1,:) = kdata(:,:,2*f,:);
        
            % split the kspace locations into odd and even frames too
            ovx = [1:nviews] + (2*f-2)*nviews; 
            evx = [1:nviews] + (2*f-1)*nviews;
            
            oddlocs(:,vx,:) = klocs(:,ovx,:);
            evenlocs(:,vx,:) = klocs(:,evx,:);
        end
        kdata = cat(3, odds,evens);
        klocs = cat(2, oddlocs,evenlocs);

        % re-calculate sizes
        ndat = size(kdata,1);   % number of data per view.
        nviews = size(kdata,2);  % number of view per frame
        nframes = size(kdata,3); % number of frames
        ncoils = size(kdata,4); % number of coils
        framesize = ndat*nviews;  % number of data per frame.
        args.frames = 1:nframes;
    end



    % set nufft arguments
    nufft_args = {N, 6*ones(1,3), 2*N, N/2, 'table', 2^10, 'minmax:kb'};

    % initialize x
    x = zeros([N(:)',length(args.frames)]);
    
    fprintf('\nData ready for recon: %d k-space locations with %d coils for %d image voxels\n', ndat*nviews, ncoils, N(1)*N(2)*N(3));

    if (args.mrf_mode==0)
        % calculate a new system operator  just once
        [A,w, omega_msk] = make_system_matrix(klocs(:,1:nviews,:), N, fov, nufft_args);
        if ncoils > 1  && args.coilwise==0  % sensitivity encoding
            fprintf('... with sense maps\n');
            A = Asense(A,args.smap);
        end

        Aold = [];
        Areg = [];
        Aregtv = [];
        L = 0;
        if args.Reg
            Aold = A;
            L = args.Reg;
            % define A'A + L*eye operator for Tikhonov Regularization
            Areg = @(x) A'*A*x + L*x;

            % define A'A + D*eye operator for TV regularization           
            Aregtv = @(x) A'*A*x + L*[diff(x) ; 0] ;
            
        end

        
        % loop through frames and recon
        
        %for i = 1:length(args.frames)
        
        % if we are using the parallel pool , uncomment this.
        % set up the parallel pool
        if isempty(gcp('nocreate')), parpool(4), end
        parfor i = 1:length(args.frames)

            framen = args.frames(i);

            % get data for current frame
            % (LHG: dimensions will be (Ndat*Nshots*Nechoes x Ncoils)
            b = reshape(kdata(:,:,framen,:),[],ncoils);
            b = b(omega_msk,:);

            % initialize with density compensated adjoint solution
            fprintf("frame %d/%d: initializing solution x0 = A'*(w.*b)\n", i, length(args.frames))
            x0 = reshape( A' * (w.*b), N );
            x0 = ir_wls_init_scale(A, b, x0);
            
            if args.niter==0  % just the NUFFT - no iterations!
                x(:,:,:,i) = reshape(x0,N);
            else
                if args.Reg
                    % solve the problem with CG using regularization - this
                    % case will call Matlab's pcg instead of the cg_solve
                    % below.
                    b = Aold' * b;
                    [tmp, flag, rRes] = pcg(Areg, b(:), 1e-4, args.niter,[],[], x0(:));
                    fprintf("Tikhonov Regularized CG ended with residual fraction: %0.3g \n", rRes);
                    x(:,:,:,i) = reshape(tmp, N);

                else
                    % solve with CG - no regularization
                    x(:,:,:,i) = cg_solve(x0, A, b, args.niter, ...
                        sprintf("frame %d/%d: ", i, length(args.frames))); % prefix the output message with frame number
                end
            end
        end

    else  % MRF case
        %for i = 1:length(args.frames)
               
        if isempty(gcp('nocreate')), parpool(4), end
        parfor i = 1:length(args.frames)
            % Usually just need to do this once, but in MRF mode the system matrix
            % changes from frame to frame.
            views = [1:nviews] + nviews*(i-1);
            [A,w, omega_msk] = make_system_matrix(klocs(:,views, :), N, fov, nufft_args);

            % test without DCF
            %w=1;

            if ncoils > 1   && args.coilwise==0 % sensitivity encoding
                A = Asense(A,args.smap);
            end
            framen = args.frames(i);

            % get data for current frame
            % (LHG: dimensions will be (Ndat*Nshots*Nechoes x Ncoils)
            b = reshape(kdata(:,:,framen,:),[],ncoils);
            b = b(omega_msk,:);

            % initialize with density compensated adjoint solution
            fprintf("frame %d/%d: initializing solution x0 = A'*(w.*b)\n", i, length(args.frames))
            x0 = reshape( A' * (w.*b), N );
            x0 = ir_wls_init_scale(A, b, x0);

            if args.niter==0   % just the NUFFT - no iterations!
                x(:,:,:,i) = reshape(x0,N);
            else
                Aold = [];
                Areg = [];
                Aregtv = [];
                L = 0;
                if args.Reg
                    Aold = A;
                    L = args.Reg;
                    % define A'A + L*eye operator for Tikhonov Regularization
                    Areg = @(x) A'*A*x + L*x;

                    % define A'A + D*eye operator for TV regularization
                    Aregtv = @(x) A'*A*x + L*[diff(x) ; 0] ;

                    % solve the problem with CG using regularization - this
                    % case will call Matlab's pcg instead of the cg_solve
                    % below.
                    b = Aold' * b;
                    [tmp, flag, rRes] = pcg(Areg, b(:), 1e-5, args.niter,[],[], x0(:));
                    fprintf('Tikhonov regularized CG ended with residual fraction: %0.3g \n', rRes);
                    x(:,:,:,i) = reshape(tmp, N);
                else
                    % solve with CG
                    x(:,:,:,i) = cg_solve(x0, A, w.*b, args.niter, ...
                        sprintf('frame %d/%d: ', i, length(args.frames))); % prefix the output message with frame number
                end
            end
        end
    end

    save reconlog.mat args

end

function [A,w, omega_msk] = make_system_matrix(klocs, N, fov, nufft_args)
% function [A,w, omega_msk] = make_system_matrix(klocs, N, fov, nufft_args)
% create a system fatrix and the density compensation for the recon

    fprintf('...Making system matrix...')
    % (LHG : this scales the k-space trajectory,
    % and masks out locations what may go outside the -pi to pi range)    
    omega = 2*pi*fov(:)'./N(:)'.*reshape(klocs,[],3);
    omega_msk = vecnorm(omega,2,2) < pi;
    omega = omega(omega_msk,:);
    
    % (LHG : this creates the system matrix A as a NUFFT )
    A = Gnufft(true(N),[omega,nufft_args]); % NUFFT
    
    % calculate density compensation
    w = aslrec.pipedcf(A,10); 
    %w = 1;   % pipe-menon is not very useful in undersampled data ...?

    % use single precision to make things faster and use less memory
    A.arg.st.sn = single(A.arg.st.sn);
    A.arg.st.om = single(A.arg.st.om);
    w = single(w);
end

function x_star = cg_solve(x0, A, b, niter, msg_pfx)
% alternative:  pcg.m function in matlab

    % set default message prefix
    if nargin < 5
        msg_pfx = '';
    end

    % loop through iterations of conjugate gradient descent
    x_set = zeros([size(x0),niter+1]);
    x_star = x0;
    x_set(:,:,:,1) = x_star;

    % gradient calculation
    r = reshape(A'*(b - A*x_star), size(x_star));
    
    p = r;
    rsold = r(:)' * r(:);
    for n = 1:niter
        fprintf('%sCG iteration %d/%d, res: %.3g   \n', msg_pfx, n, niter, rsold);
         
        % calculate the new gradient descent step size
        AtAp = reshape(A'*(A*p), size(x_star));        
        alpha = rsold / (p(:)' * AtAp(:));

        % update the guess
        x_star = x_star + alpha * p ; % - 2*alpha*x_star;  % L2 regularizer here
        x_set(:,:,:,n+1) = x_star;

        % calculate new gradient update
        r = r - alpha * AtAp;       
        rsnew = r(:)' * r(:);
        p = r + (rsnew / rsold) * p;
        rsold = rsnew;

        if exist('exitcg','var')
            break % set a variable called "exitcg" to exit at current iteration when debugging
        end

    end
    
end
