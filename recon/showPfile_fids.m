function showPfile_fids


[kdata,klocs,N,fov] = aslrec.read_data([],0);


etl1 = kdata(:,:,1,1);
etl1 = etl1(:);

ndat = size(kdata,1)
nechoes = size(kdata,2)
ncoils = size(kdata,3)
plot(abs(etl1))
return