function data = loadBinF(filename);
% load stored matrix as a float datatype
% matrix data(N,M) has to be stored in folowing format:
%    __int32 N
%    __int32 M
%    float data (whole data block - stored consecutive rows)

fid = fopen(filename, 'rb');
if fid < 0
    error(['Cannot open file ' filename]);
end

N = fread(fid, 1, 'int32');
M = fread(fid, 1, 'int32');
try
    tmp = fread(fid, Inf, 'float');
    data = reshape(tmp, length(tmp)/N, N)';
    fclose(fid);


catch
    fclose(fid);
    fid = fopen(filename, 'rb');
    if fid < 0
        error(['Cannot open file ' filename]);
    end

    N = fread(fid, 1, 'int32');
    M = fread(fid, 1, 'int32');
    data = zeros(N, M);
    for k = 1:N
        data(k,:) = fread(fid, M, 'float');
    end
    fclose(fid);
end
