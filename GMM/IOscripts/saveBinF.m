function saveBinF(filename, data);
% Syntax: saveBinF(filename, data);
% save input matrix as a float datatype
% matrix data(N,M) will be stored in folowing format:
%    __int32 N
%    __int32 M
%    float data (whole data block - stored consecutive rows)


fid = fopen(filename, 'wb');
if fid < 0
    error(['Cannot open file ' filename]);
end

fwrite(fid, size(data), 'int32');
for k = 1:size(data, 1)
    fwrite(fid, data(k, :), 'float');
end
fclose(fid);
