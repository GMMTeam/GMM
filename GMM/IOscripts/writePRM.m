function writePRM(filename, data, CF)
% writePRM(filename, data[, CF])
% save in SV-ES-PARAM (*.prm) file-format
% data .. Nsamples x dim
% CF .. certainty factor, one for each vector (Nsamples x 1)

[nsamples, dim] = size(data);

if nargin < 3
    CF = ones(nsamples, 1);
end

fw = fopen(filename, 'wb');

fwrite(fw, 'SV-ES-PARAM');
fwrite(fw, dim, 'int32');
fwrite(fw, nsamples, 'int32');

fwrite(fw, CF, 'float'); 
fwrite(fw, data', 'float'); 

fclose(fw);
