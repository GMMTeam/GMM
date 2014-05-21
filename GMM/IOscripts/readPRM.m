function [data, CF] = readPRM(filename)
% [data, CF] = readPRM(filename)
% SV-ES-PARAM (*.prm) file-format
%
% data .. Nsamples x dim
% CF .. Nsamples x 1, certainty factor

fr = fopen(filename, 'rb');

ID = fread(fr, 11, 'uchar');
ID = char(ID');

if(length(ID) ~= 11)
    disp('Chybny format - jiny nez SV-ES-PARAM')
    data = 0;
    return
end

if (sum(ID == 'SV-ES-PARAM') ~= 11)
    disp('Chybny format - jiny nez SV-ES-PARAM')
    data = 0;
    return
end

dim = fread(fr, 1, 'int32');
nsamples = fread(fr, 1, 'int32');

CF = fread(fr, nsamples, 'float'); 

data = zeros(nsamples, dim);
for i=1:nsamples,
    vector = fread(fr, dim, 'float'); 
    data(i,:) = vector';
end
fclose(fr);
