function gmm = loadGMM(filename)

gmm = loadGMMBin(filename);
if isempty(gmm),
    gmm = loadGMMTxt(filename);
end

if isempty(gmm),
    error(sprintf('Cannot load UBM from file: %s', filename))
end

function gmm = loadGMMTxt(name)

fid = fopen(name, 'r');
if fid < 0,    
    gmm = [];
    return
end
s = fgets(fid);
Dim = sscanf(s, '%d');
for k = 1:Dim+1
    s = fgets(fid);
end
NumMix = sscanf(s, '%d');
for k = 1:NumMix
    s = fgets(fid);
    gmm(k).gain = sscanf(s, '%e');
    if ~isempty(find(s == 'F'))
       full = 1;
    else
       full = 0;
    end
    s = fgets(fid);
    gmm(k).mean = zeros(1, Dim);
    for kk = 1:Dim
        s = fgets(fid);
        gmm(k).mean(kk) = sscanf(s, '%e');
    end
    s = fgets(fid);
    gmm(k).var = zeros(1, Dim);
    if full
        gmm(k).C = zeros(Dim, Dim);
    end
    for kk = 1:Dim
        s = fgets(fid);
        tmpvar = sscanf(s, '%e');
        gmm(k).var(kk) = tmpvar(1);
        if full
           gmm(k).C(kk, :) = tmpvar(2:end); 
        end
    end
    s = fgets(fid);
end

fclose(fid);


function gmm = loadGMMBin(filename)

fp = fopen(filename, 'rb');
if fp < 0,    
    gmm = [];
    return
end

desc = fread(fp, 9, 'uchar=>char')';

if desc ~= 'GMM-DB-VN',
    %disp('wrong format')
    gmm = [];
    return
end

dim = fread(fp, 1, 'int16');
dimvar = dim;
if dim < 0,
    dim = -dim;
    dimvar = dim * dim;
end

fread(fp, dim, 'single');
nummix = fread(fp, 1, 'uint16');
gains = fread(fp, nummix, 'single');

gmm.gain = [];
gmm.mean = [];
gmm.var = [];

for m = 1:nummix,    
    gmm(m).mean = [];
    gmm(m).var = [];

    gmm(m).gain = gains(m);
    gmm(m).mean = fread(fp, dim, 'single')';
    if dimvar ~= dim,
        gmm(m).C = reshape(fread(fp, dimvar, 'single'), dim, dim);
        gmm(m).var = diag(gmm(m).C)';
    else
        gmm(m).var = fread(fp, dimvar, 'single')';
    end
end
fclose(fp);
