function saveGMM(model, name)
% supports only TXT format

fid = fopen(name, 'wt');
if fid < 0
    error('Cannot open file')
end
Dim = length(model(1).mean);
NumMix = length(model);
fprintf(fid, '%i\n', Dim);
fprintf(fid, '% 15.10e\n', ones(Dim, 1));
fprintf(fid, '%i\n', NumMix);

if ~isfield(model, 'C')
    %diag case
    for k = 1:NumMix
        fprintf(fid, '% 15.10e\n', model(k).gain);
        fprintf(fid, '\n');
        fprintf(fid, '% 15.10e\n', model(k).mean);
        fprintf(fid, '\n');
        fprintf(fid, '% 15.10e\n', model(k).var);
        fprintf(fid, '\n');
    end

else %fullCov case
    for k = 1:NumMix
        fprintf(fid, '% 15.10e F\n', model(k).gain);
        fprintf(fid, '\n');
        fprintf(fid, '% 15.10e\n', model(k).mean);
        fprintf(fid, '\n');
        for d = 1:Dim
            fprintf(fid, '% 15.10e', model(k).C(d, d));
            for dd = 1:Dim
                fprintf(fid, ' % 15.10e', model(k).C(d, dd));
            end
            fprintf(fid, '\n');
        end
        fprintf(fid, '\n');
    end
end
    
    
fclose(fid);


