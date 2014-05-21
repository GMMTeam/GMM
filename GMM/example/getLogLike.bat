: compute mean log-like of feature vectors stored in data.bin given GMM myModel.gmm (in TXT format: -L)
: mean log-like will be written on the standard output
getLogLike.exe -i data.bin --data-T 2 --in-GMM myModel.gmm -L --SSE --numThrd 4
getLogLike.exe -i data.bin --data-T 2 --in-GMM myModelFC.gmm -L --SSE --numThrd 4

: compute log-like of each feature vector stored in data.bin given GMM myModel.gmm (in TXT format: -L),
: the filename of file with log-likes will be the same as the input file, but will have different extension (deafult: .logL)
: log-likes will be stored in TXT format (option -T)
getLogLike.exe -i data.bin --data-T 2 --in-GMM myModel.gmm -L --for-each -o ./ -T --SSE --numThrd 4
