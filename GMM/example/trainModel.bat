: example of a command line used to train a model from scratch (--type 0) with 4 Gaussians and 8 EM iterations,
: using SSE instructions and 4 threads. Data are stored in BIN format (--data-T 2), the model is saved to 
: a file myModel.gmm in txt format (option -T)
trainGMM.exe -v -i data.bin --data-T 2 --type 0 --nummix 4 --it-EM 8 -o myModel.gmm -T --SSE --numThrd 4

: model with full covariance-matrices
trainGMM.exe -v -i data.bin --data-T 2 --type 0 --nummix 4 --it-EM 8 -o myModelFC.gmm -T --SSE --numThrd 4 --full-cov

