GMM
===

SSE/GPU-accelerated training and evaluation of Gaussian Mixture Models (GMMs)

## TRAINING

* Executable: trainGMM
* For list of possible options run the executable without any parameters or with `--help`


All the options can be specified through the command line or in an .ini file, which
can be inserted using the option `-c train.ini`

In order to train a model, one has to specify file(s) containing feature vectors.  
These can be specified as separate files (option `-i`) or as an input directory (option `-I`),
or a file containing list of input files with full path, each on a separate line (option `-I`).
All of these options (`-i` or `-I`) can be specified multiple times.  

Several formats of input files are available (see option `--data-T X`), for details
on the formats see *lib_bmssr/param/OL_Param.cpp* or provided read/write MATLAB scripts.  
Possible formats are SVES (internal format) | HTK | RAW
   
    RAW file-format (BIN): [nsamples, dimension, data]
    * nsamples -> sizeof(int32), 
    * dimension -> sizeof(int32), 
    * data -> sizeof(float)*nsamples*dimension

Gaussian Mixture Model (GMM) can be trained from scratch (see option `--type X`). In this case none
initial model has to be specified, but one has to specify the number of desired Gaussians and number
of EM iterations (i.e. `--type 0 --nummix 32 --it-EM 8`).  
The estimation algorithm works as follow:

1. Gaussian with highest weight is split to two new Gaussians, 
   hence the overall number of Gaussians in the GMM increases by one.

2. Specified number of EM iterations is performed for the new model to better fit the data

3. If the number of Gaussians or number of splits reached the number of requested Gaussians,
   the algorithm ends, else go to step 1

Note: the number of Gaussians to be split can be adjusted, see options `--split-X` and `--split-th`;
or additional stopping condition can be set, see option `--ll-Th`.

Example of a command line used to train a model from scratch with 4 Gaussians and 8 EM iterations.
The model is saved to a file myModel.gmm in txt format (option `-T`)

    trainGMM -i myInputFile1.prm -i myInputFile2.prm -I myInputDir -I myInputListFile.txt --data-T 2 --type 0 --nummix 4 --it-EM 8 -o myModel.gmm -T

Also an initial GMM can be specified through option `--in-UBM initModel.gmm`,
and it can be used to train the final GMM in several ways:

1. extend the initial model with additional Gaussians, use option `--nummix X`, where *X* has to be greater than the
   actual number of Gaussians, e.g. initModel.gmm is a GMM with 4 Gaussians, then:

        trainGMM -i myInputFile.prm --data-T 2 --in-UBM initModel.gmm --type 0 --nummix 8 --it-EM 8 -o myModel.gmm -T

2. reiterate the actual model (EM) without adding any extra Gaussians:

2.1. The init model holds prior information on the distribution of feature vectors to be modeled,
     e.g. the initModel.gmm was trained from all the available feature vectors, however we would
     like to train a model only from a subset of the original set.
     In this case one of the adaptation techniques can be used (MAP | MLLR | fMLLR). A good choice
     is the Maximum A-posterior Probability (MAP) adaptation, where one has to specify the value
     of a relevance factor (option `--tau r`). The higher is the value the more data are needed to get
     a significant change of the model parameters. If _THETA_ are the parameters to be reestimated,
     _THETAprior_ are the parameters from the initGMM, _THETApost_ are the parameters computed according
     to the input data, and _ALPHA_ = _r_ \ (_Nsamples_ + _tau_), where _Nsamples_ is the amount of given
     feature vectors and tau is the relevance factor specified by the user, the update formula is:  
     _THETAnew_ = _ALPHA_ * _THETAprior_ + (1 - _ALPHA_) * _THETApost_;  
     Note that not all the parameters have to be reestimated (e.g. because of insufficient
     amount of data, when the estimate of the full-covariance matrix could get ill-conditioned).
     To reestimate only the means and weights use options `--weight` and `--mean`, e.g.:
    
        trainGMM -i myInputFile.prm --data-T 2 --in-UBM initModel.gmm --type 2 --it-MAP 3 --tau 14 --mean --weight -o myModel.gmm -T 

2.1. The init model is a kind of random initialization and does not hold any prior information.
     Therefore all the parameters should be reestimated without the use of any relevance factor
     (do not pull the new model toward the initial model). E.g.:
    
    trainGMM -i myInputFile.prm --data-T 2 --in-UBM initModel.gmm --type 2 --it-EM 32 --tau 0 --mean --weight --var -o myModel.gmm -T 

The estimation from scratch, working with subsequent division of Gaussians, can be quite time consuming,
therefore often a random initialization of GMM is used, e.g. pick M feature vectors from the training set,
use them as initial values of GMM means, compute the global variance of feature vectors in the training set
and use it as the initial value for variances, set all the weights of Gaussian uniformly (but they
have to sum up to one!). The use a few (e.g. 32) EM iterations for the GMM to converge, and repeat
the whole process several times. At the end, use the model, which gives highest log-like on the training set,
or use all the models with some voting scheme...  

If the estimation of full-covariance matrices fails (but not only in that case), try using the option `--robust`.
It can help also in cases with diagonal covariances. It is designed for better generalization purposes, for details see:

> J. Vanek, L. Machlica, J. Psutka, "Estimation of Single-Gaussian and Gaussian Mixture Models for Pattern Recognition", Progress in Pattern Recognition, Image Analysis, Computer Vision, and Applications Lecture Notes in Computer Science, 2013.

If you train a model from huge amount of feature vectors, that would not fir into the memory at once,
you can divide the feature vectors to separate files and use the option `--file-by-file` (see also the option `--prm-buffer SIZE`). The algorithm will iterate through all the files without storing them in the memory at once.
Check and turn off the option `--model-file-mode` if it is on.

If you train a model from scratch, you could appreciate the option `--tmp-N X`, which is used to save temporary models
after specified number of splits. If something goes wrong during the estimation process, you can use one of these
models as the initial model (option `--in-UBM`) and do not have to start from the beginning.


#### option `--in-classes`
* assume a set of *N* files with feature vectors: fileFV_1.prm, fileFV_2.prm, ..., fileFV_N.prm stored in dir/myDir/
* several models can be trained each from (not strictly) different files
* at first a file myModels.txt has to be created containing information on which model to train from which file
* one line of myModels.txt represents a list of files separated by space, which should be used to train a model,
which name is the first element on the line
* example of myModels.txt:

        model_1 fileFV_1 fileFV_2 fileFV_3
        model_2 fileFV_4 fileFV_5
        model_3 fileFV_6
        ...
        model_L fileFV_1 fileFV_4 fileFV_6

* the name of the input directory, where the files are stored can be specified by option `-I dir/myDir`
* extension of files can be specified by option `--inExt .prm`
* to train the models and store them in the directory myOutDir:

        trainGMM --in-classes myModels.txt -I dir/myDir -e .prm --data-T 2 --type 0 --nummix 4 --it-EM 8 -D myOutDir -T

* instead of specifying options `-I` and `-e`, whole paths can be specified directly in myModels.txt, e.g.:

        ...
        model_2 dir/myDir/fileFV_4.prm dir/myDir/fileFV_5.prm
        ...


## LOG-LIKELIHOOD

It is straight-forward to get the log-likelihood for a given set of feature vectors once a GMM was trained.
For this purpose, the getLogLike executable was designed.

For list of possible options run the executable without any parameters or with `--help`

If only a mean log-like of all the feature vectors stored in one file is desired, you don't have to specify any output
file, the mean log-likes along with the filename of the file with feature vectors is written directly on the standard
output. In order to get rid of any notifications and warnings set the verbose mode (option `-v`) to 0 and redirect the
standard error stream. If the log-likelihood for each feature vector is requested use the option `--for-each`, and
specify the output directory, in which for each input file a new file (with extension `--outExt .ext`) is stored with
logLikes for each feature vector

    BIN file-format: [foo, nsamples, logLikes]
    * foo -> sizeof(int32), will have always value 1
    * nsamples -> sizeof(int32), 
    * logLikes -> sizeof(float)*nsamples

Command line example (option `-L` is used to load the GMM in a text format, see options):

    getLogLike -i myInputFile1.prm -i myInputFile2.prm -I myInputDir -I myInputListFile.txt --data-T 2 --in-GMM myModel.gmm -L	

or compute log-like for each feature vector (option `--for-each`) and store in TXT format (option `-T`):

    getLogLike -i myInputFile1.prm -i myInputFile2.prm -I myInputDir -I myInputListFile.txt --data-T 2 --in-GMM myModel.gmm -L --for-each -o ./ -T


## GENERAL INFO

* options specified on the command line override settings from the ini file, but options not specified on the command
line are still read from the provided ini file

* if the GPU version does end with runtime errors, please try the SSE version with more threads (options: `--SSE
--numThrd n`)

___

If you use our code, please cite one of the following papers:

> L. Machlica, J. Vanek, Z. Zajic, "Fast Estimation of Gaussian Mixture Model Parameters on GPU using CUDA", The 12th International Conference on Parallel and Distributed Computing, Applications and Technologies, 2011.

> J. Vanek, L. Machlica, J. Psutka, "Estimation of Single-Gaussian and Gaussian Mixture Models for Pattern Recognition", Progress in Pattern Recognition, Image Analysis, Computer Vision, and Applications Lecture Notes in Computer Science, 2013.


