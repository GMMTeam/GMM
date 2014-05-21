// 
// Copyright (c) 2007--2014 Lukas Machlica
// Copyright (c) 2007--2014 Jan Vanek
// 
// University of West Bohemia, Department of Cybernetics, 
// Plzen, Czech Repulic
// 
// All rights reserved.
// 
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
// 1. Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
// 3. The name of the author may not be used to endorse or promote products
//    derived from this software without specific prior written permission.
// 
// THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS OR
// IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
// OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
// IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT, INDIRECT,
// INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT
// NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
// THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE. 

#ifndef _OL_ADAPTMODEL_
#define _OL_ADAPTMODEL_

#include "matrices/Matrix4_LMa.h"
#include "general/GlobalDefine.h"
#include "model/OL_GMModel.h"

#include "tools/OL_FileList.h"
#include "param/OL_Param.h"
#include "model/OL_GMMStats.h"
#include "trainer/OL_GMMStatsEstimator.h"

#include <vector>
#include <list>
#include <hash_map>
#include <string>
#include <boost/program_options.hpp>


#define STATS_TYPE double
//#define STATS_TYPE float
#define _MIN_GLOBAL_OCC_ 100
#define _MIN_IND_SAMPLES_ 2 // min number of independent samples for a mixture to be valid
#define FROM_SCRATCH    0
#define MLLR    1
#define MAP     2
#define fMLLR   3
#define MLLR_MAP 4



class CModelAdapt {
public:

	struct OPTIONS {
		int _adaptType;
		bool _fromScratch;
		float _splitXpercAtOnce;
		float _splitTh;
		bool _applyTransforms;
		bool _saveTransforms;
		bool _mllrUBMInit;
		string _transformFileExt;	
		int _tau;
		bool _AdaptMeanBool;
		bool _AdaptVarBool;
		bool _AdaptWeightBool;
		bool _robustCov;
		float _llTh;
		float _delMix;
		float _delMixPercentageTh;	
		int _iterMLLR;
		int _iterCountEM;
		int _iterCountMAP;
		bool _getQvalMLLR;
		bool _saveMixOcc;
		bool _SSEacc;
		bool _CUDAacc;
		int _cudaGPUid;		
		unsigned int _cudaNaccBlocks;
		unsigned int _numThreads;
		float _memoryBufferSizeGB;
		float _memoryBufferSizeGB_GPU;
		int _load_type;
		unsigned int _prmSamplePeriod;
		unsigned int _dwnsmp;	
		float _min_var; // MINIMAL ALLOWED VARIANCE
		int _saveModelAfterNSplits;
		int _verbosity;  

		OPTIONS() :	
			_adaptType(4),
			_fromScratch(false),
			_splitXpercAtOnce(0.0f),
			_splitTh(0.0f),
			_applyTransforms(true),
			_saveTransforms(false),
			_mllrUBMInit(true),
			_transformFileExt(""),
			_tau(14),
			_AdaptMeanBool(),
			_AdaptVarBool(true),
			_AdaptWeightBool(false),
			_robustCov(false),
			_llTh(0.0f),
			_delMix(0.0),
			_delMixPercentageTh(),
			_iterMLLR(1),
			_iterCountEM(0),
			_iterCountMAP(1),
			_getQvalMLLR(false),
			_saveMixOcc(false),
			_SSEacc(false),
			_CUDAacc(false),
			_cudaGPUid(-1),
			_cudaNaccBlocks(8),
			_numThreads(1),
			_memoryBufferSizeGB(0.1f),
			_memoryBufferSizeGB_GPU(0.0f),
			_load_type(0),
			_saveModelAfterNSplits(-1),
			_prmSamplePeriod(1),
			_dwnsmp(1),
			_min_var(1e-6f),
			_verbosity(0) 
		{}
	};

	CModelAdapt();
	~CModelAdapt();	

	// alocates memory and stores UBM model;	
	void Initialize(GMModel *UBM);	

	// returns a copy of actual model, if model does not exist NULL is returned instead
	GMModel *GetClientModelCopy();

	// returns the GMM model (NULL if it does not exist), and the pointer to the GMM inside of this class is set to NULL
	GMModel *PickUpClientModel();

	// returns a copy of actual transformation matrix, if it does not exist NULL is returned instead
	MATRIX *GetTRNMatrixCopy();

	// returns the transformation model (NULL if it does not exist), and the pointer to the matrix inside of this class is set to NULL
	MATRIX *PickUpTRNMatrix();

	// list of options from CModelAdapt::OPTIONS is returned
	static boost::program_options::options_description *GetOptions(CModelAdapt::OPTIONS &opt);
	
	void SetOptions(CModelAdapt::OPTIONS &opt);

	// insert data, if linkPointer == true then only the pointer will be stored, otherwise the data will be copied
	void InsertData(Param& data, bool linkPointer = false);

	// insert list of files, statistics will be accumulated file-by-file 
	void InsertData(CFileList *list); 

	// insert list of files along with specification, which frames should be used for accumulation
	void InsertData(stdext::hash_map < std::string, std::list < std::vector <unsigned int> > >* framelist);

	// create model from existing one - initial UBM inserted through Initialize(GMModel *UBM)
	void AdaptModel(GMModel *UBM);

	// train a model from scratch - in each iteration, Gaussian with highest weight is split to two Gaussians
	void TrainFromScratch(unsigned int nummix, bool fullcov,
		                  GMModel* initGMM = NULL, 
						  const char *FileNameBCKUPmodel = NULL, bool saveTxtBCKUPmodel = true);

	int Save(const char *FileName, bool save_txt, bool save_in_stat_format); 


private:
	GMModel *_modelSP; // adapted speaker model
	MATRIX *_W;       // mllr transformation matrix
	vector<double> *_W_Qvals; // values of the mllr criterion for each row of W
	GMModel *_UBM;	  // UBM -> only pointer is stored
	int _nummix;	  // number of Gaussians in the UBM

	bool _configLoaded, _initialized;
	bool _mixturesDeleted, _modelTrained, _trnmatComputed;
	bool _dataInserted, _AdaptWeightOnly;
	

	GMMStats<STATS_TYPE> _stats;
	GMMStatsEstimator<STATS_TYPE> *_estimator;
	
	CFileList *_prmlist;
	stdext::hash_map <std::string, std::list < std::vector <unsigned int> > > *_framelist;
	
	Param _param;
	int _dim;		  // feature vectors dimension
	
protected:	
	int _iter_fMLLR;  // number of iterations when fMLLR matrix is computed
	// parameters loaded from configuration file
	float _memoryBufferSizeGB; // max size of memory buffer for feature extraction for file-by-file accumulation
	float _memoryBufferSizeGB_GPU; // max size of memory buffer on GPU
	int _verbosity;
	int _adaptType;		  // MLLR / MAP / fMLLR / MLLR+MAP
	bool _fromScratch;	  // train a model from scratch
	float _splitXpercAtOnce; // split multiple Gaussians at once; _splitXpercAtOnce*100 percents of Gaussians with weight > _splitTh*max_weight
	float _splitTh;
	bool _applyTransforms; // should be the model transformed after MLLR matrix have been computed?
	bool _saveTransforms;  // save MLLR matrices?
	bool _mllrUBMInit;		// initialize MLLR matrix with statistics inferred from UBM? (robustness) 
	string _transformFileExt; // .ext of outfile with transformation matrices
	bool _saveMixOcc;      // save also occupation of Gaussians? (stored in the )
	int _tau;			  // relevance factor for MAP adaptation
	int _iterMLLR;		  // num. of itarations for MLLR
	int _iterCountEM;	  // num. of EM iterations - for MAP and training from scratch
	int _iterCountMAP;	  // num. of MAP iterations (MAP only)
	bool _getQvalMLLR;    // evaluates values pf MLLR criterion for each row of matrix W
	bool _AdaptMeanBool;   // adapt means?
	bool _AdaptVarBool;    // adapt vars?
	bool _AdaptWeightBool; // adapt weights?
	bool _robustCov;	   // robust estimate of full-cov matrix
	float _llTh;		   // log-like-threshold (if model trained from scratch);
						   // if log-like after two splits < _llTh then estimation ends
	float _delMix;		  // for greater than 0, Gaussians with alfa lower then _delMix will be discarded	
	float _delMixPercentageTh; // if more than 'delMixPercentage' percents of Gaussians should be discarded, nothing happens
	float _min_var; // minimal allowed variance (for diagonal cov matrix)
	bool _SSEacc;
	bool _CUDAacc;
	int _cudaGPUid;   // specify which GPU should be used if several GPUs available, for 0 all available GPUs will be used (default = 0)
	unsigned int _cudaNaccBlocks; // how many accumulators should be computed on GPU in parallel (they are added at the end)
	int _load_type;
	int _saveModelAfterNSplits; // if training from scratch, after _saveModelAfterNSplits the actual model is saved
	unsigned int _prmSamplePeriod;
	unsigned int _dwnsmp;
	unsigned int _numThreads;

	bool _iDataChange;

	struct model_i {
		unsigned int NzeroG; // num. of Gaussianov with zero weight
		unsigned int NnonZeroG; // num. of Gaussianov with non-zero weight
		unsigned int iMaxW;  // index of Gaussian with max weight
		std::list<unsigned int> iGsorted; // sorted indexes of Gaussians according to their weight		
		std::list<unsigned int> idsZeroG; // indexes of Gaussian with zero weight
	};

	// change UBM, new UBM has to have same number of Gaussians as the old one
	void SetUBM(GMModel *new_UBM);

	void ErrorCheck(bool initialization_required,
					bool mixturesNotDeleted_required, const char *funcName);

	// check invertibility of Gi matrix in fMLLR/MLLR (condition necessary, not sufficient!)
	void CheckInvertibility();
	
	void AccumulateStats(GMModel *model, GMMStats<STATS_TYPE> &stats,
					     bool mixProbsOnly = false, bool meanStats = true,
						 bool varStats = false, bool varFull = false,
						 bool auxS = false);
				
	// discard Gaussians with small weight from the input model
	void RemoveWeakMixtures(GMModel **model);
	
	// one iteration = pull model toward data
	void IterEM(GMModel *model, int iterCount);

	// one iteration = pull model toward UBM
	void IterMAP(GMModel *model, int iterCount);

	void UpdateModelMAP(GMModel *model, GMModel *UBM);

	GMModel *AdaptModelMAP(GMModel *ubm); 

	// one EM iteration, number of Gaussians with non-zero weight is returned
	unsigned int UpdateModelEM(GMModel *model, GMMStats<STATS_TYPE> &stats,
						       bool upweight = true, bool upmean = true, bool upvar = true);

	// according to specified Gaussians (idx in mixIndexes) ki a Gi are computed (MLLR);
	// ki and Gi have to be properly allocated -> dim(ki) = dim(feature_vector+1) x 1, dim(Gi) = dim(feature_vector+1) x dim(feature_vector+1)
	double ComputeKiGi_MLLR(GMModel *model, int *mixIndexes, int Nmixes, MATRIX *ki, MATRIX *Gi, 
						    MATRIX *Ginit, MATRIX *kinit, int i);
	
	MATRIX *IterW_MLLR(GMModel *model, unsigned int iterNum, vector<double> *Qvals = NULL);
	
	void UpdateMLLRMatrix(MATRIX *W_sum, MATRIX *W_new);

	// compute a global MLLR matrix (W) - common for all means of GMM
	void ComputeW_MLLR(GMModel *model, MATRIX &W, MATRIX &ki, MATRIX &Gi, 
					   vector<MATRIX> &Ginits, vector<MATRIX> &kinits,
					   MATRIX &wi, int *mixIndexes, vector<double> *Qvals = NULL);
	void initMLLRMatrices(MATRIX **W, MATRIX **W_tmp, MATRIX &Gi, MATRIX &ki, int **mixIndexes);
	void prepareKiGiInitMatrices(vector<MATRIX> &Gis, vector<MATRIX> &kis, GMModel &UBM);

	// transform model according to already computed W
	GMModel *AdaptModelMLLR(GMModel *UBM, MATRIX *W);	

	// according to specified Gaussians (idx in mixIndexes) ki a Gi are computed (fMLLR);
	// ki and Gi have to be properly allocated -> dim(ki) = dim(feature_vector+1) x 1, dim(Gi) = dim(feature_vector+1) x dim(feature_vector+1)
	double ComputeKiGi_fMLLR(GMModel *model, int *mixIndexes, int Nmixes, MATRIX *ki, MATRIX *Gi, int i);

	// reestimate A and b matrices; W = [A,b] -> A must be initialized as an identity matrix,
	// b as a zero vector; dim(b) = dim(feature_vector) x 1, dim(A) = dim(feature_vector) x dim(feature_vector)
	void EstimateAb_fmllr(MATRIX *A, MATRIX *b, MATRIX *ki, MATRIX *Gi, MATRIX *Giinv, MATRIX *cof, float totocc, int i);

	// compute a global fMLLR matrix (W) - common for all means of GMM
	MATRIX *ComputeW_fMLLR(GMModel *model);

	void SaveTRNMatrices(const char *FileName, bool save_txt);
	
	void SaveModel(const char *FileName, bool save_txt);

	void DeleteClientModel();
	void DeleteTRNMatrices();
	void DeleteAll(); 

	bool CheckOptions();
	void CheckBeforeAdapt();

	// (robust cov) estimate number of independent frames
	void GetNIndFrames(GMMStats<STATS_TYPE> &stats, double *NindFrames);

	// (robust cov) compute scale of diagonal and non-diagonal elements of cov matrix
	float getDiagCovMultiplier(double NindFrame);
	float getOffDiagCovMultiplier(double NindFrame);
	
	double* MakeRobustStats(GMMStats<STATS_TYPE> &stats, bool return_NindFrames = false);

	// available estimators: CPU/CPU-SSE/CUDA
	void InitializeEstimator();

	// split a Gaussian to 2 in the direction of max spread of feature vectors
	void splitGaussian(GMModel &model, unsigned int m2Split, unsigned int mnew);
	void splitMultipleGaussians(GMModel &model);	

	void GetGaussStats(GMModel &model, CModelAdapt::model_i &minfo, bool getList);
};

#endif
