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

#ifdef __GNUC__
#include "general/my_inttypes.h"
#endif

#include "general/Exception.h"
#include "tools/OL_FileList.h"
#include "trainer/OL_ModelAdapt.h"
#include "trainer/OL_GMMStatsEstimator_SSE.h"
#ifdef _CUDA
#	include "trainer/OL_GMMStatsEstimator_CUDA.h"
#endif

#include <new>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>
#include <time.h>
#include <time.h>
#include <cmath> // linux: isnan(x) 
#ifdef _MKL
#	include <mkl.h>
#elif _ACML
#	include <acml.h>
#else
#	error "ACML or MKL have to be available"
#endif

#if defined(WIN32) || defined(WIN64)
#define isnan(x) _isnan(x) // linux
#endif


using namespace stdext;
namespace po = boost::program_options;


namespace {
		void getEigValVecs(float *square_mat, unsigned int dim, 
		float *eig_vec, float *eig_val, 
		int &N_eig_val, int verbosity) 
	{	
	// INFO: http://www.intel.com/software/products/mkl/docs/WebHelp/lse/functn_syevr.html

#ifdef _ACML
#	define MO_INT int
#elif _MKL
#	define MO_INT MKL_INT
#endif
	
		// INPUT PARAMS:
		MO_INT N = dim; // matrix order
		MO_INT lda = dim; // first dim of 'square_mat'
		MO_INT ldz = dim; // leading dim of 'square_mat'

		MO_INT il = dim, iu = dim; // indices of eigenvalues to be returned 
		// in ascending order; 1 <= il <= iu <= N	
		MO_INT lwork = -1, liwork = -1; // workspace query - optimal size of work array 
		MO_INT N_eig_found; // output: total number of eigenvalues found (0 <= N_eig_found <= N)

		// to be searched for eigenvalues
		MO_INT iwkopt;
		MO_INT* iwork; // auxiliary workspace array

		// OUTPUT PARAMS:
		MO_INT info; // info = 0  : estimation OK
		// info = -i : the i-th parameter had an illegal value
		// info = i  : an internal error has occurred		
		MO_INT *isuppz; // support of the eigenvectors, indices 
		// indicating the nonzero elements in eig_vec

		if((isuppz = new(nothrow) MO_INT [dim]) == NULL)
			EXCEPTION_THROW(ModelSVM, "\n\tgetEigValVec(): Not enough memory!\n\t ", true);

		// is returned in 'wkopt' & 'iwkopt'
		float abstol = -1; // absolute error tolerance (-1 => default value is used)
		float vl = 0.0f, vu = numeric_limits<float>::max(); // lower and upper bounds of the interval 
		float wkopt;
		float* work; // auxiliary workspace array

		float *inmat;
		if((inmat = new(nothrow) float [dim * dim]) == NULL)
			EXCEPTION_THROW(ModelSVM, "\n\tgetEigValVec(): Not enough memory!\n\t ", true);

		memcpy(inmat, square_mat, sizeof(float) * dim * dim);

		// ask for size to alloc
#ifdef _ACML
		float *ev;
		if((ev = new(nothrow) float [dim]) == NULL)
			EXCEPTION_THROW(ModelSVM, "\n\tgetEigValVec(): Not enough memory!\n\t ", true);

		ssyevr('V', 'I', 'U', N, inmat, lda, vl, vu, il, iu,
			abstol, &N_eig_found, ev, eig_vec, ldz, isuppz, &info);
#elif _MKL
		SSYEVR("V", "I", "U", &N, inmat, &lda, &vl, &vu, &il, &iu,
			&abstol, &N_eig_found, eig_val, eig_vec, &ldz, isuppz, &wkopt, &lwork, &iwkopt, &liwork,
			&info);
		//if(verbosity)
		//	cout << " getEigValVecs(): Alloc sizes = [" << wkopt << "," << iwkopt << "]" << endl;

		// alloc auxiliary arrays
		lwork = (int) wkopt;
		if((work = new(nothrow) float [lwork]) == NULL)
			EXCEPTION_THROW(ModelSVM, "\n\tgetEigValVec(): Not enough memory!\n\t ", true);
		liwork = iwkopt;

		if((iwork = new(nothrow) MO_INT [liwork]) == NULL)
			EXCEPTION_THROW(ModelSVM, "\n\tgetEigValVec(): Not enough memory!\n\t ", true);
		
		// estimate eigen values & eigen vectors

		SSYEVR("V", "I", "U", &N, inmat, &lda, &vl, &vu, &il, &iu,
			&abstol, &N_eig_found, eig_val, eig_vec, &ldz, isuppz, work, &lwork, iwork, &liwork,
			&info);
		
		delete work;
		delete iwork;
#endif
		
		// errors occured?
		if(info != 0)
			EXCEPTION_THROW(ModelSVM, "\n\tgetEigValVec(): Estimation of eigenvalues failed [info = " << info << "]!\n\t ", true);
		
		N_eig_val = N_eig_found;

#ifdef _ACML
		*eig_val = ev[0];
		delete ev;
#endif

		delete isuppz;
		delete inmat;
	}
}
//==================================================================
CModelAdapt::CModelAdapt() {	
	this->_estimator = NULL;

	this->_configLoaded = false;
	this->_initialized = false;	
	this->_mixturesDeleted = false;
	this->_modelTrained = false;
	this->_trnmatComputed = false;	
	this->_dataInserted = false;
	this->_AdaptWeightOnly = false;
		
	this->_adaptType = MAP;
	this->_fromScratch = false;
	this->_splitXpercAtOnce = 0.0f;
	this->_splitTh = 0.0f;
	this->_applyTransforms = true;
	this->_saveTransforms = true;
	this->_mllrUBMInit = true;
	this->_transformFileExt.assign(".mat");
	this->_iter_fMLLR = 20;

	this->_UBM = NULL;	
	this->_modelSP = NULL;
	this->_W = NULL;
	this->_W_Qvals = NULL;
	
	this->_prmlist = NULL;
	this->_framelist = NULL;

	this->_nummix = 0;
	this->_dim = 0;

	this->_tau = 0;
	this->_AdaptMeanBool = true;
	this->_AdaptVarBool = false;
	this->_AdaptWeightBool = false;
	this->_robustCov = false;
	this->_llTh = 0.0f;

	this->_iterMLLR = 1;
	this->_iterCountEM = 0;
	this->_iterCountMAP = 0;
	this->_getQvalMLLR = false;
	this->_saveMixOcc = false;	
	
	this->_delMix = 0.0;
	this->_delMixPercentageTh = 50.0;

	//this->_data = NULL;
	//this->_NSamples = 0;
	this->_min_var = 1e-6f;


	this->_SSEacc = false;
	this->_CUDAacc = false;
	this->_cudaGPUid = -1;
	this->_cudaNaccBlocks = 8;	
	this->_numThreads = 1;
	this->_memoryBufferSizeGB = 0.1f;
	this->_memoryBufferSizeGB_GPU = 0.0f;

	this->_load_type = 0;
	this->_prmSamplePeriod = 1;
	this->_dwnsmp = 1;
	this->_verbosity = 0;

	this->_saveModelAfterNSplits = 0;
	this->_iDataChange = true;

} //kostruktor

//==================================================================
CModelAdapt::~CModelAdapt(){
	DeleteAll();
}

//==================================================================
void CModelAdapt::InitializeEstimator() {	
	if(_estimator != NULL)
		return;

	if(_SSEacc) {
		_estimator = new GMMStatsEstimator_SSE <STATS_TYPE>;
	}
#ifdef _CUDA
	else if(_CUDAacc)
		_estimator = new GMMStatsEstimator_CUDA <STATS_TYPE>(_cudaGPUid);
#endif
	else {
		_estimator = new GMMStatsEstimator <STATS_TYPE>;
	}
	_estimator->_min_var = _min_var;
	_estimator->_NAccBlocks_CUDA = _cudaNaccBlocks;

	if(_verbosity)
		cout << " [" << _estimator->getEstimationType().c_str() << " Estimation ON]" << endl;
}
//==================================================================
void CModelAdapt::Initialize(GMModel *UBM) {
	if(!_configLoaded)
		EXCEPTION_THROW(ModelAdaptException, "Initialize(): Load/Set configuration first! \n\t ", true);				

	if(UBM == NULL)
		EXCEPTION_THROW(ModelAdaptException, "Initialize(): UBM not given! \n\t ", true);				

	_UBM = UBM;
	_nummix = _UBM->GetNumberOfMixtures();
	_dim = _UBM->GetDimension();				
	_AdaptWeightOnly = _AdaptWeightBool & !_AdaptMeanBool & !_AdaptVarBool;

	// alloc memory for statistics according to adaptation types	
	bool fullV = UBM->GetFullCovStatus();
	switch(_adaptType) {
			case FROM_SCRATCH:
				_stats.alloc(_dim, _nummix, true, true, fullV, _robustCov);
				break;
			case MAP:
				_stats.alloc(_dim, _nummix, _AdaptMeanBool | _AdaptVarBool, _AdaptVarBool, fullV, _robustCov);
				break;
			case MLLR:
				_stats.alloc(_dim, _nummix, true, false, false);
				break;
			case fMLLR:
				_stats.alloc(_dim, _nummix, true, true, true);
				break;
			case MLLR_MAP:
				_stats.alloc(_dim, _nummix, true, _AdaptVarBool, fullV, _robustCov);
				break;
	}		

	if (_estimator == NULL)
		InitializeEstimator();

	_initialized = true;
}
//==================================================================
void CModelAdapt::InsertData(Param& data, bool linkPointer) {
	_param.Add(data, linkPointer);
	_dataInserted = true;
	_iDataChange = true;
}
//==================================================================
// continuos adaptation - file after file
void CModelAdapt::InsertData(CFileList *list) {
	if(_prmlist != NULL)
		EXCEPTION_THROW(ModelAdaptException, "InsertData(CFileList *): adaptation list already inserted! \n\t ", true);	
	_prmlist = list;
	_dataInserted = true;
	_iDataChange = true;
}
//==================================================================
// continuos adaptation - file after file - with specification of frames
void CModelAdapt::InsertData(hash_map <string, list < vector<unsigned int> > >* framelist) {
	if(_framelist != NULL)
		EXCEPTION_THROW(ModelAdaptException, "InsertData(hash_map *): adaptation list already inserted! \n\t ", true);	
	_framelist = framelist;
	_dataInserted = true;
	_iDataChange = true;
}
//==================================================================
GMModel *CModelAdapt::GetClientModelCopy() {
	
	if(!_modelTrained)
		return NULL;
	
	GMModel *modelSP;
	if((modelSP = new(nothrow) GMModel(_modelSP)) == NULL)
		EXCEPTION_THROW(ModelAdaptException, "GetClientModelCopy(): Not enough memory! \n\t ", true);	

	return modelSP;	
}
//==================================================================
GMModel *CModelAdapt::PickUpClientModel() {
	
	if(!_modelTrained)
		return NULL;
	
	GMModel *sp = _modelSP;
	_modelSP = NULL;
	_modelTrained = false;

	return sp;
}
//==================================================================
MATRIX *CModelAdapt::GetTRNMatrixCopy() {
	
	if(!_trnmatComputed)
		return NULL;
	
	MATRIX *W;
	if((W = new(nothrow) MATRIX) == NULL)
		EXCEPTION_THROW(ModelAdaptException, "GetTRNMatrixCopy(): Not enough memory! \n\t ", true);	
	if(W->Create(_W) != RETURN_OK)
		EXCEPTION_THROW(ModelAdaptException, "GetTRNMatrixCopy(): Not enough memory! \n\t ", true);	

	return W;	
}
//==================================================================
MATRIX *CModelAdapt::PickUpTRNMatrix() {
	
	if(!_trnmatComputed)
		return NULL;
	
	MATRIX *W = _W;
	_W = NULL;
	_trnmatComputed = false;

	return W;
}

//==================================================================
void CModelAdapt::DeleteAll() {
	if(_estimator != NULL)
		delete _estimator;

	DeleteTRNMatrices();
	DeleteClientModel();
}
//==================================================================
void CModelAdapt::DeleteTRNMatrices() {
	if(_W != NULL) {
		_W->Delete();
		delete _W;
		_W = NULL;		
	}
	if(_W_Qvals != NULL) {
		delete _W_Qvals;
		_W_Qvals = NULL;
	}
	_trnmatComputed = false;
}
//==================================================================
void CModelAdapt::DeleteClientModel() {
	if(_modelSP != NULL) {
		delete _modelSP;
		_modelSP = NULL;		
	}
	_modelTrained = false;
}
//==================================================================
void CModelAdapt::AccumulateStats(GMModel *model, GMMStats<STATS_TYPE> &stats,
								  bool mixProbsOnly /*= false*/, bool meanStats /*= true*/,
								  bool varStats /*= false*/, bool varFull /*= false*/, 
								  bool auxS /*= false*/)
{
#ifdef __CHECK_INPUT_PARAMS__	
	if (_estimator == NULL)
		EXCEPTION_THROW(ModelAdaptException, "AccumulateStats(): Estimator not initialized! \n\t ", true);
#endif 

	if(!_dataInserted)
		EXCEPTION_THROW(ModelAdaptException, "AccumulateStats(): Insert data first! \n\t ", true);

	if(_verbosity > 2)
		cout << "\n\t\tAccumulating statistics ... ";

	_estimator->_verbosity = _verbosity;	
	_estimator->_numThreads = _numThreads;
	_estimator->_memoryBuffDataGB = _memoryBufferSizeGB;
	_estimator->_memoryBuffDataGB_GPU = _memoryBufferSizeGB_GPU;

	if (model != NULL) {
		_estimator->insertModel(*model);
		_estimator->setModelToBeUsed(0);
	}
	_estimator->setAccFlags(mixProbsOnly, meanStats, varStats, varFull, auxS);

	clock_t time_b, time_e;	
	time_b = clock();

	if(_framelist != NULL) {
		_estimator->accumulateStatsMT(*_framelist, stats, _load_type, _prmSamplePeriod);
		_iDataChange = true;
	}

	if(_prmlist != NULL) {
		_estimator->accumulateStatsMT(*_prmlist, stats, _load_type, _dwnsmp);
		_iDataChange = true;
	}

	if(_param.GetNumberOfVectors() > 0) {
		if (!_iDataChange)
			_estimator->setSameDataFlag();
		_estimator->accumulateStatsMT(*_param.GetVectors(), _param.GetNumberOfVectors(), _param.GetVectorDim(), stats);
		_iDataChange = false;
	}

	time_e = clock();
	//cout << "&" << (time_e - time_b)/(double) CLOCKS_PER_SEC<< "&\n"; // LMA TMP

	if(_verbosity)
		cout << "[#frames accumulated = "<< stats.getTotAccSamples() << "(in " << (time_e - time_b)/(double) CLOCKS_PER_SEC<< " secs)]" << endl;
	
	if(_verbosity > 2)
		cout << "done" << endl;
}
//==================================================================
void CModelAdapt::GetNIndFrames(GMMStats<STATS_TYPE> &stats, double *NindFrames) {
	
	unsigned int dim = stats.getDim();
	unsigned int nummix = stats.getMixNum();

	double *glob_std;
	if((glob_std = new(nothrow) double[dim]) == NULL)
		EXCEPTION_THROW(ModelAdaptException, "GetNIndFrames(): Not enough memory! \n\t ", true);
	double *glob_mean;
	if((glob_mean = new(nothrow) double[dim]) == NULL)
		EXCEPTION_THROW(ModelAdaptException, "GetNIndFrames(): Not enough memory! \n\t ", true);

	double glob_mixProb = 0;
	memset(glob_std, 0, sizeof(double) * dim);
	memset(glob_mean, 0, sizeof(double) * dim);

	double sumNg = 0;
	for(unsigned int m = 0; m < nummix; m++) {
		glob_mixProb += stats[m].mixProb;
		unsigned int shift = 0;
		for(unsigned int d = 0; d < dim; d++) {
			glob_mean[d] += stats[m].mean[d];
			if(stats.hasFullCov()) {
				shift += d;
				glob_std[d] += stats[m].var[d * dim + d - shift];				
			}
			else glob_std[d] += stats[m].var[d];
		}
		NindFrames[m] = 0;
		sumNg += stats[m].aux2;
	}

	
	for(unsigned int d = 0; d < dim; d++) {
		glob_mean[d] /= glob_mixProb;
		glob_std[d] = sqrt(glob_std[d] / glob_mixProb - glob_mean[d] * glob_mean[d]);
		if (glob_std[d] == 0) 
			EXCEPTION_THROW(ModelAdaptException, "GetNIndFrames(): global variance equal zero! \n\t ", true);
	}
	

	//if(nummix > 1) printf("MII: "); //LLLLLLLLLLLLL
	 
	for(unsigned int m = 0; m < nummix; m++) {
		if (stats[m].mixProb == 0 || stats[m].aux3 == 0) {
			NindFrames[m] = 0;
		}
		else {
			for(unsigned int d = 0; d < dim; d++) {
				NindFrames[m] += stats[m].aux[d] / stats[m].aux3 / glob_std[d];
			}
			NindFrames[m] /= dim;

			//if(nummix > 1) printf("%f ", NindFrames[m]);//LLLLLLLLLLLLLLL

			// Dg = ( MII/1.12 ).^3; -> viz trainGMM.m -> 'robust' verze			
			NindFrames[m] = pow(NindFrames[m] / 1.12, 3);
			NindFrames[m] = 1 + (stats[m].aux2 * glob_mixProb / sumNg - 1) * NindFrames[m];
		}
	}	

	//LLLLLLLLLLLLLLL
	//if(nummix == 2) printf("sum: %f %f   sumSqrt: %f %f   Ng: %f %f\n", stats[0].aux3, stats[1].aux3, stats[0].aux2, stats[1].aux2, NindFrames[0], NindFrames[1]);
	//if(nummix >  2) printf("sum: %f %f %f  sumSqrt: %f %f %f  Ng: %f %f %f\n", stats[0].aux3, stats[1].aux3, stats[2].aux3, stats[0].aux2, stats[1].aux2, stats[2].aux2, NindFrames[0], NindFrames[1], NindFrames[2]);

	delete [] glob_std;
	delete [] glob_mean;
}
//==================================================================
float CModelAdapt::getDiagCovMultiplier(double NindFrame) {
	if (NindFrame < 2)
		return (float) pow(pow((1.0 + (1.0/(2.0 - 1.25))), 3.5), (3-NindFrame));

	return (float) pow((1.0 + (1.0/(NindFrame - 1.25))), 3.5);
}
float CModelAdapt::getOffDiagCovMultiplier(double NindFrame) {
	if (NindFrame < 2)
		return 0.0f;

	return (float) (1 - pow((_dim / (_dim - 1 + NindFrame)), 1.4));
}
//==================================================================
double *CModelAdapt::MakeRobustStats(GMMStats<STATS_TYPE> &stats, bool return_NindFrames) {
	
	unsigned int nummix = stats.getMixNum();
	unsigned int dim = stats.getDim();

	double *NindFrames;
	if((NindFrames = new(nothrow) double[nummix]) == NULL)
		EXCEPTION_THROW(ModelAdaptException, "MakeRobustStats(): Not enough memory! \n\t ", true);

	GetNIndFrames(stats, NindFrames);

	for(int j = 0; j < nummix; j++) {
		float dc = getDiagCovMultiplier(NindFrames[j]);
		dc *= NindFrames[j] / (NindFrames[j] - 1);		

		if (stats.hasFullCov()) {
			float odc = getOffDiagCovMultiplier(NindFrames[j]);
			unsigned int shift = 0;
			for(int k = 0; k < dim - 1; k++) {
				shift += k;
				stats[j].var[k * dim + k - shift] = (STATS_TYPE) (dc * stats[j].var[k * dim + k - shift] + (1 - dc) * pow( (double) stats[j].mean[k], 2.0) / stats[j].mixProb);
				for(int kk = k + 1; kk < dim; kk++)
					stats[j].var[k * dim + kk - shift] = (STATS_TYPE) (odc * stats[j].var[k * dim + kk - shift] + (1 - odc) * stats[j].mean[k] * stats[j].mean[kk] / stats[j].mixProb);
			}
			stats[j].var[stats.getVarDim() - 1] = (STATS_TYPE) (dc * stats[j].var[stats.getVarDim() - 1] + (1 - dc) * pow( (double) stats[j].mean[dim - 1], 2.0) / stats[j].mixProb);
		}
		else {
			for(int k = 0; k < dim; k++)
				stats[j].var[k] = (STATS_TYPE) (dc * stats[j].var[k] + (1 - dc) * pow( (double) stats[j].mean[k], 2.0) / stats[j].mixProb);
		}
	}

	if (return_NindFrames)
		return NindFrames;
	else {
		delete [] NindFrames;
		return NULL;
	}
}
//==================================================================
void CModelAdapt::splitMultipleGaussians(GMModel &model) {

	model_i minfo;
	GetGaussStats(model, minfo, _splitXpercAtOnce > 0.0);

	unsigned int Nsplit = (unsigned int) ceil(_splitXpercAtOnce * minfo.NnonZeroG);

	if (Nsplit < 2)
		splitGaussian(model, minfo.iMaxW, minfo.idsZeroG.front());
	else {
		float th = _splitTh * model.GetMixtureWeight(minfo.iMaxW);		

		unsigned int n = 0;
		std::list<unsigned int>::iterator it = minfo.iGsorted.begin();
		std::list<unsigned int>::iterator it_z = minfo.idsZeroG.begin();
		for ( ; it != minfo.iGsorted.end(); ++it, ++it_z, ++n) {
			if (model.GetMixtureWeight(*it) < th || it_z == minfo.idsZeroG.end() || n >= Nsplit)
				break;

			splitGaussian(model, *it, *it_z);
		}
	}
}
//==================================================================
void CModelAdapt::splitGaussian(GMModel &model, unsigned int m2Split, unsigned int mnew) {	
	
	int N_eig_val = 0;
	float *eig_vec, eig_val;

	unsigned int dim = model.GetDimension();
	
	GMMMixture *toSplit = model.GetMixture(m2Split);
	GMMMixture *newG = model.GetMixture(mnew);

	newG->SetVar(model.GetMixtureVar(m2Split));

	float w = model.GetMixtureWeight(m2Split) / 2.0f;
	model.SetMixWeight(mnew, w);
	model.SetMixWeight(m2Split, w);

	float *mshift = new float[dim];
	if (model.GetFullCovStatus()) {				
		if((eig_vec = new(nothrow) float[dim]) == NULL)
			EXCEPTION_THROW(ModelAdaptException, "splitGaussian(): Not enough memory! \n\t ", true);
		
		getEigValVecs(model.GetMixtureVar(m2Split), dim, eig_vec, &eig_val, N_eig_val, _verbosity);			
		
		if (N_eig_val < 1)
			EXCEPTION_THROW(ModelAdaptException, "splitGaussian(): None leading eigenvector found - zero covariance matrix? \n\t ", true);
		if (eig_val < 0)
			EXCEPTION_THROW(ModelAdaptException, "splitGaussian(): covariance matrix not positive definite! \n\t ", true);

		for (unsigned int d = 0; d < dim; d++)
			mshift[d] = sqrt(eig_val) * eig_vec[d];

		delete [] eig_vec;
	}
	else {
		for (unsigned int d = 0; d < dim; d++)
			mshift[d] = sqrt(toSplit->Var[d]) * 0.5;
	}

	float *m = model.GetMixtureMean(m2Split);
	float *mN = model.GetMixtureMean(mnew);
	for (unsigned int d = 0; d < dim; d++) {
		mN[d] = m[d] + mshift[d];
		m[d] -= mshift[d];		
	}

	delete [] mshift;
}
//==================================================================
void CModelAdapt::TrainFromScratch(unsigned int nummix, bool fullcov,
								   GMModel *initGMM, const char *FileName, bool save_txt) {

	if (initGMM != NULL && fullcov && !initGMM->GetFullCovStatus()) {
		cerr << " [WARNING: full-covariance model requested, but the inserted initial-GMM is only diagonal -> switching to diagonal GMMs]" << endl;
		fullcov = false;
	}

	//if (_verbosity > 0 && !fullcov && _robustCov) {
	//	cerr << " [WARNING: robust covariance estimates supported only with full-covariance matrices]" << endl;
	//	cerr << " [         robust will be turned off during estimation]" << endl;
	//}

	if (_verbosity)
		cout << "\n\t\tEstimating model \n\t\t(fullcov = " << fullcov << ", robust = " << _robustCov << ", nummix = " << nummix << ", log-lik-th = " << _llTh << ")";

	_adaptType = FROM_SCRATCH;
	InitializeEstimator();

	unsigned int dim;
	unsigned int maxIter;

	if(_param.GetNumberOfVectors() <= 3) {
		//throw std::exception("TrainFromScratch(): Not enough data to model training.");
		EXCEPTION_THROW(ModelAdaptException, "TrainFromScratch(): Not enough data to model training.\n\t ", true);
	}
	
	bool initGMMinserted = true;
	if (initGMM == NULL) {
		initGMMinserted = false;

		if(_verbosity)
			std::cout << "\n     [EM it = 0] ";

		// estimate: global mean and variance
		GMMStats<STATS_TYPE> global_stats;
		AccumulateStats(NULL, global_stats, false, true, true, fullcov, _robustCov);

		//global_stats.save("STATS.st", true, true, true, true);

		dim = global_stats.getDim();
		_dim = dim;
		maxIter = (unsigned int) (nummix);

		initGMM = new GMModel(dim, 1, fullcov);
		initGMM->SetMixWeight(0, 1.0f);		
		if (UpdateModelEM(initGMM, global_stats) == 0)
			EXCEPTION_THROW(ModelAdaptException, "trainFromScratch(): amount of data too low to estimate a model, or data are constant! \n\t ", true);		
	}
	else {
		if (nummix <= initGMM->GetNumberOfMixtures())
			EXCEPTION_THROW(ModelAdaptException, "trainFromScratch(): init model already contains specified number of Gaussians! \n\t ", true);

		dim = initGMM->GetDimension();
		_dim = dim;
		maxIter = nummix - initGMM->GetNumberOfMixtures();
	}
	

	// init model
	GMModel m2, m3, *model = new GMModel(dim, nummix, fullcov);
	Initialize(model);

	if (fullcov && initGMM->GetFullCovStatus()) {
		for (unsigned int m = 0; m < initGMM->GetNumberOfMixtures(); m++)
			model->SetMixParam(m, initGMM->GetMixtureMean(m), initGMM->GetMixtureVar(m), initGMM->GetMixtureWeight(m));	
	}
	else {
		for (unsigned int m = 0; m < initGMM->GetNumberOfMixtures(); m++)
			model->SetMixParam(m, initGMM->GetMixtureMean(m), initGMM->GetMixtureVarDiag(m), initGMM->GetMixtureWeight(m));	
	}

	if (nummix < 2)
		maxIter = 0;

	// if a threshold is used for log-like, at least 2 EM iterations are needed
	if (_iterCountEM < 2 && _llTh > 0.0f) {
		cout << " (num of EM iteartions < 2 -> it-EM set to 2)" << endl;
		_iterCountEM = 2;
	}
	
	float ll[3];
	ll[0] = ll[1] = ll[2] = -numeric_limits<float>::max();
	
	// estimation of model params	
	unsigned int nonZeroG = initGMM->GetNumberOfMixtures();
	for (unsigned int it = 0; it < maxIter; it++) {
		if(_verbosity)
			std::cout << "\n [Split num = " << it + 1 << "]" << std::endl;

		for (unsigned int itEM = 0; itEM < _iterCountEM; itEM++) {
			if(_verbosity)
				std::cout << "\n     [EM it = " << itEM + 1 << "] ";
			
			_stats.reset();
			AccumulateStats(model, _stats, false, true, true, fullcov, _robustCov); 
			
			if(_verbosity) {
				ll[0] = (double) (_stats.getTotLogLike() / (double) _stats.getTotAccSamples());
				std::cout << "                 [LL = " << ll[0] << "]  ";
			}			
			
			if (nonZeroG > 1)
				nonZeroG = UpdateModelEM(model, _stats);
						
			if (nonZeroG < 2) {	
				if (nonZeroG < 1) {	
					if (_verbosity)
						cerr << "[WARNING: All Gaussians discarded - keeping the initial/global model]" << endl;

					model->copy(initGMM);
				}
				break;
			}			
		}

		if (nonZeroG == 0 || nonZeroG == nummix)
			break;

		if (_llTh > 0.0f) {
			float mx = max(ll[1], ll[2]);
			if (it > 2 && ll[0] - mx < _llTh) {
				if (mx > ll[0]) {
					if (mx == ll[1])
						model->copy(&m2);
					else
						model->copy(&m3);					
				}
				break;
			}
			m3.copy(&m2);
			m2.copy(model);		
			ll[2] = ll[1];
			ll[1] = ll[0];
		}		
		if (_saveModelAfterNSplits > 0 && ((it + 1) % _saveModelAfterNSplits == 0)) {
			stringstream ss;
			ss << FileName << ".it" << setfill('0') << setw(5) << it << ".G" << setfill('0') << setw(5) << nonZeroG << ".gmm.bckup";
			model->Save(ss.str().c_str(), save_txt);
		}
				
		if(it < maxIter - 1) {
			splitMultipleGaussians(*model);
			nonZeroG += 1;
		}
	}		

	unsigned int NzeroG = model->RearrangeMixtures();	
	if (_verbosity)
		cout << "\n #Gauss (left) in GMM = " << (float) (nummix - NzeroG) << endl;

	if (_modelSP != NULL)
		delete _modelSP;
	_modelSP = model;

	if (!initGMMinserted)
		delete initGMM;

	_modelTrained = true;
}
//==================================================================
unsigned int CModelAdapt::UpdateModelEM(GMModel *model, GMMStats<STATS_TYPE> &stats,
										bool upweight, bool upmean, bool upvar) {

#ifdef __CHECK_INPUT_PARAMS__
	if(model == NULL)
		EXCEPTION_THROW(ModelAdaptException, "UpdateModelMAP(): Model not specified! \n\t ", true);

	if(model->GetNumberOfMixtures() != UBM->GetNumberOfMixtures())
		ErrorCheck(true, true, true, "UpdateModelEM()");		
	else
		ErrorCheck(true, true, false, "UpdateModelEM()");
#endif 

	// TMP
	//stats.save("H:/WORK/experimenty/recnik/UBMs/UBM0002_P0001_NIST040506_M2048_KWGMM/STATS.st", true, true, true, true);
	//stats.save("H:/WORK/experimenty/recnik/UBMs/UBM0002_P0001_NIST040506_M2048_KWGMM/MPROBS.st", true, true, false, false);

	if(_verbosity > 2)
		cout << "\t\tEstimating model ... ";
		
	int dim = model->GetDimension();
	int nummixM = model->GetNumberOfMixtures();	
	int nummixS = stats.getMixNum();	
	int nummix = (nummixM > nummixS) ? nummixS : nummixM;

	//bool robust = _robustCov && model->GetFullCovStatus();

	double *NindFrames;
	if (_robustCov)
		NindFrames = MakeRobustStats(stats, true);	
	else {		
		if((NindFrames = new(nothrow) double[nummix]) == NULL)
			EXCEPTION_THROW(ModelAdaptException, "UpdateModelEM(): Not enough memory! \n\t ", true);
		for(int j = 0; j < nummix; j++)
			NindFrames[j] = stats[j].mixProb;
	}

	float *mean;
	if((mean = new(nothrow) float[dim]) == NULL)
		EXCEPTION_THROW(ModelAdaptException, "UpdateModelEM(): Not enough memory! \n\t ", true);

	bool mixWchange = false;
	// compute weights, mean and vars for each Gaussian
	unsigned int NzeroMixs = 0;
	double gaussCorrection = 0.0;
	for(int j = 0; j < nummix; j++) 
	{
		if (model->GetMixtureWeight(j) == 0.0f) {
			++NzeroMixs;
			continue;			
		}

		GMMMixture *mixture = model->GetMixture(j);
		STATS_TYPE mixProb = stats.getProb(j);

		if (_saveMixOcc)
			mixture->setMixOcc((float) mixProb);

		if (NindFrames[j] < _MIN_IND_SAMPLES_) {
			mixture->Reset();
			model->SetMixWeight(j, 0.0f);
			mixWchange = true;
			continue;
		}		

		if (upweight) {
			float mweight = (float) (mixProb / stats.getTotAccSamples());
			model->SetMixWeight(j, mweight);			
			mixWchange = true;
		}

		gaussCorrection += model->GetMixtureWeight(j);

		if (upmean || upvar) {
			STATS_TYPE *ms_mean = const_cast<STATS_TYPE *> (stats.getMean(j));
			STATS_TYPE *ms_var = const_cast<STATS_TYPE *> (stats.getVar(j));

			for(int k = 0; k < dim; k++)
				mean[k] = (float) (ms_mean[k] / mixProb);

			if (upvar) {
				if (!model->GetFullCovStatus()) {
					for(int k = 0; k < dim; k++) {
						float var = (float) (ms_var[k] / mixProb - pow((double) mean[k], 2));
						if (var < _min_var)
							var = _min_var;
						mixture->Var[k] = var;
					}
				}
				else {
					if (!_robustCov && NindFrames[j] < dim) {
						cerr << " [WARNING: covariance matrix for Gaussian with idx=" << j << " is likely to be ill conditioned (not enough data!)]" << endl;
						cerr << " [         try to turn on --robust when working with full-covariance matrices]" << endl;
					}
					unsigned int shift = 0;
					for(int k = 0; k < dim; k++) {
						shift += k;
						for(int kk = k; kk < dim; kk++) {
							float var = (float) (ms_var[k * dim + kk - shift] / mixProb - mean[k] * mean[kk]);
							mixture->Var[k * dim + kk] = var;
							mixture->Var[kk * dim + k] = var;
						}
					}
				}
			}

			if (upmean)
				memcpy(mixture->Mean, mean, dim * sizeof(float));
		} 
	}

	//if(NzeroMixs == nummix)
	//	EXCEPTION_THROW(ModelAdaptException, "UpdateModelEM(): All Gaussians were discarded from the model - data & model too diferent! \n\t ", true);

	if(mixWchange && gaussCorrection > 0.0) {
		for(int j = 0; j < nummix; j++)
			model->SetMixWeight(j, (float) (model->GetMixtureWeight(j) / gaussCorrection));
	}

	delete [] mean;	
	delete [] NindFrames;

	if(_verbosity > 2)
		cout << "done" << endl;

	return nummix-NzeroMixs;
}
//==================================================================
void CModelAdapt::GetGaussStats(GMModel &model, CModelAdapt::model_i &minfo, bool getList) {

	minfo.iMaxW = 0;
	minfo.NzeroG = 0;
	minfo.NnonZeroG = 0;
	minfo.iGsorted.clear();
	minfo.idsZeroG.clear();	

	int nummix = model.GetNumberOfMixtures();
	float wm = 0.0f;
	for (int m = nummix - 1; m >= 0; m--) { //! if m would be unsigned int, m could never be less than 0 and the cycle would never end!
		float w = model.GetMixtureWeight(m);
		if (wm < w) {
			wm = w;
			minfo.iMaxW = m;
		}
		if (w == 0) {
			minfo.NzeroG += 1;
			minfo.idsZeroG.push_back(m);
			continue;
		}
		if (getList) {
			if (minfo.iGsorted.empty()) {
				minfo.iGsorted.push_back(m);
				continue;
			}
			if (minfo.iGsorted.size() == 1) {
				if (model.GetMixtureWeight(minfo.iGsorted.front()) > w)
					minfo.iGsorted.push_back(m);
				else
					minfo.iGsorted.push_front(m);
				continue;
			}
			std::list<unsigned int>::iterator it = minfo.iGsorted.begin();
			while (model.GetMixtureWeight(*it) > w && ++it != minfo.iGsorted.end());
			minfo.iGsorted.insert(it, m);
		}
	}	
	minfo.NnonZeroG = nummix - minfo.NzeroG;
}

//==================================================================
void CModelAdapt::UpdateModelMAP(GMModel *model, GMModel *UBM) {

	if (_tau == 0) {
		if (UpdateModelEM(model, _stats, _AdaptWeightBool, _AdaptMeanBool, _AdaptVarBool) == 0) {
			cerr << "\n [WARNING: All Gaussians discarded - keeping the initial/global model]" << endl;
			model->copy(UBM);
		}
		return;
	}

#ifdef __CHECK_INPUT_PARAMS__
	if(model == NULL)
		EXCEPTION_THROW(ModelAdaptException, "UpdateModelMAP(): Model not specified! \n\t ", true);
	if(UBM == NULL)
		EXCEPTION_THROW(ModelAdaptException, "UpdateModelMAP(): UBM not specified! \n\t ", true);

	if(model->GetNumberOfMixtures() != UBM->GetNumberOfMixtures())
		ErrorCheck(true, true, true, "UpdateModelMAP()");		
	else
		ErrorCheck(true, true, false, "UpdateModelMAP()");
#endif 
	
	// TMP
	//_stats.save("H:/WORK/experimenty/recnik/UBMs/UBM0002_P0001_NIST040506_M2048_KWGMM/STATS.st", true, true, true, true);
	//_stats.save("H:/WORK/experimenty/recnik/UBMs/UBM0002_P0001_NIST040506_M2048_KWGMM/MPROBS.st", true, true, false, false);

	if(_verbosity > 2)
		cout << "\t\tEstimating client model ... ";

	float adaptMixWeight, *adaptMean, *adaptVar;
	int nummix = model->GetNumberOfMixtures();
	int dim = model->GetDimension();
	if((adaptMean = new(nothrow) float[dim]) == NULL)
		EXCEPTION_THROW(ModelAdaptException, "UpdateModelMAP(): Not enough memory! \n\t ", true);
	
	unsigned int dimvar = dim;
	if(UBM->GetFullCovStatus()) 
		dimvar *= dim;
	if((adaptVar = new(nothrow) float[dimvar]) == NULL)
		EXCEPTION_THROW(ModelAdaptException, "UpdateModelMAP(): Not enough memory! \n\t ", true);

	//bool robust = _robustCov && _stats.hasFullCov();
		
	//double *NindFrames = NULL;
	if (_robustCov) {
		double *NindFrames = MakeRobustStats(_stats);		
		delete [] NindFrames;
	}

	double alfa, mixCorrection = 0.0;
	// compute weights, mean and vars for each Gaussian
	for(int j = 0; j < nummix; j++) 
	{
		if (UBM->GetMixtureWeight(j) == 0)
			continue;

		GMMMixture *mixture = model->GetMixture(j);
		STATS_TYPE mixProb = _stats.getProb(j);
		
		if(_saveMixOcc) 			
			mixture->setMixOcc((float) mixProb);

		if(mixProb > 0.0f) {			
			float *UBMmean = UBM->GetMixtureMean(j);
			float *UBMvar = UBM->GetMixtureVar(j);		
			float UBMmixWeight = UBM->GetMixtureWeight(j);

			alfa = mixProb/(mixProb + _tau);		
			if(_AdaptWeightBool && _stats.getTotAccSamples() > 0) {
				adaptMixWeight = (float) (alfa*(mixProb/_stats.getTotAccSamples()) + (1 - alfa)*UBMmixWeight); 
				model->SetMixWeight(j, adaptMixWeight);				
				mixCorrection += adaptMixWeight;				
			}
			if(_AdaptMeanBool || _AdaptVarBool) {
				STATS_TYPE *ms_mean = const_cast<STATS_TYPE *> (_stats.getMean(j));
				STATS_TYPE *ms_var = const_cast<STATS_TYPE *> (_stats.getVar(j));
				
				for(int k = 0; k < dim; k++)
					adaptMean[k] = (float) (alfa*ms_mean[k]/mixProb + (1 - alfa)*UBMmean[k]);					
				
				if(_AdaptVarBool) {
					unsigned int shift = 0;
					for(unsigned int k = 0; k < dim; k++) {
						if(!_stats.hasFullCov()) {
							adaptVar[k] = (float) (alfa*ms_var[k]/mixProb + (1 - alfa)*(UBMvar[k] + UBMmean[k]*UBMmean[k]) - adaptMean[k]*adaptMean[k]);
							if (adaptVar[k] < _min_var)
								adaptVar[k] = _min_var;
						}
						else {
							shift += k;
							for(unsigned int kk = k; kk < dim; kk++) {
								adaptVar[k * dim + kk] = (float) (alfa *ms_var[k * dim + kk - shift]/mixProb + (1 - alfa)*(UBMvar[k * dim + kk] + UBMmean[k]*UBMmean[kk]) - adaptMean[k]*adaptMean[kk]);
								adaptVar[kk * dim + k] = adaptVar[k * dim + kk];
							}
						}
					}
				}
				mixture->SetMean(adaptMean);
				if(_AdaptVarBool)
					mixture->SetVar(adaptVar);
			}
		} // if mixProb
	} 
	if(_AdaptWeightBool && mixCorrection != 0) {
		for(int j = 0; j < nummix; j++) {
			model->SetMixWeight(j, model->GetMixtureWeight(j)/((float)mixCorrection));
		}
	}
	delete [] adaptMean;
	delete [] adaptVar;

	if(_verbosity > 2)
		cout << "done" << endl;
}
//==================================================================
void CModelAdapt::RemoveWeakMixtures(GMModel **model) {
	int *deleteMixtureBool, delModel_nummix = 0, adaptMixIND = 0;	
	double alfa, delMixCorrection = 0.0;	

#ifdef __CHECK_INPUT_PARAMS__
	if(model == NULL || *model == NULL)
		EXCEPTION_THROW(ModelAdaptException, "RemoveWeakMixtures(): Model not specified! \n\t ", true);
	ErrorCheck(true, true, false, "RemoveWeakMixtures()");
#endif

	int nummix = (*model)->GetNumberOfMixtures();
	int dim = (*model)->GetDimension();
	if((deleteMixtureBool = new(nothrow) int[nummix]) == NULL) 
		EXCEPTION_THROW(ModelAdaptException, "RemoveWeakMixtures(): Not enough memory! \n\t ", true);
		
	for(int j = 0; j < nummix; j++) {	
		STATS_TYPE mixProb = _stats.getProb(j);
		alfa = mixProb/(mixProb + _tau);		
		if(alfa >= _delMix) {
			delMixCorrection += (*model)->GetMixtureWeight(j);			
			deleteMixtureBool[j] = 0;
			delModel_nummix += 1;	
		}	
		else deleteMixtureBool[j] = 1;
	}
	float delprob = 100*(1 - delModel_nummix/((float) nummix));
	if(delprob > _delMixPercentageTh) {	
		cerr << "WARNING: RemoveWeakMixtures(): None of the mixtures deleted" << endl;
		cerr << "(" << delprob << "% [>" << _delMixPercentageTh << "%] should be removed!)" << endl;
	}	
	else if(delModel_nummix < nummix) {
		if(_verbosity > 1) 
			cout << "\t" << delprob << "% of mixtures will be removed" << endl;
		GMModel *delMixModel = new(nothrow) GMModel(dim, delModel_nummix, (*model)->GetFullCovStatus());
		for(int j = 0; j < nummix; j++) {
			STATS_TYPE mixProb = _stats.getProb(j);
			if(deleteMixtureBool[j] == 0) {
				GMMMixture *mixture = delMixModel->GetMixture(adaptMixIND);

				// weights have to be recomputed, because some of the Gaussians were discardpretoze niektore zlozky boli vymazane, je treba vahu znovu prepocitat
				delMixModel->SetMixWeight(adaptMixIND, ((*model)->GetMixtureWeight(j)/ (float) delMixCorrection));
				mixture->SetMean((*model)->GetMixtureMean(j));
				mixture->SetVar((*model)->GetMixtureVar(j));
				if(_saveMixOcc)
					delMixModel->GetMixture(adaptMixIND)->setMixOcc((float) mixProb);
				adaptMixIND += 1;
			} 
		} 	
		delete *model;
		*model = delMixModel;
		_mixturesDeleted = true;
	}
	delete deleteMixtureBool;
}
//==================================================================
void CModelAdapt::AdaptModel(GMModel *UBM) {

	if (_adaptType == FROM_SCRATCH)
		return;

	CheckBeforeAdapt();	
	Initialize(UBM);

	switch(_adaptType) {
		case MAP: 
			_modelSP = AdaptModelMAP(_UBM);
			_modelTrained = true;
			break;
		case MLLR:
			if (_getQvalMLLR)
				_W_Qvals = new vector<double>;
			_W = IterW_MLLR(_UBM, _iterMLLR, _W_Qvals);
			_trnmatComputed = true;
			if(_applyTransforms) {
				_modelSP = AdaptModelMLLR(_UBM, _W);
				_modelTrained = true;
			}			
			break;
		case fMLLR:
			_W = ComputeW_fMLLR(_UBM);
			_trnmatComputed = true;
			break;
		case MLLR_MAP:
			MATRIX *Wfoo = IterW_MLLR(_UBM, _iterMLLR);
			if(Wfoo != NULL) {				
				GMModel *model_mllr = AdaptModelMLLR(_UBM, Wfoo);
				_modelSP = AdaptModelMAP(model_mllr);
				
				Wfoo->Delete();
				delete Wfoo;
				delete model_mllr;
			}
			else {
				cerr << "[WARNING: not nough data to perform MLLR adaptation -> only MAP performed]" << endl;
				_modelSP = AdaptModelMAP(_UBM);
			}
			_modelTrained = true;
			break;
		//default:
		//	EXCEPTION_THROW(ModelAdaptException, "AdaptModel(): Unknown adaptation type! \n\t ", true);
		//	break;
	}	

	if(_delMix && _modelTrained) {
		if(_verbosity > 1)
			cout << endl << "\tdelMix ON [delMix=" << _delMix << ", delMixPercentage=" << _delMixPercentageTh << "%]" << endl; 
		// if model has not changed (_iterCountEM = 0 & _iterCountMAP = 0) => at least Gaussians 
		// with small weight are discarded from UBM (still, statistics have to be computed)
		if(_stats.getTotAccSamples() < 1) {						
			_stats.reset();
			AccumulateStats(_modelSP, _stats, true, false);
		}
		RemoveWeakMixtures(&_modelSP);		
	} 
	else {
		if(_verbosity > 1)
			cout << endl << "\tdelMix OFF" << endl; 
	}
}
//==================================================================
void CModelAdapt::CheckBeforeAdapt() {
	if(!_dataInserted)
		EXCEPTION_THROW(ModelAdaptException, "AdaptModel(): Insert adaptation data first! \n\t ", true);

	if((_adaptType == MAP || _adaptType == MLLR_MAP || _applyTransforms == true) && _modelSP != NULL) {
		EXCEPTION_THROW(ModelAdaptException, "AdaptModel(): Model already adapted! \n\t ", true);
		//if(_verbosity) {			
		//	cout << "AdaptModel(): Model already adapted!" << endl;
		//	cout << "Before next adaptation delete (DeleteClientModel()) or pick up (PickUpClientModel()) the existing model!" << endl;
		//}
		//return 1;
	}
	if(_adaptType != MAP && _W != NULL) {
		EXCEPTION_THROW(ModelAdaptException, "AdaptModel(): Transform. matrix already computed! \n\t ", true);
		//if(_verbosity) {
		//	cout << "AdaptModel(): Transform. matrix already computed!" << endl;
		//	cout << "Before next adaptation delete (DeleteTRNMatrices()) or pick up (PickUpTRNMatrices()) the existing matrix!" << endl;
		//}
		//return 1;
	}
}
//==================================================================
GMModel *CModelAdapt::AdaptModelMAP(GMModel *ubm) {			
	
#ifdef __CHECK_INPUT_PARAMS__
	if(ubm == NULL)
		EXCEPTION_THROW(ModelAdaptException, "AdaptModelMAP(): UBM model not specified! \n\t ", true);
	ErrorCheck(true, false, true, "AdaptModel()");
#endif

	GMModel *model;
	if((model = new(nothrow) GMModel(ubm)) == NULL)
		EXCEPTION_THROW(ModelAdaptException, "AdaptModel(): Not enough memory! \n\t ", true);

	if(_iterCountEM == 0 && _iterCountMAP == 0) {
			cerr << "AdaptModel(): WARNING: iterCountEM = iterCountMAP = 0 => nothing to do (speaker model == UBM)!" << endl;		
	}

	if(_verbosity)
		cout << " [MAP adaptation: itEM=" << _iterCountEM << ", itMAP=" << _iterCountMAP << "] ";
	if(_iterCountEM)
		IterEM(model, _iterCountEM);
	if(_iterCountMAP)
		IterMAP(model, _iterCountMAP);

	return model;
} //CModelAdapt::AdaptModel
//==================================================================
void CModelAdapt::IterEM(GMModel *model, int iterCount) {
		
#ifdef __CHECK_INPUT_PARAMS__
	if(model == NULL)
		EXCEPTION_THROW(ModelAdaptException, "IterEM(): Model not specified! \n\t ", true);
	ErrorCheck(true, false, false, "IterEM()");
#endif

	if(_verbosity > 2)
		cout << endl << "\t[IterEM]";
		
	for(int i = 0; i < iterCount; i++) {
		_stats.reset();		
		AccumulateStats(model, _stats, _AdaptWeightOnly, _AdaptMeanBool, 
					    _AdaptVarBool, model->GetFullCovStatus(), _robustCov);
		GMModel model_temp(model);
		UpdateModelMAP(model, &model_temp);
		if(_verbosity && _verbosity < 3)
			cout << ".";
	}
}
//==================================================================
void CModelAdapt::IterMAP(GMModel *model, int iterCount) {

#ifdef __CHECK_INPUT_PARAMS__
	if(model == NULL)
		EXCEPTION_THROW(ModelAdaptException, "IterMAP(): Model not specified! \n\t ", true);
	ErrorCheck(true, false, true, "IterMAP()");
#endif 

	if(_verbosity > 2)
		cout << endl << "\t[IterMAP]";
	
	for(int i = 0; i < iterCount; i++) { 
		_stats.reset();
		AccumulateStats(model, _stats, _AdaptWeightOnly, _AdaptMeanBool, 
					    _AdaptVarBool, model->GetFullCovStatus(), _robustCov);
		UpdateModelMAP(model, _UBM);
		if(_verbosity && _verbosity < 3)
			cout << ".";
	}
}
//==================================================================
void CModelAdapt::UpdateMLLRMatrix(MATRIX *W_sum, MATRIX *W_new) {
	
	if(W_sum->size_x != W_new->size_x || W_sum->size_y != W_new->size_y)
		EXCEPTION_THROW(ModelAdaptException, "UpdateMLLRMatrix(): Incompatible matrix sizes!\n\t ", true);
	
	MATRIX W_bckup;
	if(W_bckup.Create(W_sum) != RETURN_OK)
		EXCEPTION_THROW(ModelAdaptException, "UpdateMLLRMatrix(): Not enough memory!\n\t ", true);
	
	W_sum->SetZero();
	int k, x, y;
	for(k = 0; k < W_sum->size_x; k++) {
		for(x = 0; x < W_sum->size_x; x++) {
			for(y = 0; y < W_sum->size_x; y++) {
				W_sum->da_ta[k*W_sum->size_y + x] += W_new->GetXY(k, y)*W_bckup.GetXY(y, x);			
			}
			W_sum->da_ta[k*W_sum->size_y + y] += W_new->GetXY(k, x)*W_bckup.GetXY(x, y);
		}		
		W_sum->da_ta[k*W_sum->size_y + x] += W_new->GetXY(k, x);
	}	
	W_bckup.Delete();
}
//==================================================================
void CModelAdapt::initMLLRMatrices(MATRIX **W, MATRIX **W_tmp, MATRIX &Gi, MATRIX &ki, int **mixIndexes) {

	if((*W = new(nothrow) MATRIX) == NULL)
		EXCEPTION_THROW(ModelAdaptException, "initMLLRMatrices(): Not enough memory!\n\t ", true);
	if((*W_tmp = new(nothrow) MATRIX) == NULL)
		EXCEPTION_THROW(ModelAdaptException, "initMLLRMatrices(): Not enough memory!\n\t ", true);

	if((*W)->Create(_dim, _dim+1) != RETURN_OK)
		EXCEPTION_THROW(ModelAdaptException, "initMLLRMatrices(): Not enough memory!\n\t ", true);
	if((*W_tmp)->Create(_dim, _dim+1) != RETURN_OK)
		EXCEPTION_THROW(ModelAdaptException, "initMLLRMatrices(): Not enough memory!\n\t ", true);
	if(Gi.Create(_dim+1, _dim+1) != RETURN_OK)
		EXCEPTION_THROW(ModelAdaptException, "initMLLRMatrices(): Not enough memory!\n\t ", true);
	if(ki.Create(_dim+1, 1) != RETURN_OK)
		EXCEPTION_THROW(ModelAdaptException, "initMLLRMatrices(): Not enough memory!\n\t ", true);

	if((*mixIndexes = new(nothrow) int[_nummix]) == NULL)
		EXCEPTION_THROW(ModelAdaptException, "initMLLRMatrices(): Not enough memory!\n\t ", true);
	for(int m = 0; m < _nummix; m++) 
		(*mixIndexes)[m] = m;

}
//==================================================================
MATRIX *CModelAdapt::IterW_MLLR(GMModel *model, unsigned int iterNum, vector<double> *Qvals) {	

	MATRIX Gi, ki, wi, *W, *W_tmp;
	
	int* mixIndexes;
	initMLLRMatrices(&W, &W_tmp, Gi, ki, &mixIndexes);

	vector<MATRIX> Ginits, kinits;
	if(_mllrUBMInit) {
		if(_verbosity)
			cout << " [MLLR init from UBM]" << endl;
		prepareKiGiInitMatrices(Ginits, kinits, *_UBM);
	}

	ComputeW_MLLR(model, *W, ki, Gi, Ginits, kinits, wi, mixIndexes, Qvals);
	if(iterNum > 1) {
		GMModel *modelSP, *model_foo;
		if((modelSP = new(nothrow) GMModel(model)) == NULL)
			EXCEPTION_THROW(ModelAdaptException, "IterW_MLLR(): Not enough memory!\n\t ", true);
		
		model_foo = AdaptModelMLLR(modelSP, W);
		delete modelSP;
		modelSP = model_foo;

		for(unsigned int i = 1; i < iterNum-1; i++) {
			ComputeW_MLLR(modelSP, *W_tmp, ki, Gi, Ginits, kinits, wi, mixIndexes); // Qvals needed only in the last iteration
			model_foo = AdaptModelMLLR(modelSP, W_tmp);
			delete modelSP;
			modelSP = model_foo;

			UpdateMLLRMatrix(W, W_tmp);		
		}
		ComputeW_MLLR(modelSP, *W_tmp, ki, Gi, Ginits, kinits, wi, mixIndexes, Qvals);
		UpdateMLLRMatrix(W, W_tmp);		
		delete model_foo;
	}

	for(int i = 0; i < Ginits.size(); i++) {
		Ginits[i].Delete();
		kinits[i].Delete();
	}
	W_tmp->Delete();
	delete W_tmp;

	Gi.Delete();
	ki.Delete();
	wi.Delete();
	delete [] mixIndexes;
	
	return W;
}
//==================================================================
void CModelAdapt::ComputeW_MLLR(GMModel *model, MATRIX &W, MATRIX &ki, MATRIX &Gi, 
								vector<MATRIX> &Ginits, vector<MATRIX> &kinits, 
								MATRIX &wi, int *mixIndexes, vector<double> *Qvals) 
{	

#ifdef __CHECK_INPUT_PARAMS__
	if(model == NULL)
		EXCEPTION_THROW(ModelAdaptException, "ComputeW_MLLR(): Model not specified!\n\t ", true);
	ErrorCheck(true, false, false, "ComputeW_MLLR()");
#endif

	if(_verbosity)
		cout << " [MLLR adaptation]" << endl;

	_stats.reset();
	AccumulateStats(model, _stats, false, true, false, false, false);
	
	// TMP
	//_stats.save("C:/WORK/experimenty/ruozne/x111_debug_ModelADAPT/STATS.st", true, true, true, false);

	CheckInvertibility();

	if(_verbosity > 1)
		cout << "\tEstimation of Gi & ki => wi ";

	if (Qvals != NULL) 
		Qvals->clear();

	MATRIX *Ginit = NULL, *kinit = NULL, foo;
	for(int i = 0; i < _dim; i++) {
		if(_verbosity > 1)
			cout << ".";
		
		if(Ginits.size() > 0 && kinits.size() > 0) {
			Ginit = &Ginits[i];
			kinit = &kinits[i];
		}
		ComputeKiGi_MLLR(model, mixIndexes, _nummix, &ki, &Gi, Ginit, kinit, i);

		if(wi.Copy(&Gi) != RETURN_OK)
			EXCEPTION_THROW(ModelAdaptException, "ComputeW_MLLR(): matrix copy failed!\n\t ", true);
		wi.Invert();
		if(wi.Mult(&ki) != RETURN_OK)
			EXCEPTION_THROW(ModelAdaptException, "ComputeW_MLLR(): Mult()!\n\t ", true);
		memcpy(W.da_ta + i*W.size_y, wi.da_ta, (_dim+1) * sizeof(wi.da_ta[0]));

		// vypocet hodnoty kriteria	
		if (Qvals != NULL) {			
			if(foo.MultAB(&Gi, &wi) != RETURN_OK)
				EXCEPTION_THROW(ModelAdaptException, "ComputeW_MLLR(): MultAB()!\n\t ", true);		

			double Qk, Q = 0.0;
			for(int k = 0; k <= _dim; k++) {
				Qk = wi.GetXY(k, 0) * (ki.GetXY(k,0) - 0.5*foo.GetXY(k,0));
				Q += Qk;
			}
			Qvals->push_back(Q);
		}		

	}

	foo.Delete();

	if(_verbosity > 1)
		cout << endl;
}
//==================================================================
void CModelAdapt::prepareKiGiInitMatrices(vector<MATRIX> &Gis, vector<MATRIX> &kis, GMModel &UBM) {
	unsigned int M = UBM.GetNumberOfMixtures();
	unsigned int D = UBM.GetDimension();

	Gis.resize(D);
	kis.resize(D);
	for(unsigned int k = 0; k < D; k++) {
		Gis[k].Create(D+1, D+1);
		Gis[k].SetZero();
		kis[k].Create(D+1, 1);
		kis[k].SetZero();
	}

	//vector<MATRIX> Gp, kp;
	//Gp.Create(D+1, D+1);
	//kp.Create(D+1, 1);
	double cG, ck;
	for(unsigned int m = 0; m < M; m++) {
		float *var = UBM.GetMixtureVarDiag(m);
		float *mean = UBM.GetMixtureMean(m);		
		for(unsigned int k = 0; k < D; k++) {			
			if (var[k] < _min_var)
				cG = 1/_min_var;
			else cG = 1/var[k];				
			ck = mean[k] * cG;
			for(unsigned int d = 0; d < D; d++) {
				kis[k].da_ta[d] += ck * mean[d];
				Gis[k].da_ta[D * (D+1) + d] += cG * mean[d];
				Gis[k].da_ta[d * (D+1) + D] += cG * mean[d];

				for(unsigned int dd = 0; dd < D; dd++) {
					Gis[k].da_ta[d * (D+1) + dd] += cG * mean[d] * mean[dd];
				}
				Gis[k].da_ta[d * (D+1) + d] += cG * var[d];
			}
			Gis[k].da_ta[(D+1)*(D+1) - 1] += cG;
			kis[k].da_ta[D] += ck; 
		}
	}
}
//==================================================================
double CModelAdapt::ComputeKiGi_MLLR(GMModel *model, int *mixIndexes, int Nmixes, MATRIX *ki, MATRIX *Gi, 
									 MATRIX *Ginit, MATRIX *kinit, int i) {

#ifdef __CHECK_INPUT_PARAMS__
	if(model == NULL)
		EXCEPTION_THROW(ModelAdaptException, "ComputeKG_MLLR(): Model not specified!\n\t ", true);
	if(mixIndexes == NULL)
		EXCEPTION_THROW(ModelAdaptException, "ComputeKG_MLLR(): Mixture indexes not specified!\n\t ", true);
	if(Nmixes < 0)
		EXCEPTION_THROW(ModelAdaptException, "ComputeKG_MLLR(): Nmixes < 0!\n\t ", true);
	if(ki == NULL)
		EXCEPTION_THROW(ModelAdaptException, "ComputeKG_MLLR(): MATRIX ki has to be allocated!\n\t ", true);
	if(Gi == NULL || *Gi == NULL)
		EXCEPTION_THROW(ModelAdaptException, "ComputeKG_MLLR(): MATRIX Gi has to be allocated!\n\t ", true);
	if(i < 0 || i >= _dim)
		EXCEPTION_THROW(ModelAdaptException, "ComputeKG_MLLR(): Row index 'i' = "<< i << " out of bounds [" << 0 << "," << _dim - 1 << "]!\n\t ", true);
	ErrorCheck(true, true, false, "ComputeKG_MLLR()");
#endif
	if(Ginit != NULL && kinit != NULL) {
		Gi->Copy(Ginit);
		ki->Copy(kinit);
	}
	else {
		Gi->SetZero();
		ki->SetZero();
	}

	double multc_m, multc_v, totocc = 0.0;
	float *modelMean, *modelVar;
	STATS_TYPE *ms_mean;
	for(int idx = 0; idx < Nmixes; idx++) {	
		int m = mixIndexes[idx];

		STATS_TYPE mixProb = _stats.getProb(m);
		if(mixProb > 0) {
			totocc += mixProb;
			modelVar = const_cast<float *> (model->GetMixtureVarDiag(m));
			modelMean = const_cast<float *> (model->GetMixtureMean(m));
			ms_mean =const_cast<STATS_TYPE *> ( _stats.getMean(m));						

			multc_m = ms_mean[i]/modelVar[i];
			multc_v = mixProb/modelVar[i];
			for(int k = 0; k < _dim; k++) {
				ki->da_ta[k] += multc_m*modelMean[k];
				for(int kk = k; kk < _dim; kk++) {
					Gi->da_ta[k *Gi->size_y + kk] += multc_v*modelMean[k]*modelMean[kk];
					Gi->SetXY(kk, k, Gi->GetXY(k, kk)); // faster without the condition (i != ii) 
				} 
				Gi->da_ta[k*Gi->size_y + _dim] += multc_v*modelMean[k];
				Gi->SetXY(_dim, k, Gi->GetXY(k, _dim));
			} // for k
			ki->da_ta[_dim] += multc_m;
			Gi->da_ta[_dim*Gi->size_y + _dim] += multc_v;
		}		
	} // for m	
	return totocc;
}
//==================================================================
GMModel *CModelAdapt::AdaptModelMLLR(GMModel *model, MATRIX *W) {
// POZN: fce sa da zjednodusit!!! miesto mean_copy je mozne pouzivat data z parametru funkce> model!
#ifdef __CHECK_INPUT_PARAMS__
	if(model == NULL)
		EXCEPTION_THROW(ModelAdaptException, "MLLR_modelUpdate(): Model not specified!\n\t ", true);
	if(W == NULL)
		EXCEPTION_THROW(ModelAdaptException, "MLLR_modelUpdate(): MATRIX W not specified!\n\t ", true);
#endif	

	GMModel *upmodel;
	if((upmodel = new(nothrow) GMModel(model)) == NULL)
		EXCEPTION_THROW(ModelAdaptException, "AdaptModelMLLR(): Not enough memory! \n\t ", true);

	float *mean, *mean_copy;
	if((mean_copy = new(nothrow) float [_dim]) == NULL)
		EXCEPTION_THROW(ModelAdaptException, "AdaptModelMLLR(): Not enough memory! \n\t ", true);
	for(int i = 0; i < _nummix; i++) {
		mean = upmodel->GetMixtureMean(i);
		memcpy(mean_copy, mean, sizeof(float)*_dim);
		for(int k = 0; k < _dim; k++){
			double v = 0.0;
			for(int kk = 0; kk < _dim; kk++) {
					v += W->da_ta[k*W->size_y + kk]*mean_copy[kk];
			}
			mean[k] = (float) (v + W->da_ta[k*W->size_y + _dim]);
		}		
	}
	delete [] mean_copy;
	return upmodel;
}
//==================================================================
MATRIX *CModelAdapt::ComputeW_fMLLR(GMModel *model) {	

#ifdef __CHECK_INPUT_PARAMS__
	if(model == NULL)
		EXCEPTION_THROW(ModelAdaptException, "ComputeW_MLLR(): Model not specified!\n\t ", true);
	if(Data == NULL)
		EXCEPTION_THROW(ModelAdaptException, "ComputeW_MLLR(): Data not given!\n\t ", true);
	ErrorCheck(true, false, false, "ComputeW_MLLR()");
#endif

	if(_verbosity)
		cout << " [fMLLR adaptation] " << endl;

	_stats.reset();
	AccumulateStats(model, _stats, false, true, true, true, false);

	CheckInvertibility();

	MATRIX *W, A, b, **k, **G, **Ginv;
	if((W = new(nothrow) MATRIX) == NULL)
		EXCEPTION_THROW(ModelAdaptException, "ComputeW_fMLLR(): Not enough memory!\n\t ", true);
	if(W->Create(_dim, _dim+1) != RETURN_OK)
		EXCEPTION_THROW(ModelAdaptException, "ComputeW_fMLLR(): Not enough memory!\n\t ", true);		
	if(A.Create(_dim, _dim) != RETURN_OK)
		EXCEPTION_THROW(ModelAdaptException, "ComputeW_fMLLR(): Not enough memory!\n\t ", true);
	for(int k = 0; k < _dim; k++)
		A.SetXY(k, k, 1.0);
	if(b.Create(_dim, 1) != RETURN_OK)
		EXCEPTION_THROW(ModelAdaptException, "ComputeW_fMLLR(): Not enough memory!\n\t ", true);
	if((k = new(nothrow) MATRIX* [_dim]) == NULL)
		EXCEPTION_THROW(ModelAdaptException, "ComputeW_fMLLR(): Not enough memory!\n\t ", true);
	if((G = new(nothrow) MATRIX* [_dim]) == NULL)
		EXCEPTION_THROW(ModelAdaptException, "ComputeW_fMLLR(): Not enough memory!\n\t ", true);
	if((Ginv = new(nothrow) MATRIX* [_dim]) == NULL)
		EXCEPTION_THROW(ModelAdaptException, "ComputeW_fMLLR(): Not enough memory!\n\t ", true);

	int *mixIndexes;
	if((mixIndexes = new(nothrow) int[_nummix]) == NULL)
		EXCEPTION_THROW(ModelAdaptException, "ComputeW_fMLLR(): Not enough memory!\n\t ", true);
	for(int m = 0; m < _nummix; m++) 
		mixIndexes[m] = m;

	if(_verbosity > 1)
		cout << "\tEstimation of Gi & ki " << endl;
	for(int i = 0; i < _dim; i++) {
		if((G[i] = new(nothrow) MATRIX) == NULL)
			EXCEPTION_THROW(ModelAdaptException, "ComputeW_fMLLR(): Not enough memory!\n\t ", true);		
		if(G[i]->Create(_dim+1, _dim+1) != RETURN_OK)
			EXCEPTION_THROW(ModelAdaptException, "ComputeW_fMLLR(): Not enough memory!\n\t ", true);
		if((k[i] = new(nothrow) MATRIX) == NULL)
			EXCEPTION_THROW(ModelAdaptException, "ComputeW_fMLLR(): Not enough memory!\n\t ", true);
		if(k[i]->Create(_dim+1, 1) != RETURN_OK)
			EXCEPTION_THROW(ModelAdaptException, "ComputeW_fMLLR(): Not enough memory!\n\t ", true);
		// vypocitam k[i] & G[i]
		ComputeKiGi_fMLLR(model, mixIndexes, _nummix, k[i], G[i], i);
		// invertujem Gi
		if((Ginv[i] = new(nothrow) MATRIX) == NULL)
			EXCEPTION_THROW(ModelAdaptException, "ComputeW_fMLLR(): Not enough memory!\n\t ", true);		
		if(Ginv[i]->Create(G[i]) != RETURN_OK)
			EXCEPTION_THROW(ModelAdaptException, "ComputeW_fMLLR(): Not enough memory!\n\t ", true);
		Ginv[i]->Invert();
	}
	
	if(_verbosity > 1)
		cout << "\tEstimation of A & b [iter_fMLLR = " << _iter_fMLLR << "]" << endl;

	MATRIX cofmat, cof;
	if(cof.Create(_dim+1,1) != RETURN_OK)
		EXCEPTION_THROW(ModelAdaptException, "ComputeW_fMLLR(): Not enough memory!\n\t ", true);	
	for(int iter = 0; iter < _iter_fMLLR; iter++) {
		// get the cofactor of matrix A
		if(cofmat.Create(&A) != RETURN_OK)
			EXCEPTION_THROW(ModelAdaptException, "ComputeW_fMLLR(): Not enough memory!\n\t ", true);	
		if(cofmat.Invert() != RETURN_OK)
			EXCEPTION_THROW(ModelAdaptException, "ComputeW_fMLLR(): Matrix invertion failed!\n\t ", true);	
		cofmat.Mult(A.Det());
		// estimate A and b
		for(int i = 0; i < _dim; i++) {
			if(_verbosity > 1)
				cout << ".";
			// cofactor (vector) -> columns are taken, because cofmat was not transposed
			for(int kk = 0; kk < _dim; kk++)
				cof.SetXY(kk, 0, cofmat.GetXY(kk, i));
			// A & b
			EstimateAb_fmllr(&A, &b, k[i], G[i], Ginv[i], &cof, (float) _stats.getTotAccSamples(), i);
		}
		cofmat.Delete();
	}
	if(_verbosity > 1)
		cout << endl;

	// W = [A,b]
	for(int i = 0; i < _dim; i++) {
		for(int j = 0; j < _dim; j++)
			W->SetXY(i, j, A.GetXY(i, j));		
		W->SetXY(i, _dim, b.GetXY(i, 0));
		// delete
		G[i]->Delete();
		delete G[i];
		Ginv[i]->Delete();
		delete Ginv[i];
		k[i]->Delete();
		delete k[i];
	}
	
	A.Cofact(-1, -1);
	A.Delete();
	b.Delete();
	cof.Delete();
	delete [] mixIndexes;
	delete [] Ginv;
	delete [] G;	
	delete [] k;

	return W;
}
//==================================================================
void CModelAdapt::EstimateAb_fmllr(MATRIX *A, MATRIX *b, MATRIX *ki, MATRIX *Gi, MATRIX *Giinv, MATRIX *cof, float totocc, int i) {
	MATRIX w[2], bb, cc, foo;
	double aa, f[2], Q[2];
	
	// coefficients of the quadratic eq.
	if(foo.MultAtB(cof, Giinv) != RETURN_OK)
		EXCEPTION_THROW(ModelAdaptException, "EstimateAb_fmllr(): MultAtB()!\n\t ", true);
	aa = totocc;
	if(bb.MultAB(&foo, ki) != RETURN_OK)
		EXCEPTION_THROW(ModelAdaptException, "EstimateAb_fmllr(): MultAtB()!\n\t ", true);
	bb.Mult(-1.0);
	if(cc.MultAB(&foo, cof) != RETURN_OK)
		EXCEPTION_THROW(ModelAdaptException, "EstimateAb_fmllr(): MultAtB()!\n\t ", true);
	cc.Mult(-1.0);

	// solving the quadratic equation
	double sqrtD = sqrt(pow(bb.GetXY(0,0),2) - 4*aa*(cc.GetXY(0,0)));
	if(sqrtD < 0 || isnan(sqrtD))
		EXCEPTION_THROW(ModelAdaptException, "EstimateAb_fmllr(): Discriminant < 0!\n\t ", true);
	f[0] = (-bb.GetXY(0,0) + sqrtD)/(2*aa);
	f[1] = (-bb.GetXY(0,0) - sqrtD)/(2*aa);

	bb.Delete();
	cc.Delete();

	// which solution does maximalize the criterion?
	for(int j = 0; j < 2; j++) {
		if(f[j] == 0)
			EXCEPTION_THROW(ModelAdaptException, "EstimateAb_fmllr(): Dividing by zero!\n\t ", true);
		if(w[j].Create(_dim+1, 1) != RETURN_OK)
			EXCEPTION_THROW(ModelAdaptException, "EstimateAb_fmllr(): Not enough memory!\n\t ", true);
		for(int k = 0; k <= _dim; k++) {
			w[j].da_ta[k] = cof->GetXY(k, 0)/f[j] + ki->GetXY(k, 0);
		}
		w[j].Trans();
		if(w[j].Mult(Giinv) != RETURN_OK)
			EXCEPTION_THROW(ModelAdaptException, "EstimateAb_fmllr(): Mult()!\n\t ", true);
		w[j].Trans();
		// criterion
		Q[j] = 0.0;
		if(foo.MultAB(Gi, &w[j]) != RETURN_OK)
			EXCEPTION_THROW(ModelAdaptException, "EstimateAb_fmllr(): MultAB()!\n\t ", true);		
		// double add_term = 0.0;
		for(int k = 0; k <= _dim; k++) {
			Q[j] += w[j].GetXY(k, 0) * (ki->GetXY(k,0) - 0.5*foo.GetXY(k,0));	// LMa 4.6.2009 (-0.5)
			//add_term += w[j].GetXY(k, 0)*cof->GetXY(k, 0);
		}
		// ak by som ho chcel pouzit, malo byt tam byt este krat 'totocc' a nemala by tam byt 
		// absolutna hodnota! - lenze potom moze byt imaginarne, pretoze kofaktor je pocitany zo starej matice!!
		// Q[j] += log(fabs(add_term));  // LMa 15.10.2012
	}	
	int idx = (Q[0] > Q[1]) ? 0 : 1;	
	for(int k = 0; k < _dim; k++)
		A->SetXY(i, k, w[idx].GetXY(k, 0));
	b->SetXY(i, 0, w[idx].GetXY(_dim, 0));
	
	// delete	
	w[0].Delete();
	w[1].Delete();
	foo.Delete();
}
//==================================================================
double CModelAdapt::ComputeKiGi_fMLLR(GMModel *model, int *mixIndexes, int Nmixes, MATRIX *ki, MATRIX *Gi, int i) {

#ifdef __CHECK_INPUT_PARAMS__
	if(model == NULL)
		EXCEPTION_THROW(ModelAdaptException, "ComputeKG_MLLR(): Model not specified!\n\t ", true);
	if(mixIndexes == NULL)
		EXCEPTION_THROW(ModelAdaptException, "ComputeKG_MLLR(): Mixture indexes not specified!\n\t ", true);
	if(Nmixes < 0)
		EXCEPTION_THROW(ModelAdaptException, "ComputeKG_MLLR(): Nmixes < 0!\n\t ", true);
	if(ki == NULL)
		EXCEPTION_THROW(ModelAdaptException, "ComputeKG_MLLR(): MATRIX ki has to be allocated!\n\t ", true);
	if(Gi == NULL || *Gi == NULL)
		EXCEPTION_THROW(ModelAdaptException, "ComputeKG_MLLR(): MATRIX Gi has to be allocated!\n\t ", true);
	if(i < 0 || i >= _dim)
		EXCEPTION_THROW(ModelAdaptException, "ComputeKG_MLLR(): Row index 'i' = "<< i << " out of bounds [" << 0 << "," << _dim - 1 << "]!\n\t ", true);
	ErrorCheck(true, true, false, "ComputeKG_MLLR()");
#endif	

	Gi->SetZero();
	ki->SetZero();

	double multc_m, multc_v, totocc = 0.0;
	float *modelMean, *modelVar;
	STATS_TYPE *ms_mean, *ms_var;
	for(unsigned int idx = 0; idx < Nmixes; idx++) {	
		unsigned int m = mixIndexes[idx];

		STATS_TYPE mixProb = _stats.getProb(m);
		if(mixProb > 0) {
			totocc += mixProb;
			modelVar = model->GetMixtureVarDiag(m);
			modelMean = model->GetMixtureMean(m);
			ms_mean = const_cast<STATS_TYPE *> (_stats.getMean(m));
			ms_var = const_cast<STATS_TYPE *> (_stats.getVar(m));

			multc_m = modelMean[i]/modelVar[i];
			multc_v = 1/modelVar[i];

			unsigned int shift = 0;
			for(unsigned int k = 0; k < _dim; k++) {
				ki->da_ta[k] += multc_m*ms_mean[k];
				shift += k;
				for(unsigned int kk = k; kk < _dim; kk++) {
					Gi->da_ta[k*Gi->size_y + kk] += multc_v*ms_var[k * _dim + kk - shift];
					Gi->SetXY(kk, k, Gi->GetXY(k,kk)); // bez podmienky (i != ii) rychlejsie - funkcnost rovnaka!
				}
				Gi->da_ta[k*Gi->size_y + _dim] += multc_v*ms_mean[k];
				Gi->SetXY(_dim, k, Gi->GetXY(k, _dim));
			} // for k
			ki->da_ta[_dim] += multc_m*mixProb;
			Gi->da_ta[_dim*Gi->size_y + _dim] += multc_v*mixProb;
		}
	} // for m

	return totocc;
}
//==================================================================

void CModelAdapt::CheckInvertibility() {

	if(_nummix < _dim+1) {
		EXCEPTION_THROW(ModelAdaptException, "CheckInvertibility(): In order to use MLLR/fMLLR, number of Gaussians in the model has to be greater than the dimension of feature vectors (use MAP adaptation)!\n\t ", true);
	}

	int activeMixtures = 0;
	double occtot = 0.0;
	for(int i = 0; i < _nummix; i++) {
		STATS_TYPE mixProb = _stats.getProb(i);
		if(mixProb > 0) {
			++activeMixtures;
			occtot += mixProb;
		}
	}
	if(activeMixtures < _dim+1) {
		//if(_verbosity > 1)
		//	cout << "\tWARNING: CheckInvertibility(): To ensure invertibility of matrix Gi, count of active mixtures has to be >= dimension of feature vectors!" << endl;
		EXCEPTION_THROW(ModelAdaptException, "CheckInvertibility(): To ensure invertibility of matrix Gi, count of active mixtures has to be >= dimension of feature vectors (increase amount of data!)!\n\t ", true);
		//return false;
	}
	if(occtot * _dim < _dim*(_dim + 1)) {
		//if(_verbosity > 1)
		//	cout << "\tWARNING: CheckInvertibility(): Not enough data to compute reliable estimate of transformation matrix W!" << endl;
		EXCEPTION_THROW(ModelAdaptException, "CheckInvertibility(): Number of parameters to be estimated is higher than the number of available data (increase amount of data!)!\n\t ", true);
		//return false;
	}
}
//==================================================================
void CModelAdapt::SetUBM(GMModel *UBM) {

#ifdef __CHECK_INPUT_PARAMS__
	ErrorCheck(true, false, true, "SetUBM()");
#endif

	if((UBM->GetDimension() != _UBM->GetDimension()) || 
	   (UBM->GetNumberOfMixtures() != _UBM->GetNumberOfMixtures()))
		EXCEPTION_THROW(ModelAdaptException, "SetUBM(): New UBM incompatible! \n\t ", true);
	_UBM = UBM;
}
//==================================================================
void CModelAdapt::ErrorCheck(bool initialization_required,
							 bool mixturesNotDeleted_required, const char *funcName) {
	if(initialization_required && !_initialized)
		EXCEPTION_THROW(ModelAdaptException, " " << funcName << ": Adaptation not initialized!\n\t", true);
	if(mixturesNotDeleted_required && _mixturesDeleted)
		EXCEPTION_THROW(ModelAdaptException, " " << funcName << ": Model no longer consistent (some mixtures were deleted)!\n\t ", true);
}
//==================================================================
void CModelAdapt::SaveTRNMatrices(const char *FileName, bool save_txt) {
	
	string outfile(FileName);
	size_t dotl = outfile.rfind(".");
	if (dotl != string::npos)
		outfile.erase(dotl, outfile.length()-dotl);
	outfile += _transformFileExt;

	ofstream out(outfile.c_str(), ios::out | ios::trunc);
	if(out.fail())
		EXCEPTION_THROW(MainException, "Save(): Unable to open output file " << outfile << "!\n\t", true);
	
	switch(_adaptType) {
		case MLLR:
			out << " MLLR" << endl;
			break;
		case fMLLR:
			out << " fMLLR" << endl;
			break;
	}
	// format> "_numOfmatrices_\t_XdimOfmat_ _YdimOfmat_"
	out << " 1\t" << _W->size_x << " " << _W->size_y << endl;
	for(int i = 0; i < _W->size_x; i++) {
		for(int j = 0; j < _W->size_y; j++)
			out << " " << _W->GetXY(i,j);
		out << endl;
	}
	out.close();
}
//==================================================================
int CModelAdapt::Save(const char *FileName, bool save_txt, bool save_in_stat_format) {

#ifdef __CHECK_INPUT_PARAMS__
	if(FileName == NULL)
		EXCEPTION_THROW(ModelAdaptException, "Save(): Output model filename not specified!\n\t ", true);
#endif
	// not enough data => matrix W was not calculated => nothing to be stored!
	if(!_modelTrained && !_trnmatComputed) {
		if(_verbosity) {
			cout << "Missing client model or transformation matrix." << endl;
			cout << "(Most likely because of insuffiecient amount of adaptation data.)" << endl;
		}
		return 1;
		// EXCEPTION_THROW(MainException, "Save(): Missing client model or transformation matrix!\n\t", true);
	}

	if(_trnmatComputed && (!_modelTrained || _saveTransforms || _adaptType == fMLLR)) {
		if(_verbosity > 1)
			cout << "(SAVE: trn matrix)" << endl;
		if(!save_in_stat_format)
			SaveTRNMatrices(FileName, save_txt);
		else {
			GMMStats<float> stats;
			stats.alloc(_dim+1, _dim, true, false, false);
			for (unsigned int m = 0; m < _dim; m++) {
				if (_getQvalMLLR)
					stats._ms[m].mixProb = _W_Qvals->at(m);
				else
					stats._ms[m].mixProb = 1.0f;
								
				for (unsigned int d = 0; d <= _dim; d++)
					stats._ms[m].mean[d] = _W->da_ta[m * (_dim + 1) + d];
			}
			stats.save(FileName, save_txt);
		}
	}

	if(_modelTrained) {
		if(_verbosity > 1)
			cout << "(SAVE: model)" << endl;
		if(!save_in_stat_format)
			_modelSP->Save(FileName, save_txt);
		else {
			GMMStats<float> stats;
			stats.alloc(_dim, _nummix, true, true, false);
			for (unsigned int m = 0; m < _modelSP->GetNumberOfMixtures(); m++) {
				stats._ms[m].mixProb = _modelSP->GetMixtureWeight(m);
				
				float *mean = _modelSP->GetMixtureMean(m);
				float *var = _modelSP->GetMixtureVarDiag(m);
				for (unsigned int d = 0; d < _dim; d++) {
					stats._ms[m].mean[d] = mean[d];
					stats._ms[m].var[d] = var[d];
				}
			}
			stats.save(FileName, save_txt);
		}
	}
	return 0;
} // Save
//==================================================================
void CModelAdapt::SetOptions(CModelAdapt::OPTIONS &opt) {
	_verbosity = opt._verbosity;  
	_adaptType = opt._adaptType;
	_fromScratch = opt._fromScratch;
	_splitXpercAtOnce = opt._splitXpercAtOnce;
	_splitTh = opt._splitTh;
	_applyTransforms = opt._applyTransforms;
	_saveTransforms = opt._saveTransforms;
	_mllrUBMInit = opt._mllrUBMInit;
	_transformFileExt.assign(opt._transformFileExt);
	_saveMixOcc = opt._saveMixOcc;
	_tau = opt._tau;
	_iterMLLR = opt._iterMLLR;
	_iterCountEM = opt._iterCountEM;
	_iterCountMAP = opt._iterCountMAP;
	_getQvalMLLR = opt._getQvalMLLR;
	_AdaptMeanBool = opt._AdaptMeanBool;
	_AdaptVarBool = opt._AdaptVarBool;
	_AdaptWeightBool = opt._AdaptWeightBool;
	_robustCov = opt._robustCov;
	_llTh = opt._llTh;
	_delMix = opt._delMix;
	_delMixPercentageTh = opt._delMixPercentageTh;
	_SSEacc = opt._SSEacc;
	_CUDAacc = opt._CUDAacc;
	_cudaGPUid = opt._cudaGPUid;
	_cudaNaccBlocks = opt._cudaNaccBlocks;
	_numThreads = opt._numThreads;
	_load_type = opt._load_type;
	_saveModelAfterNSplits = opt._saveModelAfterNSplits;
	_min_var = opt._min_var;
	_prmSamplePeriod = opt._prmSamplePeriod;
	_dwnsmp = opt._dwnsmp;
	_memoryBufferSizeGB = opt._memoryBufferSizeGB;
	_memoryBufferSizeGB_GPU = opt._memoryBufferSizeGB_GPU;
	
	CheckOptions();

	_configLoaded = true;
}
//==================================================================
po::options_description *CModelAdapt::GetOptions(CModelAdapt::OPTIONS &opt) {

	po::options_description *desc;
	if((desc = new(nothrow) po::options_description("Options-ADAPT")) == NULL)
		EXCEPTION_THROW(ModelAdaptException, "Constructor(): Not enough memory! \n\t ", true);						

	try {				
		// parse command line		
		desc->add_options()
			("type,t", po::value<int>(&opt._adaptType),
			"FROM_SCRATCH=0, MLLR=1, MAP=2, fMLLR=3, MLLR+MAP=4\nnote: if FROM_SCRATCH also --nummix option has to be specified")

			("split-X", po::value<float>(&opt._splitXpercAtOnce)->default_value(0.0f),
			"(FROM_SCRATCH) number from interval <0,1>; if > 0 => 'split-X'*100 percents of available Gaussians will be split at once in each iteration;\nweight of these Gaussians must be greater than 'split-th'*max_gauss_weight")

			("split-th", po::value<float>(&opt._splitTh)->default_value(0.0f),
			"(FROM_SCRATCH) number from interval <0,1>; see option --split-X for more details; note: --split-th=1 is equal to --split-X=0")

			("robust,R", po::value<bool>(&opt._robustCov)->implicit_value(true)->default_value(false),
			"(FROM_SCRATCH|MAP) robust estimation of the GMM full covariance matrices")

			("ll-Th", po::value<float>(&opt._llTh)->default_value(0.0f),
			"(FROM_SCRATCH) if the difference of log-likelihoods of models between two iterations is lower than the threshold, the estimation will end; otherwise it will continue until the number of iterations reaches number of specified Gaussians")

			("it-EM", po::value<int>(&opt._iterCountEM)->default_value(0),
			"(MAP|FROM_SCRATCH) after each iteration UBM is replaced by the newly adapted model => computation of statistics => adaptation => ...")

			("it-MAP", po::value<int>(&opt._iterCountMAP)->default_value(0),
			"(MAP) statistic are computed from the last adapted model, but UBM is used for weighing")

			("mean,M", po::value<bool>(&opt._AdaptMeanBool)->implicit_value(true)->default_value(false),
			"(MAP) mixture mean will be adapted")

			("var,V", po::value<bool>(&opt._AdaptVarBool)->implicit_value(true)->default_value(false),
			"(MAP) mixture variance will be adapted")

			("weight,W", po::value<bool>(&opt._AdaptWeightBool)->implicit_value(true)->default_value(false),
			"(MAP) mixture weight will be adapted")

			("tau,r", po::value<int>(&opt._tau)->default_value(0),
			"(MAP) relevance factor")

			("SSE", po::value<bool>(&opt._SSEacc)->implicit_value(true)->default_value(false),
			"accumulate statistics using SSE instructions (speed boost)")

#ifdef _CUDA
			("CUDA", po::value<bool>(&opt._CUDAacc)->implicit_value(true)->default_value(false),
			"accumulate statistics using GPU CUDA (speed boost)")

			("GPU-id,G", po::value<int>(&opt._cudaGPUid)->default_value(-1),
			"if more GPUs are available, which one should be used (0 - (#available_devices-1)); for --GPU-id=-1 the computation will be distributed among all available GPUs")

			("NB-GPU", po::value<unsigned int>(&opt._cudaNaccBlocks)->default_value(8),
			"number of accumulators on GPU - summed at the end; usefull (speed up) for lower amount of Gaussians (~32)")

			("prm-buffer-gpu", po::value<float>(&opt._memoryBufferSizeGB_GPU)->default_value(0.0f),
			"maximum size of memory used on GPU given in GB")
#endif
			("min-var", po::value<float>(&opt._min_var)->default_value(1e-6f),
			"minimal allowed variance occurring in the model with diagonal covariances")

			("numThrd", po::value<unsigned int>(&opt._numThreads)->default_value(1),
			"number of CPUs used")

			("prm-buffer", po::value<float>(&opt._memoryBufferSizeGB)->default_value(0.05f),
			"maximum size of a buffer used to load the feature vectors given in GB; takes effect only if file-by-file = true")

			("tmp-N", po::value<int>(&opt._saveModelAfterNSplits)->default_value(0),
			"(FROM_SCRATCH) save a temporary model after each tmp-N splits")

			("mat-ext,X", po::value<string>(&opt._transformFileExt)->default_value(".mat"),
			"output extension of file with transformation matrix")

			("apply-trn,a", po::value<bool>(&opt._applyTransforms)->implicit_value(true)->default_value(false),
			"(MLLR) after adaptation MLLR matrix is used to transform the model")

			("save-trn,s", po::value<bool>(&opt._saveTransforms)->implicit_value(true)->default_value(false),
			"(MLLR) save transformation matrices")

			("mllr-init", po::value<bool>(&opt._mllrUBMInit)->implicit_value(true)->default_value(false),
			"(MLLR) MLLR will be initialized with statistics extracted from UBM to increase the robustness of the estimation (useful mainly when only a few data are available)")
			
			("it-mllr", po::value<int>(&opt._iterMLLR)->default_value(1),
			"(MLLR) number of MLLR iterations")
			
			("mllr-Qval", po::value<bool>(&opt._getQvalMLLR)->implicit_value(true)->default_value(false),
			"(MLLR) values of the MLLR criterion for each row of W will be computed; they will be saved when save-in-stat-form is true => they are stored as stats.mixProbs")

			("del-mix", po::value<float>(&opt._delMix)->default_value(0),
			"if > 0 => mixtures with occupation < del-mix are discarded")

			("del-mix-p", po::value<float>(&opt._delMixPercentageTh)->default_value(50),
			"none of the mixtures is discarded if more than del-mix-p % of mixtures should be discarded")
			;
	}		
	catch(exception &e)
	{
		EXCEPTION_THROW(ModelAdaptException, " " << e.what() << "\n\t", true);
	}	
	return desc;
}

// ===================================================================================
bool CModelAdapt::CheckOptions() {

	if (_fromScratch) {
		_adaptType = FROM_SCRATCH;
	}

	if(_min_var < numeric_limits<float>::min()) {
		cerr << endl << "\tWARNING: \"min-var\" < min value of float! - set to default = 1.18e-038\n";			
		_min_var = numeric_limits<float>::min();
	}
	if(_adaptType != FROM_SCRATCH && _adaptType != MLLR && _adaptType != MAP && _adaptType != fMLLR && _adaptType != MLLR_MAP) {
		cerr << endl << "\tWARNING: \"type\" unrecognized, set to default = MAP\n";			
		_adaptType = MAP;
	}
	if(_adaptType == MAP || _adaptType == MLLR_MAP) {
		if(!_AdaptMeanBool && !_AdaptVarBool && !_AdaptWeightBool) {
			EXCEPTION_THROW(ModelAdaptException, " Nothing to do for MAP adaptation, nor mean, nor var, nor weight adaptation set (see options -M, -V, -W)" , true);
			//if(_verbosity) {
			//	cout << endl << " Nothing to do for MAP adaptation," << endl;
			//	cout << "mean, var, weight adaptation not set (see options -M,-V,-W)" << endl;
			//	cout << "\t-> mean adaptation ON" << endl;
			//}
			//_AdaptMeanBool = true;
		}
	}
	if(_SSEacc && _CUDAacc) {
		cerr << endl << "\tWARNING: Both \"SSE\" & \"CUDA\" set -> \"CUDAacc\" is going to be used (if possible)\n";
		_SSEacc = false;
	}
	if(_iterMLLR < 1) {
		cerr << endl << "\tWARNING: \"it-MLLR\" < 1! - set to default = 1\n";
		_iterMLLR = 1;
	}
	if(_tau < 0) {
		cerr << endl << "\tWARNING: \"tau\" < 0! - set to default = 0\n";			
		_tau = 0;
	}
	if(_iterCountEM < 0) {
		cerr << endl << "\tWARNING: \"it-EM\" < 0! - set to default = 0\n";			
		_iterCountEM = 0; 
	}
	if(_iterCountMAP < 0) {
		cerr << endl << "\tWARNING: \"it-MAP\" < 0! - set to default = 0\n";			
		_iterCountMAP = 0;
	}
	if(_delMix < 0.0) {
		cerr << endl << "\tWARNING: \"del-mix\" < 0.0! - set to default = 0.0\n";			
		_delMix = 0.0;
	}
	if(_delMixPercentageTh < 0.0 || _delMixPercentageTh >= 100.0) {
		cerr << endl << "\tWARNING: \"delMixPercentageTh\" out of bounds [0,100)! - set to default = 50.0\n";			
		_delMixPercentageTh = 50.0;
	}
	if(!_fromScratch && (!_AdaptVarBool || _adaptType == MLLR || _adaptType == fMLLR)) {
		_robustCov = false;
	}
	if(_llTh < 0.0f) {
		_llTh = 0.0f;
	}
	if(_cudaNaccBlocks < 0) {
		cerr << endl << "\tWARNING: \"NB-GPU\" < 0! - set to default = 8\n";			
		_cudaNaccBlocks = 8;
	}
	if(_splitXpercAtOnce > 1.0f) {
		cerr << endl << "\tWARNING: \"split-X\" > 1, but has to be in <0,1>! - set to 1\n";			
		_splitXpercAtOnce = 1.0f;
	}
	if(_splitXpercAtOnce < 0.0f) {
		cerr << endl << "\tWARNING: \"split-X\" < 0, but has to be in <0,1>! - set to 0\n";			
		_splitXpercAtOnce = 0.0f;
	}
	if(_splitTh > 1.0f) {
		cerr << endl << "\tWARNING: \"split-th\" > 1, but has to be in <0,1>! - set to 1\n";			
		_splitTh = 1.0f;
	}
	if(_splitTh < 0.0f) {
		cerr << endl << "\tWARNING: \"split-th\" < 0, but has to be in <0,1>! - set to 0\n";			
		_splitTh = 0.0f;
	}
	return true;
}
