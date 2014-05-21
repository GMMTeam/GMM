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

#ifndef _OL_STATS_EST_
#define _OL_STATS_EST_

#include "model/OL_GMMStats.h"
#include "model/OL_GMModel.h"
#include "tools/OL_FileList.h"

#include <vector>
#include <string.h>
#include <list>
#ifdef __GNUC__
#	include <ext/hash_map>
#	define stdext __gnu_cxx
#else
#	include <hash_map>
#endif

#ifndef _SHARE_DATA_
#	define _SHARE_DATA_
#endif

// ATTENTION: MASK HANDLING for the file-by-file accumulation uses only ONE MASK FOR ALL FILES,
//            hence one can not use different masks for different files in this regime;

// TS = Template Statistics, TV = Template Vectors, TL = Template Likelihood (variables)
template <typename TS, typename TV = float, typename TL = float>
class GMMStatsEstimator {
public:

	TL _minLogLike, _minGamma;
	float _min_var;
	unsigned int _numThreads;
	unsigned int _NAccBlocks_CUDA;
	float _memoryBuffDataGB;
	float _memoryBuffDataGB_GPU;
	int _verbosity;
	
	// signalization of numerical stability troubles when gammas are computed -> function compGammas(...); 
	// has to be set to false (reset) manually!
	bool _fNumStabilityTroubles; 	

	// used to restrict mixtures that participate in the computation of LogLike of a feature 
	// vector - these mixture indexes will be stored in 'vector< vector<unsigned int> > maskIdxs'
	// NOTE: CUDA is able to fill the mask, however CUDA can't use it
	struct logLikeMask {
		unsigned int N;   // >= NSamples
		unsigned int dim; // == numMix
		
		TL *logLikes;    // size N x 1
		TL **gammas;     // size N x dim
		std::vector< std::vector<unsigned int> > maskIdxs; // size N x Q, where Q varies in <0,M> .. M = number of mixtures
	};

	GMMStatsEstimator();
	virtual ~GMMStatsEstimator();

	// estimataion types = Classic CPU | SSE | GPU | ..
	virtual const std::string& getEstimationType() const;

	// e.g extract log-values, convert to SSE-ready model, ...; 
	// several models can be added (usefull in recognition, saves time), new group
	// of models is signalized by 'first=true' parameter - all previous models
	// will be discarded
	// note: should be 'const GMModel&', but GMModel not prepared
	virtual void insertModel (GMModel& model, bool first = true); 

	// decide which model from the set of inserted models should be used
	virtual void setModelToBeUsed (unsigned int n);


	// 'TL *logLikes, **gammas' have to be properly allocated (N >= NSamples, dim == _nummix) 
	virtual void mask2fill (logLikeMask& mask);
	
	// 'vector< vector<unsigned int> > maskIdxs' will be used in order to restrict the mixtures that
	// are used to compute the vector logLike; NOTE: see function estimateMaskIdxs() 
	virtual void mask2use (logLikeMask& mask);

	// returns the actual mask; call if the use of the logLikeMask should be disabled;
	virtual logLikeMask* withdrawMask();	

	// establish the mixtures of interest according to already 
	// filled 'TL *logLikes, *gammas' from structure 'logLikeMask' 
	static void estimateMaskIdxs(logLikeMask& mask, TL minGamma);



	// what should be accumulated - need not to be called if the accumulation process
	// should follow the allocated variables (means, vars) in GMMStats<TS>
	void setAccFlags(bool mixProbsOnly, bool meanStats, 
					 bool varStats, bool varFull, 
					 bool auxStats = false);
	
	// reset flags => the accumulation will follow the allocated variables (means, vars) in GMMStats<TS>
	void resetAccFlags();



	// return value: loglike of the vector 'vec'; if return value == _minLogLike 
	// than gamma[i] = _minGamma
	virtual TL compGammas (TV* vec, unsigned int dim, std::vector<TL>& gammas,
						   unsigned int* mixIdxs = NULL, unsigned int Nmi = 0);
	
	virtual TL computeMixLogLike (unsigned int mixnum, TV* vec, unsigned int dim);
	virtual TL computeMixLogLikeFullCov (unsigned int mixnum, TV* vec, unsigned int dim);

	virtual TL compVecLogLike (TV* vec, unsigned int dim,
							   unsigned int* mixIdxs = NULL, unsigned int Nmi = 0);

	// - accumulate GMMStats - according to pre-allocated variables in GMMStats<TS> or 
	//   according to flags set via setAccFlags();
	// - accumulate GMMStats from given vectors - multi-threaded (MT) version
	virtual void accumulateStatsMT (TV** vectors, unsigned int NSamples, unsigned int dim,
								    GMMStats<TS>& gmms);

	// accumulate GMMStats file-by-file from a list - multi-threaded (MT) version;
	// TV must be float as class Param works with floats
	virtual void accumulateStatsMT (CFileList& list, GMMStats<TS>& gmms, int prm_type, 
									unsigned int dwnsmp = 1);

	// accumulate GMMStats file-by-file-frames from a list - multi-threaded (MT) version;
	// only part of the input files will be loaded according to the given intervals, where
	// 'file2ranges' is a map: prm-filename -> intervals <x[0],x[1]) stored in a list, and 
	// 'x' is vector<unsigned int>
	// TV must be float as class Param works with floats; 
	virtual void accumulateStatsMT (stdext::hash_map <std::string, std::list < std::vector<unsigned int> > > &file2ranges, 
									GMMStats<TS>& gmms, int prm_type, unsigned int samplePeriod);

	// returns logLike for inserted vectors in 'outLogLikes', 
	// array 'outLogLikes' must be properly allocated - - multi-threaded (MT) version
	virtual void compVecsLogLikeMT (TV** vecs, unsigned int NSamples, unsigned int dim,
									TL* outLogLikes);



	// if the data provided for accumulation (ACC) or logLike estimation (LLE) are the same
	// as in the previous iteration (usefull to set when the data have to be prepared
	// before each ACC/LLE, but the data does not change between ACC/LLE;
	// Note: the flag has to be set before each ACC/LLE since at the end of ACC/LLE 
	//       it will be reseted 
	void setSameDataFlag() {_fSameDataProvided = true;}

	// get new instance of this class
	virtual GMMStatsEstimator<TS, TV, TL>* getNewInstance();

	template <typename T>
	static void getMeanGrad (GMModel& model, const GMMStats<TS>& stats, 
							 std::vector<T>& grad, double normval = 1.0);

	static void centralizeStats (GMMStats<TS>& stats, GMModel& model, 
								 bool centrMeans = true, bool centrVars = false, 
								 bool centrFVars = false);

protected:

	static void addLog (TL& d, TL dd);
	static void addLogAprox (TL& d, TL dd);
	
	// called by programmer - the '_flagsSet' is not set!
	void setAccFlags(GMMStats<TS>& stats);
	void checkProperAllocation(GMMStats<TS>& stats);

	// returns unnormalized gammas (= mixture log likes) & vector (overall) logLikelihood
	virtual TL getLikeAndGammas (TV* vec, unsigned int dim, std::vector<TL>& gammas,
								 unsigned int* mixIdxs, unsigned int Nmi);

	virtual void compVecsLogLike (TV** vecs, unsigned int NSamples, unsigned int dim,
								  TL* outLogLikes, unsigned int offset = 0);

	virtual void accumulateStats (TV** vecs, unsigned int NSamples, unsigned int dim,
								  GMMStats<TS>& gmms, unsigned int offset = 0);

	// accumulate one vector
	virtual void accumulateStats (TV* vec, unsigned int dim, GMMStats<TS>& gmms, std::vector<TL>& gammas,
								  unsigned int* mixIdxs = NULL, unsigned int Nmi = 0);

	// accumulate one vector
	virtual void accumulateAuxStats (TV* vec1, TV* vec2, TV* vec3, unsigned int dim, 
									 GMMStats<TS>& gmms, std::vector<TL>& gammas,
								     unsigned int* mixIdxs = NULL, unsigned int Nmi = 0);

	// handles one thread
	void threadFuncAcc (TV** vectors, unsigned int NSamples, unsigned int dim,
						GMMStats<TS>* gmms, unsigned int offset = 0);
	void threadFuncLL (TV** vectors, unsigned int NSamples, unsigned int dim,
					   TL* outLogLikes, unsigned int offset = 0);


	// shares pointers/variables & allocates private memory
	virtual void shareData (GMMStatsEstimator<TS, TV, TL>& est);	
	
	// initialize internal data/buffers - assuming that basic variables 
	// as _dim, _nummix,.. are already initialized
	virtual void initializeInternalData();

	// clear internal variables used in case of multiple GMMs
	virtual void clearGroups();
	
	static void getCovInverse (float* icov, float* cov, unsigned int dim, unsigned int dim_aligned);
		
	static TL getLogDet (float* cov, unsigned int dim);

	std::string _estimType; // SSE | CUDA | classic CPU
	bool _fSameDataProvided;

	// mask handling
	logLikeMask *_mask;
	bool _useMask, _fillMask;

	// GMM
	GMModel *_model;
	bool _modelIn;

	// multiple GMMs
	std::vector <GMModel *> _modelsGroup;
	
	float *_icov;

	TL *_logGconst, *_logWeights;
	std::vector <TL *> _logGconstGroup, _logWeightsGroup;	
	unsigned int _nummixAlloc, _nummix, _dim, _dimVar;
	
	// thread management	
	bool _threadsFailed, *_sharedThrdFailedStatus; // trace thread error status
	bool *_sharedfNumStabilityTroubles; // share between threads

	// accumulation properties
	bool _flagsSet, _fMixProbOnly, _fMean, _fVar, _fFullVar, _fAux;

	// constants
	static const unsigned int MIN_FRAMES_FOR_THREAD;
	static const double MINUS_LOG_2PI;
	static const float LPDIF; // threshold in log domain -> exp(-_LPDIF) ~= 0	
	//static const float MIN_VAR;

private:	
	// class specific buffers
	std::vector<TL> _gammas;
	
	// memory management when threads are in use
	unsigned int _referenceCounter, *_sharedReferenecCounter;

	// prohibited
	GMMStatsEstimator (const GMMStatsEstimator<TS, TV, TL>&) {printf("Copy ctor GMMStatsEstimator prohibited!\n");};
	GMMStatsEstimator<TS, TV, TL>& operator= (const GMMStatsEstimator<TS, TV, TL>&);
};

// definition of constants
template <typename TS, typename TV, typename TL>
const double GMMStatsEstimator<TS, TV, TL>::MINUS_LOG_2PI = -1.837877066409345;

template <typename TS, typename TV, typename TL>
const float GMMStatsEstimator<TS, TV, TL>::LPDIF = 15.0f;

template <typename TS, typename TV, typename TL>
const unsigned int GMMStatsEstimator<TS, TV, TL>::MIN_FRAMES_FOR_THREAD = 100;

//template <typename TS, typename TV, typename TL>
//const float GMMStatsEstimator<TS, TV, TL>::MIN_VAR = 1e-6f;


#include "trainer/OL_GMMStatsEstimator.cpp"

#endif

