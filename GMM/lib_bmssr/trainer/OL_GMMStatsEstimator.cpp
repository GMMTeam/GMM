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

#ifdef _OL_STATS_EST_

#ifdef __GNUC__
#	include "general/my_inttypes.h"
#else
#	define isnan(x) _isnan(x) // kuoli linuchu
#	define finite(x) _finite(x) // kuoli linuchu
#endif

#include "param/OL_Param.h"

#ifdef _MKL
#	include <mkl.h>
#elif _ACML
#	include <acml.h>
#else
#	error "ACML or MKL have to be available"
#endif


#include <new>
#include <sstream>
#include <iostream>
#include <math.h>
//#include <float.h>
#include <boost/thread.hpp>

#ifdef __GNUC__
namespace __gnu_cxx
{
        template<> struct hash< std::string >
        {
                size_t operator()( const std::string& x ) const
                {
                        return hash< const char* >()( x.c_str() );
                }
        };
}
#endif


namespace {
	boost::mutex the_except_mutex, the_dtor_mutex, the_sharedmem_mutex;
	typedef std::vector<unsigned int> element_range;
	typedef stdext::hash_map <std::string, std::list <element_range> > file_2_ranges;
}


template <typename TS, typename TV, typename TL>
GMMStatsEstimator<TS, TV, TL>::GMMStatsEstimator() 
: _minLogLike(-1e+20f),
_minGamma(1e-4f),
_min_var(1e-6f),
_numThreads(1),
_NAccBlocks_CUDA(8),
_verbosity(1),
_fNumStabilityTroubles(false),
_fSameDataProvided(false),
_mask(NULL),
_useMask(false), 
_fillMask(false),
_model(NULL), 
_icov(NULL),
_logGconst(NULL), 
_logWeights(NULL),
_nummix(0), 
_nummixAlloc(0),
_dim(0),
_dimVar(0),
_modelIn(false),
_threadsFailed(false),
_sharedThrdFailedStatus(&_threadsFailed),
_sharedfNumStabilityTroubles(&_fNumStabilityTroubles),
_flagsSet(false),
_fMixProbOnly(false), 
_fMean(false), 
_fVar(false), 
_fFullVar(false),
_fAux(false), 
_referenceCounter(0),
_sharedReferenecCounter(&_referenceCounter)
{
	_estimType = std::string("Classic CPU");
}



template <typename TS, typename TV, typename TL>
GMMStatsEstimator<TS, TV, TL>::~GMMStatsEstimator() {

#ifdef _SHARE_DATA_
	boost::mutex::scoped_lock lock(the_dtor_mutex);
	if(*_sharedReferenecCounter == 0) {
		clearGroups();
		if(_logGconst != NULL)
			delete [] _logGconst;
		if(_logWeights != NULL)
			delete [] _logWeights;
		if(_icov != NULL)
			delete [] _icov;
	}
	else --(*_sharedReferenecCounter);
#else
	clearGroups();
	if(_logGconst != NULL)
		delete [] _logGconst;
	if(_logWeights != NULL)
		delete [] _logWeights;
	if(_icov != NULL)
		delete [] _icov;
#endif
}



template <typename TS, typename TV, typename TL>
const std::string& GMMStatsEstimator<TS, TV, TL>::getEstimationType() const {
	return _estimType;
}



template <typename TS, typename TV, typename TL>
void GMMStatsEstimator<TS, TV, TL>::getCovInverse (float* icov, float* cov, unsigned int dim, unsigned int dim_aligned) {

#ifdef _MKL
	MKL_INT info;
	MKL_INT mkl_dim = (MKL_INT) dim;
	MKL_INT *N = &mkl_dim;
	MKL_INT mkl_dim_al = (MKL_INT) dim_aligned;
	MKL_INT *LDA = &mkl_dim_al;
	char uplo = 'L';
	char *L = &uplo;
#elif _ACML
	int info;
	int N = dim;
	int LDA = dim_aligned;
	char L = 'L';
#endif

	memset(icov, 0, sizeof(float) * dim_aligned * dim_aligned);
	for (unsigned int i = 0; i < dim; i++) {
		memcpy(icov + i*dim_aligned, cov + i*dim, sizeof(float) * dim);
	}

	// http://www.math.utah.edu/software/lapack/lapack-s/spotrf.html
	spotrf(L, N, icov, LDA, &info);
	if(info != 0)
		throw std::runtime_error("getCovInverse(): cholesky factorization failed!");

	// http://www.math.utah.edu/software/lapack/lapack-s/spotri.html
	spotri(L, N, icov, LDA, &info);	
	if(info != 0)
		throw std::runtime_error("getCovInverse(): inversion failed!");

	for (unsigned int i = 0; i < dim; i++) {
		for (unsigned int ii = 0; ii < i; ii++) {
			icov[i * dim_aligned + ii] = icov[ii * dim_aligned + i];
		}
	}
}



template <typename TS, typename TV, typename TL>
TL GMMStatsEstimator<TS, TV, TL>::getLogDet (float* cov, unsigned int dim) {
	
	float* chol = new float [dim * dim];
	memcpy(chol, cov, sizeof(float) * dim * dim);

#ifdef _MKL
	MKL_INT info;
	MKL_INT mkl_dim = (MKL_INT) dim;
	MKL_INT *N = &mkl_dim;
	char uplo = 'L'; 
	char *L = &uplo;
#elif _ACML
	int info;
	int N = dim;
	char L = 'L'; 
#endif

	// http://www.math.utah.edu/software/lapack/lapack-s/spotrf.html
	spotrf(L, N, chol, N, &info);
	if(info != 0)
		throw std::runtime_error("getLogDet(): cholesky factorization failed!");

	TL det = 0;
	for (unsigned int i = 0; i < dim; i++) {
		if (chol[i * dim + i] < 0.0f)
			throw std::runtime_error("getLogDet(): cholesky factorization failed!");
		det += (TL) log(chol[i * dim + i]);
	}

	delete [] chol;

	return 2*det;
}



template <typename TS, typename TV, typename TL>
void GMMStatsEstimator<TS, TV, TL>::insertModel (GMModel& model, bool first) {

	if(first)
		clearGroups();

	unsigned int nummixAlloc = model.GetNumberOfMixtures();
	unsigned int dim = model.GetDimension();
	unsigned int dimVar;
	
	if (model.GetFullCovStatus())
		dimVar = dim * dim;
	else dimVar = dim;
	
	if(!first && dim != _dim)
		throw std::runtime_error("Models (GMMs) have to have same dimensions");		

	// place Gaussians with 0 weight at the end
	unsigned int nZeroG = model.RearrangeMixtures();		
	unsigned int nummix = nummixAlloc - nZeroG;	

	if(nummixAlloc > _nummixAlloc || !first) {
		if(first) {
			if(_logGconst != NULL)
				delete [] _logGconst;
			if(_logWeights != NULL)
				delete [] _logWeights;
			_gammas.resize(nummixAlloc);
		}

		_logGconst = new TL[nummixAlloc];
		_logWeights = new TL[nummixAlloc];
		_icov = new float[nummixAlloc * dimVar];
	}

	for(unsigned int m = 0; m < nummix; m++) {		
		TL w = (TL) model.GetMixtureWeight(m);
		_logWeights[m] = (w > 0) ? log(w) : _minLogLike;				
	}

	if (!model.GetFullCovStatus()) {
		for(unsigned int m = 0; m < nummix; m++) {		
			float *var = model.GetMixtureVarDiag(m);

			_logGconst[m] = 0;
			for(unsigned int n = 0; n < dimVar; n++) {
				if(var[n] < _min_var) {
					//std::cout << "Warning: insertModel(): model contains very low variances [mix = " << m << "]!" << std::endl;
					// _logGconst[m] += (var[n] > 0) ? log(var[n]) : _minLogLike;
					std::stringstream error_report;
					error_report << "insertModel(): model contains very low variances [" << var[n] << "]!";
					throw std::runtime_error(error_report.str().c_str());
				}
				_icov[m * dimVar + n] = 1.0f / var[n];
				_logGconst[m] += (TL) log(var[n]);
			}
			_logGconst[m] = (TL) (MINUS_LOG_2PI * dim/2 - _logGconst[m]/2);
		}
	}
	else {
		for(unsigned int m = 0; m < nummix; m++) {		
			float *var = model.GetMixtureVar(m);
			getCovInverse (_icov + m * dimVar, var, dim, dim);
			
			_logGconst[m] = getLogDet (var, dim);
			_logGconst[m] = (TL) (MINUS_LOG_2PI * dim/2 - _logGconst[m]/2);			
		}
	}
	
	_model = &model;
	_nummix = nummix;
	_nummixAlloc = nummixAlloc;
	_dim = dim;
	_dimVar = dimVar;

	_modelsGroup.push_back(_model);
	_logGconstGroup.push_back(_logGconst);
	_logWeightsGroup.push_back(_logWeights);	

	_modelIn = true;
}



template <typename TS, typename TV, typename TL>
void GMMStatsEstimator<TS, TV, TL>::setModelToBeUsed (unsigned int n) {
	_model = _modelsGroup.at(n);
	_logGconst = _logGconstGroup.at(n);
	_logWeights = _logWeightsGroup.at(n);
	_nummixAlloc = _model->GetNumberOfMixtures();
	_nummix = _model->GetNumberOfNZWMixtures();
	_gammas.resize(_nummixAlloc);
}



template <typename TS, typename TV, typename TL>
void GMMStatsEstimator<TS, TV, TL>::clearGroups() {
			
	if (_modelsGroup.size() > 0) {
		_model = _modelsGroup.at(0);
		_logGconst = _logGconstGroup.at(0);
		_logWeights = _logWeightsGroup.at(0);
	}

	// do not delete the memory of the first model - may be needed for next group
	for(unsigned int it = 1; it < _modelsGroup.size(); it++) {
		delete [] _logGconstGroup[it];
		delete [] _logWeightsGroup[it];
	}

	_logGconstGroup.clear();
	_logWeightsGroup.clear();
	_modelsGroup.clear();
}



template <typename TS, typename TV, typename TL>
inline void GMMStatsEstimator<TS, TV, TL>::mask2fill (logLikeMask& mask) {
	_fillMask = true;
	_useMask = false;
	_mask = &mask;
}



template <typename TS, typename TV, typename TL>
inline void GMMStatsEstimator<TS, TV, TL>::mask2use (logLikeMask& mask) {
	_useMask = true;
	_fillMask = false;
	_mask = &mask;
}



template <typename TS, typename TV, typename TL>
inline typename GMMStatsEstimator<TS, TV, TL>::logLikeMask* GMMStatsEstimator<TS, TV, TL>::withdrawMask() {
	_fillMask = false;
	_useMask = false;
	
	logLikeMask* m = _mask;
	_mask = NULL;

	return m;
}



template <typename TS, typename TV, typename TL>
void GMMStatsEstimator<TS, TV, TL>::estimateMaskIdxs(logLikeMask& mask, TL minGamma) {

	mask.maskIdxs.resize(mask.N);
	for(unsigned int i = 0; i < mask.N; i++) {
		mask.maskIdxs[i].reserve(mask.dim);
		for(unsigned int j = 0; j < mask.dim; j++) {
			if(mask.gammas[i][j] > minGamma)
				mask.maskIdxs[i].push_back(j);
		}
	}
}



template <typename TS, typename TV, typename TL>
inline void GMMStatsEstimator<TS, TV, TL>::setAccFlags(bool mixProbsOnly, bool meanStats, 
													   bool varStats, bool varFull, bool auxStats) 
{
	_flagsSet = true;
	_fMixProbOnly = mixProbsOnly; 
	_fMean = meanStats;	
	_fVar = varStats;
	_fAux = auxStats;


	//if(_fVar)
	//	_fFullVar = varFull;
	//else _fFullVar = false;
	_fFullVar = varFull;
	if(_fFullVar)
		_fVar = true;
}



template <typename TS, typename TV, typename TL>
inline void GMMStatsEstimator<TS, TV, TL>::setAccFlags(GMMStats<TS>& gmms) {
	_fMixProbOnly = !gmms._allM && !gmms._allV;
	_fMean = gmms._allM;
	_fVar = gmms._allV;
	_fAux = gmms._allA;
	
	//if(_fVar)
	//	_fFullVar = gmms._allFV;
	//else _fFullVar = false;

	_fFullVar = gmms._allFV;
	if(_fFullVar)
		_fVar = true;
}



template <typename TS, typename TV, typename TL>
inline void GMMStatsEstimator<TS, TV, TL>::resetAccFlags() {
	_flagsSet = false;
}



template <typename TS, typename TV, typename TL>
TL GMMStatsEstimator<TS, TV, TL>::getLikeAndGammas (TV* vec, unsigned int dim, std::vector<TL>& gammas,
													unsigned int* mixIdxs, unsigned int Nmi) 
{
#ifndef _EXCLUDE_SAFETY_CONDS_
	if(!_modelIn)
		throw std::logic_error("getLikeAndGammas(): None model inserted! \n\t");
	if(gammas.size() < _nummix)
		gammas.resize(_nummix);
#endif

	if(dim != _dim)
		throw std::logic_error("getLikeAndGammas(): Inconsistent model & feature vector - different dimensions! \n\t");
	
	TL maxLL, sumLL;
	maxLL = sumLL = _minLogLike;	
	
	unsigned int index, M = (mixIdxs != NULL) ? Nmi : _nummix;
	if(_dim != _dimVar) {
		for(unsigned int i = 0; i < M; i++) {
			index = (mixIdxs != NULL) ? mixIdxs[i] : i; 

#ifndef _EXCLUDE_SAFETY_CONDS_
			if (index > _nummix) 
				throw std::out_of_range("getLikeAndGammas(): Specified index out of bounds! \n\t");		
#endif
			if (_logWeights[index] == _minLogLike) {
				gammas[index] = _minLogLike;
				continue;
			}

			gammas[index] = computeMixLogLikeFullCov(index, vec, dim) + _logWeights[index];

			if(maxLL < gammas[index]) 
				maxLL = gammas[index];

			addLog(sumLL, gammas[index]);
		}
	}
	else {
		for(unsigned int i = 0; i < M; i++) {
			index = (mixIdxs != NULL) ? mixIdxs[i] : i; 

#ifndef _EXCLUDE_SAFETY_CONDS_
			if (index > _nummix) 
				throw std::out_of_range("getLikeAndGammas(): Specified index out of bounds! \n\t");		
#endif
			if (_logWeights[index] == _minLogLike) {
				gammas[index] = _minLogLike;
				continue;
			}

			gammas[index] = computeMixLogLike(index, vec, dim) + _logWeights[index];

			if(maxLL < gammas[index]) 
				maxLL = gammas[index];

			addLog(sumLL, gammas[index]);
		}
	}

	if(maxLL == _minLogLike) {
		*_sharedfNumStabilityTroubles = true; // => SpeedUp2 should be disabled
		return _minLogLike;
	}

	return sumLL;
}



template <typename TS, typename TV, typename TL>
TL GMMStatsEstimator<TS, TV, TL>::compGammas (TV* vec, unsigned int dim, std::vector<TL>& gammas,
											  unsigned int* mixIdxs, unsigned int Nmi) 
{
#ifndef _EXCLUDE_SAFETY_CONDS_
	if(!_modelIn)
		throw std::logic_error("compGammas(): None model inserted! \n\t");
	if(gammas.size() < _nummix)
		gammas.resize(_nummix);
#endif

	//if(_nummix < 2) {
	//	gammas[0] = static_cast<TL> (1.0);
	//	return _minLogLike;
	//}

	TL llike = getLikeAndGammas(vec, dim, gammas, mixIdxs, Nmi);

	if(_nummix == 1) {
		gammas[0] = 1.0f;
		return llike;
	}

	unsigned int index, M = (mixIdxs != NULL) ? Nmi : _nummix;
	if (llike == _minLogLike) {
		std::cerr << "Warning: compGammas(): Numerical stability trouble (outliers?)!" << std::endl;
		
		for(unsigned int i = 0; i < M; i++) {
			index = (mixIdxs != NULL) ? mixIdxs[i] : i;
			gammas[index] = _minGamma;
		}
		// ?? following lines have to be reconsidered - what about outliers ?? 
		//for(unsigned int i = 0; i < M; i++) {
		//	index = (mixIdxs != NULL) ? mixIdxs[i] : i;
		//	gammas[index] = (TL) (1.0f/M);
		//}				
	}
	else {
		for(unsigned int i = 0; i < M; i++) {
			index = (mixIdxs != NULL) ? mixIdxs[i] : i;
			gammas[index] = (TL) (gammas[index] > _minLogLike) * exp(gammas[index] - llike);
		} 
	}
	return llike;
}



template <typename TS, typename TV, typename TL>
TL GMMStatsEstimator<TS, TV, TL>::computeMixLogLike (unsigned int mixnum, TV* vec, unsigned int dim) {

#ifndef _EXCLUDE_SAFETY_CONDS_
	if (mixnum > _nummix) 
		throw std::out_of_range("computeMixLogLike(): Specified index out of bounds! \n\t");		

	if(!_modelIn)
		throw std::logic_error("computeMixLogLike(): None model inserted! \n\t");
#endif

	if(dim != _dim)
		throw std::logic_error("computeMixLogLike(): Inconsistent model & feature vector - different dimensions! \n\t");

	float *mean = _model->GetMixtureMean(mixnum);	

	TL exponent = 0.0;
	for(unsigned int i = 0; i < _dim; i++)
		exponent += ((TL) (vec[i] - mean[i])) * ((TL) (vec[i] - mean[i])) * (TL) (_icov[mixnum * dim + i]);
	
	TL ll = _logGconst[mixnum] - 0.5f * exponent;

	if(!finite(ll) || isnan(ll))
		return _minLogLike;

	return ll;
}



template <typename TS, typename TV, typename TL>
TL GMMStatsEstimator<TS, TV, TL>::computeMixLogLikeFullCov (unsigned int mixnum, TV* vec, unsigned int dim) {

#ifndef _EXCLUDE_SAFETY_CONDS_
	if (mixnum > _nummix) 
		throw std::out_of_range("computeMixLogLikeFullCov(): Specified index out of bounds! \n\t");		

	if(!_modelIn)
		throw std::logic_error("computeMixLogLikeFullCov(): None model inserted! \n\t");
#endif

	if(dim != _dim)
		throw std::logic_error("computeMixLogLikeFullCov(): Inconsistent model & feature vector - different dimensions! \n\t");

	float *mean = _model->GetMixtureMean(mixnum);	

	TL exponent = 0.0;
	float *icov = _icov + mixnum * _dimVar;
	for(unsigned int i = 0; i < _dim; i++) {
		for(unsigned int ii = 0; ii < _dim; ii++) {
			exponent += ((TL) (vec[i] - mean[i])) * ((TL) (vec[ii] - mean[ii])) * ((TL) icov[i * dim + ii]);
		}
	}
	
	TL ll = _logGconst[mixnum] - 0.5f * exponent;

	if(!finite(ll) || isnan(ll))
		return _minLogLike;

	return ll;
}



template <typename TS, typename TV, typename TL>
TL GMMStatsEstimator<TS, TV, TL>::compVecLogLike (TV* vec, unsigned int dim,
												  unsigned int* mixIdxs, unsigned int Nmi) 
{
#ifndef _EXCLUDE_SAFETY_CONDS_
	if(!_modelIn)
		throw std::logic_error("compVecLogLike(): None model inserted! \n\t");
#endif

	return getLikeAndGammas(vec, dim, _gammas, mixIdxs, Nmi);
}



template <typename TS, typename TV, typename TL>
void GMMStatsEstimator<TS, TV, TL>::accumulateStatsMT (CFileList &list, GMMStats<TS>& gmms, int prm_type, 
													   unsigned int dwnsmp) 
{
	static float giga = 1024.0f * 1024.0f * 1024.0f;
	unsigned int NSamples = 0, prmdim = 0;
	int iRet;
	char *infile;
	
	if(_verbosity)
		std::cout << std::endl << " \t\t[ACC: file-by-file -- #files = " << list.ListLength() << "]" << std::endl;

	//_fSameDataProvided = false; // not necessary to be set here

	Param param, paramload;
	list.Rewind();
	while(list.GetItemName(&infile)) {		
		
		switch (prm_type)
		{
		case(SVES_PRM_IN): 
			iRet = paramload.Load (infile, dwnsmp);
			break;
		case(HTK_PRM_IN): 
			iRet = paramload.LoadHTK (infile, dwnsmp);
			break;
		case(RAW_PRM_IN): 
			iRet = paramload.LoadRaw (infile, dwnsmp);
			break;
		default:
			throw std::runtime_error("accumulateStatsMT(): Unknown input data type specified, use --help and option --in-type");
		}		

		if(iRet != 0) {
			std::cerr << std::endl << " WARNING: Param file [" << infile << "] not found -> skipped!" << std::endl;
			continue;
		}
		
		NSamples += (unsigned int) paramload.GetNumberOfVectors();
		prmdim = (unsigned int) paramload.GetVectorDim();

		if(NSamples < 1) {
			std::cerr << std::endl << " WARNING: Param file [" << infile << "] empty" << std::endl;
			continue;
		}
		param.Add(paramload, true);
		paramload.CleanMemory(true);

		if (_memoryBuffDataGB > NSamples * prmdim * sizeof(float) / giga)
			continue;

		if(_verbosity)
			std::cout << ".";

		accumulateStatsMT(*param.GetVectors(), NSamples, prmdim, gmms);

		param.CleanMemory();
		NSamples = 0;
	}
	if(NSamples > 0) {
		if(_verbosity)
			std::cout << ".";

		accumulateStatsMT(*param.GetVectors(), NSamples, prmdim, gmms);
	}
	if(_verbosity)
		std::cout << "\n\t\t";
}



template <typename TS, typename TV, typename TL>
void GMMStatsEstimator<TS, TV, TL>::accumulateStatsMT (file_2_ranges &file2ranges, GMMStats<TS>& gmms,
													   int prm_type, unsigned int samplePeriod) 
{
	static float giga = 1024.0f * 1024.0f * 1024.0f;
	unsigned int NSamples = 0, prmdim = 0;
	int iRet;

	if(_verbosity)
		std::cout << std::endl << " \t\t[ACC: file-by-file-frames -- #files = " << file2ranges.size() << "]" << std::endl;

	//_fSameDataProvided = false; // not necessary to be set here
	
	Param param, paramload;

	if(prm_type != HTK_PRM_IN) {
		param.SetSamplePeriod(samplePeriod);
		paramload.SetSamplePeriod(samplePeriod);
	}

	for (file_2_ranges::iterator it1 = file2ranges.begin(); it1 != file2ranges.end(); it1++) 
	{	
		switch (prm_type)
		{
		case(SVES_PRM_IN): 
			iRet = paramload.Load ((*it1).first.c_str(), file2ranges[(*it1).first]);
			break;
		case(HTK_PRM_IN): 
			iRet = paramload.LoadHTK ((*it1).first.c_str(), file2ranges[(*it1).first]);
			break;
		case(RAW_PRM_IN): 
			iRet = paramload.LoadRaw ((*it1).first.c_str(), file2ranges[(*it1).first]);
			break;
		default:
			throw std::runtime_error("accumulateStatsMT(): Unknown input data type specified, use --help and option --in-type");
		}

		if(iRet != 0) {			
			std::cerr << std::endl << " WARNING: Param file [" << (*it1).first << "] not found -> skipped!" << std::endl;
			continue;
		}
		
		NSamples += (unsigned int) paramload.GetNumberOfVectors();
		prmdim = (unsigned int) paramload.GetVectorDim();

		if(NSamples < 1) {
			std::cerr << std::endl << " WARNING: Param file [" << (*it1).first << "] empty" << std::endl;
			continue;
		}
		param.Add(paramload, true);
		paramload.CleanMemory(true);

		if (_memoryBuffDataGB > NSamples * prmdim * sizeof(float) / giga)
			continue;

		if(_verbosity)
			std::cout << ".";

		accumulateStatsMT(*param.GetVectors(), NSamples, prmdim, gmms);

		param.CleanMemory();
		NSamples = 0;
	}
	if(NSamples > 0) {
		if(_verbosity)
			std::cout << ".";

		accumulateStatsMT(*param.GetVectors(), NSamples, prmdim, gmms);
	}
	if(_verbosity)
		std::cout << "\n\t\t";
}



template <typename TS, typename TV, typename TL>
void GMMStatsEstimator<TS, TV, TL>::checkProperAllocation(GMMStats<TS>& gmms)
{
	if(!_fMixProbOnly && !_fMean && !_fVar)
		throw std::logic_error("accumulateStats(): Nothing to do! \n\t");

	if(_fMixProbOnly && !gmms._allocated)
		throw std::logic_error("accumulateStats(): Incorrect GMMStats allocation! \n\t");

	if(!_fMixProbOnly && _fMean && !gmms._allM)
		throw std::logic_error("accumulateStats(): Incorrect GMMStats allocation! \n\t");

	if(!_fMixProbOnly && _fVar && !gmms._allV)
		throw std::logic_error("accumulateStats(): Incorrect GMMStats allocation! \n\t");

	if(!_fMixProbOnly && _fFullVar && !gmms._allFV)
		throw std::logic_error("accumulateStats(): Incorrect GMMStats allocation! \n\t");

	if(!_fMixProbOnly && _fAux && !gmms._allA)
		throw std::logic_error("accumulateStats(): Incorrect GMMStats allocation! \n\t");
}



template <typename TS, typename TV, typename TL>
void GMMStatsEstimator<TS, TV, TL>::compVecsLogLikeMT (TV** vecs, unsigned int NSamples, unsigned int dim,
													   TL* outLogLikes) 
{
#ifndef _EXCLUDE_SAFETY_CONDS_
	if(!_modelIn)
		throw std::logic_error("compVecsLogLikeMT(): None model inserted! \n\t");		
	if(_mask != NULL && (_mask->N < NSamples || _mask->dim < _nummix))
		throw std::logic_error("compVecsLogLikeMT(): logLikeMask improperly allocated! \n\t");
#endif

	// prepare threads
	_threadsFailed = false;

	unsigned int thrdNum = NSamples / MIN_FRAMES_FOR_THREAD;
	if(thrdNum > _numThreads)
		thrdNum = _numThreads;

	// if only 1 thread requested
	if(thrdNum < 2) {
		compVecsLogLike(vecs, NSamples, dim, outLogLikes);
		_fSameDataProvided = false;
		return;
	}
	
	boost::thread **vlakna = new boost::thread* [thrdNum];

	unsigned int shift = 0, Npart = NSamples / thrdNum;	
	vlakna[0] = new boost::thread(boost::bind(&GMMStatsEstimator<TS, TV, TL>::threadFuncLL, this, vecs, Npart, dim, outLogLikes, 0));

	for(unsigned int i = 1; i < thrdNum; i++)
	{			
		shift += Npart;

		// in the last step take all the remaining vectors
		if(i+1 == thrdNum)							
			Npart = NSamples - shift;
		
		vlakna[i] = NULL;
		try {		
			vlakna[i] = new boost::thread(boost::bind(&GMMStatsEstimator<TS, TV, TL>::threadFuncLL, this, vecs, Npart, dim, outLogLikes, shift));
		} 
		catch (...) {						
			delete vlakna[i];	
			
			boost::mutex::scoped_lock lock(the_except_mutex);
			if(!_threadsFailed)
				_threadsFailed = true;

			thrdNum = i;

			break;
		}
	}

	// wait for all the threads
	for(unsigned int i = 0; i < thrdNum; i++) {
		vlakna[i]->join();							
		delete vlakna[i];		
	}	

	delete [] vlakna;	

	bool fail = _threadsFailed;
	_threadsFailed = false; // reset 
	_fSameDataProvided = false; // reset

	if(fail)
		throw std::runtime_error("compVecsLogLikeMT(): LogLike estimation failed!");
}



template <typename TS, typename TV, typename TL>
void GMMStatsEstimator<TS, TV, TL>::accumulateStatsMT (TV** vectors, unsigned int NSamples, unsigned int dim,
													   GMMStats<TS>& gmms)
{	
	// initialization & allocation check
	if(!_flagsSet)
		setAccFlags(gmms);

	// prepare threads
	_threadsFailed = false;

	unsigned int thrdNum = NSamples / MIN_FRAMES_FOR_THREAD;
	if(thrdNum > _numThreads)
		thrdNum = _numThreads;

	// if only 1 thread requested
	if(thrdNum < 2) {
		accumulateStats(vectors, NSamples, dim, gmms);
		_fSameDataProvided = false;
		return;
	}
		
	// create threads & temporary statistics
	GMMStats<TS> **stats = new GMMStats<TS>* [thrdNum-1];
	boost::thread **vlakna = new boost::thread* [thrdNum]; //TMP 6.3.2014

	unsigned int Npart = NSamples / thrdNum;	
	vlakna[0] = new boost::thread(boost::bind(&GMMStatsEstimator<TS, TV, TL>::threadFuncAcc, this, vectors, Npart, dim, &gmms, 0)); //TMP 6.3.2014
	//threadFuncAcc(vectors, Npart, dim, &gmms, 0);

	unsigned int shift = 0;
	for(unsigned int i = 1; i < thrdNum; i++)
	{			
		shift += Npart;

		// in the last step take all the remaining vectors
		if(i+1 == thrdNum)							
			Npart = NSamples - shift;
		
		stats[i-1] = NULL;
		vlakna[i] = NULL; //TMP 6.3.2014
		try {
			if(gmms.isAllocated())
				stats[i-1] = new GMMStats<TS>(gmms.getDim(), gmms.getMixNum(), _fMean, _fVar, _fFullVar, _fAux);			
			else
				stats[i-1] = new GMMStats<TS>;
			vlakna[i] = new boost::thread(boost::bind(&GMMStatsEstimator<TS, TV, TL>::threadFuncAcc, this, vectors, Npart, dim, stats[i-1], shift)); //TMP 6.3.2014
			//threadFuncAcc(vectors, Npart, dim, stats[i-1], shift);
		} 
		catch (...) {			
			delete stats[i-1];
			delete vlakna[i];	//TMP 6.3.2014
			
			boost::mutex::scoped_lock lock(the_except_mutex);
			if(!_threadsFailed)
				_threadsFailed = true;

			_threadsFailed = true;
			thrdNum = i;

			break;
		}
	}

	vlakna[0]->join(); //TMP 6.3.2014
	delete vlakna[0]; //TMP 6.3.2014

	// wait for all the threads
	for(unsigned int i = 1; i < thrdNum; i++) {
		vlakna[i]->join(); //TMP 6.3.2014
		
		// join results on success
		if(!_threadsFailed)
			gmms += *stats[i-1];
		
		delete stats[i-1];
		delete vlakna[i]; //TMP 6.3.2014
	}	

	delete [] vlakna; //TMP 6.3.2014
	delete [] stats;

	bool fail = _threadsFailed;
	_threadsFailed = false; // reset 
	_fSameDataProvided = false; // reset

	if(fail)
		throw std::runtime_error("accumulateStatsMT(): accumulation failed!");
}



template <typename TS, typename TV, typename TL>
void GMMStatsEstimator<TS, TV, TL>::threadFuncAcc (TV** vectors, unsigned int NSamples, unsigned int dim,
												   GMMStats<TS>* gmms, unsigned int offset)
{
	try {		
		GMMStatsEstimator<TS, TV, TL>* estThrd = getNewInstance();
		
		shareData(*estThrd);
		estThrd->initializeInternalData();
		estThrd->accumulateStats(vectors, NSamples, dim, *gmms, offset);
		
		delete estThrd;
	} 
	catch (std::exception &e) {		
		boost::mutex::scoped_lock lock(the_except_mutex);
		std::cout << e.what() << std::endl;
		if(!_threadsFailed)
			_threadsFailed = true;
	}
	catch (...) {
		boost::mutex::scoped_lock lock(the_except_mutex);
		std::cout << "Unknown exception caught when running threads" << std::endl;
		if(!_threadsFailed)
			_threadsFailed = true;
	}
}



template <typename TS, typename TV, typename TL>
void GMMStatsEstimator<TS, TV, TL>::threadFuncLL (TV** vectors, unsigned int NSamples, unsigned int dim,
												  TL* outLogLikes, unsigned int offset)
{
	try {		
		GMMStatsEstimator<TS, TV, TL>* estThrd = getNewInstance();
		
		shareData(*estThrd);
		estThrd->initializeInternalData();
		estThrd->compVecsLogLike(vectors, NSamples, dim, outLogLikes, offset);

		delete estThrd;
	} 
	catch (std::exception &e) {
		boost::mutex::scoped_lock lock(the_except_mutex);
		std::cout << e.what() << std::endl;
		if(!_threadsFailed)
			_threadsFailed = true;
	}
	catch (...) {
		boost::mutex::scoped_lock lock(the_except_mutex);
		std::cout << "Unknown exception caught when running threads" << std::endl;
		if(!_threadsFailed)
			_threadsFailed = true;
	}
}



template <typename TS, typename TV, typename TL>
GMMStatsEstimator<TS, TV, TL>* GMMStatsEstimator<TS, TV, TL>::getNewInstance() {
	return new GMMStatsEstimator<TS, TV, TL>;
}



template <typename TS, typename TV, typename TL>
void GMMStatsEstimator<TS, TV, TL>::initializeInternalData() {

//#ifndef _EXCLUDE_SAFETY_CONDS_
//	if(!_modelIn)
//		throw std::logic_error("accumulateStats(): None model inserted! \n\t");		
//#endif

	// class-unique data - memory allocation
	_gammas.resize(_nummixAlloc);
}



template <typename TS, typename TV, typename TL>
void GMMStatsEstimator<TS, TV, TL>::shareData (GMMStatsEstimator<TS, TV, TL>& est) 
{
	// shared data
	est._minLogLike = _minLogLike;
	est._minGamma = _minGamma;
	est._verbosity = _verbosity;	
	est._mask = _mask;
	est._useMask = _useMask;
	est._fillMask = _fillMask;
	est._model = _model;	
	est._modelIn = _modelIn;

   // zapnut _SHARE_DATA_ a staci alokovat na _nummix miesto _nummixAlloc
#ifdef _SHARE_DATA_
	est._icov = _icov;
	est._logGconst = _logGconst;
	est._logWeights = _logWeights;	
#else
	est._icov = new float[_nummix * _dimVar];
	est._logGconst = new TL[_nummix];
	est._logWeights = new TL[_nummix];
	memcpy(est._icov, _icov, sizeof(TL)*_nummix * _dimVar);
	memcpy(est._logGconst, _logGconst, sizeof(TL)*_nummix);
	memcpy(est._logWeights, _logWeights, sizeof(TL)*_nummix);
#endif
	
	est._nummix = _nummix;
	est._nummixAlloc = _nummixAlloc;
	est._dim = _dim;
	est._dimVar = _dimVar;
	est._sharedReferenecCounter = &_referenceCounter;
	est._sharedThrdFailedStatus = &_threadsFailed;
	est._sharedfNumStabilityTroubles = &_fNumStabilityTroubles;
	est._flagsSet = _flagsSet;
	est._fMixProbOnly = _fMixProbOnly;
	est._fMean = _fMean;
	est._fVar = _fVar;
	est._fFullVar = _fFullVar;
	est._fAux = _fAux;

	// increase reference counter to indicate shared memory (used in destructor)
	{
		boost::mutex::scoped_lock lock(the_sharedmem_mutex);
		++_referenceCounter;
	}
}



template <typename TS, typename TV, typename TL>
void GMMStatsEstimator<TS, TV, TL>::accumulateStats (TV** vecs, unsigned int NSamples, unsigned int dim,
													 GMMStats<TS>& gmms, unsigned int offset)
{
	if(!_modelIn) {
		if(!gmms.isAllocated()) {
			gmms.alloc(dim, 1, _fMean, _fVar, _fFullVar, _fAux);
			gmms.reset();
		}

#ifndef _EXCLUDE_SAFETY_CONDS_
		checkProperAllocation(gmms);
#endif

		std::vector<TL> gammas;
		std::vector<TL> gammas_old;
		gammas.push_back((TL) 1.0);
		gammas_old.push_back((TL) 1.0);
		unsigned int mixidx = 0;
		for(unsigned int i = 0; i < NSamples; i++) 
		{
			if(*_sharedThrdFailedStatus)
				throw std::runtime_error("accumulateStats(): one of the threads failed - thread terminating");

			accumulateStats(vecs[offset + i], dim, gmms, gammas, &mixidx, 1);			

			if(_fAux) {
				if (i > 0) accumulateAuxStats(vecs[offset + i - 1], vecs[offset + i], dim, gmms, gammas, gammas_old, &mixidx, &mixidx, 1, 1);
			}
		}
	}
	else {
		if(!gmms.isAllocated()) {
			gmms.alloc(_dim, _nummixAlloc, _fMean, _fVar, _fFullVar, _fAux);
			gmms.reset();
		}

		if(dim != _dim || dim != gmms._dim)
			throw std::logic_error("accumulateStats(): Inconsistent model & feature dimensions!");		

#ifndef _EXCLUDE_SAFETY_CONDS_
		if(_mask != NULL && (_mask->N < NSamples || _mask->dim < _nummix))
			throw std::logic_error("accumulateStats(): logLikeMask improperly allocated! \n\t");
		checkProperAllocation(gmms);
#endif

		TL ll; // frame logLike
		unsigned int *mixidx = NULL, Nmi = 0;
		std::vector<TL> _gammas_old;
		unsigned int *mixidx_old=NULL;
		unsigned int Nmi_old = 0;
		for(unsigned int i = 0; i < NSamples; i++) 
		{				
			if(*_sharedThrdFailedStatus)
				throw std::runtime_error("accumulateStats(): one of the threads failed - thread terminating");

			if(_useMask) {
				mixidx = &_mask->maskIdxs[offset + i][0];
				Nmi = _mask->maskIdxs[offset + i].size();
			}

			ll = compGammas(vecs[offset + i], dim, _gammas, mixidx, Nmi);
			gmms._totLogLike += ll;

			if(_fillMask) {			
				memcpy(_mask->gammas[offset + i], &_gammas[0], sizeof(_gammas[0])*_nummix);
				_mask->logLikes[offset + i] = ll;
			}

			accumulateStats(vecs[offset + i], dim, gmms, _gammas, mixidx, Nmi);			

			if(_fAux) {
				if (i == 0) {
					if(mixidx != NULL) mixidx_old = new unsigned int [_nummix];
					Nmi_old = Nmi;
				} else {
					accumulateAuxStats(vecs[offset + i - 1], vecs[offset + i], dim, gmms, _gammas, _gammas_old, mixidx, mixidx_old, Nmi, Nmi_old);
				}
				_gammas_old = _gammas;
				if(mixidx != NULL) {
					memcpy(mixidx_old, mixidx, sizeof(int) * Nmi);
					Nmi_old = Nmi;
					if(i == NSamples - 1) delete[] mixidx_old;
				}
			}
		}
	}

	gmms._totalAccSamples += NSamples;
}



template <typename TS, typename TV, typename TL>
void GMMStatsEstimator<TS, TV, TL>::compVecsLogLike (TV** vecs, unsigned int NSamples, unsigned int dim,
												     TL* outLogLikes, unsigned int offset)
{
	if(dim != _dim)
		throw std::logic_error("compVecsLogLike(): Inconsistent model & feature dimensions!");		
	
	TL ll; // frame logLike
	unsigned int *mixidx = NULL, Nmi = 0;
	for(unsigned int i = 0; i < NSamples; i++) 
	{				
		if(*_sharedThrdFailedStatus)
			throw std::runtime_error("compVecsLogLike(): one of the threads failed - thread terminating");
				
		if(_useMask) {
			mixidx = &_mask->maskIdxs[offset + i][0];
			Nmi = _mask->maskIdxs[offset + i].size();
		}
		
		if(!_fillMask)
			outLogLikes[offset + i] = getLikeAndGammas(vecs[offset + i], dim, _gammas, mixidx, Nmi);				
		else 
		{			
			ll = compGammas(vecs[offset + i], dim, _gammas, mixidx, Nmi);

			memcpy(_mask->gammas[offset + i], &_gammas[0], sizeof(_gammas[0])*_nummix);
			_mask->logLikes[offset + i] = ll;

			outLogLikes[offset + i] = ll;
		}
	}
}



//template <typename TS, typename TV, typename TL>
//void GMMStatsEstimator<TS, TV, TL>::accumulateAuxStats (TV* vec1, TV* vec2, TV* vec3, unsigned int dim, 
//														GMMStats<TS>& gmms, std::vector<TL>& gammas,
//													    unsigned int* mixIdxs, unsigned int Nmi)
//{
//	unsigned int index, M = (mixIdxs != NULL) ? Nmi : _nummix;
//	for(unsigned int i = 0; i < M; i++) {
//		index = (mixIdxs != NULL) ? mixIdxs[i] : i; 
//		if(gammas[index] <= _minGamma)
//			continue;
//
//		unsigned int shift = 0;
//		for(unsigned int k = 0; k < dim; k++) {
//			gmms._ms[index].aux[k] += (TS) (0.5 * gammas[index] * ( fabs(vec2[k] - vec1[k]) + fabs(vec3[k] - vec2[k]) ) );
//		} // for k
//		gmms._ms[index].aux2 += sqrt(gammas[index]);
//	}
//}

//new D variant 
template <typename TS, typename TV, typename TL>
void GMMStatsEstimator<TS, TV, TL>::accumulateAuxStats (TV* vec1, TV* vec2, unsigned int dim, 
														GMMStats<TS>& gmms, std::vector<TL>& gammas, std::vector<TL>& gammas_old,
													    unsigned int* mixIdxs, unsigned int* mixIdxs_old, unsigned int Nmi, unsigned int Nmi_old)
{

	//no Gaussians pruning:
	if(mixIdxs == NULL) {
		//compute gammas norm factor
		TL gs = 0;
		for(unsigned int i = 0; i < _nummix; i++) {
			TL g = gammas[i] * gammas_old[i];
			if(g > _minGamma) gs += g;
		}
		if(gs > 0) gs = 1/gs;

		for(unsigned int i = 0; i < _nummix; i++) {
			TL g = gs * gammas[i] * gammas_old[i];
			if(gammas[i] > _minGamma) gmms._ms[i].aux2 += sqrt((TS)gammas[i]);
			if(g <= _minGamma) continue;
			for(unsigned int k = 0; k < dim; k++) {
				TV ad = fabs(vec2[k] - vec1[k]);
				gmms._ms[i].aux[k] += (TS) (g * ad);
			} // for k
			gmms._ms[i].aux3 += g;
			////LLLLLLLLLLLLLLLLLLL
			//if(_nummix==2) {
			//	FILE *fid = fopen("xxa.txt", "a");

			//	fclose(fid);
			//}
			////LLLLLLLLLLLLLLLLLLL
		}

	} else {
		//two gammas best lists
		
		unsigned int i1 = 0;
		unsigned int i2 = 0;
		//compute gammas norm factor
		TL gs = 0;
		while(true) {
			unsigned int id1 = mixIdxs_old[i1];
			unsigned int id2 = mixIdxs[i2];
			TL g = (id1 == id2) * gammas[id2] * gammas_old[id1];
			if(g > _minGamma) gs += g;
			i1 += (id1 <= id2);
			i2 += (id2 <= id1);
			if(i1 >= Nmi_old) break;
			if(i2 >= Nmi) break;
		}
		if(gs > 0) gs = 1/gs;

		i1 = 0;
		i2 = 0;
		while(true) {
			unsigned int id1 = mixIdxs_old[i1];
			unsigned int id2 = mixIdxs[i2];
			TL g = (id1 == id2) * gs * gammas[id2] * gammas_old[id1];
			if(g > _minGamma) {
				for(unsigned int k = 0; k < dim; k++) {
					TV ad = fabs(vec2[k] - vec1[k]);
					gmms._ms[id1].aux[k] += (TS) (g * ad);
				} // for k
				gmms._ms[id1].aux3 += g;
			}
			i1 += (id1 <= id2);
			i2 += (id2 <= id1);
			if(i1 >= Nmi_old) break;
			if(i2 >= Nmi) break;
		}
		for(unsigned int i = 0; i < Nmi; i++) {
			unsigned int id = mixIdxs[i];
			gmms._ms[id].aux2 += sqrt((TS)gammas[id]);
		}

	}

}


template <typename TS, typename TV, typename TL>
void GMMStatsEstimator<TS, TV, TL>::accumulateStats (TV* vec, unsigned int dim, GMMStats<TS>& gmms, std::vector<TL>& gammas,
													 unsigned int* mixIdxs, unsigned int Nmi)
{
	unsigned int index, M = (mixIdxs != NULL) ? Nmi : _nummix;

	////LLLLLLLLLLLLLLLLLLL
	//if(M == 2) {
	//	FILE *fid = fopen("xxx.txt", "a");
	//	fprintf(fid, "%e %e\n", gammas[0], gammas[1]);
	//	fclose(fid);
	//}
	////LLLLLLLLLLLLLLLLLL


	for(unsigned int i = 0; i < M; i++) {
		index = (mixIdxs != NULL) ? mixIdxs[i] : i; 
		if(gammas[index] <= _minGamma)
			continue;

		gmms._ms[index].mixProb += (TS) gammas[index];

		if(_fMixProbOnly)
			continue;

		unsigned int shift = 0;
		for(unsigned int k = 0; k < dim; k++) {
			if(_fMean)
				gmms._ms[index].mean[k] += (TS) (gammas[index] * vec[k]); 
			if(_fVar) {
				shift += k;
				if(_fFullVar) {
					// je mozne nahradit lapack::SSPR() 
					for(unsigned int kk = k; kk < dim; kk++)
						gmms._ms[index].var[k * dim + kk - shift] += (TS) (gammas[index] * vec[k] * vec[kk]);
				}
				else {
					if(gmms._allFV)
						gmms._ms[index].var[k * dim + k - shift] += (TS) (gammas[index] * vec[k] * vec[k]);
					else
						gmms._ms[index].var[k] += (TS) (gammas[index] * vec[k] * vec[k]);
				}
			}
		} // for k
	} 
}



template <typename TS, typename TV, typename TL>
template <typename T>
void GMMStatsEstimator<TS, TV, TL>::getMeanGrad (GMModel& model, const GMMStats<TS>& stats, 
												 std::vector<T>& grad, double normval)
{
	unsigned int D = model.GetDimension();
	unsigned int M = model.GetNumberOfMixtures();	
	
	if(grad.size() < M*D)
		grad.resize(M*D);

	const TS *m_stats;
	TS mixProb;

	float *m_model, *v_model;
	for(unsigned int m = 0; m < M; m++) {
		m_model = model.GetMixtureMean(m);
		v_model = model.GetMixtureVarDiag(m);
		m_stats = stats.getMean(m);
		mixProb = stats.getProb(m);				
		for(unsigned int d = 0; d < D; d++)
			grad[m * D + d] = (T) ((m_stats[d] - mixProb*m_model[d]) / (T) (v_model[d]*normval));
	}
}



template <typename TS, typename TV, typename TL>
void GMMStatsEstimator<TS, TV, TL>::centralizeStats (GMMStats<TS>& stats, GMModel& model,
													 bool centrMeans, bool centrVars, bool centrFVars)
{
	unsigned int D = model.GetDimension();
	unsigned int M = model.GetNumberOfMixtures();	

	const TS *m_stats;
	TS mixProb;

	float *m_model;
	for(unsigned int m = 0; m < M; m++) {
		m_model = model.GetMixtureMean(m);
		m_stats = stats.getMean(m);
		mixProb = stats.getProb(m);	
		
		unsigned int shift = 0;
		for(unsigned int k = 0; k < D; k++) {
			if(centrMeans && stats._allM)
				stats._ms[m].mean[k] -= (TS) (mixProb * m_model[k]); 

			if(centrVars && stats._allV) {
				shift += k;
				if(centrFVars && stats._allFV) {					
					for(unsigned int kk = k; kk < D; kk++)
						stats._ms[m].var[k * D + kk - shift] -= (TS) (m_stats[k] * m_model[kk] + m_stats[kk] * m_model[k] - mixProb * m_model[k] * m_model[kk]);
				}
				else {
					if(stats._allFV)
						stats._ms[m].var[k * D + k - shift] -= (TS) (2 * m_stats[k] * m_model[k] - mixProb * m_model[k] * m_model[k]);
					else
						stats._ms[m].var[k] -= (TS) (2 * m_stats[k] * m_model[k] - mixProb * m_model[k] * m_model[k]);
				}
			}
		} // for k
	}
}



template <typename TS, typename TV, typename TL>
inline void GMMStatsEstimator<TS, TV, TL>::addLog (TL& p1, TL p2) {
	if(p1 - LPDIF > p2)
		return;

	if(p1>p2)
		p1 = (p1 + log(1.0f+exp(p2-p1)));
	else
		p1 = (p2 + log(1.0f+exp(p1-p2)));
}



// robust and fast AddLog - faster aproximation
template <typename TS, typename TV, typename TL>
inline void GMMStatsEstimator<TS, TV, TL>::addLogAprox (TL& p1, TL p2) {
	TL y, d;
	int n;

	if(p1>p2) {
		y = p1;
		d = p1 - p2;
	} else {
		y = p2;
		d = p2 - p1;
	}
	if(d > 6.0f) {p1 = y; return;}
	n = (int) d;

	switch(n) {

	case 0:
        y = y + 1.1678898008531799e-001f*d*d + -4.9574118267684197e-001f*d +6.9276461379079390e-001f;
		break;
	case 1:
        y = y + 7.4939387987078332e-002f*d*d - 4.0959403625058377e-001f*d + 6.4717578194953718e-001f;
		break; 
	case 2:
        y = y + 3.5561412387495495e-002f*d*d - 2.5515429569966075e-001f*d + 4.9450019773793175e-001f;
		break;
	case 3:
        y = y + 1.4505410387159802e-002f*d*d - 1.3152400320902408e-001f*d + 3.1238448119950923e-001f;
		break;
	case 4:
        y = y + 5.5512795095136935e-003f*d*d - 6.1216380576786453e-002f*d + 1.7410441415833702e-001f;
		break;
	case 5:
        y = y + 2.0725526520460253e-003f*d*d - 2.6969656372233891e-002f*d + 8.9715433724479501e-002f;
   
	}
	p1 = y;
}



#endif

