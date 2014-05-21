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

#ifdef _OL_STATS_EST_SSE_

#include <xmmintrin.h>
#include <boost/thread/mutex.hpp>

//#include <typeinfo>

#ifdef __GNUC__
#	include "general/my_inttypes.h"
#	define _aligned_malloc _mm_malloc
#	define _aligned_free _mm_free
#else
#	include <intrin.h>
#	define isnan(x) _isnan(x) // kuoli linuchu
#	define finite(x) _finite(x) // kuoli linuchu
#endif


namespace {
	boost::mutex the_dtor_mutex_sse, the_sharedmem_mutex_sse;	
}


template <typename TS, typename TL>
GMMStatsEstimator_SSE<TS, TL>::GMMStatsEstimator_SSE() 
: GMMStatsEstimator<TS, float, TL>(),
_meanBuff(NULL),
_ivarBuff(NULL),
_alignedDim(0),
_alignedDimVar(0),
_alignedData(NULL),
_referenceCounterSSE(0), 
_sharedReferenecCounterSSE(&_referenceCounterSSE)
{
	this->_estimType = std::string("SSE");
};



template <typename TS, typename TL>
GMMStatsEstimator_SSE<TS, TL>::~GMMStatsEstimator_SSE() {	
	
	// shared data
#ifdef _SHARE_DATA_
	boost::mutex::scoped_lock lock(the_dtor_mutex_sse);
	if(*_sharedReferenecCounterSSE == 0) {
		clearGroups();
		if(_meanBuff != NULL)
			_aligned_free(_meanBuff);
		if(_ivarBuff != NULL)
			_aligned_free(_ivarBuff);
	}
	else --(*_sharedReferenecCounterSSE);
#else
	clearGroups();
	if(_meanBuff != NULL)
		_aligned_free(_meanBuff);
	if(_ivarBuff != NULL)
		_aligned_free(_ivarBuff);
#endif

	// class-unique data
	if(_alignedData != NULL)
		_aligned_free(_alignedData);
}



template <typename TS, typename TL>
void GMMStatsEstimator_SSE<TS, TL>::insertModel (GMModel& model, bool first) {	
	
	if(first)
		clearGroups();

	bool realloc = false;
	if(first && (model.GetNumberOfMixtures() > this->_nummixAlloc || model.GetDimension() != this->_dim)) {
		if(_meanBuff != NULL)
			delete [] _meanBuff;
		if(_ivarBuff != NULL)
			delete [] _ivarBuff;
		if(_alignedData != NULL)
			delete [] _alignedData;
		realloc = true;
	}

	GMMStatsEstimator<TS, float, TL>::insertModel(model, first);
	this->_modelIn = false; // reset - model still not in!

	
	unsigned int alignedDim = static_cast<unsigned int> (DIM_BLOCK * ceil( this->_dim / (float) DIM_BLOCK));
	unsigned int alignedDimVar;
	if (model.GetFullCovStatus())
		alignedDimVar = alignedDim * alignedDim;
	else alignedDimVar = alignedDim;

	if(realloc || !first) {
		if((_meanBuff = (float *) _aligned_malloc(alignedDim * this->_nummixAlloc * sizeof(float), DIM_BLOCK * sizeof(float))) == NULL)
			throw std::runtime_error("insertModel(): Not enough memory!\n\t");

		if((_ivarBuff = (float *) _aligned_malloc(alignedDimVar * this->_nummixAlloc * sizeof(float), DIM_BLOCK * sizeof(float))) == NULL) {
			delete [] _meanBuff;
			throw std::runtime_error("insertModel(): Not enough memory!\n\t");
		}
		if(first) {
			if((_alignedData = (float *) _aligned_malloc(alignedDim * sizeof(float), DIM_BLOCK * sizeof(float))) == NULL) {
				delete [] _meanBuff;
				delete [] _ivarBuff;
				throw std::runtime_error("insertModel(): Not enough memory!\n\t"); 
			}
			memset(_alignedData, 0, sizeof(float) * alignedDim);
		}
	}

	memset(_meanBuff, 0, sizeof(float) * alignedDim);
	for(unsigned int m = 0; m < this->_nummix; m++)
		memcpy(_meanBuff + m * alignedDim, model.GetMixtureMean(m), sizeof(float)*this->_dim);					

	if (!model.GetFullCovStatus()) {
		memset(_ivarBuff, 0, sizeof(float) * alignedDimVar * this->_nummixAlloc);
		for(unsigned int m = 0; m < this->_nummix; m++)
			memcpy(_ivarBuff + m * alignedDim, this->_icov + m * this->_dimVar, sizeof(float) * this->_dimVar);			
	}
	else {
		memset(_ivarBuff, 0, sizeof(float) * alignedDimVar * this->_nummixAlloc);
		for(unsigned int m = 0; m < this->_nummix; m++) {										
			for(unsigned int d = 0; d < this->_dim; d++)
				memcpy(_ivarBuff + m * alignedDimVar + d * alignedDim, this->_icov + m * this->_dimVar + d * this->_dim, sizeof(float) * this->_dim);
		}
	}

	_meanBuffGroup.push_back(_meanBuff);
	_ivarBuffGroup.push_back(_ivarBuff);

	_alignedDim = alignedDim;
	_alignedDimVar = alignedDimVar;
	this->_modelIn = true;
}



template <typename TS, typename TL>
void GMMStatsEstimator_SSE<TS, TL>::setModelToBeUsed (unsigned int n) {

	GMMStatsEstimator<TS, float, TL>::setModelToBeUsed(n);
	
	_meanBuff = _meanBuffGroup.at(n);
	_ivarBuff = _ivarBuffGroup.at(n);
}



template <typename TS, typename TL>
void GMMStatsEstimator_SSE<TS, TL>::clearGroups() {

	if (this->_modelsGroup.size() > 0) {
		_meanBuff = _meanBuffGroup.at(0);
		_ivarBuff = _ivarBuffGroup.at(0);
	}

	// do not delete the memory of the first model - may be needed for next group
	for(unsigned int it = 1; it < this->_modelsGroup.size(); it++) {
		_aligned_free(_meanBuffGroup[it]);		
		_aligned_free(_ivarBuffGroup[it]);		
	}

	_meanBuffGroup.clear();
	_ivarBuffGroup.clear();

	GMMStatsEstimator<TS, float, TL>::clearGroups();
}



template <typename TS, typename TL>
TL GMMStatsEstimator_SSE<TS, TL>::computeMixLogLike (unsigned int mixnum, float* vec, unsigned int dim) {	
			
#ifndef _EXCLUDE_SAFETY_CONDS_
	if (mixnum > this->_nummix) 
		throw std::out_of_range("computeMixLogLike(): Specified index out of bounds! \n\t");		

	if(!this->_modelIn)
		throw std::logic_error("computeMixLogLike(): None model inserted! \n\t");

#endif

	if(dim != this->_dim)
		throw std::logic_error("computeMixLogLike(): Inconsistent model & feature vector - different dimensions! \n\t");

	// SSE model
	unsigned int dim_B = _alignedDim/DIM_BLOCK;
	__m128* mean = (__m128*) (_meanBuff + mixnum * _alignedDim);
	__m128* ivar = (__m128*) (_ivarBuff + mixnum * _alignedDim);
		
	memcpy(_alignedData, vec, dim * sizeof(float)); 
	__m128* data = (__m128*) _alignedData;

	// SSE computation
	__m128 foo1 = _mm_sub_ps(*data, *mean);		  // m1 = x - m
	__m128 foo2 = _mm_mul_ps(foo1, foo1);		  // m2 = (x - m)^2
	__m128 foo3, foo4 = _mm_mul_ps(foo2, *ivar);  // m3 = ((x - m)^2) * ivar	

	for(unsigned int d = 1; d < dim_B; d++) {
		mean++; ivar++; data++;
		foo1 = _mm_sub_ps(*data, *mean);  // m1 = x - m
		foo2 = _mm_mul_ps(foo1, foo1);    // m2 = (x - m)^2
		foo3 = _mm_mul_ps(foo2, *ivar);   // m3 = ((x - m)^2) * ivar
		foo4 = _mm_add_ps(foo4, foo3);		
	}
	// sum up all 4 elements (floats) in 'foo4'
	foo1 = _mm_add_ps(foo4, _mm_movehl_ps(foo4, foo4));
	foo1 = _mm_add_ss(foo1, _mm_shuffle_ps(foo1, foo1, 1));

	//first element in foo1 is the sum of elements in foo4
	TL exponent = * (TL*) (&foo1);

	TL ll = this->_logGconst[mixnum] - 0.5f * exponent;

	if(!finite(ll) || isnan(ll))
		return this->_minLogLike;

	return ll;
}



template <typename TS, typename TL>
TL GMMStatsEstimator_SSE<TS, TL>::computeMixLogLikeFullCov (unsigned int mixnum, float* vec, unsigned int dim) {	
			
#ifndef _EXCLUDE_SAFETY_CONDS_
	if (mixnum > this->_nummix) 
		throw std::out_of_range("computeMixLogLike(): Specified index out of bounds! \n\t");		

	if(!this->_modelIn)
		throw std::logic_error("computeMixLogLike(): None model inserted! \n\t");

#endif

	if(dim != this->_dim)
		throw std::logic_error("computeMixLogLike(): Inconsistent model & feature vector - different dimensions! \n\t");

	// SSE model
	unsigned int dim_B = _alignedDim/DIM_BLOCK;
	__m128* mean = (__m128*) (_meanBuff + mixnum * _alignedDim);
	__m128* ivar = (__m128*) (_ivarBuff + mixnum * _alignedDimVar);
		
	memcpy(_alignedData, vec, dim * sizeof(float)); 
	__m128* data = (__m128*) _alignedData;


	__m128 sum = _mm_setzero_ps();
	for(unsigned int d = 0; d < dim_B; d++) {
		__m128 dx1 = _mm_sub_ps(data[d], mean[d]);		  // m1 = x - m

		for(unsigned int dd = 0; dd < dim_B; dd++) {
			__m128 foo, foo2;
			__m128 dx2 = _mm_sub_ps(data[dd], mean[dd]);     // m2 = x - m
			
			foo = _mm_mul_ps(dx2, ivar[0]);
			foo2 = _mm_mul_ps(foo, _mm_set_ps1(((float*) &dx1)[0]));
			//foo2 = _mm_mul_ps(foo, _mm_set_ps1(dx1.m128_f32[0]));
			sum = _mm_add_ps(sum, foo2);

			foo = _mm_mul_ps(dx2, ivar[dim_B]);
			foo2 = _mm_mul_ps(foo, _mm_set_ps1(((float*) &dx1)[1]));
			//foo2 = _mm_mul_ps(foo, _mm_set_ps1(dx1.m128_f32[1]));
			sum = _mm_add_ps(sum, foo2);

			foo = _mm_mul_ps(dx2, ivar[2*dim_B]);
			foo2 = _mm_mul_ps(foo, _mm_set_ps1(((float*) &dx1)[2]));
			//foo2 = _mm_mul_ps(foo, _mm_set_ps1(dx1.m128_f32[2]));
			sum = _mm_add_ps(sum, foo2);

			foo = _mm_mul_ps(dx2, ivar[3*dim_B]);
			foo2 = _mm_mul_ps(foo, _mm_set_ps1(((float*) &dx1)[3]));
			//foo2 = _mm_mul_ps(foo, _mm_set_ps1(dx1.m128_f32[3]));
			sum = _mm_add_ps(sum, foo2);

			ivar++;
		}
		ivar += 3 * dim_B;
	}

	// sum up all 4 elements (floats) in 'foo4'
	__m128 foo1 = _mm_add_ps(sum, _mm_movehl_ps(sum, sum));
	foo1 = _mm_add_ss(foo1, _mm_shuffle_ps(foo1, foo1, 1));

	//first element in foo1 is the sum of elements in foo4
	TL exponent = * (TL*) (&foo1);

	TL ll = this->_logGconst[mixnum] - 0.5f * exponent;

	if(!finite(ll) || isnan(ll))
		return this->_minLogLike;

	return ll;
}



template <typename TS, typename TL>
GMMStatsEstimator<TS, float, TL>* GMMStatsEstimator_SSE<TS, TL>::getNewInstance() {
	return new GMMStatsEstimator_SSE<TS, TL>;
}



template <typename TS, typename TL>
void GMMStatsEstimator_SSE<TS, TL>::initializeInternalData() {
	
	GMMStatsEstimator<TS, float, TL>::initializeInternalData();

	// class-unique data - memory allocation	
	if((_alignedData = (float *) _aligned_malloc(_alignedDim * sizeof(float), DIM_BLOCK * sizeof(float))) == NULL)
		throw std::runtime_error("initializeInternalData(): Not enough memory!\n\t"); 		

	memset(_alignedData + this->_dim, 0, sizeof(float)*(_alignedDim - this->_dim));
}

	
	
template <typename TS, typename TL>
void GMMStatsEstimator_SSE<TS, TL>::shareData (GMMStatsEstimator<TS, float, TL>& est) {
	
	// share base class data
	GMMStatsEstimator<TS, float, TL>::shareData(est);

	// share this class data;
	// test whether 'est' is a parent of GMMStatsEstimator_SSE => return
	GMMStatsEstimator_SSE<TS, TL>* est_SSE = dynamic_cast< GMMStatsEstimator_SSE<TS, TL>* > (&est);	
	if(est_SSE == NULL)
		return;

	// shared data
#ifdef _SHARE_DATA_
	est_SSE->_meanBuff = _meanBuff;
	est_SSE->_ivarBuff = _ivarBuff;
#else
	if((est_SSE->_meanBuff = (float *) _aligned_malloc(_alignedDim * this->_nummixAlloc * sizeof(float), DIM_BLOCK * sizeof(float))) == NULL)
		throw std::runtime_error("shareData(): Not enough memory!\n\t");

	if((est_SSE->_ivarBuff = (float *) _aligned_malloc(_alignedDimVar * this->_nummixAlloc * sizeof(float), DIM_BLOCK * sizeof(float))) == NULL) {
		delete [] est_SSE->_meanBuff;
		throw std::runtime_error("shareData(): Not enough memory!\n\t");
	}
	memcpy(est_SSE->_meanBuff, _meanBuff, sizeof(float)*_alignedDim*this->_nummixAlloc);
	memcpy(est_SSE->_ivarBuff, _ivarBuff, sizeof(float)*_alignedDimVar*this->_nummixAlloc);
#endif

	est_SSE->_alignedDim = _alignedDim;
	est_SSE->_alignedDimVar = _alignedDimVar;
	est_SSE->_sharedReferenecCounterSSE = &_referenceCounterSSE;
	
	// increase reference counter to indicate shared memory (used in destructor)
	{
		boost::mutex::scoped_lock lock(the_sharedmem_mutex_sse);
		++_referenceCounterSSE;
	}
}

#endif

