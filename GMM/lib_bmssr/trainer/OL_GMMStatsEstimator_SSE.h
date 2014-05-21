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

#ifndef _OL_STATS_EST_SSE_
#define _OL_STATS_EST_SSE_

#include "trainer/OL_GMMStatsEstimator.h"
#include <vector>

template <typename TS, typename TL = float>
class GMMStatsEstimator_SSE : public GMMStatsEstimator<TS, float, TL>  {
public:

	GMMStatsEstimator_SSE();
	virtual ~GMMStatsEstimator_SSE();

	virtual void insertModel (GMModel& model, bool first = true); 
	
	virtual void setModelToBeUsed (unsigned int n);

	virtual TL computeMixLogLike (unsigned int mixnum, float* vec, unsigned int dim);
	virtual TL computeMixLogLikeFullCov (unsigned int mixnum, float* vec, unsigned int dim);

	virtual GMMStatsEstimator<TS, float, TL>* getNewInstance();

protected:

	virtual void clearGroups();

	// handles one thread
	virtual void shareData (GMMStatsEstimator<TS, float, TL>& est);		
	virtual void initializeInternalData();

	// model SSE
	float *_meanBuff; 	
	float *_ivarBuff;	
	std::vector <float *> _meanBuffGroup, _ivarBuffGroup;	

	unsigned int _alignedDim;
	unsigned int _alignedDimVar;

private:
	static const unsigned int DIM_BLOCK;

	// data buffer
	float *_alignedData;

	// memory management when threads are in use
	unsigned int _referenceCounterSSE, *_sharedReferenecCounterSSE;

	// prohibited
	GMMStatsEstimator_SSE (const GMMStatsEstimator_SSE<TS, TL>&);
	GMMStatsEstimator_SSE<TS, TL>& operator= (const GMMStatsEstimator_SSE<TS, TL>&);
};

// definition of constants
template <typename TS, typename TL>
const unsigned int GMMStatsEstimator_SSE<TS, TL>::DIM_BLOCK = 4;


#include "trainer/OL_GMMStatsEstimator_SSE.cpp"

#endif