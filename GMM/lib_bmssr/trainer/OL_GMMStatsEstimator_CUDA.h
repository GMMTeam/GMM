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

#ifndef _OL_STATS_EST_CUDA_
#define _OL_STATS_EST_CUDA_

#include "trainer/OL_GMMStatsEstimator.h"
#include "trainer/OL_GMMStatsEstimator_SSE.h"
#include "trainer/CU_GMMStatsEstimator_CUDA.h"

template <typename TS>
class GMMStatsEstimator_CUDA : public GMMStatsEstimator_SSE <TS, float> {
public:

	GMMStatsEstimator_CUDA(int GPU_id = -1);
	virtual ~GMMStatsEstimator_CUDA();

	virtual void insertModel (GMModel& model, bool first = true); 

	virtual void setModelToBeUsed (unsigned int n);

	virtual void accumulateStatsMT (float** vectors, unsigned int NSamples, unsigned int dim,
								    GMMStats<TS>& gmms);

	virtual void compVecsLogLikeMT (float** vecs, unsigned int NSamples, unsigned int dim,
									float* outLogLikes);

	virtual GMMStatsEstimator<TS, float, float>* getNewInstance();

protected:
	
	void getMinNSamples2GPU (unsigned int &minNSamplesGPU, unsigned int &minNSamplesGPUprocess,
							 unsigned int NSamples, unsigned int dim, unsigned int aligDimAux,
							 bool allocStats);

	static void alignVectors (float **vecs, unsigned int NSamples, unsigned int dim,
							  float **vecsAligned, unsigned int &NframesAligned, unsigned int &alignedDim);

	void export2GMMStats (GMMStats<TS>& gmms, float *meanS, float *varS, float *mixProbS, float *auxS,
						  unsigned int dim, unsigned int nummix);

	//void fillMaskwithGammasAndLogLikes (GMMStatsEstimator<TS, float, float>::logLikeMask& mask, unsigned int NSamples);

	void reAllocGammaAndLogLikeBuffers (unsigned int alignedNSamples, unsigned int alignedNMix);
	
	void reallocOutArrays();

	void updateOptionsOnGPU();

	// model CUDA-rdy
	float* _meanBuffC;
	float* _ivarBuffC;
	float* _gConstBuffC;
	
	float* _auxStatsBuffC;
	float* _gammasBuffC;
	float* _logLikesBuffC;

	unsigned int _alignedDimC;
	unsigned int _alignedDimVarC;
	unsigned int _alignedNMixC;
	unsigned int _alignedNMixAllocC;
	unsigned int _alignedDimAuxC;

	bool _ivarFullAlloc;

	float* _paramCacheC;
	unsigned int _alignedNSamplesC;

	bool _cudaDeviceFound;	

	std::vector<GMMStatsEstimator_GPU*> _GPUs;

private:
	// prohibited
	GMMStatsEstimator_CUDA (const GMMStatsEstimator_CUDA<TS>&);
	GMMStatsEstimator_CUDA<TS>& operator= (const GMMStatsEstimator_CUDA<TS>&);

};


#include "trainer/OL_GMMStatsEstimator_CUDA.cpp"

#endif
