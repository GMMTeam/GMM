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

#ifdef _OL_STATS_EST_CUDA_

#include "trainer/CU_GMMStatsEstimator_CUDA.h"
#include <iostream>
#include <limits>

template <typename TS>
GMMStatsEstimator_CUDA<TS>::GMMStatsEstimator_CUDA(int GPU_id)
: GMMStatsEstimator_SSE<TS, float> (),
_meanBuffC(NULL),
_ivarBuffC(NULL),
_gConstBuffC(NULL),
_auxStatsBuffC(NULL),
_gammasBuffC(NULL),
_logLikesBuffC(NULL),
_alignedDimC(0),
_alignedDimVarC(0),
_alignedNMixC(0),
_alignedDimAuxC(0),
_ivarFullAlloc(false),
_paramCacheC(NULL),
_alignedNSamplesC(0)
{		
	int NgpuDevices = GMMStatsEstimator_GPU::deviceCount();

	if (NgpuDevices == 0) {
		this->_estimType = std::string("SSE");
		_cudaDeviceFound = false;
	}
	else {		
		if (GPU_id > -1) {			
			GPU_id *= (GPU_id < NgpuDevices);
			_GPUs.push_back(new GMMStatsEstimator_GPU(GPU_id));
		}
		else {			
			for (unsigned int i = 0; i < NgpuDevices; i++)
				_GPUs.push_back(new GMMStatsEstimator_GPU(i));
		}
		this->_estimType = std::string("GPU CUDA");
		_cudaDeviceFound = true;

		// initialize options
		updateOptionsOnGPU();
	}
}



template <typename TS>
GMMStatsEstimator_CUDA<TS>::~GMMStatsEstimator_CUDA() 
{
	// NO THROWS IN DESTRUCTOR!

	if(_cudaDeviceFound) {
		if(_meanBuffC != NULL)
			GMMStatsEstimator_GPU::freeOnHost((void **) &_meanBuffC);
		
		if(_ivarBuffC != NULL)
			GMMStatsEstimator_GPU::freeOnHost((void **) &_ivarBuffC);
		
		if(_gConstBuffC != NULL)
			GMMStatsEstimator_GPU::freeOnHost((void **) &_gConstBuffC);
		
		if(_auxStatsBuffC != NULL)
			GMMStatsEstimator_GPU::freeOnHost((void **) &_auxStatsBuffC);

		if(_paramCacheC != NULL)
			GMMStatsEstimator_GPU::freeOnHost((void **) &_paramCacheC);

		if(_gammasBuffC != NULL)
			GMMStatsEstimator_GPU::freeOnHost((void **) &_gammasBuffC);

		if(_logLikesBuffC != NULL)
			GMMStatsEstimator_GPU::freeOnHost((void **) &_logLikesBuffC);

		for (unsigned int n = 0; n < _GPUs.size(); n++)
			delete _GPUs[n];
	}
}


template <typename TS>
void GMMStatsEstimator_CUDA<TS>::updateOptionsOnGPU()
{
	for (unsigned int i = 0; i < _GPUs.size(); i++) {
		_GPUs[i]->_opt.minGamma = this->_minGamma;
		_GPUs[i]->_opt.minLogLike = this->_minLogLike;
		_GPUs[i]->_opt.frames_acc_blocks = this->_NAccBlocks_CUDA;
		_GPUs[i]->_opt.verbosity = this->_verbosity;
	}
}



template <typename TS>
void GMMStatsEstimator_CUDA<TS>::insertModel (GMModel& model, bool first) 
{
	bool realloc = false;
	if(model.GetNumberOfMixtures() > this->_nummixAlloc || model.GetDimension() != this->_dim) 
		realloc = true;

	GMMStatsEstimator_SSE<TS, float>::insertModel(model, first);
	if(!_cudaDeviceFound)
		return;

	this->_modelIn = false; // reset - model still not in!

	const unsigned int DIM_BLOCK = GMMStatsEstimator_GPU::DIM_BLOCK;
	const unsigned int GAUSS_BLOCK = GMMStatsEstimator_GPU::GAUSS_BLOCK;
	const unsigned int FRAME_BLOCK = GMMStatsEstimator_GPU::FRAME_BLOCK;

	unsigned int alignedDim = static_cast<unsigned int> (DIM_BLOCK * ceil( this->_dim / (float) DIM_BLOCK));
	unsigned int alignedNMix = static_cast<unsigned int> (GAUSS_BLOCK * ceil( this->_nummix / (float) GAUSS_BLOCK));	
	unsigned int alignedNMixAlloc = static_cast<unsigned int> (GAUSS_BLOCK * ceil( this->_nummixAlloc / (float) GAUSS_BLOCK));	
	unsigned int alignedDimVar;

	if(model.GetFullCovStatus()) {
		unsigned int foo = alignedDim / DIM_BLOCK;
		alignedDimVar = (DIM_BLOCK*DIM_BLOCK) / 2 * (foo * (foo + 1));
	}
	else alignedDimVar = alignedDim;

	if(realloc) {
		// delete occupied memory
		if(_meanBuffC != NULL)
			GMMStatsEstimator_GPU::freeOnHost((void **) &_meanBuffC);				
		
		if(_ivarBuffC != NULL)
			GMMStatsEstimator_GPU::freeOnHost((void **) &_ivarBuffC);				
		
		if(_gConstBuffC != NULL)
			GMMStatsEstimator_GPU::freeOnHost((void **) &_gConstBuffC);
				
		
		// alloc new memory
		GMMStatsEstimator_GPU::allocOnHost((void **) &_meanBuffC, alignedDim * alignedNMixAlloc * sizeof(float));			
		GMMStatsEstimator_GPU::allocOnHost((void **) &_ivarBuffC, alignedDimVar * alignedNMixAlloc * sizeof(float));
		GMMStatsEstimator_GPU::allocOnHost((void **) &_gConstBuffC, alignedNMixAlloc * sizeof(float));		

		_ivarFullAlloc = false;
	}

	memset(_meanBuffC, 0, alignedDim * alignedNMixAlloc * sizeof(float));
	memset(_ivarBuffC, 0, alignedDimVar * alignedNMixAlloc * sizeof(float));
	memset(_gConstBuffC, 0, alignedNMix * sizeof(float));


	// prepare aligned GMM means & gConst
	unsigned int db_idx, gblock_idx, buff_shift, i, j;
	for(unsigned int m = 0; m < this->_nummix; m++) 
	{	
		gblock_idx = m / GAUSS_BLOCK;
		buff_shift = gblock_idx * (GAUSS_BLOCK * alignedDim);

		float* mean = model.GetMixtureMean(m);				

		j = m % GAUSS_BLOCK;
		for(unsigned int d = 0; d < this->_dim; d++) 
		{
			db_idx = d / DIM_BLOCK;
			i = d % DIM_BLOCK;						

			_meanBuffC[buff_shift + db_idx * DIM_BLOCK * GAUSS_BLOCK + j * DIM_BLOCK + i] = mean[d];
		}
		_gConstBuffC[m] = this->_logGconst[m] + this->_logWeights[m];
	}
	// fill unused elements with 0
	for(unsigned int m = this->_nummix; m < alignedNMixAlloc; m++) 
		_gConstBuffC[m] = this->_minLogLike;

	if (!model.GetFullCovStatus()) {
		// prepare aligned GMM vars (diag)
		unsigned int db_idx, gblock_idx, buff_shift, i, j;
		for(unsigned int m = 0; m < this->_nummix; m++) 
		{	
			gblock_idx = m / GAUSS_BLOCK;
			buff_shift = gblock_idx * (GAUSS_BLOCK * alignedDim);

			float* var = model.GetMixtureVarDiag(m);

			j = m % GAUSS_BLOCK;
			for(unsigned int d = 0; d < this->_dim; d++) 
			{
				db_idx = d / DIM_BLOCK;
				i = d % DIM_BLOCK;						

				// MIN-VAR test already performed - see GMMStatsEstimator::insertModel()
				//if(var[d] < MIN_VAR)
				//	throw std::runtime_error("insertModel(): model contains very low variances!");

				_ivarBuffC[buff_shift + db_idx * DIM_BLOCK * GAUSS_BLOCK + j * DIM_BLOCK + i] = 1.0f / var[d];
			}
		}
	}
	else {						
		unsigned int Ngblock = alignedNMix / GAUSS_BLOCK;
		for(unsigned int gb = 0; gb < Ngblock; gb++) {
			unsigned int counter = 0;
			for(unsigned int k = 0; k < alignedDim / DIM_BLOCK; k++) {
				for(unsigned int kk = 0; kk <= k; kk++) {
					//if(gb == 0) printf("IVAR(%d, %d)", k, kk);
					for(unsigned int d = 0; d < DIM_BLOCK; d++) {
						for(unsigned int g = 0; g < GAUSS_BLOCK; g++) {
							if (gb * GAUSS_BLOCK + g < this->_nummix) {
								for(unsigned int dd = 0; dd < DIM_BLOCK; dd++) {
									unsigned int x = k * DIM_BLOCK + d;
									unsigned int y = kk * DIM_BLOCK + dd;
									if (x >= y) {
										float foo = this->_ivarBuff[(gb * GAUSS_BLOCK + g) * this->_alignedDimVar + x * alignedDim + y];
										if (x > y)
											foo *= 2;
										//if(gb+g == 0) printf(" %f (%d)", foo, gb * GAUSS_BLOCK * alignedDimVar + counter * DIM_BLOCK * DIM_BLOCK * GAUSS_BLOCK + d * GAUSS_BLOCK * DIM_BLOCK + g * DIM_BLOCK + dd);
										_ivarBuffC[gb * GAUSS_BLOCK * alignedDimVar + counter * DIM_BLOCK * DIM_BLOCK * GAUSS_BLOCK + d * GAUSS_BLOCK * DIM_BLOCK + g * DIM_BLOCK + dd] = foo;
										//} else {
										//	if(gb+g == 0) printf(" 0.0 (%d)", gb * GAUSS_BLOCK * alignedDimVar + counter * DIM_BLOCK * DIM_BLOCK * GAUSS_BLOCK + d * GAUSS_BLOCK * DIM_BLOCK + g * DIM_BLOCK + dd);
									}
								}
							}
						}
						//if(gb == 0) printf("\n", k, kk);
					}
					//if(gb == 0) printf("\n", k, kk);
					counter++;
				}				
			}			
		}
	}

	for (unsigned int n = 0; n < _GPUs.size(); n++) 
	{		
		unsigned int sharedMemPerBlock = _GPUs[n]->getSharedMemPerBlock();
		if (model.GetFullCovStatus() && sharedMemPerBlock / sizeof(float) / FRAME_BLOCK < this->_dim)	
			throw std::runtime_error("insertModel(): Dimensions too high to fit to GPU shared memory!");	

		_GPUs[n]->uploadModel(_meanBuffC, _ivarBuffC, _gConstBuffC, 
			this->_nummix, alignedNMix, alignedNMixAlloc, 
			alignedDim, alignedDimVar, first);
	}

	_alignedDimC = alignedDim;
	_alignedDimVarC = alignedDimVar;
	_alignedNMixC = alignedNMix;
	_alignedNMixAllocC = alignedNMixAlloc;
	this->_modelIn = true;
}



template <typename TS>
void GMMStatsEstimator_CUDA<TS>::setModelToBeUsed (unsigned int n) {

	GMMStatsEstimator_SSE<TS, float>::setModelToBeUsed(n);
	for (unsigned int i = 0; i < _GPUs.size(); i++) {
		_GPUs[n]->setModelToBeUsed(n);
	}
}



template <typename TS>
void GMMStatsEstimator_CUDA<TS>::alignVectors(float **vecs, unsigned int NSamples, unsigned int dim, 
											  float **vecsAligned, unsigned int &NframesAligned, 
											  unsigned int &alignedDim) 
{
	const unsigned int DIM_BLOCK = GMMStatsEstimator_GPU::DIM_BLOCK;
	const unsigned int FRAME_BLOCK = GMMStatsEstimator_GPU::FRAME_BLOCK;

	unsigned int aN = GMMStatsEstimator_GPU::alignUP<unsigned int> (NSamples, FRAME_BLOCK);	
	unsigned int aDim = static_cast<unsigned int> (DIM_BLOCK * ceil(dim/(float) DIM_BLOCK));

	if(NframesAligned * alignedDim < aN * aDim) {
		GMMStatsEstimator_GPU::freeOnHost ((void **) vecsAligned);
		GMMStatsEstimator_GPU::allocOnHost ((void **) vecsAligned, aN * aDim * sizeof(float));
		NframesAligned = aN;
		alignedDim = aDim;
	}

	float foo;
	for(unsigned int i = 0; i < NframesAligned/FRAME_BLOCK; i++)
	{
		for(unsigned int j = 0; j < alignedDim/DIM_BLOCK; j++) 
		{
			for(unsigned int k = 0; k < DIM_BLOCK; k++) 
			{
				for(unsigned int n = 0; n < FRAME_BLOCK; n++) 
				{
					if(((j * DIM_BLOCK + k) < dim) && ((i * FRAME_BLOCK + n) < NSamples))
						foo = vecs[i * FRAME_BLOCK + n][j * DIM_BLOCK + k];
					else
						foo = 0.0f;

					(*vecsAligned)[i * alignedDim * FRAME_BLOCK + j * DIM_BLOCK * FRAME_BLOCK + k * FRAME_BLOCK + n] = foo;
				}
			}
		}
	}
}



template <typename TS>
void GMMStatsEstimator_CUDA<TS>::compVecsLogLikeMT (float** vecs, unsigned int NSamples, unsigned int dim,
													float* outLogLikes)
{
#ifndef _EXCLUDE_SAFETY_CONDS_
	if(!this->_modelIn)
		throw logic_error("compVecsLogLikeMT(): None model inserted!");
#endif	

	unsigned int NGPUs = _GPUs.size();

	//if(!_cudaDeviceFound || NSamples < GMMStatsEstimator_GPU::FRAME_BLOCK || this->_nummix < 5) {	
	if(!_cudaDeviceFound || NSamples < NGPUs * GMMStatsEstimator_GPU::FRAME_BLOCK) {
		if (this->_verbosity > 3)
			std::cout <<"\n [#samples or #Gaussians to low -> switching to SSE] ";

		GMMStatsEstimator_SSE<TS, float>::compVecsLogLikeMT (vecs, NSamples, dim, outLogLikes);		
		return;
	}

	if (this->_fillMask)
		throw std::runtime_error("compVecsLogLikeMT(): mask on GPU not supported");

	updateOptionsOnGPU();
	
	unsigned int NS_1gpu = GMMStatsEstimator_GPU::alignDiv<unsigned int> (NSamples, NGPUs);

	unsigned int NSamplesGPU, NSamplesGPUprocess; 
	getMinNSamples2GPU(NSamplesGPU, NSamplesGPUprocess, NS_1gpu, dim, _alignedDimAuxC, false);

	unsigned int numBlocks = GMMStatsEstimator_GPU::alignDiv<unsigned int> (NS_1gpu, NSamplesGPU);	
	for(unsigned int n = 0; n < numBlocks; n++) 
	{
		for (unsigned int i = 0; i < NGPUs; i++) 
		{
			unsigned int offset = i * NS_1gpu + n * NSamplesGPU; 
			unsigned int NSamplesTmpGPU = ((n+1) * NSamplesGPU > NS_1gpu) ? NS_1gpu - n * NSamplesGPU : NSamplesGPU;
			NSamplesTmpGPU = (offset + NSamplesTmpGPU > NSamples) ? NSamples - offset : NSamplesTmpGPU;

			if (!this->_fSameDataProvided || _GPUs[i]->getNSamplesOnGPU() == 0 || numBlocks > 1) {
				alignVectors (vecs + offset, NSamplesTmpGPU, dim, &_paramCacheC, _alignedNSamplesC, _alignedDimC);
				_GPUs[i]->uploadParam(_paramCacheC, _alignedDimC, NSamplesTmpGPU, NSamplesGPUprocess, true);
			}
			_GPUs[i]->compLogLikes();

		}
		for (unsigned int i = 0; i < NGPUs; i++)
			_GPUs[i]->getLogLikes(outLogLikes + i * NS_1gpu + n * NSamplesGPU);
	}

	// NaNs?
	float sum = 0.0f;
	for (unsigned int n = 0; n < NSamples; n++)
		sum += outLogLikes[n];
	if (sum * 0.0f != 0.0f)
		throw std::runtime_error("compVecsLogLikeMT(): overall log-likelihood = NaN!");

	// reset flag
	this->_fSameDataProvided = false;
}



template <typename TS>
void GMMStatsEstimator_CUDA<TS>::accumulateStatsMT (float** vectors, unsigned int NSamples, unsigned int dim, 
													GMMStats<TS>& gmms) 
{	
	unsigned int NGPUs = _GPUs.size();

	//if(!_cudaDeviceFound || NSamples < GMMStatsEstimator_GPU::FRAME_BLOCK || !this->_modelIn || this->_nummix < 5) {
	if(!_cudaDeviceFound || NSamples < NGPUs * GMMStatsEstimator_GPU::FRAME_BLOCK || !this->_modelIn) {
		if (this->_verbosity > 3)
			std::cout <<"\n [#samples or #Gaussians to low -> switching to SSE] ";

 		GMMStatsEstimator_SSE<TS, float>::accumulateStatsMT (vectors, NSamples, dim, gmms);		
		return;
	}

	if(!this->_flagsSet)
		setAccFlags(gmms);

#ifndef _EXCLUDE_SAFETY_CONDS_
	checkProperAllocation(gmms);
	if(dim != gmms.getDim())
		throw logic_error("accumulateStatsMT(): Inconsistent feature & statistic dimensions!");		
	if(_mask != NULL && (_mask->N < NSamples || _mask->dim < this->_nummixAlloc))
		throw std::logic_error("accumulateStatsMT(): logLikeMask improperly allocated! \n\t");	
#endif	

	if (this->_fillMask)
		throw std::runtime_error("accumulateStatsMT(): mask on GPU not supported");

	updateOptionsOnGPU();
	reallocOutArrays();

	unsigned int NS_1gpu = GMMStatsEstimator_GPU::alignDiv<unsigned int> (NSamples, NGPUs);

	unsigned int NSamplesGPU, NSamplesGPUprocess; 
	getMinNSamples2GPU(NSamplesGPU, NSamplesGPUprocess, NS_1gpu, dim, _alignedDimAuxC, true);
	//NSamplesGPU = 21235; // LLLLLLLLLLLLLLLLLL
	//NSamplesGPUprocess = 7623; // LLLLLLLLLLLLLLLLLL

	float totLL;
	unsigned int numBlocks = GMMStatsEstimator_GPU::alignDiv<unsigned int> (NS_1gpu, NSamplesGPU);
	for(unsigned int n = 0; n < numBlocks; n++) 
	{
		for (unsigned int i = 0; i < NGPUs; i++) 
		{
			unsigned int offset = i * NS_1gpu + n * NSamplesGPU; 
			unsigned int NSamplesTmpGPU = ((n+1) * NSamplesGPU > NS_1gpu) ? NS_1gpu - n * NSamplesGPU : NSamplesGPU;
			NSamplesTmpGPU = (offset + NSamplesTmpGPU > NSamples) ? NSamples - offset : NSamplesTmpGPU;

			if (!this->_fSameDataProvided || _GPUs[i]->getNSamplesOnGPU() == 0 || numBlocks > 1) {
				alignVectors (vectors + offset, NSamplesTmpGPU, dim, &_paramCacheC, _alignedNSamplesC, _alignedDimC);
				_GPUs[i]->uploadParam(_paramCacheC, _alignedDimC, NSamplesTmpGPU, NSamplesGPUprocess, false);
			}
			_GPUs[i]->compAccStats(this->_fFullVar, this->_fAux);
			gmms._totalAccSamples += NSamplesTmpGPU;
		}
		for (unsigned int i = 0; i < NGPUs; i++) {

			// NOTE: _meanBuffC = meanStats, _ivarBuffC = varStats, _gConstBuffC = mixProbs, _alignedDimAuxC = auxStats		
			_GPUs[i]->getAccStats(totLL, _meanBuffC, _ivarBuffC, _gConstBuffC, _auxStatsBuffC, this->_fFullVar);
			if (totLL * 0.0f != 0.0f)
				throw std::runtime_error("accumulateStatsMT(): overall log-likelihood = NaN!");
			export2GMMStats(gmms, _meanBuffC, _ivarBuffC, _gConstBuffC, _auxStatsBuffC, this->_dim, this->_nummix);
			gmms._totLogLike += totLL;
		}		
	}	

	//if(useMask) {
	//	reAllocGammaAndLogLikeBuffers(_alignedNSamplesC, _alignedNMixAllocC);
	//	_GPUs[i]->getGammasAndLL(_gammasBuffC, _logLikesBuffC, _alignedNSamplesC, _alignedNMixC);
	//	fillMaskwithGammasAndLogLikes(*_mask, NSamples);
	//}

#ifdef _DEBUG
	// ----------------------------------------------------------------------------------
	// DEBUG ----------------------------------------------------------------------------
	//static logLikeMask gammasAndLL;
	//if (gammasAndLL.N < NSamples) {
	//	gammasAndLL.N = NSamples;
	//	gammasAndLL.dim = this->_nummix;
	//	try {
	//		gammasAndLL.logLikes = new float [NSamples]; // NOT DELETED
	//		gammasAndLL.gammas = new float* [_alignedNSamplesC]; // NOT DELETED
	//		gammasAndLL.gammas[0] = new float [_alignedNSamplesC * this->_nummix]; // NOT DELETED
	//		for (unsigned int n = 1; n < _alignedNSamplesC; n++)
	//			gammasAndLL.gammas[n] = &gammasAndLL.gammas[0][this->_nummix * n];
	//	}
	//	catch(bad_alloc&) {
	//		throw bad_alloc("accumulateStatsMT(): Not enough memory for gammas");
	//	}
	//}
	//fillMaskwithGammasAndLogLikes(gammasAndLL, NSamples);
	//
	//std::ofstream file("D:/WORK/experimenty/ruozne/73_MLF_GMM/_TEST/TMP_MASK.txt");	
	//file << gammasAndLL.N << " " << gammasAndLL.dim << std::endl;
	//for(unsigned int i = 0; i < gammasAndLL.N; i++) {
	//	file << gammasAndLL.logLikes[i] << " ";
	//}
	//file << std::endl << std::endl;

	//for(unsigned int i = 0; i < _alignedNSamplesC; i++) {
	//	for(unsigned int j = 0; j < this->_nummix; j++) {
	//		file << gammasAndLL.gammas[i][j] << " ";
	//	}
	//	file << std::endl;
	//}
	//file << std::endl;
	//file.close();	
	// DEBUG ----------------------------------------------------------------------------
	// ----------------------------------------------------------------------------------
#endif
	
	// gmms.save("__tmp_stats.txt", true); // LLLLLLLLLLLLLL

	// reset flag
	this->_fSameDataProvided = false;
}

template <typename TS>
void GMMStatsEstimator_CUDA<TS>::getMinNSamples2GPU (unsigned int &minNSamplesGPU, unsigned int &minNSamplesGPUprocess,
													 unsigned int NSamples, unsigned int dim, unsigned int aligDimAux,
													 bool allocStats)
{
	static bool firstRun = true;
	static unsigned int NSamplesGPU = std::numeric_limits<unsigned int>::max();
	static unsigned int NSamplesGPUprocess = std::numeric_limits<unsigned int>::max();

	if (firstRun) {
		unsigned int m1, m2; 	
		for (unsigned int i = 0; i < _GPUs.size(); i++) {
			_GPUs[i]->getNSamples2GPU (m1, m2, NSamples, dim, this->_NAccBlocks_CUDA, aligDimAux, allocStats, this->_memoryBuffDataGB_GPU);
			if (m1 < NSamplesGPU) {
				NSamplesGPU = m1;
				NSamplesGPUprocess = m2;
			}
		}
		if (this->_verbosity) {
			unsigned int numBlocks = GMMStatsEstimator_GPU::alignDiv<unsigned int> (NSamples, NSamplesGPU);
			std::cout <<"\n     [GPU]" << std::endl;
			std::cout <<"\n        #samples on GPU = " << NSamplesGPU << ", uploaded to GPU in " << numBlocks << " block(s)";
			std::cout <<"\n\t                            ";
		}
		firstRun = false;
	}
	minNSamplesGPU = NSamplesGPU;
	minNSamplesGPUprocess = NSamplesGPUprocess;
}



template <typename TS>
void GMMStatsEstimator_CUDA<TS>::reallocOutArrays()
{
	// if necessary realloc '_ivarBuffC' in order to retrieve the accumulated vars from GPU (see below)
	if(this->_fFullVar && !_ivarFullAlloc) {		
		unsigned int varDim = (_alignedDimC * (_alignedDimC + GMMStatsEstimator_GPU::DIM_BLOCK)) / 2;
		GMMStatsEstimator_GPU::freeOnHost((void **) &_ivarBuffC);
		GMMStatsEstimator_GPU::allocOnHost((void **) &_ivarBuffC, varDim * _alignedNMixAllocC * sizeof(float));
		_ivarFullAlloc = true;
	}

	if(this->_fAux) {
		unsigned int adA = _alignedNMixAllocC * (_alignedDimC + 2); // store aux2 and aux3 in the last two dimensions of auxiliary stats
		if(_alignedDimAuxC != adA) {			
			GMMStatsEstimator_GPU::freeOnHost((void **) &_auxStatsBuffC);
			GMMStatsEstimator_GPU::allocOnHost((void **) &_auxStatsBuffC, adA * sizeof(float));
			_alignedDimAuxC = adA;
		}
	}
}



//template <typename TS>
//void GMMStatsEstimator_CUDA<TS>::fillMaskwithGammasAndLogLikes (GMMStatsEstimator<TS, float, float>::logLikeMask& mask, unsigned int NSamples) 
//{
//	//reAllocGammaAndLogLikeBuffers(_alignedNSamplesC, _alignedNMixAllocC);
//	//cuda_getGammasAndLL (_gammasBuffC, _logLikesBuffC, _alignedNSamplesC, _alignedNMixC);
//	memcpy(mask.logLikes, _logLikesBuffC, NSamples * sizeof(float));
//
//	unsigned int sBlock = GMMStatsEstimator_GPU::FRAME_BLOCK / 2;	
//	for (unsigned int n = 0; n < _alignedNSamplesC / sBlock; n++) {
//		for (unsigned int m = 0; m < this->_nummix; m++) {
//			for (unsigned int x = 0; x < sBlock; x++) {
//				if (sBlock * n + x >= NSamples)
//					break;
//				if (_logLikesBuffC[x] > this->_minLogLike)
//					mask.gammas[sBlock * n + x][m] = exp(_gammasBuffC[sBlock * _alignedNMixC * n + sBlock * m + x] - _logLikesBuffC[sBlock * n + x]);
//				else
//					mask.gammas[sBlock * n + x][m] = 0.0f;
//			}
//		}
//	}		
//}



template <typename TS>
void GMMStatsEstimator_CUDA<TS>::export2GMMStats(GMMStats<TS>& gmms, 
												 float *meanS, float *varS, float *mixProbS, float *auxS,
												 unsigned int dim, unsigned int nummix) 
{

#ifndef _EXCLUDE_SAFETY_CONDS_
	if(!this->_modelIn)
		throw logic_error("export2GMMStats(): None model inserted!");
	if(dim != this->_dim || nummix > this->_nummixAlloc)
		throw logic_error("export2GMMStats(): Inconsistent dimensionality or number of mixtures!");	
#endif	

	unsigned int DIM_BLOCK = GMMStatsEstimator_GPU::DIM_BLOCK;
	unsigned int varDim = _alignedDimC * (_alignedDimC + DIM_BLOCK)/2;

	for (unsigned int i = 0; i < nummix; i++) {	
		gmms._ms[i].mixProb += (TS) mixProbS[i];

		if(this->_fMixProbOnly)
			continue;

		unsigned int shift = 0, aShift = 0;
		for (unsigned int d = 0; d < this->_dim; d++) 
		{
			if(this->_fAux)
				gmms._ms[i].aux[d] += (TS) auxS[i * (_alignedDimC + 2) + d];							

			if(this->_fMean)
				gmms._ms[i].mean[d] += (TS) meanS[i * _alignedDimC + d];

			if(this->_fVar) {
				if(!this->_fFullVar) {
					shift += d;
					if(gmms._allFV)
						gmms._ms[i].var[d * gmms._dim + d - shift] += (TS) varS[i * _alignedDimC + d];
					else
						gmms._ms[i].var[d] += (TS) varS[i * _alignedDimC + d];
				}
				else {
					float *mixvar = &_ivarBuffC[i * varDim];
					unsigned int off = aShift + d % DIM_BLOCK;

					for(unsigned int dd = 0; dd < this->_dim - d; dd++)						
						gmms._ms[i].var[shift + dd] += (TS) mixvar[off + dd];

					shift += this->_dim - d;
					aShift += _alignedDimC - (d/DIM_BLOCK)*DIM_BLOCK;
				}
			}
		} // d < dim
		if(this->_fAux) {
			gmms._ms[i].aux2 += (TS) auxS[i * (_alignedDimC + 2) + _alignedDimC];
			gmms._ms[i].aux3 += (TS) auxS[i * (_alignedDimC + 2) + _alignedDimC + 1];
		}
	}
}



template <typename TS>
GMMStatsEstimator<TS, float, float>* GMMStatsEstimator_CUDA<TS>::getNewInstance() {
	return new GMMStatsEstimator_CUDA<TS>;
}



template <typename TS>
void GMMStatsEstimator_CUDA<TS>::reAllocGammaAndLogLikeBuffers (unsigned int alignedNSamples, unsigned int alignedNMix) {

#ifndef _EXCLUDE_SAFETY_CONDS_
	if(!this->_modelIn)
		throw logic_error("reAllocGammaAndLogLikeBuffers(): None model inserted!");
#endif

	static unsigned int maxAlignedNsamples = alignedNSamples;
	static unsigned int maxAlignedMixs = alignedNMix;

	if(maxAlignedNsamples * maxAlignedMixs < alignedNSamples * alignedNMix || maxAlignedNsamples < alignedNSamples) {
		maxAlignedNsamples = alignedNSamples;
		maxAlignedMixs = alignedNMix;
	}
	else if(_logLikesBuffC != NULL)
		return;

	if(_gammasBuffC != NULL)
		GMMStatsEstimator_GPU::freeOnHost((void **) &_gammasBuffC);			

	if(_logLikesBuffC != NULL)
		GMMStatsEstimator_GPU::freeOnHost((void **) &_logLikesBuffC);

	GMMStatsEstimator_GPU::allocOnHost((void **) &_gammasBuffC, alignedNSamples * alignedNMix * sizeof(float));
	GMMStatsEstimator_GPU::allocOnHost((void **) &_logLikesBuffC, alignedNSamples * sizeof(float));
}

#endif
