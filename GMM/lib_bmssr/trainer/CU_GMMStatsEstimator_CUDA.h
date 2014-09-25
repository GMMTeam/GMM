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

#ifndef _CU_STATS_EST_CUDA_
#define _CU_STATS_EST_CUDA_

#ifndef _CUDA
#	define float4 float
#else
#	include "vector_types.h"
#endif



class GMMStatsEstimator_GPU {
public:

	static const unsigned int DIM_BLOCK;      // feature vector dimension aligned to a natural multiple of DIM_BLOCK
	static const unsigned int FRAME_BLOCK;    // number of frames handled in one thread (minimal number of frames in order to use CUDA)
	static const unsigned int GAUSS_BLOCK;    // number of gaussians computed at once (= number of threads in one block) - note: DIM_BLOCK*FRAME_BLOCK == BLOCK
	static const unsigned int DATA_BLOCK;
	static const unsigned int MAX_GRID_SIZE;  // max dimension of a grid
	static const unsigned int RESET_BLOCK_SIZE;		
	static const unsigned int MODELMEM_INCREASE;
	static const unsigned int MIN_UNUSED_MEM_GPU; // given in MB; how much memory should be left free on GPU
	static const unsigned int MIN_FRAMES_ON_GPU; // given in MB; how much memory should be left free on GPU

	struct OPTIONS {
		float minLogLike;
		float minGamma;
		unsigned int frames_acc_blocks;	// #blocks to be accumulated separately (blocks summed up at the end);
										// set only ONCE and before the model will be uploaded
		int verbosity;
		bool throw_errors;
	} _opt;
	
	struct model {
		float4* d_means;
		float4* d_ivars;
		float* d_Gconsts;
		
		unsigned int dim;      // aligned to DIM_BLOCK		
		unsigned int dimVar;   // aligned to DIM_BLOCK and only upper triangular stored

		unsigned int nummix;            // last inserted model mixture count - aligned to GAUSS_BLOCK
		unsigned int nummix_unaligned;  // last inserted model mixture count - unaligned		
		unsigned int nummix_allocS;     // aligned to GAUSS_BLOCK

		bool d_modelAlloc;
	};

	struct param {
		float* d_vecs;
		
		unsigned int dim;               // aligned to DIM_BLOCK		
		unsigned int Nframes;           // last inserted param vector count - aligned to FRAME_BLOCK
		unsigned int Nframes_unaligned; // last inserted param vector count - not aligned
		unsigned int Nframes_allocS;    // maximum number of frames with dim 'dim' that can be stored in d_vecs 
										// without any need to reallocate the array 'd_vecs' - aligned to FRAME_BLOCK
		unsigned int Nframes_processed; // how many frames will be accumulated at once - aligned to FRAME_BLOCK; based on maximum size of GRID
		
		bool d_paramAlloc;
	};

	struct stats {
		float* d_meanStats;
		unsigned int statMeanSize;

		float* d_varStats;
		unsigned int statVarSize;
		
		bool accFullVar;

		float* d_mixProb;
		unsigned int statProbSize;		

		float* d_auxStats;
		unsigned int statAuxSize;
		
		bool d_statsAlloc;
	};

	struct likes {
		float4* d_gammas;
		unsigned int gammasSize; // size = nummix * Nframes (both aligned)

		float4* d_ll;            // LogLike
		float4* d_ll_begin;		// points to the begining of d_ll
		unsigned int llSize;     // size = Nframes_processed (aligned) OR ll_storeAll => size = Nframes (aligned) 
		bool ll_storeAll;

		float4* d_aux_ll;	//normalisation LL to compute aux statistics

		float* d_totll;  // overall LogLike
	};

	GMMStatsEstimator_GPU (int GPU_id);
	~GMMStatsEstimator_GPU();

	
	// several models can be uploaded to the GPU -> for 2nd/3rd/.. set first = false
	void uploadModel (float* meanBuffC, float* ivarBuffC, float* gConstBuffC, 
					  unsigned int NMix, unsigned int alignedNMix, unsigned int alignedNMixAlloc, 
					  unsigned int alignedDim, unsigned int alignedDimVar, bool first = true);
	
	// input parameter 'NSamples' is the number of provided frames;
	// the fcn returns the count of frames that will fit into the GPU memory ('NSamplesGPU')
	// and also the number of frames that can be processed at once ('NSamplesGPUprocess')
	void getNSamples2GPU (unsigned int &NSamplesGPU, unsigned int &NSamplesGPUprocess,
						  unsigned int NSamples, unsigned int dim, 
						  unsigned int NAccBlocks, unsigned int aligDimAux,
						  bool allocStats, float maxGPUMem2UseGB);

	// note: Nframes, dim -> not aligned!
	// Nframes2processAtOnce -> how many frames will be in one call of a kernel function -> depends 
	// on the maximum number of blocks in GRID and available GPU memory
	// requestLogLikeForEachFrame -> whether likelihood for each frame should be stored separately
	void uploadParam (float* alignedVecs, unsigned int alignedDim, unsigned int Nframes, 
					  unsigned int Nframes2processAtOnce, bool requestLogLikeForEachFrame);

	// set model to be used if several models were uploaded
	void setModelToBeUsed (unsigned int n);

	// logLikes & statistics
	void compAccStats (bool fullVarAcc, bool auxStats);

	void getAccStats (float &totLL, float* meanStat, float* varStat, float* mixProb, float* auxStats, 
					  bool fullVarAcc);

	// logLikes only
	void compLogLikes();
	void getLogLikes (float* outLogLikes);

	// Assuming that gammas & logLikes were already computed using compLogLikeAndGammas()
	// note: size of gammas   = NSamples * number_of_mixtures; size of loglikes = NSamples
	void getGammasAndLL (float *gammas, float *loglikes,
						 unsigned int alignedNframes, unsigned int alignedNmix);
	
	void printInfo();
	unsigned int getSharedMemPerBlock();
	unsigned int getNSamplesOnGPU();
	
	// -- STATIC FUNCTIONs ---------------------------
	static int deviceCount();

	template <class T>
	static T alignUP (T x, T block2align);
	template <class T>
	static T alignDiv (T x, T block2align);

	// use to boost the speed of GPU->CPU, CPU->GPU transfer
	static void allocOnHost (void **data, unsigned int size);
	static void freeOnHost (void **data);

	static bool checkError (int err, int verbosity = 0, bool throw_errors = true);

protected:

	stats *_stats;
	model *_model;
	param *_prm;
	likes *_likes;

	// store several models
	model **_gmodels;
	unsigned int _gmodelsSize;      // number of models present
	unsigned int _gmodelsAllocSize; // allocation size


	void accumulateStats (unsigned int shiftFrames, unsigned int NSamplesTmp, 
						  bool accAuxStats, bool fullVarAcc);
	void compLogLikeAndGammas (unsigned int shiftFrames, unsigned int alignedNframes);


	void allocAccStats (unsigned int alignedNMix, unsigned int alignedDim, 
						bool fullVarAcc, bool allocAuxStats);	
	void allocModel (unsigned int alignedNMix, unsigned int alignedDim, unsigned int alignedDimVar);			
	void allocParam (unsigned int alignedDim, unsigned int NframesAligned);	
	void allocGammas (unsigned int alignedNframes, unsigned int nummix_allocS,
					  unsigned int ext_ll_size);


	void sumAccStats(bool auxStats);

	void resetAccStats();
	void reallocGroups (unsigned int NGroups);	


	void eraseModel();
	void eraseGroups (bool release);
	void eraseParam();
	void eraseGammas();	
	void eraseAccStats();


private:
	unsigned int _sharedMemPerBlock;
	
	void *_pContextGPU;
	int _GPU_id;

	// prohibited
	GMMStatsEstimator_GPU (const GMMStatsEstimator_GPU&);
	GMMStatsEstimator_GPU& operator= (const GMMStatsEstimator_GPU&);

	// --- CONTEXT -- multiple GPUs
	static void createContext (int GPU_id, void **pctx);
	static void destroyContext (void **pctx);
	void pushGPU();
	void popGPU();
};

#endif
