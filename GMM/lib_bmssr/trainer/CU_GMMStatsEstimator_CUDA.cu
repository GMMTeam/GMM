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

#include "trainer/CU_GMMStatsEstimator_CUDA.h"

#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <sstream>
#include <stdexcept>

// #define GPU_SAFE_MODE // turn on synchronization of kernel executions - slowing down the execution
// #define GPU_CUDA_TIMER // trace execution times of particular kernels
#define GPU_CUDA_CONTEXT_AVAIL

#ifdef GPU_CUDA_CONTEXT_AVAIL
#    include <cuda.h>
#else
#    include <cuda_runtime.h>
#endif

#ifdef GPU_CUDA_TIMER
    float GPU_kernel_timer1 = 0;
    float GPU_kernel_timer2 = 0;
    float GPU_kernel_timer3 = 0;
    cudaEvent_t start1, start2, start3, stop1, stop2, stop3;
#endif

const unsigned int GMMStatsEstimator_GPU::DIM_BLOCK  = 4;
const unsigned int GMMStatsEstimator_GPU::FRAME_BLOCK = 8;
const unsigned int GMMStatsEstimator_GPU::GAUSS_BLOCK = 32;
const unsigned int GMMStatsEstimator_GPU::DATA_BLOCK = DIM_BLOCK * FRAME_BLOCK;
const unsigned int GMMStatsEstimator_GPU::RESET_BLOCK_SIZE = 128;
const unsigned int GMMStatsEstimator_GPU::MAX_GRID_SIZE = (32*1024-1);
const unsigned int GMMStatsEstimator_GPU::MODELMEM_INCREASE = 100;
const unsigned int GMMStatsEstimator_GPU::MIN_UNUSED_MEM_GPU = 50;
const unsigned int GMMStatsEstimator_GPU::MIN_FRAMES_ON_GPU = 100 * DIM_BLOCK * FRAME_BLOCK;

// include kernels
#include "trainer/CU_GMMStatsEstimator_CUDAkernels.cuh"

template unsigned int GMMStatsEstimator_GPU::alignUP (unsigned int x, unsigned int block2align);
template unsigned int GMMStatsEstimator_GPU::alignDiv (unsigned int x, unsigned int block2align);

GMMStatsEstimator_GPU::GMMStatsEstimator_GPU(int GPU_id) 
: _stats(NULL),
_model(NULL),
_prm(NULL),
_likes(NULL),
_gmodels(NULL),
_gmodelsSize(0),
_gmodelsAllocSize(0),
_sharedMemPerBlock(0),
_pContextGPU(NULL),
_GPU_id(-1)
{
    _opt.minLogLike = -1e+20f;
    _opt.minGamma = 1e-4f;
    _opt.frames_acc_blocks = 8;
    _opt.verbosity = 0;
    _opt.throw_errors = true;

    _model = new model;
    _model->d_means = NULL;
    _model->d_ivars = NULL;
    _model->d_Gconsts = NULL;
    _model->dim = 0;
    _model->dimVar = 0;
    _model->nummix = 0;
    _model->nummix_unaligned = 0;
    _model->nummix_allocS = 0;
    _model->d_modelAlloc = false;
    
    _gmodels = new model* [MODELMEM_INCREASE];
    _gmodelsAllocSize = MODELMEM_INCREASE;
    _gmodels[0] = _model;
    _gmodelsSize = 1;
    
    _prm = new param;
    _prm->d_vecs = NULL;
    _prm->dim = 0;
    _prm->Nframes_unaligned = 0;
    _prm->Nframes = 0;
    _prm->Nframes_allocS = 0;
    _prm->d_paramAlloc = false;

    _stats = new stats;
    _stats->d_meanStats = NULL;
    _stats->statMeanSize = 0;
    _stats->d_varStats = NULL;
    _stats->statProbSize = 0;
    _stats->accFullVar = false;
    _stats->d_mixProb = NULL;
    _stats->statVarSize = 0;
    _stats->d_auxStats = NULL;
    _stats->statAuxSize = 0;        
    _stats->d_statsAlloc = false;

    _likes = new likes;
    _likes->d_gammas = NULL;
    _likes->gammasSize = 0;
    _likes->d_ll = NULL;
    _likes->d_ll_begin = NULL;
    _likes->llSize = 0;
    _likes->ll_storeAll = false;
    _likes->d_totll = NULL;
    _likes->d_aux_ll = NULL;
    
    int devN = deviceCount();    
    if (devN > GPU_id) {
        createContext(GPU_id, &_pContextGPU);

        cudaSetDevice(GPU_id);
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, GPU_id);
        _sharedMemPerBlock = deviceProp.sharedMemPerBlock;
        _GPU_id = GPU_id;
    }
}



GMMStatsEstimator_GPU::~GMMStatsEstimator_GPU() 
{    
    pushGPU();
    eraseGroups(false);

    eraseModel();
    eraseParam();
    eraseGammas();

    delete _model;
    delete _prm;
    delete _stats;
    delete _likes;

    destroyContext(&_pContextGPU);
}



int GMMStatsEstimator_GPU::deviceCount()
{
    int deviceCount = 0;
    checkError( cudaGetDeviceCount(&deviceCount) );
    return deviceCount;
}


unsigned int GMMStatsEstimator_GPU::getSharedMemPerBlock() {
    return _sharedMemPerBlock;
}



void GMMStatsEstimator_GPU::printInfo() {

    printf("\nCUDA GPU Estimator - version 2.0 (1.2.2013)\n");
    
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    if(deviceCount > 0)
        printf("Number of CUDA compatible device(s) found: %d\n", deviceCount);
    else {
        printf("No CUDA compatible device found.\n");
        return;
    }
    
    if(deviceCount < _GPU_id) {
        printf("Device with ID %d was not detected.\n", _GPU_id);
        return;
    }

    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, _GPU_id);
    printf("Device ID: %d\n", _GPU_id);
    printf("Device name: %s\n", deviceProp.name);
    printf("GPU memory: %d MB\n", deviceProp.totalGlobalMem/1024/1024);
    printf("Core clock: %d MHz\n", deviceProp.clockRate/1000);
    if(strstr(deviceProp.name, "CPU") != NULL) {
        printf("Only GPU emulation available.\n");
    }
    //cudaSetDevice(deviceID);
}


void GMMStatsEstimator_GPU::allocOnHost(void **data, unsigned int size) {    
    checkError(  cudaMallocHost(data, size) );
    //*data = (void *) new char [size];    
}



void GMMStatsEstimator_GPU::freeOnHost(void **data) {
    checkError( cudaFreeHost(*data) );    
    *data = NULL;
    //delete [] *data;    
}




void GMMStatsEstimator_GPU::reallocGroups(unsigned int NGroups) {    

    if (NGroups <= _gmodelsAllocSize) 
        return;

    unsigned int new_size = NGroups + MODELMEM_INCREASE;

    model **tmp_gmodels;
    tmp_gmodels = new model* [new_size];
    memcpy(tmp_gmodels, _gmodels, _gmodelsAllocSize * sizeof(model*));

    delete [] _gmodels;
    _gmodels = tmp_gmodels;

    _gmodelsAllocSize = new_size;
}



void GMMStatsEstimator_GPU::eraseGroups(bool release) 
{
    if (_gmodelsAllocSize < 1) {
        _gmodelsSize = 0;
        return;
    }

    // leave the first model in
    if (_gmodelsSize > 0) 
        _model = _gmodels[0];

    for(unsigned int n = 1; n < _gmodelsSize; n++) 
    {        
        checkError( cudaFree(_gmodels[n]->d_means) );
        checkError( cudaFree(_gmodels[n]->d_ivars) );
        checkError( cudaFree(_gmodels[n]->d_Gconsts) );

        delete _gmodels[n];
        _gmodels[n] = NULL;
    }
    
    if (!release) 
    {
        delete [] _gmodels;
        _gmodels = NULL;
        _gmodelsAllocSize = 0;
    }

    _gmodelsSize = 1;
}



void GMMStatsEstimator_GPU::uploadModel (float* meanBuffC, float* ivarBuffC, float* gConstBuffC, 
                                         unsigned int NMix, unsigned int alignedNMix, unsigned int alignedNMixAlloc, 
                                         unsigned int alignedDim, unsigned int alignedDimVar, bool first)
{
    pushGPU();

    if(first)
        eraseGroups(true);
    else {
        _model = new model;
        _gmodelsSize++;
        reallocGroups(_gmodelsSize);
        _gmodels[_gmodelsSize - 1] = _model;
    }

    if (!_model->d_modelAlloc ||
        alignedNMixAlloc > _model->nummix_allocS || 
        alignedDim != _model->dim || 
        alignedDimVar != _model->dimVar )
    {                    
        allocModel(alignedNMixAlloc, alignedDim, alignedDimVar);
    }    

    checkError( cudaMemcpy(_model->d_means, meanBuffC, alignedNMixAlloc * alignedDim * sizeof(float), cudaMemcpyHostToDevice) );
    checkError( cudaMemcpy(_model->d_ivars, ivarBuffC, alignedNMixAlloc * alignedDimVar * sizeof(float), cudaMemcpyHostToDevice) );
    checkError( cudaMemcpy(_model->d_Gconsts, gConstBuffC, alignedNMixAlloc * sizeof(float), cudaMemcpyHostToDevice) );
    
    _model->dim = alignedDim;
    _model->dimVar = alignedDimVar;
    _model->nummix = alignedNMix;
    _model->nummix_unaligned = NMix;
    _model->nummix_allocS = alignedNMixAlloc;    

    popGPU();
}



void GMMStatsEstimator_GPU::eraseModel() {    

    checkError( cudaFree(_model->d_means) );
    _model->d_means = NULL;

    checkError( cudaFree(_model->d_ivars) );
    _model->d_ivars = NULL;

    checkError( cudaFree(_model->d_Gconsts) );
    _model->d_Gconsts = NULL;

    _model->nummix_allocS = 0;
    _model->nummix = 0;
    _model->dim = 0;

    _model->d_modelAlloc = false;
#ifdef GPU_CUDA_TIMER
    //init events for GPU timers
    cudaEventDestroy(start1);
    cudaEventDestroy(start2);
    cudaEventDestroy(start3);
    cudaEventDestroy(stop1);
    cudaEventDestroy(stop2);
    cudaEventDestroy(stop3);

    //print total elapsed times
    printf("GPU elapsed time in kernel 1 : %6.3f\n", GPU_kernel_timer1);
    printf("GPU elapsed time in kernel 2 : %6.3f\n", GPU_kernel_timer2);
    printf("GPU elapsed time in kernel 3 : %6.3f\n", GPU_kernel_timer3);
    printf("Total GPU time in all kernels: %6.3f\n", GPU_kernel_timer1+GPU_kernel_timer2+GPU_kernel_timer3);
#endif    
}



void GMMStatsEstimator_GPU::setModelToBeUsed(unsigned int n) 
{
#ifndef _EXCLUDE_SAFETY_CONDS_
    if (n > _gmodelsSize)
        throw std::runtime_error("setModelToBeUsed(): requested model out of bounds!");
#endif

    pushGPU();

    _model = _gmodels[n];

    popGPU();
}



void GMMStatsEstimator_GPU::allocModel(unsigned int alignedNMix, unsigned int alignedDim, unsigned int alignedDimVar) 
{
    eraseModel();

    checkError( cudaMalloc((void **)& _model->d_means, alignedNMix * alignedDim * sizeof(float)) );
    checkError( cudaMalloc((void **)& _model->d_ivars, alignedNMix * alignedDimVar * sizeof(float)) );
    checkError( cudaMalloc((void **)& _model->d_Gconsts, alignedNMix * sizeof(float)) );    

    _model->nummix_allocS = alignedNMix;
    _model->dim = alignedDim;
    _model->dimVar = alignedDimVar;

    _model->d_modelAlloc = true;
#ifdef GPU_CUDA_TIMER
    //init events for GPU timers
    cudaEventCreate(&start1);
    cudaEventCreate(&start2);
    cudaEventCreate(&start3);
    cudaEventCreate(&stop1);
    cudaEventCreate(&stop2);
    cudaEventCreate(&stop3);
    GPU_kernel_timer1 = 0.0f;
    GPU_kernel_timer2 = 0.0f;
    GPU_kernel_timer3 = 0.0f;
#endif
}



void GMMStatsEstimator_GPU::eraseParam() 
{
    checkError( cudaFree(_prm->d_vecs) );
    
    _prm->d_vecs = NULL;

    _prm->Nframes_allocS = 0;
    _prm->Nframes = 0;
    _prm->dim = 0;

    _prm->d_paramAlloc = false;
}



void GMMStatsEstimator_GPU::allocParam(unsigned int alignedDim, unsigned int NframesAligned) 
{
    eraseParam();

    checkError( cudaMalloc((void **)& _prm->d_vecs, NframesAligned * alignedDim * sizeof(float)) );

    _prm->Nframes_allocS = NframesAligned;
    _prm->dim = alignedDim;

    _prm->d_paramAlloc = true;
}



void GMMStatsEstimator_GPU::uploadParam (float* alignedVecs, unsigned int alignedDim, 
                                         unsigned int Nframes, unsigned int Nframes2processAtOnce,
                                         bool requestLogLikeForEachFrame) 
{
    pushGPU();

    unsigned int NframesAligned = alignUP<unsigned int> (Nframes, FRAME_BLOCK);
    unsigned int NframesAlignedPOnce = alignUP<unsigned int> (Nframes2processAtOnce, FRAME_BLOCK);    

#ifndef _EXCLUDE_SAFETY_CONDS_
    if (!_model->d_modelAlloc)
        throw std::runtime_error("uploadParam(): insert model first!");
    if (_model->dim != alignedDim)
        throw std::runtime_error("uploadParam(): dimension mismatch!");
#endif

    // if allocated but too small => realloc
    if (!_prm->d_paramAlloc || 
        NframesAligned > _prm->Nframes_allocS || 
        alignedDim != _prm->dim)
    {
        allocParam(alignedDim, NframesAligned);
    }
            
    // if likelihood for each frame should be kept
    unsigned int ext_ll_size = (unsigned int) requestLogLikeForEachFrame * (NframesAligned - NframesAlignedPOnce);
    if ((NframesAlignedPOnce + ext_ll_size) > _likes->llSize || 
        NframesAlignedPOnce * _model->nummix_allocS > _likes->gammasSize)
    {        
        allocGammas(NframesAlignedPOnce, _model->nummix_allocS, ext_ll_size);
        _likes->ll_storeAll = requestLogLikeForEachFrame;
    }

    checkError( cudaMemcpy(_prm->d_vecs, alignedVecs, NframesAligned * alignedDim * sizeof(float), cudaMemcpyHostToDevice) );
    
    _prm->Nframes_processed = NframesAlignedPOnce;
    _prm->Nframes_unaligned = Nframes;
    _prm->Nframes = NframesAligned;
    popGPU();
}



unsigned int GMMStatsEstimator_GPU::getNSamplesOnGPU() {
    return _prm->Nframes_unaligned;
}



void GMMStatsEstimator_GPU::eraseGammas() {

    checkError( cudaFree(_likes->d_gammas) );

    _likes->d_gammas = NULL;
    _likes->gammasSize = 0;

    checkError( cudaFree(_likes->d_ll) );
    checkError( cudaFree(_likes->d_totll) );
    checkError( cudaFree(_likes->d_aux_ll) );
    
    _likes->d_totll = NULL;    
    _likes->d_ll = NULL;
    _likes->d_ll_begin = NULL;
    
    _likes->d_aux_ll = NULL;

    _likes->llSize = 0;
    _likes->ll_storeAll = false;
}



void GMMStatsEstimator_GPU::allocGammas(unsigned int alignedNframes, unsigned int nummix_allocS, unsigned int ext_ll_size) {
    
    eraseGammas();

    unsigned int alignedSize = alignedNframes * nummix_allocS;

    checkError( cudaMalloc((void **)& _likes->d_gammas, alignedSize * sizeof(float)) );
    checkError( cudaMalloc((void **)& _likes->d_ll, (alignedNframes + ext_ll_size) * sizeof(float)) );
    checkError( cudaMalloc((void **)& _likes->d_totll, sizeof(float)) );
    checkError( cudaMalloc((void **)& _likes->d_aux_ll, (alignedNframes) * sizeof(float)) );

    _likes->d_ll_begin = _likes->d_ll;
    _likes->gammasSize = alignedSize;
    _likes->llSize = alignedNframes + ext_ll_size;
}



void GMMStatsEstimator_GPU::allocAccStats (unsigned int alignedNMix, unsigned int alignedDim, 
                                           bool fullVarAcc, bool allocAuxStats) 
{
    eraseAccStats();
    
    if (allocAuxStats)
        checkError( cudaMalloc((void **)& _stats->d_auxStats, _opt.frames_acc_blocks * alignedNMix * (alignedDim + 2) * sizeof(float)) );

    checkError( cudaMalloc((void **)& _stats->d_meanStats, _opt.frames_acc_blocks * alignedNMix * alignedDim * sizeof(float)) );

    unsigned int varDim = alignedDim;
    if(fullVarAcc) {
        varDim = (alignedDim * (alignedDim + DIM_BLOCK)) / 2;    // the diagonal part has to be alligned according to DIM_BLOCK
        _stats->accFullVar = true;
    }

    checkError( cudaMalloc((void **)& _stats->d_varStats, _opt.frames_acc_blocks * alignedNMix * varDim * sizeof(float)) );
    checkError( cudaMalloc((void **)& _stats->d_mixProb, _opt.frames_acc_blocks * alignedNMix * sizeof(float)) );    

    _stats->statMeanSize = alignedNMix * alignedDim;
    _stats->statVarSize = alignedNMix * varDim;    
    _stats->statProbSize = alignedNMix;
    if (allocAuxStats)
        _stats->statAuxSize = alignedNMix * (alignedDim + 2);

    _stats->d_statsAlloc = true;
}



void GMMStatsEstimator_GPU::eraseAccStats() {

    checkError( cudaFree(_stats->d_meanStats) );
    checkError( cudaFree(_stats->d_varStats) );
    checkError( cudaFree(_stats->d_mixProb) );
    checkError( cudaFree(_stats->d_auxStats) );

    _stats->d_meanStats = NULL;
    _stats->d_varStats = NULL;
    _stats->d_mixProb = NULL;
    _stats->d_auxStats = NULL;

    _stats->statAuxSize = 0;
    _stats->statMeanSize = 0;
    _stats->statVarSize = 0;
    _stats->statProbSize = 0;
    _stats->accFullVar = false;

    _stats->d_statsAlloc = false;
}


void GMMStatsEstimator_GPU::resetAccStats() {

    if (_stats->statAuxSize > 0) {
        cudaMemsetAsync(_stats->d_auxStats, 0, sizeof(float) * _stats->statAuxSize * _opt.frames_acc_blocks);
        //setZeroArray <<< _stats->statAuxSize/RESET_BLOCK_SIZE, RESET_BLOCK_SIZE >>>
        //    (_stats->d_auxStats, _stats->statAuxSize);
    }
    if (_stats->statMeanSize > 0) {
        cudaMemsetAsync(_stats->d_meanStats, 0, sizeof(float) * _stats->statMeanSize * _opt.frames_acc_blocks);
        //setZeroArray <<< _stats->statMeanSize/RESET_BLOCK_SIZE, RESET_BLOCK_SIZE >>>
        //    (_stats->d_meanStats, _stats->statMeanSize);
    }
    if (_stats->statVarSize > 0) {
        cudaMemsetAsync(_stats->d_varStats, 0, sizeof(float) * _stats->statVarSize * _opt.frames_acc_blocks);
        //setZeroArray <<< _stats->statVarSize/RESET_BLOCK_SIZE, RESET_BLOCK_SIZE >>>
        //    (_stats->d_varStats, _stats->statVarSize);
    }
    if (_stats->statProbSize > 0) {
        cudaMemsetAsync(_stats->d_mixProb, 0, sizeof(float) * _stats->statProbSize * _opt.frames_acc_blocks);
        //setZeroArray <<< _stats->statProbSize/RESET_BLOCK_SIZE, RESET_BLOCK_SIZE >>>
        //    (_stats->d_mixProb, _stats->statProbSize);
    }
    if (_likes->d_totll != NULL)
        checkError( cudaMemsetAsync(_likes->d_totll, 0, sizeof(float)) );

#ifdef GPU_SAFE_MODE    
    cudaThreadSynchronize();
    checkError( cudaGetLastError() );
#endif
}



void GMMStatsEstimator_GPU::compLogLikes() 
{
    pushGPU();
    
#ifndef _EXCLUDE_SAFETY_CONDS_
    if (!_model->d_modelAlloc || !_prm->d_paramAlloc)
        throw std::runtime_error("getLogLikes(): none data!");
    if (_prm->Nframes > _likes->llSize || _model->dim != _prm->dim)
        throw std::runtime_error("getLogLikes(): dimension or #sample mismatch!");
#endif
    
    unsigned int NframeBlocks = alignDiv<unsigned int> (_prm->Nframes_processed / FRAME_BLOCK, MAX_GRID_SIZE);
    unsigned int shiftFrames = 0;
    for (unsigned int iBlock = 0; iBlock < NframeBlocks; iBlock++) 
    {            
        unsigned int NSamplesTmp = ((iBlock+1) * _prm->Nframes_processed > _prm->Nframes) ? _prm->Nframes - iBlock * _prm->Nframes_processed : _prm->Nframes_processed;

        compLogLikeAndGammas(shiftFrames, NSamplesTmp);            
        shiftFrames += NSamplesTmp;
    }
    
    popGPU();
}



void GMMStatsEstimator_GPU::getLogLikes (float* outLogLikes) 
{
    pushGPU();    

#ifndef _EXCLUDE_SAFETY_CONDS_
    if (_prm->Nframes_processed < _prm->Nframes && !_likes->ll_storeAll)
        throw std::runtime_error("getLogLikes(): not all the loglikes were stored -> set 'll_storeAll' = true before uploading param!");
#endif
    
    checkError( cudaMemcpy(outLogLikes, _likes->d_ll_begin, _prm->Nframes_unaligned *  sizeof(float), cudaMemcpyDeviceToHost) );
    popGPU();
}



void GMMStatsEstimator_GPU::compLogLikeAndGammas(unsigned int shiftFrames, unsigned int alignedNframes) 
{
    
#ifdef GPU_CUDA_TIMER
    cudaEventRecord( start1, 0 );
#endif

    // estimate raw (unnormalized) gammas (store in _likes->d_gammas) for each frame in _prm->d_vecs    
    if(_model->dimVar != _model->dim) {        
        gammasKernelFull <<< dim3(alignedNframes/FRAME_BLOCK, _model->nummix/GAUSS_BLOCK), GAUSS_BLOCK, FRAME_BLOCK*_model->dim*sizeof(float) >>> 
            (*_model, *_prm, *_likes, shiftFrames);    
    }
    else {        
        if(_prm->dim <= 128) {
            gammasKernel <<< dim3(alignedNframes/FRAME_BLOCK, _model->nummix/GAUSS_BLOCK), GAUSS_BLOCK, FRAME_BLOCK*_model->dim*sizeof(float) >>> (*_model, *_prm, *_likes, shiftFrames);
        } else {
            const unsigned int DIMS_PER_BLOCK = 1024/DATA_BLOCK;
            gammasKernelLargeDim <<< dim3(alignedNframes/FRAME_BLOCK, _model->nummix/GAUSS_BLOCK), GAUSS_BLOCK, FRAME_BLOCK*DIMS_PER_BLOCK*DIM_BLOCK*sizeof(float) >>> (*_model, *_prm, *_likes, shiftFrames);
        }
    }

#ifdef GPU_CUDA_TIMER
    cudaEventRecord( stop1, 0 );
#endif

#ifdef GPU_SAFE_MODE
    cudaThreadSynchronize();
    checkError( cudaGetLastError() );
#endif

#ifdef GPU_CUDA_TIMER
    cudaEventRecord( start2, 0 );
#endif

    // sum raw (unnormalized) gammas for each frame along all the gaussians;
    // note: data stored in float4 => #blocks = alignedLlsSize/4    
    _likes->d_ll = _likes->d_ll_begin + _likes->ll_storeAll * (shiftFrames / 4);
    logLikeKernel <<< alignedNframes/4, GAUSS_BLOCK >>> 
        (*_likes, _model->nummix_unaligned, _model->nummix, _opt.minLogLike);
    
#ifdef GPU_CUDA_TIMER
    cudaEventRecord( stop2, 0 );
#endif

#ifdef GPU_SAFE_MODE
    cudaThreadSynchronize();
    checkError( cudaGetLastError() );
#endif
}



// ! assuming that gammas & logLikes were already computed using compLogLikeAndGammas()
// note: size of gammas   = NSamples * number_of_mixtures
//         size of loglikes = NSamples
void GMMStatsEstimator_GPU::getGammasAndLL (float *gammas, float *loglikes,
                                            unsigned int alignedNframes, unsigned int alignedNmix)
{
    pushGPU();

#ifndef _EXCLUDE_SAFETY_CONDS_    
    if (alignedNframes > _prm->Nframes_processed || alignedNmix > _model->nummix)
        throw std::runtime_error("getGammasAndLL(): nummix or #sample mismatch!!");
    if (_prm->Nframes_processed < _prm->Nframes)
        throw std::runtime_error("getGammasAndLL(): not all the gammas and loglikes were stored -> mask unavailable!");
#endif
    
    checkError( cudaMemcpy(gammas, _likes->d_gammas, alignedNmix * alignedNframes * sizeof(float), cudaMemcpyDeviceToHost) );
    checkError( cudaMemcpy(loglikes, _likes->d_ll_begin, alignedNframes * sizeof(float), cudaMemcpyDeviceToHost) );
    popGPU();
}



void GMMStatsEstimator_GPU::compAccStats (bool fullVarAcc, bool auxStats) 
{
#ifndef _EXCLUDE_SAFETY_CONDS_    
    if (!_model->d_modelAlloc || !_prm->d_paramAlloc)
        throw std::runtime_error("compAccStats(): none data");
    if (_model->dim != _prm->dim)
        throw std::runtime_error("compAccStats(): dimension mismatch");
#endif

    pushGPU();

    // cannot be moved to uploadModel() since auxStats and fullVarAcc has not to be known in advance
    if ( _model->nummix_allocS * _model->dim > _stats->statMeanSize || 
         (fullVarAcc && !_stats->accFullVar) )
    {
        allocAccStats (_model->nummix_allocS, _model->dim, fullVarAcc, auxStats);
    }
    resetAccStats();
        
    unsigned int NframeBlocks = alignDiv<unsigned int> (_prm->Nframes_unaligned, _prm->Nframes_processed);
    unsigned int shiftFrames = 0;
    for (unsigned int iBlock = 0; iBlock < NframeBlocks; iBlock++) 
    {            
        unsigned int NSamplesTmp = ((iBlock+1) * _prm->Nframes_processed > _prm->Nframes) ? _prm->Nframes - iBlock * _prm->Nframes_processed : _prm->Nframes_processed;
        
        accumulateStats(shiftFrames, NSamplesTmp, auxStats, fullVarAcc);
        shiftFrames += NSamplesTmp;        
    }
    
    if(_opt.frames_acc_blocks > 0)
        sumAccStats(auxStats);

    popGPU();
}



//void GMMStatsEstimator_GPU::sumAccStats(bool auxStats)
//{
//    int numBlocks;
//    numBlocks = (_stats->statMeanSize / _model->nummix_allocS * _model->nummix) / DIM_BLOCK / GAUSS_BLOCK;
//    addArrays4Kernel <<< numBlocks, GAUSS_BLOCK >>>
//        (_stats->d_meanStats, _opt.frames_acc_blocks, _stats->statMeanSize / _model->nummix_allocS * _model->nummix);
//
//#ifdef GPU_SAFE_MODE
//        cudaThreadSynchronize();
//        checkError( cudaGetLastError() );
//#endif
//    
//    numBlocks = (_stats->statVarSize / _model->nummix_allocS * _model->nummix) / DIM_BLOCK / GAUSS_BLOCK;
//    addArrays4Kernel <<< numBlocks, GAUSS_BLOCK >>>
//        (_stats->d_varStats, _opt.frames_acc_blocks, _stats->statVarSize / _model->nummix_allocS * _model->nummix);
//
//#ifdef GPU_SAFE_MODE
//        cudaThreadSynchronize();
//        checkError( cudaGetLastError() );
//#endif
//
//    numBlocks = (_stats->statProbSize / _model->nummix_allocS * _model->nummix) / GAUSS_BLOCK;
//    addArraysKernel <<< numBlocks, GAUSS_BLOCK >>>
//        (_stats->d_mixProb, _opt.frames_acc_blocks, _stats->statProbSize / _model->nummix_allocS * _model->nummix);
//
//#ifdef GPU_SAFE_MODE
//        cudaThreadSynchronize();
//        checkError( cudaGetLastError() );
//#endif
//
//    //LLLLLLLLLL
//    printfCUDADATAfloat(_stats->d_mixProb, 256, "\n");
//
//    if (auxStats)
//    {
//        numBlocks = (_stats->statAuxSize / _model->nummix_allocS * _model->nummix) / GAUSS_BLOCK;
//        addArraysKernel <<< numBlocks, GAUSS_BLOCK >>>
//            (_stats->d_auxStats, _opt.frames_acc_blocks, _stats->statAuxSize / _model->nummix_allocS * _model->nummix);
//    }
//
//#ifdef GPU_SAFE_MODE
//        cudaThreadSynchronize();
//        checkError( cudaGetLastError() );
//#endif
//}

void GMMStatsEstimator_GPU::sumAccStats(bool auxStats)
{
    addArrays4Kernel <<< _stats->statMeanSize / DIM_BLOCK / GAUSS_BLOCK, GAUSS_BLOCK >>>
        (_stats->d_meanStats, _opt.frames_acc_blocks, _stats->statMeanSize);

#ifdef GPU_SAFE_MODE
        cudaThreadSynchronize();
        checkError( cudaGetLastError() );
#endif


    addArrays4Kernel <<< _stats->statVarSize / DIM_BLOCK / GAUSS_BLOCK, GAUSS_BLOCK >>>
        (_stats->d_varStats, _opt.frames_acc_blocks, _stats->statVarSize);

#ifdef GPU_SAFE_MODE
        cudaThreadSynchronize();
        checkError( cudaGetLastError() );
#endif

    addArraysKernel <<< _stats->statProbSize / GAUSS_BLOCK, GAUSS_BLOCK >>>
        (_stats->d_mixProb, _opt.frames_acc_blocks, _stats->statProbSize);

#ifdef GPU_SAFE_MODE
        cudaThreadSynchronize();
        checkError( cudaGetLastError() );
#endif

    if (auxStats)
    {
        addArraysKernel <<< _stats->statAuxSize / GAUSS_BLOCK, GAUSS_BLOCK >>>
            (_stats->d_auxStats, _opt.frames_acc_blocks, _stats->statAuxSize);

#ifdef GPU_SAFE_MODE
        cudaThreadSynchronize();
        checkError( cudaGetLastError() );
#endif
    }
}

void GMMStatsEstimator_GPU::getAccStats (float &totLL, float *meanStats, float *varStats, float *mixProbs, float* auxStats, 
                                         bool fullVarAcc) 
{
    pushGPU();    
    
    checkError( cudaMemcpy(&totLL, _likes->d_totll, sizeof(float), cudaMemcpyDeviceToHost) );

    if(meanStats != NULL)
        checkError( cudaMemcpy(meanStats, _stats->d_meanStats, _model->nummix_unaligned * _prm->dim * sizeof(float), cudaMemcpyDeviceToHost) );
    
    if (varStats != NULL) {
        unsigned int aVarDim = fullVarAcc ? (_prm->dim * (_prm->dim + DIM_BLOCK)) / 2 : _prm->dim;
        checkError( cudaMemcpy(varStats, _stats->d_varStats, _model->nummix_unaligned * aVarDim * sizeof(float), cudaMemcpyDeviceToHost) );
    }

    if(mixProbs != NULL)
        checkError( cudaMemcpy(mixProbs, _stats->d_mixProb, _model->nummix_unaligned * sizeof(float), cudaMemcpyDeviceToHost) );

    if(auxStats != NULL)
        checkError( cudaMemcpy(auxStats, _stats->d_auxStats, _model->nummix_unaligned * (_prm->dim + 2) * sizeof(float), cudaMemcpyDeviceToHost) );
    
    popGPU();
}



void GMMStatsEstimator_GPU::accumulateStats(unsigned int shiftFrames, unsigned int NSamplesTmp, 
                                            bool accAuxStats, bool fullVarAcc)
{
    compLogLikeAndGammas(shiftFrames, NSamplesTmp);
    
    //LLLLLLLLLLLLLL
    //printfCUDADATAfloat((float*) _likes->d_gammas, 135); 

    gammasNormKernel <<< NSamplesTmp/FRAME_BLOCK, GAUSS_BLOCK >>> 
        (*_likes, _opt, _model->nummix, _prm->Nframes_unaligned - shiftFrames);

#ifdef GPU_SAFE_MODE        
        cudaThreadSynchronize();
        checkError( cudaGetLastError() );
#endif

    //LLLLLLLLLLLLLL
//        {
//        FILE *fid = fopen("xx.txt", "w");
//        float *foo = new float[NSamplesTmp*_model->nummix];
//        GMMStatsEstimator_GPU::checkError( cudaMemcpy(foo, _likes->d_gammas, sizeof(float)*NSamplesTmp*_model->nummix, cudaMemcpyDeviceToHost) );
//    for(int i = 0; i<NSamplesTmp*_model->nummix; i++) {
//        fprintf(fid, "%f\n", foo[i]);
//    }
//    delete foo;
//    fclose(fid);
//}

#ifdef GPU_CUDA_TIMER
    cudaEventRecord( start3, 0 );
#endif

    if(!fullVarAcc)
    {
        accStatsDiagKernel <<< dim3(_model->dim/DIM_BLOCK, _model->nummix/GAUSS_BLOCK * _opt.frames_acc_blocks), GAUSS_BLOCK >>> 
            (*_prm, *_likes, *_stats, _model->nummix, shiftFrames);
    }
    else 
    {
        unsigned int dimVarDBxDB = _model->dimVar / (DIM_BLOCK * DIM_BLOCK);
        accStatsFullKernel <<< dim3(_model->nummix_unaligned, _opt.frames_acc_blocks), dimVarDBxDB >>> 
            (*_prm, *_likes, *_stats, _opt, _model->nummix, shiftFrames);
    }

#ifdef GPU_SAFE_MODE        
        cudaThreadSynchronize();
        checkError( cudaGetLastError() );
#endif

        //LLLLLLLLLLLLLL
        //printfCUDADATAfloat((float*) _stats->d_mixProb, 256); 


    if(accAuxStats) {

        normAuxGammasKernel <<< NSamplesTmp/4, GAUSS_BLOCK >>> (*_likes, _model->nummix_unaligned, _model->nummix, _opt.minGamma);

#ifdef GPU_SAFE_MODE        
        cudaThreadSynchronize();
        checkError( cudaGetLastError() );
#endif

        //LLLLLLLLLLLLLL
        //printfCUDADATAfloat((float*) _likes->d_aux_ll, 8); 
        //printfCUDADATAfloat((float*) _likes->d_gammas, 8); 
        //dumpCUDADATAfloat((float*) _likes->d_aux_ll, 837, 1, "gammas_norm_cuda.txt"); 

        accAuxStatsKernel <<< dim3(_model->dim/DIM_BLOCK, _model->nummix/GAUSS_BLOCK * _opt.frames_acc_blocks), dim3(GAUSS_BLOCK / DIM_BLOCK, DIM_BLOCK) >>> 
            (*_prm, *_likes, *_stats, _opt, _model->nummix, shiftFrames);

#ifdef GPU_SAFE_MODE        
        cudaThreadSynchronize();
        checkError( cudaGetLastError() );
#endif

        //LLLLLLLLLLLLLL
        //printfCUDADATAfloat((float*) _stats->d_auxStats, 38*2);
        //dumpCUDADATAfloat((float*) _stats->d_auxStats, 32*38, 8, "aux_cuda.txt");

    }

#ifdef GPU_CUDA_TIMER
    cudaEventRecord( stop3, 0 );
#endif

    unsigned int NS_GB_aligned = alignUP<unsigned int> (NSamplesTmp, 4*GAUSS_BLOCK);     
    sumLogLike <<< NS_GB_aligned / GAUSS_BLOCK, GAUSS_BLOCK >>>
        (*_likes, _prm->Nframes_unaligned - shiftFrames, NSamplesTmp);

#ifdef GPU_SAFE_MODE        
        cudaThreadSynchronize();
        checkError( cudaGetLastError() );
#endif    

#ifdef GPU_CUDA_TIMER
    //update GPU timers
    float time;
    cudaEventElapsedTime( &time, start1, stop1 );
    GPU_kernel_timer1 += time;
    cudaEventElapsedTime( &time, start2, stop2 );
    GPU_kernel_timer2 += time;
    cudaEventElapsedTime( &time, start3, stop3 );
    GPU_kernel_timer3 += time;
#endif
}



void GMMStatsEstimator_GPU::createContext(int GPU_id, void **pctx) 
{
#ifdef GPU_CUDA_CONTEXT_AVAIL
    CUdevice device;
    *pctx = (void *) new CUcontext();
    
    checkError( cuInit(0) );
    checkError( cuDeviceGet(&device, GPU_id) );
    checkError( cuCtxCreate((CUcontext *) (*pctx), CU_CTX_SCHED_AUTO, device) );
#endif
}


void GMMStatsEstimator_GPU::destroyContext(void **pctx) 
{       
#ifdef GPU_CUDA_CONTEXT_AVAIL
    checkError( cuCtxDestroy(*(CUcontext *) (*pctx)) );
        
    delete (CUcontext *) (*pctx);    
    *pctx = NULL;
#endif
}


void GMMStatsEstimator_GPU::popGPU() 
{
#ifdef GPU_CUDA_CONTEXT_AVAIL
    checkError( cuCtxPopCurrent(NULL) );
#endif
}


void GMMStatsEstimator_GPU::pushGPU()
{      
#ifdef GPU_CUDA_CONTEXT_AVAIL
    checkError( cuCtxPushCurrent(* (CUcontext *) _pContextGPU) );
#else
    checkError( cudaSetDevice(_GPU_id) );
#endif
}



template <class T>
T GMMStatsEstimator_GPU::alignUP(T x, T block2align) 
{
    return block2align * (x / block2align + (x % block2align > 0));        
}



template <class T>
T GMMStatsEstimator_GPU::alignDiv(T x, T block2align) 
{
    return (x / block2align + (x % block2align > 0));
}



bool GMMStatsEstimator_GPU::checkError(int error, int verbosity, bool throw_errors)
{    
    cudaError err = (cudaError) error;
    if(err != cudaSuccess) 
    {
        std::stringstream e;
        e <<  "CUDA Error #" << e << ": " << cudaGetErrorString(err) << " (file: " << __FILE__ << ")";
        
        if (throw_errors)
            throw std::runtime_error(e.str().c_str());

        if(verbosity > 0)
            std::cout << e.str() << std::endl;

        return false;
    }
    else return true;
}



//void GMMStatsEstimator_GPU::getNSamples2GPU (unsigned int &NSamplesGPU, unsigned int &NSamplesGPUprocess, 
//                                             unsigned int NSamples, unsigned int dim, 
//                                             unsigned int NAccBlocks, unsigned int aligDimAux,
//                                             bool allocStats, float maxGPUMem2UseGB)
//{        
//#ifndef _EXCLUDE_SAFETY_CONDS_    
//    if (!_model->d_modelAlloc)
//        throw std::runtime_error("getNSamples2GPU(): insert model first!");
//#endif
//
//    pushGPU();    
//    
//    size_t maxGPUMem2Use = static_cast<size_t> (maxGPUMem2UseGB * 1024.0f * 1024.0f * 1024.0f);    
//
//    size_t freeMem = 0, totMem = 0;
//#ifdef GPU_CUDA_CONTEXT_AVAIL
//    checkError( cuMemGetInfo(&freeMem, &totMem) );
//    if (freeMem < maxGPUMem2Use)
//        maxGPUMem2Use = freeMem;
//#else
//    cudaDeviceProp deviceProp;
//    checkError( cudaGetDeviceProperties(&deviceProp, _GPU_id) );
//    totMem = deviceProp.totalGlobalMem;
//    if (maxGPUMem2Use == 0)
//        maxGPUMem2Use = 0.7 * totMem;
//#endif
//
//    if (maxGPUMem2Use > totMem)
//        maxGPUMem2Use = 0.7 * totMem;
//
//    if (maxGPUMem2Use > 0)
//        freeMem = maxGPUMem2Use;
//
//    size_t min_unused_memory = MIN_UNUSED_MEM_GPU << 20;
//    size_t min_frames_buff_size = DIM_BLOCK * FRAME_BLOCK << 20;
//
//    size_t min_Nframes = min_frames_buff_size / dim / sizeof(float);
//    //size_t mem_SampCPU = NSamples * dim * sizeof(float);
//
//    size_t mem_GammasGPU = (_model->nummix_allocS + 1) * min_Nframes * sizeof(float);
//    size_t mem_StatsGPU = 0;
//    if (allocStats)
//        mem_StatsGPU = NAccBlocks * ( aligDimAux + _model->nummix_allocS * (1 + _model->dim + _model->dimVar) ) * sizeof(float);
//
//    size_t toAllocGPU = mem_StatsGPU + mem_GammasGPU + min_frames_buff_size;
//
//    if (toAllocGPU + min_unused_memory > freeMem)
//        throw std::runtime_error("getNSamples2GPU(): not enough free memory on GPU");
//
//    size_t memGPUAvailable = freeMem - mem_StatsGPU - min_unused_memory;
//
//    // how many frames can be allocated on GPU along with gammas
//    size_t mem_per_frame = (size_t) ((_model->nummix_allocS + 1) * sizeof(float) + _model->dim * sizeof(float));
//    
//    // NSamplesGPU = avail_mem / mem_for_1_frame_and_adjacent_gammas
//    NSamplesGPU = memGPUAvailable / mem_per_frame;
//
//    if (NSamplesGPU > NSamples)
//        NSamplesGPU = NSamples;
//
//    // align to FRAME_BLOCK
//    size_t NGPUa = alignUP<size_t> (NSamplesGPU, FRAME_BLOCK);        
//
//    // test whether it is possible to process all the frames (max grid size to small?)
//    NSamplesGPUprocess = NSamplesGPU;
//    if (NGPUa / FRAME_BLOCK > MAX_GRID_SIZE) 
//    {
//        // how many frames fit into the grid
//        size_t max_memSGridGPU = MAX_GRID_SIZE * FRAME_BLOCK * mem_per_frame;
//        memGPUAvailable -= max_memSGridGPU;
//
//        size_t NS_res = memGPUAvailable / (_model->dim * sizeof(float));
//
//        NSamplesGPU = MAX_GRID_SIZE * FRAME_BLOCK + NS_res;
//        NSamplesGPUprocess = MAX_GRID_SIZE * FRAME_BLOCK;
//    }    
//
//    popGPU();
//}
void GMMStatsEstimator_GPU::getNSamples2GPU (unsigned int &NSamplesGPU, unsigned int &NSamplesGPUprocess, 
                                             unsigned int NSamples, unsigned int dim, 
                                             unsigned int NAccBlocks, unsigned int aligDimAux,
                                             bool allocStats, float maxGPUMem2UseGB)
{        
#ifndef _EXCLUDE_SAFETY_CONDS_    
    if (!_model->d_modelAlloc)
        throw std::runtime_error("getNSamples2GPU(): insert model first!");
#endif

    pushGPU();    
    
    size_t maxGPUMem2Use = static_cast<size_t> (maxGPUMem2UseGB * 1024.0f * 1024.0f);    // KB

    size_t freeMem = 0, totMem = 0;
#ifdef GPU_CUDA_CONTEXT_AVAIL
    checkError( cuMemGetInfo(&freeMem, &totMem) );
    freeMem >>= 10; // KB
    totMem >>= 10; // KB
    if (freeMem < maxGPUMem2Use)
        maxGPUMem2Use = freeMem;
#else
    cudaDeviceProp deviceProp;
    checkError( cudaGetDeviceProperties(&deviceProp, _GPU_id) );
    totMem = deviceProp.totalGlobalMem;
    totMem >>= 10; // KB
    if (maxGPUMem2Use == 0)
        maxGPUMem2Use = 0.7 * totMem;
#endif

    if (maxGPUMem2Use > totMem)
        maxGPUMem2Use = 0.7 * totMem;

    if (maxGPUMem2Use > 0)
        freeMem = maxGPUMem2Use;

    size_t min_unused_memory = MIN_UNUSED_MEM_GPU << 10; // KB

    size_t min_Nframes = MIN_FRAMES_ON_GPU;
    size_t min_frames_buff_size = min_Nframes * dim * sizeof(float);
    min_frames_buff_size >>= 10; // KB

    size_t mem_GammasGPU = (_model->nummix_allocS + 2) * min_Nframes * sizeof(float);
    mem_GammasGPU >>= 10; // KB
    size_t mem_StatsGPU = 0;
    if (allocStats) {
        mem_StatsGPU = NAccBlocks * ( aligDimAux + _model->nummix_allocS * (1 + _model->dim + _model->dimVar) ) * sizeof(float);
        mem_StatsGPU >>= 10; // KB        
    }

    size_t toAllocGPU = mem_StatsGPU + mem_GammasGPU + min_frames_buff_size; // KB    

    if (toAllocGPU + min_unused_memory > freeMem)
        throw std::runtime_error("getNSamples2GPU(): not enough free memory on GPU");

    size_t memGPUAvailable = freeMem - mem_StatsGPU - min_unused_memory; // KB

    // how many frames can be allocated on GPU along with gammas
    size_t mem_per_frame = (size_t) ( (_model->nummix_allocS + 1 + _model->dim) * sizeof(float) ); // B
    
    // NSamplesGPU = avail_mem / mem_for_1_frame_and_adjacent_gammas
    NSamplesGPU = 1024 * (memGPUAvailable / mem_per_frame);

    if (NSamplesGPU > NSamples)
        NSamplesGPU = NSamples;

    // align to FRAME_BLOCK
    size_t NGPUa = alignUP<size_t> (NSamplesGPU, FRAME_BLOCK);        

    // test whether it is possible to process all the frames (max grid size to small?)
    NSamplesGPUprocess = NSamplesGPU;
    if (NGPUa / FRAME_BLOCK > MAX_GRID_SIZE) 
    {
        // how many frames fit into the grid        
        size_t max_memSGridGPU = MAX_GRID_SIZE * mem_per_frame;
        max_memSGridGPU >>= 10; // KB
        max_memSGridGPU *= FRAME_BLOCK;
        memGPUAvailable -= max_memSGridGPU;

        size_t NS_res = 1024 * (memGPUAvailable / (_model->dim * sizeof(float)));

        NSamplesGPU = MAX_GRID_SIZE * FRAME_BLOCK + NS_res;
        NSamplesGPUprocess = MAX_GRID_SIZE * FRAME_BLOCK;
    }
    
    if (NSamplesGPU > NSamples)
        NSamplesGPU = NSamples;

    popGPU();
}
