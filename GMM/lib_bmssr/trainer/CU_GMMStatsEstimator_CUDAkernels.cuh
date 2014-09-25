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


#ifndef _CU_STATS_EST_CUDA_KERNELS_
#define _CU_STATS_EST_CUDA_KERNELS_

#define ALIGN_UP(x, alig) ((alig)*((x)/(alig) + ((x)%(alig)>0)))
#define ALIGN_DIV(x, y) ((x)/(y) + ((x)%(y)>0))

#define FLOAT4toREGARRAY(f, reg_offset, f4) {float4 value = f4; f[reg_offset+0] = value.x; f[reg_offset+1] = value.y; f[reg_offset+2] = value.z; f[reg_offset+3] = value.w;}
#define make_float4_REGARRAY(f, reg_offset) make_float4(f[reg_offset+0], f[reg_offset+1], f[reg_offset+2], f[reg_offset+3])

#if __CUDA_ARCH__ < 200
#if __CUDA_ARCH__ == 100
#error "CUDA arch 1.0 not supproted, change CUDA arch to 1.1 or higher"
#endif
#define atomicAdd(address, value) {float old = value; float new_old; do {new_old = atomicExch(address, 0.0f); new_old += old; } while ((old = atomicExch(address, new_old))!=0.0f);}
#endif
#define LOCAL_REDUCTION(x, buff, rNUM_THREADS, rWARP_SIZE) __shared__ float buff[rNUM_THREADS];buff[tid] = x;__syncthreads();if(tid < rWARP_SIZE){for(int i = rWARP_SIZE + tid; i < rNUM_THREADS; i += rWARP_SIZE){x += buff[i];}buff[tid] = x;}__syncthreads();if(tid < rWARP_SIZE) {for(int i = 1; i < rWARP_SIZE; i ++) {x += buff[i%rNUM_THREADS];}}

#define KERNEL_GPU_SAFE_MODE

void dumpCUDADATAfloat(float *dd_data, int rows, int cols, const char *filename) 
{
	float *foo = new float[rows*cols];
	GMMStatsEstimator_GPU::checkError( cudaMemcpy(foo, dd_data, sizeof(float)*rows*cols, cudaMemcpyDeviceToHost) );
	FILE *fid = fopen(filename, "w");
	for(int i = 0; i<rows; i++) {
		for(int ii = 0; ii<cols; ii++) {
			fprintf(fid, " %f", foo[i*cols+ii]);
		}
		fprintf(fid, "\n");
	}
	fclose(fid);
	delete foo;
}

void printfCUDADATAfloat(float *dd_data, int num, char*str=NULL) 
{
	float *foo = new float[num];
	GMMStatsEstimator_GPU::checkError( cudaMemcpy(foo, dd_data, sizeof(float)*num, cudaMemcpyDeviceToHost) );
	if(str != NULL) 
		printf("%s\n", str);
	for(int i = 0; i<num; i++) 
		printf("%f\n", foo[i]);
	printf("\n");
	delete foo;
}

namespace {
	__device__ float addLog(float f1, float f2) {
		float mx;
		mx = max(f1, f2);
		return mx + __logf(1.0f + __expf(min(f1, f2) - mx));
	}

	__device__ float4 addLog4(float4 f1, float4 f2) {
		float4 f;
		f.x = addLog(f1.x, f2.x);
		f.y = addLog(f1.y, f2.y);
		f.z = addLog(f1.z, f2.z);
		f.w = addLog(f1.w, f2.w);

		return f;
	}

	__device__ void checkLogLike(float4 &f1, float Th) {
		f1.x = max(f1.x, Th);
		f1.y = max(f1.y, Th);
		f1.z = max(f1.z, Th);
		f1.w = max(f1.w, Th);
		f1.x = (isnan(f1.x) || isinf(f1.x))? Th : f1.x;
		f1.y = (isnan(f1.y) || isinf(f1.y))? Th : f1.y;
		f1.z = (isnan(f1.z) || isinf(f1.z))? Th : f1.z;
		f1.w = (isnan(f1.w) || isinf(f1.w))? Th : f1.w;
	}

	__device__ void checkLogLike(float &f1, float Th) {
		f1 = max(f1, Th);
		f1 = (isnan(f1) || isinf(f1))? Th : f1;
	}

	//__global__ void setZeroArray(float *data, unsigned int num)
	//{
	//	unsigned int offset = blockIdx.x*blockDim.x + threadIdx.x;
	//	if(offset < num)
	//		data[offset] = 0.0f;
	//}
}



// //<<< _model->nummix, _model->dim, (_model->dim+2)*FRAME_BLOCK*sizeof(float) >>> 
//__global__ void accAuxStatsKernel_OLD (GMMStatsEstimator_GPU::param p,
//								       GMMStatsEstimator_GPU::likes l,
//								       GMMStatsEstimator_GPU::stats s,
//								       GMMStatsEstimator_GPU::OPTIONS opt,
//								       unsigned int nummix,
//								       unsigned int shiftFrames)
//{
//	unsigned int ix = blockIdx.x;
//	unsigned int tid = threadIdx.x;
//
//	const unsigned int FRAME_BLOCK = GMMStatsEstimator_GPU::FRAME_BLOCK;
//	const unsigned int DIM_BLOCK = GMMStatsEstimator_GPU::DIM_BLOCK;
//
//	extern __shared__ float xx[];
//	float *gamma = xx + p.dim*FRAME_BLOCK;
//	//float *buffLL = gamma + FRAME_BLOCK;
//
//	//"-1th" frame init
//	float auxAcc = 0.0f; //mean abs diff accumulator		
//	float g_old = 0.0f;
//
//	float x_old = p.d_vecs[shiftFrames * p.dim + 1 + tid*FRAME_BLOCK]; //init by second frame
//	float dx_old = fabs(p.d_vecs[shiftFrames * p.dim + tid*FRAME_BLOCK] - x_old);
//	
//	//unsigned int NSamples_max = p.Nframes_unaligned - shiftFrames;
//	unsigned int NSamples = (shiftFrames + p.Nframes_processed < p.Nframes) ? p.Nframes_processed : p.Nframes - shiftFrames;
//
//	float sqrtgm = 0;
//	for(unsigned int i=0; i< NSamples / FRAME_BLOCK; i++) {
//
//		__syncthreads();
//
//		//load gammas
//		for(unsigned int f = tid; f < FRAME_BLOCK; f += blockDim.x) {
//			gamma[f] = ((float*) l.d_gammas)[4*ix + i * FRAME_BLOCK * nummix + 4 * (f/DIM_BLOCK) * nummix + f%DIM_BLOCK];
//			gamma[f] *= 0.5f;
//		}
//
//		__syncthreads();
//		
//		//load param into shared memory - in frame by frame order (bank conflicts)
//		for(unsigned int m = tid; m < FRAME_BLOCK * p.dim; m += blockDim.x)
//			xx[(m % FRAME_BLOCK) * p.dim + (m / FRAME_BLOCK)] = p.d_vecs[shiftFrames * p.dim + i * FRAME_BLOCK * p.dim + m];
//		
//		__syncthreads();
//
//		//calculate differences
//		float dx = fabs(x_old - xx[tid]);
//		auxAcc += g_old * (dx_old + dx);
//		dx_old = dx;
//
//#pragma unroll
//		for(unsigned int f = 0; f < FRAME_BLOCK - 1; f++) {
//			dx = fabs(xx[f * p.dim + tid] - xx[(f + 1) * p.dim + tid]);
//			auxAcc += gamma[f] * (dx + dx_old);
//			sqrtgm += sqrt(gamma[f]);
//			dx_old = dx;
//		}
//		x_old = xx[(FRAME_BLOCK - 1) * p.dim + tid];
//		g_old = gamma[FRAME_BLOCK - 1];
//	} //for i - Nframes/FRAME_BLOCK
//
//	//last frame:
//	auxAcc += 2.0f * g_old * dx_old;
//
//	s.d_auxStats[ix * (blockDim.x + 1) + tid] += auxAcc;
//	if (tid == 0)
//		s.d_auxStats[ix * (blockDim.x + 1) + blockDim.x] += sqrtgm;	
//} //accAuxStatsKernel

////different variant than in Matlab - worse results in some cases - mean diffetences instead of convert gammas (geometric mean or posterior probability)
////<<< dim3(_model->dim/DIM_BLOCK, _model->nummix/GAUSS_BLOCK * _opt.frames_acc_blocks), dim3(GAUSS_BLOCK / DIM_BLOCK, DIM_BLOCK) >>>
//__global__ void accAuxStatsKernel (GMMStatsEstimator_GPU::param p,
//								   GMMStatsEstimator_GPU::likes l,
//								   GMMStatsEstimator_GPU::stats s,
//								   GMMStatsEstimator_GPU::OPTIONS opt,
//								   unsigned int nummix,
//								   unsigned int shiftFrames)
//{	
//	const unsigned int FRAME_BLOCK = GMMStatsEstimator_GPU::FRAME_BLOCK;
//	const unsigned int DIM_BLOCK = GMMStatsEstimator_GPU::DIM_BLOCK;
//	const unsigned int GAUSS_BLOCK = GMMStatsEstimator_GPU::GAUSS_BLOCK;
//
//	unsigned int ix = blockIdx.x;
//	unsigned int iy = blockIdx.y % (nummix/GAUSS_BLOCK);
//	unsigned int iz = blockIdx.y / (nummix/GAUSS_BLOCK);
//	unsigned int nBlocks = (gridDim.y * GAUSS_BLOCK) / nummix;
//	unsigned int tid = threadIdx.y * blockDim.x + threadIdx.x;
//
//	unsigned int NSamplesInBlock = ALIGN_UP( ALIGN_DIV(p.Nframes_processed, nBlocks), FRAME_BLOCK );
//	shiftFrames += iz * NSamplesInBlock;
//	if (shiftFrames > p.Nframes)
//		return;
//		
//	unsigned int NSamples = ((iz+1) * NSamplesInBlock < p.Nframes_processed) ? NSamplesInBlock : p.Nframes_processed - iz * NSamplesInBlock;
//	NSamples = (shiftFrames + NSamplesInBlock < p.Nframes) ? NSamples : p.Nframes - shiftFrames;
//
//	__shared__ float xx[DIM_BLOCK][4*FRAME_BLOCK]; //four bloks: recent, actual, and future blocks, agmented by difference-block
//
//	float gamma[FRAME_BLOCK];
//	float sqrtgm = 0.0f;
//	float auxAcc[DIM_BLOCK];
//
//	#pragma unroll
//	for(int d=0; d < DIM_BLOCK; d++) auxAcc[d] = 0.0;
//
//	p.d_vecs += shiftFrames * p.dim + ix * FRAME_BLOCK * DIM_BLOCK; //move the pointer by shift and dimension
//
//	float foo2 = p.d_vecs[tid];
//	if(threadIdx.x == 1) xx[threadIdx.y][2*FRAME_BLOCK - 1] = foo2; //int the recent block in reverse order
//	xx[threadIdx.y][2*FRAME_BLOCK + threadIdx.x] = foo2; //init the future block in normal order
//	__syncthreads();
//		
//	unsigned int offset = iz * nummix * (NSamplesInBlock / 4) + iy * GAUSS_BLOCK + tid;	
//	for(unsigned int i=0; i< NSamples / FRAME_BLOCK; i++) {
//
//		__syncthreads();
//
//		//load gammas
//		#pragma unroll
//		for(unsigned int f = 0; f < FRAME_BLOCK/4; f ++) {			
//			FLOAT4toREGARRAY(gamma, 4*f, l.d_gammas[offset + f * nummix]);
//		}
//		offset += FRAME_BLOCK/4 * nummix;
//
//		//move future frame block into actual place
//		xx[threadIdx.y][threadIdx.x] = xx[threadIdx.y][FRAME_BLOCK+threadIdx.x];
//		__syncthreads();
//		xx[threadIdx.y][FRAME_BLOCK+threadIdx.x] = xx[threadIdx.y][2*FRAME_BLOCK+threadIdx.x];
//		__syncthreads();
//		p.d_vecs += FRAME_BLOCK * p.dim;
//		if(i+1 < NSamples / FRAME_BLOCK) xx[threadIdx.y][2*FRAME_BLOCK+threadIdx.x] = p.d_vecs[tid];
//		__syncthreads();
//
//		////LLLLLLLLLLLLLLLLLL
//		//if(tid==0 && iz == 0 && iy == 0 && ix == 0 && i==NSamples / FRAME_BLOCK - 1) {
//		//	for (int f=0;f<3*FRAME_BLOCK;f++)
//		//		printf("\nFFFF: %d %e", f, xx[0][f]); //LLLLLLLLL
//		//}
//		//__syncthreads();
//		////EEEEEEEEEEEEEEEEE
//
//		//compute differences
//		float dx1 = fabs(xx[threadIdx.y][FRAME_BLOCK+threadIdx.x-1] - xx[threadIdx.y][FRAME_BLOCK+threadIdx.x]);
//		float dx2 = fabs(xx[threadIdx.y][FRAME_BLOCK+threadIdx.x] - xx[threadIdx.y][FRAME_BLOCK+threadIdx.x+1]);
//		if(i*FRAME_BLOCK + threadIdx.x + 1 == p.Nframes_unaligned) dx2 = dx1; //end fix
//		xx[threadIdx.y][3*FRAME_BLOCK+threadIdx.x] = 0.5f * (dx1 + dx2);
//		__syncthreads();
//
//		//accumulate
//		if(ix==0) { 
//			#pragma unroll
//			for(int f=0; f < FRAME_BLOCK; f++) {				
//				sqrtgm += sqrt(gamma[f]);
//			}
//		}
//		#pragma unroll
//		for(int f=0; f < FRAME_BLOCK; f++) {
//			//if(tid==0 && iz == 0 && iy == 0 && ix == 0 && i==NSamples / FRAME_BLOCK - 1) printf("\nXXXX: %d %e %e", f, gamma[f], xx[0][3*FRAME_BLOCK+f]); //LLLLLLLLL
//			#pragma unroll
//			for(int d=0; d < DIM_BLOCK; d++) {
//				auxAcc[d] += gamma[f] * xx[d][3*FRAME_BLOCK+f];
//			}
//		}
//	} //for i - Nframes/FRAME_BLOCK
//
//	__syncthreads();
//
//	float foo[DIM_BLOCK];
//	#pragma unroll
//	for(int d=0; d < DIM_BLOCK; d++) {
//		foo[d] = auxAcc[d] + s.d_auxStats[iz * s.statAuxSize + (p.dim + 1) * (iy * GAUSS_BLOCK + tid) + ix * DIM_BLOCK + d];
//	}
//	#pragma unroll
//	for(int d=0; d < DIM_BLOCK; d++) {
//		s.d_auxStats[iz * s.statAuxSize + (p.dim + 1) * (iy * GAUSS_BLOCK + tid) + ix * DIM_BLOCK + d] = foo[d];
//	}
//	
//	if (ix == 0) {
//		s.d_auxStats[iz * s.statAuxSize + (p.dim + 1) * (iy * GAUSS_BLOCK + tid) + p.dim] += sqrtgm;
//		//printf("GGG %d %d %d %e\n", iy, iz, iz * s.statAuxSize + (p.dim + 1) * (iy * GAUSS_BLOCK + tid) + p.dim, sqrtgm); //LLLLLLLLL
//	}
//
//} //accAuxStatsKernel 

//different variant than in CPU/SSE - geometric mean istead of posterior probability ofgammas - it requires global norm of new gammas
//<<< dim3(_model->dim/DIM_BLOCK, _model->nummix/GAUSS_BLOCK * _opt.frames_acc_blocks), dim3(GAUSS_BLOCK / DIM_BLOCK, DIM_BLOCK) >>>
__global__ void accAuxStatsKernel (GMMStatsEstimator_GPU::param p,
								   GMMStatsEstimator_GPU::likes l,
								   GMMStatsEstimator_GPU::stats s,
								   GMMStatsEstimator_GPU::OPTIONS opt,
								   unsigned int nummix,
								   unsigned int shiftFrames)
{	
	const unsigned int FRAME_BLOCK = GMMStatsEstimator_GPU::FRAME_BLOCK;
	const unsigned int DIM_BLOCK = GMMStatsEstimator_GPU::DIM_BLOCK;
	const unsigned int GAUSS_BLOCK = GMMStatsEstimator_GPU::GAUSS_BLOCK;

	unsigned int ix = blockIdx.x;
	unsigned int iy = blockIdx.y % (nummix/GAUSS_BLOCK);
	unsigned int iz = blockIdx.y / (nummix/GAUSS_BLOCK);
	unsigned int nBlocks = (gridDim.y * GAUSS_BLOCK) / nummix;
	unsigned int tid = threadIdx.y * blockDim.x + threadIdx.x;

	unsigned int NSamplesInBlock = ALIGN_UP( ALIGN_DIV(p.Nframes_processed, nBlocks), FRAME_BLOCK );
	shiftFrames += iz * NSamplesInBlock;
	if (shiftFrames > p.Nframes)
		return;
		
	unsigned int NSamples = ((iz+1) * NSamplesInBlock < p.Nframes_processed) ? NSamplesInBlock : p.Nframes_processed - iz * NSamplesInBlock;
	NSamples = (shiftFrames + NSamplesInBlock < p.Nframes) ? NSamples : p.Nframes - shiftFrames;

	__shared__ float xx[DIM_BLOCK][3*FRAME_BLOCK]; //three bloks: recent, actual, agmented by the difference-block

	float gamma[FRAME_BLOCK+1];
	float sqrtgm = 0.0f;
	float gm = 0.0f;
	__shared__ float gn[FRAME_BLOCK];
	float auxAcc[DIM_BLOCK];

	#pragma unroll
	for(int d=0; d < DIM_BLOCK; d++) auxAcc[d] = 0.0;

	p.d_vecs += shiftFrames * p.dim + ix * FRAME_BLOCK * DIM_BLOCK; //move the pointer by shift and dimension
	l.d_aux_ll += iz * NSamplesInBlock/4; //float4

	unsigned int offset = iz * nummix * (NSamplesInBlock / 4) + iy * GAUSS_BLOCK + tid;	

	float foo2;
	if(iz == 0) {
		foo2 = p.d_vecs[tid];
		if(threadIdx.x == 1) xx[threadIdx.y][2*FRAME_BLOCK - 1] = foo2; //int the recent block in reverse order
	} else {
		foo2 = (p.d_vecs - FRAME_BLOCK * p.dim)[tid];
		xx[threadIdx.y][FRAME_BLOCK + threadIdx.x] = foo2; //init the future block in normal order
		//load gammas
		#pragma unroll
		for(unsigned int f = 0; f < FRAME_BLOCK/4; f ++) {			
			FLOAT4toREGARRAY(gamma, 1+4*f, l.d_gammas[offset + f * nummix - FRAME_BLOCK/4 * nummix]);
		}
	}
	__syncthreads();
		
	for(unsigned int i=0; i< NSamples / FRAME_BLOCK; i++) {
	//for(unsigned int i=0; i< 1; i++) { //LLLLLLLL

		__syncthreads();

		//copy last gamma to old
		gamma[0] = gamma[FRAME_BLOCK];
		//load gammas
		#pragma unroll
		for(unsigned int f = 0; f < FRAME_BLOCK/4; f ++) {			
			FLOAT4toREGARRAY(gamma, 1+4*f, l.d_gammas[offset + f * nummix]);
		}
		offset += FRAME_BLOCK/4 * nummix;
		//set gamma of -1 frame to zero:
		if(iz == 0 && i==0) gamma[0] = 0.0f;

		//move future frame block into actual place
		xx[threadIdx.y][threadIdx.x] = xx[threadIdx.y][FRAME_BLOCK+threadIdx.x];
		__syncthreads();
		//xx[threadIdx.y][FRAME_BLOCK+threadIdx.x] = xx[threadIdx.y][2*FRAME_BLOCK+threadIdx.x];
		//__syncthreads();
		xx[threadIdx.y][FRAME_BLOCK+threadIdx.x] = p.d_vecs[tid];
		p.d_vecs += FRAME_BLOCK * p.dim;

		if(tid < FRAME_BLOCK) {
			gn[tid] = ((float*)&(l.d_aux_ll[i*2]))[tid];
			gn[tid] = (gn[tid] == 0)? 0.0f : 1/gn[tid];
		}
		__syncthreads();

		////LLLLLLLLLLLLLLLLLL
		//if(tid==0 && iz == 0 && iy == 0 && ix == 0 && i==NSamples / FRAME_BLOCK - 1) {
		//	for (int f=0;f<3*FRAME_BLOCK;f++)
		//		printf("\nFFFF: %d %e", f, xx[0][f]); //LLLLLLLLL
		//}
		//__syncthreads();
		////EEEEEEEEEEEEEEEEE

		//compute differences
		float dx1 = fabs(xx[threadIdx.y][FRAME_BLOCK+threadIdx.x-1] - xx[threadIdx.y][FRAME_BLOCK+threadIdx.x]);
		//float dx2 = fabs(xx[threadIdx.y][FRAME_BLOCK+threadIdx.x] - xx[threadIdx.y][FRAME_BLOCK+threadIdx.x+1]);
		//if(i*FRAME_BLOCK + threadIdx.x + 1 == p.Nframes_unaligned) dx2 = dx1; //end fix
		xx[threadIdx.y][2*FRAME_BLOCK+threadIdx.x] = dx1;//LLLLLLLLLLLLL
		__syncthreads();

		//accumulate
//		if(ix==0) { 
			#pragma unroll
			for(int f=0; f < FRAME_BLOCK; f++) {				
				float g = (gamma[f] * gamma[f+1]) * gn[f];
				sqrtgm += (gamma[f+1] > opt.minGamma) * sqrt(gamma[f+1]);
				gm += g;
				gamma[f] = g;
			}
//		}
		//if(iz == 1 && iy == 0 && ix == 0 && i==0) printf("\nXXXX: %d %d %e", threadIdx.y, threadIdx.x, xx[threadIdx.y][threadIdx.x]); //LLLLLLLLL
		//if(iz == 1 && iy == 0 && ix == 0 && i==0) printf("\nYYYY: %d %d %e", threadIdx.y, threadIdx.x, xx[threadIdx.y][FRAME_BLOCK+threadIdx.x]); //LLLLLLLLL
		//if(iz == 1 && iy == 0 && ix == 0 && i==0) printf("\nZZZZ: %d %d %e", threadIdx.y, threadIdx.x, xx[threadIdx.y][2*FRAME_BLOCK+threadIdx.x]); //LLLLLLLLL

		#pragma unroll
		for(int f=0; f < FRAME_BLOCK; f++) {
			#pragma unroll
			for(int d=0; d < DIM_BLOCK; d++) {
				auxAcc[d] += gamma[f] * xx[d][2*FRAME_BLOCK+f];
				//auxAcc[d] += (i == 104) * gamma[f] * xx[d][2*FRAME_BLOCK+f];
			}
		}
	} //for i - Nframes/FRAME_BLOCK

	__syncthreads();

	float foo[DIM_BLOCK];
	#pragma unroll
	for(int d=0; d < DIM_BLOCK; d++) {
		foo[d] = auxAcc[d] + s.d_auxStats[iz * s.statAuxSize + (p.dim + 2) * (iy * GAUSS_BLOCK + tid) + ix * DIM_BLOCK + d];
	}
	#pragma unroll
	for(int d=0; d < DIM_BLOCK; d++) {
		s.d_auxStats[iz * s.statAuxSize + (p.dim + 2) * (iy * GAUSS_BLOCK + tid) + ix * DIM_BLOCK + d] = foo[d];
	}
	
	if (ix == 0) {
		s.d_auxStats[iz * s.statAuxSize + (p.dim + 2) * (iy * GAUSS_BLOCK + tid) + p.dim] += sqrtgm;
		s.d_auxStats[iz * s.statAuxSize + (p.dim + 2) * (iy * GAUSS_BLOCK + tid) + p.dim + 1] += gm;
		//printf("GGG %d %d %d %e\n", iy, iz, iz * s.statAuxSize + (p.dim + 2) * (iy * GAUSS_BLOCK + tid) + p.dim, sqrtgm); //LLLLLLLLL
	}

} //accAuxStatsKernel_debug



// full covariance matrix will be accumulated - only upper triangle stored
// accStatsFullKernel <<<dim3(_model->nummix_unaligned, _opt.frames_acc_blocks), dimVar4x4 >>>
__global__ void accStatsFullKernel (GMMStatsEstimator_GPU::param p,
								    GMMStatsEstimator_GPU::likes l,
								    GMMStatsEstimator_GPU::stats s,
								    GMMStatsEstimator_GPU::OPTIONS opt,
								    unsigned int nummix,								   								   
								    unsigned int shiftFrames) 
{		
	unsigned int ix = blockIdx.x;
	unsigned int iy = blockIdx.y;
	unsigned int nBlocks = gridDim.y;
	unsigned int tid = threadIdx.x;

	const unsigned int FRAME_BLOCK = GMMStatsEstimator_GPU::FRAME_BLOCK;
	const unsigned int DIM_BLOCK = GMMStatsEstimator_GPU::DIM_BLOCK;
	const unsigned int DATA_BLOCK = GMMStatsEstimator_GPU::DATA_BLOCK;
	const unsigned int GAUSS_BLOCK = GMMStatsEstimator_GPU::GAUSS_BLOCK;

	unsigned int NSamplesInBlock = ALIGN_UP( ALIGN_DIV(p.Nframes_processed, nBlocks), FRAME_BLOCK );
	shiftFrames += iy * NSamplesInBlock;
	if (shiftFrames > p.Nframes)
		return;
	
	unsigned int NSamples = ((iy+1) * NSamplesInBlock < p.Nframes_processed) ? NSamplesInBlock : p.Nframes_processed - iy * NSamplesInBlock;
	NSamples = (shiftFrames + NSamplesInBlock < p.Nframes) ? NSamples : p.Nframes - shiftFrames;

	uint2 idx;
	unsigned int offset = 0;
	unsigned int oVarOffsetTmp = 0;
	unsigned int oVarOffset;
	for(unsigned int x = 0; x < p.dim / DIM_BLOCK; x++)  {
		for(unsigned int y = 0; y < p.dim / DIM_BLOCK - x; y++) {
			if (offset == tid) {
				idx.x = x;
				idx.y = y;
				oVarOffset = oVarOffsetTmp;
			}
			++offset;
		}
		oVarOffsetTmp += p.dim / DIM_BLOCK - x;
	}

	float accMean[DIM_BLOCK];
	float accVar[DIM_BLOCK][DIM_BLOCK];
	float mixProb = 0.0f;

#pragma unroll
	// reset accumulators
	for (unsigned int i = 0; i < DIM_BLOCK; i++) {
		accMean[i] = 0.0f;	
#pragma unroll
		for (unsigned int ii = 0; ii < DIM_BLOCK; ii++)
			accVar[i][ii] = 0.0f;
	}

	offset = shiftFrames * p.dim;
	for(unsigned int x = 0; x < NSamples / FRAME_BLOCK; x++)
	{		
		__syncthreads();

		__shared__ float gamma4[FRAME_BLOCK];
		
		for(unsigned int f = tid; f < FRAME_BLOCK; f += blockDim.x)
			gamma4[f] = ((float*) l.d_gammas)[iy * NSamplesInBlock * nummix + 4*ix + x * FRAME_BLOCK * nummix + 4 * (f/DIM_BLOCK) * nummix + f%DIM_BLOCK];
							
		__syncthreads();
				
		for(unsigned int f = 0; f < FRAME_BLOCK; f++) {
			float g = gamma4[f];
			if (g == 0.0f) 
				continue;			
			
			float data_x[4];
#pragma unroll
			for(unsigned int d = 0; d < DIM_BLOCK; d++)
				data_x[d] = p.d_vecs[offset + f + idx.x * DATA_BLOCK + d * FRAME_BLOCK];

			float data_y[4];
#pragma unroll
			for(unsigned int d = 0; d < DIM_BLOCK; d++)
				data_y[d] = p.d_vecs[offset + f + (idx.y + idx.x) * DATA_BLOCK + d * FRAME_BLOCK];		
			
			// compute mixProb
			mixProb += g;

			// accumnulate means only if two data_blocks coincide							
#pragma unroll
			for(unsigned int d = 0; d < DIM_BLOCK; d++) {
				accMean[d] += g * data_y[d];
#pragma unroll
				for(unsigned int dd = 0; dd < DIM_BLOCK; dd++) {
					accVar[d][dd] += g * data_x[d] * data_y[dd];
				}
			}
		} // f < FRAME_BLOCK
		offset = offset + p.dim * FRAME_BLOCK;		
	} // x < NSamples/FRAME_BLOCK

	unsigned int _ix = ix / GAUSS_BLOCK;
	unsigned int _tid = ix % GAUSS_BLOCK;
	if(tid < p.dim / DIM_BLOCK) 
	{		
		offset = iy * (s.statMeanSize / 4) + (_ix * GAUSS_BLOCK * p.dim / DIM_BLOCK) + p.dim / DIM_BLOCK * _tid + idx.y;
		float4 foo = ((float4 *) s.d_meanStats)[offset];
		((float4 *) s.d_meanStats)[offset] = make_float4(accMean[0] + foo.x, accMean[1] + foo.y, accMean[2] + foo.z, accMean[3] + foo.w);
	}

	// offset for variance
	unsigned int varBlockS = p.dim / DIM_BLOCK * (p.dim / DIM_BLOCK + 1) / 2 * DIM_BLOCK;
	offset = iy * (s.statVarSize / 4) + ix * varBlockS + oVarOffset * DIM_BLOCK + idx.y;

	for (unsigned int i = 0; i < DIM_BLOCK; i++) {
		float4 foo = ((float4 *) s.d_varStats)[offset + i * (p.dim / DIM_BLOCK - idx.x)];
		((float4 *) s.d_varStats)[offset + i * (p.dim / DIM_BLOCK - idx.x)] = make_float4(accVar[i][0] + foo.x, accVar[i][1] + foo.y, accVar[i][2] + foo.z, accVar[i][3] + foo.w);
	}

	if(tid == 0)
		s.d_mixProb[ix + iy * s.statProbSize] += mixProb;
}



//accStatsDiagKernel <<< dim3(_model->dim/DIM_BLOCK, _model->nummix/GAUSS_BLOCK * _opt.frames_acc_blocks), GAUSS_BLOCK >>> 
// diag covariance matrix will be accumulated - only upper triangle stored
__global__ void accStatsDiagKernel (GMMStatsEstimator_GPU::param p,
								    GMMStatsEstimator_GPU::likes l,
								    GMMStatsEstimator_GPU::stats s,
								    unsigned int nummix, 								   
								    unsigned int shiftFrames)
{
	const unsigned int FRAME_BLOCK = GMMStatsEstimator_GPU::FRAME_BLOCK;
	const unsigned int DATA_BLOCK = GMMStatsEstimator_GPU::DATA_BLOCK;
	const unsigned int GAUSS_BLOCK = GMMStatsEstimator_GPU::GAUSS_BLOCK;

	unsigned int ix = blockIdx.x;
	unsigned int iy = blockIdx.y % (nummix/GAUSS_BLOCK);
	unsigned int iz = blockIdx.y / (nummix/GAUSS_BLOCK);
	unsigned int blockDimZ = (gridDim.y * GAUSS_BLOCK) / nummix;
	unsigned int tid = threadIdx.x;

	unsigned int NSamples = (shiftFrames + p.Nframes_processed < p.Nframes) ? p.Nframes_processed : p.Nframes - shiftFrames;

	__shared__ float buff[2 * DATA_BLOCK];
	__shared__ float *buff2;

	float accMean[4], accVar[4];
	float gamma[FRAME_BLOCK];

#pragma unroll
	for(int d=0;d<4;d++) {
		accMean[d] = 0.0f;
		accVar[d] = 0.0f;
	}
	float mixProb = 0.0f;
	
	buff2 = buff + DATA_BLOCK; //shared data^2 buffer

	//unsigned int NframesInBlock = GMMStatsEstimator_GPU::alignUP<unsigned int> (NSamples / FRAME_BLOCK, blockDimZ) / blockDimZ;
	unsigned int NframesInBlock = ALIGN_UP(NSamples / FRAME_BLOCK, blockDimZ) / blockDimZ;	
	unsigned int limit = min(NSamples / FRAME_BLOCK, NframesInBlock * (iz + 1));

	unsigned int offset = NframesInBlock * iz * (FRAME_BLOCK / 4) * nummix + iy * blockDim.x + tid;
	for(unsigned int i = NframesInBlock * iz; i < limit; i++) 
	{		
		__syncthreads();

		unsigned int dataOffset = shiftFrames * p.dim + ix * DATA_BLOCK + i * p.dim * FRAME_BLOCK;
		float tmp = p.d_vecs[dataOffset + tid];
		buff[tid] = tmp;
		buff2[tid] = tmp * tmp;
		
		__syncthreads();
		
		FLOAT4toREGARRAY(gamma, 0, l.d_gammas[offset]);
		FLOAT4toREGARRAY(gamma, 4, l.d_gammas[offset + nummix]);
		//((float4*)gamma)[0] = l.d_gammas[offset];
		//((float4*)gamma)[1] = l.d_gammas[offset + nummix];

		offset += 2 * nummix;

		if (ix == 0) 
		{
#pragma unroll
			for(int f = 0; f < FRAME_BLOCK; f++) {
				mixProb += gamma[f];
#pragma unroll
				for(int d = 0; d < 4; d++)
				{
					accMean[d] += gamma[f] * buff[FRAME_BLOCK * d + f];
					accVar[d] += gamma[f] * buff2[FRAME_BLOCK * d + f];
				} 
			}
		} 
		else 
		{
#pragma unroll
			for(int f = 0; f < FRAME_BLOCK; f++) {
#pragma unroll
				for(int d=0;d<4;d++) 
				{
					accMean[d] += gamma[f] * buff[FRAME_BLOCK * d + f];
					accVar[d] += gamma[f] * buff2[FRAME_BLOCK * d + f];
				}
			}
		}
	} //end for i

	nummix = s.statProbSize; //store according to full alloc size of the model
	unsigned int addr = (iz * gridDim.x * nummix) + (iy * gridDim.x * blockDim.x) + ix + gridDim.x * tid;
		
	float4 foo = ((float4 *) s.d_meanStats)[addr];
	((float4 *) s.d_meanStats)[addr] = make_float4(accMean[0] + foo.x, accMean[1] + foo.y, accMean[2] + foo.z, accMean[3] + foo.w);

	foo = ((float4 *) s.d_varStats)[addr];
	((float4 *) s.d_varStats)[addr] = make_float4(accVar[0] + foo.x, accVar[1] + foo.y, accVar[2] + foo.z, accVar[3] + foo.w);

	if (ix == 0)
		s.d_mixProb[iz * nummix + iy * blockDim.x + tid] += mixProb;
}


// gammasKernel <<< dim3(alignedNframes/FRAME_BLOCK, alignedNmix/GAUSS_BLOCK), GAUSS_BLOCK, FRAME_BLOCK*_model->dim*sizeof(float) >>> 
//__global__ void gammasKernel(float4 *gammas, int dim, unsigned int shiftFrames)
__global__ void gammasKernel (GMMStatsEstimator_GPU::model m, 
							  GMMStatsEstimator_GPU::param p, 
							  GMMStatsEstimator_GPU::likes l, 
							  unsigned int shiftFrames)
{	
	unsigned int ix = blockIdx.x;
	unsigned int iy = blockIdx.y;
	unsigned int tid = threadIdx.x;

	const unsigned int FRAME_BLOCK = GMMStatsEstimator_GPU::FRAME_BLOCK;
	const unsigned int DATA_BLOCK = GMMStatsEstimator_GPU::DATA_BLOCK;
	const unsigned int DIM_BLOCK = GMMStatsEstimator_GPU::DIM_BLOCK;

	//__shared__ float xx[FRAME_BLOCK * MAX_DIM];
	extern __shared__ float xx[];

	//load param into shared memory
	for(unsigned int m = 0; m < p.dim / DIM_BLOCK; m++) { 
		__syncthreads();
		xx[m * DATA_BLOCK + tid] = p.d_vecs[shiftFrames * p.dim + ix * FRAME_BLOCK * p.dim + m * DATA_BLOCK + tid];
	}
		
	// array in registry - accumulated exponents (along dimension) of gaussians for 8 frames
	float acc[FRAME_BLOCK];

#pragma unroll
	for(unsigned int i = 0; i < FRAME_BLOCK; i++) {
		acc[i] = 0.0f;
	}
	// set addres index to begin of block
	unsigned int addr = iy * blockDim.x * p.dim/4; // note: float4 => dim/4

	// cycle through all dimensions
	for(unsigned int k = 0; k < p.dim / DIM_BLOCK; k++) {

		// load model
		__syncthreads();

		float mean[4], ivar[4];
		((float4*) mean)[0] = m.d_means[addr + tid];
		((float4*) ivar)[0] = m.d_ivars[addr + tid];


		// first/second/third/fourth dimension in the block - 8 frames are processed
#pragma unroll
		for(unsigned int j = 0; j < 4; j++) {
#pragma unroll
			for(unsigned int i = 0; i < FRAME_BLOCK; i++) {			
				float tmp = xx[DATA_BLOCK * k + j*FRAME_BLOCK + i] - mean[j];
				acc[i] += tmp * tmp * ivar[j];
			}
		}			
		addr += blockDim.x;
	}

	__syncthreads();
	float gc = m.d_Gconsts[iy * blockDim.x + tid];

	// finalise and add LogProb
#pragma unroll		
	for(unsigned int i = 0; i < FRAME_BLOCK; i++) {
		acc[i] = -0.5f * acc[i] + gc;
	}
	__syncthreads();

	// store results
	addr = 2 * ix * gridDim.y * blockDim.x + iy * blockDim.x;
	l.d_gammas[addr + tid] = ((float4*) acc)[0]; // make_float4(acc00, acc01, acc02, acc03);
	l.d_gammas[addr + gridDim.y * blockDim.x + tid] = ((float4*) acc)[1]; //make_float4(acc04, acc05, acc06, acc07);
}

// gammasKernelLargeDim <<< dim3(alignedNframes/FRAME_BLOCK, alignedNmix/GAUSS_BLOCK), GAUSS_BLOCK, 1024*sizeof(float) >>> 
__global__ void gammasKernelLargeDim (GMMStatsEstimator_GPU::model m, 
							  GMMStatsEstimator_GPU::param p, 
							  GMMStatsEstimator_GPU::likes l, 
							  unsigned int shiftFrames)
{	
	unsigned int ix = blockIdx.x;
	unsigned int iy = blockIdx.y;
	unsigned int tid = threadIdx.x;

	const unsigned int FRAME_BLOCK = GMMStatsEstimator_GPU::FRAME_BLOCK;
	const unsigned int DATA_BLOCK = GMMStatsEstimator_GPU::DATA_BLOCK;
	const unsigned int DIM_BLOCK = GMMStatsEstimator_GPU::DIM_BLOCK;
	const unsigned int DIMS_PER_BLOCK = 1024/DATA_BLOCK;
	
	int dimBlocks = ALIGN_DIV(p.dim/DIM_BLOCK, DIMS_PER_BLOCK);

	//__shared__ float xx[FRAME_BLOCK * MAX_DIM];
	extern __shared__ float xx[];

	// array in registry - accumulated exponents (along dimension) of gaussians for 8 frames
	float acc[FRAME_BLOCK];

	#pragma unroll
	for(unsigned int i = 0; i < FRAME_BLOCK; i++) {
		acc[i] = 0.0f;
	}

	// set addres index to begin of block
	unsigned int addr = iy * blockDim.x * p.dim/4; // note: float4 => dim/4
	
	//main BLOCKSDIM loop - for large dimesions
	for(int dm = 0; dm < dimBlocks; dm++) {

		//load param into shared memory
		int maxDim = min(DIMS_PER_BLOCK, p.dim/DIM_BLOCK - dm * DIMS_PER_BLOCK);
		for(unsigned int d = 0; d < maxDim; d++) { 
			__syncthreads();
			xx[d * DATA_BLOCK + tid] = p.d_vecs[shiftFrames * p.dim + ix * FRAME_BLOCK * p.dim + (d+DIMS_PER_BLOCK*dm) * DATA_BLOCK + tid];
		}
			
		// cycle through all dimensions
		for(unsigned int k = 0; k < maxDim; k++) {

			// load model
			__syncthreads();

			float mean[4], ivar[4];
			((float4*) mean)[0] = m.d_means[addr + tid];
			((float4*) ivar)[0] = m.d_ivars[addr + tid];


			// first/second/third/fourth dimension in the block - 8 frames are processed
	#pragma unroll
			for(unsigned int j = 0; j < 4; j++) {
	#pragma unroll
				for(unsigned int i = 0; i < FRAME_BLOCK; i++) {			
					float tmp = xx[DATA_BLOCK * k + j*FRAME_BLOCK + i] - mean[j];
					acc[i] += tmp * tmp * ivar[j];
				}
			}			
			addr += blockDim.x;
		}//for k
	}//for dm

	__syncthreads();
	float gc = m.d_Gconsts[iy * blockDim.x + tid];

	// finalise and add LogProb
#pragma unroll		
	for(unsigned int i = 0; i < FRAME_BLOCK; i++) {
		acc[i] = -0.5f * acc[i] + gc;
	}
	__syncthreads();

	// store results
	addr = 2 * ix * gridDim.y * blockDim.x + iy * blockDim.x;
	l.d_gammas[addr + tid] = ((float4*) acc)[0]; // make_float4(acc00, acc01, acc02, acc03);
	l.d_gammas[addr + gridDim.y * blockDim.x + tid] = ((float4*) acc)[1]; //make_float4(acc04, acc05, acc06, acc07);
}


// gammasKernel <<< dim3(alignedNframes/FRAME_BLOCK, alignedNmix/GAUSS_BLOCK), GAUSS_BLOCK, FRAME_BLOCK*_model->dim*sizeof(float) >>> 
// (float4 *gammas, int dim, int dimV_full, unsigned int shiftFrames)
__global__ void gammasKernelFull (GMMStatsEstimator_GPU::model m, 
								  GMMStatsEstimator_GPU::param p, 
								  GMMStatsEstimator_GPU::likes l, 
								  unsigned int shiftFrames)
{	
	unsigned int ix = blockIdx.x;
	unsigned int iy = blockIdx.y;
	unsigned int tid = threadIdx.x;

	const unsigned int FRAME_BLOCK = GMMStatsEstimator_GPU::FRAME_BLOCK;
	const unsigned int DATA_BLOCK = GMMStatsEstimator_GPU::DATA_BLOCK;
	const unsigned int DIM_BLOCK = GMMStatsEstimator_GPU::DIM_BLOCK;

	extern __shared__ float xx[];

	//load param into shared memory
	for(unsigned int d = 0; d < p.dim / DIM_BLOCK; d++) { 
		//__syncthreads();
		xx[d * DATA_BLOCK + tid] = p.d_vecs[shiftFrames * p.dim + ix * FRAME_BLOCK * p.dim + d * DATA_BLOCK + tid];
	}
	__syncthreads();
		
	// array in registry - accumulated exponents (along dimension) of gaussians for 8 frames
	float acc[FRAME_BLOCK];

#pragma unroll
	for(unsigned int i = 0; i < FRAME_BLOCK; i++) {
		acc[i] = 0.0f;
	}
	// set addres index to begin of block
	unsigned int addr = iy * blockDim.x * p.dim/4 + tid; // note: float4 => p.dim/4
	unsigned int addrV = iy * blockDim.x * m.dimVar/4 + tid; // note: float4

	// cycle through all dimensions
	for(unsigned int k = 0; k < p.dim / DIM_BLOCK; k++) {

		__syncthreads();
		
		// load model
		float mean[4], mean2[4], ivar[4];
		((float4*) mean)[0] = m.d_means[addr + k * blockDim.x];

		for(unsigned int kk = 0; kk <= k; kk++) {
			if (k == kk)
				((float4*) mean2)[0] = ((float4*) mean)[0];
			else
				((float4*) mean2)[0] = m.d_means[addr + kk * blockDim.x];
			
#pragma unroll
			for(int z = 0; z < DIM_BLOCK; z++) 
			{				
				((float4*) ivar)[0] = m.d_ivars[addrV];
				addrV += blockDim.x;

#pragma unroll
				for(int x = 0; x < FRAME_BLOCK; x++) {
					float tmp = xx[DATA_BLOCK * k + x + FRAME_BLOCK * z] - mean[z];
#pragma unroll
					for(int y = 0;y < DIM_BLOCK; y++) {
						acc[x] += tmp * (xx[DATA_BLOCK * kk + x + FRAME_BLOCK * y] - mean2[y]) * ivar[y];
					}
				}
				__syncthreads();
			}
		}
	}

	__syncthreads();
	float gc = m.d_Gconsts[iy * blockDim.x + tid];

	// finalise and add LogProb
#pragma unroll		
	for(unsigned int i = 0; i < FRAME_BLOCK; i++) {
		acc[i] = -0.5f * acc[i] + gc;
	}
	__syncthreads();

	// store results
	addr = 2 * ix * gridDim.y * blockDim.x + iy * blockDim.x;
	l.d_gammas[addr + tid] = ((float4*) acc)[0]; // make_float4(acc00, acc01, acc02, acc03);
	l.d_gammas[addr + gridDim.y * blockDim.x + tid] = ((float4*) acc)[1]; //make_float4(acc04, acc05, acc06, acc07);
}



// logLikeKernel <<< alignedNframes/4, GAUSS_BLOCK >>>
__global__ void logLikeKernel (GMMStatsEstimator_GPU::likes l, 
							   unsigned int nummix, unsigned int nummixAligned, 
							   float minLogLike) 
{	
	int ix = blockIdx.x;
	int tid = threadIdx.x;
	const int numSteps = nummixAligned/blockDim.x;
	float4 minLL = make_float4(minLogLike, minLogLike, minLogLike, minLogLike);
	float4 tmp;
	l.d_gammas += ix * nummixAligned; //shift gammas pointer	

	const unsigned int GAUSS_BLOCK = GMMStatsEstimator_GPU::DATA_BLOCK;

#ifdef KERNEL_GPU_SAFE_MODE
	float4 sumLL = (tid < nummix) ? l.d_gammas[tid] : minLL;	
#else
	unsigned int offset = ix * nummixAligned + tid;
	//float4 sumLL = gammas[tid];
	float4 sumLL = l.d_gammas[offset];
#endif

	for(unsigned int i = 1; i < numSteps; i++) {	
#ifdef KERNEL_GPU_SAFE_MODE
		tmp = (blockDim.x*i+tid<nummix)? l.d_gammas[i * blockDim.x + tid] : minLL;
#else
		//tmp = l.d_gammas[i * blockDim.x + tid];
		offset += blockDim.x;
		tmp = l.d_gammas[offset];
#endif
		sumLL = addLog4(sumLL, tmp);
	}

	// parallel sum
	__shared__ float buff[4 * GAUSS_BLOCK];
	
	((float4*)buff)[tid] = sumLL; //bank conflicts!
	__syncthreads();

	buff[tid] = addLog(buff[tid], buff[GAUSS_BLOCK+tid]);

	__syncthreads();
	buff[32+tid] = addLog(buff[2*GAUSS_BLOCK+tid], buff[3*GAUSS_BLOCK+tid]);
	__syncthreads();
	if(GAUSS_BLOCK >= 256) {
		buff[tid] = addLog(buff[tid], buff[256+tid]);
		__syncthreads();
	}
	if(GAUSS_BLOCK >= 128) {
		buff[tid] = addLog(buff[tid], buff[128+tid]);
		__syncthreads();
	}
	if(GAUSS_BLOCK >= 64) {
		buff[tid] = addLog(buff[tid], buff[64+tid]);
		__syncthreads();
	}
	if(tid<32) {
		volatile float *_buff = buff;
		_buff[tid] = addLog(_buff[tid], _buff[32+tid]);
		_buff[tid] = addLog(_buff[tid], _buff[16+tid]);
		_buff[tid] = addLog(_buff[tid], _buff[8+tid]);
		_buff[tid] = addLog(_buff[tid], _buff[4+tid]);
	}

	// store logLike
	if(tid < 4) {
//#ifdef KERNEL_GPU_SAFE_MODE
		float tmp1 = buff[tid];
		checkLogLike(tmp1, minLogLike);
		((float*)&(l.d_ll[blockIdx.x]))[tid] = tmp1;
//#else
//		((float*)&(l.d_ll[blockIdx.x]))[tid] = buff[tid];
//#endif
	}
}

//<<< NS_GB_aligned / GAUSS_BLOCK, GAUSS_BLOCK >>>
__global__ void sumLogLike (GMMStatsEstimator_GPU::likes l, 
							unsigned int NSamples, unsigned int NS_aligned)
{
	unsigned int ix = blockIdx.x;
	unsigned int tid = threadIdx.x;

	float4 buffLL={0,0,0,0};

	unsigned int x = ix * blockDim.x + tid;
	if (x < NS_aligned / 4)
		buffLL = l.d_ll[x];

	buffLL.x = (4 * x + 0 < NSamples)? buffLL.x : 0.0f;
	buffLL.y = (4 * x + 1 < NSamples)? buffLL.y : 0.0f;
	buffLL.z = (4 * x + 2 < NSamples)? buffLL.z : 0.0f;
	buffLL.w = (4 * x + 3 < NSamples)? buffLL.w : 0.0f;

	float totLL = buffLL.x + buffLL.y + buffLL.z + buffLL.w;
	LOCAL_REDUCTION(totLL, sh_buff, GMMStatsEstimator_GPU::GAUSS_BLOCK, 32)
	if(tid==0)
		atomicAdd(l.d_totll, totLL);


	////LLLL:
	//__shared__ float sum;
	//if(tid == 0) sum = 0.0f;
	//__syncthreads();

	//atomicAdd(&sum, buffLL.x + buffLL.y + buffLL.z + buffLL.w);

	//__syncthreads();

	//if(tid==0)
	//	atomicAdd(l.d_totll, sum);
}



//<<< alignedNframes/FRAME_BLOCK, GAUSS_BLOCK >>> 
__global__ void gammasNormKernel(GMMStatsEstimator_GPU::likes l, 
								 GMMStatsEstimator_GPU::OPTIONS opt,
								 unsigned int nummixAligned, unsigned int NSamples_unaligned) 
{
	int ix = blockIdx.x;
	int tid = threadIdx.x;
	const int numSteps = nummixAligned/blockDim.x;

	const unsigned int FRAME_BLOCK = GMMStatsEstimator_GPU::FRAME_BLOCK;
	const unsigned int GAUSS_BLOCK = GMMStatsEstimator_GPU::DATA_BLOCK;

	float gamma[FRAME_BLOCK];
	__shared__ float buffLL[FRAME_BLOCK];				


	unsigned int offset = ix * 2 * nummixAligned + tid;		

	if(tid < 2)	
		//((float4*)&buffLL)[tid] = l.d_ll[2 * ix + tid];
		FLOAT4toREGARRAY(buffLL, 4*tid, l.d_ll[2 * ix + tid]);
	
	__syncthreads();

	if(ix < gridDim.x-1) { //all block except the last one

		for(int i=0; i<numSteps; i++) {

			//load gammas
			//((float4*)gamma)[0] = l.d_gammas[offset];
			//((float4*)gamma)[1] = l.d_gammas[offset+nummixAligned];
			FLOAT4toREGARRAY(gamma, 0, l.d_gammas[offset]);
			FLOAT4toREGARRAY(gamma, 4, l.d_gammas[offset+nummixAligned]);			

			// normalize and check minLogLike threshold
#pragma unroll
			for(int f=0;f<FRAME_BLOCK;f++) {
				gamma[f] = (buffLL[f] > opt.minLogLike) * __expf(gamma[f] - buffLL[f]);
				gamma[f] *= (gamma[f] > opt.minGamma);
			}

			//store gammas
			l.d_gammas[offset] = make_float4_REGARRAY(gamma, 0); //((float4*)gamma)[0];
			l.d_gammas[offset+nummixAligned] = make_float4_REGARRAY(gamma, 4); //((float4*)gamma)[1];
			
			offset += GAUSS_BLOCK;
		} //end for i

	} 
	else 
	{ //last block
		for(int i=0; i<numSteps; i++) 
		{
			//load gammas
			//((float4*)gamma)[0] = l.d_gammas[offset];
			//((float4*)gamma)[1] = l.d_gammas[offset + nummixAligned];
			FLOAT4toREGARRAY(gamma, 0, l.d_gammas[offset]);
			FLOAT4toREGARRAY(gamma, 4, l.d_gammas[offset+nummixAligned]);

			// normalize and check minLogLike threshold
#pragma unroll
			for(int f=0;f<FRAME_BLOCK;f++) {
				gamma[f] = (buffLL[f] > opt.minLogLike) * __expf(gamma[f] - buffLL[f]);
				gamma[f] *= (gamma[f] > opt.minGamma);
				gamma[f] *= (ix * FRAME_BLOCK + f < NSamples_unaligned); //LAST BLOCK modification
			}

			//store gammas
			l.d_gammas[offset] = make_float4_REGARRAY(gamma, 0); //((float4*) gamma)[0];
			l.d_gammas[offset+nummixAligned] = make_float4_REGARRAY(gamma, 4); //((float4*) gamma)[1];
			
			offset += GAUSS_BLOCK;
		} //end for i
	} //end last block
} //gammasNormKernel



// <<< array_size / 4 / NTHRDS, NTHRDS >>>
__global__ void addArrays4Kernel (float* d_arr, unsigned int Nblocks, unsigned int Bsize)
{		
	unsigned int ix = blockIdx.x;
	unsigned int tid = threadIdx.x;
	
	unsigned int offset = ix * blockDim.x;

	float4 sum = ((float4*) d_arr) [offset + tid];
	for (unsigned int i = 1; i < Nblocks; i++) {
		__syncthreads();
		float4 foo = ((float4*) d_arr) [offset + i * (Bsize / 4) + tid];
		sum.x += foo.x;
		sum.y += foo.y;
		sum.z += foo.z;
		sum.w += foo.w;
	}

	__syncthreads();
	
	FLOAT4toREGARRAY(d_arr, 4*(offset + tid), sum);
	//((float4*) d_arr) [offset + tid] = sum;
}
// <<< array_size / NTHRDS, NTHRDS >>>
__global__ void addArraysKernel (float* d_arr, unsigned int Nblocks, unsigned int Bsize)
{		
	unsigned int ix = blockIdx.x;
	unsigned int tid = threadIdx.x;
	
	unsigned int offset = ix * blockDim.x;

	float sum = d_arr[offset + tid];
	for (unsigned int i = 1; i < Nblocks; i++) {
		__syncthreads();		
		sum += d_arr [offset + i * Bsize + tid];
	}

	__syncthreads();
	
	d_arr[offset + tid] = sum;
}


//gammas are non-log (after exp) - normal multiply and sum
// normAuxGammasKernel <<< alignedNframes/4, GAUSS_BLOCK >>>
__global__ void normAuxGammasKernel (GMMStatsEstimator_GPU::likes l, 
							   unsigned int nummix, unsigned int nummixAligned, float minGamma) 
{	
	int ix = blockIdx.x;
	int tid = threadIdx.x;
	const int numSteps = nummixAligned/blockDim.x;
	l.d_gammas += ix * nummixAligned; //shift gammas pointer	
	float4 *old_gammas =  l.d_gammas - nummixAligned;

	const unsigned int GAUSS_BLOCK = GMMStatsEstimator_GPU::DATA_BLOCK;

	float sumLL[4];
	float4 gL = make_float4(0,0,0,0);
	float gbuff[5];
#pragma unroll
		for(int f=0; f<4; f++) sumLL[f] = 0;

	for(unsigned int i = 0; i < numSteps; i++) {
		  
		if(ix > 0) gL = old_gammas[i * blockDim.x + tid];
		FLOAT4toREGARRAY(gbuff, 1, l.d_gammas[i * blockDim.x + tid]);
		gbuff[0] = gL.w;
#pragma unroll
		for(int f=0; f<4; f++) {
			float g = gbuff[f]*gbuff[f+1];
			sumLL[f] += (g > minGamma) * g;
		}  
	}

	// parallel sum
	__shared__ float buff[4 * GAUSS_BLOCK];
	
	((float4*)buff)[tid] = make_float4_REGARRAY(sumLL, 0); //bank conflicts!
	__syncthreads();

	buff[tid] += buff[GAUSS_BLOCK+tid];

	__syncthreads();
	buff[32+tid] = buff[2*GAUSS_BLOCK+tid] + buff[3*GAUSS_BLOCK+tid];
	__syncthreads();
	if(GAUSS_BLOCK >= 256) {
		buff[tid] += buff[256+tid];
		__syncthreads();
	}
	if(GAUSS_BLOCK >= 128) {
		buff[tid] += buff[128+tid];
		__syncthreads();
	}
	if(GAUSS_BLOCK >= 64) {
		buff[tid] += buff[64+tid];
		__syncthreads();
	}
	if(tid<32) {
		volatile float *_buff = buff;
		_buff[tid] += _buff[32+tid];
		_buff[tid] += _buff[16+tid];
		_buff[tid] += _buff[8+tid];
		_buff[tid] += _buff[4+tid];
	}

	// store logLike
	if(tid < 4) {
		float tmp1 = buff[tid];
		((float*)&(l.d_aux_ll[blockIdx.x]))[tid] = tmp1;
	}
}

#endif
