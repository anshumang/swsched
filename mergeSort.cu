/*
 * Copyright 1993-2014 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */




/*
 * Based on "Designing efficient sorting algorithms for manycore GPUs"
 * by Nadathur Satish, Mark Harris, and Michael Garland
 * http://mgarland.org/files/papers/gpusort-ipdps09.pdf
 *
 * Victor Podlozhnyuk 09/24/2009
 */



#include <assert.h>
#include <helper_cuda.h>
#include "mergeSort_common.h"
#include <iostream>
#include <sys/time.h>


////////////////////////////////////////////////////////////////////////////////
// Helper functions
////////////////////////////////////////////////////////////////////////////////
static inline __host__ __device__ uint iDivUp(uint a, uint b)
{
    return ((a % b) == 0) ? (a / b) : (a / b + 1);
}

static inline __host__ __device__ uint getSampleCount(uint dividend)
{
    return iDivUp(dividend, SAMPLE_STRIDE);
}

#define W (sizeof(uint) * 8)
static inline __device__ uint nextPowerOfTwo(uint x)
{
    /*
        --x;
        x |= x >> 1;
        x |= x >> 2;
        x |= x >> 4;
        x |= x >> 8;
        x |= x >> 16;
        return ++x;
    */
    return 1U << (W - __clz(x - 1));
}

template<uint sortDir> static inline __device__ uint binarySearchInclusive(uint val, uint *data, uint L, uint stride)
{
    if (L == 0)
    {
        return 0;
    }

    uint pos = 0;

    for (; stride > 0; stride >>= 1)
    {
        uint newPos = umin(pos + stride, L);

        if ((sortDir && (data[newPos - 1] <= val)) || (!sortDir && (data[newPos - 1] >= val)))
        {
            pos = newPos;
        }
    }

    return pos;
}

template<uint sortDir> static inline __device__ uint binarySearchExclusive(uint val, uint *data, uint L, uint stride)
{
    if (L == 0)
    {
        return 0;
    }

    uint pos = 0;

    for (; stride > 0; stride >>= 1)
    {
        uint newPos = umin(pos + stride, L);

        if ((sortDir && (data[newPos - 1] < val)) || (!sortDir && (data[newPos - 1] > val)))
        {
            pos = newPos;
        }
    }

    return pos;
}



////////////////////////////////////////////////////////////////////////////////
// Bottom-level merge sort (binary search-based)
////////////////////////////////////////////////////////////////////////////////
template<uint sortDir> __global__ void mergeSortSharedKernel(
    uint *d_DstKey,
    uint *d_DstVal,
    uint *d_SrcKey,
    uint *d_SrcVal,
    uint arrayLength
)
{
    __shared__ uint s_key[SHARED_SIZE_LIMIT];
    __shared__ uint s_val[SHARED_SIZE_LIMIT];

    d_SrcKey += blockIdx.x * SHARED_SIZE_LIMIT + threadIdx.x;
    d_SrcVal += blockIdx.x * SHARED_SIZE_LIMIT + threadIdx.x;
    d_DstKey += blockIdx.x * SHARED_SIZE_LIMIT + threadIdx.x;
    d_DstVal += blockIdx.x * SHARED_SIZE_LIMIT + threadIdx.x;
    s_key[threadIdx.x +                       0] = d_SrcKey[                      0];
    s_val[threadIdx.x +                       0] = d_SrcVal[                      0];
    s_key[threadIdx.x + (SHARED_SIZE_LIMIT / 2)] = d_SrcKey[(SHARED_SIZE_LIMIT / 2)];
    s_val[threadIdx.x + (SHARED_SIZE_LIMIT / 2)] = d_SrcVal[(SHARED_SIZE_LIMIT / 2)];

    for (uint stride = 1; stride < arrayLength; stride <<= 1)
    {
        uint     lPos = threadIdx.x & (stride - 1);
        uint *baseKey = s_key + 2 * (threadIdx.x - lPos);
        uint *baseVal = s_val + 2 * (threadIdx.x - lPos);

        __syncthreads();
        uint keyA = baseKey[lPos +      0];
        uint valA = baseVal[lPos +      0];
        uint keyB = baseKey[lPos + stride];
        uint valB = baseVal[lPos + stride];
        uint posA = binarySearchExclusive<sortDir>(keyA, baseKey + stride, stride, stride) + lPos;
        uint posB = binarySearchInclusive<sortDir>(keyB, baseKey +      0, stride, stride) + lPos;

        __syncthreads();
        baseKey[posA] = keyA;
        baseVal[posA] = valA;
        baseKey[posB] = keyB;
        baseVal[posB] = valB;
    }

    __syncthreads();
    d_DstKey[                      0] = s_key[threadIdx.x +                       0];
    d_DstVal[                      0] = s_val[threadIdx.x +                       0];
    d_DstKey[(SHARED_SIZE_LIMIT / 2)] = s_key[threadIdx.x + (SHARED_SIZE_LIMIT / 2)];
    d_DstVal[(SHARED_SIZE_LIMIT / 2)] = s_val[threadIdx.x + (SHARED_SIZE_LIMIT / 2)];
}

static void mergeSortShared(
    uint *d_DstKey,
    uint *d_DstVal,
    uint *d_SrcKey,
    uint *d_SrcVal,
    uint batchSize,
    uint arrayLength,
    uint sortDir
)
{
    if (arrayLength < 2)
    {
        return;
    }

    assert(SHARED_SIZE_LIMIT % arrayLength == 0);
    assert(((batchSize * arrayLength) % SHARED_SIZE_LIMIT) == 0);
    uint  blockCount = batchSize * arrayLength / SHARED_SIZE_LIMIT;
    uint threadCount = SHARED_SIZE_LIMIT / 2;

    if (sortDir)
    {
	    cudaError_t k_err;
	    int num_blocks=0;
	    k_err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&num_blocks, mergeSortSharedKernel<1U>, threadCount, 0); 
	    if(k_err != cudaSuccess){
		    std::cerr << "cudaOccupancyMaxActiveBlocksPerMultiprocessor failed with error " << k_err << std::endl;}
	    std::cerr << "mergeSortSharedKernel occ " << num_blocks << std::endl;
	struct timeval start, end;
	gettimeofday(&start, NULL);
        mergeSortSharedKernel<1U><<<blockCount, threadCount>>>(d_DstKey, d_DstVal, d_SrcKey, d_SrcVal, arrayLength);
	cudaDeviceSynchronize();
	gettimeofday(&end, NULL);
	std::cerr << "mergeSortShared " << blockCount << " " << threadCount << " " << (end.tv_sec - start.tv_sec)*1000000 + (end.tv_usec - start.tv_usec) << std::endl;
        getLastCudaError("mergeSortShared<1><<<>>> failed\n");
    }
    else
    {
        mergeSortSharedKernel<0U><<<blockCount, threadCount>>>(d_DstKey, d_DstVal, d_SrcKey, d_SrcVal, arrayLength);
        //getLastCudaError("mergeSortShared<0><<<>>> failed\n");
    }
}

template<uint sortDir> __device__ void mergeSortSharedKernel_minion(
    uint *d_DstKey,
    uint *d_DstVal,
    uint *d_SrcKey,
    uint *d_SrcVal,
    uint arrayLength,
    uint num_iter
)
{
    __shared__ uint s_key[SHARED_SIZE_LIMIT];
    __shared__ uint s_val[SHARED_SIZE_LIMIT];

    size_t new_block_idx_x = blockIdx.x + num_iter*60;
    //d_SrcKey += blockIdx.x * SHARED_SIZE_LIMIT + threadIdx.x;
    //d_SrcVal += blockIdx.x * SHARED_SIZE_LIMIT + threadIdx.x;
    //d_DstKey += blockIdx.x * SHARED_SIZE_LIMIT + threadIdx.x;
    //d_DstVal += blockIdx.x * SHARED_SIZE_LIMIT + threadIdx.x;
    d_SrcKey += new_block_idx_x * SHARED_SIZE_LIMIT + threadIdx.x;
    d_SrcVal += new_block_idx_x * SHARED_SIZE_LIMIT + threadIdx.x;
    d_DstKey += new_block_idx_x * SHARED_SIZE_LIMIT + threadIdx.x;
    d_DstVal += new_block_idx_x * SHARED_SIZE_LIMIT + threadIdx.x;
    s_key[threadIdx.x +                       0] = d_SrcKey[                      0];
    s_val[threadIdx.x +                       0] = d_SrcVal[                      0];
    s_key[threadIdx.x + (SHARED_SIZE_LIMIT / 2)] = d_SrcKey[(SHARED_SIZE_LIMIT / 2)];
    s_val[threadIdx.x + (SHARED_SIZE_LIMIT / 2)] = d_SrcVal[(SHARED_SIZE_LIMIT / 2)];

    for (uint stride = 1; stride < arrayLength; stride <<= 1)
    {
        uint     lPos = threadIdx.x & (stride - 1);
        uint *baseKey = s_key + 2 * (threadIdx.x - lPos);
        uint *baseVal = s_val + 2 * (threadIdx.x - lPos);

        __syncthreads();
        uint keyA = baseKey[lPos +      0];
        uint valA = baseVal[lPos +      0];
        uint keyB = baseKey[lPos + stride];
        uint valB = baseVal[lPos + stride];
        uint posA = binarySearchExclusive<sortDir>(keyA, baseKey + stride, stride, stride) + lPos;
        uint posB = binarySearchInclusive<sortDir>(keyB, baseKey +      0, stride, stride) + lPos;

        __syncthreads();
        baseKey[posA] = keyA;
        baseVal[posA] = valA;
        baseKey[posB] = keyB;
        baseVal[posB] = valB;
    }

    __syncthreads();
    d_DstKey[                      0] = s_key[threadIdx.x +                       0];
    d_DstVal[                      0] = s_val[threadIdx.x +                       0];
    d_DstKey[(SHARED_SIZE_LIMIT / 2)] = s_key[threadIdx.x + (SHARED_SIZE_LIMIT / 2)];
    d_DstVal[(SHARED_SIZE_LIMIT / 2)] = s_val[threadIdx.x + (SHARED_SIZE_LIMIT / 2)];
}

__device__ uint d_d2h_flag=0;
__device__ uint d_num_blocks_completed=0;
__device__ uint d_num_blocks_started=0;
__device__ uint d_h2d_flag=0;
__device__ long d_grid_size=60;

__global__ void super_kernel(
    uint *d_cm_flag,
    uint *d_DstKey,
    uint *d_DstVal,
    uint *d_SrcKey,
    uint *d_SrcVal,
    uint arrayLength)
{
     long long int l_timestamp = 0;
     unsigned int smidx, warpidx;
     /*if((blockIdx.x==0)||(blockIdx.x==15)||(blockIdx.x==30)||(blockIdx.x==45)||(blockIdx.x==60))
     {
       asm("mov.u32 %0, %%smid;" : "=r"(smidx));
       asm("mov.u32 %0, %%warpid;" : "=r"(warpidx));
       l_timestamp = clock64();
       //printf("%d %d %d %d %ld\n",smidx, warpidx, gridDim.x, blockDim.x, l_timestamp);
       printf("%d %d %d %d %d %d %ld\n",threadIdx.x, blockIdx.x, smidx, warpidx, gridDim.x, blockDim.x, l_timestamp);
     }*/
     if(threadIdx.x==0)
     {
        asm("mov.u32 %0, %%smid;" : "=r"(smidx));
        asm("mov.u32 %0, %%warpid;" : "=r"(warpidx));
        l_timestamp = clock64();
        /*if(blockIdx.x%60==0)
        {
          printf("start %d %d %d %ld\n", blockIdx.x, smidx, warpidx, l_timestamp);
        }*/
        uint l_h2d_flag=0;
        do
        {
           l_h2d_flag = *d_cm_flag;
           __threadfence_system();
        }while(l_h2d_flag==0);
      }
     __syncthreads();
     __shared__ bool s_grid_done;
     int iter=4369/*273*/, iter_completed=0;
    while(iter_completed<4370/*273*/){
     if(iter_completed==4369){
         if(blockIdx.x>=4){ /*4 blocks remaining, so use block indices 0-3, that is, choose 1 block each from 4 SMs (ALTERNATIVE : choose all 4 block indices from 1 SM)*/
            break;
         }
         if(threadIdx.x==0){
         printf("extra dispatch %d %d\n", blockIdx.x, smidx);}
     }
     if(threadIdx.x==0)
     {
	s_grid_done=false;
        //if(blockIdx.x%60==0)
        //{
          //printf("start %d %d %d %ld %ld\n", blockIdx.x, smidx, warpidx, l_timestamp, clock64());
        //}
        //if(d_num_blocks_started == 0)
        //if(/*(blockIdx.x==0)&&*/(iter_completed==0))
        //{
            //printf("start %d %d %d %ld\n", blockIdx.x, smidx, warpidx, clock64());
        //}
        atomicAdd(&d_num_blocks_started, 1);
     }
     __syncthreads();
     mergeSortSharedKernel_minion<1>(d_DstKey, d_DstVal, d_SrcKey, d_SrcVal, arrayLength, iter_completed);
     __syncthreads();
     //if(threadIdx.x==0)
     //{
        //l_timestamp = clock64();
        //printf("end %d %d %ld\n", r, blockIdx.x, l_timestamp);
     //}
     iter_completed++;
     if(threadIdx.x==0)
     {
         atomicAdd(&d_num_blocks_completed, 1);
         //if(d_num_blocks_completed==gridDim.x)
         //if(d_num_blocks_completed==d_grid_size)
         //if(iter_completed==273)
         if((blockIdx.x==0)&&(iter_completed==4369/*273*/))
         {
             s_grid_done=true;
             //printf("end %d %d %d %ld\n", blockIdx.x, smidx, warpidx, clock64());
         }
     }
     __syncthreads();
     /*if(s_grid_done==true){
       break;
     }*/
    }
     /*if((threadIdx.x==0)&&(blockIdx.x==0))
     {
         int l_num_blocks_completed;
         do
         {
           l_num_blocks_completed = atomicCAS(&d_num_blocks_completed, gridDim.x-1 ,0);
         }while(l_num_blocks_completed!=gridDim.x-1);
         atomicExch(&d_d2h_flag, 1);
         printf("%ld \n", clock64());
     }*/
}

template<uint sortDir> __device__ void mergeSortSharedKernel_minion_elastic_blockDim(
    uint *d_DstKey,
    uint *d_DstVal,
    uint *d_SrcKey,
    uint *d_SrcVal,
    uint arrayLength,
    uint num_iter,
    bool partial_dispatch,
    uint factor
)
{
    __shared__ uint s_key[SHARED_SIZE_LIMIT];
    __shared__ uint s_val[SHARED_SIZE_LIMIT];

    uint extra_iter=0;
    if(partial_dispatch){
       extra_iter=1;
    }
    size_t new_block_idx_x = blockIdx.x + num_iter*60 + extra_iter;
    size_t new_thread_idx_x = threadIdx.x*factor; //when supervisor block smaller than app block
    //threadIdx.x%factor; // when app block smaller than supervisor block  
    //d_SrcKey += blockIdx.x * SHARED_SIZE_LIMIT + threadIdx.x;
    //d_SrcVal += blockIdx.x * SHARED_SIZE_LIMIT + threadIdx.x;
    //d_DstKey += blockIdx.x * SHARED_SIZE_LIMIT + threadIdx.x;
    //d_DstVal += blockIdx.x * SHARED_SIZE_LIMIT + threadIdx.x;
    d_SrcKey += new_block_idx_x * SHARED_SIZE_LIMIT + threadIdx.x;
    d_SrcVal += new_block_idx_x * SHARED_SIZE_LIMIT + threadIdx.x;
    d_DstKey += new_block_idx_x * SHARED_SIZE_LIMIT + threadIdx.x;
    d_DstVal += new_block_idx_x * SHARED_SIZE_LIMIT + threadIdx.x;
    s_key[threadIdx.x +                       0] = d_SrcKey[                      0];
    s_val[threadIdx.x +                       0] = d_SrcVal[                      0];
    s_key[threadIdx.x + (SHARED_SIZE_LIMIT / 2)] = d_SrcKey[(SHARED_SIZE_LIMIT / 2)];
    s_val[threadIdx.x + (SHARED_SIZE_LIMIT / 2)] = d_SrcVal[(SHARED_SIZE_LIMIT / 2)];

    for (uint stride = 1; stride < arrayLength; stride <<= 1)
    {
        uint     lPos = threadIdx.x & (stride - 1);
        uint *baseKey = s_key + 2 * (threadIdx.x - lPos);
        uint *baseVal = s_val + 2 * (threadIdx.x - lPos);

        __syncthreads();
        uint keyA = baseKey[lPos +      0];
        uint valA = baseVal[lPos +      0];
        uint keyB = baseKey[lPos + stride];
        uint valB = baseVal[lPos + stride];
        uint posA = binarySearchExclusive<sortDir>(keyA, baseKey + stride, stride, stride) + lPos;
        uint posB = binarySearchInclusive<sortDir>(keyB, baseKey +      0, stride, stride) + lPos;

        __syncthreads();
        baseKey[posA] = keyA;
        baseVal[posA] = valA;
        baseKey[posB] = keyB;
        baseVal[posB] = valB;
    }

    __syncthreads();
    d_DstKey[                      0] = s_key[threadIdx.x +                       0];
    d_DstVal[                      0] = s_val[threadIdx.x +                       0];
    d_DstKey[(SHARED_SIZE_LIMIT / 2)] = s_key[threadIdx.x + (SHARED_SIZE_LIMIT / 2)];
    d_DstVal[(SHARED_SIZE_LIMIT / 2)] = s_val[threadIdx.x + (SHARED_SIZE_LIMIT / 2)];
}

__global__ void super_kernel_elastic_blockDim(
    uint *d_cm_flag,
    uint *d_DstKey,
    uint *d_DstVal,
    uint *d_SrcKey,
    uint *d_SrcVal,
    uint arrayLength)
{
     long long int l_timestamp = 0;
     unsigned int smidx, warpidx;
     /*if((blockIdx.x==0)||(blockIdx.x==15)||(blockIdx.x==30)||(blockIdx.x==45)||(blockIdx.x==60))
     {
       asm("mov.u32 %0, %%smid;" : "=r"(smidx));
       asm("mov.u32 %0, %%warpid;" : "=r"(warpidx));
       l_timestamp = clock64();
       //printf("%d %d %d %d %ld\n",smidx, warpidx, gridDim.x, blockDim.x, l_timestamp);
       printf("%d %d %d %d %d %d %ld\n",threadIdx.x, blockIdx.x, smidx, warpidx, gridDim.x, blockDim.x, l_timestamp);
     }*/
     if(threadIdx.x==0)
     {
        asm("mov.u32 %0, %%smid;" : "=r"(smidx));
        asm("mov.u32 %0, %%warpid;" : "=r"(warpidx));
        l_timestamp = clock64();
        /*if(blockIdx.x%60==0)
        {
          printf("start %d %d %d %ld\n", blockIdx.x, smidx, warpidx, l_timestamp);
        }*/
        uint l_h2d_flag=0;
        do
        {
           l_h2d_flag = *d_cm_flag;
           __threadfence_system();
        }while(l_h2d_flag==0);
      }
     __syncthreads();
     __shared__ bool s_grid_done;
     int iter=4369/*273*/, iter_completed=0;
    while(iter_completed<4369/*273*/){
     if(threadIdx.x==0)
     {
	s_grid_done=false;
        //if(blockIdx.x%60==0)
        //{
          //printf("start %d %d %d %ld %ld\n", blockIdx.x, smidx, warpidx, l_timestamp, clock64());
        //}
        //if(d_num_blocks_started == 0)
        /*if((blockIdx.x==0)&&(iter_completed==0))
        {
            printf("start %d %d %d %ld\n", blockIdx.x, smidx, warpidx, clock64());
        }*/
        atomicAdd(&d_num_blocks_started, 1);
     }
     __syncthreads();
     mergeSortSharedKernel_minion_elastic_blockDim<1>(d_DstKey, d_DstVal, d_SrcKey, d_SrcVal, arrayLength, iter_completed, false, 1);
     __syncthreads();
     //if(threadIdx.x==0)
     //{
        //l_timestamp = clock64();
        //printf("end %d %d %ld\n", r, blockIdx.x, l_timestamp);
     //}
     iter_completed++;
     if(threadIdx.x==0)
     {
         atomicAdd(&d_num_blocks_completed, 1);
         //if(d_num_blocks_completed==gridDim.x)
         //if(d_num_blocks_completed==d_grid_size)
         //if(iter_completed==273)
         if((blockIdx.x==0)&&(iter_completed==4369/*273*/))
         {
             s_grid_done=true;
             //printf("end %d %d %d %ld\n", blockIdx.x, smidx, warpidx, clock64());
         }
     }
     __syncthreads();
     /*if(s_grid_done==true){
       break;
     }*/
    }
     /*if((threadIdx.x==0)&&(blockIdx.x==0))
     {
         int l_num_blocks_completed;
         do
         {
           l_num_blocks_completed = atomicCAS(&d_num_blocks_completed, gridDim.x-1 ,0);
         }while(l_num_blocks_completed!=gridDim.x-1);
         atomicExch(&d_d2h_flag, 1);
         printf("%ld \n", clock64());
     }*/
}

////////////////////////////////////////////////////////////////////////////////
// Merge step 1: generate sample ranks
////////////////////////////////////////////////////////////////////////////////
template<uint sortDir> __global__ void generateSampleRanksKernel(
    uint *d_RanksA,
    uint *d_RanksB,
    uint *d_SrcKey,
    uint stride,
    uint N,
    uint threadCount
)
{
    uint pos = blockIdx.x * blockDim.x + threadIdx.x;

    if (pos >= threadCount)
    {
        return;
    }

    const uint           i = pos & ((stride / SAMPLE_STRIDE) - 1);
    const uint segmentBase = (pos - i) * (2 * SAMPLE_STRIDE);
    d_SrcKey += segmentBase;
    d_RanksA += segmentBase / SAMPLE_STRIDE;
    d_RanksB += segmentBase / SAMPLE_STRIDE;

    const uint segmentElementsA = stride;
    const uint segmentElementsB = umin(stride, N - segmentBase - stride);
    const uint  segmentSamplesA = getSampleCount(segmentElementsA);
    const uint  segmentSamplesB = getSampleCount(segmentElementsB);

    if (i < segmentSamplesA)
    {
        d_RanksA[i] = i * SAMPLE_STRIDE;
        d_RanksB[i] = binarySearchExclusive<sortDir>(
                          d_SrcKey[i * SAMPLE_STRIDE], d_SrcKey + stride,
                          segmentElementsB, nextPowerOfTwo(segmentElementsB)
                      );
    }

    if (i < segmentSamplesB)
    {
        d_RanksB[(stride / SAMPLE_STRIDE) + i] = i * SAMPLE_STRIDE;
        d_RanksA[(stride / SAMPLE_STRIDE) + i] = binarySearchInclusive<sortDir>(
                                                     d_SrcKey[stride + i * SAMPLE_STRIDE], d_SrcKey + 0,
                                                     segmentElementsA, nextPowerOfTwo(segmentElementsA)
                                                 );
    }
}

static void generateSampleRanks(
    uint *d_RanksA,
    uint *d_RanksB,
    uint *d_SrcKey,
    uint stride,
    uint N,
    uint sortDir
)
{
    uint lastSegmentElements = N % (2 * stride);
    uint         threadCount = (lastSegmentElements > stride) ? (N + 2 * stride - lastSegmentElements) / (2 * SAMPLE_STRIDE) : (N - lastSegmentElements) / (2 * SAMPLE_STRIDE);

    if (sortDir)
    {
	    cudaError_t k_err;
	    int num_blocks=0;
	    k_err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&num_blocks, generateSampleRanksKernel<1U>, 256, 0); 
	    if(k_err != cudaSuccess){
		    std::cerr << "cudaOccupancyMaxActiveBlocksPerMultiprocessor failed with error " << k_err << std::endl;}
	    std::cerr << "generateSampleRanksKernel occ " << num_blocks << std::endl;
        struct timeval start, end;
        gettimeofday(&start, NULL);
        generateSampleRanksKernel<1U><<<iDivUp(threadCount, 256), 256>>>(d_RanksA, d_RanksB, d_SrcKey, stride, N, threadCount);
        cudaDeviceSynchronize();
        gettimeofday(&end, NULL);
        std::cerr << "generateSampleRanksKernel " << sortDir << " " << iDivUp(threadCount, 256) << " " << (end.tv_sec - start.tv_sec)*1000000 + (end.tv_usec - start.tv_usec) << std::endl;
        getLastCudaError("generateSampleRanksKernel<1U><<<>>> failed\n");
    }
    else
    {
        struct timeval start, end;
        gettimeofday(&start, NULL);
        generateSampleRanksKernel<0U><<<iDivUp(threadCount, 256), 256>>>(d_RanksA, d_RanksB, d_SrcKey, stride, N, threadCount);
        cudaDeviceSynchronize();
        gettimeofday(&end, NULL);
        std::cerr << "generateSampleRanksKernel " << sortDir << " " << iDivUp(threadCount, 256) << " " << (end.tv_sec - start.tv_sec)*1000000 + (end.tv_usec - start.tv_usec) << std::endl;
        getLastCudaError("generateSampleRanksKernel<0U><<<>>> failed\n");
    }
}



////////////////////////////////////////////////////////////////////////////////
// Merge step 2: generate sample ranks and indices
////////////////////////////////////////////////////////////////////////////////
__global__ void mergeRanksAndIndicesKernel(
    uint *d_Limits,
    uint *d_Ranks,
    uint stride,
    uint N,
    uint threadCount
)
{
    uint pos = blockIdx.x * blockDim.x + threadIdx.x;

    if (pos >= threadCount)
    {
        return;
    }

    const uint           i = pos & ((stride / SAMPLE_STRIDE) - 1);
    const uint segmentBase = (pos - i) * (2 * SAMPLE_STRIDE);
    d_Ranks  += (pos - i) * 2;
    d_Limits += (pos - i) * 2;

    const uint segmentElementsA = stride;
    const uint segmentElementsB = umin(stride, N - segmentBase - stride);
    const uint  segmentSamplesA = getSampleCount(segmentElementsA);
    const uint  segmentSamplesB = getSampleCount(segmentElementsB);

    if (i < segmentSamplesA)
    {
        uint dstPos = binarySearchExclusive<1U>(d_Ranks[i], d_Ranks + segmentSamplesA, segmentSamplesB, nextPowerOfTwo(segmentSamplesB)) + i;
        d_Limits[dstPos] = d_Ranks[i];
    }

    if (i < segmentSamplesB)
    {
        uint dstPos = binarySearchInclusive<1U>(d_Ranks[segmentSamplesA + i], d_Ranks, segmentSamplesA, nextPowerOfTwo(segmentSamplesA)) + i;
        d_Limits[dstPos] = d_Ranks[segmentSamplesA + i];
    }
}

static void mergeRanksAndIndices(
    uint *d_LimitsA,
    uint *d_LimitsB,
    uint *d_RanksA,
    uint *d_RanksB,
    uint stride,
    uint N
)
{
    uint lastSegmentElements = N % (2 * stride);
    uint         threadCount = (lastSegmentElements > stride) ? (N + 2 * stride - lastSegmentElements) / (2 * SAMPLE_STRIDE) : (N - lastSegmentElements) / (2 * SAMPLE_STRIDE);

	    cudaError_t k_err;
	    int num_blocks=0;
	    k_err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&num_blocks, mergeRanksAndIndicesKernel, 256, 0); 
	    if(k_err != cudaSuccess){
		    std::cerr << "cudaOccupancyMaxActiveBlocksPerMultiprocessor failed with error " << k_err << std::endl;}
	    std::cerr << "mergeRanksAndIndicesKernel occ " << num_blocks << std::endl;
    struct timeval start, end;
    gettimeofday(&start, NULL);
    mergeRanksAndIndicesKernel<<<iDivUp(threadCount, 256), 256>>>(
        d_LimitsA,
        d_RanksA,
        stride,
        N,
        threadCount
    );
    cudaDeviceSynchronize();
    gettimeofday(&end, NULL);
    std::cerr << "mergeRanksAndIndicesKernelA " << " " << iDivUp(threadCount, 256) << " " << (end.tv_sec - start.tv_sec)*1000000 + (end.tv_usec - start.tv_usec) << std::endl;
    getLastCudaError("mergeRanksAndIndicesKernel(A)<<<>>> failed\n");

    gettimeofday(&start, NULL);
    mergeRanksAndIndicesKernel<<<iDivUp(threadCount, 256), 256>>>(
        d_LimitsB,
        d_RanksB,
        stride,
        N,
        threadCount
    );
    cudaDeviceSynchronize();
    gettimeofday(&end, NULL);
    std::cerr << "mergeRanksAndIndicesKernelB " << " " << iDivUp(threadCount, 256) << " " << (end.tv_sec - start.tv_sec)*1000000 + (end.tv_usec - start.tv_usec) << std::endl;
    getLastCudaError("mergeRanksAndIndicesKernel(B)<<<>>> failed\n");
}



////////////////////////////////////////////////////////////////////////////////
// Merge step 3: merge elementary intervals
////////////////////////////////////////////////////////////////////////////////
template<uint sortDir> inline __device__ void merge(
    uint *dstKey,
    uint *dstVal,
    uint *srcAKey,
    uint *srcAVal,
    uint *srcBKey,
    uint *srcBVal,
    uint lenA,
    uint nPowTwoLenA,
    uint lenB,
    uint nPowTwoLenB
)
{
    uint keyA, valA, keyB, valB, dstPosA, dstPosB;

    if (threadIdx.x < lenA)
    {
        keyA = srcAKey[threadIdx.x];
        valA = srcAVal[threadIdx.x];
        dstPosA = binarySearchExclusive<sortDir>(keyA, srcBKey, lenB, nPowTwoLenB) + threadIdx.x;
    }

    if (threadIdx.x < lenB)
    {
        keyB = srcBKey[threadIdx.x];
        valB = srcBVal[threadIdx.x];
        dstPosB = binarySearchInclusive<sortDir>(keyB, srcAKey, lenA, nPowTwoLenA) + threadIdx.x;
    }

    __syncthreads();

    if (threadIdx.x < lenA)
    {
        dstKey[dstPosA] = keyA;
        dstVal[dstPosA] = valA;
    }

    if (threadIdx.x < lenB)
    {
        dstKey[dstPosB] = keyB;
        dstVal[dstPosB] = valB;
    }
}

template<uint sortDir> __global__ void mergeElementaryIntervalsKernel(
    uint *d_DstKey,
    uint *d_DstVal,
    uint *d_SrcKey,
    uint *d_SrcVal,
    uint *d_LimitsA,
    uint *d_LimitsB,
    uint stride,
    uint N
)
{
    __shared__ uint s_key[2 * SAMPLE_STRIDE];
    __shared__ uint s_val[2 * SAMPLE_STRIDE];

    const uint   intervalI = blockIdx.x & ((2 * stride) / SAMPLE_STRIDE - 1);
    const uint segmentBase = (blockIdx.x - intervalI) * SAMPLE_STRIDE;
    d_SrcKey += segmentBase;
    d_SrcVal += segmentBase;
    d_DstKey += segmentBase;
    d_DstVal += segmentBase;

    //Set up threadblock-wide parameters
    __shared__ uint startSrcA, startSrcB, lenSrcA, lenSrcB, startDstA, startDstB;

    if (threadIdx.x == 0)
    {
        uint segmentElementsA = stride;
        uint segmentElementsB = umin(stride, N - segmentBase - stride);
        uint  segmentSamplesA = getSampleCount(segmentElementsA);
        uint  segmentSamplesB = getSampleCount(segmentElementsB);
        uint   segmentSamples = segmentSamplesA + segmentSamplesB;

        startSrcA    = d_LimitsA[blockIdx.x];
        startSrcB    = d_LimitsB[blockIdx.x];
        uint endSrcA = (intervalI + 1 < segmentSamples) ? d_LimitsA[blockIdx.x + 1] : segmentElementsA;
        uint endSrcB = (intervalI + 1 < segmentSamples) ? d_LimitsB[blockIdx.x + 1] : segmentElementsB;
        lenSrcA      = endSrcA - startSrcA;
        lenSrcB      = endSrcB - startSrcB;
        startDstA    = startSrcA + startSrcB;
        startDstB    = startDstA + lenSrcA;
    }

    //Load main input data
    __syncthreads();

    if (threadIdx.x < lenSrcA)
    {
        s_key[threadIdx.x +             0] = d_SrcKey[0 + startSrcA + threadIdx.x];
        s_val[threadIdx.x +             0] = d_SrcVal[0 + startSrcA + threadIdx.x];
    }

    if (threadIdx.x < lenSrcB)
    {
        s_key[threadIdx.x + SAMPLE_STRIDE] = d_SrcKey[stride + startSrcB + threadIdx.x];
        s_val[threadIdx.x + SAMPLE_STRIDE] = d_SrcVal[stride + startSrcB + threadIdx.x];
    }

    //Merge data in shared memory
    __syncthreads();
    merge<sortDir>(
        s_key,
        s_val,
        s_key + 0,
        s_val + 0,
        s_key + SAMPLE_STRIDE,
        s_val + SAMPLE_STRIDE,
        lenSrcA, SAMPLE_STRIDE,
        lenSrcB, SAMPLE_STRIDE
    );

    //Store merged data
    __syncthreads();

    if (threadIdx.x < lenSrcA)
    {
        d_DstKey[startDstA + threadIdx.x] = s_key[threadIdx.x];
        d_DstVal[startDstA + threadIdx.x] = s_val[threadIdx.x];
    }

    if (threadIdx.x < lenSrcB)
    {
        d_DstKey[startDstB + threadIdx.x] = s_key[lenSrcA + threadIdx.x];
        d_DstVal[startDstB + threadIdx.x] = s_val[lenSrcA + threadIdx.x];
    }
}

static void mergeElementaryIntervals(
    uint *d_DstKey,
    uint *d_DstVal,
    uint *d_SrcKey,
    uint *d_SrcVal,
    uint *d_LimitsA,
    uint *d_LimitsB,
    uint stride,
    uint N,
    uint sortDir
)
{
    uint lastSegmentElements = N % (2 * stride);
    uint          mergePairs = (lastSegmentElements > stride) ? getSampleCount(N) : (N - lastSegmentElements) / SAMPLE_STRIDE;

    if (sortDir)
    {
	    cudaError_t k_err;
	    int num_blocks=0;
	    k_err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&num_blocks, mergeElementaryIntervalsKernel<1U>, SAMPLE_STRIDE, 0); 
	    if(k_err != cudaSuccess){
		    std::cerr << "cudaOccupancyMaxActiveBlocksPerMultiprocessor failed with error " << k_err << std::endl;}
	    std::cerr << "mergeElementaryIntervalsKernel occ " << num_blocks << std::endl;
        struct timeval start, end;
        gettimeofday(&start, NULL);
        mergeElementaryIntervalsKernel<1U><<<mergePairs, SAMPLE_STRIDE>>>(
            d_DstKey,
            d_DstVal,
            d_SrcKey,
            d_SrcVal,
            d_LimitsA,
            d_LimitsB,
            stride,
            N
        );
	cudaDeviceSynchronize();
	gettimeofday(&end, NULL);
	std::cerr << "mergeElementaryIntervalsKernel " << sortDir << " " << mergePairs << " " << (end.tv_sec - start.tv_sec)*1000000 + (end.tv_usec - start.tv_usec) << std::endl;
        getLastCudaError("mergeElementaryIntervalsKernel<1> failed\n");
    }
    else
    {
        struct timeval start, end;
        gettimeofday(&start, NULL);
        mergeElementaryIntervalsKernel<0U><<<mergePairs, SAMPLE_STRIDE>>>(
            d_DstKey,
            d_DstVal,
            d_SrcKey,
            d_SrcVal,
            d_LimitsA,
            d_LimitsB,
            stride,
            N
        );
	cudaDeviceSynchronize();
	gettimeofday(&end, NULL);
	std::cerr << "mergeElementaryIntervalsKernel " << sortDir << " " << mergePairs << " " << (end.tv_sec - start.tv_sec)*1000000 + (end.tv_usec - start.tv_usec) << std::endl;
        getLastCudaError("mergeElementaryIntervalsKernel<0> failed\n");
    }
}



extern "C" void bitonicSortShared(
    uint *d_DstKey,
    uint *d_DstVal,
    uint *d_SrcKey,
    uint *d_SrcVal,
    uint batchSize,
    uint arrayLength,
    uint sortDir
);

extern "C" void bitonicMergeElementaryIntervals(
    uint *d_DstKey,
    uint *d_DstVal,
    uint *d_SrcKey,
    uint *d_SrcVal,
    uint *d_LimitsA,
    uint *d_LimitsB,
    uint stride,
    uint N,
    uint sortDir
);



static uint *d_RanksA, *d_RanksB, *d_LimitsA, *d_LimitsB;
static const uint MAX_SAMPLE_COUNT = 64 * 32 * 1024;//32768;

extern "C" void initMergeSort(void)
{
    checkCudaErrors(cudaMalloc((void **)&d_RanksA,  MAX_SAMPLE_COUNT * sizeof(uint)));
    checkCudaErrors(cudaMalloc((void **)&d_RanksB,  MAX_SAMPLE_COUNT * sizeof(uint)));
    checkCudaErrors(cudaMalloc((void **)&d_LimitsA, MAX_SAMPLE_COUNT * sizeof(uint)));
    checkCudaErrors(cudaMalloc((void **)&d_LimitsB, MAX_SAMPLE_COUNT * sizeof(uint)));

}

extern "C" void closeMergeSort(void)
{
    checkCudaErrors(cudaFree(d_RanksA));
    checkCudaErrors(cudaFree(d_RanksB));
    checkCudaErrors(cudaFree(d_LimitsB));
    checkCudaErrors(cudaFree(d_LimitsA));

}

extern "C" void mergeSort(
    uint *d_DstKey,
    uint *d_DstVal,
    uint *d_BufKey,
    uint *d_BufVal,
    uint *d_SrcKey,
    uint *d_SrcVal,
    uint N,
    uint sortDir
)
{
    uint stageCount = 0;

    for (uint stride = SHARED_SIZE_LIMIT; stride < N; stride <<= 1, stageCount++);

    uint *ikey, *ival, *okey, *oval;

    if (stageCount & 1)
    {
        ikey = d_BufKey;
        ival = d_BufVal;
        okey = d_DstKey;
        oval = d_DstVal;
    }
    else
    {
        ikey = d_DstKey;
        ival = d_DstVal;
        okey = d_BufKey;
        oval = d_BufVal;
    }

    assert(N <= (SAMPLE_STRIDE * MAX_SAMPLE_COUNT));
    assert(N % SHARED_SIZE_LIMIT == 0);
    struct timeval start, end;
    int in_zero_keys=0, in_non_zero_keys=0, in_zero_vals=0, in_non_zero_vals=0, out_zero_keys=0, out_non_zero_keys=0, out_zero_vals=0, out_non_zero_vals=0;
    uint *h_key1, *h_val1, *h_key2, *h_val2, *h_key3, *h_val3;
    h_key1 = (uint *)calloc(N, sizeof(uint));
    h_val1 = (uint *)calloc(N, sizeof(uint));
    h_key2 = (uint *)calloc(N, sizeof(uint));
    h_val2 = (uint *)calloc(N, sizeof(uint));
    h_key3 = (uint *)calloc(N, sizeof(uint));
    h_val3 = (uint *)calloc(N, sizeof(uint));
    checkCudaErrors(cudaMemcpy(h_key3, d_SrcKey, N*sizeof(uint), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(h_val3, d_SrcVal, N*sizeof(uint), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(h_key1, ikey, N*sizeof(uint), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(h_val1, ival, N*sizeof(uint), cudaMemcpyDeviceToHost));
    gettimeofday(&start, NULL);
    for(int i=0; i<N; i++){
      if(h_key3[i]==0){
         in_zero_keys++;
      }else{
         in_non_zero_keys++;
      }
      if(h_val3[i]==0){
         in_zero_vals++;
      }else{
         in_non_zero_vals++;
      }
      if(h_key1[i]==0){
         out_zero_keys++;
      }else{
         out_non_zero_keys++;
      }
      if(h_val1[i]==0){
         out_zero_vals++;
      }else{
         out_non_zero_vals++;
      }
    }
    gettimeofday(&end, NULL);
    std::cerr << "Output vs Input inspection " << (end.tv_sec - start.tv_sec)*1000000 + (end.tv_usec - start.tv_usec) << std::endl;
    std::cerr << in_zero_keys << " " << in_non_zero_keys << " " << in_zero_vals << " " << in_non_zero_vals << std::endl;
    std::cerr << out_zero_keys << " " << out_non_zero_keys << " " << out_zero_vals << " " << out_non_zero_vals << std::endl;

    mergeSortShared(ikey, ival, d_SrcKey, d_SrcVal, N / SHARED_SIZE_LIMIT, SHARED_SIZE_LIMIT, sortDir);
    gettimeofday(&start, NULL);
    checkCudaErrors(cudaMemcpy(h_key1, ikey, N*sizeof(uint), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(h_val1, ival, N*sizeof(uint), cudaMemcpyDeviceToHost));
    gettimeofday(&end, NULL);
    std::cerr << "D2D " << (end.tv_sec - start.tv_sec)*1000000 + (end.tv_usec - start.tv_usec) << std::endl;

    out_zero_keys=out_non_zero_keys=out_zero_vals=out_non_zero_vals=0;
    for(int i=0; i<N; i++){
      if(h_key1[i]==0){
         out_zero_keys++;
      }else{
         out_non_zero_keys++;
      }
      if(h_val1[i]==0){
         out_zero_vals++;
      }else{
         out_non_zero_vals++;
      }
    }
    std::cerr << out_zero_keys << " " << out_non_zero_keys << " " << out_zero_vals << " " << out_non_zero_vals << std::endl;

    checkCudaErrors(cudaMemset(ikey, 0, N*sizeof(uint)));
    checkCudaErrors(cudaMemset(ival, 0, N*sizeof(uint)));

    uint *d_cm_flag=NULL;
    cudaMalloc(&d_cm_flag, sizeof(uint));

    cudaError_t k_err;
    cudaStream_t k_strm;
    k_err = cudaStreamCreateWithFlags(&k_strm, cudaStreamNonBlocking);
    if(k_err != cudaSuccess){
      std::cerr << "cudaStreamCreateWithFlags failed with error " << k_err << std::endl;}
    cudaEvent_t k_ev;
    cudaEventCreateWithFlags(&k_ev, cudaEventBlockingSync);
    if(k_err != cudaSuccess){
      std::cerr << "cudaEventCreateWithFlags failed with error " << k_err << std::endl;}
    
    uint l_h2d_flag=1;

    /*gettimeofday(&start, NULL);
    cudaMemcpy(d_cm_flag, &l_h2d_flag, sizeof(uint), cudaMemcpyHostToDevice);
    gettimeofday(&end, NULL);
    std::cerr << "H2D Flag write pre " << (end.tv_sec - start.tv_sec)*1000000 + (end.tv_usec - start.tv_usec) << std::endl;*/
    int num_blocks=0;
    k_err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&num_blocks, super_kernel, 512, 0); 
    if(k_err != cudaSuccess){
      std::cerr << "cudaOccupancyMaxActiveBlocksPerMultiprocessor failed with error " << k_err << std::endl;}
    std::cerr << "super_kernel occ " << num_blocks << std::endl;
    gettimeofday(&start, NULL);
    //super_kernel<<</*192*15*/512*32, 512/*64*/>>>(d_cm_flag, ikey, ival, d_SrcKey, d_SrcVal, SHARED_SIZE_LIMIT);
    //super_kernel<<<16384, 512, 0, k_strm>>>(d_cm_flag, ikey, ival, d_SrcKey, d_SrcVal, SHARED_SIZE_LIMIT);
    super_kernel<<<60, 512, 0, k_strm>>>(d_cm_flag, ikey, ival, d_SrcKey, d_SrcVal, SHARED_SIZE_LIMIT);
    k_err = cudaEventRecord(k_ev, k_strm);
    if(k_err != cudaSuccess){
      std::cerr << "cudaEventRecord failed with error " << k_err << std::endl;}
    //cudaDeviceSynchronize();
    gettimeofday(&end, NULL);
    std::cerr << "super_kernel+event submitted " << (end.tv_sec - start.tv_sec)*1000000 + (end.tv_usec - start.tv_usec) << std::endl;

    gettimeofday(&start, NULL);
    k_err = cudaMemcpy(d_cm_flag, &l_h2d_flag, sizeof(uint), cudaMemcpyHostToDevice);
    if(k_err != cudaSuccess){
       std::cerr << "cudaMemcpy failed with error " << k_err << std::endl;}
    gettimeofday(&end, NULL);
    std::cerr << "H2D Flag write post(super_kernel) " << (end.tv_sec - start.tv_sec)*1000000 + (end.tv_usec - start.tv_usec) << std::endl;

    gettimeofday(&start, NULL);
    k_err = cudaEventSynchronize(k_ev);
    if(k_err != cudaSuccess){
      std::cerr << "cudaEventSynchronize failed with error " << k_err << std::endl;}
    gettimeofday(&end, NULL);
    std::cerr << "super_kernel completed " << (end.tv_sec - start.tv_sec)*1000000 + (end.tv_usec - start.tv_usec) << std::endl;
    
    gettimeofday(&start, NULL);
    checkCudaErrors(cudaMemcpy(h_key2, ikey, N*sizeof(uint), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(h_val2, ival, N*sizeof(uint), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemset(ikey, 0, N*sizeof(uint)));
    checkCudaErrors(cudaMemset(ival, 0, N*sizeof(uint)));
    gettimeofday(&end, NULL);
    std::cerr << "D2D+Memset " << (end.tv_sec - start.tv_sec)*1000000 + (end.tv_usec - start.tv_usec) << std::endl;

    gettimeofday(&start, NULL);
    for(int i=0; i<N; i++){
        if(h_key1[i]!=h_key2[i]){
          std::cerr << "Mismatch(key) at " << i << " " << h_key1[i] << " " << h_key2[i] << std::endl;
        }
        if(h_val1[i]!=h_val2[i]){
          std::cerr << "Mismatch(val) at " << i << " " << h_val1[i] << " " << h_val2[i] << std::endl;
        }
    }
    gettimeofday(&end, NULL);
    std::cerr << "Validation " << (end.tv_sec - start.tv_sec)*1000000 + (end.tv_usec - start.tv_usec) << std::endl;

    mergeSortShared(ikey, ival, d_SrcKey, d_SrcVal, N / SHARED_SIZE_LIMIT, SHARED_SIZE_LIMIT, sortDir);

    gettimeofday(&start, NULL);
    super_kernel_elastic_blockDim<<<60, 512, 0, k_strm>>>(d_cm_flag, ikey, ival, d_SrcKey, d_SrcVal, SHARED_SIZE_LIMIT);
    k_err = cudaEventRecord(k_ev, k_strm);
    if(k_err != cudaSuccess){
      std::cerr << "cudaEventRecord failed with error " << k_err << std::endl;}
    gettimeofday(&end, NULL);
    std::cerr << "super_kernel_elastic_blockDim+event submitted " << (end.tv_sec - start.tv_sec)*1000000 + (end.tv_usec - start.tv_usec) << std::endl;

    gettimeofday(&start, NULL);
    k_err = cudaMemcpy(d_cm_flag, &l_h2d_flag, sizeof(uint), cudaMemcpyHostToDevice);
    if(k_err != cudaSuccess){
       std::cerr << "cudaMemcpy failed with error " << k_err << std::endl;}
    gettimeofday(&end, NULL);
    std::cerr << "H2D Flag write post(super_kernel_elastic_blockDim) " << (end.tv_sec - start.tv_sec)*1000000 + (end.tv_usec - start.tv_usec) << std::endl;

    gettimeofday(&start, NULL);
    k_err = cudaEventSynchronize(k_ev);
    if(k_err != cudaSuccess){
      std::cerr << "cudaEventSynchronize failed with error " << k_err << std::endl;}
    gettimeofday(&end, NULL);
    std::cerr << "super_kernel_elastic_blockDim completed " << (end.tv_sec - start.tv_sec)*1000000 + (end.tv_usec - start.tv_usec) << std::endl;
    
#if 1
    for (uint stride = SHARED_SIZE_LIMIT; stride < N; stride <<= 1)
    {
        uint lastSegmentElements = N % (2 * stride);

        //Find sample ranks and prepare for limiters merge
        generateSampleRanks(d_RanksA, d_RanksB, ikey, stride, N, sortDir);

        //Merge ranks and indices
        mergeRanksAndIndices(d_LimitsA, d_LimitsB, d_RanksA, d_RanksB, stride, N);

        //Merge elementary intervals
        mergeElementaryIntervals(okey, oval, ikey, ival, d_LimitsA, d_LimitsB, stride, N, sortDir);

        if (lastSegmentElements && (lastSegmentElements <= stride))
        {
            //Last merge segment consists of a single array which just needs to be passed through
		gettimeofday(&start, NULL);
            checkCudaErrors(cudaMemcpy(okey + (N - lastSegmentElements), ikey + (N - lastSegmentElements), lastSegmentElements * sizeof(uint), cudaMemcpyDeviceToDevice));
	    gettimeofday(&end, NULL);
	    std::cerr << "D2D okey " << lastSegmentElements * sizeof(uint) << " " << (end.tv_sec - start.tv_sec)*1000000 + (end.tv_usec - start.tv_usec) << std::endl;
	    gettimeofday(&start, NULL);
            checkCudaErrors(cudaMemcpy(oval + (N - lastSegmentElements), ival + (N - lastSegmentElements), lastSegmentElements * sizeof(uint), cudaMemcpyDeviceToDevice));
	    gettimeofday(&end, NULL);
	    std::cerr << "D2D oval " << lastSegmentElements * sizeof(uint) << " " << (end.tv_sec - start.tv_sec)*1000000 + (end.tv_usec - start.tv_usec) << std::endl;
        }

        uint *t;
        t = ikey;
        ikey = okey;
        okey = t;
        t = ival;
        ival = oval;
        oval = t;
    }
#endif
}

