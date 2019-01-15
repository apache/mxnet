/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */
/*!
 * Copyright (c) 2018 by Contributors
 * \file binary_inference_convolution-inl.h
 * \brief
 * \ref: https://arxiv.org/abs/1705.09864
 * \author MatthieuCourbariaux (Github), HPI-DeepLearning
*/

#include "./xnor_kernels.h"

namespace xnor_cuda {

// 32 single float array ->  32 bits BINARY_WORD
__device__ BINARY_WORD concatenate(float* array)
{
    BINARY_WORD rvalue=0;
    BINARY_WORD sign;

    for (int i = 0; i < BITS_PER_BINARY_WORD; i++)
    {
        sign = (array[i]>=0);
        rvalue |= (sign << i);
    }
    
    return rvalue;
}

//concatinate in standard directions: (ROW_top->ROW_down {COL_left->COL_right} )
__global__ void concatenate_rows_kernel(float *a, BINARY_WORD *b, int size)
{ 
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i<size) b[i] = concatenate(&a[i*BITS_PER_BINARY_WORD]);
}

//concatinate column, processing directions: (COL_left->COL_right {ROW_top->ROW_down} ) 
/*	this method should give the same result as the following implementation:
 *   for(int y=0; y<(n/BITS_PER_BINARY_WORD); y++){
      for(int x=0; x < k; ++x){          
        BINARY_WORD rvalue=0;
        BINARY_WORD sign;    
        for(int b=0; b<BITS_PER_BINARY_WORD; ++b){
          sign = (col[(y*BITS_PER_BINARY_WORD+b)*k + x]>=0);          
          rvalue |= (sign << b);
        }
        b_col[y*k + x] = rvalue;
      }
    }    
 */
__global__ void concatenate_cols_kernel(float *a, BINARY_WORD *b, int n, int k)
{   
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    
    if(j<k){        
    	float * array = new float[BITS_PER_BINARY_WORD];
        for(int i=0; i<n; i+=BITS_PER_BINARY_WORD){       	            
            for(int bit=0; bit<BITS_PER_BINARY_WORD;bit++) 
            	array[bit] = a[j + k*(i+bit)];            
            b[j+k*i/BITS_PER_BINARY_WORD]=concatenate(array);             
        }   
        delete[] array;      
    }
}

// CUDA tutorial: http://www.nvidia.com/docs/IO/116711/sc11-cuda-c-basics.pdf
// http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#shared-memory
// A is shape (m,n), B is shape (n,k) and C is shape (m,k)
__global__ void xnor_gemm(BINARY_WORD* A, BINARY_WORD* B, float* C, int m, int n, int k, int block_size) {
    
    // Block row and column
    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;
    
    // Thread row and column within Csub
    int row = threadIdx.y;
    int col = threadIdx.x;

    // Each thread block computes one sub-matrix Csub of C
    float* Csub = &C[block_size * k * blockRow + block_size * blockCol];

    // Shared memory used to store Asub and Bsub respectively
    extern __shared__ BINARY_WORD sm[];
    BINARY_WORD* As = sm;
    BINARY_WORD* Bs = (BINARY_WORD*)&As[block_size*block_size];
    
    // Each thread computes one element of Csub
    // by accumulating results into Cvalue
    float Cvalue = 0;
    
    // Loop over all the sub-matrices of A and B that are
    // required to compute Csub
    // Multiply each pair of sub-matrices together
    // and accumulate the results
    for (int i = 0; i < (n / block_size); ++i) {
    
        // Get sub-matrix Asub of A
        BINARY_WORD* Asub = &A[block_size * blockRow * n + block_size * i];
        
        // Get sub-matrix Bsub of B
        BINARY_WORD* Bsub = &B[block_size * k * i + block_size * blockCol];
        
        // Load Asub and Bsub from device memory to shared memory
        // Each thread loads one element of each sub-matrix
        As[row*block_size + col] = Asub[row*n+col];
        Bs[row*block_size + col] = Bsub[row*k+col];
    
        // Synchronize to make sure the sub-matrices are loaded
        // before starting the computation
        __syncthreads();
        
        // Multiply Asub and Bsub together
        // apply xnor and popcount: 
        //CUDA has population count intrinsics for both 32-bit and 64-bit types. (__popc() and __popcll())
        for (int j = 0; j < block_size; ++j){
            #if BINARY_WORD_64 == 1
                Cvalue += (float)__popcll(~(As[row*block_size + j]^Bs[j*block_size + col]));
            #else //BINARY_WORD_32 == 1
                Cvalue += (float)__popc(~(As[row*block_size + j]^Bs[j*block_size + col]));
            #endif            
        }
        
        // Synchronize to make sure that the preceding
        // computation is done before loading two new
        // sub-matrices of A and B in the next iteration
        __syncthreads();
    }
    
    // Write Csub to device memory
    // Each thread writes one element    
    if(col + blockCol* block_size< k && row + blockRow* block_size< m) Csub[row*k+col] = Cvalue;
}
}// namespace xnor_cuda