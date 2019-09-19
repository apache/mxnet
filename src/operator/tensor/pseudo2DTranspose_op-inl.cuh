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
 * Copyright (c) 2019 by Contributors
 * \file pseudo2DTranspose_op-inl.cuh
 * \brief pseudo 2D transpose
 * \author Dawid Tracz
 */

#ifndef MXNET_OPERATOR_TENSOR_PSEUDO2DTRANSPOSE_OP_INL_CUH_
#define MXNET_OPERATOR_TENSOR_PSEUDO2DTRANSPOSE_OP_INL_CUH_

#include <mxnet/tuple.h>
#include <mxnet/tensor_blob.h>
#include <mshadow/base.h>
#include <algorithm>
#include <utility>
#include "../../common/cuda_utils.h"


namespace mxnet {
namespace op {
namespace cuda {


template <typename DType, typename CType>
__global__ void transpose_pseudo2D(DType* out, DType* inp, const uint32_t m, const uint32_t n) {
  const uint32_t TSR = sizeof(CType)/sizeof(DType);  // TypeSizeRatio
  const uint32_t chunked_n = n/TSR;
  const uint32_t chunked_m = m/TSR;

  union transp_t {
   CType valChunk;
   DType values[TSR];
  };

  __shared__ DType d_shm[1024*TSR*TSR];
  CType* c_shm = reinterpret_cast<CType*>(d_shm);

  CType* cInp = reinterpret_cast<CType*>(inp);
  CType* cOut = reinterpret_cast<CType*>(out);

  uint32_t offset = blockIdx.z*m*chunked_n
                  + blockIdx.y*blockDim.y*TSR*chunked_n
                  + blockIdx.x*blockDim.x;

  if ((blockIdx.x*blockDim.x + threadIdx.x)*TSR < n
   && (blockIdx.y*blockDim.y + threadIdx.y)*TSR < m) {
    // read from global memory to shared
    #pragma unroll
    for (uint32_t i=0; i<TSR; i++) {
      uint32_t shmIdx = (TSR*threadIdx.y + i)*blockDim.x + threadIdx.x;
      c_shm[shmIdx] = cInp[offset + (TSR*threadIdx.y + i)*chunked_n + threadIdx.x];
    }
    __syncthreads();

    // read from shared to registers
    transp_t tmp[TSR];
    #pragma unroll
    for (uint32_t i=0; i<TSR; i++) {
      #pragma unroll
      for (int j=0; j<TSR; j++) {
        uint32_t shmIdx = (TSR*threadIdx.y + j)*blockDim.x*TSR + TSR*threadIdx.x + i;
        tmp[i].values[j] = d_shm[shmIdx];
      }
    }
    __syncthreads();

    // write back to global output
    offset = blockIdx.z*m*chunked_n + blockIdx.x*blockDim.x*TSR*chunked_m + blockIdx.y*blockDim.y;
    #pragma unroll
    for (uint32_t i=0; i<TSR; i++) {
        cOut[offset + (TSR*threadIdx.x + i)*chunked_m + threadIdx.y] = tmp[i].valChunk;
    }
  }
}

}  // namespace cuda


/*!
 * \brief Calls proper version of kernel `transpose_pseudo2D`
 *        basing on chosen type sizes.
 * \param dTypeSize Size of data type.
 * \param cTypeSize Size of type that should be use to copy.
 * \param grid Grid dimensions for the kernel.
 * \param block Block dimensions for the kernel.
 * \param stream Strem to run kernel.
 * \param out Pointer to output memory.
 * \param inp Pointer to input memory.
 * \param m First of tensor dimensions.
 * \param n Second of tensor dimensions.
 */
inline void call_transpose_pseudo2D(uint32_t dTypeSize, uint32_t cTypeSize,
                                   dim3 grid, dim3 block, cudaStream_t stream,
                                   void* out, void* inp, const uint32_t m, const uint32_t n) {
  switch (dTypeSize) {
   case (1): {
    uint8_t* d_outPtr = reinterpret_cast<uint8_t*>(out);
    uint8_t* d_inpPtr = reinterpret_cast<uint8_t*>(inp);
    switch (cTypeSize) {
     case (1):
      cuda::transpose_pseudo2D<uint8_t, uint8_t><<<grid, block, 0, stream>>>(d_outPtr, d_inpPtr, m, n);
      break;
     case (2):
      cuda::transpose_pseudo2D<uint8_t, uint16_t><<<grid, block, 0, stream>>>(d_outPtr, d_inpPtr, m, n);
      break;
     case (4):
      cuda::transpose_pseudo2D<uint8_t, uint32_t><<<grid, block, 0, stream>>>(d_outPtr, d_inpPtr, m, n);
      break;
     case (8):
      // case guarded against in function getBestCopyTypeSize
      LOG(FATAL) << "cuda::transpose_pseudo2D<uint8_t, uint64_t> would take too much shared memory";
     default:
      LOG(FATAL) << "Unsupported type combination";
    }
    break;
   }
   case (2): {
    uint16_t* d_outPtr = reinterpret_cast<uint16_t*>(out);
    uint16_t* d_inpPtr = reinterpret_cast<uint16_t*>(inp);
    switch (cTypeSize) {
     case (2):
      cuda::transpose_pseudo2D<uint16_t, uint16_t><<<grid, block, 0, stream>>>(d_outPtr, d_inpPtr, m, n);
      break;
     case (4):
      cuda::transpose_pseudo2D<uint16_t, uint32_t><<<grid, block, 0, stream>>>(d_outPtr, d_inpPtr, m, n);
      break;
     case (8):
      cuda::transpose_pseudo2D<uint16_t, uint64_t><<<grid, block, 0, stream>>>(d_outPtr, d_inpPtr, m, n);
      break;
     default:
      LOG(FATAL) << "Unsupported type combination";
    }
    break;
   }
   case (4): {
    uint32_t* d_outPtr = reinterpret_cast<uint32_t*>(out);
    uint32_t* d_inpPtr = reinterpret_cast<uint32_t*>(inp);
    switch (cTypeSize) {
     case (4):
      cuda::transpose_pseudo2D<uint32_t, uint32_t><<<grid, block, 0, stream>>>(d_outPtr, d_inpPtr, m, n);
      break;
     case (8):
      cuda::transpose_pseudo2D<uint32_t, uint64_t><<<grid, block, 0, stream>>>(d_outPtr, d_inpPtr, m, n);
      break;
     default:
      LOG(FATAL) << "Unsupported type combination";
    }
    break;
   }
   case (8): {
    uint64_t* d_outPtr = reinterpret_cast<uint64_t*>(out);
    uint64_t* d_inpPtr = reinterpret_cast<uint64_t*>(inp);
    switch (cTypeSize) {
     case (8):
      cuda::transpose_pseudo2D<uint64_t, uint64_t><<<grid, block, 0, stream>>>(d_outPtr, d_inpPtr, m, n);
      break;
     default:
      LOG(FATAL) << "Unsupported type combination";
    }
    break;
   }
   default:
    LOG(FATAL) << "Unsupported type combination";
  }
  auto cuErr = cudaPeekAtLastError();
  if (cuErr != cudaSuccess) {
      std::cerr << "Transpose kernel failure: " << cudaGetErrorString(cuErr) << std::endl;
      exit(EXIT_FAILURE);
  }
}


/*!
 * \brief Checks if function `transpose_pseudo2D` can be used
 *        to perform transpose operation with given params.
 * \param params Parameters (axes) of the transpose.
 */
inline bool isPseudo2DTranspose(const TShape& params) {
  size_t n_swpDims = 1;
  int i=0;
  while (i < params.ndim() && i == params[i])
    i++;  // leading dimensions
  while (i+1 < params.ndim()) {
    if(params[i]+1 != params[i+1])
      n_swpDims++;
    i++;
  }
  return n_swpDims == 2;
}


struct pseudo2DSizes {
  uint32_t leadDimS;
  uint32_t M;
  uint32_t N;
};

/*!
 * \brief Calculates total size of last two dimension batches
 *        (according to description of transpose_pseudo2D function).
 * \param shape Shape of tensor to transpose.
 * \param params Parameters (axes) of the transpose.
 */
inline pseudo2DSizes getPackedTransposeDimensions(const TShape& shape,
                                                  const TShape& params) {
  auto ndim = params.ndim();
  pseudo2DSizes sizes;
  sizes.leadDimS = 1;
  int i=0;
  while (i < ndim && i == params[i]) {
    sizes.leadDimS *= shape[i];
    i++;
  }
  sizes.N = shape[params[i++]];
  while (i < ndim && params[i]-1 == params[i-1]) {
    sizes.N *= shape[params[i]];
    i++;
  }
  sizes.M = shape[params[i++]];
  while (i < ndim && params[i]-1 == params[i-1]) {
    sizes.M *= shape[params[i]];
    i++;
  }
  CHECK_EQ(i, ndim) << "Too many dimensions to transpose";
  return sizes;
}


inline int32_t getBestCopyTypeSize(size_t dTypeSize, uint32_t sizeM, uint32_t sizeN) {
  uint32_t cTypeSize = std::max((size_t)8, dTypeSize);
  while (cTypeSize > dTypeSize) {
    auto tsr = cTypeSize/dTypeSize;
    if (sizeM % tsr != 0 || sizeN % tsr != 0)
      cTypeSize /= 2;
    else
      break;
  }
  // if the cTypeSize is 8x dTypeSize then kernel would require 64kB shared memory
  if(cTypeSize == 8 && dTypeSize == 1)
    cTypeSize = 4;
  return cTypeSize;
}


/*!
 * \brief Transpose given tensor according to params.
 *        Supports only transposes that satisfy:
 *        Exists n and m such that:
 *        params = (0, ..., n-1, n+m, ..., params.size, n, ..., n+m-1)
 *        Example: (0, 2, 3, 1) or (0, 3, 1, 2), but not (0, 2, 1, 3).
 * \param outBlob Tensor blob to store result.
 * \param inpBlob Tensor blob with input data.
 * \param params Parameters (axes) of the transpose.
 * \param s Poinster to GPU stream.
 */
template <typename DType, typename gpu>
void transpose_pseudo2D(const TBlob& outBlob, const TBlob& inpBlob,
                        const TShape& params, mshadow::Stream<gpu>* s) {
  const TShape& shape = inpBlob.shape_;
  CHECK_EQ(shape.ndim(), params.ndim());
  auto ndim = params.ndim();

  auto sizes = getPackedTransposeDimensions(shape, params);

  uint32_t cTypeSize = getBestCopyTypeSize(sizeof(DType), sizes.M, sizes.N);
  // Type Size Ratio
  const uint32_t TSR = cTypeSize/sizeof(DType);
  CHECK_EQ(cTypeSize, sizeof(DType)*TSR);

  // calculate kernel params
  uint32_t nThreadsPerBlock = 32*32/4;  // value chosen empirically
  uint32_t thdsY = 1;
  uint32_t thdsX = 1;
  if(sizes.M < sizes.N) {
    while(sizes.M > thdsY && thdsY < 32) {
      thdsY *= 2;
    }
    thdsX = nThreadsPerBlock/thdsY;
  } else {
    while(sizes.N > thdsX && thdsX < 32) {
      thdsX *= 2;
    }
    thdsY = nThreadsPerBlock/thdsX;
  }
  uint32_t blocksY = (sizes.M-1)/(thdsY*TSR) + 1;
  uint32_t blocksX = (sizes.N-1)/(thdsX*TSR) + 1;

  dim3 grid(blocksX, blocksY, sizes.leadDimS);
  dim3 block(thdsX, thdsY, 1);

  cudaStream_t stream = mshadow::Stream<gpu>::GetStream(s);
  call_transpose_pseudo2D(sizeof(DType), cTypeSize,
                          grid, block, stream,
                          outBlob.dptr_, inpBlob.dptr_, sizes.M, sizes.N);
}

}  // namespace op
}  // namespace mxnet


#endif  // MXNET_OPERATOR_TENSOR_PSEUDO2DTRANSPOSE_OP_INL_CUH_
