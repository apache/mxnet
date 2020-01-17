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

/*!
 * \brief The `transpose_pseudo2D` based on chosen vectorized types. It transposes an array of
 *    shape (k, m, n) to (k, n, m)
 * \param out Pointer to output memory.
 * \param inp Pointer to input memory.
 * \param m First of tensor dimensions.
 * \param n Second of tensor dimensions.
 * \param nIterY The number of iterations in the y-dim of the thread to cover all rows. (1-->m)
 * \param nIterZ The number of iterations in the z-dim of the thread to cover all rows. (1-->k)
 * \tparam DType Data type
 * \tparam CType The type to load the data.
 * \tparam is_addto Whether to perform out += transpose(data) or out = transpose(data)
 */
template <typename DType, typename CType, bool is_addto>
__global__ void transpose_pseudo2D(DType* out, DType* inp,
                                   const index_t m, const index_t n,
                                   const index_t nIterY, const index_t nIterZ) {
  // Calculate the TypeSizeRatio
  const index_t TSR = sizeof(CType) / sizeof(DType) > 0 ? sizeof(CType) / sizeof(DType) : 1;
  const index_t chunked_n = n/TSR;
  const index_t chunked_m = m/TSR;

  extern __shared__ char buf[];
  DType* d_shm = reinterpret_cast<DType*>(buf);
  CType* c_shm = reinterpret_cast<CType*>(buf);

  CType* cInp = reinterpret_cast<CType*>(inp);
  CType* cOut = reinterpret_cast<CType*>(out);

  for (index_t iterZ = 0; iterZ < nIterZ; iterZ++) {
    const index_t blockIdx_z = gridDim.z*iterZ + blockIdx.z;
    for (index_t iterY = 0; iterY < nIterY; iterY++) {
      const index_t blockIdx_y = gridDim.y*iterY + blockIdx.y;

      index_t offset = blockIdx_z*m*chunked_n
                     + blockIdx_y*blockDim.y*TSR*chunked_n
                     + (index_t)blockIdx.x*blockDim.x;

      if ((blockIdx.x*blockDim.x + threadIdx.x)*TSR < n
       && (blockIdx_y*blockDim.y + threadIdx.y)*TSR < m) {
        // read from global memory to shared
        #pragma unroll
        for (index_t i = 0; i < TSR; i++) {
          index_t shmIdx = (TSR*threadIdx.y + i)*blockDim.x + threadIdx.x;
          c_shm[shmIdx] = cInp[offset + (TSR*threadIdx.y + i)*chunked_n + threadIdx.x];
        }
        __syncthreads();

        // read from shared to local registers
        CType tmp[TSR];
        #pragma unroll
        for (index_t i = 0; i < TSR; i++) {
          DType* tmp_dptr = reinterpret_cast<DType*>(&tmp[i]);
          #pragma unroll
          for (int j = 0; j < TSR; j++) {
            index_t shmIdx = (TSR*threadIdx.y + j)*blockDim.x*TSR + TSR*threadIdx.x + i;
            tmp_dptr[j] = d_shm[shmIdx];
          }
        }
        __syncthreads();

        // write back to global output
        offset = blockIdx_z*m*chunked_n + blockIdx.x*blockDim.x*TSR*chunked_m
                                        + blockIdx_y*blockDim.y;
        #pragma unroll
        for (index_t i = 0; i < TSR; i++) {
          if (is_addto) {
            DType* tmp_dptr = reinterpret_cast<DType*>(&tmp[i]);
            #pragma unroll
            for (int j = 0; j < TSR; j++) {
              out[TSR * (offset + (TSR*threadIdx.x + i)*chunked_m + threadIdx.y) + j]
                += tmp_dptr[j];
            }
          } else {
            cOut[offset + (TSR*threadIdx.x + i)*chunked_m + threadIdx.y] = tmp[i];
          }
        }
      }
    }
  }
}

}  // namespace cuda


/*!
 * \brief Calls proper version of kernel `transpose_pseudo2D`
 *        basing on chosen type sizes.
 * \param cTypeSize Size of type that should be use to copy.
 * \param grid Grid dimensions for the kernel.
 * \param block Block dimensions for the kernel.
 * \param stream Strem to run kernel.
 * \param out Pointer to output memory.
 * \param inp Pointer to input memory.
 * \param m First of tensor dimensions.
 * \param n Second of tensor dimensions.
 * \tparam DType Data type
 * \tparam is_addto Whether to trigger add the transpose result to the output tensor.
 */
template <typename DType, bool is_addto>
inline void call_transpose_pseudo2D(index_t cTypeSize,
                                    dim3 grid, dim3 block, cudaStream_t stream,
                                    DType* d_outPtr, DType* d_inpPtr,
                                    const index_t m, const index_t n,
                                    const index_t nIterY, const index_t nIterZ) {
  const int nshared = 1024 * cTypeSize / sizeof(DType) * cTypeSize;
  switch (cTypeSize) {
    case (1):
      cuda::transpose_pseudo2D<DType, uint8_t, is_addto><<<grid, block, nshared, stream>>>
                              (d_outPtr, d_inpPtr, m, n, nIterY, nIterZ);
      break;
    case (2):
      cuda::transpose_pseudo2D<DType, uint16_t, is_addto><<<grid, block, nshared, stream>>>
                              (d_outPtr, d_inpPtr, m, n, nIterY, nIterZ);
      break;
    case (4):
      cuda::transpose_pseudo2D<DType, uint32_t, is_addto><<<grid, block, nshared, stream>>>
                              (d_outPtr, d_inpPtr, m, n, nIterY, nIterZ);
      break;
    case (8):
      cuda::transpose_pseudo2D<DType, uint64_t, is_addto><<<grid, block, nshared, stream>>>
                              (d_outPtr, d_inpPtr, m, n, nIterY, nIterZ);
      break;
    default:
      LOG(FATAL) << "Unsupported type combination. " << "Copy type size = " << cTypeSize;
  }
  auto cuErr = cudaPeekAtLastError();
  CHECK_EQ(cuErr, cudaSuccess) << "TransposePseudo2D kernel failure: "
                               << cudaGetErrorString(cuErr) << ". "
                               << "block: (" << block.x << "," << block.y << "," << block.z << ")"
                               << " grid: (" << grid.x << "," << grid.y << "," << grid.z << ")";
}


/*!
 * \brief Checks if function `transpose_pseudo2D` can be used
 *        to perform transpose operation with given params.
 * \param params Parameters (axes) of the transpose.
 */
inline bool isPseudo2DTranspose(const TShape& params) {
  index_t n_swpDims = 1;
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
  index_t leadDimS;
  index_t M;
  index_t N;
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


inline int32_t getBestCopyTypeSize(index_t dTypeSize, index_t sizeM, index_t sizeN) {
  index_t cTypeSize = std::max((index_t)8, dTypeSize);
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


inline std::pair<dim3, dim3> calculateKernelParams(pseudo2DSizes sizes, const index_t TSR) {
  index_t nThreadsPerBlock = 32*32/4;  // value chosen empirically
  index_t thdsY = 1;
  index_t thdsX = 1;
  while(sizes.N/TSR > thdsX && thdsX < 32) {
    thdsX *= 2;
  }
  thdsY = nThreadsPerBlock/thdsX;
  thdsY = std::min(sizes.M/TSR, thdsY);
  index_t blocksY = (sizes.M/TSR-1)/thdsY + 1;
  index_t blocksX = (sizes.N/TSR-1)/thdsX + 1;

  dim3 grid(blocksX, blocksY, sizes.leadDimS);
  dim3 block(thdsX, thdsY);
  return {grid, block};
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
 * \param is_addto Whether to add the transpose result to the outBlob
 * \param s Pointer to GPU stream.
 */
template <typename DType, bool is_addto>
void transpose_pseudo2D(const TBlob& outBlob, const TBlob& inpBlob,
                        const TShape& params, mshadow::Stream<gpu>* s) {
  const TShape& shape = inpBlob.shape_;
  CHECK_EQ(shape.ndim(), params.ndim());
  auto sizes = getPackedTransposeDimensions(shape, params);

  index_t cTypeSize = getBestCopyTypeSize(sizeof(DType), sizes.M, sizes.N);
  // Type Size Ratio
  const index_t TSR = cTypeSize/sizeof(DType);
  CHECK_EQ(cTypeSize, sizeof(DType)*TSR);

  auto pair = calculateKernelParams(sizes, TSR);
  dim3 grid = pair.first;
  dim3 block = pair.second;
  index_t nIterY = 1;
  if (grid.y > std::numeric_limits<uint16_t>::max()) {
    nIterY = (grid.y - 1)/(std::numeric_limits<uint16_t>::max() - 1) + 1;
    grid.y = (grid.y - 1)/nIterY + 1;
  }
  index_t nIterZ = 1;
  if (grid.z > std::numeric_limits<uint16_t>::max()) {
    nIterZ = (grid.z - 1)/(std::numeric_limits<uint16_t>::max() - 1) + 1;
    grid.z = (grid.z - 1)/nIterZ + 1;
  }

  cudaStream_t stream = mshadow::Stream<gpu>::GetStream(s);
  call_transpose_pseudo2D<DType, is_addto>
      (cTypeSize, grid, block, stream,
       outBlob.dptr<DType>(), inpBlob.dptr<DType>(),
       sizes.M, sizes.N, nIterY, nIterZ);
}

}  // namespace op
}  // namespace mxnet


#endif  // MXNET_OPERATOR_TENSOR_PSEUDO2DTRANSPOSE_OP_INL_CUH_
