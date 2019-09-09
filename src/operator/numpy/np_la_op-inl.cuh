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
 * \file np_la_op-inl.cuh
 * \brief CUDA implementations for linalg norm
*/

#ifndef MXNET_OPERATOR_NUMPY_NP_LA_OP_INL_CUH_
#define MXNET_OPERATOR_NUMPY_NP_LA_OP_INL_CUH_

using namespace mshadow::cuda;

#define KERNEL_UNROLL_SWITCH(do_unroll, unrollAmount, unrollVar, ...) \
  if (do_unroll) {                                                    \
    const int unrollVar = unrollAmount;                               \
    {__VA_ARGS__}                                                     \
  } else {                                                            \
    const int unrollVar = 1;                                          \
    {__VA_ARGS__}                                                     \
  }

template<typename Reducer, int ndim, typename AType, typename DType, typename OType, typename OP, int unroll>
__launch_bounds__(nthread_reduce)
__global__ void norm_reduce_kernel(const int N, const int M, const bool addto,
                                   const DType* __restrict big, OType *small,
                                   const Shape<ndim> big_shape0, const Shape<ndim> small_shape,
                                   const Shape<ndim> big_shape, const Shape<ndim> big_stride,
                                   const int Mnext, const bool do_transpose, const double ord) {
  extern __shared__ char shTileChar[];
  AType* shTile = (AType*)(shTileChar);
  const int tid = threadIdx.x + threadIdx.y*blockDim.x;
  const int bx = (do_transpose) ? blockDim.y : blockDim.x;
  const int by = (do_transpose) ? blockDim.x : blockDim.y;
  const int tidx = (do_transpose) ? tid / by : threadIdx.x;
  const int tidy = (do_transpose) ? tid % by : threadIdx.y;
  for (int m0 = blockIdx.y; m0 < Mnext; m0 += gridDim.y) {
    // This TB handles M range [Mstart, ...., Mend - 1]
    const int Mstart = (int)((uint64_t)M*(uint64_t)m0/(uint64_t)Mnext);
    const int Mend   = (int)((uint64_t)M*(uint64_t)(m0 + 1)/(uint64_t)Mnext);
    for (int idx0 = blockIdx.x*bx; idx0 < N; idx0 += bx*gridDim.x) {
      int idx = idx0 + tidx;
      Shape<ndim> coord = unravel(idx, small_shape);
      int idx_big0 = ravel(coord, big_shape0);

      AType val;
      double residual= ord;
      Reducer::SetInitValue(val, residual);
      if (idx < N) {
        for (int k = tidy + Mstart; k < Mend; k += by*unroll) {
          int idx_big[unroll];
#pragma unroll
          for (int u=0;u < unroll;u++) {
            idx_big[u] = idx_big0 + unravel_dot(k + u*by, big_shape, big_stride);
          }
          DType tmp[unroll];
#pragma unroll
          for (int u=0;u < unroll;u++) {
            if (k + u*by < Mend) {
              tmp[u] = OP::Map(big[idx_big[u]]);
            }
          }
#pragma unroll
          for (int u=0;u < unroll;u++) {
            if (k + u*by < Mend) Reducer::Reduce(val, AType(tmp[u]), residual);
          }
        }
      }

      // Shared memory block bx * by. Reduction is along by. Final result is in tidy=0
      if (by > 1) {
        // Fix bx to avoid bank conflicts. Assumes warpSize number of banks
        const int fbx = (do_transpose && ((bx & (warpSize - 1)) == 0)) ? (bx + 1) : bx;
        const int it0 = tidx + tidy*fbx;
        shTile[it0 * 2] = val;
        shTile[it0 * 2 + 1] = residual;
        __syncthreads();
        for (int t=1;t < by; t <<= 1) {
          AType tmp, tmp_residual;
          Reducer::SetInitValue(tmp, tmp_residual);
          if (tidy + t < by) {
            tmp = shTile[(it0 + t*fbx) * 2];
            tmp_residual = shTile[(it0 + t*fbx) * 2 + 1];
          }
          __syncthreads();
          Reducer::Merge(shTile[it0 * 2], shTile[it0 * 2 + 1], tmp, tmp_residual);
          __syncthreads();
        }
        if (idx < N && tidy == 0) {
          Reducer::Finalize(shTile[tidx * 2], shTile[tidx * 2 + 1]);
          assign(&small[idx + m0*N], addto, OType(shTile[tidx * 2]));
        }
      } else {
        if (idx < N) {
          Reducer::Finalize(val, residual);
          assign(&small[idx + m0*N], addto, OType(val));
        }
      }
    }
  }
}

// Simple reduction of lines when M is small
template<typename Reducer, typename DType>
__launch_bounds__(kMaxThreadsPerBlock)
__global__ void norm_reduce_lines_kernel(const int N, const int M, const bool addto,
                                         const int small_in_stride, const DType* __restrict small_in, DType *small_out,
                                         const double ord) {
  for (int idx = threadIdx.x + blockIdx.x*blockDim.x; idx < N; idx += blockDim.x*gridDim.x) {
    DType val;
    double  residual = ord;
    Reducer::SetInitValue(val, residual);
    for (int k = 0; k < M; k++) {
      Reducer::Reduce(val, small_in[idx + k*small_in_stride], residual);
    }
    if (idx < N) {
      Reducer::Finalize(val, residual);
      assign(&small_out[idx], addto, val);
    }
  }
}

template<typename Reducer, int ndim, typename AType, typename DType, typename OType, typename OP>
__global__ void power_reduce_kernel_M1(const int N, const bool addto,
                                       const DType* __restrict big, OType *small, const Shape<ndim> bshape,
                                       const Shape<ndim> sshape, const double ord) {
  for (int idx = threadIdx.x + blockIdx.x*blockDim.x; idx < N; idx += blockDim.x*gridDim.x) {
    Shape<ndim> coord = unravel(idx, sshape);
    int j = ravel(coord, bshape);
    AType val;
    double residual = ord;
    Reducer::SetInitValue(val, residual);
    Reducer::Reduce(val, AType(OP::Map(big[j])), residual);
    Reducer::Finalize(val, residual);
    assign(&small[idx], addto, OType(val));
  }
}

template<typename Reducer, int ndim, typename AType, typename DType, typename OType, typename OP>
void PowerReduceImpl(cudaStream_t stream, const TBlob& small, const OpReqType req,
                     const TBlob& big, const Tensor<gpu, 1, char>& workspace,
                     const ReduceImplConfig<ndim>& config, const double ord) {
  if (config.M == 1) {
    power_reduce_kernel_M1<Reducer, ndim, AType, DType, OType, OP>
        <<< config.kernel_1.gridDim, config.kernel_1.blockDim, 0, stream >>>(
        config.N, req == kAddTo, big.dptr<DType>(), small.dptr<OType>(), big.shape_.get<ndim>(),
            small.shape_.get<ndim>(), ord);
    MSHADOW_CUDA_POST_KERNEL_CHECK(power_reduce_kernel_M1);
  } else {
    OType* small_dptr = small.dptr<OType>();
    bool addto = (req == kAddTo);
    if (config.Mnext > 1) {
      // small_dptr[] is N*Mnext*sizeof(DType) bytes
      small_dptr = reinterpret_cast<OType*>(workspace.dptr_);
      addto = false;
      // Check that the workspace is contigiuous
      CHECK_EQ(workspace.CheckContiguous(), true);
      // Check that we have enough storage
      CHECK_GE(workspace.size(0), config.workspace_size);
    }

    const int by = (config.kernel_1.do_transpose) ? config.kernel_1.blockDim.x : config.kernel_1.blockDim.y;
    const bool do_unroll = (config.M / (by*config.Mnext) >= config.unroll_reduce);
    KERNEL_UNROLL_SWITCH(do_unroll, ReduceImplConfig<ndim>::unroll_reduce, UNROLL, {
      norm_reduce_kernel<Reducer, ndim, AType, DType, OType, OP, UNROLL>
      <<< config.kernel_1.gridDim, config.kernel_1.blockDim, config.kernel_1.shMemSize, stream>>>(
          config.N, config.M, addto, big.dptr<DType>(), small_dptr, big.shape_.get<ndim>(),
              small.shape_.get<ndim>(), config.rshape, config.rstride, config.Mnext,
              config.kernel_1.do_transpose, ord);
    });
    MSHADOW_CUDA_POST_KERNEL_CHECK(reduce_kernel);

    if (config.Mnext > 1) {
      norm_reduce_lines_kernel<Reducer, OType>
      <<< config.kernel_2.gridSize, config.kernel_2.blockSize, 0, stream >>>
          (config.N, config.Mnext, req == kAddTo, config.N, small_dptr, small.dptr<OType>(), ord);
      MSHADOW_CUDA_POST_KERNEL_CHECK(norm_reduce_lines_kernel);
    }
  }
}
template<typename Reducer, int ndim, typename DType, typename OP, bool safe_acc = false>
void PowerReduce(Stream <gpu> *s,
                 const TBlob &small,
                 const OpReqType req,
                 const Tensor<gpu, 1, char> &workspace,
                 const TBlob &big,
                 const double ord) {
  if (req == kNullOp) return;
  cudaStream_t stream = Stream<gpu>::GetStream(s);
  ReduceImplConfig<ndim> config =
      ConfigureReduceImpl<ndim, DType>(small.shape_, big.shape_, NULL, NULL);
  if (safe_acc) {
    // TODO(haojin2): Use real-only type swtich for windows temporarily due to CI issues.
#ifndef _WIN32
    MXNET_ACC_TYPE_SWITCH(mshadow::DataType<DType>::kFlag, DataType, AType, {
        typedef typename std::conditional<safe_acc, AType, DataType>::type AccType;
        MSHADOW_TYPE_SWITCH(small.type_flag_, OType, {
          typedef typename std::conditional<safe_acc, OType, DataType>::type
              OutType;
          config = ConfigureReduceImpl<ndim, AccType>(small.shape_,
                                                      big.shape_,
                                                      NULL,
                                                      NULL);
          PowerReduceImpl<Reducer, ndim, AccType, DataType, OutType, OP>(
              stream, small, req, big, workspace, config, ord);
        });
    });
#else
    MXNET_REAL_ACC_TYPE_SWITCH(mshadow::DataType<DType>::kFlag, DataType, AType, {
      typedef typename std::conditional<safe_acc, AType, DataType>::type AccType;
      MSHADOW_TYPE_SWITCH(small.type_flag_, OType, {
        typedef typename std::conditional<safe_acc, OType, DataType>::type OutType;
        config = ConfigureReduceImpl<ndim, AccType>(small.shape_, big.shape_, NULL, NULL);
        PowerReduceImpl<Reducer, ndim, AccType, DataType, OutType, OP>(
          stream, small, req, big, workspace, config, ord);
      });
    });
#endif
  } else {
    PowerReduceImpl<Reducer, ndim, DType, DType, DType, OP>(stream, small, req, big, workspace, config, ord);
  }
}
#endif  // MXNET_OPERATOR_NUMPY_NP_LA_OP_INL_CUH_
