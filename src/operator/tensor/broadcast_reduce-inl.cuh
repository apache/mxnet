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
 * Copyright (c) 2015-2020 by Contributors
 * \file broadcast_reduce-inl.cuh
 * \brief CUDA implementations for binary broadcast and reduce
 * \author Antti-Pekka Hynninen, Przemyslaw Tredak
*/
#ifndef MXNET_OPERATOR_TENSOR_BROADCAST_REDUCE_INL_CUH_
#define MXNET_OPERATOR_TENSOR_BROADCAST_REDUCE_INL_CUH_

using namespace mshadow::cuda;

template<typename Reducer, int ndim, typename AType, typename DType, typename OType, typename OP, int unroll,
         typename IndexOP = mxnet::op::mshadow_op::set_index_no_op<AType, int>>
__launch_bounds__(nthread_reduce)
__global__ void reduce_kernel(const int N, const int M, const bool addto,
                              const DType* __restrict big, OType *small,
                              const Shape<ndim> big_shape0, const Shape<ndim> small_shape,
                              const Shape<ndim> big_shape, const Shape<ndim> big_stride,
                              const int Mnext, const bool do_transpose) {
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
      Shape<ndim> coord = mxnet_op::unravel(idx, small_shape);
      int idx_big0 = mxnet_op::ravel(coord, big_shape0);

      AType val, residual;
      Reducer::SetInitValue(val, residual);
      if (idx < N) {
        for (int k = tidy + Mstart; k < Mend; k += by*unroll) {
          int idx_big[unroll];
          #pragma unroll
          for (int u=0;u < unroll;u++) {
            idx_big[u] = idx_big0 + mxnet_op::unravel_dot(k + u*by, big_shape, big_stride);
          }
          AType tmp[unroll];
          #pragma unroll
          for (int u=0;u < unroll;u++) {
            if (k + u*by < Mend) {
              tmp[u] = OP::Map(big[idx_big[u]]);
              // argmin/max, set IndexedNum.idx
              if (IndexOP::do_op)
                IndexOP::Op(&tmp[u], k+u*by);
            }
          }
          #pragma unroll
          for (int u=0;u < unroll;u++) {
            if (k + u*by < Mend) Reducer::Reduce(val, tmp[u], residual);
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
        for (int t=1;t < by;t <<= 1) {
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

template<typename Reducer, int ndim, typename DType, typename OP1, typename OP2, int unroll>
__launch_bounds__(nthread_reduce)
__global__ void reduce_kernel(const int N, const int M, const bool addto,
                              const DType* __restrict big, const DType* __restrict lhs,
                              const DType* __restrict rhs, DType *small,
                              const Shape<ndim> big_shape0, const Shape<ndim> lhs_shape0,
                              const Shape<ndim> rhs_shape0, const Shape<ndim> small_shape,
                              const Shape<ndim> big_shape, const Shape<ndim> lhs_shape,
                              const Shape<ndim> rhs_shape, const Shape<ndim> big_stride,
                              const Shape<ndim> lhs_stride, const Shape<ndim> rhs_stride,
                              const int Mnext, const bool do_transpose) {
  extern __shared__ char shTileChar[];
  DType* shTile = (DType*)(shTileChar);
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
      Shape<ndim> coord = mxnet_op::unravel(idx, small_shape);
      int idx_big0 = mxnet_op::ravel(coord, big_shape0);
      int idx_lhs0 = mxnet_op::ravel(coord, lhs_shape0);
      int idx_rhs0 = mxnet_op::ravel(coord, rhs_shape0);

      DType val, residual;
      Reducer::SetInitValue(val, residual);
      if (idx < N) {
        for (int k = tidy + Mstart; k < Mend; k += by*unroll) {
          int idx_big[unroll];
          int idx_lhs[unroll];
          int idx_rhs[unroll];
          #pragma unroll
          for (int u=0;u < unroll;u++) {
            idx_big[u] = idx_big0 + mxnet_op::unravel_dot(k + u*by, big_shape, big_stride);
            idx_lhs[u] = idx_lhs0 + mxnet_op::unravel_dot(k + u*by, lhs_shape, lhs_stride);
            idx_rhs[u] = idx_rhs0 + mxnet_op::unravel_dot(k + u*by, rhs_shape, rhs_stride);
          }
          DType tmp[unroll];
          #pragma unroll
          for (int u=0;u < unroll;u++) {
            if (k + u*by < Mend) {
              tmp[u] = OP1::Map(big[idx_big[u]], OP2::Map(lhs[idx_lhs[u]], rhs[idx_rhs[u]]));
            }
          }
          #pragma unroll
          for (int u=0;u < unroll;u++) {
            if (k + u*by < Mend) Reducer::Reduce(val, tmp[u], residual);
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
        for (int t=1;t < by;t <<= 1) {
          DType tmp, tmp_residual;
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
          assign(&small[idx + m0*N], addto, shTile[tidx * 2]);
        }
      } else {
        if (idx < N) {
          Reducer::Finalize(val, residual);
          assign(&small[idx + m0*N], addto, val);
        }
      }
    }
  }
}

// Simple reduction of lines when M is small
template<typename Reducer, typename DType>
__launch_bounds__(kMaxThreadsPerBlock)
__global__ void reduce_lines_kernel(const int N, const int M, const bool addto,
  const int small_in_stride, const DType* __restrict small_in, DType *small_out) {
  for (int idx = threadIdx.x + blockIdx.x*blockDim.x; idx < N; idx += blockDim.x*gridDim.x) {

    DType val, residual;
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
__launch_bounds__(kMaxThreadsPerBlock)
__global__ void reduce_kernel_M1(const int N, const bool addto,
                                const DType* __restrict big, OType *small, const Shape<ndim> bshape,
                                const Shape<ndim> sshape) {
  for (int idx = threadIdx.x + blockIdx.x*blockDim.x; idx < N; idx += blockDim.x*gridDim.x) {
    Shape<ndim> coord = mxnet_op::unravel(idx, sshape);
    int j = mxnet_op::ravel(coord, bshape);
    AType val, residual, temp = OP::Map(big[j]);
    Reducer::SetInitValue(val, residual);
    Reducer::Reduce(val, temp, residual);
    Reducer::Finalize(val, residual);
    assign(&small[idx], addto, OType(val));
  }
}

template<typename Reducer, int ndim, typename DType, typename OP1, typename OP2>
__launch_bounds__(kMaxThreadsPerBlock)
__global__ void reduce_kernel_M1(const int N, const bool addto,
                                 const DType* __restrict big,
                                 const DType* __restrict lhs,
                                 const DType* __restrict rhs,
                                 DType *small,
                                 const Shape<ndim> big_shape,
                                 const Shape<ndim> lhs_shape,
                                 const Shape<ndim> rhs_shape,
                                 const Shape<ndim> small_shape) {
  for (int idx = threadIdx.x + blockIdx.x*blockDim.x; idx < N; idx += blockDim.x*gridDim.x) {
    Shape<ndim> coord = mxnet_op::unravel(idx, small_shape);
    int idx_big = mxnet_op::ravel(coord, big_shape);
    int idx_lhs = mxnet_op::ravel(coord, lhs_shape);
    int idx_rhs = mxnet_op::ravel(coord, rhs_shape);
    DType val, residual;
    Reducer::SetInitValue(val, residual);
    Reducer::Reduce(val, OP1::Map(big[idx_big], OP2::Map(lhs[idx_lhs], rhs[idx_rhs])), residual);
    Reducer::Finalize(val, residual);
    assign(&small[idx], addto, val);
  }
}

#define KERNEL_UNROLL_SWITCH(do_unroll, unrollAmount, unrollVar, ...) \
  if (do_unroll) {                                                    \
    const int unrollVar = unrollAmount;                               \
    {__VA_ARGS__}                                                     \
  } else {                                                            \
    const int unrollVar = 1;                                          \
    {__VA_ARGS__}                                                     \
  }

template<typename Reducer, int ndim, typename AType, typename DType, typename OType, typename OP,
	 typename IndexOP = mxnet::op::mshadow_op::set_index_no_op<AType, int>>
void ReduceImpl(cudaStream_t stream, const TBlob& small, const OpReqType req,
                const TBlob& big, const Tensor<gpu, 1, char>& workspace,
                const ReduceImplConfig& config) {
  if (config.M == 1) {
    reduce_kernel_M1<Reducer, ndim, AType, DType, OType, OP>
    <<< config.kernel_1.gridDim, config.kernel_1.blockDim, 0, stream >>>(
      config.N, req == kAddTo, big.dptr<DType>(), reinterpret_cast<OType*>(small.dptr_),
      big.shape_.get<ndim>(), small.shape_.get<ndim>());
    MSHADOW_CUDA_POST_KERNEL_CHECK(reduce_kernel_M1);
  } else {
    OType* small_dptr = reinterpret_cast<OType*>(small.dptr_);
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

    const int by = (config.kernel_1.do_transpose) ?
      config.kernel_1.blockDim.x : config.kernel_1.blockDim.y;
    const bool do_unroll = ( config.M / (by*config.Mnext) >= unroll_reduce );
    KERNEL_UNROLL_SWITCH(do_unroll, unroll_reduce, UNROLL, {
      reduce_kernel<Reducer, ndim, AType, DType, OType, OP, UNROLL, IndexOP>
      <<< config.kernel_1.gridDim, config.kernel_1.blockDim, config.kernel_1.shMemSize, stream>>>(
        config.N, config.M, addto, big.dptr<DType>(), small_dptr, big.shape_.get<ndim>(),
        small.shape_.get<ndim>(), config.rshape.get<ndim>(), config.rstride.get<ndim>(),
        config.Mnext, config.kernel_1.do_transpose);
    });
    MSHADOW_CUDA_POST_KERNEL_CHECK(reduce_kernel);

    if (config.Mnext > 1) {
      reduce_lines_kernel<Reducer, OType>
      <<< config.kernel_2.gridSize, config.kernel_2.blockSize, 0, stream >>>
        (config.N, config.Mnext, req == kAddTo, config.N, small_dptr,
         reinterpret_cast<OType*>(small.dptr_));
      MSHADOW_CUDA_POST_KERNEL_CHECK(reduce_lines_kernel);
    }
  }
}

template<typename Reducer, int ndim, typename DType, typename OP1, typename OP2>
void ReduceImpl(cudaStream_t stream, const TBlob& small, const TBlob& lhs, const TBlob& rhs,
                const OpReqType req, const TBlob& big, const Tensor<gpu, 1, char>& workspace,
                const ReduceImplConfig& config) {
  if (config.M == 1) {
    reduce_kernel_M1<Reducer, ndim, DType, OP1, OP2>
    <<< config.kernel_1.gridDim, config.kernel_1.blockDim, 0, stream >>>(
      config.N, req == kAddTo, big.dptr<DType>(), lhs.dptr<DType>(), rhs.dptr<DType>(),
      small.dptr<DType>(), big.shape_.get<ndim>(), lhs.shape_.get<ndim>(),
      rhs.shape_.get<ndim>(), small.shape_.get<ndim>());
    MSHADOW_CUDA_POST_KERNEL_CHECK(reduce_kernel_M1);
  } else {
    DType* small_dptr = small.dptr<DType>();
    bool addto = (req == kAddTo);
    if (config.Mnext > 1) {
      // small_dptr[] is N*Mnext*sizeof(DType) bytes
      small_dptr = reinterpret_cast<DType*>(workspace.dptr_);
      addto = false;
      // Check that the workspace is contigiuous
      CHECK_EQ(workspace.CheckContiguous(), true);
      // Check that we have enough storage
      CHECK_GE(workspace.size(0), config.workspace_size);
    }

    const int by = (config.kernel_1.do_transpose) ?
      config.kernel_1.blockDim.x : config.kernel_1.blockDim.y;
    const bool do_unroll = ( config.M / (by*config.Mnext) >= unroll_reduce );
    KERNEL_UNROLL_SWITCH(do_unroll, unroll_reduce, UNROLL, {
      reduce_kernel<Reducer, ndim, DType, OP1, OP2, UNROLL>
      <<< config.kernel_1.gridDim, config.kernel_1.blockDim, config.kernel_1.shMemSize, stream>>>(
        config.N, config.M, addto, big.dptr<DType>(), lhs.dptr<DType>(), rhs.dptr<DType>(),
        small_dptr, big.shape_.get<ndim>(), lhs.shape_.get<ndim>(),
        rhs.shape_.get<ndim>(), small.shape_.get<ndim>(), config.rshape.get<ndim>(),
        config.lhs_shape.get<ndim>(), config.rhs_shape.get<ndim>(), config.rstride.get<ndim>(),
        config.lhs_stride.get<ndim>(), config.rhs_stride.get<ndim>(), config.Mnext,
        config.kernel_1.do_transpose);
      MSHADOW_CUDA_POST_KERNEL_CHECK(reduce_kernel);
    });

    if (config.Mnext > 1) {
      reduce_lines_kernel<Reducer, DType>
      <<< config.kernel_2.gridSize, config.kernel_2.blockSize, 0, stream >>>
        (config.N, config.Mnext, req == kAddTo, config.N, small_dptr, small.dptr<DType>());
      MSHADOW_CUDA_POST_KERNEL_CHECK(reduce_lines_kernel);
    }
  }
}

#undef KERNEL_UNROLL_SWITCH

template<typename Reducer, int ndim, typename DType, typename OP, bool safe_acc = false>
void Reduce(Stream<gpu> *s, const TBlob& small, const OpReqType req,
            const Tensor<gpu, 1, char>& workspace, const TBlob& big) {
  if (req == kNullOp) return;
  cudaStream_t stream = Stream<gpu>::GetStream(s);
  ReduceImplConfig config(small.shape_, big.shape_, nullptr, nullptr, sizeof(DType));
  if (safe_acc) {
    MXNET_ACC_TYPE_SWITCH(mshadow::DataType<DType>::kFlag, DataType, AType, {
      typedef typename std::conditional<safe_acc, AType, DataType>::type AccType;
      MSHADOW_TYPE_SWITCH(small.type_flag_, OType, {
        typedef typename std::conditional<safe_acc, OType, DataType>::type OutType;
        config = ReduceImplConfig(small.shape_, big.shape_, nullptr, nullptr,
                                  sizeof(AccType));
        ReduceImpl<Reducer, ndim, AccType, DataType, OutType, OP>(
          stream, small, req, big, workspace, config);
      });
    });
  } else {
    ReduceImpl<Reducer, ndim, DType, DType, DType, OP>(stream, small, req, big, workspace, config);
  }
}

template<typename Reducer, int ndim, typename DType, typename OP, bool safe_acc = false>
void ReduceBool(Stream<gpu> *s, const TBlob& small, const OpReqType req,
                const Tensor<gpu, 1, char>& workspace, const TBlob& big) {
  if (req == kNullOp) return;
  cudaStream_t stream = Stream<gpu>::GetStream(s);
  ReduceImplConfig config(small.shape_, big.shape_, nullptr, nullptr, sizeof(DType));
  ReduceImpl<Reducer, ndim, bool, DType, bool, OP>(stream, small, req, big, workspace, config);
}

template <typename Reducer, int ndim, typename DType, typename OP>
void ReduceWithExtraMem(Stream<gpu>* s, const TBlob& small, const OpReqType req,
                        const Tensor<gpu, 1, char>& workspace, const TBlob& big) {};

template<typename Reducer, int ndim, typename DType, typename OP1, typename OP2>
void Reduce(Stream<gpu> *s, const TBlob& small, const OpReqType req,
            const Tensor<gpu, 1, char>& workspace, const TBlob& big,
            const TBlob& lhs, const TBlob& rhs) {
  if (req == kNullOp) return;
  cudaStream_t stream = Stream<gpu>::GetStream(s);
  ReduceImplConfig config(small.shape_, big.shape_, &lhs.shape_, &rhs.shape_, sizeof(DType));
  ReduceImpl<Reducer, ndim, DType, OP1, OP2>(stream, small, lhs, rhs, req, big, workspace, config);
}

#endif  //MXNET_OPERATOR_TENSOR_BROADCAST_REDUCE_INL_CUH_
