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
 *  Copyright (c) 2015-2017 by Contributors
 * \file broadcast_reduce-inl.h
 * \brief CPU-specific Function definition of broadcast and reduce operators
 */
#ifndef MXNET_OPERATOR_TENSOR_BROADCAST_REDUCE_INL_H_
#define MXNET_OPERATOR_TENSOR_BROADCAST_REDUCE_INL_H_

#include <mxnet/operator_util.h>
#include <algorithm>
#include <vector>
#include <string>
#include <utility>
#include "../mshadow_op.h"
#include "../operator_common.h"

namespace mxnet {
namespace op {
namespace broadcast {
using namespace mshadow;

const int MAX_DIM = 5;

template<int ndim>
MSHADOW_XINLINE Shape<ndim> calc_stride(const Shape<ndim>& shape) {
  Shape<ndim> stride;
  index_t cumprod = 1;
  #pragma unroll
  for (int i = ndim - 1; i >= 0; --i) {
    stride[i] = (shape[i] > 1) ? cumprod : 0;
    cumprod *= shape[i];
  }
  return stride;
}

template<int ndim>
MSHADOW_XINLINE void unravel_dot(const index_t idx, const Shape<ndim>& shape,
  const Shape<ndim>& stridej, const Shape<ndim>& stridek, index_t* j, index_t* k) {
  *j = 0;
  *k = 0;
  #pragma unroll
  for (index_t i = ndim-1, idx_t = idx; i >=0; --i) {
    const auto tmp = idx_t / shape[i];
    const auto coord = idx_t - tmp*shape[i];
    *j += coord*stridej[i];
    *k += coord*stridek[i];
    idx_t = tmp;
  }
}

template<int ndim>
MSHADOW_XINLINE Shape<ndim> unravel(const index_t idx, const Shape<ndim>& shape) {
  Shape<ndim> ret;
  #pragma unroll
  for (index_t i = ndim-1, j = idx; i >=0; --i) {
    auto tmp = j / shape[i];
    ret[i] = j - tmp*shape[i];
    j = tmp;
  }
  return ret;
}

template<int ndim>
MSHADOW_XINLINE index_t ravel(const Shape<ndim>& coord, const Shape<ndim>& shape) {
  index_t ret = 0;
  #pragma unroll
  for (index_t i = 0; i < ndim; ++i) {
    ret = ret * shape[i] + (shape[i] > 1) * coord[i];
  }
  return ret;
}

template<int ndim>
MSHADOW_XINLINE int diff(const Shape<ndim>& small, const Shape<ndim>& big, Shape<ndim>* dims,
  Shape<ndim>* stride) {
  int mdim = 0;
  #pragma unroll
  for (int i = 0; i < ndim; ++i) {
    mdim += small[i] != big[i];
    (*dims)[i] = (*stride)[i] = 1;
  }

  index_t s = 1;
  #pragma unroll
  for (int i = ndim - 1, j = mdim; i >= 0; --i) {
    if (small[i] != big[i]) {
      --j;
      (*stride)[j] = s;
      (*dims)[j] = big[i];
    }
    s *= big[i];
  }
  return mdim;
}

template<int ndim>
MSHADOW_XINLINE index_t unravel_dot(const index_t idx, const Shape<ndim>& shape,
  const Shape<ndim>& stride) {
  index_t ret = 0;
  #pragma unroll
  for (index_t i = ndim-1, j = idx; i >=0; --i) {
    auto tmp = j / shape[i];
    ret += (j - tmp*shape[i])*stride[i];
    j = tmp;
  }
  return ret;
}

template<int ndim>
MSHADOW_XINLINE index_t dot(const Shape<ndim>& coord, const Shape<ndim>& stride) {
  index_t ret = 0;
  #pragma unroll
  for (int i = 0; i < ndim; ++i)
    ret += coord[i] * stride[i];
  return ret;
}

template<typename DType>
MSHADOW_XINLINE void assign(DType* dst, const bool addto, const DType src) {
  if (addto) {
    *dst += src;
  } else {
    *dst = src;
  }
}

template<int ndim, typename DType, typename OP>
MSHADOW_XINLINE void binary_broadcast_assign(const index_t idx, const bool addto,
                                             const DType* __restrict lhs,
                                             const DType* __restrict rhs, DType* out,
                                             const Shape<ndim>& lshape, const Shape<ndim>& rshape,
                                             const Shape<ndim>& oshape) {
  const Shape<ndim> coord = unravel(idx, oshape);
  const index_t j = ravel(coord, lshape);
  const index_t k = ravel(coord, rshape);
  assign(&out[idx], addto, OP::Map(lhs[j], rhs[k]));
}

template<typename Reducer, int ndim, typename AType, typename DType, typename OType, typename OP>
MSHADOW_XINLINE void seq_reduce_assign(const index_t idx, const size_t M, const bool addto,
                                       const DType* __restrict big, OType *small,
                                       const Shape<ndim>& bshape, const Shape<ndim>& sshape,
                                       const Shape<ndim>& rshape, const Shape<ndim>& rstride) {
  Shape<ndim> coord = unravel(idx, sshape);
  index_t j = ravel(coord, bshape);
  AType val, residual;
  Reducer::SetInitValue(val, residual);
  for (size_t k = 0; k < M; ++k) {
    coord = unravel(k, rshape);
    Reducer::Reduce(val, AType(OP::Map(big[j + dot(coord, rstride)])), residual);
  }
  Reducer::Finalize(val, residual);
  assign(&small[idx], addto, OType(val));
}

#ifdef __CUDACC__
#include "broadcast_reduce-inl.cuh"

#else

template<int ndim, typename DType, typename OP>
void binary_broadcast_compute(const size_t N, const bool addto, const DType *lhs,
                              const DType *rhs, DType *out, const Shape<ndim> lshape,
                              const Shape<ndim> rshape, const Shape<ndim> oshape) {
  for (size_t idx = 0; idx < N; ++idx) {
    binary_broadcast_assign<ndim, DType, OP>(idx, addto, lhs, rhs, out, lshape, rshape, oshape);
  }
}

template<int ndim, typename DType, typename OP>
void BinaryBroadcastComputeImpl(Stream<cpu> *s, const OpReqType req,
                                const TBlob& lhs, const TBlob& rhs, const TBlob& out) {
  if (req == kNullOp) return;
  size_t N = out.shape_.Size();
  binary_broadcast_compute<ndim, DType, OP>(N, req == kAddTo, lhs.dptr<DType>(), rhs.dptr<DType>(),
                           out.dptr<DType>(), lhs.shape_.get<ndim>(), rhs.shape_.get<ndim>(),
                           out.shape_.get<ndim>());
}

template<typename Reducer, int ndim, typename AType, typename DType, typename OType, typename OP>
void seq_reduce_compute(const size_t N, const size_t M, const bool addto,
                        const DType *big, OType *small, const Shape<ndim> bshape,
                        const Shape<ndim> sshape, const Shape<ndim> rshape,
                        const Shape<ndim> rstride) {
  #pragma omp parallel for num_threads(engine::OpenMP::Get()->GetRecommendedOMPThreadCount())
  for (index_t idx = 0; idx < static_cast<index_t>(N); ++idx) {
    seq_reduce_assign<Reducer, ndim, AType, DType, OType, OP>(idx, M, addto, big, small,
        bshape, sshape, rshape, rstride);
  }
}

template <typename Reducer, int ndim, typename DType, typename OP>
void seq_reduce_compute_extra_mem(const size_t N, const size_t M, const bool addto,
                                  const DType* big, DType* small,
                                  const Shape<ndim> bshape,
                                  const Shape<ndim> sshape,
                                  const Shape<ndim> rshape,
                                  const Shape<ndim> rstride,
                                  const index_t* ws_dptr) {
  #pragma omp parallel for num_threads(engine::OpenMP::Get()->GetRecommendedOMPThreadCount())
  for (index_t idx = 0; idx < static_cast<index_t>(N); ++idx) {
    Shape<ndim> coord = unravel(idx, sshape);
    index_t j = ravel(coord, bshape);
    DType val, residual;
    Reducer::SetInitValue(val, residual);
    for (size_t k = 0; k < M; ++k) {
      Reducer::Reduce(val, OP::Map(big[j + ws_dptr[k]]), residual);
    }
    assign(&small[idx], addto, val);
  }
}

template <typename Reducer, int ndim, typename DType, typename OP, bool safe_acc = false>
void Reduce(Stream<cpu>* s, const TBlob& small, const OpReqType req,
            const Tensor<cpu, 1, char>& workspace, const TBlob& big) {
  if (req == kNullOp) return;
  Shape<ndim> rshape, rstride;
  diff(small.shape_.get<ndim>(), big.shape_.get<ndim>(), &rshape, &rstride);
  size_t N = small.shape_.Size(), M = rshape.Size();
  if (!safe_acc) {
    seq_reduce_compute<Reducer, ndim, DType, DType, DType, OP>(
      N, M, req == kAddTo, big.dptr<DType>(), small.dptr<DType>(),
      big.shape_.get<ndim>(), small.shape_.get<ndim>(), rshape, rstride);
  } else {
    MXNET_ACC_TYPE_SWITCH(mshadow::DataType<DType>::kFlag, DataType, AType, {
      typedef typename std::conditional<safe_acc, AType, DataType>::type AccType;
      MSHADOW_TYPE_SWITCH_WITH_BOOL(small.type_flag_, OType, {
        typedef typename std::conditional<safe_acc, OType, DataType>::type OutType;
        seq_reduce_compute<Reducer, ndim, AccType, DataType, OutType, OP>(
          N, M, req == kAddTo, big.dptr<DataType>(), small.dptr<OutType>(),
          big.shape_.get<ndim>(), small.shape_.get<ndim>(), rshape, rstride);
      });
    });
  }
}

template <typename Reducer, int ndim, typename DType, typename OP>
void ReduceWithExtraMem(Stream<cpu>* s, const TBlob& small, const OpReqType req,
                        const Tensor<cpu, 1, char>& workspace, const TBlob& big) {
  using namespace mxnet_op;
  if (req == kNullOp) return;
  Shape<ndim> rshape, rstride;
  diff(small.shape_.get<ndim>(), big.shape_.get<ndim>(), &rshape, &rstride);
  index_t* ws_dptr = reinterpret_cast<index_t*>(workspace.dptr_);
  size_t N = small.shape_.Size(), M = rshape.Size();
  #pragma omp parallel for num_threads(engine::OpenMP::Get()->GetRecommendedOMPThreadCount())
  for (index_t k = 0; k < static_cast<index_t>(M); k++) {
    Shape<ndim> coord = unravel(k, rshape);
    ws_dptr[k] = dot(coord, rstride);
  }

  seq_reduce_compute_extra_mem<Reducer, ndim, DType, OP>(
    N, M, req == kAddTo, big.dptr<DType>(), small.dptr<DType>(), big.shape_.get<ndim>(),
    small.shape_.get<ndim>(), rshape, rstride, ws_dptr);
}

template<int ndim, typename DType>
size_t ReduceWorkspaceSize(Stream<cpu> *s, const mxnet::TShape& small, const OpReqType req,
                           const mxnet::TShape& big) {
  return 0;
}

template<int ndim, typename DType>
size_t ReduceWorkspaceSize(Stream<cpu> *s, const mxnet::TShape& small, const OpReqType req,
                           const mxnet::TShape& big, const mxnet::TShape& lhs,
                           const mxnet::TShape& rhs) {
  return 0;
}

template<typename Reducer, int ndim, typename DType, typename OP1, typename OP2>
MSHADOW_XINLINE void seq_reduce_assign(const index_t idx, const size_t M, const bool addto,
                                       const DType* __restrict big, const DType* __restrict lhs,
                                       const DType* __restrict rhs, DType *small,
                                       const Shape<ndim>& big_shape, const Shape<ndim>& lhs_shape0,
                                       const Shape<ndim>& rhs_shape0,
                                       const Shape<ndim>& small_shape, const Shape<ndim>& rshape,
                                       const Shape<ndim>& lhs_shape, const Shape<ndim>& rhs_shape,
                                       const Shape<ndim>& rstride, const Shape<ndim>& lhs_stride,
                                       const Shape<ndim>& rhs_stride) {
  Shape<ndim> coord = unravel(idx, small_shape);
  const index_t idx_big0 = ravel(coord, big_shape);
  const index_t idx_lhs0 = ravel(coord, lhs_shape0);
  const index_t idx_rhs0 = ravel(coord, rhs_shape0);
  DType val, residual;
  Reducer::SetInitValue(val, residual);
  for (size_t k = 0; k < M; ++k) {
    Shape<ndim> coord_big = unravel(k, rshape);
    index_t idx_big = idx_big0 + dot(coord_big, rstride);

    Shape<ndim> coord_lhs = unravel(k, lhs_shape);
    index_t idx_lhs = idx_lhs0 + dot(coord_lhs, lhs_stride);

    Shape<ndim> coord_rhs = unravel(k, rhs_shape);
    index_t idx_rhs = idx_rhs0 + dot(coord_rhs, rhs_stride);

    Reducer::Reduce(val, OP1::Map(big[idx_big], OP2::Map(lhs[idx_lhs], rhs[idx_rhs])), residual);
  }
  Reducer::Finalize(val, residual);
  assign(&small[idx], addto, val);
}

template<typename Reducer, int ndim, typename DType, typename OP1, typename OP2>
void seq_reduce_compute(const size_t N, const size_t M, const bool addto,
                        const DType *big, const DType *lhs, const DType *rhs, DType *small,
                        const Shape<ndim> big_shape, const Shape<ndim> small_shape,
                        const Shape<ndim> rshape, const Shape<ndim> rstride,
                        const Shape<ndim> lhs_shape, const Shape<ndim> lhs_stride,
                        const Shape<ndim> rhs_shape, const Shape<ndim> rhs_stride,
                        const Shape<ndim>& lhs_shape0, const Shape<ndim>& rhs_shape0) {
  #pragma omp parallel for num_threads(engine::OpenMP::Get()->GetRecommendedOMPThreadCount())
  for (index_t idx = 0; idx < static_cast<index_t>(N); ++idx) {
    seq_reduce_assign<Reducer, ndim, DType, OP1, OP2>(idx, M, addto, big, lhs, rhs, small,
      big_shape, lhs_shape0, rhs_shape0, small_shape, rshape, lhs_shape, rhs_shape, rstride,
      lhs_stride, rhs_stride);
  }
}

template<typename Reducer, int ndim, typename DType, typename OP1, typename OP2>
void Reduce(Stream<cpu> *s, const TBlob& small, const OpReqType req,
            const Tensor<cpu, 1, char>& workspace, const TBlob& big, const TBlob& lhs,
            const TBlob& rhs) {
  if (req == kNullOp) return;
  Shape<ndim> rshape, rstride;
  diff(small.shape_.get<ndim>(), big.shape_.get<ndim>(), &rshape, &rstride);
  size_t N = small.shape_.Size();
  size_t M = rshape.Size();

  Shape<ndim> lhs_shape, lhs_stride;
  diff(small.shape_.get<ndim>(), lhs.shape_.get<ndim>(), &lhs_shape, &lhs_stride);

  Shape<ndim> rhs_shape, rhs_stride;
  diff(small.shape_.get<ndim>(), rhs.shape_.get<ndim>(), &rhs_shape, &rhs_stride);

  seq_reduce_compute<Reducer, ndim, DType, OP1, OP2>(
    N, M, req == kAddTo,
    big.dptr<DType>(), lhs.dptr<DType>(), rhs.dptr<DType>(), small.dptr<DType>(),
    big.shape_.get<ndim>(), small.shape_.get<ndim>(),
    rshape, rstride,
    lhs_shape, lhs_stride,
    rhs_shape, rhs_stride,
    lhs.shape_.get<ndim>(), rhs.shape_.get<ndim>());
}

#endif
}  // namespace broadcast
}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_TENSOR_BROADCAST_REDUCE_INL_H_
