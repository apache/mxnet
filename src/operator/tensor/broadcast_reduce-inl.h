/*!
 *  Copyright (c) 2015-2017 by Contributors
 * \file broadcast_reduce_kernel.h
 * \brief Function defintion of elementwise unary operators
 */
#ifndef MXNET_OPERATOR_TENSOR_BROADCAST_REDUCE_INL_H_
#define MXNET_OPERATOR_TENSOR_BROADCAST_REDUCE_INL_H_

#include <mxnet/operator_util.h>
#include <algorithm>
#include <vector>
#include <string>
#include <utility>
#include "../mshadow_op.h"
#include "../elemwise_op_common.h"
#include "./elemwise_binary_op.h"
#include "../operator_common.h"

namespace mxnet {
namespace op {
namespace broadcast {
using namespace mshadow;

const int MAX_DIM = 5;
using CShape = Shape<MAX_DIM>;

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
MSHADOW_XINLINE void unravel_dot(const int idx, const Shape<ndim>& shape,
  const Shape<ndim>& stridej, const Shape<ndim>& stridek, int* j, int* k) {
  *j = 0;
  *k = 0;
  #pragma unroll
  for (int i = ndim-1, idx_t = idx; i >=0; --i) {
    const int tmp = idx_t / shape[i];
    const int coord = idx_t - tmp*shape[i];
    *j += coord*stridej[i];
    *k += coord*stridek[i];
    idx_t = tmp;
  }
}

template<int ndim>
MSHADOW_XINLINE Shape<ndim> unravel(const int idx, const Shape<ndim>& shape) {
  Shape<ndim> ret;
  for (int i = ndim-1, j = idx; i >=0; --i) {
    int tmp = j / shape[i];
    ret[i] = j - tmp*shape[i];
    j = tmp;
  }
  return ret;
}

template<int ndim>
MSHADOW_XINLINE int ravel(const Shape<ndim>& coord, const Shape<ndim>& shape) {
  int ret = 0;
  for (int i = 0; i < ndim; ++i) {
    ret = ret * shape[i] + (shape[i] > 1) * coord[i];
  }
  return ret;
}

template<int ndim>
MSHADOW_XINLINE int diff(const Shape<ndim>& small, const Shape<ndim>& big, Shape<ndim>* dims,
  Shape<ndim>* stride) {
  int mdim = 0;
  for (int i = 0; i < ndim; ++i) {
    mdim += small[i] != big[i];
    (*dims)[i] = (*stride)[i] = 1;
  }
  for (int i = ndim-1, j = mdim, s = 1; i >= 0; --i) {
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
MSHADOW_XINLINE int unravel_dot(const int idx, const Shape<ndim>& shape,
  const Shape<ndim>& stride) {
  int ret = 0;
  for (int i = ndim-1, j = idx; i >=0; --i) {
    int tmp = j / shape[i];
    ret += (j - tmp*shape[i])*stride[i];
    j = tmp;
  }
  return ret;
}

template<int ndim>
MSHADOW_XINLINE int dot(const Shape<ndim>& coord, const Shape<ndim>& stride) {
  int ret = 0;
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

template<typename DType, typename OP>
MSHADOW_XINLINE void binary_broadcast_assign(const int idx, const bool addto,
                                             const DType* __restrict lhs,
                                             const DType* __restrict rhs, DType* out,
                                             const CShape& lshape, const CShape& rshape,
                                             const CShape& oshape) {
  const CShape coord = unravel(idx, oshape);
  const int j = ravel(coord, lshape);
  const int k = ravel(coord, rshape);
  assign(&out[idx], addto, OP::Map(lhs[j], rhs[k]));
}

template<typename Reducer, typename DType, typename OP>
MSHADOW_XINLINE void seq_reduce_assign(const int idx, const int M, const bool addto,
                                       const DType* __restrict big, DType *small,
                                       const CShape& bshape, const CShape& sshape,
                                       const CShape& rshape, const CShape& rstride) {
  CShape coord = unravel(idx, sshape);
  int j = ravel(coord, bshape);
  DType val;
  Reducer::SetInitValue(val);
  for (int k = 0; k < M; ++k) {
    coord = unravel(k, rshape);
    Reducer::Reduce(val, OP::Map(big[j + dot(coord, rstride)]));
  }
  assign(&small[idx], addto, val);
}

#ifdef __CUDACC__
#include "broadcast_reduce-inl.cuh"

#else

template<typename DType>
using CTensor = Tensor<cpu, MAX_DIM, DType>;

template<typename DType, typename OP>
void binary_broadcast_compute(const int N, const bool addto, const DType *lhs,
                              const DType *rhs, DType *out, const CShape lshape,
                              const CShape rshape, const CShape oshape) {
  for (int idx = 0; idx < N; ++idx) {
    binary_broadcast_assign<DType, OP>(idx, addto, lhs, rhs, out, lshape, rshape, oshape);
  }
}

template<typename DType, typename OP>
void BinaryBroadcastComputeImpl(Stream<cpu> *s, const OpReqType req,
                                const TBlob& lhs, const TBlob& rhs, const TBlob& out) {
  if (req == kNullOp) return;
  int N = out.shape_.Size();
  binary_broadcast_compute<DType, OP>(N, req == kAddTo, lhs.dptr<DType>(), rhs.dptr<DType>(),
                           out.dptr<DType>(), lhs.shape_.get<MAX_DIM>(), rhs.shape_.get<MAX_DIM>(),
                           out.shape_.get<MAX_DIM>());
}

template<typename Reducer, typename DType, typename OP>
void seq_reduce_compute(const int N, const int M, const bool addto,
                        const DType *big, DType *small, const CShape bshape, const CShape sshape,
                        const CShape rshape, const CShape rstride) {
  for (int idx = 0; idx < N; ++idx) {
    seq_reduce_assign<Reducer, DType, OP>(idx, M, addto, big, small, bshape, sshape, rshape,
      rstride);
  }
}

template<typename Reducer, typename DType, typename OP>
void Reduce(Stream<cpu> *s, const TBlob& small, const OpReqType req,
            const Tensor<cpu, 1, char>& workspace, const TBlob& big) {
  if (req == kNullOp) return;
  CShape rshape, rstride;
  int mdim = diff(small.shape_.get<MAX_DIM>(), big.shape_.get<MAX_DIM>(), &rshape, &rstride);
  int N = small.shape_.Size(), M = rshape.Size();
  seq_reduce_compute<Reducer, DType, OP>(
    N, M, req == kAddTo, big.dptr<DType>(), small.dptr<DType>(), big.shape_.get<MAX_DIM>(),
    small.shape_.get<MAX_DIM>(), rshape, rstride);
}

template<typename DType>
size_t ReduceWorkspaceSize(Stream<cpu> *s, const TBlob& small, const OpReqType req,
                           const TBlob& big) {
  return 0;
}

template<typename DType>
size_t ReduceWorkspaceSize(Stream<cpu> *s, const TBlob& small, const OpReqType req,
                           const TBlob& big, const TBlob& lhs, const TBlob& rhs) {
  return 0;
}

template<typename Reducer, typename DType, typename OP1, typename OP2>
MSHADOW_XINLINE void seq_reduce_assign(const int idx, const int M, const bool addto,
                                       const DType* __restrict big, const DType* __restrict lhs,
                                       const DType* __restrict rhs, DType *small,
                                       const CShape& big_shape, const CShape& lhs_shape0,
                                       const CShape& rhs_shape0, const CShape& small_shape,
                                       const CShape& rshape, const CShape& lhs_shape,
                                       const CShape& rhs_shape, const CShape& rstride,
                                       const CShape& lhs_stride, const CShape& rhs_stride) {
  CShape coord = unravel(idx, small_shape);
  const int idx_big0 = ravel(coord, big_shape);
  const int idx_lhs0 = ravel(coord, lhs_shape0);
  const int idx_rhs0 = ravel(coord, rhs_shape0);
  DType val;
  Reducer::SetInitValue(val);
  for (int k = 0; k < M; ++k) {
    CShape coord_big = unravel(k, rshape);
    int idx_big = idx_big0 + dot(coord_big, rstride);

    CShape coord_lhs = unravel(k, lhs_shape);
    int idx_lhs = idx_lhs0 + dot(coord_lhs, lhs_stride);

    CShape coord_rhs = unravel(k, rhs_shape);
    int idx_rhs = idx_rhs0 + dot(coord_rhs, rhs_stride);

    Reducer::Reduce(val, OP1::Map(big[idx_big], OP2::Map(lhs[idx_lhs], rhs[idx_rhs]) ) );
  }
  assign(&small[idx], addto, val);
}

template<typename Reducer, typename DType, typename OP1, typename OP2>
void seq_reduce_compute(const int N, const int M, const bool addto,
                        const DType *big, const DType *lhs, const DType *rhs, DType *small,
                        const CShape big_shape, const CShape small_shape,
                        const CShape rshape, const CShape rstride,
                        const CShape lhs_shape, const CShape lhs_stride,
                        const CShape rhs_shape, const CShape rhs_stride,
                        const CShape& lhs_shape0, const CShape& rhs_shape0) {
  for (int idx = 0; idx < N; ++idx) {
    seq_reduce_assign<Reducer, DType, OP1, OP2>(idx, M, addto, big, lhs, rhs, small, big_shape,
      lhs_shape0, rhs_shape0, small_shape, rshape, lhs_shape, rhs_shape, rstride,
      lhs_stride, rhs_stride);
  }
}

template<typename Reducer, typename DType, typename OP1, typename OP2>
void Reduce(Stream<cpu> *s, const TBlob& small, const OpReqType req,
            const Tensor<cpu, 1, char>& workspace, const TBlob& big, const TBlob& lhs,
            const TBlob& rhs) {
  if (req == kNullOp) return;
  CShape rshape, rstride;
  diff(small.shape_.get<MAX_DIM>(), big.shape_.get<MAX_DIM>(), &rshape, &rstride);
  int N = small.shape_.Size();
  int M = rshape.Size();

  CShape lhs_shape, lhs_stride;
  diff(small.shape_.get<MAX_DIM>(), lhs.shape_.get<MAX_DIM>(), &lhs_shape, &lhs_stride);

  CShape rhs_shape, rhs_stride;
  diff(small.shape_.get<MAX_DIM>(), rhs.shape_.get<MAX_DIM>(), &rhs_shape, &rhs_stride);

  seq_reduce_compute<Reducer, DType, OP1, OP2>(
    N, M, req == kAddTo,
    big.dptr<DType>(), lhs.dptr<DType>(), rhs.dptr<DType>(), small.dptr<DType>(),
    big.shape_.get<MAX_DIM>(), small.shape_.get<MAX_DIM>(),
    rshape, rstride,
    lhs_shape, lhs_stride,
    rhs_shape, rhs_stride,
    lhs.shape_.get<MAX_DIM>(), rhs.shape_.get<MAX_DIM>());
}

#endif
}  // namespace broadcast
}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_TENSOR_BROADCAST_REDUCE_INL_H_
