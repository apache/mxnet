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
  #pragma unroll
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
  #pragma unroll
  for (int i = 0; i < ndim; ++i) {
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
  #pragma unroll
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
  #pragma unroll
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
MSHADOW_XINLINE void binary_broadcast_assign(const int idx, const bool addto,
                                             const DType* __restrict lhs,
                                             const DType* __restrict rhs, DType* out,
                                             const Shape<ndim>& lshape, const Shape<ndim>& rshape,
                                             const Shape<ndim>& oshape) {
  const Shape<ndim> coord = unravel(idx, oshape);
  const int j = ravel(coord, lshape);
  const int k = ravel(coord, rshape);
  assign(&out[idx], addto, OP::Map(lhs[j], rhs[k]));
}

template<typename Reducer, int ndim, typename DType, typename OP>
MSHADOW_XINLINE void seq_reduce_assign(const int idx, const int M, const bool addto,
                                       const DType* __restrict big, DType *small,
                                       const Shape<ndim>& bshape, const Shape<ndim>& sshape,
                                       const Shape<ndim>& rshape, const Shape<ndim>& rstride) {
  Shape<ndim> coord = unravel(idx, sshape);
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

template<int ndim, typename DType, typename OP>
void binary_broadcast_compute(const int N, const bool addto, const DType *lhs,
                              const DType *rhs, DType *out, const Shape<ndim> lshape,
                              const Shape<ndim> rshape, const Shape<ndim> oshape) {
  for (int idx = 0; idx < N; ++idx) {
    binary_broadcast_assign<ndim, DType, OP>(idx, addto, lhs, rhs, out, lshape, rshape, oshape);
  }
}

template<int ndim, typename DType, typename OP>
void BinaryBroadcastComputeImpl(Stream<cpu> *s, const OpReqType req,
                                const TBlob& lhs, const TBlob& rhs, const TBlob& out) {
  if (req == kNullOp) return;
  int N = out.shape_.Size();
  binary_broadcast_compute<ndim, DType, OP>(N, req == kAddTo, lhs.dptr<DType>(), rhs.dptr<DType>(),
                           out.dptr<DType>(), lhs.shape_.get<ndim>(), rhs.shape_.get<ndim>(),
                           out.shape_.get<ndim>());
}

template<typename Reducer, int ndim, typename DType, typename OP>
void seq_reduce_compute(const int N, const int M, const bool addto,
                        const DType *big, DType *small, const Shape<ndim> bshape,
                        const Shape<ndim> sshape, const Shape<ndim> rshape,
                        const Shape<ndim> rstride) {
  for (int idx = 0; idx < N; ++idx) {
    seq_reduce_assign<Reducer, ndim, DType, OP>(idx, M, addto, big, small, bshape, sshape, rshape,
      rstride);
  }
}

template<typename Reducer, int ndim, typename DType, typename OP>
void Reduce(Stream<cpu> *s, const TBlob& small, const OpReqType req,
            const Tensor<cpu, 1, char>& workspace, const TBlob& big) {
  if (req == kNullOp) return;
  Shape<ndim> rshape, rstride;
  int mdim = diff(small.shape_.get<ndim>(), big.shape_.get<ndim>(), &rshape, &rstride);
  int N = small.shape_.Size(), M = rshape.Size();
  seq_reduce_compute<Reducer, ndim, DType, OP>(
    N, M, req == kAddTo, big.dptr<DType>(), small.dptr<DType>(), big.shape_.get<ndim>(),
    small.shape_.get<ndim>(), rshape, rstride);
}

template<int ndim, typename DType>
size_t ReduceWorkspaceSize(Stream<cpu> *s, const TBlob& small, const OpReqType req,
                           const TBlob& big) {
  return 0;
}

template<int ndim, typename DType>
size_t ReduceWorkspaceSize(Stream<cpu> *s, const TBlob& small, const OpReqType req,
                           const TBlob& big, const TBlob& lhs, const TBlob& rhs) {
  return 0;
}

template<typename Reducer, int ndim, typename DType, typename OP1, typename OP2>
MSHADOW_XINLINE void seq_reduce_assign(const int idx, const int M, const bool addto,
                                       const DType* __restrict big, const DType* __restrict lhs,
                                       const DType* __restrict rhs, DType *small,
                                       const Shape<ndim>& big_shape, const Shape<ndim>& lhs_shape0,
                                       const Shape<ndim>& rhs_shape0,
                                       const Shape<ndim>& small_shape, const Shape<ndim>& rshape,
                                       const Shape<ndim>& lhs_shape, const Shape<ndim>& rhs_shape,
                                       const Shape<ndim>& rstride, const Shape<ndim>& lhs_stride,
                                       const Shape<ndim>& rhs_stride) {
  Shape<ndim> coord = unravel(idx, small_shape);
  const int idx_big0 = ravel(coord, big_shape);
  const int idx_lhs0 = ravel(coord, lhs_shape0);
  const int idx_rhs0 = ravel(coord, rhs_shape0);
  DType val;
  Reducer::SetInitValue(val);
  for (int k = 0; k < M; ++k) {
    Shape<ndim> coord_big = unravel(k, rshape);
    int idx_big = idx_big0 + dot(coord_big, rstride);

    Shape<ndim> coord_lhs = unravel(k, lhs_shape);
    int idx_lhs = idx_lhs0 + dot(coord_lhs, lhs_stride);

    Shape<ndim> coord_rhs = unravel(k, rhs_shape);
    int idx_rhs = idx_rhs0 + dot(coord_rhs, rhs_stride);

    Reducer::Reduce(val, OP1::Map(big[idx_big], OP2::Map(lhs[idx_lhs], rhs[idx_rhs]) ) );
  }
  assign(&small[idx], addto, val);
}

template<typename Reducer, int ndim, typename DType, typename OP1, typename OP2>
void seq_reduce_compute(const int N, const int M, const bool addto,
                        const DType *big, const DType *lhs, const DType *rhs, DType *small,
                        const Shape<ndim> big_shape, const Shape<ndim> small_shape,
                        const Shape<ndim> rshape, const Shape<ndim> rstride,
                        const Shape<ndim> lhs_shape, const Shape<ndim> lhs_stride,
                        const Shape<ndim> rhs_shape, const Shape<ndim> rhs_stride,
                        const Shape<ndim>& lhs_shape0, const Shape<ndim>& rhs_shape0) {
  for (int idx = 0; idx < N; ++idx) {
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
  int N = small.shape_.Size();
  int M = rshape.Size();

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
