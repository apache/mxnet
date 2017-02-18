/*!
 *  Copyright (c) 2015 by Contributors
 * \file broadcast_reduce_kernel.h
 * \brief Function defintion of elementwise unary operators
 */
#ifndef MXNET_OPERATOR_TENSOR_BROADCAST_REDUCE_KERNEL_OP_H_
#define MXNET_OPERATOR_TENSOR_BROADCAST_REDUCE_KERNEL_OP_H_

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
MSHADOW_XINLINE Shape<ndim> unravel(const int idx, const Shape<ndim>& shape) {
  Shape<ndim> ret;
  for (int i = ndim-1, j = idx; i >=0; --i) {
    ret[i] = j % shape[i];
    j /= shape[i];
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
MSHADOW_XINLINE int diff(const Shape<ndim>& small, const Shape<ndim>& big, Shape<ndim> &dims, Shape<ndim> &stride) {
  int mdim = 0;
  for (int i = 0; i < ndim; ++i) {
    mdim += small[i] != big[i];
    dims[i] = stride[i] = 1;
  }
  for (int i = ndim-1, j = mdim, s = 1; i >= 0; --i) {
    if (small[i] != big[i]) {
      --j;
      stride[j] = s;
      dims[j] = big[i];
    }
    s *= big[i];
  }
  return mdim;
}

template<int ndim>
MSHADOW_XINLINE int dot(const Shape<ndim>& coord, const Shape<ndim>& stride) {
  int ret = 0;
  for (int i = 0; i < ndim; ++i) 
    ret += coord[i] * stride[i];
  return ret;
}

template<typename DType>
MSHADOW_XINLINE void assign(DType& dst, const bool addto, const DType src) {
  if (addto) {
    dst += src;
  } else {
    dst = src;
  }
}

template<typename DType, typename OP>
MSHADOW_XINLINE void binary_broadcast_assign(const int idx, const bool addto,
                                             const DType* lhs, const DType* rhs, DType* out,
                                             const CShape& lshape, const CShape& rshape,
                                             const CShape& oshape) {
  const CShape coord = unravel(idx, oshape);
  const int j = ravel(coord, lshape);
  const int k = ravel(coord, rshape);
  assign(out[idx], addto, OP::Map(lhs[j], rhs[k]));
}

template<typename Reducer, typename DType, typename OP>
MSHADOW_XINLINE void seq_reduce_assign(const int idx, const int M, const bool addto,
                                       const DType *big, DType *small, const CShape& bshape, const CShape& sshape,
                                       const CShape& rshape, const CShape& rstride) {
  CShape coord = unravel(idx, sshape);
  int j = ravel(coord, bshape);
  DType val;
  Reducer::SetInitValue(val);
  for (int k = 0; k < M; ++k) {
    coord = unravel(k, rshape);
    Reducer::Reduce(val, OP::Map(big[j + dot(coord, rstride)]));
  }
  assign(small[idx], addto, val);
}

#ifdef __CUDACC__

template<typename DType>
using CTensor = Tensor<gpu, MAX_DIM, DType>;

template<typename DType, typename OP>
__global__ void binary_broadcast_kernel(const int N, const bool addto, const DType *lhs,
                                        const DType *rhs, DType *out, const CShape lshape,
                                        const CShape rshape, const CShape oshape) {
  for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < N; idx += blockDim.x * gridDim.x) {
    binary_broadcast_assign<DType, OP>(idx, addto, lhs, rhs, out, lshape, rshape, oshape);
  }
}

template<typename DType, typename OP>
void BinaryBroadcastComputeImpl(Stream<gpu> *s, const OpReqType req,
                                const CTensor<DType>& lhs, const CTensor<DType>& rhs, CTensor<DType> out) {
  using namespace mshadow::cuda;
  if (req == kNullOp) return;
  cudaStream_t stream = Stream<gpu>::GetStream(s);
  int N = out.shape_.Size();
  int ngrid = std::min(kMaxGridNum, (N + kBaseThreadNum - 1) / kBaseThreadNum);
  binary_broadcast_kernel<DType, OP><<<ngrid, kBaseThreadNum, 0, stream>>>(
    N, req == kAddTo, lhs.dptr_, rhs.dptr_, out.dptr_, lhs.shape_, rhs.shape_, out.shape_);
}

template<typename Reducer, typename DType, typename OP>
__global__ void seq_reduce_kernel(const int N, const int M, const bool addto,
                                  const DType *big, DType *small, const CShape bshape, const CShape sshape,
                                  const CShape rshape, const CShape rstride) {
  for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < N; idx += blockDim.x * gridDim.x) {
    seq_reduce_assign<Reducer, DType, OP>(idx, M, addto, big, small, bshape, sshape, rshape, rstride);
  }
}

template<typename Reducer, typename DType, typename OP, int x_bits>
__global__ void par_reduce_kernel(const int N, const int M, const bool addto,
                                  const DType *big, DType *small, const CShape bshape, const CShape sshape,
                                  const CShape rshape, const CShape rstride) {
  __shared__ DType buf[1<<x_bits];
  for (int idx = blockIdx.x; idx < N; idx += gridDim.x) {
    CShape coord = unravel(idx, sshape);
    int j = ravel(coord, bshape);

    Reducer::SetInitValue(buf[threadIdx.x]);
    for (int k = threadIdx.x; k < M; k += blockDim.x) {
      coord = unravel(k, rshape);
      Reducer::Reduce(buf[threadIdx.x], OP::Map(big[j + dot(coord, rstride)]));
    }
    __syncthreads();
    cuda::Reduce1D<Reducer, x_bits>(buf);
    if (threadIdx.x == 0) {
      assign(small[idx], addto, buf[0]);
    }
  }
}

template<typename Reducer, typename DType, typename OP>
void Reduce(Stream<gpu> *s, CTensor<DType> small, const OpReqType req,
            const CTensor<DType>& big) {
  using namespace mshadow::cuda;
  if (req == kNullOp) return;
  cudaStream_t stream = Stream<gpu>::GetStream(s);
  CShape rshape, rstride;
  int mdim = diff(small.shape_, big.shape_, rshape, rstride);
  int N = small.shape_.Size(), M = rshape.Size();
  if (N > 32*kBaseThreadNum) {
    int ngrid = std::min(kMaxGridNum, (N + kBaseThreadNum - 1) / kBaseThreadNum);
    seq_reduce_kernel<Reducer, DType, OP><<<ngrid, kBaseThreadNum, 0, stream>>>(
      N, M, req == kAddTo, big.dptr_, small.dptr_, big.shape_, small.shape_,
      rshape, rstride);
  } else {
    int ngrid = std::min(kMaxGridNum, N);
    par_reduce_kernel<Reducer, DType, OP, kBaseThreadBits><<<ngrid, kBaseThreadNum, 0, stream>>>(
      N, M, req == kAddTo, big.dptr_, small.dptr_, big.shape_, small.shape_,
      rshape, rstride);
  }
}

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
                                const CTensor<DType>& lhs, const CTensor<DType>& rhs, CTensor<DType> out) {
  if (req == kNullOp) return;
  int N = out.shape_.Size();
  binary_broadcast_compute<DType, OP>(N, req == kAddTo, lhs.dptr_, rhs.dptr_,
                           out.dptr_, lhs.shape_, rhs.shape_, out.shape_);
}

template<typename Reducer, typename DType, typename OP>
void seq_reduce_compute(const int N, const int M, const bool addto,
                        const DType *big, DType *small, const CShape bshape, const CShape sshape,
                        const CShape rshape, const CShape rstride) {
  for (int idx = 0; idx < N; ++idx) {
    seq_reduce_assign<Reducer, DType, OP>(idx, M, addto, big, small, bshape, sshape, rshape, rstride);
  }
}

template<typename Reducer, typename DType, typename OP>
void Reduce(Stream<cpu> *s, CTensor<DType> small, const OpReqType req,
            const CTensor<DType>& big) {
  if (req == kNullOp) return;
  CShape rshape, rstride;
  int mdim = diff(small.shape_, big.shape_, rshape, rstride);
  int N = small.shape_.Size(), M = rshape.Size();
  seq_reduce_compute<Reducer, DType, OP>(
    N, M, req == kAddTo, big.dptr_, small.dptr_, big.shape_, small.shape_,
    rshape, rstride);
}

#endif
}  // broadcast
}  // op
}  // mxnet

#endif  // MXNET_OPERATOR_TENSOR_BROADCAST_REDUCE_KERNEL_OP_H_