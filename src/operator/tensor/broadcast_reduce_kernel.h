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
#ifdef __CUDACC__
#include <cub/cub.cuh>
#endif

namespace mxnet {
namespace op {
namespace broadcast {
using namespace mshadow;

/*!
 * \brief shape of a tensor with product of shape
 * \tparam dimension dimension of tensor
 */
template<int dimension>
struct ShapeWP : Shape<dimension> {
  /*! \brief dimension of current shape */
  static const int kDimension = dimension;
  index_t prod_[kDimension];
  /*! \brief default constructor, do nothing */
  MSHADOW_XINLINE ShapeWP(void) {}
  /*! \brief constuctor */
  MSHADOW_XINLINE ShapeWP(const Shape<kDimension> &s) : Shape<dimension>(s) {}
  /*!
   * \return product shape in [dimstart,dimend)
   * \param dimstart start dimension
   * \param dimend end dimension
   */
  MSHADOW_XINLINE void SetProd() {
    index_t prodval = 1;
    #pragma unroll
    for (int i = kDimension - 1; i >= 0; --i)
    {
      this->prod_[i] = (this->shape_[i] > 1) ? prodval : 0;
      prodval *= this->shape_[i];
    }
  }
};

const int MAX_DIM = 5;
using CShape = Shape<MAX_DIM>;
using CShapeWP = ShapeWP<MAX_DIM>;

template<int ndim>
MSHADOW_XINLINE void unravel_ravel(const int idx, const Shape<ndim>& shape,
  const ShapeWP<ndim>& shapej, const ShapeWP<ndim>& shapek, int& j, int& k) {
  j = 0;
  k = 0;
  #pragma unroll
  for (int i = ndim-1, idx_t = idx; i >=0; --i) {
    const int tmp = idx_t / shape[i];
    const int coord = idx_t - tmp*shape[i];
    j += coord*shapej.prod_[i];
    k += coord*shapek.prod_[i];
    idx_t = tmp;
  }
}

template<int ndim>
MSHADOW_XINLINE Shape<ndim> unravel(const int idx, const ShapeWP<ndim>& shape) {
  Shape<ndim> ret;
  for (int i = ndim-1; i >=0; --i) {
    ret[i] = (idx / shape.prod_[i]) % shape[i];
  }
  return ret;
}

template<int ndim>
MSHADOW_XINLINE void unravel_ravel(const int idx, const Shape<ndim>& shape,
  const Shape<ndim>& shapej, const Shape<ndim>& shapek, int& j, int& k) {
  j = 0;
  k = 0;
  #pragma unroll
  for (int i = ndim-1, idx_t = idx; i >=0; --i) {
    const int tmp = idx_t / shape[i];
    const int coord = idx_t - tmp*shape[i];
    j = j*shapej[i] + (shapej[i] > 1)*coord;
    k = k*shapek[i] + (shapek[i] > 1)*coord;
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
                                             const CShapeWP& lshape, const CShapeWP& rshape,
                                             const CShape& oshape) {
  int j, k;
  unravel_ravel(idx, oshape, lshape, rshape, j, k);
  assign(out[idx], addto, OP::Map(lhs[j], rhs[k]));
}

template<typename DType, typename OP>
MSHADOW_XINLINE void binary_broadcast_assign(const int idx, const bool addto,
                                             const DType* lhs, const DType* rhs, DType* out,
                                             const CShape& lshape, const CShape& rshape,
                                             const CShape& oshape) {
  // const CShape coord = unravel(idx, oshape);
  // const int j = ravel(coord, lshape);
  // const int k = ravel(coord, rshape);
  int j, k;
  unravel_ravel(idx, oshape, lshape, rshape, j, k);
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

#include <cub/device/device_reduce.cuh>

// Parallel unravel + ravel for cases where idx is constant across warp
__forceinline__ __device__ int par_unravel_ravel(const int idx, const int shape, const int prod,
  const int stride) {
  // Calculate sum elements in parallel
  int ret = ((idx / prod) % shape)*stride;
  // Add sum elements together to get the final result
  #pragma unroll
  for (int i=warpSize/2;i>=1;i/=2) ret += __shfl_xor(ret, i);
  return ret;
}

template<typename Reducer, typename DType, typename OP>
__forceinline__ __device__ void seq_reduce_assign(const int idx, const int M, const bool addto,
                                       const DType *big, DType *small, const CShape& bshape,
                                       const CShape& sshape, const int rshape_shape,
                                       const int rshape_prod, const int rstride_shape) {
  CShape coord = unravel(idx, sshape);
  int j = ravel(coord, bshape);
  DType val;
  Reducer::SetInitValue(val);
  for (int k = 0; k < M; ++k) {
    int l = par_unravel_ravel(k, rshape_shape, rshape_prod, rstride_shape);
    Reducer::Reduce(val, OP::Map(big[j + l]));
  }
  assign(small[idx], addto, val);
}

template<typename DType>
using CTensor = Tensor<gpu, MAX_DIM, DType>;

template<typename DType, typename OP>
__global__ void binary_broadcast_kernel(const int N, const bool addto, const DType *lhs,
                                        const DType *rhs, DType *out, const CShapeWP lshape,
                                        const CShapeWP rshape, const CShape oshape) {
  for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < N; idx += blockDim.x * gridDim.x) {
    binary_broadcast_assign<DType, OP>(idx, addto, lhs, rhs, out, lshape, rshape, oshape);
  }
}

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
  int nthread = kMaxThreadsPerBlock / 2;
  int ngrid = std::min(kBaseGridNum, (N + nthread - 1) / nthread);
  CShapeWP lhs_shape(lhs.shape_);
  CShapeWP rhs_shape(rhs.shape_);
  lhs_shape.SetProd();
  rhs_shape.SetProd();
  binary_broadcast_kernel<DType, OP><<<ngrid, nthread, 0, stream>>>(
    N, req == kAddTo, lhs.dptr_, rhs.dptr_, out.dptr_, lhs_shape, rhs_shape, out.shape_);
  // binary_broadcast_kernel<DType, OP><<<ngrid, kBaseThreadNum, 0, stream>>>(
  //   N, req == kAddTo, lhs.dptr_, rhs.dptr_, out.dptr_, lhs.shape_, rhs.shape_, out.shape_);
}

template<typename Reducer, typename DType, typename OP, int ndim>
__global__ void seq_reduce_kernel(const int N, const int M, const bool addto,
                                  const DType *big, DType *small, const CShape bshape,
                                  const CShape sshape, const CShapeWP rshape,
                                  const CShape rstride) {
  // Must have at least as many warp lanes as we have dimensions
  static_assert(ndim <= 32,
    "number of dimensions (ndim) must be less than or equal to the warp width (warpSize)");
  // Store rshape.prod_[], rshape.shape_[], and rstride.shape_[] into registers
  const int warpLane = threadIdx.x & (warpSize - 1);
  int rshape_prod   = 1;
  int rshape_shape  = 1;
  int rstride_shape = 0;
  if (warpLane < ndim) {
    rshape_prod   = rshape.prod_[warpLane];
    rshape_shape  = rshape[warpLane];
    rstride_shape = rstride[warpLane];
  }
  for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < N; idx += blockDim.x * gridDim.x) {
    seq_reduce_assign<Reducer, DType, OP>(idx, M, addto, big, small, bshape, sshape, rshape_shape,
      rshape_prod, rstride_shape);
  }
}

template<typename Reducer, typename DType, typename OP>
__global__ void seq_reduce_kernel(const int N, const int M, const bool addto,
                                  const DType *big, DType *small, const CShape bshape, const CShape sshape,
                                  const CShape rshape, const CShape rstride) {
  for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < N; idx += blockDim.x * gridDim.x) {
    seq_reduce_assign<Reducer, DType, OP>(idx, M, addto, big, small, bshape, sshape, rshape, rstride);
  }
}

#if 1
template<typename Reducer, typename DType, typename OP, int x_bits>
__launch_bounds__(mshadow::cuda::kMaxThreadsPerBlock)
__global__ void par_reduce_kernel(const int N, const int M, const bool addto,
                                  const DType *big, DType *small, const CShape bshape,
                                  const CShape sshape, const CShape rshape, const CShape rstride,
                                  const int Mnext, int* offsets) {
  __shared__ DType buf[1<<x_bits];

  for (int m0 = blockIdx.y; m0 < Mnext; m0 += gridDim.y) {
    // This TB handles M range [Mstart, ...., Mend - 1]
    const int Mstart = M*m0/Mnext;
    const int Mend   = M*(m0 + 1)/Mnext;
    for (int idx = blockIdx.x; idx < N; idx += gridDim.x) {
      CShape coord = unravel(idx, sshape);
      int j = ravel(coord, bshape);

      Reducer::SetInitValue(buf[threadIdx.x]);
      for (int k = threadIdx.x + Mstart; k < Mend; k += blockDim.x) {
      // for (int k = threadIdx.x; k < M; k += blockDim.x) {
        coord = unravel(k, rshape);
        Reducer::Reduce(buf[threadIdx.x], OP::Map(big[j + dot(coord, rstride)]));
      }
      __syncthreads();
      cuda::Reduce1D<Reducer, x_bits>(buf);
      if (threadIdx.x == 0) {
        assign(small[idx + m0*N], addto, buf[0]);
        if (offsets != NULL) offsets[idx] = idx*Mnext;
      }
    }
  }

  // Cap offsets
  if (threadIdx.x == 0 && blockIdx.x == 0 
    && blockIdx.y == 0 && offsets != NULL) offsets[N] = N*Mnext;
}

#else
template<typename Reducer, typename DType, typename OP, int x_bits>
__global__ void par_reduce_kernel(const int N, const int M, const bool addto,
                                  const DType *big, DType *small, const CShape bshape,
                                  const CShape sshape, const CShape rshape, const CShape rstride) {
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
#endif

template<typename Reducer, typename DType, typename OP>
__launch_bounds__(mshadow::cuda::kMaxThreadsPerBlock)
__global__ void reduce_kernel(const int N, const int M, const bool addto,
                              const DType* __restrict__ big, DType *small, const CShape bshape,
                              const CShape sshape, const CShape rshape, const CShape rstride,
                              const int Mnext, int* offsets) {
  // Size of shared memory is blockDim.x*blockDim.y*sizeof(DType)
  extern __shared__ char shTileChar[];
  DType* shTile = (DType*)(shTileChar);
  const int it0 = threadIdx.x + threadIdx.y*blockDim.x;
  for (int m0 = blockIdx.y; m0 < Mnext; m0 += gridDim.y) {
    // This TB handles M range [Mstart, ...., Mend - 1]
    const int Mstart = M*m0/Mnext;
    const int Mend   = M*(m0 + 1)/Mnext;
    for (int idx0 = blockIdx.x*blockDim.x; idx0 < N; idx0 += blockDim.x*gridDim.x) {
      int idx = idx0 + threadIdx.x;
      CShape coord = unravel(idx, sshape);
      int j = ravel(coord, bshape);

      DType val;
      Reducer::SetInitValue(val);
      if (idx < N) {
        for (int k = threadIdx.y + Mstart; k < Mend; k += blockDim.y) {
          coord = unravel(k, rshape);
          Reducer::Reduce(val, OP::Map(big[j + dot(coord, rstride)]));
        }
      }

      shTileChar[it0] = val;
      __syncthreads();
      for (int t=1;t < blockDim.y;t <<= 1) {
        DType tmp;
        Reducer::SetInitValue(tmp);
        if (threadIdx.y + t < blockDim.y) tmp = shTile[it0 + t*blockDim.x];
        __syncthreads();
        Reducer::Reduce(shTile[it0], tmp);
        __syncthreads();
      }

      if (idx < N && threadIdx.y == 0) {
        assign(small[idx + m0*N], addto, shTile[threadIdx.x]);
        if (offsets != NULL) offsets[idx] = idx*Mnext;
      }

    }
  }

  // Cap offsets
  if (threadIdx.x == 0 && threadIdx.y == 0 &&
    blockIdx.x == 0 && blockIdx.y == 0 && offsets != NULL) offsets[N] = N*Mnext;
}

template<typename DType>
__global__ void addTo_kernel(const int N, const DType* small_in, DType* small_out) {
  for (int idx = threadIdx.x + blockIdx.x*blockDim.x; idx < N; idx += blockDim.x*gridDim.x) {
    assign(small_out[idx], true, small_in[idx]);
  }
}

template<typename Reducer, typename DType, typename OP>
__global__ void reduce_kernel_M1(const int N, const bool addto,
                                const DType* __restrict__ big, DType *small, const CShape bshape,
                                const CShape sshape) {
  for (int idx = threadIdx.x + blockIdx.x*blockDim.x; idx < N; idx += blockDim.x*gridDim.x) {
    CShape coord = unravel(idx, sshape);
    int j = ravel(coord, bshape);
    assign(small[idx], addto, OP::Map(big[j]));
  }
}

#if 1

// Convert Reducer::Reduce to CUB operation
template<typename Reducer>
struct cubOP {
  template <typename DType>
  CUB_RUNTIME_FUNCTION __forceinline__ __device__
  DType operator()(const DType &a, const DType &b) const {
    DType c = a;
    Reducer::Reduce(c, b);
    return c;
  }
};

template<typename Reducer, typename DType, typename OP>
size_t ReduceImpl(Stream<gpu> *s, CTensor<DType> small, const OpReqType req,
                  const CTensor<DType>& big, mshadow::Tensor<gpu, 1, char>& workspace,
                  const bool getWorkspaceSize) {
  using namespace mshadow::cuda;
  size_t workspace_pos = 0;
  if (req == kNullOp) return workspace_pos;
  cudaStream_t stream = Stream<gpu>::GetStream(s);
  CShape rshape, rstride;
  int mdim = diff(small.shape_, big.shape_, rshape, rstride);
  int N = small.shape_.Size(), M = rshape.Size();

  LOG(INFO) << "N " << N << " M " << M;

  const int warpSize = 32;

  if (M == 1) {
    if (!getWorkspaceSize) {
      dim3 blockDim;
      dim3 gridDim;
      blockDim.x = kMaxThreadsPerBlock;
      gridDim.x = std::min((unsigned int)kBaseGridNum, (N + blockDim.x - 1)/blockDim.x);
      reduce_kernel_M1<Reducer, DType, OP><<< gridDim, blockDim, 0, stream>>>(
        N, req == kAddTo, big.dptr_, small.dptr_, big.shape_, small.shape_);
    }
  } else {

    // The goal is to saturate the GPU with enough work. This happens when there are enough
    // thread blocks to keep all SMs working.

    // First try saturating the GPU in a single pass, where Mnext = 1

    dim3 blockDim;
    dim3 gridDim;

    const int kMaxThreadBits = 10;
    const int par_reduce_lim = 1024;

    int Mnext = 1;
    // int nthreadPerM;
    int minGridSize;
    if (N < par_reduce_lim) {
      // nthreadPerM = 1 << kMaxThreadBits;
      blockDim.x = 1 << kMaxThreadBits;
      gridDim.x = std::min(kMaxGridNum, N);
      gridDim.y = std::min(kMaxGridNum, Mnext);
      int blockSize;
      cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize,
        par_reduce_kernel<Reducer, DType, OP, kMaxThreadBits>, 0, kMaxThreadsPerBlock);
    } else {
      // nthreadPerM = std::min(M, warpSize);
      // blockDim.y = std::min(M, 1);
      if (M >= N) {
        blockDim.y = std::min(M, warpSize);
      } else {
        blockDim.y = 1;
      }
      blockDim.x = (kMaxThreadsPerBlock/(blockDim.y*warpSize))*warpSize;
      // gridDim.x = std::min((unsigned int)kBaseGridNum, (N + blockDim.x - 1)/blockDim.x);
      gridDim.x = std::min((unsigned int)512, (N + blockDim.x - 1)/blockDim.x);
      gridDim.y = std::min(kMaxGridNum, Mnext);
      int shMemSize = blockDim.x*blockDim.y*sizeof(DType);
      int blockSize;
      cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize,
        reduce_kernel<Reducer, DType, OP>, shMemSize, kMaxThreadsPerBlock);
    }

    // minGridSize is the minimum number of blocks we need to saturate the GPU
    // 
    LOG(INFO) << "minGridSize " << minGridSize << " gridDim " << gridDim.x*gridDim.y;
    Mnext = (minGridSize*8 + gridDim.x*gridDim.y - 1) / (gridDim.x*gridDim.y);
    LOG(INFO) << "Mnext " << Mnext;

    // Maximum number of times TB loops
    // const int maxLoopPerTB = 64;
    // Max size of M-block each TB can handle
    // int maxMblock = nthreadPerM*maxLoopPerTB;
    // Size of M after first pass
    // Mnext = (M + maxMblock - 1) / maxMblock;
    DType* out1_dptr = small.dptr_;
    int* offsets_dptr = NULL;
    bool addto = (req == kAddTo);
    if (Mnext > 1) {
      // out1_dptr[] is N*Mnext*sizeof(DType) bytes
      out1_dptr = reinterpret_cast<DType*>(workspace.dptr_ + workspace_pos);
      workspace_pos += N*Mnext*sizeof(DType);
      // offsets_dptr[] is (N + 1)*sizeof(int) bytes
      offsets_dptr = reinterpret_cast<int*>(workspace.dptr_ + workspace_pos);
      workspace_pos += (N + 1)*sizeof(int);
      addto = false;
      // Check that the workspace is contigiuous
      if (!getWorkspaceSize) CHECK_EQ(workspace.CheckContiguous(), true);
      // Check that we have enough storage
      if (!getWorkspaceSize) CHECK_GE(workspace.size(0), workspace_pos);
      // Fix grid y-dimension
      if (N < par_reduce_lim) {
        gridDim.y = std::min(kMaxGridNum, Mnext);
      } else {
        gridDim.y = std::min(kMaxGridNum, Mnext);
      }
    }

    if (N < par_reduce_lim) {
      if (!getWorkspaceSize) {
        par_reduce_kernel<Reducer, DType, OP, kMaxThreadBits>
        <<< gridDim, blockDim, 0, stream>>>(
          N, M, addto, big.dptr_, out1_dptr, big.shape_, small.shape_,
          rshape, rstride, Mnext, offsets_dptr);
      }
    } else {
      if (!getWorkspaceSize) {
        int shMemSize = blockDim.x*blockDim.y*sizeof(DType);
        reduce_kernel<Reducer, DType, OP>
        <<< gridDim, blockDim, shMemSize, stream>>>(
          N, M, addto, big.dptr_, out1_dptr, big.shape_, small.shape_,
          rshape, rstride, Mnext, offsets_dptr);
      }
    }

    if (Mnext > 1) {
      DType initValue;
      Reducer::SetInitValue(initValue);
      cubOP<Reducer> reduceOp;

      DType* out2_dptr = small.dptr_;
      DType* temp_storage_dptr = NULL;
      if (req == kAddTo) {
        // out2_dptr[] is N*sizeof(DType) bytes
        out2_dptr = reinterpret_cast<DType*>(workspace.dptr_ + workspace_pos);
        workspace_pos += N*sizeof(DType);
        // Check that we have enough storage
        if (!getWorkspaceSize) CHECK_GE(workspace.size(0), workspace_pos);
      }

      // Get size of temporary storage temp_storage_bytes
      size_t temp_storage_bytes = 0;
      cub::DeviceSegmentedReduce::Reduce(temp_storage_dptr, temp_storage_bytes, out1_dptr, out2_dptr,
        N, offsets_dptr, offsets_dptr + 1, reduceOp, initValue, stream);

      // temp_storage_dptr[] is temp_storage_bytes bytes
      temp_storage_dptr = reinterpret_cast<DType*>(workspace.dptr_ + workspace_pos);
      workspace_pos += temp_storage_bytes;
      // Check that we have enough storage
      if (!getWorkspaceSize) CHECK_GE(workspace.size(0), workspace_pos);

      // Reduce
      if (!getWorkspaceSize) {
        cub::DeviceSegmentedReduce::Reduce(temp_storage_dptr, temp_storage_bytes, out1_dptr, out2_dptr,
          N, offsets_dptr, offsets_dptr + 1, reduceOp, initValue, stream);
      }

      if (req == kAddTo) {
        if (!getWorkspaceSize) {
          // Add out2_dptr[0 ... N - 1] to small.dptr[0 ... N - 1]
          int blockSize = kMaxThreadsPerBlock;
          int gridSize = std::min((int)kMaxGridNum, (N + blockSize - 1)/blockSize );
          addTo_kernel<<< gridSize, blockSize, 0, stream >>>(N, out2_dptr, small.dptr_);
        }
      }

    }

  }

  return workspace_pos;
}

// template<typename Reducer, typename DType, typename OP>
// size_t ReduceImpl(Stream<gpu> *s, CTensor<DType> small, const OpReqType req,
//                   const CTensor<DType>& big, mshadow::Tensor<gpu, 1, char>& workspace,
//                   const bool getWorkspaceSize);

template<typename Reducer, typename DType, typename OP>
void Reduce(Stream<gpu> *s, CTensor<DType> small, const OpReqType req,
            const CTensor<DType>& big, mshadow::Tensor<gpu, 1, char>& workspace) {
  ReduceImpl<Reducer, DType, OP>(s, small, req, big, workspace, false);
}

template<typename Reducer, typename DType, typename OP>
size_t ReduceWorkspaceSize(Stream<gpu> *s, CTensor<DType> small, const OpReqType req,
                           const CTensor<DType>& big) {
  mshadow::Tensor<gpu, 1, char> dummy_workspace;
  return ReduceImpl<Reducer, DType, OP>(s, small, req, big, dummy_workspace, true);
}

#else
template<typename Reducer, typename DType, typename OP>
void Reduce(Stream<gpu> *s, CTensor<DType> small, const OpReqType req,
            const CTensor<DType>& big, mshadow::Tensor<gpu, 1, char>& workspace) {
  using namespace mshadow::cuda;
  if (req == kNullOp) return;
  cudaStream_t stream = Stream<gpu>::GetStream(s);
  CShape rshape, rstride;
  int mdim = diff(small.shape_, big.shape_, rshape, rstride);
  int N = small.shape_.Size(), M = rshape.Size();
  LOG(INFO) << "N " << N << " M " << M;
  if (N > 32*kBaseThreadNum) {
    // int ngrid = std::min(kMaxGridNum, (N + kBaseThreadNum - 1) / kBaseThreadNum);
    // CShapeWP rshapeWP(rshape);
    // rshapeWP.SetProd();
    // seq_reduce_kernel<Reducer, DType, OP><<<ngrid, kBaseThreadNum, 0, stream>>>(
    //   N, M, req == kAddTo, big.dptr_, small.dptr_, big.shape_, small.shape_,
    //   rshapeWP, rstride);
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

template<typename Reducer, typename DType, typename OP>
size_t ReduceWorkspaceSize(Stream<gpu> *s, CTensor<DType> small, const OpReqType req,
                           const CTensor<DType>& big) {
  return 0;
}
#endif

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
            const CTensor<DType>& big, mshadow::Tensor<cpu, 1, char>& workspace) {
  if (req == kNullOp) return;
  CShape rshape, rstride;
  int mdim = diff(small.shape_, big.shape_, rshape, rstride);
  int N = small.shape_.Size(), M = rshape.Size();
  seq_reduce_compute<Reducer, DType, OP>(
    N, M, req == kAddTo, big.dptr_, small.dptr_, big.shape_, small.shape_,
    rshape, rstride);
}

template<typename Reducer, typename DType, typename OP>
size_t ReduceWorkspaceSize(Stream<cpu> *s, CTensor<DType> small, const OpReqType req,
                           const CTensor<DType>& big) {
  return 0;
}

#endif
}  // broadcast
}  // op
}  // mxnet

#endif  // MXNET_OPERATOR_TENSOR_BROADCAST_REDUCE_KERNEL_OP_H_