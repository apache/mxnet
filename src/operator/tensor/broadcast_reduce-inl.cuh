/*!
 * Copyright (c) 2015-2017 by Contributors
 * \file broadcast_reduce-inl.cuh
 * \brief CUDA implementations for binary broadcast and reduce
 * \author Antti-Pekka Hynninen
*/
#ifndef MXNET_OPERATOR_TENSOR_BROADCAST_REDUCE_INL_CUH_
#define MXNET_OPERATOR_TENSOR_BROADCAST_REDUCE_INL_CUH_

template<typename DType>
using CTensor = Tensor<gpu, MAX_DIM, DType>;

using namespace mshadow::cuda;

template<typename DType, typename OP, int unroll>
__launch_bounds__(kMaxThreadsPerBlock)
__global__ void binary_broadcast_kernel(const int N, const bool addto,
                                        const DType* __restrict lhs,
                                        const DType* __restrict rhs, DType *out,
                                        const CShape lstride, const CShape rstride,
                                        const CShape oshape) {
  for (int idx = blockIdx.x * blockDim.x * unroll + threadIdx.x; idx < N;
    idx += blockDim.x * gridDim.x * unroll)
  {
    int j[unroll];
    int k[unroll];
    DType val[unroll];
    #pragma unroll
    for (int i=0;i < unroll;i++) {
      unravel_dot(idx + i*blockDim.x, oshape, lstride, rstride, &j[i], &k[i]);
      val[i] = OP::Map(lhs[j[i]], rhs[k[i]]);
    }
    #pragma unroll
    for (int i=0;i < unroll;i++) {
      if (idx + i*blockDim.x < N) assign(&out[idx + i*blockDim.x], addto, val[i]);
    }

  }
}

template<typename DType, typename OP>
void BinaryBroadcastComputeImpl(Stream<gpu> *s, const OpReqType req,
                                const TBlob& lhs, const TBlob& rhs, const TBlob& out) {
  if (req == kNullOp) return;
  cudaStream_t stream = Stream<gpu>::GetStream(s);
  int N = out.shape_.Size();
  const int warpSize = 32;
  const int unroll = 2;
  int nthread = std::min(kMaxThreadsPerBlock, ((N + warpSize - 1)/warpSize)*warpSize );
  int ngrid = std::min(kBaseGridNum, (N + nthread*unroll - 1) / (nthread*unroll));
  CShape lstride = calc_stride(lhs.shape_.get<MAX_DIM>());
  CShape rstride = calc_stride(rhs.shape_.get<MAX_DIM>());
  binary_broadcast_kernel<DType, OP, unroll><<<ngrid, nthread, 0, stream>>>(
    N, req == kAddTo, lhs.dptr<DType>(), rhs.dptr<DType>(), out.dptr<DType>(), lstride, rstride,
    out.shape_.get<MAX_DIM>());
}

template<typename Reducer, typename DType, typename OP, int x_bits, int unroll>
__launch_bounds__(kMaxThreadsPerBlock)
__global__ void par_reduce_kernel(const int N, const int M, const bool addto,
                                  const DType* __restrict big, DType *small, const CShape bshape,
                                  const CShape sshape, const CShape rshape, const CShape rstride,
                                  const int Mnext) {
  __shared__ DType buf[1<<x_bits];

  for (int m0 = blockIdx.y; m0 < Mnext; m0 += gridDim.y) {
    // This TB handles M range [Mstart, ...., Mend - 1]
    const int Mstart = (int)((uint64_t)M*(uint64_t)m0/(uint64_t)Mnext);
    const int Mend   = (int)((uint64_t)M*(uint64_t)(m0 + 1)/(uint64_t)Mnext);
    for (int idx = blockIdx.x; idx < N; idx += gridDim.x) {
      CShape coord = unravel(idx, sshape);
      int j = ravel(coord, bshape);

      DType val;
      Reducer::SetInitValue(val);
      for (int k = threadIdx.x + Mstart; k < Mend; k += blockDim.x*unroll) {
        int kdot[unroll];
        #pragma unroll
        for (int u=0;u < unroll;u++) {
          kdot[u] = unravel_dot(k + u*blockDim.x, rshape, rstride);
        }
        DType tmp[unroll];
        #pragma unroll
        for (int u=0;u < unroll;u++) {
          if (k + u*blockDim.x < Mend) tmp[u] = OP::Map(big[j + kdot[u]]);
        }
        #pragma unroll
        for (int u=0;u < unroll;u++) {
          if (k + u*blockDim.x < Mend) Reducer::Reduce(val, tmp[u]);
        }
      }
      buf[threadIdx.x] = val;

      __syncthreads();
      Reduce1D<Reducer, x_bits>(buf);
      if (threadIdx.x == 0) {
        assign(&small[idx + m0*N], addto, buf[0]);
      }
    }
  }

}

const int nthread_reduce = kMaxThreadsPerBlock;
template<typename Reducer, typename DType, typename OP, int unroll>
__launch_bounds__(nthread_reduce)
__global__ void reduce_kernel(const int N, const int M, const bool addto,
                              const DType* __restrict big, DType *small, const CShape bshape,
                              const CShape sshape, const CShape rshape, const CShape rstride,
                              const int Mnext) {
  // Size of shared memory is blockDim.x*( (blockDim.y > 1) ? blockDim.y : 0 )*sizeof(DType)
  extern __shared__ char shTileChar[];
  DType* shTile = (DType*)(shTileChar);
  const int it0 = threadIdx.x + threadIdx.y*blockDim.x;
  for (int m0 = blockIdx.y; m0 < Mnext; m0 += gridDim.y) {
    // This TB handles M range [Mstart, ...., Mend - 1]
    const int Mstart = (int)((uint64_t)M*(uint64_t)m0/(uint64_t)Mnext);
    const int Mend   = (int)((uint64_t)M*(uint64_t)(m0 + 1)/(uint64_t)Mnext);
    for (int idx0 = blockIdx.x*blockDim.x; idx0 < N; idx0 += blockDim.x*gridDim.x) {
      int idx = idx0 + threadIdx.x;
      CShape coord = unravel(idx, sshape);
      int j = ravel(coord, bshape);

      DType val;
      Reducer::SetInitValue(val);
      if (idx < N) {
        for (int k = threadIdx.y + Mstart; k < Mend; k += blockDim.y*unroll) {
          int kdot[unroll];
          #pragma unroll
          for (int u=0;u < unroll;u++) {
            kdot[u] = unravel_dot(k + u*blockDim.y, rshape, rstride);
          }
          DType tmp[unroll];
          #pragma unroll
          for (int u=0;u < unroll;u++) {
            if (k + u*blockDim.y < Mend) tmp[u] = OP::Map(big[j + kdot[u]]);
          }
          #pragma unroll
          for (int u=0;u < unroll;u++) {
            if (k + u*blockDim.y < Mend) Reducer::Reduce(val, tmp[u]);
          }
        }
      }

      if (blockDim.y > 1) {
        shTile[it0] = val;
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
          assign(&small[idx + m0*N], addto, shTile[threadIdx.x]);
        }
      } else {
        if (idx < N) {
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
    
    DType val;
    Reducer::SetInitValue(val);
    for (int k = 0; k < M; k++) {
      Reducer::Reduce(val, small_in[idx + k*small_in_stride]);
    }

    if (idx < N) {
      assign(&small_out[idx], addto, val);
    }

  }
}

template<typename Reducer, typename DType, typename OP>
__global__ void reduce_kernel_M1(const int N, const bool addto,
                                const DType* __restrict big, DType *small, const CShape bshape,
                                const CShape sshape) {
  for (int idx = threadIdx.x + blockIdx.x*blockDim.x; idx < N; idx += blockDim.x*gridDim.x) {
    CShape coord = unravel(idx, sshape);
    int j = ravel(coord, bshape);
    assign(&small[idx], addto, OP::Map(big[j]));
  }
}

template<typename Reducer, typename DType, typename OP>
size_t ReduceImpl(Stream<gpu> *s, const TBlob& small, const OpReqType req,
                  const TBlob& big, const Tensor<gpu, 1, char>& workspace,
                  const bool getWorkspaceSize) {
  size_t workspace_pos = 0;
  if (req == kNullOp) return workspace_pos;
  cudaStream_t stream = Stream<gpu>::GetStream(s);
  CShape rshape, rstride;
  int mdim = diff(small.shape_.get<MAX_DIM>(), big.shape_.get<MAX_DIM>(), &rshape, &rstride);
  int N = small.shape_.Size(), M = rshape.Size();

  const int warpSize = 32;

  if (M == 1) {
    if (!getWorkspaceSize) {
      dim3 blockDim;
      dim3 gridDim;
      blockDim.x = kMaxThreadsPerBlock;
      gridDim.x = std::min((unsigned int)kBaseGridNum, (N + blockDim.x - 1)/blockDim.x);
      reduce_kernel_M1<Reducer, DType, OP><<< gridDim, blockDim, 0, stream>>>(
        N, req == kAddTo, big.dptr<DType>(), small.dptr<DType>(), big.shape_.get<MAX_DIM>(),
        small.shape_.get<MAX_DIM>());
    }
  } else {

    dim3 blockDim;
    dim3 gridDim;

    const int kMaxThreadBits = 10;
    const int par_reduce_lim = 32;
    const int unroll_par_reduce = 2;
    const int unroll_reduce = 4;

    int Mnext = 1;
    if (N <= par_reduce_lim) {
      blockDim.x = 1 << kMaxThreadBits;
      gridDim.x = std::min(kBaseGridNum, N);
      gridDim.y = std::min(kBaseGridNum, Mnext);
      const int maxLoopPerTB = 64;
      int maxMblock = blockDim.x*maxLoopPerTB;
      Mnext = (M + maxMblock - 1) / maxMblock;
    } else {
      const int maxLoopPerTB = 64;
      if (M >= maxLoopPerTB*32) {
        // M is large enough, choose square thread block
        blockDim.y = std::min(M, nthread_reduce/warpSize);
      } else if (M > 40) {
        // M is medium, choose rectangular thread block
        blockDim.y = 4;
      } else {
        // M is small, choose flat thread block
        blockDim.y = 1;
      }
      blockDim.x = (nthread_reduce/(blockDim.y*warpSize))*warpSize;
      gridDim.x = std::min((unsigned int)kBaseGridNum, (N + blockDim.x - 1)/blockDim.x);
      gridDim.y = std::min(kBaseGridNum, Mnext);
      // Maximum number of times we want TB to loop in M
      // Max size of M-block each TB can handle
      int maxMblock = blockDim.y*maxLoopPerTB;
      Mnext = (M + maxMblock - 1) / maxMblock;
    }

    DType* out1_dptr = small.dptr<DType>();
    bool addto = (req == kAddTo);
    if (Mnext > 1) {
      // out1_dptr[] is N*Mnext*sizeof(DType) bytes
      out1_dptr = reinterpret_cast<DType*>(workspace.dptr_ + workspace_pos);
      workspace_pos += N*Mnext*sizeof(DType);
      addto = false;
      // Check that the workspace is contigiuous
      if (!getWorkspaceSize) CHECK_EQ(workspace.CheckContiguous(), true);
      // Check that we have enough storage
      if (!getWorkspaceSize) CHECK_GE(workspace.size(0), workspace_pos);
      // Set gridDim.y to Mnext
      gridDim.y = std::min(kBaseGridNum, Mnext);
    }

    if (N <= par_reduce_lim) {
      if (!getWorkspaceSize) {
        if ( M / (blockDim.x*Mnext) >= unroll_par_reduce ) {
          par_reduce_kernel<Reducer, DType, OP, kMaxThreadBits, unroll_par_reduce>
          <<< gridDim, blockDim, 0, stream>>>(
            N, M, addto, big.dptr<DType>(), out1_dptr, big.shape_.get<MAX_DIM>(),
            small.shape_.get<MAX_DIM>(), rshape, rstride, Mnext);
        } else {
          par_reduce_kernel<Reducer, DType, OP, kMaxThreadBits, 1>
          <<< gridDim, blockDim, 0, stream>>>(
            N, M, addto, big.dptr<DType>(), out1_dptr, big.shape_.get<MAX_DIM>(),
            small.shape_.get<MAX_DIM>(), rshape, rstride, Mnext);          
        }
      }
    } else {
      if (!getWorkspaceSize) {
        int shMemSize = blockDim.x*( (blockDim.y > 1) ? blockDim.y : 0 )*sizeof(DType);
        if ( M / (blockDim.y*Mnext) >= unroll_reduce ) {
          reduce_kernel<Reducer, DType, OP, unroll_reduce>
          <<< gridDim, blockDim, shMemSize, stream>>>(
            N, M, addto, big.dptr<DType>(), out1_dptr, big.shape_.get<MAX_DIM>(),
            small.shape_.get<MAX_DIM>(), rshape, rstride, Mnext);
        } else {
          reduce_kernel<Reducer, DType, OP, 1>
          <<< gridDim, blockDim, shMemSize, stream>>>(
            N, M, addto, big.dptr<DType>(), out1_dptr, big.shape_.get<MAX_DIM>(),
            small.shape_.get<MAX_DIM>(), rshape, rstride, Mnext);          
        }
      }
    }

    if (Mnext > 1) {
      if (!getWorkspaceSize) {
        int blockSize = kMaxThreadsPerBlock;
        int gridSize = std::min((int)kBaseGridNum, (N + blockSize - 1)/blockSize );
        reduce_lines_kernel<Reducer, DType><<< gridSize, blockSize, 0, stream >>>
        (N, Mnext, req == kAddTo, N, out1_dptr, small.dptr<DType>());
      }
    }

  }

  return workspace_pos;
}

template<typename Reducer, typename DType, typename OP>
void Reduce(Stream<gpu> *s, const TBlob& small, const OpReqType req,
            const TBlob& big, const Tensor<gpu, 1, char>& workspace) {
  ReduceImpl<Reducer, DType, OP>(s, small, req, big, workspace, false);
}

template<typename Reducer, typename DType, typename OP>
size_t ReduceWorkspaceSize(Stream<gpu> *s, const TBlob& small, const OpReqType req,
                           const TBlob& big) {
  mshadow::Tensor<gpu, 1, char> dummy_workspace;
  return ReduceImpl<Reducer, DType, OP>(s, small, req, big, dummy_workspace, true);
}

#endif  //MXNET_OPERATOR_TENSOR_BROADCAST_REDUCE_INL_CUH_
