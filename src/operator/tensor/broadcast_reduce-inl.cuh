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

const int nthread_par_reduce = kMaxThreadsPerBlock;
const int nbit_par_reduce = 10;
template<typename Reducer, typename DType, typename OP, int x_bits, int unroll>
__launch_bounds__(nthread_par_reduce)
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

template<typename Reducer, typename DType, typename OP1, typename OP2, int unroll>
__launch_bounds__(nthread_par_reduce)
__global__ void par_reduce_kernel2(const int N, const int M, const bool addto,
                                  const DType* __restrict big,
                                  const DType* __restrict lhs,
                                  const DType* __restrict rhs,
                                  DType *small,
                                  const CShape big_shape0,
                                  const CShape lhs_shape0,
                                  const CShape rhs_shape0,
                                  const CShape small_shape,
                                  const CShape big_shape,
                                  const CShape lhs_shape,
                                  const CShape rhs_shape,
                                  const CShape big_stride,
                                  const CShape lhs_stride,
                                  const CShape rhs_stride,
                                  const int Mnext) {
  // Size of shared memory is blockDim.x*blockDim.y*sizeof(DType)
  extern __shared__ char shBufChar[];
  DType* shBuf = (DType*)(shBufChar);

  for (int m0 = blockIdx.y; m0 < Mnext; m0 += gridDim.y) {
    // This TB handles M range [Mstart, ...., Mend - 1]
    const int Mstart = (int)((uint64_t)M*(uint64_t)m0/(uint64_t)Mnext);
    const int Mend   = (int)((uint64_t)M*(uint64_t)(m0 + 1)/(uint64_t)Mnext);
    for (int idx = threadIdx.y + blockIdx.x*blockDim.y; idx < N; idx += gridDim.x*blockDim.y) {
      CShape coord = unravel(idx, small_shape);
      int idx_big0 = ravel(coord, big_shape0);
      int idx_lhs0 = ravel(coord, lhs_shape0);
      int idx_rhs0 = ravel(coord, rhs_shape0);

      DType val;
      Reducer::SetInitValue(val);
      for (int k = threadIdx.x + Mstart; k < Mend; k += blockDim.x*unroll) {
        int idx_big[unroll];
        int idx_lhs[unroll];
        int idx_rhs[unroll];
        #pragma unroll
        for (int u=0;u < unroll;u++) {
          idx_big[u] = idx_big0 + unravel_dot(k + u*blockDim.x, big_shape, big_stride);
          idx_lhs[u] = idx_lhs0 + unravel_dot(k + u*blockDim.x, lhs_shape, lhs_stride);
          idx_rhs[u] = idx_rhs0 + unravel_dot(k + u*blockDim.x, rhs_shape, rhs_stride);
        }
        DType tmp[unroll];
        #pragma unroll
        for (int u=0;u < unroll;u++) {
          if (k + u*blockDim.x < Mend) {
            tmp[u] = OP1::Map(big[idx_big[u]], OP2::Map(lhs[idx_lhs[u]], rhs[idx_rhs[u]]));
          }
        }
        #pragma unroll
        for (int u=0;u < unroll;u++) {
          if (k + u*blockDim.x < Mend) Reducer::Reduce(val, tmp[u]);
        }
      }
      const int t0 = threadIdx.x + threadIdx.y*blockDim.x;
      shBuf[t0] = val;

      __syncthreads();
      for (int t=1;t < blockDim.x;t <<= 1) {
        DType tmp;
        Reducer::SetInitValue(tmp);
        if (threadIdx.x + t < blockDim.x) tmp = shBuf[t0 + t];
        __syncthreads();
        Reducer::Reduce(shBuf[t0], tmp);
        __syncthreads();
      }
      if (threadIdx.x == 0) {
        assign(&small[idx + m0*N], addto, shBuf[t0]);
      }
    }
  }

}

/*
template<typename Reducer, typename DType, typename OP1, typename OP2, int x_bits, int unroll>
__launch_bounds__(nthread_par_reduce)
__global__ void par_reduce_kernel(const int N, const int M, const bool addto,
                                  const DType* __restrict big,
                                  const DType* __restrict lhs,
                                  const DType* __restrict rhs,
                                  DType *small,
                                  const CShape big_shape0,
                                  const CShape lhs_shape0,
                                  const CShape rhs_shape0,
                                  const CShape small_shape,
                                  const CShape big_shape,
                                  const CShape lhs_shape,
                                  const CShape rhs_shape,
                                  const CShape big_stride,
                                  const CShape lhs_stride,
                                  const CShape rhs_stride,
                                  const int Mnext) {
  __shared__ DType buf[1<<x_bits];

  for (int m0 = blockIdx.y; m0 < Mnext; m0 += gridDim.y) {
    // This TB handles M range [Mstart, ...., Mend - 1]
    const int Mstart = (int)((uint64_t)M*(uint64_t)m0/(uint64_t)Mnext);
    const int Mend   = (int)((uint64_t)M*(uint64_t)(m0 + 1)/(uint64_t)Mnext);
    for (int idx = blockIdx.x; idx < N; idx += gridDim.x) {
      CShape coord = unravel(idx, small_shape);
      int idx_big0 = ravel(coord, big_shape0);
      int idx_lhs0 = ravel(coord, lhs_shape0);
      int idx_rhs0 = ravel(coord, rhs_shape0);

      DType val;
      Reducer::SetInitValue(val);
      for (int k = threadIdx.x + Mstart; k < Mend; k += blockDim.x*unroll) {
        int idx_big[unroll];
        int idx_lhs[unroll];
        int idx_rhs[unroll];
        #pragma unroll
        for (int u=0;u < unroll;u++) {
          idx_big[u] = idx_big0 + unravel_dot(k + u*blockDim.x, big_shape, big_stride);
          idx_lhs[u] = idx_lhs0 + unravel_dot(k + u*blockDim.x, lhs_shape, lhs_stride);
          idx_rhs[u] = idx_rhs0 + unravel_dot(k + u*blockDim.x, rhs_shape, rhs_stride);
        }
        DType tmp[unroll];
        #pragma unroll
        for (int u=0;u < unroll;u++) {
          if (k + u*blockDim.x < Mend) {
            tmp[u] = OP1::Map(big[idx_big[u]], OP2::Map(lhs[idx_lhs[u]], rhs[idx_rhs[u]]));
          }
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
*/


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

template<typename Reducer, typename DType, typename OP1, typename OP2, int unroll>
__launch_bounds__(nthread_reduce)
__global__ void reduce_kernel(const int N, const int M, const bool addto,
                              const DType* __restrict big,
                              const DType* __restrict lhs,
                              const DType* __restrict rhs,
                              DType *small,
                              const CShape big_shape0,
                              const CShape lhs_shape0,
                              const CShape rhs_shape0,
                              const CShape small_shape,
                              const CShape big_shape,
                              const CShape lhs_shape,
                              const CShape rhs_shape,
                              const CShape big_stride,
                              const CShape lhs_stride,
                              const CShape rhs_stride,
                              const int Mnext) {
  // Size of shared memory is blockDim.x*( (blockDim.y > 1) ? blockDim.y : 0 )*sizeof(DType)
  extern __shared__ char shTileChar[];
  DType* shTile = (DType*)(shTileChar);
  const int it0 = threadIdx.x + threadIdx.y*blockDim.x;
  bool printed = false;
  for (int m0 = blockIdx.y; m0 < Mnext; m0 += gridDim.y) {
    // This TB handles M range [Mstart, ...., Mend - 1]
    const int Mstart = (int)((uint64_t)M*(uint64_t)m0/(uint64_t)Mnext);
    const int Mend   = (int)((uint64_t)M*(uint64_t)(m0 + 1)/(uint64_t)Mnext);
    for (int idx0 = blockIdx.x*blockDim.x; idx0 < N; idx0 += blockDim.x*gridDim.x) {
      int idx = idx0 + threadIdx.x;
      CShape coord = unravel(idx, small_shape);
      int idx_big0 = ravel(coord, big_shape0);
      int idx_lhs0 = ravel(coord, lhs_shape0);
      int idx_rhs0 = ravel(coord, rhs_shape0);

      DType val;
      Reducer::SetInitValue(val);
      if (idx < N) {
        for (int k = threadIdx.y + Mstart; k < Mend; k += blockDim.y*unroll) {
          int idx_big[unroll];
          int idx_lhs[unroll];
          int idx_rhs[unroll];
          #pragma unroll
          for (int u=0;u < unroll;u++) {
            idx_big[u] = idx_big0 + unravel_dot(k + u*blockDim.y, big_shape, big_stride);
            idx_lhs[u] = idx_lhs0 + unravel_dot(k + u*blockDim.y, lhs_shape, lhs_stride);
            idx_rhs[u] = idx_rhs0 + unravel_dot(k + u*blockDim.y, rhs_shape, rhs_stride);
          }
          DType tmp[unroll];
          #pragma unroll
          for (int u=0;u < unroll;u++) {
            if (k + u*blockDim.y < Mend) {
              // if (u < 2 && threadIdx.x < 32 && threadIdx.y == 0 && blockIdx.x == 0 && blockIdx.y == 0 && !printed) {
              //   printf("%d %d\n", threadIdx.x, idx_big[u]);
              // }
              DType val_lhs = lhs[idx_lhs[u]];
              DType val_rhs = rhs[idx_rhs[u]];
              DType val_big = big[idx_big[u]];
              tmp[u] = OP1::Map(val_big, OP2::Map(val_lhs, val_rhs));
              // tmp[u] = OP1::Map(big[idx_big[u]], OP2::Map(lhs[idx_lhs[u]], rhs[idx_rhs[u]]));
            }
          }
          // printed = true;
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
template<typename Reducer, typename DType, typename OP>
__launch_bounds__(kMaxThreadsPerBlock)
__global__ void reduce_lines_kernel(const int N, const int M, const bool addto,
  const int small_in_stride, const DType* __restrict small_in, DType *small_out) {
  for (int idx = threadIdx.x + blockIdx.x*blockDim.x; idx < N; idx += blockDim.x*gridDim.x) {
    
    DType val;
    Reducer::SetInitValue(val);
    for (int k = 0; k < M; k++) {
      Reducer::Reduce(val, OP::Map(small_in[idx + k*small_in_stride]));
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

template<typename Reducer, typename DType, typename OP1, typename OP2>
__global__ void reduce_kernel_M1(const int N, const bool addto,
                                 const DType* __restrict big,
                                 const DType* __restrict lhs,
                                 const DType* __restrict rhs,
                                 DType *small,
                                 const CShape big_shape,
                                 const CShape lhs_shape,
                                 const CShape rhs_shape,
                                 const CShape small_shape) {
  for (int idx = threadIdx.x + blockIdx.x*blockDim.x; idx < N; idx += blockDim.x*gridDim.x) {
    CShape coord = unravel(idx, small_shape);
    int idx_big = ravel(coord, big_shape);
    int idx_lhs = ravel(coord, lhs_shape);
    int idx_rhs = ravel(coord, rhs_shape);
    DType val = OP1::Map(big[idx_big], OP2::Map(lhs[idx_lhs], rhs[idx_rhs]));
    assign(&small[idx], addto, val);
  }
}

// Configuration for ReduceImpl()
struct ReduceImplConfig {
  static const int warpSize = 32;
  static const int kMaxThreadBits = 10;
  static const int par_reduce_lim = 32;//1234567*2;//32;
  static const int unroll_par_reduce = 2;
  static const int unroll_reduce = 4;
  static const int maxLoopPerTB = 64;
  int N;
  int M;
  int Mnext;
  struct {
    dim3 blockDim;
    dim3 gridDim;
    int shMemSize;
  } kernel_1;
  struct {
    int blockSize;
    int gridSize;
  } kernel_2;
  size_t workspace_size;
};

template<typename DType>
ReduceImplConfig ConfigureReduceImpl(const int N, const int M) {
  ReduceImplConfig config;
  config.workspace_size = 0;
  config.N = N;
  config.M = M;

  if (config.M == 1) {
    config.kernel_1.blockDim.x = kMaxThreadsPerBlock;
    config.kernel_1.gridDim.x = std::min((unsigned int)kBaseGridNum,
      (config.N + config.kernel_1.blockDim.x - 1)/config.kernel_1.blockDim.x);
  } else {

    config.Mnext = 1;
    if (config.N <= config.par_reduce_lim) {
      // config.kernel_1.blockDim.x = 1 << nbit_par_reduce; //1 << config.kMaxThreadBits;
      // config.kernel_1.gridDim.x = std::min(kBaseGridNum, config.N);
      // config.kernel_1.gridDim.y = std::min(kBaseGridNum, config.Mnext);
      // int maxMblock = config.kernel_1.blockDim.x*config.maxLoopPerTB;
      // config.Mnext = (config.M + maxMblock - 1) / maxMblock;
      config.kernel_1.blockDim.x = ((std::min(config.M, kMaxThreadsPerBlock) +
        config.warpSize - 1)/config.warpSize)*config.warpSize;
      config.kernel_1.blockDim.y = kMaxThreadsPerBlock / config.kernel_1.blockDim.x;
      config.kernel_1.shMemSize =
        config.kernel_1.blockDim.x*config.kernel_1.blockDim.y*sizeof(DType);
      config.kernel_1.gridDim.x = std::min(kBaseGridNum, config.N);
      config.kernel_1.gridDim.y = std::min(kBaseGridNum, config.Mnext);
      int maxMblock = config.kernel_1.blockDim.x*config.maxLoopPerTB;
      config.Mnext = (config.M + maxMblock - 1) / maxMblock;
    } else {
      if (config.M >= config.maxLoopPerTB*32) {
        // M is large enough, choose square thread block
        config.kernel_1.blockDim.y = std::min(config.M, nthread_reduce/config.warpSize);
      } else if (config.M > 40) {
        // M is medium, choose rectangular thread block
        config.kernel_1.blockDim.y = 4;
      } else {
        // M is small, choose flat thread block
        config.kernel_1.blockDim.y = 1;
      }
      config.kernel_1.blockDim.x =
        (nthread_reduce/(config.kernel_1.blockDim.y*config.warpSize))*config.warpSize;
      config.kernel_1.gridDim.x = std::min((unsigned int)kBaseGridNum,
        (config.N + config.kernel_1.blockDim.x - 1)/config.kernel_1.blockDim.x);
      config.kernel_1.gridDim.y = std::min(kBaseGridNum, config.Mnext);
      // Maximum number of times we want TB to loop in M
      // Max size of M-block each TB can handle
      int maxMblock = config.kernel_1.blockDim.y*config.maxLoopPerTB;
      config.Mnext = (config.M + maxMblock - 1) / maxMblock;
    }

    if (config.Mnext > 1) {
      // small_dptr[] is N*Mnext*sizeof(DType) bytes
      config.workspace_size += config.N*config.Mnext*sizeof(DType);
      // Set gridDim.y to Mnext
      config.kernel_1.gridDim.y = std::min(kBaseGridNum, config.Mnext);
    }

    if (config.N <= config.par_reduce_lim) {
    } else {
      config.kernel_1.shMemSize =
        config.kernel_1.blockDim.x*( (config.kernel_1.blockDim.y > 1) ?
          config.kernel_1.blockDim.y : 0 )*sizeof(DType);
    }

    if (config.Mnext > 1) {
      config.kernel_2.blockSize = kMaxThreadsPerBlock;
      config.kernel_2.gridSize = std::min((int)kBaseGridNum,
        (config.N + config.kernel_2.blockSize - 1)/config.kernel_2.blockSize );
    }

  }

  return config;
}

template<typename Reducer, typename DType, typename OP>
void ReduceImpl(cudaStream_t stream, const TBlob& small, const OpReqType req,
                const TBlob& big, const Tensor<gpu, 1, char>& workspace,
                const CShape& rshape, const CShape& rstride, const ReduceImplConfig& config) {
  if (config.M == 1) {
    reduce_kernel_M1<Reducer, DType, OP>
    <<< config.kernel_1.gridDim, config.kernel_1.blockDim, 0, stream >>>(
      config.N, req == kAddTo, big.dptr<DType>(), small.dptr<DType>(), big.shape_.get<MAX_DIM>(),
      small.shape_.get<MAX_DIM>());
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

    if (config.N <= config.par_reduce_lim) {
      if ( config.M / (config.kernel_1.blockDim.x*config.Mnext) >= config.unroll_par_reduce ) {
        par_reduce_kernel<Reducer, DType, OP, nbit_par_reduce /*ReduceImplConfig::kMaxThreadBits*/,
          ReduceImplConfig::unroll_par_reduce>
        <<< config.kernel_1.gridDim, config.kernel_1.blockDim, 0, stream>>>(
          config.N, config.M, addto, big.dptr<DType>(), small_dptr, big.shape_.get<MAX_DIM>(),
          small.shape_.get<MAX_DIM>(), rshape, rstride, config.Mnext);
      } else {
        par_reduce_kernel<Reducer, DType, OP, nbit_par_reduce /*ReduceImplConfig::kMaxThreadBits*/, 1>
        <<< config.kernel_1.gridDim, config.kernel_1.blockDim, 0, stream>>>(
          config.N, config.M, addto, big.dptr<DType>(), small_dptr, big.shape_.get<MAX_DIM>(),
          small.shape_.get<MAX_DIM>(), rshape, rstride, config.Mnext);
      }
    } else {
      if ( config.M / (config.kernel_1.blockDim.y*config.Mnext) >= config.unroll_reduce ) {
        reduce_kernel<Reducer, DType, OP, ReduceImplConfig::unroll_reduce>
        <<< config.kernel_1.gridDim, config.kernel_1.blockDim,
            config.kernel_1.shMemSize, stream>>>(
          config.N, config.M, addto, big.dptr<DType>(), small_dptr, big.shape_.get<MAX_DIM>(),
          small.shape_.get<MAX_DIM>(), rshape, rstride, config.Mnext);
      } else {
        reduce_kernel<Reducer, DType, OP, 1>
        <<< config.kernel_1.gridDim, config.kernel_1.blockDim,
            config.kernel_1.shMemSize, stream>>>(
          config.N, config.M, addto, big.dptr<DType>(), small_dptr, big.shape_.get<MAX_DIM>(),
          small.shape_.get<MAX_DIM>(), rshape, rstride, config.Mnext);
      }
    }

    if (config.Mnext > 1) {
      reduce_lines_kernel<Reducer, DType, mshadow_op::identity>
      <<< config.kernel_2.gridSize, config.kernel_2.blockSize, 0, stream >>>
        (config.N, config.Mnext, req == kAddTo, config.N, small_dptr, small.dptr<DType>());
    }

  }

}

template<int ndim>
void print_shape(const Shape<ndim>& shape) {
  for (int i=0;i < ndim;i++) {
    printf("%d ", shape[i]);
  }
  printf("\n");
}

// Returns the stride with which the fastest dimension is moving.
// Used to detect memory access scatter.
template<int ndim>
MSHADOW_XINLINE int fastest_stride(const Shape<ndim>& small, const Shape<ndim>& big) {
  for (int i = ndim-1; i >= 0; --i) {
    if (big[i] != 1) {
      return (small[i] == big[i]) ? 1 : big[i];
    }
  }
  return 1;
}


template<typename Reducer, typename DType, typename OP1, typename OP2>
void ReduceImpl(cudaStream_t stream, const TBlob& small, const TBlob& lhs, const TBlob& rhs,
                const OpReqType req, const TBlob& big, const Tensor<gpu, 1, char>& workspace,
                const CShape& rshape, const CShape& rstride, const ReduceImplConfig& config) {

  CShape lhs_shape, lhs_stride;
  diff(small.shape_.get<MAX_DIM>(), lhs.shape_.get<MAX_DIM>(), &lhs_shape, &lhs_stride);

  CShape rhs_shape, rhs_stride;
  diff(small.shape_.get<MAX_DIM>(), rhs.shape_.get<MAX_DIM>(), &rhs_shape, &rhs_stride);

  int fastest_stride_big = fastest_stride(rshape, rstride);
  int fastest_stride_lhs = fastest_stride(lhs_shape, lhs_stride);
  int fastest_stride_rhs = fastest_stride(rhs_shape, rhs_stride);
  printf("fastest_stride %d %d %d\n", fastest_stride_big, fastest_stride_lhs, fastest_stride_rhs);

  if (config.M == 1) {
    printf("reduce_kernel_M1\n");
    reduce_kernel_M1<Reducer, DType, OP1, OP2>
    <<< config.kernel_1.gridDim, config.kernel_1.blockDim, 0, stream >>>(
      config.N, req == kAddTo, big.dptr<DType>(), lhs.dptr<DType>(), rhs.dptr<DType>(),
      small.dptr<DType>(), big.shape_.get<MAX_DIM>(), lhs.shape_.get<MAX_DIM>(),
      rhs.shape_.get<MAX_DIM>(), small.shape_.get<MAX_DIM>());
  } else {

    // printf("N %d M %d\n", config.N, config.M);
    // printf("rshape ");
    // print_shape(rshape);
    // printf("rstride ");
    // print_shape(rstride);

    // printf("lhs_shape ");
    // print_shape(lhs_shape);
    // printf("lhs_stride ");
    // print_shape(lhs_stride);

    // printf("rhs_shape ");
    // print_shape(rhs_shape);
    // printf("rhs_stride ");
    // print_shape(rhs_stride);

    // for (int i=0;i < 4;i++) {
    //   int idx = 0 + i;
    //   CShape coord = unravel(idx, small.shape_.get<MAX_DIM>());
    //   int idx_big0 = ravel(coord, big.shape_.get<MAX_DIM>());
    //   int idx_lhs0 = ravel(coord, lhs.shape_.get<MAX_DIM>());
    //   int idx_rhs0 = ravel(coord, rhs.shape_.get<MAX_DIM>());
    //   printf("%d | %d %d %d\n", idx, idx_big0, idx_lhs0, idx_rhs0);
    // }

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

    if (config.N <= config.par_reduce_lim) {
      if ( config.M / (config.kernel_1.blockDim.x*config.Mnext) >= config.unroll_par_reduce ) {
        printf("par_reduce_kernel 1\n");
        // par_reduce_kernel<Reducer, DType, OP1, OP2, nbit_par_reduce /*ReduceImplConfig::kMaxThreadBits*/,
        //   ReduceImplConfig::unroll_par_reduce>
        // <<< config.kernel_1.gridDim, config.kernel_1.blockDim, 0, stream>>>(
        //   config.N, config.M, addto, big.dptr<DType>(), lhs.dptr<DType>(), rhs.dptr<DType>(),
        //   small_dptr, big.shape_.get<MAX_DIM>(), lhs.shape_.get<MAX_DIM>(),
        //   rhs.shape_.get<MAX_DIM>(), small.shape_.get<MAX_DIM>(), rshape, lhs_shape, rhs_shape,
        //   rstride, lhs_stride, rhs_stride, config.Mnext);
        par_reduce_kernel2<Reducer, DType, OP1, OP2, ReduceImplConfig::unroll_par_reduce>
        <<< config.kernel_1.gridDim, config.kernel_1.blockDim, config.kernel_1.shMemSize, stream>>>(
          config.N, config.M, addto, big.dptr<DType>(), lhs.dptr<DType>(), rhs.dptr<DType>(),
          small_dptr, big.shape_.get<MAX_DIM>(), lhs.shape_.get<MAX_DIM>(),
          rhs.shape_.get<MAX_DIM>(), small.shape_.get<MAX_DIM>(), rshape, lhs_shape, rhs_shape,
          rstride, lhs_stride, rhs_stride, config.Mnext);
      } else {
        printf("par_reduce_kernel 2\n");
        // par_reduce_kernel<Reducer, DType, OP1, OP2, nbit_par_reduce /*ReduceImplConfig::kMaxThreadBits*/, 1>
        // <<< config.kernel_1.gridDim, config.kernel_1.blockDim, 0, stream>>>(
        //   config.N, config.M, addto, big.dptr<DType>(), lhs.dptr<DType>(), rhs.dptr<DType>(),
        //   small_dptr, big.shape_.get<MAX_DIM>(), lhs.shape_.get<MAX_DIM>(),
        //   rhs.shape_.get<MAX_DIM>(), small.shape_.get<MAX_DIM>(), rshape, lhs_shape, rhs_shape,
        //   rstride, lhs_stride, rhs_stride, config.Mnext);
        par_reduce_kernel2<Reducer, DType, OP1, OP2, 1>
        <<< config.kernel_1.gridDim, config.kernel_1.blockDim, config.kernel_1.shMemSize, stream>>>(
          config.N, config.M, addto, big.dptr<DType>(), lhs.dptr<DType>(), rhs.dptr<DType>(),
          small_dptr, big.shape_.get<MAX_DIM>(), lhs.shape_.get<MAX_DIM>(),
          rhs.shape_.get<MAX_DIM>(), small.shape_.get<MAX_DIM>(), rshape, lhs_shape, rhs_shape,
          rstride, lhs_stride, rhs_stride, config.Mnext);
      }
    } else {
      if ( config.M / (config.kernel_1.blockDim.y*config.Mnext) >= config.unroll_reduce ) {
        printf("reduce_kernel 1\n");
        reduce_kernel<Reducer, DType, OP1, OP2, ReduceImplConfig::unroll_reduce>
        <<< config.kernel_1.gridDim, config.kernel_1.blockDim,
            config.kernel_1.shMemSize, stream>>>(
          config.N, config.M, addto, big.dptr<DType>(), lhs.dptr<DType>(), rhs.dptr<DType>(),
          small_dptr, big.shape_.get<MAX_DIM>(), lhs.shape_.get<MAX_DIM>(),
          rhs.shape_.get<MAX_DIM>(), small.shape_.get<MAX_DIM>(), rshape, lhs_shape, rhs_shape,
          rstride, lhs_stride, rhs_stride, config.Mnext);
      } else {
        printf("reduce_kernel 2\n");
        reduce_kernel<Reducer, DType, OP1, OP2, 1>
        <<< config.kernel_1.gridDim, config.kernel_1.blockDim,
            config.kernel_1.shMemSize, stream>>>(
          config.N, config.M, addto, big.dptr<DType>(), lhs.dptr<DType>(), rhs.dptr<DType>(),
          small_dptr, big.shape_.get<MAX_DIM>(), lhs.shape_.get<MAX_DIM>(),
          rhs.shape_.get<MAX_DIM>(), small.shape_.get<MAX_DIM>(), rshape, lhs_shape, rhs_shape,
          rstride, lhs_stride, rhs_stride, config.Mnext);
      }
    }

    if (config.Mnext > 1) {
      printf("reduce_lines_kernel\n");
      reduce_lines_kernel<Reducer, DType, mshadow_op::identity>
      <<< config.kernel_2.gridSize, config.kernel_2.blockSize, 0, stream >>>
        (config.N, config.Mnext, req == kAddTo, config.N, small_dptr, small.dptr<DType>());
    }

  }

  printf("\n");
}

template<typename Reducer, typename DType, typename OP>
void Reduce(Stream<gpu> *s, const TBlob& small, const OpReqType req,
            const Tensor<gpu, 1, char>& workspace, const TBlob& big) {
  if (req == kNullOp) return;
  cudaStream_t stream = Stream<gpu>::GetStream(s);
  CShape rshape, rstride;
  diff(small.shape_.get<MAX_DIM>(), big.shape_.get<MAX_DIM>(), &rshape, &rstride);
  int N = small.shape_.Size(), M = rshape.Size();
  ReduceImplConfig config = ConfigureReduceImpl<DType>(N, M);
  ReduceImpl<Reducer, DType, OP>(stream, small, req, big, workspace, rshape, rstride, config);
}

template<typename Reducer, typename DType, typename OP1, typename OP2>
void Reduce(Stream<gpu> *s, const TBlob& small, const OpReqType req,
            const Tensor<gpu, 1, char>& workspace, const TBlob& big,
            const TBlob& lhs, const TBlob& rhs) {
  if (req == kNullOp) return;

  int fastest_stride_big = fastest_stride(small.shape_.get<MAX_DIM>(), big.shape_.get<MAX_DIM>());
  int fastest_stride_lhs = fastest_stride(small.shape_.get<MAX_DIM>(), lhs.shape_.get<MAX_DIM>());
  int fastest_stride_rhs = fastest_stride(small.shape_.get<MAX_DIM>(), rhs.shape_.get<MAX_DIM>());
  printf("fastest_stride %d %d %d\n", fastest_stride_big, fastest_stride_lhs, fastest_stride_rhs);

  // printf("small ");
  // print_shape(small.shape_.get<MAX_DIM>());

  // printf("big ");
  // print_shape(big.shape_.get<MAX_DIM>());

  // printf("lhs ");
  // print_shape(lhs.shape_.get<MAX_DIM>());

  // printf("rhs ");
  // print_shape(rhs.shape_.get<MAX_DIM>());

  cudaStream_t stream = Stream<gpu>::GetStream(s);
  CShape rshape, rstride;
  diff(small.shape_.get<MAX_DIM>(), big.shape_.get<MAX_DIM>(), &rshape, &rstride);
  int N = small.shape_.Size(), M = rshape.Size();
  ReduceImplConfig config = ConfigureReduceImpl<DType>(N, M);
  ReduceImpl<Reducer, DType, OP1, OP2>(stream, small, lhs, rhs, req, big, workspace, rshape,
    rstride, config);
}

template<typename DType>
size_t ReduceWorkspaceSize(Stream<gpu> *s, const TBlob& small, const OpReqType req,
                           const TBlob& big) {
  if (req == kNullOp) return 0;
  CShape rshape, rstride;
  diff(small.shape_.get<MAX_DIM>(), big.shape_.get<MAX_DIM>(), &rshape, &rstride);
  int N = small.shape_.Size(), M = rshape.Size();
  ReduceImplConfig config = ConfigureReduceImpl<DType>(N, M);
  return config.workspace_size;
}

#endif  //MXNET_OPERATOR_TENSOR_BROADCAST_REDUCE_INL_CUH_
