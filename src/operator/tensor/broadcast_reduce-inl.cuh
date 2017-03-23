/*!
 * Copyright (c) 2015-2017 by Contributors
 * \file broadcast_reduce-inl.cuh
 * \brief CUDA implementations for binary broadcast and reduce
 * \author Antti-Pekka Hynninen
*/
#ifndef MXNET_OPERATOR_TENSOR_BROADCAST_REDUCE_INL_CUH_
#define MXNET_OPERATOR_TENSOR_BROADCAST_REDUCE_INL_CUH_

// template<typename DType>
// using CTensor = Tensor<gpu, ndim, DType>;

using namespace mshadow::cuda;

template<int ndim, typename DType, typename OP, int unroll>
__launch_bounds__(kMaxThreadsPerBlock)
__global__ void binary_broadcast_kernel(const int N, const bool addto,
                                        const DType* __restrict lhs,
                                        const DType* __restrict rhs, DType *out,
                                        const Shape<ndim> lstride, const Shape<ndim> rstride,
                                        const Shape<ndim> oshape) {
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

template<int ndim, typename DType, typename OP>
void BinaryBroadcastComputeImpl(Stream<gpu> *s, const OpReqType req,
                                const TBlob& lhs, const TBlob& rhs, const TBlob& out) {
  if (req == kNullOp) return;
  cudaStream_t stream = Stream<gpu>::GetStream(s);
  int N = out.shape_.Size();
  const int warpSize = 32;
  const int unroll = 2;
  int nthread = std::min(kMaxThreadsPerBlock, ((N + warpSize - 1)/warpSize)*warpSize );
  int ngrid = std::min(kBaseGridNum, (N + nthread*unroll - 1) / (nthread*unroll));
  Shape<ndim> lstride = calc_stride(lhs.shape_.get<ndim>());
  Shape<ndim> rstride = calc_stride(rhs.shape_.get<ndim>());
  binary_broadcast_kernel<ndim, DType, OP, unroll><<<ngrid, nthread, 0, stream>>>(
    N, req == kAddTo, lhs.dptr<DType>(), rhs.dptr<DType>(), out.dptr<DType>(), lstride, rstride,
    out.shape_.get<ndim>());
}

// Performs batched reduction: blockDim_y number of reductions across blockDim_x
template<typename Reducer, typename DType, int blockDim_x, int blockDim_y>
__device__ __forceinline__ DType batchReduce(const DType val) {
  DType ret;
#if __CUDA_ARCH__ >= 3000
  #pragma unroll
  for (int i=16;i >= 1;i/=2) {
    Reducer::Reduce(val, (DType)__shfl_xor(val, i));
  }
  ret = val;
#else
  // Size of shared memory is blockDim_x*blockDim_y*sizeof(DType)
  volatile __shared__ DType shBuf[blockDim_y][blockDim_x];
  shBuf[threadIdx.y][threadIdx.x] = val;
  __syncthreads();
  for (int t=1;t < blockDim_x;t <<= 1) {
    DType tmp;
    Reducer::SetInitValue(tmp);
    if (threadIdx.x + t < blockDim_x) tmp = shBuf[threadIdx.y][threadIdx.x + t];
    __syncthreads();
    Reducer::Reduce(shBuf[threadIdx.y][threadIdx.x], tmp);
    __syncthreads();
  }
  ret = shBuf[threadIdx.y][threadIdx.x];
#endif
  return ret;
}

const int nthread_par_reduce = kMaxThreadsPerBlock;

template<typename Reducer, int ndim, typename DType, typename OP, int unroll, int blockDim_x, int blockDim_y>
__launch_bounds__(nthread_par_reduce)
__global__ void par_reduce_kernel(const int N, const int M, const bool addto,
                                  const DType* __restrict big,
                                  DType *small,
                                  const Shape<ndim> big_shape0,
                                  const Shape<ndim> small_shape,
                                  const Shape<ndim> big_shape,
                                  const Shape<ndim> big_stride,
                                  const int Mnext) {
  for (int m0 = blockIdx.y; m0 < Mnext; m0 += gridDim.y) {
    // This TB handles M range [Mstart, ...., Mend - 1]
    const int Mstart = (int)((uint64_t)M*(uint64_t)m0/(uint64_t)Mnext);
    const int Mend   = (int)((uint64_t)M*(uint64_t)(m0 + 1)/(uint64_t)Mnext);
    for (int idx = threadIdx.y + blockIdx.x*blockDim_y; idx < N; idx += gridDim.x*blockDim_y) {
      Shape<ndim> coord = unravel(idx, small_shape);
      int idx_big0 = ravel(coord, big_shape0);

      DType val;
      Reducer::SetInitValue(val);
      for (int k = threadIdx.x + Mstart; k < Mend; k += blockDim_x*unroll) {
        int idx_big[unroll];
        #pragma unroll
        for (int u=0;u < unroll;u++) {
          idx_big[u] = idx_big0 + unravel_dot(k + u*blockDim_x, big_shape, big_stride);
        }
        DType tmp[unroll];
        #pragma unroll
        for (int u=0;u < unroll;u++) {
          if (k + u*blockDim_x < Mend) {
            tmp[u] = OP::Map(big[idx_big[u]]);
          }
        }
        #pragma unroll
        for (int u=0;u < unroll;u++) {
          if (k + u*blockDim_x < Mend) Reducer::Reduce(val, tmp[u]);
        }
      }

      val = batchReduce<Reducer, DType, blockDim_x, blockDim_y>(val);
      if (threadIdx.x == 0) {
        assign(&small[idx + m0*N], addto, val);
      }
    }
  }

}

#if 1
template<typename Reducer, int ndim, typename DType, typename OP1, typename OP2, int unroll, int blockDim_x,
         int blockDim_y>
__launch_bounds__(nthread_par_reduce)
__global__ void par_reduce_kernel(const int N, const int M, const bool addto,
                                  const DType* __restrict big,
                                  const DType* __restrict lhs,
                                  const DType* __restrict rhs,
                                  DType *small,
                                  const Shape<ndim> big_shape0,
                                  const Shape<ndim> lhs_shape0,
                                  const Shape<ndim> rhs_shape0,
                                  const Shape<ndim> small_shape,
                                  const Shape<ndim> big_shape,
                                  const Shape<ndim> lhs_shape,
                                  const Shape<ndim> rhs_shape,
                                  const Shape<ndim> big_stride,
                                  const Shape<ndim> lhs_stride,
                                  const Shape<ndim> rhs_stride,
                                  const int Mnext, const bool do_transpose) {
  const int tid = threadIdx.x + threadIdx.y*blockDim_x;
  const int tidx = (do_transpose) ? tid / blockDim_y : threadIdx.x;
  const int tidy = (do_transpose) ? tid % blockDim_y : threadIdx.y;
  for (int m0 = blockIdx.y; m0 < Mnext; m0 += gridDim.y) {
    // This TB handles M range [Mstart, ...., Mend - 1]
    const int Mstart = (int)((uint64_t)M*(uint64_t)m0/(uint64_t)Mnext);
    const int Mend   = (int)((uint64_t)M*(uint64_t)(m0 + 1)/(uint64_t)Mnext);
    for (int idx0 = blockIdx.x*blockDim_y; idx0 < N; idx0 += gridDim.x*blockDim_y) {
      int idx = idx0 + tidy;
      Shape<ndim> coord = unravel(idx, small_shape);
      int idx_big0 = ravel(coord, big_shape0);
      int idx_lhs0 = ravel(coord, lhs_shape0);
      int idx_rhs0 = ravel(coord, rhs_shape0);

      DType val;
      Reducer::SetInitValue(val);
      if (idx < N) {
        for (int k = tidx + Mstart; k < Mend; k += blockDim_x*unroll) {
          int idx_big[unroll];
          int idx_lhs[unroll];
          int idx_rhs[unroll];
          #pragma unroll
          for (int u=0;u < unroll;u++) {
            idx_big[u] = idx_big0 + unravel_dot(k + u*blockDim_x, big_shape, big_stride);
            idx_lhs[u] = idx_lhs0 + unravel_dot(k + u*blockDim_x, lhs_shape, lhs_stride);
            idx_rhs[u] = idx_rhs0 + unravel_dot(k + u*blockDim_x, rhs_shape, rhs_stride);
          }
          DType tmp[unroll];
          #pragma unroll
          for (int u=0;u < unroll;u++) {
            if (k + u*blockDim_x < Mend) {
              tmp[u] = OP1::Map(big[idx_big[u]], OP2::Map(lhs[idx_lhs[u]], rhs[idx_rhs[u]]));
            }
          }
          #pragma unroll
          for (int u=0;u < unroll;u++) {
            if (k + u*blockDim_x < Mend) Reducer::Reduce(val, tmp[u]);
          }
        }
      }

      // Size of shared memory is (blockDim_x + 1)*blockDim_y*sizeof(DType)
      volatile __shared__ DType shBuf[blockDim_y][blockDim_x+1];
      shBuf[tidy][tidx] = val;
      __syncthreads();
      for (int t=1;t < blockDim_x;t <<= 1) {
        DType tmp;
        Reducer::SetInitValue(tmp);
        if (threadIdx.x + t < blockDim_x) tmp = shBuf[threadIdx.y][threadIdx.x + t];
        __syncthreads();
        Reducer::Reduce(shBuf[threadIdx.y][threadIdx.x], tmp);
        __syncthreads();
      }
      if (tidx == 0) {
        assign(&small[idx + m0*N], addto, shBuf[tidy][0]);
      }
    }
  }
}
#else
template<typename Reducer, int ndim, typename DType, typename OP1, typename OP2, int unroll, int blockDim_x,
         int blockDim_y>
__launch_bounds__(nthread_par_reduce)
__global__ void par_reduce_kernel(const int N, const int M, const bool addto,
                                  const DType* __restrict big,
                                  const DType* __restrict lhs,
                                  const DType* __restrict rhs,
                                  DType *small,
                                  const Shape<ndim> big_shape0,
                                  const Shape<ndim> lhs_shape0,
                                  const Shape<ndim> rhs_shape0,
                                  const Shape<ndim> small_shape,
                                  const Shape<ndim> big_shape,
                                  const Shape<ndim> lhs_shape,
                                  const Shape<ndim> rhs_shape,
                                  const Shape<ndim> big_stride,
                                  const Shape<ndim> lhs_stride,
                                  const Shape<ndim> rhs_stride,
                                  const int Mnext) {
  for (int m0 = blockIdx.y; m0 < Mnext; m0 += gridDim.y) {
    // This TB handles M range [Mstart, ...., Mend - 1]
    const int Mstart = (int)((uint64_t)M*(uint64_t)m0/(uint64_t)Mnext);
    const int Mend   = (int)((uint64_t)M*(uint64_t)(m0 + 1)/(uint64_t)Mnext);
    for (int idx = threadIdx.y + blockIdx.x*blockDim_y; idx < N; idx += gridDim.x*blockDim_y) {
      Shape<ndim> coord = unravel(idx, small_shape);
      int idx_big0 = ravel(coord, big_shape0);
      int idx_lhs0 = ravel(coord, lhs_shape0);
      int idx_rhs0 = ravel(coord, rhs_shape0);

      DType val;
      Reducer::SetInitValue(val);
      for (int k = threadIdx.x + Mstart; k < Mend; k += blockDim_x*unroll) {
        int idx_big[unroll];
        int idx_lhs[unroll];
        int idx_rhs[unroll];
        #pragma unroll
        for (int u=0;u < unroll;u++) {
          idx_big[u] = idx_big0 + unravel_dot(k + u*blockDim_x, big_shape, big_stride);
          idx_lhs[u] = idx_lhs0 + unravel_dot(k + u*blockDim_x, lhs_shape, lhs_stride);
          idx_rhs[u] = idx_rhs0 + unravel_dot(k + u*blockDim_x, rhs_shape, rhs_stride);
        }
        DType tmp[unroll];
        #pragma unroll
        for (int u=0;u < unroll;u++) {
          if (k + u*blockDim_x < Mend) {
            tmp[u] = OP1::Map(big[idx_big[u]], OP2::Map(lhs[idx_lhs[u]], rhs[idx_rhs[u]]));
          }
        }
        #pragma unroll
        for (int u=0;u < unroll;u++) {
          if (k + u*blockDim_x < Mend) Reducer::Reduce(val, tmp[u]);
        }
      }
      val = batchReduce<Reducer, DType, blockDim_x, blockDim_y>(val);
      if (threadIdx.x == 0) {
        assign(&small[idx + m0*N], addto, val);
      }
    }
  }
}
#endif

const int nthread_reduce = kMaxThreadsPerBlock;
template<typename Reducer, int ndim, typename DType, typename OP, int unroll>
__launch_bounds__(nthread_reduce)
__global__ void reduce_kernel(const int N, const int M, const bool addto,
                              const DType* __restrict big, DType *small, const Shape<ndim> bshape,
                              const Shape<ndim> sshape, const Shape<ndim> rshape, const Shape<ndim> rstride,
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
      Shape<ndim> coord = unravel(idx, sshape);
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

#if 1
template<typename Reducer, int ndim, typename DType, typename OP1, typename OP2, int unroll>
__launch_bounds__(nthread_reduce)
__global__ void reduce_kernel(const int N, const int M, const bool addto,
                              const DType* __restrict big,
                              const DType* __restrict lhs,
                              const DType* __restrict rhs,
                              DType *small,
                              const Shape<ndim> big_shape0,
                              const Shape<ndim> lhs_shape0,
                              const Shape<ndim> rhs_shape0,
                              const Shape<ndim> small_shape,
                              const Shape<ndim> big_shape,
                              const Shape<ndim> lhs_shape,
                              const Shape<ndim> rhs_shape,
                              const Shape<ndim> big_stride,
                              const Shape<ndim> lhs_stride,
                              const Shape<ndim> rhs_stride,
                              const int Mnext, const bool do_transpose) {
  // Size of shared memory is blockDim.x*( (blockDim.y > 1) ? blockDim.y : 0 )*sizeof(DType)
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
      Shape<ndim> coord = unravel(idx, small_shape);
      int idx_big0 = ravel(coord, big_shape0);
      int idx_lhs0 = ravel(coord, lhs_shape0);
      int idx_rhs0 = ravel(coord, rhs_shape0);

      DType val;
      Reducer::SetInitValue(val);
      if (idx < N) {
        for (int k = tidy + Mstart; k < Mend; k += by*unroll) {
          int idx_big[unroll];
          int idx_lhs[unroll];
          int idx_rhs[unroll];
          #pragma unroll
          for (int u=0;u < unroll;u++) {
            idx_big[u] = idx_big0 + unravel_dot(k + u*by, big_shape, big_stride);
            idx_lhs[u] = idx_lhs0 + unravel_dot(k + u*by, lhs_shape, lhs_stride);
            idx_rhs[u] = idx_rhs0 + unravel_dot(k + u*by, rhs_shape, rhs_stride);
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
            if (k + u*by < Mend) Reducer::Reduce(val, tmp[u]);
          }
        }
      }

      if (blockDim.y > 1) {
        shTile[tidx + tidy*(bx + 1)] = val;
        const int it0 = threadIdx.x + threadIdx.y*(blockDim.x + 1);
        __syncthreads();
        for (int t=1;t < blockDim.y;t <<= 1) {
          DType tmp;
          Reducer::SetInitValue(tmp);
          if (threadIdx.y + t < blockDim.y) tmp = shTile[it0 + t*(blockDim.x + 1)];
          __syncthreads();
          Reducer::Reduce(shTile[it0], tmp);
          __syncthreads();
        }
        if (idx < N && tidy == 0) {
          assign(&small[idx + m0*N], addto, shTile[tidx]);
        }
      } else {
        if (idx < N) {
          assign(&small[idx + m0*N], addto, val);
        }        
      }
    }
  }

}
#else
template<typename Reducer, typename DType, typename OP1, typename OP2, int unroll, int blockDim_x,
         int blockDim_y>
__launch_bounds__(nthread_reduce)
__global__ void reduce_kernel(const int N, const int M, const bool addto,
                              const DType* __restrict big,
                              const DType* __restrict lhs,
                              const DType* __restrict rhs,
                              DType *small,
                              const Shape<ndim> big_shape0,
                              const Shape<ndim> lhs_shape0,
                              const Shape<ndim> rhs_shape0,
                              const Shape<ndim> small_shape,
                              const Shape<ndim> big_shape,
                              const Shape<ndim> lhs_shape,
                              const Shape<ndim> rhs_shape,
                              const Shape<ndim> big_stride,
                              const Shape<ndim> lhs_stride,
                              const Shape<ndim> rhs_stride,
                              const int Mnext) {
  // Size of shared memory is blockDim.x*( (blockDim.y > 1) ? blockDim.y : 0 )*sizeof(DType)
  volatile DType shTile[blockDim_y][blockDim_x + 1];
  for (int m0 = blockIdx.y; m0 < Mnext; m0 += gridDim.y) {
    // This TB handles M range [Mstart, ...., Mend - 1]
    const int Mstart = (int)((uint64_t)M*(uint64_t)m0/(uint64_t)Mnext);
    const int Mend   = (int)((uint64_t)M*(uint64_t)(m0 + 1)/(uint64_t)Mnext);
    for (int idx0 = blockIdx.x*blockDim_x; idx0 < N; idx0 += blockDim_x*gridDim.x) {
      int idx = idx0 + threadIdx.y;
      Shape<ndim> coord = unravel(idx, small_shape);
      int idx_big0 = ravel(coord, big_shape0);
      int idx_lhs0 = ravel(coord, lhs_shape0);
      int idx_rhs0 = ravel(coord, rhs_shape0);

      DType val;
      Reducer::SetInitValue(val);
      if (idx < N) {
        for (int k = threadIdx.x + Mstart; k < Mend; k += blockDim_y*unroll) {
          int idx_big[unroll];
          int idx_lhs[unroll];
          int idx_rhs[unroll];
          #pragma unroll
          for (int u=0;u < unroll;u++) {
            idx_big[u] = idx_big0 + unravel_dot(k + u*blockDim_y, big_shape, big_stride);
            idx_lhs[u] = idx_lhs0 + unravel_dot(k + u*blockDim_y, lhs_shape, lhs_stride);
            idx_rhs[u] = idx_rhs0 + unravel_dot(k + u*blockDim_y, rhs_shape, rhs_stride);
          }
          DType tmp[unroll];
          #pragma unroll
          for (int u=0;u < unroll;u++) {
            if (k + u*blockDim_y < Mend) {
              tmp[u] = OP1::Map(big[idx_big[u]], OP2::Map(lhs[idx_lhs[u]], rhs[idx_rhs[u]]));
            }
          }
          #pragma unroll
          for (int u=0;u < unroll;u++) {
            if (k + u*blockDim_y < Mend) Reducer::Reduce(val, tmp[u]);
          }
        }
      }

      if (blockDim_y > 1) {
        shTile[threadIdx.x][threadIdx.y] = val;
        __syncthreads();
        for (int t=1;t < blockDim_y;t <<= 1) {
          DType tmp;
          Reducer::SetInitValue(tmp);
          if (threadIdx.y + t < blockDim_y) tmp = shTile[threadIdx.y + t][threadIdx.x];
          // __syncthreads();
          Reducer::Reduce(shTile[threadIdx.y][threadIdx.x], tmp);
          // __syncthreads();
        }
        if (idx < N && threadIdx.y == 0) {
          assign(&small[idx + m0*N], addto, shTile[0][threadIdx.x]);
        }
      } else {
        if (idx < N) {
          assign(&small[idx + m0*N], addto, val);
        }        
      }

    }
  }

}
#endif

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

template<typename Reducer, int ndim, typename DType, typename OP>
__global__ void reduce_kernel_M1(const int N, const bool addto,
                                const DType* __restrict big, DType *small, const Shape<ndim> bshape,
                                const Shape<ndim> sshape) {
  for (int idx = threadIdx.x + blockIdx.x*blockDim.x; idx < N; idx += blockDim.x*gridDim.x) {
    Shape<ndim> coord = unravel(idx, sshape);
    int j = ravel(coord, bshape);
    assign(&small[idx], addto, OP::Map(big[j]));
  }
}

template<typename Reducer, int ndim, typename DType, typename OP1, typename OP2>
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
    Shape<ndim> coord = unravel(idx, small_shape);
    int idx_big = ravel(coord, big_shape);
    int idx_lhs = ravel(coord, lhs_shape);
    int idx_rhs = ravel(coord, rhs_shape);
    DType val = OP1::Map(big[idx_big], OP2::Map(lhs[idx_lhs], rhs[idx_rhs]));
    assign(&small[idx], addto, val);
  }
}

// Returns the stride with which the fastest dimension is moving.
// Used to detect memory access scatter.
template<int ndim>
MSHADOW_XINLINE int fastest_stride(const Shape<ndim>& small, const Shape<ndim>& big, 
  const Shape<ndim>& big_stride) {
  for (int i = ndim-1; i >= 0; --i) {
    if (big[i] != 1) {
      return (small[i] == big[i]) ? 1 : big_stride[i];
    }
  }
  return 1;
}

// Returns a/b integer division rounded up
template<typename Type>
Type ceil_idiv(const Type a, const Type b) {
  return (a + b - 1)/b;
}

// Configuration for ReduceImpl()
template<int ndim>
struct ReduceImplConfig {
  static const int warpSize = 32;
  static const int par_reduce_lim = 32;
  static const int unroll_par_reduce = 2;
  static const int unroll_reduce = 2;//4;
  static const int maxLoopPerTB = 64;
  bool do_par_reduce;
  int N;
  int M;
  int Mnext;
  struct {
    dim3 blockDim;
    dim3 gridDim;
    int shMemSize;
    bool do_transpose;
  } kernel_1;
  struct {
    int blockSize;
    int gridSize;
  } kernel_2;
  size_t workspace_size;

  Shape<ndim> rshape, rstride;
  Shape<ndim> lhs_shape, lhs_stride;
  Shape<ndim> rhs_shape, rhs_stride;
};

static inline uint64_t calc_num_load(const int X, const int Y, const int* strides) {
  const int warpSize = ReduceImplConfig<1>::warpSize;
  // Number of full warps
  uint64_t num_full_warp = X / warpSize;
  // Length of the partial warp i.e. number of threads that are performing loads
  uint64_t len_part_warp = X % warpSize;

  uint64_t num_load_full = (std::min(warpSize, strides[0]) +
    std::min(warpSize, strides[1]) +
    std::min(warpSize, strides[2]))*num_full_warp;

  uint64_t num_load_part =
  (std::min(len_part_warp, ceil_idiv<uint64_t>(len_part_warp*strides[0], warpSize)) +
    std::min(len_part_warp, ceil_idiv<uint64_t>(len_part_warp*strides[1], warpSize)) +
    std::min(len_part_warp, ceil_idiv<uint64_t>(len_part_warp*strides[2], warpSize)))*
  (len_part_warp != 0);

  uint64_t num_load = (num_load_full + num_load_part)*(uint64_t)Y;
  // printf("num_load_full %ld num_load_part %ld\n", num_load_full, num_load_part);
  return num_load;
}

template<int ndim, typename DType>
ReduceImplConfig<ndim> ConfigureReduceImpl(const TBlob& small, const TBlob& big, const TBlob* lhs,
  const TBlob* rhs) {

  ReduceImplConfig<ndim> config;

  diff(small.shape_.get<ndim>(), big.shape_.get<ndim>(), &config.rshape, &config.rstride);
  config.N = small.shape_.Size();
  config.M = config.rshape.Size();

  printf("N %d M %d\n", config.N, config.M);

  bool multiOp = false;
  if (lhs != NULL) {
    CHECK_NOTNULL(rhs);
    diff(small.shape_.get<ndim>(), lhs->shape_.get<ndim>(), &config.lhs_shape,
      &config.lhs_stride);
    diff(small.shape_.get<ndim>(), rhs->shape_.get<ndim>(), &config.rhs_shape,
      &config.rhs_stride);
    multiOp = true;
  }

  config.workspace_size = 0;

  if (config.M == 1) {
    config.kernel_1.blockDim.x = kMaxThreadsPerBlock;
    config.kernel_1.gridDim.x = std::min((unsigned int)kBaseGridNum,
      (config.N + config.kernel_1.blockDim.x - 1)/config.kernel_1.blockDim.x);
  } else {

    int reduce_strides[3];
    reduce_strides[0] = fastest_stride(small.shape_.get<ndim>(), big.shape_.get<ndim>(),
      big.shape_.get<ndim>());
    reduce_strides[1] = (multiOp) ? fastest_stride(small.shape_.get<ndim>(),
      lhs->shape_.get<ndim>(), lhs->shape_.get<ndim>()) : 1;
    reduce_strides[2] = (multiOp) ? fastest_stride(small.shape_.get<ndim>(),
      rhs->shape_.get<ndim>(), rhs->shape_.get<ndim>()) : 1;
    printf("reduce_strides %d %d %d\n", reduce_strides[0], reduce_strides[1], reduce_strides[2]);

    int reduce_strides_transp[3];
    reduce_strides_transp[0] = fastest_stride(small.shape_.get<ndim>(), config.rshape,
      config.rstride);
    reduce_strides_transp[1] = (multiOp) ?
      fastest_stride(small.shape_.get<ndim>(), config.lhs_shape, config.lhs_stride) : 1;
    reduce_strides_transp[2] = (multiOp) ?
      fastest_stride(small.shape_.get<ndim>(), config.rhs_shape, config.rhs_stride) : 1;
    printf("reduce_strides_transp %d %d %d\n", reduce_strides_transp[0], reduce_strides_transp[1], reduce_strides_transp[2]);

    int par_reduce_strides[3];
    par_reduce_strides[0] = fastest_stride(config.rshape, config.rstride, config.rstride);
    par_reduce_strides[1] = (multiOp) ? fastest_stride(config.lhs_shape, config.lhs_stride,
      config.lhs_stride) : 1;
    par_reduce_strides[2] = (multiOp) ? fastest_stride(config.rhs_shape, config.rhs_stride,
      config.rhs_stride) : 1;
    printf("par_reduce_strides %d %d %d\n", par_reduce_strides[0], par_reduce_strides[1], par_reduce_strides[2]);

    int nreduce_stride_large = (int)(reduce_strides[0] >= 2) + (int)(reduce_strides[1] >= 2) +
      (int)(reduce_strides[2] >= 2);

    int npar_reduce_stride_large = (int)(par_reduce_strides[0] >= 2) +
      (int)(par_reduce_strides[1] >= 2) + (int)(par_reduce_strides[2] >= 2);

    // if (multiOp) {
    //   config.do_par_reduce = (par_reduce_strides[0] == 1 && par_reduce_strides[1] == 1 &&
    //                           par_reduce_strides[2] == 1 && nreduce_stride_large >= 2) ||
    //                          (config.N <= config.par_reduce_lim);
    // } else {
    //   config.do_par_reduce = (par_reduce_strides[0] == 1 && par_reduce_strides[1] == 1 &&
    //                           par_reduce_strides[2] == 1 && nreduce_stride_large >= 1) ||
    //                          (config.N <= config.par_reduce_lim);
    // }

    uint64_t num_par_load = calc_num_load(config.N, config.M, par_reduce_strides);
    // uint64_t num_par_load_transp = calc_num_load(config.M, config.N, par_reduce_strides_transp);

    uint64_t num_load = calc_num_load(config.N, config.M, reduce_strides);
    uint64_t num_load_transp = calc_num_load(config.M, config.N, reduce_strides_transp);

    printf("num_load %ld num_load_transp %ld num_par_load %ld\n", num_load, num_load_transp, num_par_load);

    // config.do_par_reduce = (num_par_load <= std::min(num_load, num_load_transp));

    config.do_par_reduce = false;//(config.N <= config.par_reduce_lim);

    config.Mnext = 1;
    if (config.do_par_reduce) {

      // config.kernel_1.blockDim.x = nthread_par_reduce;
      config.kernel_1.blockDim.x =
        (config.M >= nthread_par_reduce) ? nthread_par_reduce : config.warpSize;
      config.kernel_1.blockDim.y = nthread_par_reduce / config.kernel_1.blockDim.x;

      if (multiOp) {
        config.kernel_1.do_transpose = (npar_reduce_stride_large >= 2) &&
          (config.kernel_1.blockDim.y > 1);
      } else {
        config.kernel_1.do_transpose = (npar_reduce_stride_large >= 1) &&
          (config.kernel_1.blockDim.y > 1);
      }

      // config.kernel_1.shMemSize =
      //   (config.kernel_1.blockDim.x + 1)*config.kernel_1.blockDim.y*sizeof(DType);
      config.kernel_1.gridDim.x = std::min(kBaseGridNum, config.N);
      config.kernel_1.gridDim.y = std::min(kBaseGridNum, config.Mnext);
      int maxMblock = config.kernel_1.blockDim.x*config.maxLoopPerTB;
      config.Mnext = (config.M + maxMblock - 1) / maxMblock;
    } else {

      config.kernel_1.do_transpose = (num_load > num_load_transp);

      // if (multiOp) {
      //   config.kernel_1.do_transpose = (nreduce_stride_large >= 2) && (config.M >= 32);
      // } else {
      //   config.kernel_1.do_transpose = (nreduce_stride_large >= 1) && (config.M >= 32);
      // }


      if (config.kernel_1.do_transpose) {
        // if (config.M >= nthread_par_reduce) {
        //   config.kernel_1.blockDim.x = config.warpSize;
        // } else {
        //   config.kernel_1.blockDim.x = nthread_reduce;
        // }
        config.kernel_1.blockDim.x = 1;
        config.kernel_1.blockDim.y = nthread_reduce;
        // config.kernel_1.blockDim.x =
        //   (config.M >= nthread_reduce) ? nthread_reduce : config.warpSize;
        // config.kernel_1.blockDim.y = nthread_reduce / config.kernel_1.blockDim.x;
      } else {
        // if (config.M >= config.maxLoopPerTB*32)
        if (config.M >= 32)
        {
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
      }

      printf("bx by %d %d do_transpose %d\n",
        config.kernel_1.blockDim.x, config.kernel_1.blockDim.y, config.kernel_1.do_transpose);

      // config.kernel_1.blockDim.y = config.warpSize;

      config.kernel_1.shMemSize =
        (config.kernel_1.blockDim.x + 1)*( (config.kernel_1.blockDim.y > 1) ?
          config.kernel_1.blockDim.y : 0 )*sizeof(DType);

      if (config.kernel_1.do_transpose) {
        config.kernel_1.gridDim.x = std::min(kBaseGridNum, config.N);
        config.kernel_1.gridDim.y = std::min(kBaseGridNum, config.Mnext);

        // config.kernel_1.gridDim.x = std::min((unsigned int)kBaseGridNum,
        //   (config.N + config.kernel_1.blockDim.x - 1)/config.kernel_1.blockDim.x);
        // config.kernel_1.gridDim.y = std::min(kBaseGridNum, config.Mnext);

        // Maximum number of times we want TB to loop in M
        // Max size of M-block each TB can handle
        int maxMblock = config.kernel_1.blockDim.x*config.maxLoopPerTB;
        config.Mnext = (config.M + maxMblock - 1) / maxMblock;
      } else {
        config.kernel_1.gridDim.x = std::min((unsigned int)kBaseGridNum,
          (config.N + config.kernel_1.blockDim.x - 1)/config.kernel_1.blockDim.x);
        config.kernel_1.gridDim.y = std::min(kBaseGridNum, config.Mnext);

        // Maximum number of times we want TB to loop in M
        // Max size of M-block each TB can handle
        int maxMblock = config.kernel_1.blockDim.y*config.maxLoopPerTB;
        config.Mnext = (config.M + maxMblock - 1) / maxMblock;
      }
    }

    if (config.Mnext > 1) {
      // small_dptr[] is N*Mnext*sizeof(DType) bytes
      config.workspace_size += config.N*config.Mnext*sizeof(DType);
      // Set gridDim.y to Mnext
      config.kernel_1.gridDim.y = std::min(kBaseGridNum, config.Mnext);
    }

    printf("Mnext %d\n", config.Mnext);

    if (config.Mnext > 1) {
      config.kernel_2.blockSize = kMaxThreadsPerBlock;
      config.kernel_2.gridSize = std::min((int)kBaseGridNum,
        (config.N + config.kernel_2.blockSize - 1)/config.kernel_2.blockSize );
    }

  }

  return config;
}

template<int ndim>
void print_shape(const Shape<ndim>& shape) {
  for (int i=0;i < ndim;i++) {
    printf("%d ", shape[i]);
  }
  printf("\n");
}

#define KERNEL_UNROLL_SWITCH(do_unroll, unroll, UnRoll, ...)  \
  if (do_unroll) {                                            \
    const int UnRoll = unroll;                                \
    {__VA_ARGS__}                                             \
  } else {                                                    \
    const int UnRoll = 1;                                     \
    {__VA_ARGS__}                                             \
  }

#define KERNEL_BLOCKDIM_SWITCH(blockDim_y, BlockDim_y, ...)       \
  switch (blockDim_y) {                                           \
    case ReduceImplConfig<ndim>::warpSize:                        \
      {                                                           \
        const int BlockDim_y = ReduceImplConfig<ndim>::warpSize;  \
        {__VA_ARGS__}                                             \
      }                                                           \
    break;                                                        \
    case 4:                                                       \
      {                                                           \
        const int BlockDim_y = 4;                                 \
        {__VA_ARGS__}                                             \
      }                                                           \
    break;                                                        \
    case 1:                                                       \
      {                                                           \
        const int BlockDim_y = 1;                                 \
        {__VA_ARGS__}                                             \
      }                                                           \
    break;                                                        \
  }  

#define KERNEL_SWITCH(do_unroll, unroll, blockDim_y, ...)   \
  switch (blockDim_y) {                                     \
    case ReduceImplConfig<ndim>::warpSize:                        \
      if (do_unroll) {                                      \
        const int UNROLL = unroll;                          \
        const int BLOCKDIM_Y = ReduceImplConfig<ndim>::warpSize;  \
        {__VA_ARGS__}                                       \
      } else {                                              \
        const int UNROLL = 1;                               \
        const int BLOCKDIM_Y = ReduceImplConfig<ndim>::warpSize;  \
        {__VA_ARGS__}                                       \
      }                                                     \
    break;                                                  \
    case 4:                                                 \
      if (do_unroll) {                                      \
        const int UNROLL = unroll;                          \
        const int BLOCKDIM_Y = 4;                           \
        {__VA_ARGS__}                                       \
      } else {                                              \
        const int UNROLL = 1;                               \
        const int BLOCKDIM_Y = 4;                           \
        {__VA_ARGS__}                                       \
      }                                                     \
    break;                                                  \
    case 1:                                                 \
      if (do_unroll) {                                      \
        const int UNROLL = unroll;                          \
        const int BLOCKDIM_Y = 1;                           \
        {__VA_ARGS__}                                       \
      } else {                                              \
        const int UNROLL = 1;                               \
        const int BLOCKDIM_Y = 1;                           \
        {__VA_ARGS__}                                       \
      }                                                     \
    break;                                                  \
    default:                                                \
    LOG(FATAL) << "Unknown blockDim_y " << blockDim_y;      \
  }

template<typename Reducer, int ndim, typename DType, typename OP>
void ReduceImpl(cudaStream_t stream, const TBlob& small, const OpReqType req,
                const TBlob& big, const Tensor<gpu, 1, char>& workspace,
                const ReduceImplConfig<ndim>& config) {
  if (config.M == 1) {
    reduce_kernel_M1<Reducer, ndim, DType, OP>
    <<< config.kernel_1.gridDim, config.kernel_1.blockDim, 0, stream >>>(
      config.N, req == kAddTo, big.dptr<DType>(), small.dptr<DType>(), big.shape_.get<ndim>(),
      small.shape_.get<ndim>());
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

    if (config.do_par_reduce) {
      if ( config.M / (config.kernel_1.blockDim.x*config.Mnext) >= config.unroll_par_reduce ) {
        if (config.kernel_1.blockDim.y == 1) {
          par_reduce_kernel<Reducer, ndim, DType, OP, ReduceImplConfig<ndim>::unroll_par_reduce,
            nthread_par_reduce, 1>
          <<< config.kernel_1.gridDim, config.kernel_1.blockDim, 0, stream>>>(
            config.N, config.M, addto, big.dptr<DType>(), small_dptr, big.shape_.get<ndim>(),
            small.shape_.get<ndim>(), config.rshape, config.rstride, config.Mnext);
        } else {
          par_reduce_kernel<Reducer, ndim, DType, OP, ReduceImplConfig<ndim>::unroll_par_reduce,
            ReduceImplConfig<ndim>::warpSize, nthread_par_reduce/ReduceImplConfig<ndim>::warpSize>
          <<< config.kernel_1.gridDim, config.kernel_1.blockDim, 0, stream>>>(
            config.N, config.M, addto, big.dptr<DType>(), small_dptr, big.shape_.get<ndim>(),
            small.shape_.get<ndim>(), config.rshape, config.rstride, config.Mnext);          
        }
      } else {
        if (config.kernel_1.blockDim.y == 1) {
          par_reduce_kernel<Reducer, ndim, DType, OP, 1, nthread_par_reduce, 1>
          <<< config.kernel_1.gridDim, config.kernel_1.blockDim, 0, stream>>>(
            config.N, config.M, addto, big.dptr<DType>(), small_dptr, big.shape_.get<ndim>(),
            small.shape_.get<ndim>(), config.rshape, config.rstride, config.Mnext);
        } else {
          par_reduce_kernel<Reducer, ndim, DType, OP, 1,
            ReduceImplConfig<ndim>::warpSize, nthread_par_reduce/ReduceImplConfig<ndim>::warpSize>
          <<< config.kernel_1.gridDim, config.kernel_1.blockDim, 0, stream>>>(
            config.N, config.M, addto, big.dptr<DType>(), small_dptr, big.shape_.get<ndim>(),
            small.shape_.get<ndim>(), config.rshape, config.rstride, config.Mnext);
        }
      }
    } else {
      if ( config.M / (config.kernel_1.blockDim.y*config.Mnext) >= config.unroll_reduce ) {
        reduce_kernel<Reducer, ndim, DType, OP, ReduceImplConfig<ndim>::unroll_reduce>
        <<< config.kernel_1.gridDim, config.kernel_1.blockDim,
            config.kernel_1.shMemSize, stream>>>(
          config.N, config.M, addto, big.dptr<DType>(), small_dptr, big.shape_.get<ndim>(),
          small.shape_.get<ndim>(), config.rshape, config.rstride, config.Mnext);
      } else {
        reduce_kernel<Reducer, ndim, DType, OP, 1>
        <<< config.kernel_1.gridDim, config.kernel_1.blockDim,
            config.kernel_1.shMemSize, stream>>>(
          config.N, config.M, addto, big.dptr<DType>(), small_dptr, big.shape_.get<ndim>(),
          small.shape_.get<ndim>(), config.rshape, config.rstride, config.Mnext);
      }
    }

    if (config.Mnext > 1) {
      reduce_lines_kernel<Reducer, DType>
      <<< config.kernel_2.gridSize, config.kernel_2.blockSize, 0, stream >>>
        (config.N, config.Mnext, req == kAddTo, config.N, small_dptr, small.dptr<DType>());
    }

  }

}

template<typename Reducer, int ndim, typename DType, typename OP1, typename OP2>
void ReduceImpl(cudaStream_t stream, const TBlob& small, const TBlob& lhs, const TBlob& rhs,
                const OpReqType req, const TBlob& big, const Tensor<gpu, 1, char>& workspace,
                const ReduceImplConfig<ndim>& config) {

  // int cbit = 0;
  if (config.M == 1) {
    reduce_kernel_M1<Reducer, ndim, DType, OP1, OP2>
    <<< config.kernel_1.gridDim, config.kernel_1.blockDim, 0, stream >>>(
      config.N, req == kAddTo, big.dptr<DType>(), lhs.dptr<DType>(), rhs.dptr<DType>(),
      small.dptr<DType>(), big.shape_.get<ndim>(), lhs.shape_.get<ndim>(),
      rhs.shape_.get<ndim>(), small.shape_.get<ndim>());
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

    // cbit |= 1 << 0;

    if (config.do_par_reduce) {

      const bool do_unroll =
      ( config.M / (config.kernel_1.blockDim.x*config.Mnext) >= config.unroll_par_reduce );

#if 1
      KERNEL_SWITCH(do_unroll, ReduceImplConfig<ndim>::unroll_par_reduce, config.kernel_1.blockDim.y, {
        par_reduce_kernel<Reducer, ndim, DType, OP1, OP2, UNROLL, nthread_par_reduce/BLOCKDIM_Y,
                          BLOCKDIM_Y>
        <<< config.kernel_1.gridDim, config.kernel_1.blockDim, 0, stream>>>(
          config.N, config.M, addto, big.dptr<DType>(), lhs.dptr<DType>(), rhs.dptr<DType>(),
          small_dptr, big.shape_.get<ndim>(), lhs.shape_.get<ndim>(),
          rhs.shape_.get<ndim>(), small.shape_.get<ndim>(), config.rshape, config.lhs_shape,
          config.rhs_shape, config.rstride, config.lhs_stride, config.rhs_stride, config.Mnext,
          config.kernel_1.do_transpose);
      })
#else
      KERNEL_SWITCH(do_unroll, ReduceImplConfig<ndim>::unroll_par_reduce, config.kernel_1.blockDim.y, {
        par_reduce_kernel<Reducer, ndim, DType, OP1, OP2, UNROLL, nthread_par_reduce/BLOCKDIM_Y,
                          BLOCKDIM_Y>
        <<< config.kernel_1.gridDim, config.kernel_1.blockDim, 0, stream>>>(
          config.N, config.M, addto, big.dptr<DType>(), lhs.dptr<DType>(), rhs.dptr<DType>(),
          small_dptr, big.shape_.get<ndim>(), lhs.shape_.get<ndim>(),
          rhs.shape_.get<ndim>(), small.shape_.get<ndim>(), config.rshape, config.lhs_shape,
          config.rhs_shape, config.rstride, config.lhs_stride, config.rhs_stride, config.Mnext);
      })
#endif
      // if ( config.M / (config.kernel_1.blockDim.x*config.Mnext) >= config.unroll_par_reduce ) {
      //   if (config.kernel_1.blockDim.y == 1) {
      //     par_reduce_kernel<Reducer, DType, OP1, OP2, ReduceImplConfig<ndim>::unroll_par_reduce,
      //       nthread_par_reduce, 1>
      //     <<< config.kernel_1.gridDim, config.kernel_1.blockDim, 0, stream>>>(
      //       config.N, config.M, addto, big.dptr<DType>(), lhs.dptr<DType>(), rhs.dptr<DType>(),
      //       small_dptr, big.shape_.get<ndim>(), lhs.shape_.get<ndim>(),
      //       rhs.shape_.get<ndim>(), small.shape_.get<ndim>(), config.rshape, config.lhs_shape,
      //       config.rhs_shape, config.rstride, config.lhs_stride, config.rhs_stride, config.Mnext);
      //   } else {
      //     par_reduce_kernel<Reducer, DType, OP1, OP2, ReduceImplConfig<ndim>::unroll_par_reduce,
      //       ReduceImplConfig<ndim>::warpSize, nthread_par_reduce/ReduceImplConfig<ndim>::warpSize>
      //     <<< config.kernel_1.gridDim, config.kernel_1.blockDim, 0, stream>>>(
      //       config.N, config.M, addto, big.dptr<DType>(), lhs.dptr<DType>(), rhs.dptr<DType>(),
      //       small_dptr, big.shape_.get<ndim>(), lhs.shape_.get<ndim>(),
      //       rhs.shape_.get<ndim>(), small.shape_.get<ndim>(), config.rshape, config.lhs_shape,
      //       config.rhs_shape, config.rstride, config.lhs_stride, config.rhs_stride, config.Mnext);
      //   }
      // } else {
      //   if (config.kernel_1.blockDim.y == 1) {
      //     par_reduce_kernel<Reducer, DType, OP1, OP2, 1, nthread_par_reduce, 1>
      //     <<< config.kernel_1.gridDim, config.kernel_1.blockDim, 0, stream>>>(
      //       config.N, config.M, addto, big.dptr<DType>(), lhs.dptr<DType>(), rhs.dptr<DType>(),
      //       small_dptr, big.shape_.get<ndim>(), lhs.shape_.get<ndim>(),
      //       rhs.shape_.get<ndim>(), small.shape_.get<ndim>(), config.rshape, config.lhs_shape,
      //       config.rhs_shape, config.rstride, config.lhs_stride, config.rhs_stride, config.Mnext);
      //   } else {
      //     par_reduce_kernel<Reducer, DType, OP1, OP2, 1,
      //       ReduceImplConfig<ndim>::warpSize, nthread_par_reduce/ReduceImplConfig<ndim>::warpSize>
      //     <<< config.kernel_1.gridDim, config.kernel_1.blockDim, 0, stream>>>(
      //       config.N, config.M, addto, big.dptr<DType>(), lhs.dptr<DType>(), rhs.dptr<DType>(),
      //       small_dptr, big.shape_.get<ndim>(), lhs.shape_.get<ndim>(),
      //       rhs.shape_.get<ndim>(), small.shape_.get<ndim>(), config.rshape, config.lhs_shape,
      //       config.rhs_shape, config.rstride, config.lhs_stride, config.rhs_stride, config.Mnext);
      //   }
      //   // cbit |= 1 << 1;
      // }
    } else {
      // cbit |= 1 << 2;
#if 0
      const bool do_unroll =
        (config.M / (config.kernel_1.blockDim.y*config.Mnext) >= config.unroll_reduce );
      KERNEL_SWITCH(do_unroll, ReduceImplConfig<ndim>::unroll_reduce, config.kernel_1.blockDim.y, {
        reduce_kernel<Reducer, ndim, DType, OP1, OP2, UNROLL, nthread_reduce/BLOCKDIM_Y, BLOCKDIM_Y>
        <<< config.kernel_1.gridDim, config.kernel_1.blockDim, 0, stream>>>(
          config.N, config.M, addto, big.dptr<DType>(), lhs.dptr<DType>(), rhs.dptr<DType>(),
          small_dptr, big.shape_.get<ndim>(), lhs.shape_.get<ndim>(),
          rhs.shape_.get<ndim>(), small.shape_.get<ndim>(), config.rshape, config.lhs_shape,
          config.rhs_shape, config.rstride, config.lhs_stride, config.rhs_stride, config.Mnext);
      })
#else
      if ( config.M / (config.kernel_1.blockDim.y*config.Mnext) >= config.unroll_reduce ) {
        reduce_kernel<Reducer, ndim, DType, OP1, OP2, ReduceImplConfig<ndim>::unroll_reduce>
        <<< config.kernel_1.gridDim, config.kernel_1.blockDim,
            config.kernel_1.shMemSize, stream>>>(
          config.N, config.M, addto, big.dptr<DType>(), lhs.dptr<DType>(), rhs.dptr<DType>(),
          small_dptr, big.shape_.get<ndim>(), lhs.shape_.get<ndim>(),
          rhs.shape_.get<ndim>(), small.shape_.get<ndim>(), config.rshape, config.lhs_shape,
          config.rhs_shape, config.rstride, config.lhs_stride, config.rhs_stride, config.Mnext,
          config.kernel_1.do_transpose);
      } else {
        reduce_kernel<Reducer, ndim, DType, OP1, OP2, 1>
        <<< config.kernel_1.gridDim, config.kernel_1.blockDim,
            config.kernel_1.shMemSize, stream>>>(
          config.N, config.M, addto, big.dptr<DType>(), lhs.dptr<DType>(), rhs.dptr<DType>(),
          small_dptr, big.shape_.get<ndim>(), lhs.shape_.get<ndim>(),
          rhs.shape_.get<ndim>(), small.shape_.get<ndim>(), config.rshape, config.lhs_shape,
          config.rhs_shape, config.rstride, config.lhs_stride, config.rhs_stride, config.Mnext,
          config.kernel_1.do_transpose);
      }
#endif
    }

    if (config.Mnext > 1) {
      reduce_lines_kernel<Reducer, DType>
      <<< config.kernel_2.gridSize, config.kernel_2.blockSize, 0, stream >>>
        (config.N, config.Mnext, req == kAddTo, config.N, small_dptr, small.dptr<DType>());
      // cbit |= 1 << 3;
    }

  }

  // printf(" cbit%d ", cbit);

  // printf("\n");
}

#undef KERNEL_SWITCH

template<typename Reducer, int ndim, typename DType, typename OP>
void Reduce(Stream<gpu> *s, const TBlob& small, const OpReqType req,
            const Tensor<gpu, 1, char>& workspace, const TBlob& big) {
  if (req == kNullOp) return;
  cudaStream_t stream = Stream<gpu>::GetStream(s);
  ReduceImplConfig<ndim> config = ConfigureReduceImpl<ndim, DType>(small, big, NULL, NULL);
  ReduceImpl<Reducer, ndim, DType, OP>(stream, small, req, big, workspace, config);
}

template<typename Reducer, int ndim, typename DType, typename OP1, typename OP2>
void Reduce(Stream<gpu> *s, const TBlob& small, const OpReqType req,
            const Tensor<gpu, 1, char>& workspace, const TBlob& big,
            const TBlob& lhs, const TBlob& rhs) {
  if (req == kNullOp) return;
  cudaStream_t stream = Stream<gpu>::GetStream(s);
  ReduceImplConfig<ndim> config = ConfigureReduceImpl<ndim, DType>(small, big, &lhs, &rhs);
  ReduceImpl<Reducer, ndim, DType, OP1, OP2>(stream, small, lhs, rhs, req, big, workspace, config);
}

template<int ndim, typename DType>
size_t ReduceWorkspaceSize(Stream<gpu> *s, const TBlob& small, const OpReqType req,
                           const TBlob& big) {
  if (req == kNullOp) return 0;
  ReduceImplConfig<ndim> config = ConfigureReduceImpl<ndim, DType>(small, big, NULL, NULL);
  return config.workspace_size;
}

template<int ndim, typename DType>
size_t ReduceWorkspaceSize(Stream<gpu> *s, const TBlob& small, const OpReqType req,
                           const TBlob& big, const TBlob& lhs, const TBlob& rhs) {
  if (req == kNullOp) return 0;
  ReduceImplConfig<ndim> config = ConfigureReduceImpl<ndim, DType>(small, big, &lhs, &rhs);
  return config.workspace_size;
}

#endif  //MXNET_OPERATOR_TENSOR_BROADCAST_REDUCE_INL_CUH_
