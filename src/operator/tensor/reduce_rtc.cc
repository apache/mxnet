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

#include "broadcast_reduce-inl.h"
#include "elemwise_unary_op.h"

#if MXNET_USE_CUDA
#include "../../common/cuda/rtc.h"
#endif  // MXNET_USE_CUDA

using namespace mshadow;

namespace mxnet {
namespace op {
namespace broadcast {

#if MXNET_USE_CUDA

namespace {

constexpr int nthread_reduce = 512;
constexpr int kBaseGridNum = 1024;

int diff(const TShape& small, const TShape& big, TShape* dims,
  TShape* stride) {
  int ndim = small.ndim();
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

constexpr int warpSize = 32;
constexpr int unroll_reduce = 2;
constexpr int maxLoopPerTB = 64;

// Returns a/b integer division rounded up
template<typename Type>
Type ceil_idiv(const Type a, const Type b) {
  return (a + b - 1)/b;
}

uint64_t calc_num_load(const int X, const int Y, const int* strides) {
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
  return num_load;
}

struct RTCReduceImplConfig {
  index_t N;
  index_t M;
  index_t Mnext;
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

  TShape rshape, rstride;
  TShape lhs_shape, lhs_stride;
  TShape rhs_shape, rhs_stride;

  RTCReduceImplConfig(const ::mxnet::TShape& small, const ::mxnet::TShape& big,
                      const size_t type_size, const ::mxnet::TShape* lhs,
                      const ::mxnet::TShape* rhs) :
    rshape(small.ndim(), 1), rstride(small.ndim(), 1),
    lhs_shape(small.ndim(), 1), lhs_stride(small.ndim(), 1),
    rhs_shape(small.ndim(), 1), rhs_stride(small.ndim(), 1) {
    int ndim = small.ndim();

    diff(small, big, &rshape, &rstride);
    N = small.Size();

    M = rshape[0];
    for (int i = 1; i < ndim; ++i) {
      M *= rshape[i];
    }

    bool multiOp = false;
    if (lhs != nullptr) {
      CHECK_NOTNULL(rhs);
      diff(small, *lhs, &lhs_shape, &lhs_stride);
      diff(small, *rhs, &rhs_shape, &rhs_stride);
      multiOp = true;
    }

    workspace_size = 0;

    if (M == 1) {
      kernel_1.blockDim.x = nthread_reduce;
      kernel_1.gridDim.x = std::min((unsigned int)kBaseGridNum,
          (N + kernel_1.blockDim.x - 1)/kernel_1.blockDim.x);
    } else {

      int reduce_strides[3];
      reduce_strides[0] = fastest_stride(small, big, big);
      reduce_strides[1] = (multiOp) ? fastest_stride(small, *lhs, *lhs) : 1;
      reduce_strides[2] = (multiOp) ? fastest_stride(small, *rhs, *rhs) : 1;

      int reduce_strides_transp[3];
      reduce_strides_transp[0] = fastest_stride(small, rshape, rstride);
      reduce_strides_transp[1] = (multiOp) ?
        fastest_stride(small, lhs_shape, lhs_stride) : 1;
      reduce_strides_transp[2] = (multiOp) ?
        fastest_stride(small, rhs_shape, rhs_stride) : 1;

      uint64_t num_load = calc_num_load(N, M, reduce_strides);
      uint64_t num_load_transp = calc_num_load(M, N, reduce_strides_transp);

      Mnext = 1;
      kernel_1.do_transpose = (num_load > num_load_transp);

      kernel_1.blockDim.x = 0;
      kernel_1.blockDim.y = 0;

      if (kernel_1.do_transpose) {
        // Fastest thread ID goes through M
        // Loop over N has step size kernel_1.blockDim.y
        if (N < 8) {
          kernel_1.blockDim.y = 1;
        } else if (N < 256) {
          kernel_1.blockDim.y = 4;
        } else {
          if (M < 8) {
            kernel_1.blockDim.x = 1;
          } else if (M < 256) {
            kernel_1.blockDim.x = 4;
          } else {
            kernel_1.blockDim.x = warpSize;
          }
        }
      } else {
        // Fastest thread ID goes through N
        // Loop over M has step size kernel_1.blockDim.y
        if (M < 8) {
          kernel_1.blockDim.y = 1;
        } else if (M < 256) {
          kernel_1.blockDim.y = 4;
        } else {
          if (N < 8) {
            kernel_1.blockDim.x = 1;
          } else if (N < 256) {
            kernel_1.blockDim.x = 4;
          } else {
            kernel_1.blockDim.x = warpSize;
          }
        }
      }

      if (kernel_1.blockDim.x == 0 && kernel_1.blockDim.y == 0) {
        LOG(FATAL) << "Unable to set blockDim";
      } else if (kernel_1.blockDim.x == 0) {
        kernel_1.blockDim.x = nthread_reduce / kernel_1.blockDim.y;
      } else if (kernel_1.blockDim.y == 0) {
        kernel_1.blockDim.y = nthread_reduce / kernel_1.blockDim.x;
      }

      if (kernel_1.do_transpose) {
        // Fastest thread ID goes through M
        kernel_1.gridDim.x = std::min((unsigned int)kBaseGridNum,
            ceil_idiv<unsigned int>(N, kernel_1.blockDim.y));
        kernel_1.gridDim.y = std::min(kBaseGridNum, Mnext);
        int by = kernel_1.blockDim.y;
        if (kernel_1.blockDim.y % warpSize == 0) {
          // Fix shared memory bank conflict
          by++;
        }
        kernel_1.shMemSize = (kernel_1.blockDim.x > 1) ?
          kernel_1.blockDim.x*by*type_size * 2 : 0;
        // Maximum number of times we want TB to loop in M
        // Max size of M-block each TB can handle
        int maxMblock = kernel_1.blockDim.x*maxLoopPerTB;
        Mnext = (M + maxMblock - 1) / maxMblock;
      } else {
        // Fastest thread ID goes through N
        kernel_1.gridDim.x = std::min((unsigned int)kBaseGridNum,
            ceil_idiv<unsigned int>(N, kernel_1.blockDim.x));
        kernel_1.gridDim.y = std::min(kBaseGridNum, Mnext);
        kernel_1.shMemSize = (kernel_1.blockDim.y > 1) ?
          kernel_1.blockDim.x*kernel_1.blockDim.y*type_size * 2 : 0;
        // Maximum number of times we want TB to loop in M
        // Max size of M-block each TB can handle
        int maxMblock = kernel_1.blockDim.y*maxLoopPerTB;
        Mnext = (M + maxMblock - 1) / maxMblock;
      }

      if (Mnext > 1) {
        // small_dptr[] is N*Mnext*type_size bytes
        workspace_size += N*Mnext*sizeof(double);
        // Set gridDim.y to Mnext
        kernel_1.gridDim.y = std::min(kBaseGridNum, Mnext);
      }

      if (Mnext > 1) {
        kernel_2.blockSize = nthread_reduce;
        kernel_2.gridSize = std::min((int)kBaseGridNum,
            (N + kernel_2.blockSize - 1)/kernel_2.blockSize );
      }

    }
  }

};

struct reduce_kernel_params {
  index_t big_shape[MAX_DIM];
  index_t small_shape[MAX_DIM];
  index_t rshape[MAX_DIM];
  index_t rstride[MAX_DIM];
};

const char reduce_kernel_code[] = R"code(
struct reduce_kernel_params {
  index_t big_shape[util::MAX_DIM];
  index_t small_shape[util::MAX_DIM];
  index_t rshape[util::MAX_DIM];
  index_t rstride[util::MAX_DIM];
};

__global__ void reduce_kernel(const int N, const int M, const bool addto,
                              const InputType0* __restrict big,
                              OutputType0 *small,
                              const reduce_kernel_params params,
                              const int Mnext) {
  extern __shared__ char shTileChar[];
  using IType = AccType<InputType0>;
  using OType = AccType<OutputType0>;
  using AType = typename IType::type;
  AType* shTile = (AType*)(shTileChar);
  const int tid = threadIdx.x + threadIdx.y*blockDim.x;
  const int bx = (do_transpose) ? blockDim.y : blockDim.x;
  const int by = (do_transpose) ? blockDim.x : blockDim.y;
  const int tidx = (do_transpose) ? tid / by : threadIdx.x;
  const int tidy = (do_transpose) ? tid % by : threadIdx.y;
  for (int m0 = blockIdx.y; m0 < Mnext; m0 += gridDim.y) {
    // This TB handles M range [Mstart, ...., Mend - 1]
    const index_t Mstart = (index_t)((int64)M*(int64)m0/(int64)Mnext);
    const index_t Mend   = (index_t)((int64)M*(int64)(m0 + 1)/(int64)Mnext);
    for (index_t idx0 = blockIdx.x*bx; idx0 < N; idx0 += bx*gridDim.x) {
      int idx = idx0 + tidx;
      index_t idx_big0 = util::unravel_ravel<ndim>(idx, params.small_shape, params.big_shape);

      AType val, residual;
      REDUCER::SetInitValue(val, residual);
      if (idx < N) {
        for (index_t k = tidy + Mstart; k < Mend; k += by*UNROLL) {
          index_t idx_big[UNROLL];
          #pragma unroll
          for (int u=0;u < UNROLL;u++) {
            idx_big[u] = idx_big0 + util::unravel_dot<ndim>(k + u*by, params.rshape,
                                                            params.rstride);
          }
          typename OType::type tmp[UNROLL];
          #pragma unroll
          for (int u=0;u < UNROLL;u++) {
            if (k + u*by < Mend) {
              tmp[u] = OP(OType::from(big[idx_big[u]]));
            }
          }
          #pragma unroll
          for (int u=0;u < UNROLL;u++) {
            if (k + u*by < Mend) REDUCER::Reduce(val, tmp[u], residual);
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
          REDUCER::SetInitValue(tmp, tmp_residual);
          if (tidy + t < by) {
            tmp = shTile[(it0 + t*fbx) * 2];
            tmp_residual = shTile[(it0 + t*fbx) * 2 + 1];
          }
          __syncthreads();
          REDUCER::Merge(shTile[it0 * 2], shTile[it0 * 2 + 1], tmp, tmp_residual);
          __syncthreads();
        }
        if (idx < N && tidy == 0) {
          if (addto) {
            small[idx + m0 * N] = OType::to(op::add(OType::from(small[idx + m0 * N]),
                                                    shTile[tidx * 2]));
          } else {
            small[idx + m0 * N] = OType::to(shTile[tidx * 2]);
          }
        }
      } else {
        if (idx < N) {
          if (addto) {
            small[idx + m0 * N] = OType::to(op::add(OType::from(small[idx + m0 * N]),
                                                    val));
          } else {
            small[idx + m0 * N] = OType::to(val);
          }
        }
      }
    }
  }
}
)code";

const char reduce_lines_kernel_code[] = R"code(
__global__ void reduce_lines_kernel(const index_t N, const index_t M,
                                    const index_t small_in_stride,
                                    const OutputType0* __restrict small_in,
                                    OutputType0 *small_out) {
  using OType = AccType<OutputType0>;
  for (index_t idx = threadIdx.x + blockIdx.x*blockDim.x; idx < N; idx += blockDim.x*gridDim.x) {
    typename OType::type val, residual;
    REDUCER::SetInitValue(val, residual);
    for (int k = 0; k < M; k++) {
      REDUCER::Reduce(val,
        OType::from(reinterpret_cast<const OutputType0*>(small_in)[idx + k*small_in_stride]),
        residual);
    }

    if (idx < N) {
      if (req == OpReqType::kAddTo) {
        small_out[idx] = OType::to(op::add(OType::from(small_out[idx]), val));
      } else {
        small_out[idx] = OType::to(val);
      }
    }

  }
}
)code";

void RTCReduceImpl(Stream<gpu> *s, const TBlob& small, const bool addto,
                const TBlob& big, const Tensor<gpu, 1, char>& workspace,
                const RTCReduceImplConfig& config, const int ndim,
                const std::string &common_code, int dev_id) {
  using namespace common::cuda::rtc;
  void* small_dptr = small.dptr_;
  bool first_kernel_addto = addto;
  if (config.Mnext > 1) {
    // small_dptr[] is N*Mnext*sizeof(DType) bytes
    small_dptr = workspace.dptr_;
    first_kernel_addto = false;
    // Check that the workspace is contigiuous
    CHECK_EQ(workspace.CheckContiguous(), true);
    // Check that we have enough storage
    CHECK_GE(workspace.size(0), config.workspace_size);
  }

  const int by = (config.kernel_1.do_transpose) ?
    config.kernel_1.blockDim.x : config.kernel_1.blockDim.y;
  const bool do_unroll = ( config.M / (by*config.Mnext) >= unroll_reduce );
  std::string code = common_code +
                     "#define UNROLL " +
                     (do_unroll ? std::to_string(unroll_reduce) : "1") +
                     "\n"
                     "const bool do_transpose = " +
                     (config.kernel_1.do_transpose ? "true" : "false") +
                     ";\n"
                     "using InputType0 = " +
                     util::mshadow_type_info(big.type_flag_).name +
                     ";\n"
                     "using OutputType0 = " +
                     util::mshadow_type_info(small.type_flag_).name +
                     ";\n";

  reduce_kernel_params param {};
  for (int i = 0; i < ndim; ++i) {
    param.big_shape[i] = big.shape_[i];
    param.small_shape[i] = small.shape_[i];
    param.rshape[i] = config.rshape[i];
    param.rstride[i] = config.rstride[i];
  }

  std::vector<const void*> args;
  args.emplace_back(&config.N);
  args.emplace_back(&config.M);
  args.emplace_back(&first_kernel_addto);
  args.emplace_back(&big.dptr_);
  args.emplace_back(&small_dptr);
  args.emplace_back(&param);
  args.emplace_back(&config.Mnext);

  auto reduce_kernel_func = get_function(code + reduce_kernel_code, "reduce_kernel", dev_id);
  launch(reduce_kernel_func, config.kernel_1.gridDim, config.kernel_1.blockDim, config.kernel_1.shMemSize, s, &args);

  if (config.Mnext > 1) {
    args.resize(0);
    args.emplace_back(&config.N);
    args.emplace_back(&config.Mnext);
    args.emplace_back(&config.N);
    args.emplace_back(&small_dptr);
    args.emplace_back(&small.dptr_);

    auto reduce_lines_kernel_func = get_function(code + reduce_lines_kernel_code,
                                                 "reduce_lines_kernel", dev_id);
    launch(reduce_lines_kernel_func, config.kernel_2.gridSize,
           config.kernel_2.blockSize, 0, s, &args);
  }
}


}  // namespace

void RTCReduce(const NodeAttrs& attrs,
               const OpContext& ctx,
               const TBlob& small,
               const OpReqType req,
               const Tensor<gpu, 1, char>& workspace,
               const TBlob& big,
               const std::string& reducer,
               int ndim,
               const std::string& OP) {
  using namespace mxnet::common::cuda::rtc;
  if (req == kNullOp) return;
  Stream<gpu> *s = ctx.get_stream<gpu>();
  size_t type_size = util::mshadow_type_info(small.type_flag_).size;
  if (small.type_flag_ == mshadow::kFloat16) {
    type_size = sizeof(float);
  }
  RTCReduceImplConfig config(small.shape_, big.shape_, type_size, nullptr, nullptr);
  if (config.M == 1) {
    // With M == 1 result is just (possibly reshaped) OP(big)
    UnaryRTCCompute {OP} (attrs, ctx, {big}, {req}, {small});
  } else {
    std::string common_code = std::string("const OpReqType req = ") +
                              util::to_string(req) +
                              ";\n"
                              "#define OP op::" +
                              OP +
                              "\n"
                              "#define REDUCER " +
                              reducer +
                              "\n"
                              "const int ndim = " +
                              std::to_string(ndim) +
                              ";\n";
    RTCReduceImpl(s, small, req == kAddTo, big, workspace, config,
                  ndim, common_code, ctx.run_ctx.ctx.dev_id);
  }
}

#endif  // MXNET_USE_CUDA

}  // namespace broadcast
}  // namespace op
}  // namespace mxnet
