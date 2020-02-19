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
 *  Copyright (c) 2020 by Contributors
 * \file elemwise_op.cuh
 * \brief GPU helpers for elementwise operators
 */

#ifndef MXNET_OPERATOR_TENSOR_ELEMWISE_OP_CUH_
#define MXNET_OPERATOR_TENSOR_ELEMWISE_OP_CUH_

#include <cuda_runtime.h>
#include "../operator_common.h"
#include "../../common/cuda_utils.h"

#include <vector>

#if MXNET_USE_CUDA

namespace mxnet {
namespace op {

template <typename DType, typename LType>
class VectorizedStorage {
 public:
  constexpr static int nvec = sizeof(LType) / sizeof(DType);
  union vectorized_storage {
    LType aligned;
    DType separate[nvec];  // NOLINT(*)

    MSHADOW_XINLINE vectorized_storage() {}
    MSHADOW_XINLINE ~vectorized_storage() {}
  } scratch_;
};

template <typename DType, typename LType, bool aligned = false>
class VectorizedAccessor {
 public:
  VectorizedStorage<typename std::remove_const<DType>::type,
                    typename std::remove_const<LType>::type> storage_;

  LType* aligned_ptr_;
  DType* unaligned_ptr_;
  int alignment_;
  index_t n_elems_;

  MSHADOW_XINLINE VectorizedAccessor(DType* ptr, const index_t N) {
    unaligned_ptr_ = ptr;
    if (aligned) {
      alignment_ = 0;
      aligned_ptr_ = reinterpret_cast<LType*>(ptr);
      n_elems_ = (N + storage_.nvec - 1) / storage_.nvec;
    } else {
      size_t ptr_as_number = reinterpret_cast<size_t>(ptr);
      alignment_ = (ptr_as_number % sizeof(LType)) / sizeof(DType);
      aligned_ptr_ = reinterpret_cast<LType*>(ptr - alignment_);
      n_elems_ = (N + alignment_ + storage_.nvec - 1) / storage_.nvec;
    }
  }

  MSHADOW_XINLINE index_t num_aligned_elements() const {
    return n_elems_;
  }

  MSHADOW_XINLINE void load(const index_t id, const index_t N) {
    if (aligned) {
      storage_.scratch_.aligned = aligned_ptr_[id];
    } else {
      if (id > 0 && id < n_elems_ - 1) {
      storage_.scratch_.aligned = aligned_ptr_[id];
      } else {
#pragma unroll
        for (int j = 0; j < storage_.nvec; ++j) {
          DType* ptr = reinterpret_cast<DType*>(&(aligned_ptr_[id])) + j;
          if (reinterpret_cast<size_t>(ptr) >= reinterpret_cast<size_t>(unaligned_ptr_) &&
              reinterpret_cast<size_t>(ptr) < reinterpret_cast<size_t>(unaligned_ptr_ + N)) {
            storage_.scratch_.separate[j] = *ptr;
          }
        }
      }
    }
  }
};

template <typename DType, typename LType, bool aligned = false>
class VectorizedLoader : public VectorizedAccessor<const DType, const LType, aligned> {
 public:
  MSHADOW_XINLINE VectorizedLoader(const DType* ptr, const index_t N) :
    VectorizedAccessor<const DType, const LType, aligned>(ptr, N) {
  }
};

template <typename DType, typename LType, bool aligned = false>
class VectorizedStorer : public VectorizedAccessor<DType, LType, aligned> {
 public:
  MSHADOW_XINLINE VectorizedStorer(DType* ptr, const index_t N) :
    VectorizedAccessor<DType, LType, aligned>(ptr, N) {
  }

  MSHADOW_XINLINE void store(const index_t id, const index_t N) {
    if (aligned) {
      this->aligned_ptr_[id] = this->storage_.scratch_.aligned;
    } else {
      if (id > 0 && id < this->n_elems_ - 1) {
        this->aligned_ptr_[id] = this->storage_.scratch_.aligned;
      } else {
#pragma unroll
        for (int j = 0; j < this->storage_.nvec; ++j) {
          DType* ptr = reinterpret_cast<DType*>(&(this->aligned_ptr_[id])) + j;
          if (reinterpret_cast<size_t>(ptr) >= reinterpret_cast<size_t>(this->unaligned_ptr_) &&
              reinterpret_cast<size_t>(ptr) < reinterpret_cast<size_t>(this->unaligned_ptr_ + N)) {
            *ptr = this->storage_.scratch_.separate[j];
          }
        }
      }
    }
  }
};

namespace {

template <typename DType, int NumInputs, int NumOutputs>
struct VectorizedKernelParams {
  const DType* inputs[NumInputs];
  DType* outputs[NumOutputs];
};


template <bool aligned, typename DType, typename LType, typename OP, int req>
__global__ void VectorizedBinaryKernelFwd(const VectorizedKernelParams<DType, 2, 1> params,
                                          const index_t N) {
  VectorizedLoader<DType, LType, aligned> loader0(params.inputs[0], N);
  VectorizedLoader<DType, LType, aligned> loader1(params.inputs[1], N);
  VectorizedStorer<DType, LType, aligned> storer(params.outputs[0], N);

  const index_t M = loader0.num_aligned_elements();

  for (index_t tid = blockIdx.x * blockDim.x + threadIdx.x;
       tid < M;
       tid += gridDim.x * blockDim.x) {
    loader0.load(tid, N);
    loader1.load(tid, N);
    if (req == kAddTo) {
      storer.load(tid, N);
    }
#pragma unroll
    for (int i = 0; i < loader0.storage_.nvec; ++i) {
      DType temp = OP::Map(loader0.storage_.scratch_.separate[i],
                           loader1.storage_.scratch_.separate[i]);

      if (req == kAddTo) {
        storer.storage_.scratch_.separate[i] += temp;
      } else {
        storer.storage_.scratch_.separate[i] = temp;
      }
    }
    storer.store(tid, N);
  }
}

template <bool aligned, typename DType, typename LType,
          typename LOP, typename ROP, int lreq, int rreq>
__global__ void VectorizedBinaryKernelBwdUseNone(const VectorizedKernelParams<DType, 1, 2> params,
                                                 const index_t N) {
  VectorizedLoader<DType, LType, aligned> loader(params.inputs[0], N);
  VectorizedStorer<DType, LType, aligned> lstorer(params.outputs[0], N);
  VectorizedStorer<DType, LType, aligned> rstorer(params.outputs[1], N);

  const index_t M = loader.num_aligned_elements();

  for (index_t tid = blockIdx.x * blockDim.x + threadIdx.x;
       tid < M;
       tid += gridDim.x * blockDim.x) {
    loader.load(tid, N);
    if (lreq == kAddTo) {
      lstorer.load(tid, N);
    }
    if (rreq == kAddTo) {
      rstorer.load(tid, N);
    }
#pragma unroll
    for (int i = 0; i < loader.storage_.nvec; ++i) {
      DType inp = loader.storage_.scratch_.separate[i];
      if (!((std::is_same<LOP, mshadow_op::identity>::value && lreq == kWriteInplace) ||
            lreq == kNullOp)) {
        DType ltemp = LOP::Map(inp);
        if (lreq == kAddTo) {
          lstorer.storage_.scratch_.separate[i] += ltemp;
        } else {
          lstorer.storage_.scratch_.separate[i] = ltemp;
        }
        lstorer.store(tid, N);
      }
      if (!((std::is_same<ROP, mshadow_op::identity>::value && rreq == kWriteInplace) ||
            rreq == kNullOp)) {
        DType rtemp = ROP::Map(inp);

        if (rreq == kAddTo) {
          rstorer.storage_.scratch_.separate[i] += rtemp;
        } else {
          rstorer.storage_.scratch_.separate[i] = rtemp;
        }
        rstorer.store(tid, N);
      }
    }
  }
}

template <typename DType, typename OP, int req>
class VectorizedBinaryFwd {
 public:
  using ParamType = VectorizedKernelParams<DType, 2, 1>;

  template <bool aligned, typename LType>
  static void Launch(const index_t blocks, const index_t threads,
                     cudaStream_t stream,
                     const ParamType params, const index_t N) {
    VectorizedBinaryKernelFwd<aligned, DType, LType, OP, req>
      <<<blocks, threads, 0, stream>>>(params, N);
  }
};

template <typename DType, typename LOP, typename ROP, int lreq, int rreq>
class VectorizedBinaryBwdUseNone {
 public:
  using ParamType = VectorizedKernelParams<DType, 1, 2>;

  template <bool aligned, typename LType>
  static void Launch(const index_t blocks, const index_t threads,
                     cudaStream_t stream,
                     const ParamType params, const index_t N) {
    VectorizedBinaryKernelBwdUseNone<aligned, DType, LType, LOP, ROP, lreq, rreq>
      <<<blocks, threads, 0, stream>>>(params, N);
  }
};

enum class Alignment {
  SAME_ALIGNED,
  SAME_UNALIGNED,
  DIFFERENT
};

template <typename LType, typename DType>
int CalcAlignment(const DType* ptr) {
  size_t ptr_as_number = reinterpret_cast<size_t>(ptr);
  return ptr_as_number % sizeof(LType);
}

template <typename LType, typename DType, int N, int M>
Alignment CheckAlignment(const VectorizedKernelParams<DType, N, M>& params) {
  int align = -1;

  for (const DType* ptr : params.inputs) {
    int new_align = CalcAlignment<LType>(ptr);
    if (align == -1) {
      align = new_align;
    } else {
      if (align != new_align) {
        return Alignment::DIFFERENT;
      }
    }
  }

  for (const DType* ptr : params.outputs) {
    int new_align = CalcAlignment<LType>(ptr);
    if (align == -1) {
      align = new_align;
    } else {
      if (align != new_align) {
        return Alignment::DIFFERENT;
      }
    }
  }

  return align == 0 ? Alignment::SAME_ALIGNED
                    : Alignment::SAME_UNALIGNED;
}

template <typename DType, typename LType, typename Kernel>
void VectorizedKernelLauncher(const index_t size, mshadow::Stream<gpu>* s, typename Kernel::ParamType params) {
  static_assert(sizeof(LType) >= sizeof(DType), "Load type is smaller than operand type");
  if (size != 0) {
    cudaStream_t stream = mshadow::Stream<gpu>::GetStream(s);
    constexpr int nvec = sizeof(LType) / sizeof(DType);
    VectorizedLoader<DType, LType> l(params.inputs[0], size);
    size_t num_elements = l.num_aligned_elements();
    constexpr int threads = 512;
    constexpr int max_blocks = 65535;
    index_t blocks = std::min(static_cast<int>((num_elements + threads - 1) / threads),
                              max_blocks);
    auto align = CheckAlignment<LType, DType>(params);
    if (align == Alignment::SAME_ALIGNED && (size % nvec == 0)) {
      Kernel::template Launch<true, LType>(blocks, threads, stream, params, size);
    } else {
      if (align != Alignment::DIFFERENT) {
        Kernel::template Launch<false, LType>(blocks, threads, stream, params, size);
      } else {
        index_t blocks = std::min(static_cast<int>((size + threads - 1) /
                                                   threads),
                                  max_blocks);
        // If the pointers are aligned differently we cannot vectorize
        Kernel::template Launch<true, DType>(blocks, threads, stream, params, size);
      }
    }
  }
}

template<typename OP>
void VectorizedCompute(const nnvm::NodeAttrs &attrs,
                       const OpContext &ctx,
                       const std::vector<TBlob> &inputs,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &outputs) {
  if (req[0] == kNullOp) return;
  mshadow::Stream<gpu> *s = ctx.get_stream<gpu>();
  CHECK_EQ(inputs.size(), 2U);
  CHECK_EQ(outputs.size(), 1U);
  MXNET_ASSIGN_REQ_SWITCH(req[0], Req, {
    MSHADOW_TYPE_SWITCH(outputs[0].type_flag_, DType, {
      using LType = uint4;
      using Kernel = VectorizedBinaryFwd<DType, OP, Req>;

      const index_t size = outputs[0].Size();
      typename Kernel::ParamType params;
      params.inputs[0] = inputs[0].dptr<DType>();
      params.inputs[1] = inputs[1].dptr<DType>();
      params.outputs[0] = outputs[0].dptr<DType>();

      VectorizedKernelLauncher<DType, LType, Kernel>(size, s, params);
    });
  });
}

template<typename LOP, typename ROP>
void VectorizedBackwardUseNoneCompute(const nnvm::NodeAttrs &attrs,
                                      const OpContext &ctx,
                                      const std::vector<TBlob> &inputs,
                                      const std::vector<OpReqType> &req,
                                      const std::vector<TBlob> &outputs) {
  mshadow::Stream<gpu> *s = ctx.get_stream<gpu>();
  cudaStream_t stream = mshadow::Stream<gpu>::GetStream(s);

  MSHADOW_TYPE_SWITCH(inputs[0].type_flag_, DType, {
    const index_t size = inputs[0].Size();
    if (req[0] != kNullOp || req[1] != kNullOp) {
      MXNET_ASSIGN_REQ_SWITCH(req[0], lreq, {
        MXNET_ASSIGN_REQ_SWITCH(req[1], rreq, {
          using LType = uint4;
          using Kernel = VectorizedBinaryBwdUseNone<DType, LOP, ROP, lreq, rreq>;

          typename Kernel::ParamType params;
          params.inputs[0] = inputs[0].dptr<DType>();
          params.outputs[0] = outputs[0].dptr<DType>();
          params.outputs[1] = outputs[1].dptr<DType>();

          VectorizedKernelLauncher<DType, LType, Kernel>(size, s, params);
        });
      });
    }
  });
}

}  // namespace

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_USE_CUDA
#endif  // MXNET_OPERATOR_TENSOR_ELEMWISE_OP_CUH_
