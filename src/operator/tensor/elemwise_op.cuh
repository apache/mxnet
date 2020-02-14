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

template <typename LType>
MSHADOW_XINLINE void ldg(LType* dst, const LType* src) {
    *dst = *src;
}

template <>
MSHADOW_XINLINE void ldg(double* dst, const double* src) {
    double temp;
    asm volatile ("ld.global.f64 %0, [%1];" :
                  "=d"(temp) :
                  "l"(src));
    *dst = temp;
}

/*template <>*/
/*MSHADOW_XINLINE void ldg(uint64_t* dst, const uint64_t* src) {*/
    /*uint64_t temp;*/
    /*asm volatile ("ld.global.u64 %0, [%1];" :*/
                  /*"=l"(temp) :*/
                  /*"l"(src));*/
    /**dst = temp;*/
/*}*/

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
      ldg<typename std::remove_const<LType>::type>(&(storage_.scratch_.aligned),
                                                   aligned_ptr_ + id);
    } else {
      if (id > 0 && id < n_elems_ - 1) {
        ldg<typename std::remove_const<LType>::type>(&(storage_.scratch_.aligned),
                                                     aligned_ptr_ + id);
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

template <bool aligned, typename DType, typename LType, typename OP, int req>
__global__ void VectorizedElementwiseKernel(DType* output, const DType* input0, const DType* input1, index_t N) {
  VectorizedLoader<DType, LType, aligned> loader0(input0, N);
  VectorizedLoader<DType, LType, aligned> loader1(input1, N);
  VectorizedStorer<DType, LType, aligned> storer(output, N);

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

template <typename LType, typename DType>
Alignment CheckAlignment(const std::vector<DType*>& pointers) {
  int align = -1;
  for (const DType* ptr : pointers) {
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

size_t minthree(const size_t a, const size_t b, const size_t c) {
  return a < b ? (a < c ? a : c) : (b < c ? b : c);
}

}  // namespace

template<typename OP>
void VectorizedCompute(const nnvm::NodeAttrs &attrs,
                       const OpContext &ctx,
                       const std::vector<TBlob> &inputs,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &outputs) {
  using namespace mxnet_op;
  if (req[0] == kNullOp) return;
  Stream<gpu> *s = ctx.get_stream<gpu>();
  CHECK_EQ(inputs.size(), 2U);
  CHECK_EQ(outputs.size(), 1U);
  MXNET_ASSIGN_REQ_SWITCH(req[0], Req, {
      if (dmlc::GetEnv("DEBUG_VECTOR", false)) {
        MSHADOW_TYPE_SWITCH_WITH_HALF2(outputs[0].type_flag_, DType, {
          const size_t size = (minthree(outputs[0].Size(), inputs[0].Size(), inputs[1].Size())
          + DataType<DType>::kLanes - 1) / DataType<DType>::kLanes;
          if (size != 0) {
            Kernel<mxnet_op::op_with_req<OP, Req>, gpu>::Launch(s, size,
            outputs[0].dptr<DType>(),
            inputs[0].dptr<DType>(), inputs[1].dptr<DType>());
          }
        });
      } else {
        MSHADOW_TYPE_SWITCH(outputs[0].type_flag_, DType, {
          using LType = uint4;
          static_assert(sizeof(LType) >= sizeof(DType), "Load type is smaller than operand type");
          if (outputs[0].Size() != 0) {
            cudaStream_t stream = mshadow::Stream<gpu>::GetStream(s);
            constexpr int nvec = sizeof(LType) / sizeof(DType);
            VectorizedLoader<DType, LType> l(outputs[0].dptr<DType>(), outputs[0].Size());
            size_t num_elements = l.num_aligned_elements();
            constexpr int threads = 512;
            index_t blocks = std::min(static_cast<int>((num_elements + threads - 1) / threads),
                                      65535);
            auto align = CheckAlignment<LType, DType>({outputs[0].dptr<DType>(),
                                                       inputs[0].dptr<DType>(),
                                                       inputs[1].dptr<DType>()});
            if (align == Alignment::SAME_ALIGNED && (outputs[0].Size() % nvec == 0)) {
              VectorizedElementwiseKernel<true, DType, LType, OP, Req>
                <<<blocks, threads, 0, stream>>>(outputs[0].dptr<DType>(),
                                                 inputs[0].dptr<DType>(),
                                                 inputs[1].dptr<DType>(),
                                                 outputs[0].Size());
            } else {
              if (align != Alignment::DIFFERENT) {
                VectorizedElementwiseKernel<false, DType, LType, OP, Req>
                  <<<blocks, threads, 0, stream>>>(outputs[0].dptr<DType>(),
                                                   inputs[0].dptr<DType>(),
                                                   inputs[1].dptr<DType>(),
                                                   outputs[0].Size());
              } else {
                index_t blocks = std::min(static_cast<int>((outputs[0].Size() + threads - 1) /
                                                           threads),
                                          65535);
                // If the pointers are aligned differently we cannot vectorize
                VectorizedElementwiseKernel<true, DType, DType, OP, Req>
                  <<<blocks, threads, 0, stream>>>(outputs[0].dptr<DType>(),
                                                   inputs[0].dptr<DType>(),
                                                   inputs[1].dptr<DType>(),
                                                   outputs[0].Size());
              }
            }
          }
        });
      }
  });
}

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_USE_CUDA
#endif  // MXNET_OPERATOR_TENSOR_ELEMWISE_OP_CUH_
