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
 * \file cuda_vectorization.cuh
 * \brief GPU helpers for vectorized memory accesses
 */

#ifndef MXNET_COMMON_CUDA_VECTORIZATION_CUH_
#define MXNET_COMMON_CUDA_VECTORIZATION_CUH_

#if MXNET_USE_CUDA && __CUDACC__

#include <cuda_runtime.h>
#include "cuda_utils.h"


namespace mxnet {
namespace common {
namespace cuda {

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
  using StorageType = VectorizedStorage<typename std::remove_const<DType>::type,
                                        typename std::remove_const<LType>::type>;
  StorageType storage_;

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

  MSHADOW_XINLINE DType* separate() {
    return storage_.scratch_.separate;
  }

  MSHADOW_XINLINE constexpr int nvec() const {
    return storage_.nvec;
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

template <typename LType, typename DType, typename Params>
Alignment CheckAlignment(const Params& params) {
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

constexpr int vectorized_kernel_thread_num = 512;

}  // namespace

template <typename DType, typename LType, typename Kernel>
void VectorizedKernelLauncher(const index_t size, mshadow::Stream<gpu>* s, typename Kernel::ParamType params) {
  static_assert(sizeof(LType) >= sizeof(DType), "Load type is smaller than operand type");
  if (size != 0) {
    cudaStream_t stream = mshadow::Stream<gpu>::GetStream(s);
    constexpr int nvec = sizeof(LType) / sizeof(DType);
    VectorizedLoader<DType, LType> l(params.inputs[0], size);
    size_t num_elements = l.num_aligned_elements();
    constexpr int threads = vectorized_kernel_thread_num;
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

}  // namespace cuda
}  // namespace common
}  // namespace mxnet

#endif  // MXNET_USE_CUDA && __CUDACC__

#endif  // MXNET_COMMON_CUDA_VECTORIZATION_CUH_
