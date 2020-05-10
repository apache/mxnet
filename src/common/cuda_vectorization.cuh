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

/* \brief Helper class that enables storing multiple values of type DType
          as 1 value of type LType.
*/
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

/* \brief Helper class that enables accessing multiple values of type DType
          as 1 value of type LType. Additional aligned template argument
          allows performance optimizations if the pointer and the size of
          the allocation is aligned to sizeof(LType) / sizeof(DType) elements.
*/
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

  MSHADOW_XINLINE VectorizedAccessor(DType* ptr, const index_t size) {
    unaligned_ptr_ = ptr;
    if (aligned) {
      alignment_ = 0;
      aligned_ptr_ = reinterpret_cast<LType*>(ptr);
      n_elems_ = (size + storage_.nvec - 1) / storage_.nvec;
    } else {
      size_t ptr_as_number = reinterpret_cast<size_t>(ptr);
      alignment_ = (ptr_as_number % sizeof(LType)) / sizeof(DType);
      aligned_ptr_ = reinterpret_cast<LType*>(ptr - alignment_);
      n_elems_ = (size + alignment_ + storage_.nvec - 1) / storage_.nvec;
    }
  }

  /* \brief Alignment of the input pointer in elements. */
  MSHADOW_XINLINE int alignment() const {
    return alignment_;
  }

  /* \brief Access to separate elements. */
  MSHADOW_XINLINE DType* separate() {
    return storage_.scratch_.separate;
  }

  /* \brief Number of elements stored. */
  MSHADOW_XINLINE constexpr int nvec() const {
    return storage_.nvec;
  }

  /* \brief Number of aligned elements that span the entire input tensor. */
  MSHADOW_XINLINE index_t num_aligned_elements() const {
    return n_elems_;
  }

  /* \brief Load values from the input.
     \param id Aligned index of the element.
     \param N size of the tensor.
  */
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

/* \brief Class used for vectorized read-only access. */
template <typename DType, typename LType, bool aligned = false>
class VectorizedLoader : public VectorizedAccessor<const DType, const LType, aligned> {
 public:
  MSHADOW_XINLINE VectorizedLoader(const DType* ptr, const index_t N) :
    VectorizedAccessor<const DType, const LType, aligned>(ptr, N) {
  }
};

/* \brief Class used for vectorized writable access. */
template <typename DType, typename LType, bool aligned = false>
class VectorizedStorer : public VectorizedAccessor<DType, LType, aligned> {
 public:
  MSHADOW_XINLINE VectorizedStorer(DType* ptr, const index_t N) :
    VectorizedAccessor<DType, LType, aligned>(ptr, N) {
  }

  /* \brief Store values to the output.
     \param id Aligned index of the element.
     \param N size of the tensor.
  */
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
  SAME_ALIGNED,  // All tensors aligned
  SAME_UNALIGNED,  // All tensors have the same misalignment
  DIFFERENT  // Tensors have different alignment
};

template <typename LType, typename DType>
int CalcAlignment(const DType* ptr) {
  size_t ptr_as_number = reinterpret_cast<size_t>(ptr);
  return ptr_as_number % sizeof(LType);
}

/* \brief Check alignment of the inputs and outputs when cast to LType*.
   \param params Structuce containing arrays with inputs' and outputs' pointers
   \param lead_dim Leading dimension of the tensors.
   \param other_dim The size of the other dimensions of the tensors.
*/
template <typename LType, typename DType, typename Params>
Alignment CheckAlignment(const Params& params, const index_t lead_dim, const index_t other_dim) {
  int align = -1;
  constexpr int nvec = sizeof(LType) / sizeof(DType);

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

  if ((other_dim != 1) &&
      (lead_dim % nvec != 0)) {
    return Alignment::DIFFERENT;
  }

  if ((align == 0) &&
      (lead_dim % nvec == 0)) {
    return Alignment::SAME_ALIGNED;
  } else {
    return Alignment::SAME_UNALIGNED;
  }
}

constexpr int vectorized_kernel_thread_num = 512;

}  // namespace

/* \brief Helper launcher function for the vectorized kernels. Checks for alignment of the
          input and output tensors and launches a proper template.
   \param lead_dim Leading dimension of the tensors.
   \param other_dim The size of the other dimensions.
   \param s Stream which should be used for launching the kernel.
   \param params Input parameters to the kernel. Needs to contain at least 2 arrays of DType*:
                 inputs and outputs, which contain input and output pointers.
*/
template <typename DType, typename LType, typename Kernel>
void VectorizedKernelLauncher(const index_t lead_dim,
                              const index_t other_dim,
                              mshadow::Stream<gpu>* s,
                              typename Kernel::ParamType params) {
  static_assert(sizeof(LType) >= sizeof(DType), "Load type is smaller than operand type");
  if (lead_dim * other_dim != 0) {
    cudaStream_t stream = mshadow::Stream<gpu>::GetStream(s);
    VectorizedLoader<DType, LType> l(params.inputs[0], lead_dim);
    size_t num_elements = other_dim * l.num_aligned_elements();
    constexpr int threads = vectorized_kernel_thread_num;
    constexpr int max_blocks = 65535;
    index_t blocks = std::min(static_cast<int>((num_elements + threads - 1) / threads),
                              max_blocks);
    auto align = CheckAlignment<LType, DType>(params, lead_dim, other_dim);
    switch (align) {
      case Alignment::SAME_ALIGNED:
        Kernel::template Launch<true, LType>(blocks, threads, stream, params, lead_dim, other_dim);
        break;
      case Alignment::SAME_UNALIGNED:
        Kernel::template Launch<false, LType>(blocks, threads, stream, params, lead_dim, other_dim);
        break;
      case Alignment::DIFFERENT: {
        const index_t size = lead_dim * other_dim;
        index_t blocks = std::min(static_cast<int>((size + threads - 1) /
                                                   threads),
                                  max_blocks);
        // If the pointers are aligned differently we cannot vectorize
        Kernel::template Launch<true, DType>(blocks, threads, stream, params, lead_dim, other_dim);
        break;
      }
    }
  }
}

}  // namespace cuda
}  // namespace common
}  // namespace mxnet

#endif  // MXNET_USE_CUDA && __CUDACC__

#endif  // MXNET_COMMON_CUDA_VECTORIZATION_CUH_
