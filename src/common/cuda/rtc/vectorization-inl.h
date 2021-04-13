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

#ifndef MXNET_COMMON_CUDA_RTC_VECTORIZATION_INL_H_
#define MXNET_COMMON_CUDA_RTC_VECTORIZATION_INL_H_

#include <mxnet/base.h>

#if MXNET_USE_CUDA

#include <sstream>
#include <string>
#include <vector>
#include <algorithm>

#include "../rtc.h"
#include "../../utils.h"

namespace mxnet {
namespace common {
namespace cuda {
namespace rtc {

const char vectorization_support_string[] = R"code(

namespace vector {

template <int size>
struct VectorType {
    static_assert(size <= 32, "VectorType needs to have size of at most 32B");
};

template <>
struct VectorType<1> {
  using type = char;
};

template <>
struct VectorType<2> {
  using type = short;
};


template <>
struct VectorType<4> {
  using type = int;
};

template <>
struct VectorType<8> {
  using type = long long;
};

template <>
struct VectorType<16> {
  using type = ulonglong2;
};

template <>
struct VectorType<32> {
  using type = ulonglong4;
};

template <typename DType>
__device__ inline DType add_elem(const DType& x, const DType& y) {
  return x + y;
}

template <>
__device__ inline half add_elem(const half& x, const half& y) {
  return __float2half(__half2float(x) + __half2float(y));
}

/* \brief Helper class that enables storing multiple values of type DType
          as 1 value of type LType.
*/
template <typename DType, int n>
class VectorizedStorage {
 public:
  using LType = typename VectorType<sizeof(DType) * n>::type;
  constexpr static int nvec = n;
  union vectorized_storage {
    LType aligned;
    DType separate[nvec];  // NOLINT(*)

    inline __device__ vectorized_storage() {}
    inline __device__ ~vectorized_storage() {}
  } scratch_;

  inline __device__ VectorizedStorage() {}
  inline __device__ VectorizedStorage (const VectorizedStorage<DType, n>& y2) {
      scratch_.aligned = y2.scratch_.aligned;
  }
  inline __device__ VectorizedStorage (const LType &y2) {
      scratch_.aligned = y2;
  }
  inline __device__ VectorizedStorage<DType, n>& operator+=(
      const VectorizedStorage<DType, n>& rhs) {
    #pragma unroll
    for (int i = 0; i < nvec; ++i) {
      scratch_.separate[i] = add_elem(scratch_.separate[i], rhs.scratch_.separate[i]);
    }
    return *this;
  }
  inline __device__ ~VectorizedStorage() {}
};

// Returns const LType is DType is const
template <typename DType, typename LType>
struct select_const {
  using type = LType;
};

template <typename DType, typename LType>
struct select_const<const DType, LType> {
  using type = const LType;
};

template <typename DType>
struct remove_const {
  using type = DType;
};

template <typename DType>
struct remove_const<const DType> {
  using type = DType;
};


/* \brief Helper class that enables accessing multiple values of type DType
          as 1 value of type LType. Additional aligned template argument
          allows performance optimizations if the pointer and the size of
          the allocation is aligned to sizeof(LType) / sizeof(DType) elements.
*/
template <typename DType, int nvec, bool aligned = false>
class VectorizedAccessor {
 public:
  using StorageType = VectorizedStorage<typename remove_const<DType>::type,
                                        nvec>;
  using LType = typename select_const<DType, typename StorageType::LType>::type;
  StorageType storage_;

  LType* aligned_ptr_;
  DType* unaligned_ptr_;
  int alignment_;
  index_t n_elems_;

  inline __device__ VectorizedAccessor(DType* const ptr, const index_t size) {
    unaligned_ptr_ = ptr;
    if (aligned) {
      alignment_ = 0;
      aligned_ptr_ = reinterpret_cast<LType*>(ptr);
      n_elems_ = (size + nvec- 1) / nvec;
    } else {
      size_t ptr_as_number = reinterpret_cast<size_t>(ptr);
      alignment_ = (ptr_as_number % sizeof(LType)) / sizeof(DType);
      aligned_ptr_ = reinterpret_cast<LType*>(ptr - alignment_);
      n_elems_ = (size + alignment_ + nvec - 1) / nvec;
    }
  }

  /* \brief Alignment of the input pointer in elements. */
  inline __device__ int alignment() const {
    return alignment_;
  }

  /* \brief Access to separate elements. */
  inline __device__ DType* separate() {
    return storage_.scratch_.separate;
  }

  /* \brief Number of aligned elements that span the entire input tensor. */
  inline __device__ index_t num_aligned_elements() const {
    return n_elems_;
  }

  /* \brief Load values from the input.
     \param id Aligned index of the element.
     \param N size of the tensor.
  */
  inline __device__ void load(const index_t id, const index_t N) {
    if (aligned) {
      storage_.scratch_.aligned = aligned_ptr_[id];
    } else {
      if (id > 0 && id < n_elems_ - 1) {
        storage_.scratch_.aligned = aligned_ptr_[id];
      } else {
#pragma unroll
        for (int j = 0; j < nvec; ++j) {
          DType* ptr = reinterpret_cast<DType*>(&(aligned_ptr_[id])) + j;
          if (reinterpret_cast<size_t>(ptr) >= reinterpret_cast<size_t>(unaligned_ptr_) &&
              reinterpret_cast<size_t>(ptr) < reinterpret_cast<size_t>(unaligned_ptr_ + N)) {
            storage_.scratch_.separate[j] = *ptr;
          } else {
            storage_.scratch_.separate[j] = DType();
          }
        }
      }
    }
  }
};

/* \brief Class used for vectorized read-only access. */
template <typename DType, int nvec, bool aligned = false>
class VectorizedLoader : public VectorizedAccessor<const DType, nvec, aligned> {
 public:
  inline __device__ VectorizedLoader(const DType* ptr, const index_t N) :
    VectorizedAccessor<const DType, nvec, aligned>(ptr, N) {
  }
};

/* \brief Class used for vectorized writable access. */
template <typename DType, int nvec, bool aligned = false>
class VectorizedStorer : public VectorizedAccessor<DType, nvec, aligned> {
 public:
  inline __device__ VectorizedStorer(DType* ptr, const index_t N) :
    VectorizedAccessor<DType, nvec, aligned>(ptr, N) {
  }

  /* \brief Store values to the output.
     \param id Aligned index of the element.
     \param N size of the tensor.
  */
  inline __device__ void store(const index_t id, const index_t N) {
    if (aligned) {
      this->aligned_ptr_[id] = this->storage_.scratch_.aligned;
    } else {
      if (id > 0 && id < this->n_elems_ - 1) {
        this->aligned_ptr_[id] = this->storage_.scratch_.aligned;
      } else {
#pragma unroll
        for (int j = 0; j < nvec; ++j) {
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

}  // namespace vector

)code";

namespace {

inline index_t get_num_aligned_elements(const void *ptr, const index_t lead_dim,
                                        const int nvec, const int size) {
  size_t ptr_as_number = reinterpret_cast<size_t>(ptr);
  int alignment = (ptr_as_number % (nvec * size)) / size;
  return (lead_dim + alignment + nvec - 1) / nvec;
}

enum class Alignment {
  SAME_ALIGNED,  // All tensors aligned
  SAME_UNALIGNED,  // All tensors have the same misalignment
  DIFFERENT  // Tensors have different alignment
};

inline int CalcAlignment(const void *ptr, const int size) {
  size_t ptr_as_number = reinterpret_cast<size_t>(ptr);
  return ptr_as_number % size;
}

/* \brief Check alignment of the inputs and outputs when using vectorized accesses.
   \param params Structure containing arrays with inputs' and outputs' pointers
   \param lead_dim Leading dimension of the tensors.
   \param other_dim The size of the other dimensions of the tensors.
   \param nvec Length of the vector.
   \param inputs Inputs to the operator.
   \param outputs Outputs of the operator.
*/
template <typename Params>
Alignment CheckAlignment(const Params& params, const index_t lead_dim,
                         const index_t other_dim, const int nvec,
                         const std::vector<TBlob> &inputs,
                         const std::vector<TBlob> &outputs) {
  using namespace common;
  int align = -1;

  size_t i = 0;
  for (const void *ptr : params.inputs) {
    if (ptr != nullptr) {
      int new_align = CalcAlignment(ptr,
                                    mshadow_type_info(inputs[i].type_flag_).size * nvec);
      if (align == -1) {
        align = new_align;
      } else {
        if (align != new_align) {
          return Alignment::DIFFERENT;
        }
      }
    }
    ++i;
  }

  i = 0;
  for (const void *ptr : params.outputs) {
    if (ptr != nullptr) {
      int new_align = CalcAlignment(ptr,
                                    mshadow_type_info(outputs[i].type_flag_).size * nvec);
      if (align == -1) {
        align = new_align;
      } else {
        if (align != new_align) {
          return Alignment::DIFFERENT;
        }
      }
    }
    ++i;
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

/*! \brief Launcher helper for the kernels using vectorization.
 *  \param parameters of the kernel (e.g. values of the template arguments)
 *  \param kernel_name name of the kernel
 *  \param code used for compilation of the kernel if not found in cache
 *  \param nvec length of the vector used for loading/storing data
 *  \param lead_dim size of leading dimension of the tensors
 *  \param other_dim maximum of the total size of all the other dimensions of the tensors
 *  \param s stream used to launch the kernel
 *  \param inputs to the kernel
 *  \param outputs of the kernel
 *  \param dev_id id of the devide which the kernel will be launched on
 *  \param lead_input_num number of input to use for checking alignment
 *                        (in case only a subset of inputs is used vectorized).
 *                        Default is 0.
 */
template <typename Params>
void VectorizedKernelRTCLauncher(const std::string &parameters,
                                 const std::string &kernel_name,
                                 const std::string &code,
                                 int nvec,
                                 const index_t lead_dim,
                                 const index_t other_dim,
                                 mshadow::Stream<gpu> *s,
                                 const Params params,
                                 const std::vector<TBlob> &inputs,
                                 const std::vector<TBlob> &outputs,
                                 const int dev_id,
                                 const int lead_input_num = 0) {
  const index_t N = lead_dim * other_dim;
  nvec = std::min(nvec, 4);  // Use at most 4-wide vectors
  if (N != 0) {
    auto align = CheckAlignment(params, lead_dim, other_dim,
                                nvec, inputs, outputs);
    std::string kernel_builder;
    kernel_builder.reserve(2560);

    // Fill input types
    int counter = 0;
    for (const auto& input : inputs) {
      const auto& type_info = common::mshadow_type_info(input.type_flag_);
      kernel_builder += "using InputType";
      kernel_builder += std::to_string(counter);
      kernel_builder += " = ";
      kernel_builder += type_info.name;
      kernel_builder += ";\n";
      ++counter;
    }

    // Fill output types
    counter = 0;
    for (const auto& output : outputs) {
      const auto& type_info = common::mshadow_type_info(output.type_flag_);
      kernel_builder += "using OutputType";
      kernel_builder += std::to_string(counter);
      kernel_builder += " = ";
      kernel_builder += type_info.name;
      kernel_builder += ";\n";
      ++counter;
    }

    switch (align) {
      case Alignment::SAME_ALIGNED:
        kernel_builder += "const bool aligned = true;\n"
                          "const int nvec = ";
        kernel_builder += std::to_string(nvec);
        kernel_builder += ";\n";
        break;
      case Alignment::SAME_UNALIGNED:
        kernel_builder += "const bool aligned = false;\n"
                          "const int nvec = ";
        kernel_builder += std::to_string(nvec);
        kernel_builder += ";\n";
        break;
      case Alignment::DIFFERENT: {
        // If the pointers are aligned differently we cannot vectorize
        kernel_builder += "const bool aligned = true;\n"
                          "const int nvec = 1;\n";
        nvec = 1;
        break;
      }
    }

    kernel_builder += parameters;

    index_t num_aligned_elements = get_num_aligned_elements(
                                    params.inputs[lead_input_num],
                                    lead_dim, nvec,
                                    common::mshadow_type_info(
                                      inputs[lead_input_num].type_flag_).size);
    size_t num_elements = other_dim * num_aligned_elements;
    constexpr int threads = vectorized_kernel_thread_num;
    constexpr int max_blocks = 65535;
    index_t blocks = std::min(static_cast<int>((num_elements + threads - 1) / threads),
                              max_blocks);
    std::vector<const void*> args = {&params, &lead_dim, &other_dim,
                                     &N, &num_aligned_elements};
    auto function = common::cuda::rtc::get_function(kernel_builder,
                                                    kernel_name,
                                                    code,
                                                    dev_id);

    common::cuda::rtc::launch(function,
                              {static_cast<unsigned int>(blocks), 1, 1},
                              {static_cast<unsigned int>(threads), 1, 1},
                              0, s, &args);
  }
}


}  // namespace rtc
}  // namespace cuda
}  // namespace common
}  // namespace mxnet

#endif  // MXNET_USE_CUDA

#endif  // MXNET_COMMON_CUDA_RTC_VECTORIZATION_INL_H_
