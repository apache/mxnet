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
 * \file test_util.h
 * \brief unit test performance analysis functions
 * \author Chris Olivier
 */
#ifndef TEST_UTIL_H_
#define TEST_UTIL_H_

#include <gtest/gtest.h>
#include <mxnet/storage.h>
#include <mxnet/ndarray.h>
#include <string>
#include <vector>
#include <sstream>
#include <random>

#include "../../../src/ndarray/ndarray_function.h"

#if MXNET_USE_VTUNE
#include <ittnotify.h>
#endif

namespace mxnet {
namespace test {

extern bool unitTestsWithCuda;
extern bool debug_output;
extern bool quick_test;
extern bool performance_run;
extern bool csv;
extern bool thread_safety_force_cpu;

template <typename DType>
inline size_t shapeMemorySize(const mxnet::TShape& shape) {
  return shape.Size() * sizeof(DType);
}

class BlobMemory {
 public:
  explicit inline BlobMemory(const bool isGPU) : isGPU_(isGPU) {
    this->handle_.dptr = nullptr;
  }
  inline ~BlobMemory() {
    Free();
  }
  void* Alloc(const size_t size) {
    CHECK_GT(size, 0U);  // You've probably made a mistake
    mxnet::Context context = isGPU_ ? mxnet::Context::GPU(0) : mxnet::Context{};
    Storage* storage       = mxnet::Storage::Get();
    handle_                = storage->Alloc(size, context);
    return handle_.dptr;
  }
  void Free() {
    mxnet::Storage::Get()->DirectFree(handle_);
    handle_.dptr = nullptr;
    handle_.size = 0;
  }
  size_t Size() const {
    return handle_.size;
  }

 private:
  const bool isGPU_;
  Storage::Handle handle_;
};

class StandaloneBlob : public TBlob {
 public:
  inline StandaloneBlob(const mxnet::TShape& shape, const bool isGPU, const int dtype)
      : TBlob(nullptr, shape, isGPU ? gpu::kDevMask : cpu::kDevMask, dtype),
        memory_(std::make_shared<BlobMemory>(isGPU)) {
    MSHADOW_TYPE_SWITCH(
        dtype, DType, { this->dptr_ = memory_->Alloc(shapeMemorySize<DType>(shape)); });
  }
  inline ~StandaloneBlob() {
    this->dptr_ = nullptr;
  }
  inline size_t MemorySize() const {
    return memory_->Size();
  }

 private:
  /*! \brief Locally allocated memory block for this blob */
  std::shared_ptr<BlobMemory> memory_;
};

/*!
 * \brief Access a TBlob's data on the CPU within the scope of this object
 * Overloaded () operator returns the CPU-bound TBlob
 * RAII will copy the data back to the GPU (if it was a GPU blob)
 */
class CAccessAsCPU {
 public:
  CAccessAsCPU(const RunContext& run_ctx, const TBlob& src, bool copy_back_result = true)
      : run_ctx_(run_ctx), src_(src), copy_back_result_(copy_back_result) {
#if MXNET_USE_CUDA
    if (run_ctx.ctx.dev_type == Context::kCPU) {
      blob_ = src;
    } else {
      Context cpu_ctx, gpu_ctx = run_ctx.ctx;
      cpu_ctx.dev_type = Context::kCPU;
      cpu_ctx.dev_id   = 0;
      NDArray on_cpu(src.shape_, cpu_ctx, false, src_.type_flag_);
      on_cpu.CheckAndAlloc();
      blob_ = on_cpu.data();
      run_ctx.get_stream<gpu>()->Wait();
      mxnet::ndarray::Copy<gpu, cpu>(src, &blob_, cpu_ctx, gpu_ctx, run_ctx);
      run_ctx.get_stream<gpu>()->Wait();
      on_cpu_ = on_cpu;
    }
#else
    blob_ = src;
#endif
  }
  ~CAccessAsCPU() {
#if MXNET_USE_CUDA
    if (copy_back_result_) {
      // Copy back from GPU to CPU
      if (run_ctx_.ctx.dev_type == Context::kGPU) {
        Context cpu_ctx, gpu_ctx = run_ctx_.ctx;
        cpu_ctx.dev_type = Context::kCPU;
        cpu_ctx.dev_id   = 0;
        run_ctx_.get_stream<gpu>()->Wait();
        mxnet::ndarray::Copy<cpu, gpu>(blob_, &src_, gpu_ctx, cpu_ctx, run_ctx_);
        run_ctx_.get_stream<gpu>()->Wait();
      }
    }
#endif
  }
  inline const TBlob& operator()() const {
    return blob_;
  }

 private:
  const RunContext run_ctx_;
  TBlob src_;
  const bool copy_back_result_;
  NDArray on_cpu_;
  TBlob blob_;
};

/*!
 * \brief Access data blob as if on the CPU via a callback
 * \tparam Type of callback Function to call with CPU-data NDArray
 * \param src Source NDArray (on GPU or CPU)
 * \param run_ctx Run context
 * \param cb Callback Function to call with CPU-data NDArray
 */
template <typename CallbackFunction>
inline void AccessAsCPU(const NDArray& src, const RunContext& run_ctx, CallbackFunction cb) {
#if MXNET_USE_CUDA
  if (src.ctx().dev_type == Context::kCPU) {
    cb(src);
  } else {
    Context cpu_ctx, gpu_ctx = src.ctx();
    cpu_ctx.dev_type = Context::kCPU;
    cpu_ctx.dev_id   = 0;
    NDArray on_cpu(src.shape(), cpu_ctx, false, src.dtype());
    on_cpu.CheckAndAlloc();
    TBlob tmp1 = on_cpu.data();
    run_ctx.get_stream<gpu>()->Wait();
    mxnet::ndarray::Copy<gpu, cpu>(src.data(), &tmp1, cpu_ctx, gpu_ctx, run_ctx);
    run_ctx.get_stream<gpu>()->Wait();
    cb(on_cpu);
    TBlob tmp2 = src.data();
    mxnet::ndarray::Copy<cpu, gpu>(on_cpu.data(), &tmp2, gpu_ctx, cpu_ctx, run_ctx);
    run_ctx.get_stream<gpu>()->Wait();
  }
#else
  cb(src);
#endif
}

/*!
 * \brief Access data blob as if on the CPU via a callback
 * \tparam Type of callback Function to call with CPU-data NDArray
 * \param src Source TBlob (on GPU or CPU)
 * \param run_ctx Run context
 * \param cb Callback Function to call with CPU-data TBlob
 */
template <typename CallbackFunction>
inline void AccessAsCPU(const TBlob& src, const RunContext& run_ctx, CallbackFunction cb) {
#if MXNET_USE_CUDA
  if (run_ctx.ctx.dev_type == Context::kCPU) {
    cb(src);
  } else {
    cb(CAccessAsCPU(run_ctx, src, true)());
  }
#else
  cb(src);
#endif
}

constexpr const size_t MPRINT_PRECISION = 5;
template <typename DType>
inline void fill(const RunContext& run_ctx, const TBlob& _blob, const DType val) {
  AccessAsCPU(_blob, run_ctx, [val](const TBlob& blob) {
    MSHADOW_TYPE_SWITCH(blob.type_flag_, DTypeX, {
      DTypeX* p1 = blob.dptr<DTypeX>();
      for (size_t i = 0, n = blob.Size(); i < n; ++i) {
        *p1++ = val;
      }
    });
  });
}

template <typename DType>
inline void try_fill(const RunContext& run_ctx, const TBlob* blob, const DType val) {
  if (blob) {
    fill(run_ctx, *blob, val);
  }
}

template <typename DType, typename Stream>
inline void dump(Stream* os, const TBlob& blob, const char* suffix = "f") {
  DType* p1 = blob.dptr<DType>();
  for (size_t i = 0, n = blob.Size(); i < n; ++i) {
    if (i) {
      *os << ", ";
    }
    const DType val = *p1++;

    std::stringstream stream;
    stream << val;
    std::string ss = stream.str();
    if (suffix && *suffix == 'f') {
      if (std::find(ss.begin(), ss.end(), '.') == ss.end()) {
        ss += ".0";
      }
    }
    *os << ss << suffix;
  }
}

/*! \brief Return reference to data at position indexes */
inline index_t getMult(const mxnet::TShape& shape, const index_t axis) {
  return axis < shape.ndim() ? shape[axis] : 1;
}

/*! \brief offset, given indices such as bn, channel, depth, row, column */
inline index_t offset(const mxnet::TShape& shape, const std::vector<size_t>& indices) {
  const size_t dim = shape.ndim();
  CHECK_LE(indices.size(), dim);
  size_t offset = 0;
  for (size_t i = 0; i < dim; ++i) {
    offset *= shape[i];
    if (indices.size() > i) {
      CHECK_LT(indices[i], shape[i]);
      offset += indices[i];
    }
  }
  return offset;
}

/*! \brief Return reference to data at position indexes */
template <typename DType>
inline const DType& data_at(const TBlob* blob, const std::vector<size_t>& indices) {
  return blob->dptr<DType>()[offset(blob->shape_, indices)];
}

/*! \brief Set data at position indexes */
template <typename DType>
inline DType& data_ref(const TBlob* blob, const std::vector<size_t>& indices) {
  return blob->dptr<DType>()[offset(blob->shape_, indices)];
}

inline std::string repeatedStr(const char* s,
                               const signed int count,
                               const bool trailSpace = false) {
  if (count <= 0) {
    return std::string();
  } else if (count == 1) {
    std::stringstream str;
    str << s << " ";
    return str.str();
  } else {
    std::stringstream str;
    for (int x = 0; x < count; ++x) {
      str << s;
    }
    if (trailSpace) {
      str << " ";
    }
    return str.str();
  }
}

/*! \brief Pretty print a shape with optional label */
template <typename StreamType>
inline StreamType& print_shape(StreamType* _os,
                               const std::string& label,
                               const mxnet::TShape& shape,
                               const bool add_endl = true) {
  if (!label.empty()) {
    *_os << label << ": ";
  }
  *_os << "(";
  for (size_t i = 0, n = shape.ndim(); i < n; ++i) {
    if (i) {
      *_os << ", ";
    }
    *_os << shape[i];
  }
  *_os << ")";
  if (add_endl) {
    *_os << std::endl;
  } else {
    *_os << " ";
  }
  return *_os << std::flush;
}

/*! \brief Pretty print a 1D, 2D, or 3D blob */
template <typename DType, typename StreamType>
inline StreamType& print_blob_(const RunContext& ctx,
                               StreamType* _os,
                               const TBlob& blob,
                               const bool doChannels = true,
                               const bool doBatches  = true,
                               const bool add_endl   = true) {
#if MXNET_USE_CUDA
  if (blob.dev_mask() == gpu::kDevMask) {
    return print_blob_<DType>(
        ctx, _os, CAccessAsCPU(ctx, blob, false)(), doChannels, doBatches, add_endl);
  }
#endif  // MXNET_USE_CUDA

  StreamType& os   = *_os;
  const size_t dim = static_cast<size_t>(blob.ndim());

  if (dim == 1) {
    // probably a 1d tensor (mshadow::Tensor is deprecated)
    TBlob changed(blob.dptr<DType>(), mxnet::TShape(3, -1), blob.dev_mask(), blob.dev_id());
    changed.shape_[0] = 1;
    changed.shape_[1] = 1;
    changed.shape_[2] = blob.shape_[0];
    return print_blob_<DType>(ctx, &os, changed, false, false, add_endl);
  } else if (dim == 2) {
    // probably a 2d tensor (mshadow::Tensor is deprecated)
    TBlob changed(blob.dptr<DType>(), mxnet::TShape(4, -1), blob.dev_mask(), blob.dev_id());
    changed.shape_[0] = 1;
    changed.shape_[1] = 1;
    changed.shape_[2] = blob.shape_[0];
    changed.shape_[3] = blob.shape_[1];
    return print_blob_<DType>(ctx, &os, changed, false, false, add_endl);
  }
  CHECK_GE(dim, 3U) << "Invalid dimension zero (0)";

  const size_t batchSize = blob.size(0);

  size_t channels = 1;
  size_t depth    = 1;
  size_t height   = 1;
  size_t width    = 1;
  if (dim > 1) {
    channels = blob.size(1);
    if (dim > 2) {
      if (dim == 3) {
        width = blob.size(2);
      } else if (dim == 4) {
        height = blob.size(2);
        width  = blob.size(3);
      } else {
        depth = blob.size(2);
        if (dim > 3) {
          height = blob.size(3);
          if (dim > 4) {
            width = blob.size(4);
          }
        }
      }
    }
  }

  for (size_t r = 0; r < height; ++r) {
    for (size_t thisBatch = 0; thisBatch < batchSize; ++thisBatch) {
      if (doBatches) {
        std::stringstream ss;
        if (doBatches && !thisBatch) {
          os << "|";
        }
        ss << "N" << thisBatch << "| ";
        const std::string nns = ss.str();
        if (!r) {
          os << nns;
        } else {
          os << repeatedStr(" ", nns.size());
        }
      }
      for (size_t thisChannel = 0; thisChannel < channels; ++thisChannel) {
        os << "[";
        for (size_t c = 0; c < width; ++c) {
          if (c) {
            os << ", ";
          }
          for (size_t dd = 0; dd < depth; ++dd) {
            DType val;
            switch (dim) {
              case 3:
                val = data_at<DType>(&blob, {thisBatch, thisChannel, c});
                break;
              case 4:
                val = data_at<DType>(&blob, {thisBatch, thisChannel, r, c});
                break;
              case 5:
                val = data_at<DType>(&blob, {thisBatch, thisChannel, dd, r, c});
                break;
              default:
                LOG(FATAL) << "Unsupported blob dimension" << dim;
                val = DType(0);
                break;
            }
            os << repeatedStr("(", dd);
            os << std::fixed << std::setw(7) << std::setprecision(MPRINT_PRECISION) << std::right
               << val << " ";
            os << repeatedStr(")", dd, true);
          }
        }
        os << "]  ";
        if (!doChannels) {
          break;
        }
      }
      if (!doBatches) {
        break;
      } else {
        os << " |" << std::flush;
      }
    }
    if (r < height - 1) {
      os << std::endl;
    }
  }
  if (!height) {
    os << "[]";
    if (add_endl) {
      os << std::endl;
    }
  } else if (!add_endl) {
    os << " ";
  } else {
    os << std::endl;
  }
  os << std::flush;
  return os;
}

template <typename StreamType>
inline StreamType& print(const RunContext& ctx,
                         StreamType* _os,
                         const TBlob& blob,
                         const bool doChannels = true,
                         const bool doBatches  = true,
                         const bool add_endl   = true) {
  MSHADOW_TYPE_SWITCH(blob.type_flag_, DType, {
    print_blob_<DType>(ctx, _os, blob, doChannels, doBatches, add_endl);
  });
  return *_os;
}

template <typename StreamType>
inline StreamType& print(const RunContext& ctx,
                         StreamType* _os,
                         const std::string& label,
                         const TBlob& blob,
                         const bool doChannels = true,
                         bool doBatches        = true,
                         const bool add_endl   = true) {
  if (!label.empty()) {
    *_os << label << ": ";
  }
  return print(ctx, _os, blob, doChannels, doBatches, add_endl);
}

template <typename StreamType>
inline StreamType& print(const RunContext& ctx,
                         StreamType* _os,
                         const std::string& label,
                         const NDArray& arr) {
  if (!label.empty()) {
    *_os << label << ": ";
  }
  switch (arr.storage_type()) {
    case kRowSparseStorage: {
      // data
      const mxnet::TShape& shape = arr.shape();
      print_shape(_os, "[row_sparse] main shape", shape, false);
      const mxnet::TShape& storage_shape = arr.storage_shape();
      const bool is_one_row              = storage_shape[0] < 2;
      print_shape(_os, "storage shape", storage_shape, false);
      print(ctx, _os, arr.data(), true, true, !is_one_row);

      // indices
      const mxnet::TShape& indices_shape = arr.aux_shape(rowsparse::kIdx);
      print_shape(_os, "indices shape", indices_shape, false);
      print(ctx, _os, arr.aux_data(rowsparse::kIdx), true, true, false) << std::endl;
      break;
    }
    case kCSRStorage: {
      // data
      const mxnet::TShape& shape = arr.shape();
      print_shape(_os, "[CSR] main shape", shape, false);
      const mxnet::TShape& storage_shape = arr.storage_shape();
      const bool is_one_row              = storage_shape[0] < 2;
      print_shape(_os, "storage shape", storage_shape, false);
      print(ctx, _os, arr.data(), true, true, !is_one_row);

      // row ptrs
      const mxnet::TShape& ind_ptr_shape = arr.aux_shape(csr::kIndPtr);
      print_shape(_os, "row ptrs shape", ind_ptr_shape, false);
      print(ctx, _os, arr.aux_data(csr::kIndPtr), true, true, false) << std::endl;

      // col indices
      const mxnet::TShape& indices_shape = arr.aux_shape(csr::kIdx);
      print_shape(_os, "col indices shape", indices_shape, false);
      print(ctx, _os, arr.aux_data(csr::kIdx), true, true, false) << std::endl;

      break;
    }
    case kDefaultStorage: {
      // data
      const mxnet::TShape& shape = arr.shape();
      const bool is_one_row      = shape[0] < 2;
      print_shape(_os, "[dense] main shape", shape, !is_one_row);
      print(ctx, _os, arr.data(), true, true, !is_one_row) << std::endl;
      break;
    }
    default:
      CHECK(false) << "Unsupported storage type:" << arr.storage_type();
      break;
  }
  return *_os << std::flush;
}

inline void print(const RunContext& ctx,
                  const std::string& label,
                  const std::string& var,
                  const std::vector<NDArray>& arrays) {
  std::cout << label << std::endl;
  for (size_t x = 0, n = arrays.size(); x < n; ++x) {
    std::stringstream ss;
    ss << var << "[" << x << "]";
    test::print(ctx, &std::cout, ss.str(), arrays[x]);
  }
}

inline void print(const RunContext& ctx,
                  const std::string& label,
                  const std::string& var,
                  const std::vector<TBlob>& arrays) {
  std::cout << label << std::endl;
  for (size_t x = 0, n = arrays.size(); x < n; ++x) {
    std::stringstream ss;
    ss << var << "[" << x << "]";
    test::print(ctx, &std::cout, ss.str(), arrays[x], true, true, false);
  }
}

inline std::string demangle(const char* name) {
#if defined(__GLIBCXX__) || defined(_LIBCPP_VERSION)
  int status = -4;  // some arbitrary value to eliminate the compiler warning
  std::unique_ptr<char, void (*)(void*)> res{abi::__cxa_demangle(name, nullptr, nullptr, &status),
                                             &std::free};
  return status ? name : res.get();
#else
  return name;
#endif
}

template <typename T>
inline std::string type_name() {
  return demangle(typeid(T).name());
}

#define PRINT_NDARRAYS(__ctx$, __var) test::print(__ctx$, __FUNCTION__, #__var, __var)
#define PRINT_OP_AND_ARRAYS(__ctx$, __op, __var)                                       \
  test::print(__ctx$,                                                                  \
              __FUNCTION__,                                                            \
              static_cast<std::stringstream*>(                                         \
                  &(std::stringstream() << #__var << "<" << type_name<__op>() << ">")) \
                  ->str(),                                                             \
              __var)
#define PRINT_OP2_AND_ARRAYS(__ctx$, __op1, __op2, __var)  test::print(__ctx$, __FUNCTION__, \
  static_cast<std::stringstream *>(&(std::stringstream() << #__var << \
  "<" << type_name<__op1>().name()) << ", " \
  << type_name<__op2>() << ">"))->str(), __var)

/*! \brief Fill blob with some pattern defined by the getNextData() callback
 * Pattern fill in the defined order (important for analysis):
 *  1D: batch item -> channel -> depth -> row -> col
 *  2D: batch item -> channel -> row -> col
 *  3D: batch item -> channel -> col
 */
template <typename GetNextData>
static inline void patternFill(const RunContext& run_ctx,
                               const TBlob* _blob,
                               GetNextData getNextData) {
  AccessAsCPU(*_blob, run_ctx, [getNextData](const TBlob& blob) {
    const size_t dim = static_cast<size_t>(blob.ndim());
    CHECK_LE(dim, 5U) << "Will need to handle above 3 dimensions (another for loop)";
    const size_t num             = blob.size(0);
    const size_t channels        = dim > 1 ? blob.size(1) : 1;
    const size_t depth           = dim > 2 ? blob.size(2) : 1;
    const size_t height          = dim > 3 ? blob.size(3) : 1;
    const size_t width           = dim > 4 ? blob.size(4) : 1;
    const size_t numberOfIndexes = blob.shape_.Size();
    for (size_t n = 0; n < num; ++n) {
      if (dim > 1) {
        for (size_t ch = 0; ch < channels; ++ch) {
          if (dim > 2) {
            for (size_t d = 0; d < depth; ++d) {
              if (dim > 3) {
                for (size_t row = 0; row < height; ++row) {
                  if (dim > 4) {
                    for (size_t col = 0; col < width; ++col) {
                      if (dim == 5) {
                        const size_t idx = test::offset(blob.shape_, {n, ch, d, row, col});
                        CHECK_LT(idx, numberOfIndexes);
                        MSHADOW_TYPE_SWITCH(blob.type_flag_, ThisDataType, {
                          ThisDataType& f = blob.dptr<ThisDataType>()[idx];
                          f               = getNextData();
                        });
                      } else {
                        CHECK(dim <= 5) << "Unimplemented dimension: " << dim;
                      }
                    }
                  } else {
                    const size_t idx = test::offset(blob.shape_, {n, ch, d, row});
                    CHECK_LT(idx, numberOfIndexes);
                    MSHADOW_TYPE_SWITCH(blob.type_flag_, ThisDataType, {
                      ThisDataType& f = blob.dptr<ThisDataType>()[idx];
                      f               = getNextData();
                    });
                  }
                }
              } else {
                const size_t idx = test::offset(blob.shape_, {n, ch, d});
                CHECK_LT(idx, numberOfIndexes);
                MSHADOW_TYPE_SWITCH(blob.type_flag_, ThisDataType, {
                  ThisDataType& f = blob.dptr<ThisDataType>()[idx];
                  f               = getNextData();
                });
              }
            }
          } else {
            const size_t idx = test::offset(blob.shape_, {n, ch});
            CHECK_LT(idx, numberOfIndexes);
            MSHADOW_TYPE_SWITCH(blob.type_flag_, ThisDataType, {
              ThisDataType& f = blob.dptr<ThisDataType>()[idx];
              f               = getNextData();
            });
          }
        }
      } else {
        const size_t idx = test::offset(blob.shape_, {n});
        CHECK_LT(idx, numberOfIndexes);
        MSHADOW_TYPE_SWITCH(blob.type_flag_, ThisDataType, {
          ThisDataType& f = blob.dptr<ThisDataType>()[idx];
          f               = getNextData();
        });
      }
    }
  });
}

/*! \brief Return a random number within a given range (inclusive) */
template <class ScalarType>
inline ScalarType rangedRand(const ScalarType min, const ScalarType max) {
  uint64_t num_bins = static_cast<uint64_t>(max + 1), num_rand = static_cast<uint64_t>(RAND_MAX),
           bin_size = num_rand / num_bins, defect = num_rand % num_bins;
  ScalarType x;
  do {
    x = std::rand();
  } while (num_rand - defect <= (uint64_t)x);

  return static_cast<ScalarType>(x / bin_size + min);
}

/*!
 * \brief Deterministically compare mxnet::TShape objects as less-than,
 *        for use in stl sorted key such as map and set
 * \param s1 First shape
 * \param s2 Second shape
 * \return true if s1 is less than s2
 */
inline bool operator<(const mxnet::TShape& s1, const mxnet::TShape& s2) {
  if (s1.Size() == s2.Size()) {
    if (s1.ndim() == s2.ndim()) {
      for (size_t i = 0, n = s1.ndim(); i < n; ++i) {
        if (s1[i] == s2[i]) {
          continue;
        }
        return s1[i] < s2[i];
      }
      return false;
    }
    return s1.ndim() < s2.ndim();
  }
  return s1.Size() < s2.Size();
}

/*!
 * \brief Deterministically compare a vector of mxnet::TShape objects as less-than,
 *        for use in stl sorted key such as map and set
 * \param v1 First vector of shapes
 * \param v2 Second vector of shapes
 * \return true if v1 is less than v2
 */
inline bool operator<(const std::vector<mxnet::TShape>& v1, const std::vector<mxnet::TShape>& v2) {
  if (v1.size() == v2.size()) {
    for (size_t i = 0, n = v1.size(); i < n; ++i) {
      if (v1[i] == v2[i]) {
        continue;
      }
      return v1[i] < v2[i];
    }
    return false;
  }
  return v1.size() < v2.size();
}

/*!
 * \brief std::less compare structure for compating vectors of shapes for stl sorted containers
 */
struct less_shapevect {
  bool operator()(const std::vector<mxnet::TShape>& v1,
                  const std::vector<mxnet::TShape>& v2) const {
    if (v1.size() == v2.size()) {
      for (size_t i = 0, n = v1.size(); i < n; ++i) {
        if (v1[i] == v2[i]) {
          continue;
        }
        return v1[i] < v2[i];
      }
      return false;
    }
    return v1.size() < v2.size();
  }
};

inline std::string pretty_num(uint64_t val) {
  if (!test::csv) {
    std::string res, s = std::to_string(val);
    size_t ctr = 0;
    for (int i = static_cast<int>(s.size()) - 1; i >= 0; --i, ++ctr) {
      if (ctr && (ctr % 3) == 0) {
        res += ",";
      }
      res.push_back(s[i]);
    }
    std::reverse(res.begin(), res.end());
    return res;
  } else {
    return std::to_string(val);
  }
}

/*! \brief Change a value during the scope of this declaration */
template <typename T>
struct ScopeSet {
  inline ScopeSet(T* var, const T tempValue) : var_(*var), saveValue_(var) {
    *var = tempValue;
  }
  inline ~ScopeSet() {
    var_ = saveValue_;
  }
  T& var_;
  T saveValue_;
};

static inline void AssertEqual(const std::vector<NDArray*>& in_arrs,
                               const std::vector<NDArray*>& out_arrs,
                               float rtol           = 1e-5,
                               float atol           = 1e-8,
                               bool test_first_only = false) {
  for (size_t j = 0; j < in_arrs.size(); ++j) {
    // When test_all is fir
    if (test_first_only && j == 1) {
      return;
    }
    NDArray tmp1 = *in_arrs[j];
    NDArray tmp2 = *out_arrs[j];
    if (tmp1.ctx().dev_type == mxnet::Context::kGPU) {
      tmp1 = tmp1.Copy(mxnet::Context::CPU(0));
      tmp2 = tmp2.Copy(mxnet::Context::CPU(0));
      tmp1.WaitToRead();
      tmp2.WaitToRead();
    }
#if MXNET_USE_ONEDNN == 1
    tmp1 = tmp1.Reorder2Default();
    tmp2 = tmp2.Reorder2Default();
#endif
    EXPECT_EQ(tmp1.shape().Size(), tmp2.shape().Size());
    TBlob blob1                 = tmp1.data();
    TBlob blob2                 = tmp2.data();
    mshadow::default_real_t* d1 = static_cast<mshadow::default_real_t*>(blob1.dptr_);
    mshadow::default_real_t* d2 = static_cast<mshadow::default_real_t*>(blob2.dptr_);
    for (int i = 0; i < tmp1.shape().Size(); i++) {
      float abs_err = fabs((d1[i]) - (d2[i]));
      ASSERT_LE(abs_err, (atol + rtol * fabs(d2[i])))
          << "index: " << i << ", " << d1[i] << " vs " << d2[i];
    }
  }
}

}  // namespace test
}  // namespace mxnet

#if defined(_MSC_VER)
inline void usleep(__int64 usec) {
  HANDLE timer;
  LARGE_INTEGER ft;

  // Convert to 100 nanosecond interval, negative value indicates relative time
  ft.QuadPart = -(10 * usec);

  timer = CreateWaitableTimer(NULL, TRUE, NULL);
  SetWaitableTimer(timer, &ft, 0, NULL, NULL, 0);
  WaitForSingleObject(timer, INFINITE);
  CloseHandle(timer);
}
#endif  // _WIN32

#endif  // TEST_UTIL_H_
