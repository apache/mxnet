/*!
 * Copyright (c) 2017 by Contributors
 * \file test_util.h
 * \brief unit test performance analysis functions
 * \author Chris Olivier
*/
#ifndef TESTS_CPP_INCLUDE_TEST_UTIL_H_
#define TESTS_CPP_INCLUDE_TEST_UTIL_H_

#include <gtest/gtest.h>
#include <mxnet/storage.h>
#include <string>
#include <vector>
#include <sstream>

#if MXNET_USE_VTUNE
#include <ittnotify.h>
#endif

namespace mxnet {
namespace test {

extern bool unitTestsWithCuda;
extern bool debugOutput;

/*! \brief Pause VTune analysis */
struct VTunePause {
  inline VTunePause() {
#if MXNET_USE_VTUNE
    __itt_pause();
#endif
  }
  inline ~VTunePause() {
#if MXNET_USE_VTUNE
    __itt_resume();
#endif
  }
};

/*! \brief Resume VTune analysis */
struct VTuneResume {
  inline VTuneResume() {
#if MXNET_USE_VTUNE
    __itt_resume();
#endif
  }
  inline ~VTuneResume() {
#if MXNET_USE_VTUNE
    __itt_pause();
#endif
  }
};

constexpr const size_t MPRINT_PRECISION = 5;

template<typename DType>
inline void fill(const TBlob& blob, const DType val) {
  DType *p1 = blob.dptr<DType>();
  for (size_t i = 0, n = blob.Size(); i < n; ++i) {
    *p1++ = val;
  }
}

template<typename DType>
inline void fill(const TBlob& blob, const DType *valArray) {
  DType *p1 = blob.dptr<DType>();
  for (size_t i = 0, n = blob.Size(); i < n; ++i) {
    *p1++ = *valArray++;
  }
}

template<typename DType>
inline void try_fill(const std::vector<TBlob>& container, size_t index, const DType value) {
  if (index < container.size()) {
    test::fill(container[index], value);
  }
}

template<typename DType, typename Stream>
inline void dump(Stream *os, const TBlob& blob, const char *suffix = "f") {
  DType *p1 = blob.dptr<DType>();
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
inline index_t getMult(const TShape& shape, const index_t axis) {
  return axis < shape.ndim() ? shape[axis] : 1;
}

/*! \brief offset, given indices such as bn, channel, depth, row, column */
inline index_t offset(const TShape& shape, const std::vector<size_t>& indices) {
  const size_t dim = shape.ndim();
  CHECK_LE(indices.size(), dim);
  size_t offset = 0;
  for (size_t i = 0; i < dim; ++i) {
    offset *= shape[i];
    if (indices.size() > i) {
      CHECK_GE(indices[i], 0U);
      CHECK_LT(indices[i], shape[i]);
      offset += indices[i];
    }
  }
  return offset;
}

/*! \brief Return reference to data at position indexes */
template<typename DType>
inline const DType& data_at(const TBlob *blob, const std::vector<size_t>& indices) {
  return blob->dptr<DType>()[offset(blob->shape_, indices)];
}

/*! \brief Set data at position indexes */
template<typename DType>
inline DType& data_ref(const TBlob *blob, const std::vector<size_t>& indices) {
  return blob->dptr<DType>()[offset(blob->shape_, indices)];
}

inline std::string repeatedStr(const char *s, const signed int count,
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

/*! \brief Pretty print a 1D, 2D, or 3D blob */
template<typename DType, typename StreamType>
inline StreamType& print_blob(StreamType *_os, const TBlob &blob,
                              bool doChannels = true, bool doBatches = true) {
  StreamType& os = *_os;
  const size_t dim = static_cast<size_t>(blob.ndim());

  if (dim == 1) {
    // probably a tensor (mshadow::Tensor is deprecated)
    TBlob changed(blob.dptr<DType>(), TShape(3), blob.dev_mask(), blob.dev_id());
    changed.shape_[0] = 1;
    changed.shape_[1] = 1;
    changed.shape_[2] = blob.shape_[0];
    return print_blob<DType>(&os, changed, false, false);
  } else if (dim == 2) {
    // probably a tensor (mshadow::Tensor is deprecated)
    TBlob changed(blob.dptr<DType>(), TShape(4), blob.dev_mask(), blob.dev_id());
    changed.shape_[0] = 1;
    changed.shape_[1] = 1;
    changed.shape_[2] = blob.shape_[0];
    changed.shape_[3] = blob.shape_[1];
    return print_blob<DType>(&os, changed, false, false);
  }
  CHECK_GE(dim, 3U) << "Invalid dimension zero (0)";

  const size_t batchSize = blob.size(0);

  size_t channels = 1;
  size_t depth  = 1;
  size_t height = 1;
  size_t width  = 1;
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

  os << std::endl;
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
        for (size_t c = 0; c < width; ++c) {
          if (c) {
            os << ", ";
          } else {
            os << "[";
          }
          for (size_t dd = 0; dd < depth; ++dd) {
            DType val;
            switch (dim) {
              case 3:
                val = data_at<DType>(&blob, {thisBatch, thisChannel, c });
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
            os << std::fixed << std::setw(7) << std::setprecision(MPRINT_PRECISION)
               << std::right << val << " ";
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
        os << " |" << std::flush;;
      }
    }
    os << std::endl;
  }
  os << std::endl << std::flush;
  return os;
}

template<typename DType>
inline size_t shapeMemorySize(const TShape& shape) {
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
  void *Alloc(const size_t size) {
    CHECK_GT(size, 0U);  // You've probably made a mistake
    mxnet::Context context = isGPU_ ? mxnet::Context::GPU(0) : mxnet::Context{};
    Storage *storage = mxnet::Storage::Get();
    handle_ = storage->Alloc(size, context);
    return handle_.dptr;
  }
  void Free() {
    if (handle_.dptr) {
      Storage *storage = mxnet::Storage::Get();
      storage->DirectFree(handle_);
      handle_.dptr = nullptr;
    }
  }

 private:
  const bool      isGPU_;
  Storage::Handle handle_;
};

class StandaloneBlob : public TBlob {
 public:
  inline StandaloneBlob(const TShape& shape, const bool isGPU, const int dtype)
    : TBlob(nullptr, shape, isGPU ? gpu::kDevMask : cpu::kDevMask, dtype)
      , memory_(isGPU) {
    MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
      this->dptr_ = memory_.Alloc(shapeMemorySize<DType>(shape)); });
  }
  inline ~StandaloneBlob() {
    this->dptr_ = nullptr;
    memory_.Free();
  }
 private:
  /*! \brief Locally allocated memory block for this blob */
  BlobMemory  memory_;
};

/*! \brief Fill blob with some pattern defined by the getNextData() callback
 * Pattern fill in the defined order (important for analysis):
 *  1D: batch item -> channel -> depth -> row -> col
 *  2D: batch item -> channel -> row -> col
 *  3D: batch item -> channel -> col
 */
template<typename DType, typename GetNextData>
static inline void patternFill(TBlob *blob, GetNextData getNextData) {
  const size_t dim = blob->ndim();
  CHECK_LE(dim, 5U) << "Will need to handle above 3 dimensions (another for loop)";
  const size_t num = blob->size(0);
  const size_t channels = dim > 1 ? blob->size(1) : 1;
  const size_t depth = dim > 2 ? blob->size(2) : 1;
  const size_t height = dim > 3 ? blob->size(3) : 1;
  const size_t width = dim > 4 ? blob->size(4) : 1;
  const size_t numberOfIndexes = blob->shape_.Size();
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
                      const size_t idx = test::offset(blob->shape_, {n, ch, d, row, col});
                      CHECK_LT(idx, numberOfIndexes);
                      DType &f = blob->dptr<DType>()[idx];
                      f = getNextData();
                    } else {
                      CHECK(dim <= 5) << "Unimplemented dimension: " << dim;
                    }
                  }
                } else {
                  const size_t idx = test::offset(blob->shape_, {n, ch, d, row});
                  CHECK_LT(idx, numberOfIndexes);
                  DType &f = blob->dptr<DType>()[idx];
                  f = getNextData();
                }
              }
            } else {
              const size_t idx = test::offset(blob->shape_, {n, ch, d});
              CHECK_LT(idx, numberOfIndexes);
              DType &f = blob->dptr<DType>()[idx];
              f = getNextData();
            }
          }
        } else {
          const size_t idx = test::offset(blob->shape_, {n, ch});
          CHECK_LT(idx, numberOfIndexes);
          DType &f = blob->dptr<DType>()[idx];
          f = getNextData();
        }
      }
    } else {
      const size_t idx = test::offset(blob->shape_, {n});
      CHECK_LT(idx, numberOfIndexes);
      DType &f = blob->dptr<DType>()[idx];
      f = getNextData();
    }
  }
}

/*! \brief Return a random number within a given range (inclusive) */
template<class ScalarType>
inline ScalarType rangedRand(const ScalarType min, const ScalarType max) {
  uint64_t num_bins = static_cast<uint64_t>(max + 1),
    num_rand = static_cast<uint64_t>(RAND_MAX),
    bin_size = num_rand / num_bins,
    defect   = num_rand % num_bins;
  ScalarType x;
  do {
    x = random();
  } while (num_rand - defect <= (uint64_t)x);

  return static_cast<ScalarType>(x / bin_size + min);
}

/*! \brief Change a value during the scope of this declaration */
template<typename T>
struct ScopeSet {
  inline ScopeSet(T *var, const T tempValue)
    : var_(*var)
      , saveValue_(var) {
    *var = tempValue;
  }
  inline ~ScopeSet() {
    var_ = saveValue_;
  }
  T& var_;
  T  saveValue_;
};


}  // namespace test
}  // namespace mxnet

#endif  // TESTS_CPP_INCLUDE_TEST_UTIL_H_
