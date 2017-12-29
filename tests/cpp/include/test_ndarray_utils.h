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
#ifndef TEST_NDARRAY_UTILS_H_
#define TEST_NDARRAY_UTILS_H_

#include <unistd.h>
#include <dmlc/logging.h>
#include <gtest/gtest.h>
#include <mxnet/engine.h>
#include <mxnet/ndarray.h>
#include <cstdio>
#include <vector>
#include <cstdlib>
#include <string>
#include <map>
#include "test_util.h"
#include "test_op.h"

namespace mxnet {
namespace test {

#define ROW_SPARSE_IDX_TYPE mshadow::kInt64

using namespace mxnet;
#define TEST_DTYPE float
#define TEST_ITYPE int32_t

inline void CheckDataRegion(const TBlob &src, const TBlob &dst) {
  auto size = src.shape_.Size() * mshadow::mshadow_sizeof(src.type_flag_);
  auto equals = memcmp(src.dptr_, dst.dptr_, size);
  EXPECT_EQ(equals, 0);
}

inline unsigned gen_rand_seed() {
  time_t timer;
  ::time(&timer);
  return static_cast<unsigned>(timer);
}

inline float RandFloat() {
  static unsigned seed = gen_rand_seed();
  double v = rand_r(&seed) * 1.0 / RAND_MAX;
  return static_cast<float>(v);
}

// Get an NDArray with provided indices, prepared for a RowSparse NDArray.
inline NDArray RspIdxND(const TShape shape, const Context ctx,
                        const std::vector<TEST_ITYPE> &values) {
  NDArray nd(shape, ctx, false, ROW_SPARSE_IDX_TYPE);
  size_t num_val = values.size();
  MSHADOW_TYPE_SWITCH(nd.dtype(), DType, {
    auto tensor = nd.data().FlatTo1D<cpu, DType>();
    for (size_t i = 0; i < num_val; i++) {
      tensor[i] = values[i];
    }
  });
  return nd;
}

// Get a dense NDArray with provided values.
inline NDArray DnsND(const TShape shape, const Context ctx, std::vector<TEST_DTYPE> vs) {
  NDArray nd(shape, ctx, false);
  size_t num_val = shape.Size();
  // generate random values
  while (vs.size() < num_val) {
    auto v = RandFloat();
    vs.emplace_back(v);
  }
  CHECK_EQ(vs.size(), nd.shape().Size());
  MSHADOW_TYPE_SWITCH(nd.dtype(), DType, {
    auto tensor = nd.data().FlatTo1D<cpu, DType>();
    for (size_t i = 0; i < num_val; i++) {
      tensor[i] = vs[i];
    }
  });
  return nd;
}

template<typename xpu>
static void inline CopyBlob(mshadow::Stream<xpu> *s,
                            const TBlob& dest_blob,
                            const TBlob& src_blob) {
  using namespace mshadow;
  using namespace mshadow::expr;
  CHECK_EQ(src_blob.type_flag_, dest_blob.type_flag_);
  CHECK_EQ(src_blob.shape_, dest_blob.shape_);
  MSHADOW_TYPE_SWITCH(src_blob.type_flag_, DType, {
    // Check if the pointers are the same (in-place operation needs no copy)
    if (src_blob.dptr<DType>() != dest_blob.dptr<DType>()) {
      mshadow::Copy(dest_blob.FlatTo1D<xpu, DType>(s), src_blob.FlatTo1D<xpu, DType>(s), s);
    }
  });
}

// Get a RowSparse NDArray with provided indices and values
inline NDArray RspND(const TShape shape, const Context ctx, const std::vector<TEST_ITYPE> idx,
              std::vector<TEST_DTYPE> vals) {
  CHECK(shape.ndim() <= 2) << "High dimensional row sparse not implemented yet";
  index_t num_rows = idx.size();
  index_t num_cols = vals.size() / idx.size();
  // create index NDArray
  NDArray index = RspIdxND(mshadow::Shape1(num_rows), ctx, idx);
  print(&std::cout, "index", index);
  CHECK_EQ(vals.size() % idx.size(), 0);
  // create value NDArray
  NDArray data = DnsND(mshadow::Shape2(num_rows, num_cols), ctx, vals);
  print(&std::cout, "data", data);
  // create result nd
  std::vector<TShape> aux_shapes = {mshadow::Shape1(num_rows)};
  NDArray nd(kRowSparseStorage, shape, ctx, false, mshadow::default_type_flag,
             {}, aux_shapes);

  mshadow::Stream<cpu> *s = nullptr;
  CopyBlob(s, nd.aux_data(rowsparse::kIdx), index.data());
  CopyBlob(s, nd.data(), data.data());

  print(&std::cout, "nd", nd);
  return nd;
}

/*! \brief Array - utility class to construct sparse arrays
 *  \warning This class is not meant to run in a production environment.  Since it is for unit tests only,
 *           simplicity has been chosen over performance.
 **/
template<typename DType>
class Array {
  typedef std::map<size_t, std::map<size_t, DType> > TItems;
  static constexpr double EPSILON = 1e-5;

  static const char *st2str(const NDArrayStorageType storageType) {
    switch (storageType) {
      case kDefaultStorage:
        return "kDefaultStorage";
      case kRowSparseStorage:
        return "kRowSparseStorage";
      case kCSRStorage:
        return "kCSRStorage";
      case kUndefinedStorage:
        return "kUndefinedStorage";
      default:
        LOG(FATAL) << "Unsupported storage type: " << storageType;
        return "<INVALID>";
    }
  }

  /*! \brief Remove all zero entries */
  void Prune() {
    for (typename TItems::iterator i = items_.begin(), e = items_.end();
         i != e;) {
      const size_t y = i->first;
      std::map<size_t, DType> &m = i->second;
      ++i;
      for (typename std::map<size_t, DType>::const_iterator j = m.begin(), jn = m.end();
           j != jn;) {
        const size_t x = j->first;
        const DType v = j->second;
        ++j;
        if (IsZero(v)) {
          m.erase(x);
        }
      }
      if (m.empty()) {
        items_.erase(y);
      }
    }
  }

  /*! \brief Create a dense NDArray from our mapped data */
  NDArray CreateDense(const Context& ctx) const {
    NDArray array(shape_, Context::CPU(-1));
    TBlob data = array.data();
    DType *p_data = data.dptr<DType>();
    memset(p_data, 0, array.shape().Size() * sizeof(DType));
    for (typename TItems::const_iterator i = items_.begin(), e = items_.end();
         i != e; ++i) {
      const size_t y = i->first;
      const std::map<size_t, DType> &m = i->second;
      for (typename std::map<size_t, DType>::const_iterator j = m.begin(), jn = m.end();
           j != jn; ++j) {
        const size_t x = j->first;
        const DType v = j->second;
        if (!IsZero(v)) {
          const size_t offset = mxnet::test::offset(shape_, {y, x});
          p_data[offset] = v;
        }
      }
    }
    if (ctx.dev_type == Context::kGPU) {
      NDArray argpu(shape_, ctx);
      CopyFromTo(array, &argpu);
      return argpu;
    } else {
      return array;
    }
  }

 public:
  Array() = default;

  explicit Array(const TShape &shape)
    : shape_(shape) {}

  explicit Array(const NDArray &arr)
    : shape_(arr.shape()) {
    Load(arr);
  }

  void clear() {
    items_.clear();
    shape_ = TShape(0);
  }

  static inline bool IsNear(const DType v1, const DType v2) { return fabs(v2 - v1) <= EPSILON; }
  static inline bool IsZero(const DType v) { return IsNear(v, DType(0)); }

  /*! Index into value maps via: [y][x] (row, col) */
  std::map<size_t, DType> &operator[](const size_t idx) { return items_[idx]; }

  const std::map<size_t, DType> &operator[](const size_t idx) const {
    typename TItems::const_iterator i = items_.find(idx);
    if (i != items_.end()) {
      return i->second;
    }
    CHECK(false) << "Attempt to access a non-existent key in a constant map";
    return *static_cast<std::map<size_t, DType> *>(nullptr);
  }

  bool Contains(const size_t row, const size_t col) const {
    typename TItems::const_iterator i = items_.find(row);
    if (i != items_.end()) {
      typename std::map<size_t, DType>::const_iterator j = i->second.find(col);
      if (j != i->second.end()) {
        return true;
      }
    }
    return false;
  }

  /*! \brief Convert from one storage type NDArray to another */
  static NDArray Convert(const Context& ctx, const NDArray& src,
                         const NDArrayStorageType storageType) {
    std::unique_ptr<NDArray> pArray(
      storageType == kDefaultStorage
      ? new NDArray(src.shape(), ctx)
      : new NDArray(storageType, src.shape(), ctx));
    OpContext opContext;
    MXNET_CUDA_ONLY(std::unique_ptr<test::op::GPUStreamScope> gpuScope;);
    switch (ctx.dev_type) {
#if MNXNET_USE_CUDA
      case Context::kGPU:
        gpuScope.reset(new test::op::GPUStreamScope(&opContext));
        mxnet::op::CastStorageComputeImpl<gpu>(s, src, dest);
        break;
#endif  // MNXNET_USE_CUDA
      default: {  // CPU
        OpContext op_ctx;
        mxnet::op::CastStorageComputeImpl<cpu>(op_ctx, src, *pArray);
        break;
      }
    }
    return *pArray;
  }

  /*! \brief Return NDArray of given storage type representing the value maps */
  NDArray Save(const Context& ctx, const NDArrayStorageType storageType) const {
    switch (storageType) {
      case kDefaultStorage:
        return CreateDense(ctx);
      case kRowSparseStorage:
      case kCSRStorage:
        return Convert(ctx, CreateDense(ctx), storageType);
      case kUndefinedStorage:
      default:
        LOG(ERROR) << "Unsupported storage type: " << storageType;
        return NDArray(TShape(0), ctx);
    }
  }

  void Load(NDArray array) {
    clear();
    shape_ = array.shape();
    if (array.storage_type() != kDefaultStorage) {
      array = Convert(array.ctx(), array, kDefaultStorage);
    }
#if MXNET_USE_CUDA
    if (array.ctx().dev_type == Context::kGPU) {
      NDArray tmp(array.shape(), Context::CPU(-1));
      CopyFromTo(array, &tmp);
      array = tmp;
    }
#endif  // MXNET_USE_CUDA
    const TBlob blob = array.data();
    DType *p = blob.dptr<DType>();
    CHECK_EQ(shape_.ndim(), 2U);
    for (size_t row = 0, nrow = shape_[0]; row < nrow; ++row) {
      for (size_t col = 0, ncol = shape_[1]; col < ncol; ++col) {
        const size_t off = test::offset(shape_, {row, col});
        if (!IsZero(p[off])) {
          (*this)[row][col] = p[off];
        }
      }
    }
  }

  void print() const {
    for (typename TItems::const_iterator i = items_.begin(), e = items_.end();
         i != e; ++i) {
      const size_t y = i->first;
      const std::map<size_t, DType> &m = i->second;
      CHECK_EQ(m.empty(), false);  // How did it get to have an empty map?
      for (typename std::map<size_t, DType>::const_iterator j = m.begin(), jn = m.end();
           j != jn; ++j) {
        const size_t x = j->first;
        const DType v = j->second;
        if (!IsZero(v)) {
          std::cout << "[row=" << y << ", col=" << x << "]: " << v << std::endl;
        }
      }
    }
    std::cout << std::flush;
  }

 private:
  TShape shape_;
  TItems items_;
};

template<typename StreamType>
inline StreamType& print_dense(StreamType *_os, const std::string& label, const NDArray& arr) {
  MSHADOW_TYPE_SWITCH(arr.data().type_flag_, DType, {
    print(_os, label, test::Array<DType>(arr).Save(arr.ctx(), kDefaultStorage))
      << std::endl;
  });
  return *_os;
}

}  // namespace test
}  // namespace mxnet

#endif  // TEST_NDARRAY_UTILS_H_
