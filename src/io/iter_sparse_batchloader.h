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
 * \file iter_sparse_batchloader.h
 * \brief define a batch adapter to create sparse tblob batch
 */
#ifndef MXNET_IO_ITER_SPARSE_BATCHLOADER_H_
#define MXNET_IO_ITER_SPARSE_BATCHLOADER_H_

#include <mxnet/io.h>
#include <mxnet/base.h>
#include <dmlc/logging.h>
#include <mshadow/tensor.h>
#include <utility>
#include <vector>
#include <string>
#include "./inst_vector.h"
#include "./image_iter_common.h"
#include "./iter_batchloader.h"
#include "./iter_sparse.h"

namespace mxnet {
namespace io {

/*! \brief create a batch iterator from single instance iterator */
class SparseBatchLoader : public BatchLoader, public SparseIIterator<TBlobBatch> {
 public:
  explicit SparseBatchLoader(SparseIIterator<DataInst> *base):
      BatchLoader(base), sparse_base_(base) {
  }

  virtual ~SparseBatchLoader(void) {}

  inline void Init(const std::vector<std::pair<std::string, std::string> >& kwargs) {
    BatchLoader::Init(kwargs);
    data_stype_ = sparse_base_->GetStorageType(true);
    label_stype_ = sparse_base_->GetStorageType(false);
    if (param_.round_batch == 0) {
      LOG(FATAL) << "sparse batch loader doesn't support round_batch == false yet";
    }
  }

  virtual void BeforeFirst(void) {
    BatchLoader::BeforeFirst();
  }

  virtual bool Next(void) {
    out_.num_batch_padd = 0;
    out_.batch_size = param_.batch_size;
    this->head_ = 0;
    // if overflown from previous round, directly return false, until before first is called
    if (num_overflow_ != 0) return false;
    size_t top = 0;
    offsets_.clear();
    while (sparse_base_->Next()) {
      const DataInst& inst = sparse_base_->Value();
      // initialize the data buffer, only called once
      if (data_.size() == 0) this->InitData(inst);
      // initialize the number of elements in each buffer, called once per batch
      if (offsets_.size() == 0) offsets_.resize(inst.data.size(), 0);
      CopyData(inst, top);
      if (++top >= param_.batch_size) {
        SetOutputShape();
        return true;
      }
    }
    if (top != 0) {
      CHECK_NE(param_.round_batch, 0)
        << "round_batch = False is not supported for sparse data iterator";
      num_overflow_ = 0;
      sparse_base_->BeforeFirst();
      for (; top < param_.batch_size; ++top, ++num_overflow_) {
        CHECK(sparse_base_->Next()) << "number of input must be bigger than batch size";
        const DataInst& inst = sparse_base_->Value();
        // copy data
        CopyData(inst, top);
      }
      SetOutputShape();
      out_.num_batch_padd = num_overflow_;
      return true;
    }
    // no more data instance
    return false;
  }

  virtual const TBlobBatch &Value(void) const {
    return BatchLoader::Value();
  }

  virtual const NDArrayStorageType GetStorageType(bool is_data) const {
    return sparse_base_->GetStorageType(is_data);
  }

  virtual const mxnet::TShape GetShape(bool is_data) const {
    mxnet::TShape inst_shape = sparse_base_->GetShape(is_data);
    std::vector<index_t> shape_vec;
    shape_vec.push_back(param_.batch_size);
    for (index_t dim = 0; dim < inst_shape.ndim(); ++dim) {
      shape_vec.push_back(inst_shape[dim]);
    }
    return mxnet::TShape(shape_vec.begin(), shape_vec.end());
  }

 private:
  /*! \brief base sparse iterator */
  SparseIIterator<DataInst> *sparse_base_;
  /*! \brief data storage type */
  NDArrayStorageType data_stype_;
  /*! \brief data label type */
  NDArrayStorageType label_stype_;
  /*! \brief tensor offsets for slicing */
  std::vector<size_t> offsets_;
  /*! \brief tensor dtypes */
  std::vector<int> dtypes_;
  /*! \brief whether the offset correspond to an indptr array */
  std::vector<bool> indptr_;

  // check whether ith position is the indptr tensor for a CSR tensor
  inline bool IsIndPtr(size_t i) {
    auto data_num_aux = num_aux_data(data_stype_);
    auto label_num_aux = num_aux_data(label_stype_);
    auto label_indptr_offset = data_num_aux + 1 + label_num_aux;
    // data indptr
    if (i == data_num_aux && data_stype_ == kCSRStorage) {
      return true;
    }
    // label indptr
    if (i == label_indptr_offset && label_stype_ == kCSRStorage &&
        data_stype_ == kCSRStorage) {
      return true;
    }
    return false;
  }

  // initialize the data holder by using from the batch
  inline void InitData(const DataInst& first_inst) {
    CHECK(data_stype_ == kCSRStorage || label_stype_ == kCSRStorage);
    out_.data.clear();
    data_.clear();
    offsets_.clear();
    indptr_.clear();

    // num_arrays is the number of arrays in inputs
    // if both data and label are in the csr format,
    // num_arrays will be 3 + 3 = 6.
    size_t num_arrays = first_inst.data.size();
    data_.resize(num_arrays);
    offsets_.resize(num_arrays, 0);
    indptr_.resize(num_arrays, false);
    // tensor buffer sizes
    std::vector<size_t> buff_sizes(num_arrays, 0);
    dtypes_.resize(num_arrays);
    out_.data.resize(num_arrays);
    // estimate the memory required for a batch
    for (size_t i = 0; i < num_arrays; ++i) {
      // shape for indptr
      if (IsIndPtr(i)) {
        buff_sizes[i] = param_.batch_size + 1;
        indptr_[i] = true;
      } else {
        // estimated the size for the whole batch based on the first instance
        buff_sizes[i] = first_inst.data[i].Size() * param_.batch_size;
        indptr_[i] = false;
      }
      dtypes_[i] = first_inst.data[i].type_flag_;
    }

    CHECK_EQ(buff_sizes[0], buff_sizes[1]);
    // allocate buffer
    for (size_t i = 0; i < num_arrays; ++i) {
      // init object attributes
      mxnet::TShape dst_shape(mshadow::Shape1(buff_sizes[i]));
      data_[i].resize(mshadow::Shape1(buff_sizes[i]), dtypes_[i]);
      CHECK(data_[i].dptr_ != nullptr);
    }
  }

  /* \brief set the shape of the outputs based on actual shapes */
  inline void SetOutputShape() {
    for (size_t i = 0; i < out_.data.size(); i++) {
      out_.data[i] = TBlob(data_[i].dptr_, mshadow::Shape1(offsets_[i]),
                           Context::kCPU, dtypes_[i]);
    }
  }

  /* \brief increase the size of i-th data buffer by a factor of 2, while retaining the content */
  inline void ResizeBuffer(size_t src_size, size_t i) {
    MSHADOW_TYPE_SWITCH(data_[i].type_flag_, DType, {
      TBlobContainer temp;
      temp.resize(mshadow::Shape1(src_size), dtypes_[i]);
      mshadow::Copy(temp.get<cpu, 1, DType>(), data_[i].get<cpu, 1, DType>().Slice(0, src_size));
      // increase the size of space exponentially
      size_t capacity = data_[i].Size();
      capacity = capacity * 2 + 1;
      data_[i] = TBlobContainer();
      data_[i].resize(mshadow::Shape1(capacity), dtypes_[i]);
      // copy back
      mshadow::Copy(data_[i].get<cpu, 1, DType>().Slice(0, src_size), temp.get<cpu, 1, DType>());
    });
  }

  /* \brief copy the data instance to data buffer */
  void CopyData(const DataInst& inst, const size_t top) {
    int64_t unit_size = 0;
    out_.inst_index[top] = inst.index;
    for (size_t i = 0; i < inst.data.size(); ++i) {
      if (!indptr_[i]) {
        // indices and values tensor
        unit_size = inst.data[i].shape_.Size();
        MSHADOW_TYPE_SWITCH(data_[i].type_flag_, DType, {
          const size_t begin = offsets_[i];
          const size_t end = offsets_[i] + unit_size;
          size_t capacity = data_[i].Size();
          // resize the data buffer if estimated space is not sufficient
          while (capacity < end) {
            ResizeBuffer(begin, i);
            capacity = data_[i].Size();
          }
          mshadow::Copy(data_[i].get<cpu, 1, DType>().Slice(begin, end),
                        inst.data[i].get_with_shape<cpu, 1, DType>(mshadow::Shape1(unit_size)));
        });
        offsets_[i] += unit_size;
      } else {
        // indptr placeholder
        auto indptr = data_[i].get<cpu, 1, int64_t>();
        // initialize the first indptr, which is always 0
        if (top == 0) indptr[0] = 0;
        indptr[top + 1] = indptr[top] + unit_size;
        offsets_[i] = top + 2;
      }
    }
  }
};  // class BatchLoader
}  // namespace io
}  // namespace mxnet
#endif  // MXNET_IO_ITER_SPARSE_BATCHLOADER_H_
