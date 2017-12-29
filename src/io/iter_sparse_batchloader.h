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
    index_t top = 0;
    inst_cache_.clear();
    while (sparse_base_->Next()) {
      inst_cache_.emplace_back(sparse_base_->Value());
      if (inst_cache_.size() >= param_.batch_size) break;
    }
    // no more data instance
    if (inst_cache_.size() == 0) {
      return false;
    }
    if (inst_cache_.size() < param_.batch_size) {
      CHECK_GT(param_.round_batch, 0);
      num_overflow_ = 0;
      sparse_base_->BeforeFirst();
      for (; inst_cache_.size() < param_.batch_size; ++num_overflow_) {
        CHECK(sparse_base_->Next()) << "number of input must be bigger than batch size";
        inst_cache_.emplace_back(sparse_base_->Value());
      }
    }
    out_.num_batch_padd = num_overflow_;
    CHECK_EQ(inst_cache_.size(), param_.batch_size);
    this->InitDataFromBatch();
    for (size_t j = 0; j < inst_cache_.size(); j++) {
      const auto& d = inst_cache_[j];
      out_.inst_index[top] = d.index;
      // TODO(haibin) double check the type?
      int64_t unit_size = 0;
      for (size_t i = 0; i < d.data.size(); ++i) {
        // indptr tensor
        if (IsIndPtr(i)) {
          auto indptr = data_[i].get<cpu, 1, int64_t>();
          if (j == 0) indptr[0] = 0;
          indptr[j + 1] = indptr[j] + unit_size;
          offsets_[i] = j;
        } else {
          // indices and values tensor
          unit_size = d.data[i].shape_.Size();
          MSHADOW_TYPE_SWITCH(data_[i].type_flag_, DType, {
            const auto begin = offsets_[i];
            const auto end = offsets_[i] + unit_size;
            mshadow::Copy(data_[i].get<cpu, 1, DType>().Slice(begin, end),
                          d.data[i].get_with_shape<cpu, 1, DType>(mshadow::Shape1(unit_size)));
            });
          offsets_[i] += unit_size;
        }
      }
    }
    return true;
  }

  virtual const TBlobBatch &Value(void) const {
    return BatchLoader::Value();
  }

  virtual const NDArrayStorageType GetStorageType(bool is_data) const {
    return sparse_base_->GetStorageType(is_data);
  }

  virtual const TShape GetShape(bool is_data) const {
    TShape inst_shape = sparse_base_->GetShape(is_data);
    std::vector<index_t> shape_vec;
    shape_vec.push_back(param_.batch_size);
    for (index_t dim = 0; dim < inst_shape.ndim(); ++dim) {
      shape_vec.push_back(inst_shape[dim]);
    }
    return TShape(shape_vec.begin(), shape_vec.end());
  }

 private:
  /*! \brief base sparse iterator */
  SparseIIterator<DataInst> *sparse_base_;
  /*! \brief data instances */
  std::vector<DataInst> inst_cache_;
  /*! \brief data storage type */
  NDArrayStorageType data_stype_;
  /*! \brief data label type */
  NDArrayStorageType label_stype_;
  /*! \brief tensor offset for slicing */
  std::vector<size_t> offsets_;

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
    if (i == label_indptr_offset && label_stype_ == kCSRStorage && data_stype_ == kCSRStorage) {
      return true;
    }
    return false;
  }

  // initialize the data holder by using from the batch
  inline void InitDataFromBatch() {
    CHECK(data_stype_ == kCSRStorage || label_stype_ == kCSRStorage);
    CHECK_GT(inst_cache_.size(), 0);
    out_.data.clear();
    data_.clear();
    offsets_.clear();

    size_t total_size = inst_cache_[0].data.size();
    data_.resize(total_size);
    offsets_.resize(total_size, 0);
    std::vector<size_t> vec_sizes(total_size, 0);
    // accumulate the memory required for a batch
    for (size_t i = 0; i < total_size; ++i) {
      size_t size = 0;
      // vec_size for indptr
      if (IsIndPtr(i)) {
        size = param_.batch_size + 1;
      } else {
        for (const auto &d : inst_cache_) size += d.data[i].shape_.Size();
      }
      vec_sizes[i] = size;
    }

    CHECK_EQ(vec_sizes[0], vec_sizes[1]);
    for (size_t i = 0; i < total_size; ++i) {
      int src_type_flag = inst_cache_[0].data[i].type_flag_;
      // init object attributes
      TShape dst_shape(mshadow::Shape1(vec_sizes[i]));
      data_[i].resize(mshadow::Shape1(vec_sizes[i]), src_type_flag);
      CHECK(data_[i].dptr_ != nullptr);
      out_.data.push_back(TBlob(data_[i].dptr_, dst_shape, cpu::kDevMask, src_type_flag));
    }
  }
};  // class BatchLoader
}  // namespace io
}  // namespace mxnet
#endif  // MXNET_IO_ITER_SPARSE_BATCHLOADER_H_
