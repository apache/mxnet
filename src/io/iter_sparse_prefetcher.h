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
 * \file iter_sparse_prefetcher.h
 * \brief define a prefetcher using threaditer to keep k batch fetched
 */
#ifndef MXNET_IO_ITER_SPARSE_PREFETCHER_H_
#define MXNET_IO_ITER_SPARSE_PREFETCHER_H_

#include <mxnet/io.h>
#include <mxnet/base.h>
#include <mxnet/ndarray.h>
#include <dmlc/logging.h>
#include <dmlc/threadediter.h>
#include <dmlc/optional.h>
#include <mshadow/tensor.h>
#include <climits>
#include <utility>
#include <string>
#include <vector>
#include <queue>
#include <algorithm>
#include "./inst_vector.h"
#include "./image_iter_common.h"
#include "./iter_prefetcher.h"
#include "./iter_sparse.h"

namespace mxnet {
namespace io {
// iterator on sparse data
class SparsePrefetcherIter : public PrefetcherIter {
 public:
  explicit SparsePrefetcherIter(SparseIIterator<TBlobBatch>* base)
      : PrefetcherIter(base), sparse_loader_(base) {}

  ~SparsePrefetcherIter() {}

  virtual void Init(const std::vector<std::pair<std::string, std::string> >& kwargs) {
    PrefetcherIter::InitParams(kwargs);
    // use the kwarg to init batch loader
    sparse_loader_->Init(kwargs);
    iter.Init([this](DataBatch **dptr) {
        if (!sparse_loader_->Next()) return false;
        const TBlobBatch& batch = sparse_loader_->Value();
        if (*dptr == nullptr) {
          // allocate databatch
          *dptr = new DataBatch();
          (*dptr)->num_batch_padd = batch.num_batch_padd;
          // (*dptr)->data.at(0) => data
          // (*dptr)->data.at(1) => label
          (*dptr)->data.resize(2);
          (*dptr)->index.resize(batch.batch_size);
          size_t data_iter = 0;
          for (size_t i = 0; i < (*dptr)->data.size(); ++i) {
            bool is_data = i == 0;
            auto stype = this->GetStorageType(is_data);
            auto dtype = param_.dtype ? param_.dtype.value() : batch.data[data_iter].type_flag_;
            if (stype == kDefaultStorage) {
              (*dptr)->data.at(i) = NDArray(batch.data[data_iter].shape_,
                                            Context::CPU(), false, dtype);
            } else {
              (*dptr)->data.at(i) = NDArray(stype, this->GetShape(is_data),
                                            Context::CPU(), false, dtype);
            }
            data_iter += num_aux_data(stype) + 1;
          }
        }
        // copy data over
        size_t data_iter = 0;
        for (size_t i = 0; i < (*dptr)->data.size(); ++i) {
          auto& nd = ((*dptr)->data)[i];
          auto stype = nd.storage_type();
          auto& data_i = ((*dptr)->data)[i];
          if (stype == kDefaultStorage) {
            CopyFromTo(data_i.data(), batch.data[data_iter]);
          } else if (stype == kCSRStorage) {
            auto& values = batch.data[data_iter];
            auto& indices = batch.data[data_iter + 1];
            auto& indptr = batch.data[data_iter + 2];
            // allocate memory
            CHECK_EQ(indices.shape_.Size(), values.shape_.Size());
            nd.CheckAndAllocAuxData(csr::kIdx, indices.shape_);
            nd.CheckAndAllocData(values.shape_);
            nd.CheckAndAllocAuxData(csr::kIndPtr, indptr.shape_);
            // copy values, indices and indptr
            CopyFromTo(data_i.data(), values);
            CopyFromTo(data_i.aux_data(csr::kIdx), indices);
            CopyFromTo(data_i.aux_data(csr::kIndPtr), indptr);
          } else {
            LOG(FATAL) << "Storage type not implemented: " << stype;
          }
          data_iter += num_aux_data(stype) + 1;
          (*dptr)->num_batch_padd = batch.num_batch_padd;
        }
        if (batch.inst_index) {
          std::copy(batch.inst_index,
                    batch.inst_index + batch.batch_size,
                    (*dptr)->index.begin());
        }
       return true;
      },
      [this]() { sparse_loader_->BeforeFirst(); });
  }

  virtual void BeforeFirst(void) {
    PrefetcherIter::BeforeFirst();
  }

  virtual bool Next(void) {
    return PrefetcherIter::Next();
  }
  virtual const DataBatch &Value(void) const {
    return PrefetcherIter::Value();
  }

  virtual const NDArrayStorageType GetStorageType(bool is_data) const {
    return sparse_loader_->GetStorageType(is_data);
  }

  virtual const TShape GetShape(bool is_data) const {
    return sparse_loader_->GetShape(is_data);
  }

 private:
  /*! \brief internal sparse batch loader */
  SparseIIterator<TBlobBatch>* sparse_loader_;

  inline void CopyFromTo(TBlob dst, const TBlob src) {
    MSHADOW_TYPE_SWITCH(src.type_flag_, DType, {
      mshadow::Copy(dst.FlatTo1D<cpu, DType>(), src.FlatTo1D<cpu, DType>());
    });
  }
};
}  // namespace io
}  // namespace mxnet
#endif  // MXNET_IO_ITER_SPARSE_PREFETCHER_H_
