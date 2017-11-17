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
 *  Copyright (c) 2015 by Contributors
 * \file iter_batchloader.h
 * \brief define a batch adapter to create tblob batch
 */
#ifndef MXNET_IO_ITER_BATCHLOADER_H_
#define MXNET_IO_ITER_BATCHLOADER_H_

#include <mxnet/io.h>
#include <mxnet/base.h>
#include <dmlc/logging.h>
#include <mshadow/tensor.h>
#include <utility>
#include <vector>
#include <string>
#include "./inst_vector.h"
#include "./image_iter_common.h"

namespace mxnet {
namespace io {

/*! \brief create a batch iterator from single instance iterator */
class BatchLoader : public IIterator<TBlobBatch> {
 public:
  explicit BatchLoader(IIterator<DataInst> *base):
    head_(1), num_overflow_(0), base_(base) {
  }

  virtual ~BatchLoader(void) {
    delete base_;
  }

  inline void Init(const std::vector<std::pair<std::string, std::string> >& kwargs) {
    std::vector<std::pair<std::string, std::string> > kwargs_left;
    // init batch param, it could have similar param with
    kwargs_left = param_.InitAllowUnknown(kwargs);
    // Init space for out
    out_.inst_index = new unsigned[param_.batch_size];
    out_.batch_size = param_.batch_size;
    out_.data.clear();
    // init base iterator
    base_->Init(kwargs);
  }

  virtual void BeforeFirst(void) {
    if (param_.round_batch == 0 || num_overflow_ == 0) {
      // otherise, we already called before first
      base_->BeforeFirst();
    } else {
      num_overflow_ = 0;
    }
    head_ = 1;
  }

  virtual bool Next(void) {
    out_.num_batch_padd = 0;
    out_.batch_size = param_.batch_size;
    this->head_ = 0;

    // if overflow from previous round, directly return false, until before first is called
    if (num_overflow_ != 0) return false;
    index_t top = 0;

    while (base_->Next()) {
      const DataInst& d = base_->Value();
      out_.inst_index[top] = d.index;
      if (data_.size() == 0) {
        this->InitData(d);
      }
      for (size_t i = 0; i < d.data.size(); ++i) {
        CHECK_EQ(unit_size_[i], d.data[i].Size());
        MSHADOW_TYPE_SWITCH(data_[i].type_flag_, DType, {
            mshadow::Copy(
              data_[i].get<cpu, 1, DType>().Slice(top * unit_size_[i],
                                                  (top + 1) * unit_size_[i]),
              d.data[i].get_with_shape<cpu, 1, DType>(mshadow::Shape1(unit_size_[i])));
          });
      }
      if (++top >= param_.batch_size) {
        return true;
      }
    }
    if (top != 0) {
      if (param_.round_batch != 0) {
        num_overflow_ = 0;
        base_->BeforeFirst();
        for (; top < param_.batch_size; ++top, ++num_overflow_) {
          CHECK(base_->Next()) << "number of input must be bigger than batch size";
          const DataInst& d = base_->Value();
          out_.inst_index[top] = d.index;
          // copy data
          for (size_t i = 0; i < d.data.size(); ++i) {
            CHECK_EQ(unit_size_[i], d.data[i].Size());
            MSHADOW_TYPE_SWITCH(data_[i].type_flag_, DType, {
                mshadow::Copy(
                  data_[i].get<cpu, 1, DType>().Slice(top * unit_size_[i],
                                                      (top + 1) * unit_size_[i]),
                  d.data[i].get_with_shape<cpu, 1, DType>(mshadow::Shape1(unit_size_[i])));
              });
          }
        }
        out_.num_batch_padd = num_overflow_;
      } else {
        out_.num_batch_padd = param_.batch_size - top;
      }
      return true;
    }
    return false;
  }
  virtual const TBlobBatch &Value(void) const {
    return out_;
  }

 protected:
  /*! \brief batch parameters */
  BatchParam param_;
  /*! \brief output data */
  TBlobBatch out_;
  /*! \brief on first */
  int head_;
  /*! \brief number of overflow instances that readed in round_batch mode */
  int num_overflow_;
  /*! \brief tensor to hold data */
  std::vector<TBlobContainer> data_;

 private:
  /*! \brief base iterator */
  IIterator<DataInst> *base_;
  /*! \brief data shape */
  std::vector<TShape> shape_;
  /*! \brief unit size */
  std::vector<size_t> unit_size_;
  // initialize the data holder by using from the first batch.
  inline void InitData(const DataInst& first_batch) {
    shape_.resize(first_batch.data.size());
    data_.resize(first_batch.data.size());
    unit_size_.resize(first_batch.data.size());
    for (size_t i = 0; i < first_batch.data.size(); ++i) {
      TShape src_shape = first_batch.data[i].shape_;
      int src_type_flag = first_batch.data[i].type_flag_;
      // init object attributes
      std::vector<index_t> shape_vec;
      shape_vec.push_back(param_.batch_size);
      for (index_t dim = 0; dim < src_shape.ndim(); ++dim) {
        shape_vec.push_back(src_shape[dim]);
      }
      TShape dst_shape(shape_vec.begin(), shape_vec.end());
      shape_[i] = dst_shape;
      data_[i].resize(mshadow::Shape1(dst_shape.Size()), src_type_flag);
      unit_size_[i] = src_shape.Size();
      out_.data.push_back(TBlob(data_[i].dptr_, dst_shape, cpu::kDevMask, src_type_flag, 0));
    }
  }
};  // class BatchLoader
}  // namespace io
}  // namespace mxnet
#endif  // MXNET_IO_ITER_BATCHLOADER_H_
