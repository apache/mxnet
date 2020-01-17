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
 * \file dataloader.cc
 * \brief Pure c++ backed dataloader implementation
 */
#include <mxnet/io.h>

namespace mxnet {
namespace io {
template<typename DType = real_t>
class ThreadedDataLoader : public IIterator<DataBatch> {
 public:
  ImageRecordIter() : data_(nullptr) { }
  // destructor
  virtual ~ImageRecordIter(void) {
    iter_.Destroy();
    delete data_;
  }
  // constructor
  virtual void Init(const std::vector<std::pair<std::string, std::string> >& kwargs) {
    param_.InitAllowUnknown(kwargs);
    // use the kwarg to init parser
    parser_.Init(kwargs);
    // prefetch at most 4 minbatches
    iter_.set_max_capacity(4);
    // init thread iter
    iter_.Init([this](std::vector<InstVector<DType>> **dptr) {
        if (*dptr == nullptr) {
          *dptr = new std::vector<InstVector<DType>>();
        }
        return parser_.ParseNext(*dptr);
      },
      [this]() { parser_.BeforeFirst(); });
    inst_ptr_ = 0;
    rnd_.seed(kRandMagic + param_.seed);
  }
  // before first
  virtual void BeforeFirst(void) {
    iter_.BeforeFirst();
    inst_order_.clear();
    inst_ptr_ = 0;
  }

  virtual bool Next(void) {
    while (true) {
      if (inst_ptr_ < inst_order_.size()) {
        std::pair<unsigned, unsigned> p = inst_order_[inst_ptr_];
        out_ = (*data_)[p.first][p.second];
        ++inst_ptr_;
        return true;
      } else {
        if (data_ != nullptr) iter_.Recycle(&data_);
        if (!iter_.Next(&data_)) return false;
        inst_order_.clear();
        for (unsigned i = 0; i < data_->size(); ++i) {
          const InstVector<DType>& tmp = (*data_)[i];
          for (unsigned j = 0; j < tmp.Size(); ++j) {
            inst_order_.push_back(std::make_pair(i, j));
          }
        }
        // shuffle instance order if needed
        if (param_.shuffle != 0) {
          std::shuffle(inst_order_.begin(), inst_order_.end(), rnd_);
        }
        inst_ptr_ = 0;
      }
    }
    return false;
  }

  virtual const DataBatch &Value(void) const {
    return out_;
  }

 private:
  // random magic
  static const int kRandMagic = 111;
  // output instance
  DataBatch out_;
  // data ptr
  size_t inst_ptr_;
  // internal instance order
  std::vector<std::pair<unsigned, unsigned> > inst_order_;
  // data
  std::vector<InstVector<DType>> *data_;
  // internal parser
  ImageRecordIOParser<DType> parser_;
  // backend thread
  dmlc::ThreadedIter<std::vector<InstVector<DType>> > iter_;
  // parameters
  ImageRecordParam param_;
  // random number generator
  common::RANDOM_ENGINE rnd_;
};

// OLD VERSION - DEPRECATED
MXNET_REGISTER_IO_ITER(ImageRecordIter_v1)
.describe(R"code(Threaded DataLoader. 


Help loading the data with combined c++ version of ``dataset``, ``sampler``, 
``batch_sampler``, and ``batchify_fn``.


)code" ADD_FILELINE)
.add_arguments(ImageRecParserParam::__FIELDS__())
.add_arguments(ImageRecordParam::__FIELDS__())
.add_arguments(BatchParam::__FIELDS__())
.add_arguments(PrefetcherParam::__FIELDS__())
.add_arguments(ListDefaultAugParams())
.add_arguments(ImageNormalizeParam::__FIELDS__())
.set_body([]() {
    return new PrefetcherIter(
        new BatchLoader(
            new ImageNormalizeIter(
                new ImageRecordIter<real_t>())));
  });
}  // namespace io
}  // namespace mxnet