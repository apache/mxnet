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
 * \file sampler.cc
 * \brief The data sampler for access elements in dataset.
 */
#include <dmlc/parameter.h>
#include <mshadow/random.h>
#include <mxnet/base.h>
#include <mxnet/resource.h>
#include "../common/utils.h"
#include "./sampler.h"

namespace dmlc {
DMLC_REGISTRY_ENABLE(::mxnet::io::BatchSamplerReg);
}  // namespace dmlc

namespace mxnet {
namespace io {
namespace sampler_enum {
enum LastBatchType {kKeep, kDiscard, kRollover};
}

struct RandomSamplerParam : public dmlc::Parameter<RandomSamplerParam> {
    /*! \brief Length of the sequence. */
    size_t length;
    /*! \brief Random seed.*/
    int seed;
    // declare parameters
    DMLC_DECLARE_PARAMETER(RandomSamplerParam) {
        DMLC_DECLARE_FIELD(length)
            .describe("Length of the sequence.");
        DMLC_DECLARE_FIELD(seed).set_default(0)
            .describe("Random seed.");
    }
};  // struct RandomSamplerParam

DMLC_REGISTER_PARAMETER(RandomSamplerParam);

class RandomSampler : public IndexSampler {
  public:
    void Init(const std::vector<std::pair<std::string, std::string> >& kwargs) {
      param_.InitAllowUnknown(kwargs);
      indices_.resize(param_.length);
      std::iota(std::begin(indices_), std::end(indices_), 0);  // fill like arange
      rng_.reset(new common::RANDOM_ENGINE(kRandMagic + param_.seed));
    }
    void BeforeFirst(void) {
      std::shuffle(std::begin(indices_), std::end(indices_), *rng_);
      pos_ = 0;
    }
    bool Next(void) {
      if (pos_ < indices_.size()) {
        data_ = indices_[pos_];
        ++pos_;
        return true;
      }
      return false;
    }
    const int64_t &Value(void) const {
      return data_;
    }
  private:
    /*! \brief random magic number */
    static const int kRandMagic = 2333;
    /*! \brief Stored integer indices */
    std::vector<int64_t> indices_;
    /*! \brief current position for iteration */
    std::size_t pos_;
    /*! \brief data for next value */
    int64_t data_;
    /*! \brief random generator engine */
    std::unique_ptr<common::RANDOM_ENGINE> rng_;
    /*! \brief arguments */
    RandomSamplerParam param_;
};  // class RandomSampler

struct DefaultBatchSamplerParam : public dmlc::Parameter<DefaultBatchSamplerParam> {
    /*! \brief Name of the internal sampler it wraps. */
    std::string sampler_name;
    /*! \brief joined kwargs for internal sampler.
     * Example: "start:10;length:20" for SequentialSampler
     */
    std::string sampler_kwargs;
    /*! \brief Size of mini-batch. */
    size_t batch_size;
    /*! \brief Specifies how the last batch is handled if batch_size does not evenly
     *  divide sequence length. 
     */
    int last_batch;
    // declare parameters
    DMLC_DECLARE_PARAMETER(DefaultBatchSamplerParam) {
        DMLC_DECLARE_FIELD(sampler_name)
            .describe("Internal sampler name it wraps.");
        DMLC_DECLARE_FIELD(sampler_kwargs).set_default("")
            .describe("Additional kwargs for the internal sampler.");
        DMLC_DECLARE_FIELD(batch_size)
            .describe("Mini-batch size.");
        DMLC_DECLARE_FIELD(last_batch).set_default(sampler_enum::kKeep)
            .add_enum("keep", sampler_enum::kKeep)
            .add_enum("discard", sampler_enum::kDiscard)
            .add_enum("rollover", sampler_enum::kRollover)
            .describe("Specifies how the last batch is handled if batch_size does not evenly "
                      "divide sequence length. "
                      "If 'keep', the last batch will be returned directly, but will contain "
                      "less element than `batch_size` requires. "
                      "If 'discard', the last batch will be discarded. "
                      "If 'rollover', the remaining elements will be rolled over to the next "
                      "iteration.");
    }
};  // struct DefaultBatchSampler

DMLC_REGISTER_PARAMETER(DefaultBatchSamplerParam);

class DefaultBatchSampler : public BatchSampler {
  public:
    explicit DefaultBatchSampler(IndexSampler* base) : base_(base) {
    }
    void Init(const std::vector<std::pair<std::string, std::string> >& kwargs) {
      auto left_kwargs = param_.InitAllowUnknown(kwargs);
      base_->Init(left_kwargs);
      data_.reserve(param_.batch_size);
      prev_.reserve(param_.batch_size);
    }
    void BeforeFirst(void) {
      base_->BeforeFirst();
      data_.clear();
      std::swap(data_, prev_);
    }
    bool Next(void) {
      data_.clear();
      while (base_->Next()) {
        auto idx = base_->Value();
        data_.emplace_back(idx);
        if (data_.size() == param_.batch_size) {
        return true;
        }
      }
      if (data_.size() > 0) {
        if (sampler_enum::kKeep == param_.last_batch) {
          return true;
        } else if (sampler_enum::kDiscard == param_.last_batch) {
          data_.clear();
          return false;
        } else if (sampler_enum::kRollover == param_.last_batch) {
          std::swap(prev_, data_);
          data_.clear();
          return false;
        } else {
          LOG(FATAL)
            << "last_batch must be one of 'keep', 'discard', or 'rollover'";
        }
      }
      return false;
    }
    const BatchSample &Value(void) const {
      return data_;
    }
  private:
    /*! \brief rolled over indices from last epoch */
    BatchSample prev_;
    /*! \brief Stored data for next iteration */
    BatchSample data_;
    /*! \brief Arguments */
    DefaultBatchSamplerParam param_;
    /*! \brief Internal sampler */
    IndexSampler* base_;
};  // class DefaultBatchSampler

MXNET_REGISTER_IO_BATCH_SAMPLER(BatchRandomSampler)
 .describe("Batch Random sampler")
 .add_arguments(DefaultBatchSamplerParam::__FIELDS__())
 .add_arguments(RandomSamplerParam::__FIELDS__())
 .set_body([]() {
     return new DefaultBatchSampler(new RandomSampler());
   });
}  // namespace io
}  // namespace mxnet