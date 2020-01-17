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
DMLC_REGISTRY_ENABLE(::mxnet::io::DataSamplerReg);
DMLC_REGISTRY_ENABLE(::mxnet::io::DataBatchSamplerReg);
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

class RandomSampler : public DatasetSampler<int64_t> {
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

MXNET_REGISTER_IO_SAMPLER(RandomSampler)
 .describe("Random sampler")
 .add_arguments(RandomSamplerParam::__FIELDS__())
 .set_body([]() {
     return new RandomSampler();
   });

struct BatchSamplerParam : public dmlc::Parameter<BatchSamplerParam> {
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
    DMLC_DECLARE_PARAMETER(BatchSamplerParam) {
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
};  // struct BatchSamplerParam

DMLC_REGISTER_PARAMETER(BatchSamplerParam);

class BatchSampler : public DatasetSampler<BatchSample> {
  public:
    void Init(const std::vector<std::pair<std::string, std::string> >& kwargs) {
      param_.InitAllowUnknown(kwargs);
      auto sampler_kwargs = ParseKwargs(param_.sampler_kwargs);
      sampler_.reset(dmlc::Registry<DataSamplerReg>::Find(param_.sampler_name)->body());
      sampler_->Init(sampler_kwargs);
      data_.reserve(param_.batch_size);
      prev_.reserve(param_.batch_size);
    }
    void BeforeFirst(void) {
      sampler_->BeforeFirst();
      data_.clear();
      std::swap(data_, prev_);
    }
    bool Next(void) {
      data_.clear();
      while (sampler_->Next()) {
        auto idx = sampler_->Value();
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
    /*! \brief Stored internal sampler which provides index per next() */
    std::unique_ptr<DatasetSampler<int64_t> > sampler_;
    /*! \brief rolled over indices from last epoch */
    BatchSample prev_;
    /*! \brief Stored data for next iteration */
    BatchSample data_;
    /*! \brief Arguments */
    BatchSamplerParam param_;

    std::vector<std::pair<std::string, std::string> > ParseKwargs(std::string &kwargs) {
        std::vector<std::pair<std::string, std::string> > ret;
        auto pairs = dmlc::Split(kwargs, ';');
        for (auto pair : pairs) {
            auto arg = dmlc::Split(pair, ':');
            CHECK_EQ(arg.size(), 2)
                << "Unable to parse argument pair: " << pair;
            ret.emplace_back(std::make_pair(arg[0], arg[1]));
        }
        return ret;
    }
};  // class BatchSampler

}  // namespace io
}  // namespace mxnet