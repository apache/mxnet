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
#include "./sampler.h"

namespace mxnet {
namespace io {
struct RandomSamplerParam : public dmlc::Parameter<RandomSamplerParam> {
    /*! \brief Length of the sequence. */
    size_t length;
    // declare parameters
    DMLC_DECLARE_PARAMETER(RandomSamplerParam) {
        DMLC_DECLARE_FIELD(length)
            .describe("Length of the sequence.");
    }
};  // struct RandomSamplerParam

DMLC_REGISTER_PARAMETER(RandomSamplerParam);

class RandomSampler : public DatasetSampler<int64_t> {
  public:
    virtual void Init(const std::vector<std::pair<std::string, std::string> >& kwargs) {
      param_.InitAllowUnknown(kwargs);
      indices_.resize(param_.length);
      std::iota(std::begin(indices_), std::end(indices_), 0);  // fill like arange
      rng_ = mshadow::Random<mxnet::cpu, int64_t>::GetRndEngine();
    }
    virtual void BeforeFirst(void) {
      std::shuffle(std::begin(indices_), std::end(indices_), rng_);
      pos_ = 0;
    }
    virtual bool Next(void) {
      if (pos_ < indices_.size()) {
        data_ = indices_[pos_];
        ++pos_;
        return true;
      }
      return false;
    }
    virtual const int64_t &Value(void) const {
      return data_;
    }
  private:
    std::vector<int64_t> indices_;
    std::size_t pos_;
    int64_t data_;
    std::mt19937 &rng_;
    RandomSamplerParam param_;
};  // class RandomSampler

MXNET_REGISTER_IO_SAMPLER(RandomSampler)
 .describe("Random sampler")
 .add_arguments(RandomSamplerParam::__FIELDS__())
 .set_body([]() {
     return new RandomSampler();
   });

class BatchSampler : public DatasetSampler<BatchSample> {

};  // class BatchSampler
}  // namespace io
}  // namespace mxnet