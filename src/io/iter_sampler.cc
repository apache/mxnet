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
 * \file iter_sampler.cc
 * \brief The sampler iterator for access dataset elements.
 */
#include <dmlc/parameter.h>
#include <mshadow/random.h>
#include <mxnet/io.h>
#include <mxnet/base.h>
#include <mxnet/resource.h>
#include "../common/utils.h"
#include "./iter_batchloader.h"
#include "./iter_prefetcher.h"

namespace mxnet {
namespace io {
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

class RandomSampler : public IIterator<DataInst> {
  public:
    virtual void Init(const std::vector<std::pair<std::string, std::string> >& kwargs) {
      param_.InitAllowUnknown(kwargs);
      indices_.resize(param_.length);
      std::iota(std::begin(indices_), std::end(indices_), 0);  // fill like arange
      rng_.reset(new common::RANDOM_ENGINE(kRandMagic + param_.seed));
      out_.data.resize(2);  // label required by DataBatch, we can use fake label here
      out_.data[1] = TBlob(indices_.data(), TShape({1,}), cpu::kDevMask, 0);
      BeforeFirst();
    }

    virtual void BeforeFirst(void) {
      std::shuffle(std::begin(indices_), std::end(indices_), *rng_);
      pos_ = 0;
    }

    virtual int64_t GetLenHint(void) const {
      return static_cast<int64_t>(indices_.size());
    }

    virtual bool Next(void) {
      if (pos_ < indices_.size()) {
        int64_t *ptr = indices_.data() + pos_;
        out_.data[0] = TBlob(ptr, TShape({1,}), cpu::kDevMask, 0);
        ++pos_;
        return true;
      }
      return false;
    }

    virtual const DataInst &Value(void) const {
      return out_;
    }
  private:
    /*! \brief random magic number */
    static const int kRandMagic = 2333;
    /*! \brief Stored integer indices */
    std::vector<int64_t> indices_;
    /*! \brief current position for iteration */
    std::size_t pos_;
    /*! \brief data for next value */
    DataInst out_;
    /*! \brief random generator engine */
    std::unique_ptr<common::RANDOM_ENGINE> rng_;
    /*! \brief arguments */
    RandomSamplerParam param_;
};  // class RandomSampler

MXNET_REGISTER_IO_ITER(RandomSamplerIter)
.describe(R"code(Returns the random sampler iterator.
)code" ADD_FILELINE)
.add_arguments(RandomSamplerParam::__FIELDS__())
.add_arguments(BatchParam::__FIELDS__())
.add_arguments(PrefetcherParam::__FIELDS__())
.set_body([]() {
    return new PrefetcherIter(
        new BatchLoader(
            new RandomSampler()));
  });

}  // namespace io
}  // namespace mxnet