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
 *  Copyright (c) 2020 by Contributors
 * \file iter_sampler.cc
 * \brief The sampler iterator for access dataset elements.
 */
#include <dmlc/parameter.h>
#include <mshadow/random.h>
#include <mxnet/io.h>
#include <mxnet/base.h>
#include <mxnet/resource.h>
#include <memory>
#include <numeric>
#include "../common/utils.h"
#include "./iter_batchloader.h"
#include "./iter_prefetcher.h"

namespace mxnet {
namespace io {
struct SequentialSamplerParam : public dmlc::Parameter<SequentialSamplerParam> {
  /*! \brief Length of the sequence. */
  size_t length;
  /*! \brief start index.*/
  int start;
  // declare parameters
  DMLC_DECLARE_PARAMETER(SequentialSamplerParam) {
      DMLC_DECLARE_FIELD(length)
          .describe("Length of the sequence.");
      DMLC_DECLARE_FIELD(start).set_default(0)
          .describe("Start of the index.");
  }
};  // struct SequentialSamplerParam

DMLC_REGISTER_PARAMETER(SequentialSamplerParam);

class SequentialSampler : public IIterator<DataInst> {
 public:
  void Init(const std::vector<std::pair<std::string, std::string> >& kwargs) override {
    param_.InitAllowUnknown(kwargs);
    indices_.resize(param_.length);
    std::iota(std::begin(indices_), std::end(indices_), 0);  // fill like arange
    out_.data.resize(1);
  }

  void BeforeFirst() override {
    pos_ = 0;
  }

  int64_t GetLenHint() const override {
    return static_cast<int64_t>(indices_.size());
  }

  bool Next() override {
    if (pos_ < indices_.size()) {
      int64_t *ptr = indices_.data() + pos_;
      out_.data[0] = TBlob(ptr, TShape({1, }), cpu::kDevMask, 0);
      ++pos_;
      return true;
    }
    return false;
  }

  const DataInst &Value() const override {
    return out_;
  }

 private:
  /*! \brief Stored integer indices */
  std::vector<int64_t> indices_;
  /*! \brief current position for iteration */
  std::size_t pos_;
  /*! \brief data for next value */
  DataInst out_;
  /*! \brief arguments */
  SequentialSamplerParam param_;
};  // class SequentialSampler

MXNET_REGISTER_IO_ITER(SequentialSampler)
.describe(R"code(Returns the sequential sampler iterator.
)code" ADD_FILELINE)
.add_arguments(SequentialSamplerParam::__FIELDS__())
.add_arguments(BatchSamplerParam::__FIELDS__())
.set_body([]() {
    return
        new BatchSampler(
            new SequentialSampler());
  });

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

class RandomSampler : public IIterator<DataInst> {
 public:
  void Init(const std::vector<std::pair<std::string, std::string> >& kwargs) override {
    param_.InitAllowUnknown(kwargs);
    indices_.resize(param_.length);
    std::iota(std::begin(indices_), std::end(indices_), 0);  // fill like arange
    mshadow::Random<cpu> *ctx_rng = ResourceManager::Get()->Request(
      Context::CPU(), ResourceRequest::kRandom).get_random<cpu, real_t>(nullptr);
    rng_ = std::make_unique<common::RANDOM_ENGINE>(ctx_rng->GetSeed());
    out_.data.resize(1);
    BeforeFirst();
  }

  void BeforeFirst() override {
    std::shuffle(std::begin(indices_), std::end(indices_), *rng_);
    pos_ = 0;
  }

  int64_t GetLenHint() const override {
    return static_cast<int64_t>(indices_.size());
  }

  bool Next() override {
    if (pos_ < indices_.size()) {
      int64_t *ptr = indices_.data() + pos_;
      out_.data[0] = TBlob(ptr, TShape({1, }), cpu::kDevMask, 0);
      ++pos_;
      return true;
    }
    return false;
  }

  const DataInst &Value() const override {
    return out_;
  }
 private:
  /*! \brief Stored integer indices */
  std::vector<int64_t> indices_;
  /*! \brief current position for iteration */
  std::size_t pos_;
  /*! \brief data for next value */
  DataInst out_;
  /*! \brief random generator engine */
  std::unique_ptr<std::mt19937> rng_;
  /*! \brief arguments */
  RandomSamplerParam param_;
};  // class RandomSampler

MXNET_REGISTER_IO_ITER(RandomSampler)
.describe(R"code(Returns the random sampler iterator.
)code" ADD_FILELINE)
.add_arguments(RandomSamplerParam::__FIELDS__())
.add_arguments(BatchSamplerParam::__FIELDS__())
.set_body([]() {
    return new BatchSampler(
            new RandomSampler());
  });

}  // namespace io
}  // namespace mxnet
