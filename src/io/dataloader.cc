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
#include <dmlc/parameter.h>
#include <dmlc/omp.h>
#include <mxnet/io.h>

#include "./inst_vector.h"

namespace mxnet {
namespace io {
struct ThreadedDataLoaderParam : public dmlc::Parameter<ThreadedDataLoaderParam> {
    /*! \brief Multithread worker number. */
    int num_worker;
    /*! \brief dataset pointer.*/
    std::intptr_t dataset;
    /*! \brief sampler pointer.*/
    std::intptr_t sampler;
    /*! \brief batchify function pointer.*/
    std::intptr_t batchify_fn;
    /*! \brief pin memory to device id.*/
    int pin_device_id;
    // declare parameters
    DMLC_DECLARE_PARAMETER(ThreadedDataLoaderParam) {
        DMLC_DECLARE_FIELD(num_worker).set_default(0)
            .describe("Number of thread workers.");
        DMLC_DECLARE_FIELD(dataset)
            .describe("Number of thread workers.");
        DMLC_DECLARE_FIELD(sampler)
            .describe("Number of thread workers.");
        DMLC_DECLARE_FIELD(batchify_fn)
            .describe("Number of thread workers.");
        DMLC_DECLARE_FIELD(pin_device_id).set_default(-1)
            .describe("If not negative, will move data to pinned memory.");
    }
};  // struct ThreadedDataLoaderParam

DMLC_REGISTER_PARAMETER(ThreadedDataLoaderParam);

template<typename DType = real_t>
class ThreadedDataLoader : public IIterator<TBlobBatch> {
 public:
  // destructor
  virtual ~ThreadedDataLoader(void) {
  }
  // constructor
  virtual void Init(const std::vector<std::pair<std::string, std::string> >& kwargs) {
    param_.InitAllowUnknown(kwargs);
    int maxthread, threadget;
    #pragma omp parallel
    {
      // be conservative, set number of real cores
      maxthread = std::max(omp_get_num_procs(), 1);
    }
    param_.num_worker = std::min(maxthread, param_.num_worker);
    #pragma omp parallel num_threads(param_.num_worker)
    {
      threadget = omp_get_num_threads();
    }
    param_.num_worker = threadget;
    dataset_ = static_cast<Dataset*>(reinterpret_cast<void*>(param_.dataset));
    sampler_ = static_cast<IIterator<DataBatch>* >(reinterpret_cast<void*>(param_.sampler));
    batchify_fn_ = static_cast<BatchifyFunction*>(reinterpret_cast<void*>(param_.batchify_fn));
  }
  // before first
  virtual void BeforeFirst(void) {
    sampler_->BeforeFirst();
  }

  virtual int64_t GetLenHint(void) const {
    return sampler_->GetLenHint();
  }

  virtual bool Next(void) {
    bool has_next = sampler_->Next();
    if (!has_next) return false;
    auto samples = sampler_->Value();
    auto batch_size = samples.data[0].shape().Size();
    return true;
  }

  virtual const DataBatch &Value(void) const {
  }

  private:
    /*! \brief Params */
    ThreadedDataLoaderParam param_;
    /*! \brief pointer to dataset */
    Dataset *dataset_;
    /*! \brief pointer to sampler iterator */
    IIterator<DataBatch> *sampler_;
    /*! \brief pointer to batchify function */
    BatchifyFunction *batchify_fn_;
  
};  // class ThreadedDataLoader
}  // namespace io
}  // namespace mxnet