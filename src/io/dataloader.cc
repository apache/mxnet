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
#include "./iter_prefetcher.h"

namespace mxnet {
namespace io {
struct ThreadedDataLoaderParam : public dmlc::Parameter<ThreadedDataLoaderParam> {
    /*! \brief Multithread worker number. */
    int num_workers;
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
        DMLC_DECLARE_FIELD(num_workers).set_default(0)
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
  void Init(const std::vector<std::pair<std::string, std::string> >& kwargs) {
    param_.InitAllowUnknown(kwargs);
    int maxthread, threadget;
    #pragma omp parallel
    {
      // be conservative, set number of real cores
      maxthread = std::max(omp_get_num_procs(), 1);
    }
    param_.num_workers = std::min(maxthread, param_.num_workers);
    #pragma omp parallel num_threads(param_.num_workers)
    {
      threadget = omp_get_num_threads();
    }
    param_.num_workers = threadget;
    dataset_ = static_cast<Dataset*>(reinterpret_cast<void*>(param_.dataset));
    dataset_len_ = dataset_->GetLen();
    item_size_ = dataset_->GetOutputSize();
    sampler_ = static_cast<IIterator<DataBatch>* >(reinterpret_cast<void*>(param_.sampler));
    batchify_fn_ = static_cast<BatchifyFunction*>(reinterpret_cast<void*>(param_.batchify_fn));
    this->BeforeFirst();
  }
  // before first
  void BeforeFirst(void) {
    sampler_->BeforeFirst();
  }

  int64_t GetLenHint(void) const {
    return sampler_->GetLenHint();
  }

  bool Next(void) {
    bool has_next = sampler_->Next();
    if (!has_next) return false;
    auto samples = sampler_->Value();
    auto batch_size = samples.data[0].shape().Size();
    if (samples.num_batch_padd > 0) {
        // when last batch is keep but not fully filled
        // effective batch size is smaller
        batch_size -= samples.num_batch_padd;
    }
    const int64_t *idx_ptr = static_cast<int64_t*>(
        samples.data[0].data().dptr_);

    // __getitem__
    std::vector<std::vector<NDArray> > inputs(batch_size);
    for (size_t i = 0; i < batch_size; ++i) {
      // omp_exc_.Run([&] {
        inputs[i].resize(item_size_);
      // });
    }
    // omp_exc_.Rethrow();
    size_t workload = batch_size * item_size_;
    // #pragma omp parallel for num_threads(param_.num_workers)
      for (size_t i = 0; i < workload; ++i) {
        // omp_exc_.Run([&] {
          size_t x = i / item_size_;
          size_t y = i % item_size_;
          int is_scalar;
          inputs[x][y] = std::move(
              dataset_->GetItem(idx_ptr[x], y, &is_scalar));
        // });
      }
      // omp_exc_.Rethrow();

    // batchify
    auto batched_data = batchify_fn_->Batchify(inputs);
    out_.batch_size = batched_data.size();
    out_.data = batched_data;
    // #pragma omp parallel for num_threads(param_.num_workers)
      // for (size_t i = 0; i < data_.size(); ++i) {
      //   // omp_exc_.Run([&] {
      //     out_.data[i] = data_[i];
      //   // });
      // }
      // omp_exc_.Rethrow();
    return true;
  }

  const TBlobBatch &Value(void) const {
    return out_;
  }

  private:
    /*! \brief Params */
    ThreadedDataLoaderParam param_;
    /*! \brief output */
    TBlobBatch out_;
    /*! \brief pointer to dataset */
    Dataset *dataset_;
    /*! \brief dataset length */
    int64_t dataset_len_;
    /*! \brief dataset output size */
    int item_size_;
    /*! \brief pointer to sampler iterator */
    IIterator<DataBatch> *sampler_;
    /*! \brief pointer to batchify function */
    BatchifyFunction *batchify_fn_;
    /*! \brief OMPException obj to store and rethrow exceptions from omp blocks*/
    dmlc::OMPException omp_exc_;
};  // class ThreadedDataLoader

MXNET_REGISTER_IO_ITER(ThreadedDataLoader)
.describe(R"code(Returns a threaded data loader iterator.
)code" ADD_FILELINE)
.add_arguments(ThreadedDataLoaderParam::__FIELDS__())
.add_arguments(PrefetcherParam::__FIELDS__())
.set_body([]() {
    return new PrefetcherIter(
            new ThreadedDataLoader<mxnet::real_t>());
  });
}  // namespace io
}  // namespace mxnet
