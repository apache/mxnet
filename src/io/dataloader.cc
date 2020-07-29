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
 * \file dataloader.cc
 * \brief Pure c++ backed dataloader implementation
 */
#include <dmlc/parameter.h>
#include <dmlc/omp.h>
#include <mxnet/io.h>

#include "./inst_vector.h"
#include "./iter_prefetcher.h"
#include "../profiler/custom_op_profiler.h"

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
          .describe("Pointer to shared Dataset.");
      DMLC_DECLARE_FIELD(sampler)
          .describe("Pointer to Sampler.");
      DMLC_DECLARE_FIELD(batchify_fn)
          .describe("Pointer to Batchify function.");
      DMLC_DECLARE_FIELD(pin_device_id).set_default(-1)
          .describe("If not negative, will move data to pinned memory.");
  }
};  // struct ThreadedDataLoaderParam

DMLC_REGISTER_PARAMETER(ThreadedDataLoaderParam);

template<typename DType = real_t>
class ThreadedDataLoader : public IIterator<TBlobBatch> {
 public:
  ThreadedDataLoader() = default;
  // destructor
  ~ThreadedDataLoader() override = default;
  // constructor
  void Init(const std::vector<std::pair<std::string, std::string> >& kwargs) override {
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
    param_.num_workers = std::max(1, threadget);
    dataset_ = *static_cast<std::shared_ptr<Dataset>*>(reinterpret_cast<void*>(param_.dataset));
    dataset_len_ = dataset_->GetLen();
    sampler_ = static_cast<IIterator<DataBatch>* >(reinterpret_cast<void*>(param_.sampler));
    batchify_fn_ = *static_cast<BatchifyFunctionPtr*>(reinterpret_cast<void*>(param_.batchify_fn));
    this->BeforeFirst();
  }
  // before first
  void BeforeFirst() override {
    sampler_->BeforeFirst();
  }

  int64_t GetLenHint() const override {
    return sampler_->GetLenHint();
  }

  bool Next() override {
    bool has_next = sampler_->Next();
    if (!has_next) return false;
    auto samples = sampler_->Value();
    auto batch_size = samples.data[0].shape().Size();
    int real_batch_size = batch_size - samples.num_batch_padd;
    const int64_t *idx_ptr = static_cast<int64_t*>(
        samples.data[0].data().dptr_);
    std::vector<int64_t> idx_ptrs;
    idx_ptrs.assign(idx_ptr, idx_ptr + real_batch_size);

    // __getitem__
    std::vector<std::vector<NDArray> > inputs(batch_size);
    std::vector<int> is_scalars;
    bool profiling = profiler::Profiler::Get()->IsProfiling(profiler::Profiler::kImperative);
    if (profiling) {
      profiler::CustomOpProfiler::Get()->OnCustomBegin("MXThreadedDataLoaderGetItems");
    }
    #pragma omp parallel for num_threads(param_.num_workers)
    for (int i = 0; i < real_batch_size; ++i) {
      omp_exc_.Run([&] {
        auto idx = idx_ptrs[i];
        CHECK(dataset_->GetItem(idx, &inputs[i]))
          << "Error getting data # " << idx;
      });
    }
    if (profiling) {
      profiler::CustomOpProfiler::Get()->OnCustomEnd();
    }
    omp_exc_.Rethrow();

    // pad to normal batch size
    for (size_t i = real_batch_size; i < batch_size; ++i) {
      inputs[i] = inputs[0];
    }

    // batchify
    if (profiling) {
      profiler::CustomOpProfiler::Get()->OnCustomBegin("MXThreadedDataLoaderBatchify");
    }
    CHECK(batchify_fn_->Batchify(inputs, &batched_buffer_))
      << "Error call batchify inside dataloader";
    if (profiling) {
      profiler::CustomOpProfiler::Get()->OnCustomEnd();
    }
    out_.batch_size = batched_buffer_.size();
    out_.data.resize(batched_buffer_.size());
    for (size_t i = 0; i < batched_buffer_.size(); ++i) {
      out_.data[i] = batched_buffer_[i].data();
    }
    out_.num_batch_padd = samples.num_batch_padd;
    return true;
  }

  const TBlobBatch &Value() const override {
    return out_;
  }

 private:
  /*! \brief Params */
  ThreadedDataLoaderParam param_;
  /*! \brief output */
  TBlobBatch out_;
  /*! \brief batched buffer */
  std::vector<NDArray> batched_buffer_;
  /*! \brief pointer to dataset */
  std::shared_ptr<Dataset> dataset_;
  /*! \brief dataset length */
  int64_t dataset_len_;
  /*! \brief pointer to sampler iterator */
  IIterator<DataBatch> *sampler_;
  /*! \brief pointer to batchify function */
  BatchifyFunctionPtr batchify_fn_;
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
