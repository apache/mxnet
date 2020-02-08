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
  ThreadedDataLoader() {
  }
  // destructor
  virtual ~ThreadedDataLoader(void) {
    for (size_t i = 0; i < vars_.size(); ++i) {
      Engine::Get()->DeleteVariable([](mxnet::RunContext ctx) {}, Context::CPU(), vars_[i]);
    }
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
    param_.num_workers = std::max(1, threadget);
    dataset_ = *static_cast<DatasetPtr*>(reinterpret_cast<void*>(param_.dataset));
    // datasets_.clear();
    // datasets_.reserve(param_.num_workers);
    // datasets_.emplace_back(dataset);
    // for (int i = 1; i < param_.num_workers; ++i) {
    //   datasets_.emplace_back(DatasetPtr(dataset->Clone()));
    // }
    dataset_len_ = dataset_->GetLen();
    sampler_ = static_cast<IIterator<DataBatch>* >(reinterpret_cast<void*>(param_.sampler));
    batchify_fn_ = *static_cast<BatchifyFunctionPtr*>(reinterpret_cast<void*>(param_.batchify_fn));
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
    auto real_batch_size = batch_size - samples.num_batch_padd;
    const int64_t *idx_ptr = static_cast<int64_t*>(
        samples.data[0].data().dptr_);

    // __getitem__
    std::vector<std::vector<NDArray> > inputs(batch_size);
    std::vector<int> is_scalars;
    // #pragma omp parallel for num_threads(param_.num_workers)
    // const auto engine = Engine::Get();
    for (size_t i = 0; i < real_batch_size; ++i) {
      // omp_exc_.Run([&] {
      auto idx = idx_ptr[i];
      std::vector<int> is_temp;
      inputs[i] = dataset_->GetItem(idx, is_temp);
      if (i == 0) {
        is_scalars = is_temp;
      }
      // if (vars_.size() <= i) {
      //   vars_.emplace_back(engine->NewVariable());
      // }
      // engine->PushSync(
      //   [this, &inputs, &is_scalars, &idx, &i](RunContext ctx) {
      //     std::vector<int> is_temp;
      //     LOG(INFO) << "before get";
      //     inputs[i] = dataset_->GetItem(idx, is_temp);
      //     LOG(INFO) << "after get";
      //     if (i == 0) {
      //       is_scalars = is_temp;
      //     }
      //   },
      //   Context::CPU(), {}, {vars_[i]}, FnProperty::kNormal, 0, "DataLoaderGetPerItem");
        
      // });
    }
    // omp_exc_.Rethrow();

    // for (size_t i = 0; i < real_batch_size; ++i) {
    //   engine->WaitForVar(vars_[i]);
    // }
    // pad to normal batch size
    for (size_t i = real_batch_size; i < batch_size; ++i) {
      inputs[i] = inputs[0];
    }

    // batchify
    auto batched_data = batchify_fn_->Batchify(inputs);
    for (size_t i = 0; i < is_scalars.size(); ++i) {
      if (is_scalars[i] == 1) {
        // batched scalar array should have dim 1 not 2
        CHECK_EQ(batched_data[i].ndim(), 2);
        batched_data[i] = batched_data[i].reshape(
          TShape({static_cast<dim_t>(batched_data[i].Size())}));
      }
    }
    out_.batch_size = batched_data.size();
    out_.data = batched_data;
    out_.num_batch_padd = samples.num_batch_padd;
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
    DatasetPtr dataset_;
    /*! \brief dataset length */
    int64_t dataset_len_;
    /*! \brief pointer to sampler iterator */
    IIterator<DataBatch> *sampler_;
    /*! \brief pointer to batchify function */
    BatchifyFunctionPtr batchify_fn_;
    /*! \brief engine variable */
    std::vector<Engine::VarHandle> vars_;
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
