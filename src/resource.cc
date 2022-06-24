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
 * \file resource.cc
 * \brief Implementation of resource manager.
 */
#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <dmlc/thread_local.h>
#include <mxnet/base.h>
#include <mxnet/engine.h>
#include <mxnet/random_generator.h>
#include <mxnet/resource.h>
#include <limits>
#include <atomic>
#include <memory>
#include "./common/lazy_alloc_array.h"
#include "./common/utils.h"
#include "./common/cuda/utils.h"
#include "./profiler/storage_profiler.h"

namespace mxnet {
namespace resource {

// internal structure for space allocator
struct SpaceAllocator {
  // internal context
  Context ctx;
  // internal handle
  Storage::Handle handle;
  // internal CPU handle
  Storage::Handle host_handle;

  SpaceAllocator() {
    handle.dptr      = nullptr;
    handle.size      = 0;
    host_handle.dptr = nullptr;
    host_handle.size = 0;
  }

  inline void ReleaseAll() {
    Storage::Get()->DirectFree(handle);
    handle.dptr = nullptr;
    handle.size = 0;

    Storage::Get()->DirectFree(host_handle);
    host_handle.dptr = nullptr;
    host_handle.size = 0;
  }

  inline void* GetSpace(size_t size, const std::string& name) {
    if (handle.size >= size)
      return handle.dptr;

    Storage::Get()->DirectFree(handle);
    handle                = Storage::Get()->Alloc(size, ctx);
    handle.profiler_scope = "resource:";
    handle.name           = name;
#if MXNET_USE_CUDA
    profiler::GpuDeviceStorageProfiler::Get()->UpdateStorageInfo(handle);
#endif  // MXNET_USE_CUDA
    return handle.dptr;
  }

  inline void* GetHostSpace(size_t size) {
    if (host_handle.size >= size)
      return host_handle.dptr;

    Storage::Get()->DirectFree(host_handle);
    host_handle = Storage::Get()->Alloc(size, Context());
    return host_handle.dptr;
  }
};

// Implements resource manager
class ResourceManagerImpl : public ResourceManager {
 public:
  ResourceManagerImpl() noexcept(false) {
    cpu_temp_space_copy_  = dmlc::GetEnv("MXNET_CPU_TEMP_COPY", 4);
    gpu_temp_space_copy_  = dmlc::GetEnv("MXNET_GPU_TEMP_COPY", 1);
    cpu_native_rand_copy_ = dmlc::GetEnv("MXNET_CPU_PARALLEL_RAND_COPY", 1);
    gpu_native_rand_copy_ = dmlc::GetEnv("MXNET_GPU_PARALLEL_RAND_COPY", 1);
#if MXNET_USE_CUDNN == 1
    gpu_cudnn_dropout_state_copy_ = dmlc::GetEnv("MXNET_GPU_CUDNN_DROPOUT_STATE_COPY", 1);
#endif  // MXNET_USE_CUDNN == 1
    engine_ref_  = Engine::_GetSharedRef();
    storage_ref_ = Storage::_GetSharedRef();
    cpu_rand_    = std::make_unique<ResourceRandom<cpu>>(Context::CPU(), global_seed_);
    cpu_space_   = std::make_unique<ResourceTempSpace<ResourceRequest::kTempSpace>>(
        Context::CPU(), cpu_temp_space_copy_);
    cpu_parallel_rand_ = std::make_unique<ResourceParallelRandom<cpu>>(
        Context::CPU(), cpu_native_rand_copy_, global_seed_);
  }
  ~ResourceManagerImpl() override {
    // need explicit delete, before engine get killed
    cpu_rand_.reset(nullptr);
    cpu_space_.reset(nullptr);
    cpu_parallel_rand_.reset(nullptr);
#if MXNET_USE_CUDA
    gpu_rand_.Clear();
    gpu_space_.Clear();
    gpu_parallel_rand_.Clear();
#if MXNET_USE_CUDNN == 1
    gpu_cudnn_dropout_state_.Clear();
#endif  // MXNET_USE_CUDNN == 1
#endif
    if (engine_ref_ != nullptr) {
      engine_ref_ = nullptr;
    }
    if (storage_ref_ != nullptr) {
      storage_ref_ = nullptr;
    }
  }

  // request resources
  Resource Request(Context ctx, const ResourceRequest& req) override {
    if (ctx.dev_mask() == Context::kCPU) {
      switch (req.type) {
        case ResourceRequest::kRandom:
          return cpu_rand_->resource;
        case ResourceRequest::kTempSpace:
          return cpu_space_->GetNext();
        case ResourceRequest::kParallelRandom:
          return cpu_parallel_rand_->GetNext();
        default:
          LOG(FATAL) << "Unknown supported type " << req.type;
      }
    } else {
      CHECK_EQ(ctx.dev_mask(), Context::kGPU);
#if MSHADOW_USE_CUDA
      switch (req.type) {
        case ResourceRequest::kRandom: {
          return gpu_rand_
              .Get(ctx.dev_id, [ctx, this]() { return new ResourceRandom<gpu>(ctx, global_seed_); })
              ->resource;
        }
        case ResourceRequest::kTempSpace: {
          return gpu_space_
              .Get(ctx.dev_id,
                   [ctx, this]() {
                     return new ResourceTempSpace<ResourceRequest::kTempSpace>(
                         ctx, gpu_temp_space_copy_);
                   })
              ->GetNext();
        }
        case ResourceRequest::kParallelRandom: {
          return gpu_parallel_rand_
              .Get(ctx.dev_id,
                   [ctx, this]() {
                     return new ResourceParallelRandom<gpu>(
                         ctx, gpu_native_rand_copy_, global_seed_);
                   })
              ->GetNext();
        }
#if MXNET_USE_CUDNN == 1
        case ResourceRequest::kCuDNNDropoutDesc: {
          return gpu_cudnn_dropout_state_
              .Get(ctx.dev_id,
                   [ctx, this]() {
                     return new ResourceCUDNNDropout(
                         ctx, gpu_cudnn_dropout_state_copy_, global_seed_);
                   })
              ->GetNext();
        }
#endif  // MXNET_USE_CUDNN == 1
        default:
          LOG(FATAL) << "Unknown supported type " << req.type;
      }
#else
      LOG(FATAL) << MXNET_GPU_NOT_ENABLED_ERROR;
#endif
    }
    Resource ret;
    return ret;
  }

  void SeedRandom(uint32_t seed) override {
    global_seed_ = seed;
    cpu_rand_->SeedWithDeviceID(global_seed_);
    cpu_parallel_rand_->SeedWithDeviceID(global_seed_);
#if MXNET_USE_CUDA
    gpu_rand_.ForEach([seed](size_t i, ResourceRandom<gpu>* p) { p->SeedWithDeviceID(seed); });
    gpu_parallel_rand_.ForEach(
        [seed](size_t i, ResourceParallelRandom<gpu>* p) { p->SeedWithDeviceID(seed); });
#if MXNET_USE_CUDNN == 1
    gpu_cudnn_dropout_state_.ForEach([seed](size_t i, ResourceCUDNNDropout* p) {
      ResourceManagerImpl::SeedCUDNNDropout(p, seed);
    });
#endif  // MXNET_USE_CUDNN
#endif  // MXNET_USE_CUDA
  }

  void SeedRandom(Context ctx, uint32_t seed) override {
    cpu_rand_->Seed(seed);
    cpu_parallel_rand_->Seed(seed);
#if MXNET_USE_CUDA
    if (ctx.dev_type == Context::kGPU) {
      gpu_rand_.Get(ctx.dev_id, [ctx, seed, this]() { return new ResourceRandom<gpu>(ctx, seed); })
          ->Seed(seed);
      gpu_parallel_rand_
          .Get(ctx.dev_id,
               [ctx, seed, this]() {
                 return new ResourceParallelRandom<gpu>(ctx, gpu_native_rand_copy_, seed);
               })
          ->Seed(seed);
#if MXNET_USE_CUDNN == 1
      auto dropout_state = gpu_cudnn_dropout_state_.Get(ctx.dev_id, [ctx, seed, this]() {
        return new ResourceCUDNNDropout(ctx, gpu_cudnn_dropout_state_copy_, seed);
      });
      ResourceManagerImpl::SeedCUDNNDropout(dropout_state.get(), seed);
#endif  // MXNET_USE_CUDNN
    }

#endif  // MXNET_USE_CUDA
  }

 private:
  /*! \brief Maximum number of GPUs */
  static constexpr std::size_t kMaxNumGPUs = 16;
  /*! \brief Random number magic number to seed different random numbers */
  static constexpr uint32_t kRandMagic = 127UL;
  // the random number resources
  template <typename xpu>
  struct ResourceRandom {
    /*! \brief the context of the PRNG */
    Context ctx;
    /*! \brief pointer to PRNG */
    mshadow::Random<xpu>* prnd;
    /*! \brief resource representation */
    Resource resource;
    /*! \brief constructor */
    explicit ResourceRandom(Context ctx, uint32_t global_seed) : ctx(ctx) {
      mshadow::SetDevice<xpu>(ctx.dev_id);
      resource.var  = Engine::Get()->NewVariable();
      prnd          = new mshadow::Random<xpu>(ctx.dev_id + global_seed * kRandMagic);
      resource.ptr_ = prnd;
      resource.req  = ResourceRequest(ResourceRequest::kRandom);
    }
    ~ResourceRandom() {
      mshadow::Random<xpu>* r = prnd;
      Engine::Get()->DeleteVariable(
          [r](RunContext rctx) { MSHADOW_CATCH_ERROR(delete r); }, ctx, resource.var);
    }
    // set seed to a PRNG using global_seed and device id
    inline void SeedWithDeviceID(uint32_t global_seed) {
      Seed(ctx.dev_id + global_seed * kRandMagic);
    }
    // set seed to a PRNG
    inline void Seed(uint32_t seed) {
      mshadow::Random<xpu>* r = prnd;
      Engine::Get()->PushAsync(
          [r, seed](RunContext rctx,
                    Engine::CallbackOnStart on_start,
                    Engine::CallbackOnComplete on_complete) {
            on_start();
            r->set_stream(rctx.get_stream<xpu>());
            r->Seed(seed);
            on_complete();
          },
          ctx,
          {},
          {resource.var},
          FnProperty::kNormal,
          0,
          "ResourceRandomSetSeed");
    }
  };

  // temporary space resource.
  template <ResourceRequest::Type req>
  struct ResourceTempSpace {
    /*! \brief the context of the device */
    Context ctx;
    /*! \brief the underlying space */
    std::vector<SpaceAllocator> space;
    /*! \brief resource representation */
    std::vector<Resource> resource;
    /*! \brief current pointer to the round roubin allocator */
    std::atomic<size_t> curr_ptr;
    /*! \brief constructor */
    explicit ResourceTempSpace(Context ctx, size_t ncopy)
        : ctx(ctx), space(ncopy), resource(ncopy), curr_ptr(0) {
      for (size_t i = 0; i < space.size(); ++i) {
        resource[i].var  = Engine::Get()->NewVariable();
        resource[i].id   = static_cast<int32_t>(i);
        resource[i].ptr_ = &space[i];
        resource[i].req  = ResourceRequest(req);
        space[i].ctx     = ctx;
        CHECK_EQ(space[i].handle.size, 0U);
      }
    }
    virtual ~ResourceTempSpace() {
      for (size_t i = 0; i < space.size(); ++i) {
        SpaceAllocator r = space[i];
        Engine::Get()->DeleteVariable(
            [r](RunContext rctx) {
              SpaceAllocator rcpy = r;
              MSHADOW_CATCH_ERROR(rcpy.ReleaseAll());
            },
            ctx,
            resource[i].var);
      }
    }
    // get next resource in round roubin matter
    inline Resource GetNext() {
      const size_t kMaxDigit = std::numeric_limits<size_t>::max() / 2;
      size_t ptr             = ++curr_ptr;
      // reset ptr to avoid undefined behavior during overflow
      // usually this won't happen
      if (ptr > kMaxDigit) {
        curr_ptr.store((ptr + 1) % space.size());
      }
      return resource[ptr % space.size()];
    }
  };

#if MXNET_USE_CUDNN == 1
  struct ResourceCUDNNDropout : public ResourceTempSpace<ResourceRequest::kCuDNNDropoutDesc> {
    explicit ResourceCUDNNDropout(Context ctx, size_t ncopy, uint32_t global_seed)
        : ResourceTempSpace<ResourceRequest::kCuDNNDropoutDesc>(ctx, ncopy) {
      ResourceManagerImpl::SeedCUDNNDropout(this, global_seed);
    }
  };

  static void SeedCUDNNDropout(ResourceCUDNNDropout* p, uint32_t seed) {
    for (size_t i = 0; i < p->space.size(); ++i) {
      uint32_t current_seed = p->ctx.dev_id + i * kMaxNumGPUs + seed * kRandMagic;
      Resource* r           = &(p->resource[i]);
      Engine::Get()->PushAsync(
          [r, current_seed](RunContext rctx,
                            Engine::CallbackOnStart on_start,
                            Engine::CallbackOnComplete on_complete) {
            on_start();
            auto state_space             = static_cast<resource::SpaceAllocator*>(r->ptr_);
            mshadow::Stream<gpu>* stream = rctx.get_stream<gpu>();
            CHECK_EQ(state_space->ctx.dev_id, stream->dev_id)
                << "The device id of cudnn dropout state space doesn't match that from stream.";
            if (!state_space->handle.size) {
              // not allocated yet
              size_t dropout_state_size;
              CUDNN_CALL(cudnnDropoutGetStatesSize(stream->dnn_handle_, &dropout_state_size));
              // reserve GPU space
              Storage::Get()->DirectFree(
                  Storage::Get()->Alloc(dropout_state_size, state_space->ctx));
              state_space->GetSpace(dropout_state_size, "cudnn_dropout_state");
            }
            cudnnDropoutDescriptor_t temp_descriptor;
            CUDNN_CALL(cudnnCreateDropoutDescriptor(&temp_descriptor));
            CUDNN_CALL(cudnnSetDropoutDescriptor(temp_descriptor,
                                                 stream->dnn_handle_,
                                                 0.5,
                                                 state_space->handle.dptr,
                                                 state_space->handle.size,
                                                 current_seed));
            CUDNN_CALL(cudnnDestroyDropoutDescriptor(temp_descriptor));
            cudaStream_t cuda_stream = mshadow::Stream<gpu>::GetStream(stream);
            cudaStreamSynchronize(cuda_stream);
            on_complete();
          },
          p->ctx,
          {},
          {r->var},
          FnProperty::kNormal,
          0,
          "CUDNNDropoutDescriptorSeed");
    }

    p->curr_ptr.store(0);
  }

#endif  // MXNET_USE_CUDNN

  // the parallel random sampler resources
  // it use device API for GPU
  template <typename xpu>
  struct ResourceParallelRandom {
    /*! \brief the context of the PRNG */
    Context ctx;
    /*! \brief pointers to sampler */
    std::vector<common::random::RandGenerator<xpu>*> sampler;
    /*! \brief resource representation */
    std::vector<Resource> resource;
    /*! \brief current pointer to the round roubin allocator */
    std::atomic<size_t> curr_ptr;
    /*! \brief constructor */
    explicit ResourceParallelRandom(Context ctx, size_t ncopy, uint32_t global_seed)
        : ctx(ctx), sampler(ncopy), resource(ncopy), curr_ptr(0) {
      for (size_t i = 0; i < sampler.size(); ++i) {
        const uint32_t seed = ctx.dev_id + i * kMaxNumGPUs + global_seed * kRandMagic;
        resource[i].var     = Engine::Get()->NewVariable();
        common::random::RandGenerator<xpu>* r = new common::random::RandGenerator<xpu>();
        Engine::Get()->PushSync(
            [r, seed](RunContext rctx) {
              common::random::RandGenerator<xpu>::AllocState(r);
              r->Seed(rctx.get_stream<xpu>(), seed);
            },
            ctx,
            {},
            {resource[i].var},
            FnProperty::kNormal,
            0,
            "ResourceParallelRandomSetSeed");
        sampler[i]       = r;
        resource[i].ptr_ = sampler[i];
        resource[i].req  = ResourceRequest(ResourceRequest::kParallelRandom);
      }
    }
    ~ResourceParallelRandom() {
      for (size_t i = 0; i < sampler.size(); ++i) {
        common::random::RandGenerator<xpu>* r = sampler[i];
        Engine::Get()->DeleteVariable(
            [r](RunContext rctx) {
              MSHADOW_CATCH_ERROR(common::random::RandGenerator<xpu>::FreeState(r));
              MSHADOW_CATCH_ERROR(delete r);
            },
            ctx,
            resource[i].var);
      }
    }
    // set seed to a sampler using global_seed and device id
    inline void SeedWithDeviceID(uint32_t global_seed) {
      for (size_t i = 0; i < sampler.size(); ++i) {
        SeedOne(i, ctx.dev_id + i * kMaxNumGPUs + global_seed * kRandMagic);
      }
      // reset pointer to ensure the same result with the same seed.
      curr_ptr.store(0);
    }
    // set seed to a sampler
    inline void Seed(uint32_t seed) {
      for (size_t i = 0; i < sampler.size(); ++i) {
        SeedOne(i, i * kMaxNumGPUs + seed * kRandMagic);
      }
      // reset pointer to ensure the same result with the same seed.
      curr_ptr.store(0);
    }
    // set seed to a sampler
    inline void SeedOne(size_t i, uint32_t seed) {
      common::random::RandGenerator<xpu>* r = sampler[i];
      Engine::Get()->PushAsync(
          [r, seed](RunContext rctx,
                    Engine::CallbackOnStart on_start,
                    Engine::CallbackOnComplete on_complete) {
            on_start();
            r->Seed(rctx.get_stream<xpu>(), seed);
            on_complete();
          },
          ctx,
          {},
          {resource[i].var},
          FnProperty::kNormal,
          0,
          "ResourceNativeRandomSetSeed");
    }
    // get next resource in round roubin matter
    inline Resource GetNext() {
      const size_t kMaxDigit = std::numeric_limits<size_t>::max() / 2;
      size_t ptr             = ++curr_ptr;
      // reset ptr to avoid undefined behavior during overflow
      // usually this won't happen
      if (ptr > kMaxDigit) {
        curr_ptr.store((ptr + 1) % sampler.size());
      }
      return resource[ptr % sampler.size()];
    }
  };

  /*! \brief number of copies in CPU temp space */
  int cpu_temp_space_copy_;
  /*! \brief number of copies in GPU temp space */
  int gpu_temp_space_copy_;
  /*! \brief number of copies in CPU native random sampler */
  int cpu_native_rand_copy_;
  /*! \brief number of copies in GPU native random sampler */
  int gpu_native_rand_copy_;
  /*! \brief Reference to the engine */
  std::shared_ptr<Engine> engine_ref_;
  /*! \brief Reference to the storage */
  std::shared_ptr<Storage> storage_ref_;
  /*! \brief internal seed to the random number generator */
  uint32_t global_seed_{static_cast<uint32_t>(time(nullptr))};
  /*! \brief CPU random number resources */
  std::unique_ptr<ResourceRandom<cpu>> cpu_rand_;
  /*! \brief CPU temp space resources */
  std::unique_ptr<ResourceTempSpace<ResourceRequest::kTempSpace>> cpu_space_;
  /*! \brief CPU parallel random number resources */
  std::unique_ptr<ResourceParallelRandom<cpu>> cpu_parallel_rand_;
#if MXNET_USE_CUDA
  /*! \brief random number generator for GPU */
  common::LazyAllocArray<ResourceRandom<gpu>> gpu_rand_;
  /*! \brief temp space for GPU */
  common::LazyAllocArray<ResourceTempSpace<ResourceRequest::kTempSpace>> gpu_space_;
  /*! \brief GPU parallel (on device) random number resources */
  common::LazyAllocArray<ResourceParallelRandom<gpu>> gpu_parallel_rand_;
#if MXNET_USE_CUDNN == 1
  /*! \brief number of copies in GPU cudnn dropout descriptor resources */
  int gpu_cudnn_dropout_state_copy_;
  /*! \brief GPU parallel (on device) random number resources */
  common::LazyAllocArray<ResourceCUDNNDropout> gpu_cudnn_dropout_state_;
#endif  // MXNET_USE_CUDNN == 1
#endif
};
}  // namespace resource

void* Resource::get_space_internal(size_t size, const std::string& name) const {
  return static_cast<resource::SpaceAllocator*>(ptr_)->GetSpace(size, name);
}

void* Resource::get_host_space_internal(size_t size) const {
  return static_cast<resource::SpaceAllocator*>(ptr_)->GetHostSpace(size);
}

#if MXNET_USE_CUDNN == 1
void Resource::get_cudnn_dropout_desc(cudnnDropoutDescriptor_t* dropout_desc,
                                      mshadow::Stream<gpu>* stream,
                                      const float dropout,
                                      const std::string& name) const {
  CHECK_EQ(req.type, ResourceRequest::kCuDNNDropoutDesc);
  auto state_space = static_cast<resource::SpaceAllocator*>(ptr_);
  CHECK_EQ(state_space->ctx.dev_id, stream->dev_id)
      << "The device id of cudnn dropout state space doesn't match that from stream.";
  if (dropout <= 0) {
    CUDNN_CALL(
        cudnnSetDropoutDescriptor(*dropout_desc, stream->dnn_handle_, dropout, nullptr, 0, 0));
  } else {
    CHECK(state_space->handle.size > 0) << "CUDNN dropout descriptor was not initialized yet!";
    // cudnnRestoreDropoutDescriptor() introduced with cuDNN v7
    STATIC_ASSERT_CUDNN_VERSION_GE(7000);
    CUDNN_CALL(cudnnRestoreDropoutDescriptor(*dropout_desc,
                                             stream->dnn_handle_,
                                             dropout,
                                             state_space->handle.dptr,
                                             state_space->handle.size,
                                             0));
  }
}
#endif  // MXNET_USE_CUDNN == 1

ResourceManager* ResourceManager::Get() {
  typedef dmlc::ThreadLocalStore<resource::ResourceManagerImpl> inst;
  return inst::Get();
}
}  // namespace mxnet
