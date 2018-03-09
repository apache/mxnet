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
 * \file resource.cc
 * \brief Implementation of resource manager.
 */
#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <dmlc/thread_local.h>
#include <mxnet/base.h>
#include <mxnet/engine.h>
#include <mxnet/resource.h>
#include <mxnet/storage.h>
#include <limits>
#include <atomic>
#include "./common/lazy_alloc_array.h"
#include "./common/random_generator.h"
#include "./common/utils.h"

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
    handle.dptr = nullptr;
    handle.size = 0;
    host_handle.dptr = nullptr;
    host_handle.size = 0;
  }
  inline void ReleaseAll() {
    if (handle.size != 0) {
      Storage::Get()->DirectFree(handle);
      handle.size = 0;
    }
    if (host_handle.size != 0) {
      Storage::Get()->DirectFree(host_handle);
      host_handle.size = 0;
    }
  }
  inline void* GetSpace(size_t size) {
    if (handle.size >= size) return handle.dptr;
    if (handle.size != 0) {
      Storage::Get()->DirectFree(handle);
    }
    handle = Storage::Get()->Alloc(size, ctx);
    return handle.dptr;
  }

  inline void* GetHostSpace(size_t size) {
    if (host_handle.size >= size) return host_handle.dptr;
    if (host_handle.size != 0) {
      Storage::Get()->DirectFree(host_handle);
    }
    host_handle = Storage::Get()->Alloc(size, Context());
    return host_handle.dptr;
  }
};


// Implements resource manager
class ResourceManagerImpl : public ResourceManager {
 public:
  ResourceManagerImpl() noexcept(false)
      : global_seed_(0) {
    cpu_temp_space_copy_ = dmlc::GetEnv("MXNET_CPU_TEMP_COPY", 4);
    gpu_temp_space_copy_ = dmlc::GetEnv("MXNET_GPU_TEMP_COPY", 1);
    cpu_native_rand_copy_ = dmlc::GetEnv("MXNET_CPU_PARALLEL_RAND_COPY", 1);
    gpu_native_rand_copy_ = dmlc::GetEnv("MXNET_GPU_PARALLEL_RAND_COPY", 4);
    engine_ref_ = Engine::_GetSharedRef();
    storage_ref_ = Storage::_GetSharedRef();
    cpu_rand_.reset(new ResourceRandom<cpu>(
        Context::CPU(), global_seed_));
    cpu_space_.reset(new ResourceTempSpace(
        Context::CPU(), cpu_temp_space_copy_));
    cpu_parallel_rand_.reset(new ResourceParallelRandom<cpu>(
        Context::CPU(), cpu_native_rand_copy_, global_seed_));
  }
  ~ResourceManagerImpl() {
    // need explicit delete, before engine get killed
    cpu_rand_.reset(nullptr);
    cpu_space_.reset(nullptr);
    cpu_parallel_rand_.reset(nullptr);
#if MXNET_USE_CUDA
    gpu_rand_.Clear();
    gpu_space_.Clear();
    gpu_parallel_rand_.Clear();
#endif
    if (engine_ref_ != nullptr) {
      engine_ref_ = nullptr;
    }
    if (storage_ref_ != nullptr) {
      storage_ref_ = nullptr;
    }
  }

  // request resources
  Resource Request(Context ctx, const ResourceRequest &req) override {
    if (ctx.dev_mask() == Context::kCPU) {
      switch (req.type) {
        case ResourceRequest::kRandom: return cpu_rand_->resource;
        case ResourceRequest::kTempSpace: return cpu_space_->GetNext();
        case ResourceRequest::kParallelRandom: return cpu_parallel_rand_->GetNext();
        default: LOG(FATAL) << "Unknown supported type " << req.type;
      }
    } else {
      CHECK_EQ(ctx.dev_mask(), Context::kGPU);
#if MSHADOW_USE_CUDA
      switch (req.type) {
        case ResourceRequest::kRandom: {
          return gpu_rand_.Get(ctx.dev_id, [ctx, this]() {
              return new ResourceRandom<gpu>(ctx, global_seed_);
            })->resource;
        }
        case ResourceRequest::kTempSpace: {
          return gpu_space_.Get(ctx.dev_id, [ctx, this]() {
              return new ResourceTempSpace(ctx, gpu_temp_space_copy_);
            })->GetNext();
        }
        case ResourceRequest::kParallelRandom: {
          return gpu_parallel_rand_.Get(ctx.dev_id, [ctx, this]() {
            return new ResourceParallelRandom<gpu>(ctx, gpu_native_rand_copy_, global_seed_);
          })->GetNext();
        }
        default: LOG(FATAL) << "Unknown supported type " << req.type;
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
    cpu_rand_->Seed(global_seed_);
    cpu_parallel_rand_->Seed(global_seed_);
#if MXNET_USE_CUDA
    gpu_rand_.ForEach([seed](size_t i, ResourceRandom<gpu> *p) {
        p->Seed(seed);
      });
    gpu_parallel_rand_.ForEach([seed](size_t i, ResourceParallelRandom<gpu> *p) {
      p->Seed(seed);
    });
#endif
  }

 private:
  /*! \brief Maximum number of GPUs */
  static constexpr std::size_t kMaxNumGPUs = 16;
  /*! \brief Random number magic number to seed different random numbers */
  static constexpr uint32_t kRandMagic = 127UL;
  // the random number resources
  template<typename xpu>
  struct ResourceRandom {
    /*! \brief the context of the PRNG */
    Context ctx;
    /*! \brief pointer to PRNG */
    mshadow::Random<xpu> *prnd;
    /*! \brief resource representation */
    Resource resource;
    /*! \brief constructor */
    explicit ResourceRandom(Context ctx, uint32_t global_seed)
        : ctx(ctx) {
      mshadow::SetDevice<xpu>(ctx.dev_id);
      resource.var = Engine::Get()->NewVariable();
      prnd = new mshadow::Random<xpu>(ctx.dev_id + global_seed * kRandMagic);
      resource.ptr_ = prnd;
      resource.req = ResourceRequest(ResourceRequest::kRandom);
    }
    ~ResourceRandom() {
      mshadow::Random<xpu> *r = prnd;
      Engine::Get()->DeleteVariable(
          [r](RunContext rctx) {
            MSHADOW_CATCH_ERROR(delete r);
          }, ctx, resource.var);
    }
    // set seed to a PRNG
    inline void Seed(uint32_t global_seed) {
      uint32_t seed = ctx.dev_id + global_seed * kRandMagic;
      mshadow::Random<xpu> *r = prnd;
      Engine::Get()->PushAsync(
        [r, seed](RunContext rctx, Engine::CallbackOnComplete on_complete) {
          r->set_stream(rctx.get_stream<xpu>());
          r->Seed(seed);
          on_complete();
        }, ctx, {}, {resource.var},
        FnProperty::kNormal, 0, PROFILER_MESSAGE("ResourceRandomSetSeed"));
    }
  };

  // temporal space resource.
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
        resource[i].var = Engine::Get()->NewVariable();
        resource[i].id = static_cast<int32_t>(i);
        resource[i].ptr_ = &space[i];
        resource[i].req = ResourceRequest(ResourceRequest::kTempSpace);
        space[i].ctx = ctx;
        CHECK_EQ(space[i].handle.size, 0U);
      }
    }
    ~ResourceTempSpace() {
      for (size_t i = 0; i < space.size(); ++i) {
        SpaceAllocator r = space[i];
        Engine::Get()->DeleteVariable(
            [r](RunContext rctx){
              SpaceAllocator rcpy = r;
              MSHADOW_CATCH_ERROR(rcpy.ReleaseAll());
            }, ctx, resource[i].var);
      }
    }
    // get next resource in round roubin matter
    inline Resource GetNext() {
      const size_t kMaxDigit = std::numeric_limits<size_t>::max() / 2;
      size_t ptr = ++curr_ptr;
      // reset ptr to avoid undefined behavior during overflow
      // usually this won't happen
      if (ptr > kMaxDigit) {
        curr_ptr.store((ptr + 1) % space.size());
      }
      return resource[ptr % space.size()];
    }
  };

  // the parallel random sampler resources
  // it use device API for GPU
  template<typename xpu>
  struct ResourceParallelRandom {
    /*! \brief the context of the PRNG */
    Context ctx;
    /*! \brief pointers to sampler */
    std::vector<common::random::RandGenerator<xpu> *> sampler;
    /*! \brief resource representation */
    std::vector<Resource> resource;
    /*! \brief current pointer to the round roubin allocator */
    std::atomic<size_t> curr_ptr;
    /*! \brief constructor */
    explicit ResourceParallelRandom(Context ctx, size_t ncopy, uint32_t global_seed)
        : ctx(ctx), sampler(ncopy), resource(ncopy), curr_ptr(0) {
      for (size_t i = 0; i < sampler.size(); ++i) {
        const uint32_t seed = ctx.dev_id + i * kMaxNumGPUs + global_seed * kRandMagic;
        resource[i].var = Engine::Get()->NewVariable();
        common::random::RandGenerator<xpu> *r = new common::random::RandGenerator<xpu>();
        Engine::Get()->PushSync(
        [r, seed](RunContext rctx) {
          common::random::RandGenerator<xpu>::AllocState(r);
          r->Seed(rctx.get_stream<xpu>(), seed);
        }, ctx, {}, {resource[i].var},
        FnProperty::kNormal, 0, PROFILER_MESSAGE("ResourceParallelRandomSetSeed"));
        sampler[i] = r;
        resource[i].ptr_ = sampler[i];
        resource[i].req = ResourceRequest(ResourceRequest::kParallelRandom);
      }
    }
    ~ResourceParallelRandom() {
      for (size_t i = 0; i < sampler.size(); ++i) {
        common::random::RandGenerator<xpu> *r = sampler[i];
        Engine::Get()->DeleteVariable(
        [r](RunContext rctx) {
          MSHADOW_CATCH_ERROR(common::random::RandGenerator<xpu>::FreeState(r));
          MSHADOW_CATCH_ERROR(delete r);
        }, ctx, resource[i].var);
      }
    }
    // set seed to a sampler
    inline void Seed(uint32_t global_seed) {
      for (size_t i = 0; i < sampler.size(); ++i) {
        const uint32_t seed = ctx.dev_id + i * kMaxNumGPUs + global_seed * kRandMagic;
        common::random::RandGenerator<xpu> *r = sampler[i];
        Engine::Get()->PushAsync(
        [r, seed](RunContext rctx, Engine::CallbackOnComplete on_complete) {
          r->Seed(rctx.get_stream<xpu>(), seed);
          on_complete();
        }, ctx, {}, {resource[i].var},
        FnProperty::kNormal, 0, PROFILER_MESSAGE("ResourceNativeRandomSetSeed"));
      }
      // reset pointer to ensure the same result with the same seed.
      curr_ptr.store(0);
    }
    // get next resource in round roubin matter
    inline Resource GetNext() {
      const size_t kMaxDigit = std::numeric_limits<size_t>::max() / 2;
      size_t ptr = ++curr_ptr;
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
  uint32_t global_seed_;
  /*! \brief CPU random number resources */
  std::unique_ptr<ResourceRandom<cpu> > cpu_rand_;
  /*! \brief CPU temp space resources */
  std::unique_ptr<ResourceTempSpace> cpu_space_;
  /*! \brief CPU parallel random number resources */
  std::unique_ptr<ResourceParallelRandom<cpu> > cpu_parallel_rand_;
#if MXNET_USE_CUDA
  /*! \brief random number generator for GPU */
  common::LazyAllocArray<ResourceRandom<gpu> > gpu_rand_;
  /*! \brief temp space for GPU */
  common::LazyAllocArray<ResourceTempSpace> gpu_space_;
  /*! \brief GPU parallel (on device) random number resources */
  common::LazyAllocArray<ResourceParallelRandom<gpu> > gpu_parallel_rand_;
#endif
};
}  // namespace resource

void* Resource::get_space_internal(size_t size) const {
  return static_cast<resource::SpaceAllocator*>(ptr_)->GetSpace(size);
}

void* Resource::get_host_space_internal(size_t size) const {
  return static_cast<resource::SpaceAllocator*>(ptr_)->GetHostSpace(size);
}

ResourceManager* ResourceManager::Get() {
  typedef dmlc::ThreadLocalStore<resource::ResourceManagerImpl> inst;
  return inst::Get();
}
}  // namespace mxnet
