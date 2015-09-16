/*!
 *  Copyright (c) 2015 by Contributors
 * \file resource.cc
 * \brief Implementation of resource manager.
 */
#include <dmlc/logging.h>
#include <mxnet/base.h>
#include <mxnet/engine.h>
#include <mxnet/resource.h>
#include "./common/lazy_alloc_array.h"

namespace mxnet {
namespace resource {

// implements resource manager
class ResourceManagerImpl : public ResourceManager {
 public:
  ResourceManagerImpl() : global_seed_(0) {
    engine_ref_ = Engine::_GetSharedRef();
    cpu_rand_ = new ResourceRandom<cpu>(
        Context(cpu::kDevMask, 0), global_seed_);
  }
  ~ResourceManagerImpl() {
    // need explicit delete, before engine get killed
    delete cpu_rand_;
#if MXNET_USE_CUDA
    gpu_rand_.Clear();
#endif
    // release the reference to engine.
    engine_ref_ = nullptr;
  }

  // request resources
  Resource Request(Context ctx, const ResourceRequest &req) override {
    if (req.type == ResourceRequest::kRandom) {
      if (ctx.dev_mask == cpu::kDevMask) {
        return cpu_rand_->resource;
      } else {
        CHECK_EQ(ctx.dev_mask, gpu::kDevMask);
#if MSHADOW_USE_CUDA
        return gpu_rand_.Get(ctx.dev_id, [ctx, this]() {
            return new ResourceRandom<gpu>(ctx, global_seed_);
          })->resource;
#else
        LOG(FATAL) << MXNET_GPU_NOT_ENABLED_ERROR;
#endif
      }
    } else {
      LOG(FATAL) << "Unknown supported type " << req.type;
    }
    Resource ret;
    return ret;
  }

  void SeedRandom(uint32_t seed) override {
    global_seed_ = seed;
    cpu_rand_->Seed(global_seed_);
#if MXNET_USE_CUDA
    gpu_rand_.ForEach([seed](size_t i, ResourceRandom<gpu> *p) {
        p->Seed(seed);
      });
#endif
  }

 private:
  /*! \brief Maximum number of GPUs */
  static constexpr std::size_t kMaxNumGPUs = 16;
  /*! \brief Random number magic number to seed different random numbers */
  static constexpr uint32_t kRandMagic = 127UL;
  /*! \brief Reference to the engine */
  std::shared_ptr<Engine> engine_ref_;

  // the random number resources
  template<typename xpu>
  struct ResourceRandom {
    /*! \brief pointer to PRNG */
    mshadow::Random<xpu> *prnd;
    /*! \brief the context of the PRNG */
    Context ctx;
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
          [r](RunContext rctx){ delete r; }, ctx, resource.var);
    }
    // set seed to a PRNG
    inline void Seed(uint32_t global_seed) {
      uint32_t seed = ctx.dev_id + global_seed * kRandMagic;
      mshadow::Random<xpu> *r = prnd;
      Engine::Get()->PushSync([r, seed](RunContext rctx) {
          r->set_stream(rctx.get_stream<xpu>());
          r->Seed(seed);
        }, ctx, {}, {resource.var});
    }
  };
  /*! \brief internal seed to the random number generator */
  uint32_t global_seed_;
  /*! \brief CPU random number resources */
  ResourceRandom<cpu> *cpu_rand_;
#if MXNET_USE_CUDA
  /*! \brief random number generator for GPU */
  common::LazyAllocArray<ResourceRandom<gpu> > gpu_rand_;
#endif
};
}  // namespace resource

ResourceManager* ResourceManager::Get() {
  static resource::ResourceManagerImpl inst;
  return &inst;
}
}  // namespace mxnet
