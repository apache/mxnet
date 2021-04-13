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
 * Copyright (c) 2015 by Contributors
 * \file pooled_storage_manager.h
 * \brief Storage manager with a memory pool.
 */
#ifndef MXNET_STORAGE_POOLED_STORAGE_MANAGER_H_
#define MXNET_STORAGE_POOLED_STORAGE_MANAGER_H_

#include <string>
#include <vector>
#include <algorithm>
#include <mutex>
#include <tuple>
#include "./storage_manager.h"
#include "../profiler/storage_profiler.h"


namespace mxnet {
namespace storage {

typedef enum {
  pool_type,
  pool_page_size,
  large_alloc_size,
  round_linear_cutoff,
  pool_reserve,
} env_var_type;

const std::string env_var_name(const char* dev_type, env_var_type type);

#if MXNET_USE_CUDA
#define SET_DEVICE(device_store, contextHelper, ctx, flag) \
      const auto *device_store = flag? contextHelper.get()->SetCurrentDevice(ctx) : nullptr;
#define UNSET_DEVICE(device_store)    delete device_store

#define SET_GPU_PROFILER(prof, contextHelper)                          \
      auto prof = contextHelper->contextGPU()?                         \
                  profiler::GpuDeviceStorageProfiler::Get() : nullptr; \
      if (!prof->IsProfiling()) prof = nullptr

#define GPU_PROFILER_ON_FREE(prof, pntr)    if (prof) prof->OnFree(pntr)
#else
// empty macros when MxNet is compiled without CUDA support
#define SET_DEVICE(...)
#define UNSET_DEVICE(...)
#define SET_GPU_PROFILER(prof, ...)
#define GPU_PROFILER_ON_FREE(prof, ...)
#endif

/*!
 * \brief Storage manager with a memory pool for GPU/CPU/CPUPunned memory chunks
 * memory chunks which reused based on rounded size match.
 * Rounding method is defined by the template parameter BucketingStrategy.
 * Memory pool type is defined by the template parameter StoringMethod
 * Allocation/freeing of memory is done by contextHelper_, which is the pointer
 * to one of memory specific instance of the class, derived from ContextHelper
 */
template<typename BucketingStrategy, typename StoringMethod>
class PooledStorageManager : public StorageManager,
             public BucketingStrategy, public StoringMethod {
 public:
  explicit PooledStorageManager(const Context &ctx, int num_gpu_device) {
    const char *dev_type = nullptr;
    switch (dev_type_ = ctx.dev_type) {
#if MXNET_USE_CUDA
      case Context::kGPU:       contextHelper_ = std::make_unique<ContextHelperGPU>();
                                dev_type = "GPU";
                                break;
      case Context::kCPUPinned: dev_type = "CPU_PINNED";
                                if (num_gpu_device > 1) {
                                  contextHelper_ = std::make_unique<ContextHelperPinned>();
                                  dev_type_ = Context::kGPU;
                                  break;
                                }
#else
      case Context::kCPUPinned: dev_type = "CPU_PINNED";
#endif
                                dev_type_ = Context::kCPU;
      case Context::kCPU:       contextHelper_ = std::make_unique<ContextHelperCPU>();
                                dev_type = "CPU";
      default:                  break;
    }

    BucketingStrategy::InitRoundHelper(dev_type);
    StoringMethod::InitContainer(this);
    contextHelper_->set_initilal_context(ctx);

    // percentage of reserved memory
    if (dev_type) {
      const auto env_var = env_var_name(dev_type, pool_reserve);
      const size_t reserve = dmlc::GetEnv(env_var.c_str(), 5);
      const size_t total = std::get<1>(contextHelper_->getMemoryInfo());
      memory_allocation_limit_ = total * reserve / 100;
    }
  }
  /*!
   * \brief Default destructor.
   */
  ~PooledStorageManager() override {
    ReleaseAll();
  }

  void Alloc(Storage::Handle* handle) override;
  void Free(Storage::Handle handle) override {
    // Insert returned memory in cache
    std::lock_guard<std::mutex> lock(Storage::Get()->GetMutex(dev_type_));
    StoringMethod::InsertInCache(BucketingStrategy::get_bucket(handle.size), handle.dptr);
  }

  void DirectFree(Storage::Handle handle) override {
    std::lock_guard<std::mutex> lock(Storage::Get()->GetMutex(dev_type_));
    SET_DEVICE(device_store, contextHelper_, handle.ctx, true);
    contextHelper_->Free(handle.dptr);
    SET_GPU_PROFILER(profilerGPU, contextHelper_);
    GPU_PROFILER_ON_FREE(profilerGPU, handle.dptr);
    UNSET_DEVICE(device_store);
    used_memory_ -= BucketingStrategy::RoundAllocSize(handle.size);
  }

  void ReleaseAll() override {
    std::lock_guard<std::mutex> lock(Storage::Get()->GetMutex(dev_type_));
    ReleaseAllNoLock();
  }

 private:
  void ReleaseAllNoLock(bool set_device = true) {
    SET_DEVICE(device_store, contextHelper_, contextHelper_->initilal_context(), set_device);
    used_memory_ -= StoringMethod::ReleaseAllNoLock(contextHelper_.get(), this);
    UNSET_DEVICE(device_store);
  }

  bool MemoryIsAvalable(size_t roundSize) const {
    const auto free = contextHelper_->freeMemorySize();
    return free > roundSize && memory_allocation_limit_ <= free - roundSize;
  }

  // device type of used context
  Context::DeviceType dev_type_;
  // used memory
  size_t used_memory_ = 0;
  // minimum amount of memory, which will never be allocated
  size_t memory_allocation_limit_ = 0;
  // Pointer to the Helper, supporting some context-specific operations in GPU/CPU/CPUPinned context
  std::unique_ptr<ContextHelper> contextHelper_;
};

template<typename BucketingStrategy, typename StoringMethod>
void PooledStorageManager<BucketingStrategy, StoringMethod>::Alloc(Storage::Handle* handle) {
  std::lock_guard<std::mutex> lock(Storage::Get()->GetMutex(dev_type_));
  const auto bucket_id = BucketingStrategy::get_bucket(handle->size);
  size_t roundSize = 0;
  auto reuse_pool = StoringMethod::GetMemStorage(bucket_id);
  if (!reuse_pool) {
    SET_DEVICE(device_store, contextHelper_, handle->ctx, true);
    roundSize = BucketingStrategy::RoundAllocSizeForBucket(bucket_id);
    if (!MemoryIsAvalable(roundSize))
      ReleaseAllNoLock(false);

    void *ret = nullptr;
    auto e = contextHelper_->Malloc(&ret, roundSize);
    if (e) {
      // retry in case of fragmentation
      ReleaseAllNoLock(false);
      e = contextHelper_->Malloc(&ret, roundSize);
      if (e) {
        const std::string err(
#if MXNET_USE_CUDA
        dev_type_ == Context::kGPU?
           cudaGetErrorString(static_cast<cudaError_t>(e)) :
#endif
           std::strerror(errno));

        LOG(FATAL) << "Memory allocation failed " << err;
      }
    }

    UNSET_DEVICE(device_store);

    used_memory_ += roundSize;
    handle->dptr = ret;
  } else {
    // Reusing memory
    handle->dptr = reuse_pool->back();
    reuse_pool->pop_back();
  }
#if MXNET_USE_CUDA
  SET_GPU_PROFILER(profilerGPU, contextHelper_);
  if (profilerGPU) {
    if (reuse_pool)  // roundSize was not calculated
      roundSize = BucketingStrategy::RoundAllocSizeForBucket(bucket_id);

    // record the allocation event in the memory profiler
    profilerGPU->OnAlloc(*handle, roundSize, reuse_pool);
  }
#endif
}


/*!
 * \brief Base class for Rounding Method classes.
 */
class RoundHelper {
 public:
  virtual size_t get_size(size_t  /*bucket*/) const { return 0; }
  virtual std::tuple<size_t, size_t> getContainerParam() const {
     return std::tuple<size_t, size_t>(0, 0);
  }

 protected:
  void InitRoundHelper(const char* dev_type) {
    const auto env_var = env_var_name(dev_type, pool_page_size);
    page_size_ = dmlc::GetEnv(env_var.c_str(), 4096);
    if (page_size_ < NDEV) {
      LOG(FATAL) << env_var << " cannot be set to a value smaller than " << NDEV \
                    << ". Got " << page_size_ << ".";
    }
  }

  // page size
  size_t page_size_ = 0;

 private:
  // number of devices
  const size_t NDEV = 32;
};  // class RoundHelper

/*!
 * \brief Rounding method used by CPU/GPU mem pool.
 * Round up small allocs to multiple of page_size_ or large_alloc_round_size_
 */
class RoundMultiple : protected RoundHelper {
 protected:
  void InitRoundHelper(const char *dev_type) {
    RoundHelper::InitRoundHelper(dev_type);
    const auto env_var = env_var_name(dev_type, large_alloc_size);
    large_alloc_round_size_ = dmlc::GetEnv(env_var.c_str(), 2*1024*1024);
    if (large_alloc_round_size_ <= 0) {
      LOG(FATAL) << env_var << " cannot be set to a value <= 0, found: "
                 << large_alloc_round_size_;
    }
  }

  size_t RoundAllocSize(size_t size) const {
    // Round up small allocs to multiple of page_size_ to consolidate the pool lookups
    size = RoundToMultiple(size, page_size_);
    // To ensure proper freeing under some driver variants, make sure
    // large allocs entirely occupy their slabs, which cannot then be
    // locked by smaller permanent allocations sharing the slab.
    return  size > large_alloc_round_size_? RoundToMultiple(size, large_alloc_round_size_) : size;
  }
  inline size_t get_bucket(size_t size) const                   { return RoundAllocSize(size); }
  inline size_t RoundAllocSizeForBucket(size_t bucket_id) const { return bucket_id; }

 private:
  // Round a value 'x' up to the next multiple of 'multiple'
  inline static size_t RoundToMultiple(size_t x, size_t multiple) {
    return ((x + multiple - 1) / multiple) * multiple;
  }

  // size that large allocations should be rounded to, for proper freeing.
  size_t large_alloc_round_size_;
};  // class RoundMultiple

/*!
 * \brief Rounding method used by CPU/GPU mem pool.
 *
 * This Rounding method uses a mixture of nearest pow2 (exponential) rounding and
 * nearest multiple (linear) rounding to help alleviate the memory allocation stress
 * in which the default naive exact-size-match pool falls short, such as in variable-length
 * input/output cases like RNN workloads.
 *
 * \param cutoff the cutoff at which rounding is switched from exponential to linear. It's set
 * through MXNET_GPU_MEM_POOL_ROUND_LINEAR_CUTOFF / MXNET_CPU_MEM_POOL_ROUND_LINEAR_CUTOFF /
 * MXNET_CPU_PINNED_MEM_POOL_ROUND_LINEAR_CUTOFF environment variable.
 * Must be between 20 (1 MB) and 34 (16 GB).
 * Suppose the cutoff is X, the memory size buckets look like this:
 * exp2(0), exp2(1), ..., exp2(X), 2*exp2(X), 3*exp2(X), ...
 */
class RoundPower2 : public RoundHelper {
 public:
  size_t get_size(size_t bucket) const override {
    return bucket <= cut_off_? 1ul << bucket : (bucket - cut_off_ + 1) << cut_off_;
  }

 protected:
  void InitRoundHelper(const char *dev_type) {
    RoundHelper::InitRoundHelper(dev_type);
    const auto log_pager_size = common::ilog2ul(page_size_ - 1);
    if (page_size_ != 1ul << log_pager_size) {
      LOG(FATAL) << env_var_name(dev_type, pool_page_size) \
                 << " must be a power of 2. Got: " << page_size_ << ".";
    }
    page_size_ = log_pager_size;

    const auto linear_cutoff = env_var_name(dev_type, round_linear_cutoff);
    cut_off_ = dmlc::GetEnv(linear_cutoff.c_str(), 24);
    if (cut_off_ < 20 || cut_off_ > LOG2_MAX_MEM) {
      LOG(FATAL) << linear_cutoff << " cannot be set to a value " \
                 << "smaller than 20 or greater than " << LOG2_MAX_MEM << ". Got: " \
                 << cut_off_ << ".";
    }
    if (cut_off_ < page_size_) {
      LOG(FATAL) << linear_cutoff << " cannot be set to a value smaller than log2 of " \
                 << env_var_name(dev_type, pool_page_size) << ". Got: " \
                 << cut_off_ << " vs " << page_size_ << ".";
    }
  }

  inline size_t get_bucket(size_t s) const {
    const size_t log_size = common::ilog2ul(s - 1);
    if (log_size > cut_off_)
      return div_pow2_round_up(s, cut_off_) - 1 + cut_off_;

    return std::max(log_size, page_size_);
  }

  inline size_t RoundAllocSizeForBucket(size_t bucket_id) const { return get_size(bucket_id); }
  inline size_t RoundAllocSize(size_t size) const { return get_size(get_bucket(size)); }
  std::tuple<size_t, size_t> getContainerParam() const override {
    return std::make_tuple((1ul << (LOG2_MAX_MEM - cut_off_)) + cut_off_,
                       get_bucket(page_size_) - 1);
  }

 private:
  inline static int div_pow2_round_up(size_t s, int divisor_log2) {
    // (1025, 10) -> 2
    // (2048, 10) -> 2
    // (2049, 10) -> 3
    const size_t result = s >> divisor_log2;
    return static_cast<int>(result + (s > (result << divisor_log2) ? 1 : 0));
  }

  // log2 of maximum page size. 16GB
  const size_t LOG2_MAX_MEM = 34;
  // log2 of memory size before switching to exponential mode to linear mode
  size_t cut_off_ = 0;
};  // class RoundPower2


/*!
 * \brief Unordered map based storage container.
 *  The pointers to the portions of same rounded sizes memory
 *  allocated on CPU/GPU, are stored in separate vectors.
 *  These sizes are used as keys for accessing the vectors,
 *  which are the elements stored in an unordered map.
 */
class UnorderedMapContainer {
 protected:
  inline void InitContainer(const RoundHelper *p)   {}
  inline void InsertInCache(size_t key, void *dptr) { memory_pool_[key].push_back(dptr); }

  inline std::vector<void*> *GetMemStorage(size_t key) {
    auto&& reuse_it = memory_pool_.find(key);
    return reuse_it != memory_pool_.end() && reuse_it->second.size()? &reuse_it->second : nullptr;
  }

  size_t ReleaseAllNoLock(const ContextHelper *contextHelper, const RoundHelper * /*rndHelper*/) {
    SET_GPU_PROFILER(profilerGPU, contextHelper);
    size_t released_memory = 0;
    for (auto&& i : memory_pool_) {
      for (auto&& j : i.second) {
        contextHelper->Free(j);
        GPU_PROFILER_ON_FREE(profilerGPU, j);
      }
      released_memory += i.first * i.second.size();
      i.second.clear();
    }
    memory_pool_.clear();
    return released_memory;
  }

 private:
  std::unordered_map<size_t, std::vector<void *>> memory_pool_;
};  // class UnorderedMapContainer

/*!
 * \brief Vector-container based storage container. It should be used ONLY with the RoundPower2.
 *  The pointers to the portions of same rounded size allocated on
 *  GPU/CPU/CPU_Pinned memory, are stored in separate vectors.
 *  The vectors themselves are stored in the vector-container and could
 *  be accessed by the indices calculated as a functions of rounded size
 *  (see description for RoundPower2 for more details)
 */
class VectorContainer {
 protected:
  inline void InitContainer(const RoundHelper *p) {
    size_t vector_size;
    std::tie(vector_size, first_bucket_) = p->getContainerParam();
    memory_pool_ .resize(vector_size);
  }

  inline void InsertInCache(size_t idx, void *dptr) { memory_pool_[idx].push_back(dptr); }

  std::vector<void*> *GetMemStorage(size_t idx) {
    auto &&reuse_pool = memory_pool_[idx];
    return reuse_pool.size() ? &reuse_pool : nullptr;
  }

  size_t ReleaseAllNoLock(const ContextHelper *contextHelper, const RoundHelper *rndHelper) {
    SET_GPU_PROFILER(profilerGPU, contextHelper);
    size_t released_memory = 0;
    for (size_t i = first_bucket_; i < memory_pool_.size(); i++) {
      if (!memory_pool_[i].size())
        continue;

      for (auto &j : memory_pool_[i]) {
        contextHelper->Free(j);
        GPU_PROFILER_ON_FREE(profilerGPU, j);
      }
      released_memory += rndHelper->get_size(i) * memory_pool_[i].size();
      memory_pool_[i].clear();
    }
    return released_memory;
  }

 private:
  std::vector<std::vector<void*>> memory_pool_;
  size_t first_bucket_;
};  // class VectorContainer

// For backward compatibility, define previously used classes via new components.
// Just in case, if someone uses these classes in other places, besides
// the storage.cc, where the corresponding changes have already been made.
typedef PooledStorageManager<RoundMultiple, UnorderedMapContainer> GPUPooledStorageManager;
typedef PooledStorageManager<RoundPower2, VectorContainer> GPUPooledRoundedStorageManager;

}  // namespace storage
}  // namespace mxnet

#endif  // MXNET_STORAGE_POOLED_STORAGE_MANAGER_H_
