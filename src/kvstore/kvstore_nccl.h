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

/**
 * @file   kvstore_nccl.h
 * @brief  NCCL implementation of KVStore
 */
#ifndef MXNET_KVSTORE_KVSTORE_NCCL_H_
#define MXNET_KVSTORE_KVSTORE_NCCL_H_

#if MXNET_USE_NCCL

#include <mxnet/kvstore.h>
#include <nccl.h>
#include <unordered_map>
#include <bitset>
#include <vector>
#include <string>
#include <utility>
#include <functional>
#include <algorithm>
#include <tuple>
#include "./comm.h"
#include "./kvstore_local.h"
#include "../common/cuda_utils.h"

// NCCL v2 introduces NCCL_MAJOR macro for versioning,
// so if there is no such macro defined in nccl.h
// then it is NCCL v1
#ifndef NCCL_MAJOR
#define NCCL_MAJOR 1
#endif

#if NCCL_MAJOR == 1
#define ncclGroupStart()
#define ncclGroupEnd()
#define ncclNumTypes nccl_NUM_TYPES
#endif  // NCCL_MAJOR == 1

namespace mxnet {
namespace kvstore {

/**
 * \brief store data in local machine using NCCL
 */
class KVStoreNCCL : public KVStoreLocal {
 public:
  KVStoreNCCL() : KVStoreLocal() {
    // Due to aggregation, we do not use the Comm interface
    comm_ = nullptr;
    pinned_ctx_ = Context::CPUPinned(0);
    inited_ = false;
  }

  virtual ~KVStoreNCCL() {
    for (auto e : nccl_data_) {
      cudaStreamDestroy(e.second.stream);
      ncclCommDestroy(e.second.comm);
    }
  }

 private:
  void InitImpl(const std::vector<int>& keys,
                const std::vector<NDArray>& values) override {
    for (size_t i = 0; i < keys.size(); ++i) {
      CHECK(local_.find(keys[i]) == local_.end())
          << "duplicate init of key " << keys[i];
      local_[keys[i]] = values[i].Copy(pinned_ctx_);
      InitKey(keys[i], values[i].storage_type(), values[i].shape(), values[i].dtype());
    }
  }

  void PushImpl(const std::vector<int>& keys,
                const std::vector<NDArray>& values,
                int priority) override {
    std::vector<int> uniq_keys;
    std::vector<std::vector<NDArray> > grouped_vals;
    GroupKVPairsHelper(keys, values, &uniq_keys, &grouped_vals);

    std::vector<const NDArray*> merged_ptrs;
    std::vector<NDArray*> local_ptrs;
    bool nccl_called = false;

    Reduce(uniq_keys, grouped_vals, priority, &merged_ptrs);

    for (size_t i = 0; i < uniq_keys.size(); ++i) {
      int key = uniq_keys[i];
      if (grouped_vals[i].size() > 1) {
        // We issued NCCL kernels, need to synchronize
        nccl_called = true;
      }
      auto& merged = *(merged_ptrs[i]);
      NDArray& local = local_[key];
      if (updater_ != nullptr) {
        CHECK(!local.is_none()) << "key " << key << " has not been inited";
        // if merged is on gpu, we may need copy weight from cpu to gpu
        if (merged.ctx().dev_mask() != cpu::kDevMask &&
            local.ctx().dev_mask() == cpu::kDevMask) {
          local = local.Copy(merged.ctx());
        }
      }
      local_ptrs.push_back(&local);
    }

    // Sync after all reductions in a group
    if (nccl_called) {
      CommSync(merged_ptrs, priority);
    }

    for (size_t i = 0; i < uniq_keys.size(); ++i) {
      int key = uniq_keys[i];
      auto& merged = *(merged_ptrs[i]);
      NDArray& local = *(local_ptrs[i]);
      if (updater_ != nullptr) {
        // call the updater with string keys
        // if string keys are used and str_updater_ is available
        // otherwise fallback to updater_ which uses int key interface
        if (key_type_ == kStringKey && str_updater_ != nullptr) {
          // after all language bindings picks up string interface changes
          const std::string &str_key = reverse_str_key_dict_[key];
          str_updater_(str_key, merged,  &local);
        } else {
          updater_(key, merged,  &local);
        }
      } else {
        local = merged;
      }
    }
  }

  void PullImpl(const std::vector<int>& keys,
                const std::vector<NDArray*>& values,
                int priority) override {
    std::vector<int> uniq_keys;
    std::vector<std::vector<NDArray*> > grouped_vals;
    GroupKVPairsHelper(keys, values, &uniq_keys, &grouped_vals);
    std::vector<NDArray> locals;
    bool nccl_called = false;

    for (size_t i = 0; i < uniq_keys.size(); ++i) {
      int key = uniq_keys[i];
      const NDArray& local = local_[key];
      locals.push_back(local_[key]);
      CHECK(!local.is_none()) << "key " << key << " has not been inited";
      if (grouped_vals[i].size() > 1) {
        // We issued NCCL kernels, need to synchronize
        nccl_called = true;
      }
    }

    Broadcast(uniq_keys, locals, grouped_vals, priority);
    // Sync after all broadcasts in a group
    if (nccl_called) {
      const std::vector<const NDArray*> values_copy(values.begin(), values.end());
      CommSync(values_copy, priority);
    }
  }

  void PullRowSparseImpl(const std::vector<int>& keys,
                         const std::vector<std::pair<NDArray*, NDArray>>& val_rowids,
                         int priority = 0) override {
    LOG(FATAL) << "NCCL kvstore does not support sparse storage type";
  }

  void SetGradientCompression(const std::vector<std::pair<std::string, std::string> >
                                      & kwargs) override {
    LOG(FATAL) << "NCCL kvstore does not support gradient compression";
  }

 protected:
  /**
   * \brief group values on keys
   */
  template <typename T>
  void GroupKVPairsHelper(const std::vector<int>& keys,
                          const std::vector<T>& values,
                          std::vector<int> *uniq_keys,
                          std::vector<std::vector<T>> *grouped_vals) {
    // check if the storage type of a value is valid
    auto validator = [this](const int key, const T nd) -> bool {
      auto stype = ptr(nd)->storage_type();
      // valid NDArray
      if (stype == kDefaultStorage) return true;
      // invalid NDArray, abort
      LOG(FATAL) << "NCCL kvstore does not support sparse storage type";
      return false;
    };
    GroupKVPairs(keys, values, uniq_keys, grouped_vals, validator);
  }

 private:
  // Aggregated reductions
  virtual void Reduce(const std::vector<int> keys,
                      const std::vector<std::vector<NDArray>>& srcs,
                      int priority,
                      std::vector<const NDArray*>* merged_ptrs) {
    std::vector<size_t> root_ids(keys.size());
    std::vector<NDArray> reduces(keys.size());
    merged_ptrs->resize(keys.size());
    std::vector<Engine::VarHandle> const_vars;
    std::vector<Engine::VarHandle> mutate_vars;

    for (size_t k = 0; k < keys.size(); ++k) {
      auto& key = keys[k];
      auto& src = srcs[k];
      auto& root_id = root_ids[k];

      // avoid extra copy for single device, but it may bring problems for
      // abnormal usage of kvstore
      if (src.size() == 1) {
        (*merged_ptrs)[k] = &src[0];
        continue;
      }

      if (!inited_) {
        std::vector<Context> devs;
        for (const auto& a : src) {
          devs.push_back(a.ctx());
        }
        InitNCCL(devs);
        InitMergeBuffer(devs);
      }

      // Check whether we got the same set of devices
      std::vector<int> dev_ids;
      for (auto e : src) {
        dev_ids.push_back(e.ctx().dev_id);
      }
      std::sort(dev_ids.begin(), dev_ids.end());
      CHECK(device_ids_ == dev_ids) << "NCCL KVStore supports only single set of devices";

      auto& buf = merge_buf_[key];
      int root = buf.merged.ctx().dev_id;
      root_id = FindRootId(src, root);

      auto& reduce = buf.merged;
      (*merged_ptrs)[k] = &reduce;
      // Need to pass NDArrays by value to the engine
      reduces[k] = reduce;

      for (size_t i = 0; i < src.size(); ++i) {
        const_vars.push_back(src[i].var());
      }
      mutate_vars.push_back(reduce.var());
    }

    Engine::Get()->PushSync([srcs, reduces, root_ids, this](RunContext rctx) {
        std::lock_guard<std::mutex> l(Storage::Get()->GetMutex(Context::kGPU));
#if (NCCL_MAJOR > 2 || (NCCL_MAJOR == 2 && NCCL_MINOR > 1))
        ncclGroupStart();
#endif
        for (size_t k = 0; k < srcs.size(); ++k) {
          auto& src = srcs[k];
          auto& root_id = root_ids[k];
          auto& reduce = reduces[k];
          if (src.size() <= 1) {
            continue;
          }
          int root = nccl_data_[src[root_id].ctx().dev_id].rank;
          ncclGroupStart();
          for (size_t i = 0; i < src.size(); ++i) {
            NCCLEntry cur = nccl_data_[src[i].ctx().dev_id];
            if (i == root_id) {
            MSHADOW_TYPE_SWITCH(src[i].dtype(), DType,
            ncclReduce(src[i].data().dptr<DType>(),
                              reduce.data().dptr<DType>(),
                              src[i].shape().Size(),
                              GetNCCLType(src[i].dtype()),
                              ncclSum,
                              root,
                              cur.comm,
                              cur.stream););
            } else {
            MSHADOW_TYPE_SWITCH(src[i].dtype(), DType,
            ncclReduce(src[i].data().dptr<DType>(),
                              NULL,
                              src[i].shape().Size(),
                              GetNCCLType(src[i].dtype()),
                              ncclSum,
                              root,
                              cur.comm,
                              cur.stream););
            }
          }
          ncclGroupEnd();
        }
#if (NCCL_MAJOR > 2 || (NCCL_MAJOR == 2 && NCCL_MINOR > 1))
        ncclGroupEnd();
#endif
      },
      Context::CPU(),
      const_vars,
      mutate_vars,
      FnProperty::kCPUPrioritized,
      priority,
      PROFILER_MESSAGE("KVStoreReduce"));
  }

  virtual void Broadcast(const std::vector<int> keys,
      const std::vector<NDArray>& srcs,
      const std::vector<std::vector<NDArray*>>& dsts,
      int priority) {
    std::vector<size_t> root_ids(keys.size());
    std::vector<Engine::VarHandle> const_vars;
    std::vector<Engine::VarHandle> mutable_vars;

    for (size_t k = 0; k < keys.size(); ++k) {
      auto& key = keys[k];
      auto& src = srcs[k];
      auto& dst = dsts[k];
      auto& root_id = root_ids[k];

      if (!inited_) {
        // copy to a random device first
        int dev_id = key % dst.size();
        CopyFromTo(src, *dst[dev_id], priority);
        for (size_t i = 0; i < dst.size(); ++i) {
          if (i != static_cast<size_t>(dev_id)) {
            CopyFromTo(*dst[dev_id], *dst[i], priority);
          }
        }
      } else {
        auto& buf = merge_buf_[key];
        int root = src.ctx().dev_id;
        assert(root == buf.ctx().dev_id);
        root_id = FindRootId(dst, root);

        // Check whether we got the same set of devices
        std::vector<int> dev_ids;
        for (size_t i = 0; i < dst.size(); ++i) {
          auto& bcast = (i == root_id) ? src : *dst[i];
          dev_ids.push_back(bcast.ctx().dev_id);
        }
        std::sort(dev_ids.begin(), dev_ids.end());
        CHECK(device_ids_ == dev_ids) << "NCCL KVStore supports only single set of devices";

        // On root perform simple copy to the output
        CopyFromTo(src, *dst[root_id], priority);
        for (size_t i = 0; i < dst.size(); ++i) {
          if ( i != root_id)
            mutable_vars.push_back(dst[i]->var());
        }
        const_vars.push_back(src.var());
      }
    }

    // If not yet inited, then all work is already scheduled
    if (!inited_) {
      return;
    }

    // We need to capture NDArrays by value
    // in order to push to the engine
    std::vector<std::vector<NDArray>> broadcasts(dsts.size());
    for (size_t i = 0; i < dsts.size(); ++i) {
      auto& broadcast = broadcasts[i];
      broadcast.resize(dsts[i].size());
      for (size_t j = 0; j < dsts[i].size(); ++j) {
        broadcast[j] = *(dsts[i][j]);
      }
    }

    Engine::Get()->PushSync([srcs, broadcasts, root_ids, this](RunContext rctx) {
        std::lock_guard<std::mutex> l(Storage::Get()->GetMutex(Context::kGPU));
#if (NCCL_MAJOR > 2 || (NCCL_MAJOR == 2 && NCCL_MINOR > 1))
        ncclGroupStart();
#endif
        for (size_t k = 0; k < srcs.size(); ++k) {
          auto& src = srcs[k];
          auto& dst = broadcasts[k];
          auto& root_id = root_ids[k];
          if (dst.size() <= 1) {
            continue;
          }

          int root = nccl_data_[src.ctx().dev_id].rank;
          ncclGroupStart();
          for (size_t i = 0; i < dst.size(); ++i) {
            auto& bcast = (i == root_id) ? src : dst[i];
            NCCLEntry cur = nccl_data_[bcast.ctx().dev_id];
            MSHADOW_TYPE_SWITCH(bcast.dtype(), DType,
                ncclBcast(bcast.data().dptr<DType>(),
                  bcast.shape().Size(),
                  GetNCCLType(bcast.dtype()),
                  root,
                  cur.comm,
                  cur.stream););
          }
          ncclGroupEnd();
        }
#if (NCCL_MAJOR > 2 || (NCCL_MAJOR == 2 && NCCL_MINOR > 1))
        ncclGroupEnd();
#endif
      },
      Context::CPU(),
      const_vars,
      mutable_vars,
      FnProperty::kCPUPrioritized,
      priority,
      PROFILER_MESSAGE("KVStoreBCast"));
  }

  // Function that waits for NCCL collective to complete
  template <typename T>
  void CommSync(const std::vector<T>& dst, int priority) {
    std::vector<Engine::VarHandle> mutate_vars;
    for (size_t i = 0; i < dst.size(); ++i) {
        mutate_vars.push_back(ptr(dst[i])->var());
    }
    Engine::Get()->PushSync([this](RunContext rctx) {
        for (auto cur : nccl_data_) {
          CUDA_CALL(cudaSetDevice(cur.second.dev_id));
          CUDA_CALL(cudaStreamSynchronize(cur.second.stream));
        }
      },
      Context::CPU(),
      {},
      mutate_vars,
      FnProperty::kCPUPrioritized,
      priority,
      PROFILER_MESSAGE("KVStoreStreamSync"));
  }

  // Initialize single key
  void InitKey(int key, const NDArrayStorageType stype, const TShape& shape,
            int dtype = mshadow::kFloat32) {
    if (stype == kDefaultStorage) {
      key_attrs_.push_back(std::make_tuple(key, shape, dtype));
    } else {
      LOG(FATAL) << "NCCL KVStore does not support sparse storage type";
    }
  }

  ncclDataType_t GetNCCLType(int dtype) {
    switch (dtype) {
      case mshadow::kFloat32:
        return ncclFloat;
      case mshadow::kFloat16:
        return ncclHalf;
      case mshadow::kFloat64:
        return ncclDouble;
      case mshadow::kUint8:
        return ncclChar;
      case mshadow::kInt32:
        return ncclInt;
      case mshadow::kInt64:
        return ncclInt64;
      default:
        LOG(FATAL) << "Unknown type passed to NCCL KVStore";
    }
    return ncclNumTypes;
  }

  void InitNCCL(const std::vector<Context>& devs) {
    for (size_t i = 0; i < devs.size(); ++i) {
      device_ids_.push_back(devs[i].dev_id);
    }
    std::sort(device_ids_.begin(), device_ids_.end());
    std::lock_guard<std::mutex> l(Storage::Get()->GetMutex(Context::kGPU));
    std::vector<ncclComm_t> comms(devs.size());
    ncclCommInitAll(&(comms[0]), devs.size(), &(device_ids_[0]));
    for (size_t i = 0; i < devs.size(); ++i) {
      NCCLEntry e;
      e.dev_id = device_ids_[i];
      e.comm = comms[i];
      e.rank = i;
      cudaSetDevice(e.dev_id);
      cudaStreamCreate(&(e.stream));
      nccl_data_[device_ids_[i]] = e;
    }
  }

  using KeyAttrs = std::tuple<int, TShape, int>;
  void InitMergeBuffer(const std::vector<Context>& devs) {
    for (size_t i = 0; i < key_attrs_.size(); ++i) {
      int key  = std::get<0>(key_attrs_[i]);
      TShape s = std::get<1>(key_attrs_[i]);
      int type = std::get<2>(key_attrs_[i]);
      auto& buf = merge_buf_[key];
      // always use devs[0] as root
      buf.merged = NDArray(s, devs[0], false, type);
    }
    inited_ = true;
  }

  // Functions that enable templates to work on both references
  // and pointers
  template<typename T>
  const T * ptr(const T & obj) { return &obj; }

  template<typename T>
  const T * ptr(T * obj) { return obj; }

  // Find which element of the vector
  // corresponds to root dev_id
  template <typename T>
  size_t FindRootId(const std::vector<T>& vec, int root) {
    size_t root_id = -1;
    for (size_t i = 0; i < vec.size(); ++i) {
      if (ptr(vec[i])->ctx().dev_id == root) {
        root_id = i;
        break;
      }
    }
    return root_id;
  }

  std::vector<KeyAttrs> key_attrs_;
  /// \brief temporal space for pushing and pulling
  struct BufferEntry {
    /// \brief the merged value
    NDArray merged;
  };
  struct NCCLEntry {
    /// \brief device ID
    int dev_id;
    /// \brief NCCL commmunicator
    ncclComm_t comm;
    /// \brief NCCL rank
    int rank;
    /// \brief GPU stream to use with NCCL
    cudaStream_t stream;
  };
  std::unordered_map<int, BufferEntry> merge_buf_;
  std::unordered_map<int, NCCLEntry> nccl_data_;
  bool inited_;
  // \brief devices used with this KVStore
  std::vector<int> device_ids_;
};
}  // namespace kvstore
}  // namespace mxnet
#endif  // MXNET_USE_NCCL
#endif  // MXNET_KVSTORE_KVSTORE_NCCL_H_
