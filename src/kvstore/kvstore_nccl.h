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
#include "./comm.h"
#include "./kvstore_local.h"


namespace mxnet {
namespace kvstore {

/**
 * \brief store data in local machine using NCCL
 */
class KVStoreNCCL : public KVStoreLocal {
 public:
  KVStoreNCCL() : KVStoreLocal() {
    comm_ = new CommNCCL();
    pinned_ctx_ = comm_->pinned_ctx();
  }

  virtual ~KVStoreNCCL() {
    delete comm_;
    comm_ = nullptr;
  }

 private:
  void InitImpl(const std::vector<int>& keys,
                const std::vector<NDArray>& values) override {
    for (size_t i = 0; i < keys.size(); ++i) {
      CHECK(local_.find(keys[i]) == local_.end())
          << "duplicate init of key " << keys[i];
      local_[keys[i]] = values[i].Copy(pinned_ctx_);
      comm_->Init(keys[i], values[i].storage_type(), values[i].shape(), values[i].dtype());
    }
  }

  void PushImpl(const std::vector<int>& keys,
                const std::vector<NDArray>& values,
                int priority) override {
    std::vector<int> uniq_keys;
    std::vector<std::vector<NDArray> > grouped_vals;
    GroupKVPairsPushHelper(keys, values, &uniq_keys, &grouped_vals);

    std::vector<const NDArray*> merged_ptrs;
    std::vector<NDArray*> local_ptrs;
    bool nccl_called = false;

    comm_->Reduce(uniq_keys, grouped_vals, priority, &merged_ptrs);

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
      comm_->CommSync(merged_ptrs, priority);
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
    GroupKVPairsPullHelper(keys, values, &uniq_keys, &grouped_vals);
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

    comm_->Broadcast(uniq_keys, locals, grouped_vals, priority);
    // Sync after all broadcasts in a group
    if (nccl_called) {
      const std::vector<const NDArray*> values_copy(values.begin(), values.end());
      comm_->CommSync(values_copy, priority);
    }
  }

  void PullRowSparseImpl(const std::vector<int>& keys,
                         const std::vector<std::pair<NDArray*, NDArray>>& val_rowids,
                         int priority = 0) override {
    LOG(FATAL) << "NCCL kvstore does not support sparse storage type";
  }

 protected:
  /**
   * \brief group values on keys
   */
  virtual void GroupKVPairsPushHelper(const std::vector<int>& keys,
                                      const std::vector<NDArray>& values,
                                      std::vector<int> *uniq_keys,
                                      std::vector<std::vector<NDArray>> *grouped_vals) {
    // check if the storage type of a value is valid
    auto validator = [this](const int key, const NDArray& nd) -> bool {
      auto stype = nd.storage_type();
      // valid NDArray
      if (stype == kDefaultStorage) return true;
      // invalid NDArray, abort
      LOG(FATAL) << "NCCL kvstore does not support sparse storage type";
      return false;
    };
    GroupKVPairs(keys, values, uniq_keys, grouped_vals, validator);
  }

  virtual void GroupKVPairsPullHelper(const std::vector<int>& keys,
                                      const std::vector<NDArray*>& values,
                                      std::vector<int> *uniq_keys,
                                      std::vector<std::vector<NDArray*>> *grouped_vals) {
    // check if the storage type of a value is valid
    auto validator = [this](const int key, const NDArray* nd) -> bool {
      auto stype = nd->storage_type();
      // valid NDArray
      if (stype == kDefaultStorage) return true;
      // invalid NDArray, abort
      LOG(FATAL) << "NCCL kvstore does not support sparse storage type";
      return false;
    };
    GroupKVPairs(keys, values, uniq_keys, grouped_vals, validator);
  }
};
}  // namespace kvstore
}  // namespace mxnet
#endif  // MXNET_USE_NCCL
#endif  // MXNET_KVSTORE_KVSTORE_NCCL_H_
