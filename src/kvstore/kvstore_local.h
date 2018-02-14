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
 * Copyright (c) 2015 by Contributors
 * @file   kvstore_local.h
 * @brief  local implementation
 */
#ifndef MXNET_KVSTORE_KVSTORE_LOCAL_H_
#define MXNET_KVSTORE_KVSTORE_LOCAL_H_

#include <mxnet/kvstore.h>
#include <unordered_map>
#include <bitset>
#include <vector>
#include <string>
#include <utility>
#include <functional>
#include <algorithm>
#include "./comm.h"
#include "./kvstore_utils.h"

namespace mxnet {
namespace kvstore {

enum KeyType {
  kUndefinedKey = -1,
  kStringKey,
  kIntKey
};

/**
 * \brief store data in local machine
 */
class KVStoreLocal : public KVStore {
 public:
  /*
   * \param use_device_comm
   */
  explicit KVStoreLocal(bool use_device_comm) : KVStore() {
    if (use_device_comm) {
      comm_ = new CommDevice();
    } else {
      comm_ = new CommCPU();
    }
    pinned_ctx_ = comm_->pinned_ctx();
    gradient_compression_ = std::make_shared<GradientCompression>();
  }

  virtual ~KVStoreLocal() {
    delete comm_;
    comm_ = nullptr;
  }

  void Init(const std::vector<int>& keys,
            const std::vector<NDArray>& values) override {
    SetKeyType(kIntKey);
    InitImpl(keys, values);
  }

  void Init(const std::vector<std::string>& str_keys,
            const std::vector<NDArray>& values) override {
    SetKeyType(kStringKey);
    std::vector<int> keys(str_keys.size());
    for (size_t i = 0; i < str_keys.size(); ++i) {
      auto &str_key = str_keys[i];
      CHECK(str_key_dict_.find(str_key) == str_key_dict_.end())
            << "duplicate init of key " << str_key;
      auto key = next_str_key_++;
      str_key_dict_[str_key] = key;
      // record reverse mapping from int to string
      reverse_str_key_dict_[key] = str_key;
      keys[i] = key;
    }
    InitImpl(keys, values);
  }

  void Push(const std::vector<int>& keys,
            const std::vector<NDArray>& values,
            int priority) override {
    SetKeyType(kIntKey);
    PushImpl(keys, values, priority);
  }

  void Pull(const std::vector<int>& keys,
            const std::vector<NDArray*>& values,
            int priority) override {
    SetKeyType(kIntKey);
    PullImpl(keys, values, priority);
  }

  void PullRowSparse(const std::vector<int>& keys,
                     const std::vector<std::pair<NDArray*, NDArray>>& val_rowids,
                     int priority = 0) override {
    SetKeyType(kIntKey);
    PullRowSparseImpl(keys, val_rowids, priority);
  }

  void Push(const std::vector<std::string>& str_keys,
            const std::vector<NDArray>& values,
            int priority) override {
    SetKeyType(kStringKey);
    std::vector<int> keys(str_keys.size());
    LookupKeys(str_keys, &keys);
    PushImpl(keys, values, priority);
  }

  void Pull(const std::vector<std::string>& str_keys,
            const std::vector<NDArray*>& values,
            int priority) override {
    SetKeyType(kStringKey);
    std::vector<int> keys(str_keys.size());
    LookupKeys(str_keys, &keys);
    PullImpl(keys, values, priority);
  }

  void PullRowSparse(const std::vector<std::string>& str_keys,
                     const std::vector<std::pair<NDArray*, NDArray>>& val_rowids,
                     int priority = 0) override {
    SetKeyType(kStringKey);
    std::vector<int> keys(str_keys.size());
    LookupKeys(str_keys, &keys);
    PullRowSparseImpl(keys, val_rowids, priority);
  }

  void SetGradientCompression(const std::vector<std::pair<std::string, std::string> >
                              & kwargs) override {
    gradient_compression_->SetParams(kwargs);
  }

 private:
  virtual void InitImpl(const std::vector<int>& keys,
                        const std::vector<NDArray>& values) {
    for (size_t i = 0; i < keys.size(); ++i) {
      CHECK(local_.find(keys[i]) == local_.end())
          << "duplicate init of key " << keys[i];
      local_[keys[i]] = values[i].Copy(pinned_ctx_);
      comm_->Init(keys[i], values[i].storage_type(), values[i].shape(), values[i].dtype());
    }
    comm_->SetGradientCompression(gradient_compression_);
  }

  virtual void PushImpl(const std::vector<int>& keys,
                        const std::vector<NDArray>& values,
                        int priority) {
    std::vector<int> uniq_keys;
    std::vector<std::vector<NDArray> > grouped_vals;
    GroupKVPairsPush(keys, values, &uniq_keys, &grouped_vals);
    for (size_t i = 0; i < uniq_keys.size(); ++i) {
      int key = uniq_keys[i];
      const NDArray& merged = comm_->Reduce(key, grouped_vals[i], priority);
      NDArray& local = local_[key];
      if (updater_ != nullptr) {
        CHECK(!local.is_none()) << "key " << key << " has not been inited";
        // if merged is on gpu, we may need copy weight from cpu to gpu
        if (merged.ctx().dev_mask() != cpu::kDevMask &&
            local.ctx().dev_mask() == cpu::kDevMask) {
          local = local.Copy(merged.ctx());
        }
        // call the updater with string keys
        // if string keys are used and str_updater_ is available
        // otherwise fallback to updater_ which uses int key interface
        if (key_type_ == kStringKey && str_updater_ != nullptr) {
          // TODO(haibin) CHECK(str_updater_ != nullptr) if use_str_key
          // after all language bindings picks up string interface changes
          const std::string &str_key = reverse_str_key_dict_[key];
          // TODO(haibin) avoid reverse key lookup if use_str_key
          str_updater_(str_key, merged,  &local);
        } else {
          updater_(key, merged,  &local);
        }
      } else {
        if (merged.storage_type() != local.storage_type()) {
          local = merged.Copy(local.ctx());
        } else {
          local = merged;
        }
      }
    }
  }

  virtual void PullImpl(const std::vector<int>& keys,
                        const std::vector<NDArray*>& values,
                        int priority) {
    std::vector<int> uniq_keys;
    std::vector<std::vector<NDArray*> > grouped_vals;
    GroupKVPairsPull(keys, values, &uniq_keys, &grouped_vals);

    for (size_t i = 0; i < uniq_keys.size(); ++i) {
      int key = uniq_keys[i];
      const NDArray& local = local_[key];
      CHECK(!local.is_none()) << "key " << key << " has not been inited";
      comm_->Broadcast(key, local, grouped_vals[i], priority);
    }
  }

  virtual void PullRowSparseImpl(const std::vector<int>& keys,
                                 const std::vector<std::pair<NDArray*, NDArray>>& val_rowids,
                                 int priority = 0) {
    std::vector<int> uniq_keys;
    std::vector<std::vector<std::pair<NDArray*, NDArray>>> grouped_val_rowids;
    GroupKVPairsPullRsp(keys, val_rowids, &uniq_keys, &grouped_val_rowids);
    for (size_t i = 0; i < uniq_keys.size(); ++i) {
      int key = uniq_keys[i];
      const NDArray& local = local_[key];
      CHECK(!local.is_none()) << "key " << key << " has not been inited";
      CHECK_EQ(local.storage_type(), kRowSparseStorage)
               << "PullRowSparse expects row_sparse src NDArray";
      auto &target_val_rowids = grouped_val_rowids[i];
      const size_t num_vals = target_val_rowids.size();
      for (size_t j = 0; j < num_vals; j++) {
        auto &row_id = target_val_rowids[j].second;
        NDArray indices(row_id.shape(), local.ctx(), false, mshadow::kInt64);
        CopyFromTo(row_id, &indices, 0);
        Unique(&indices, priority);
        target_val_rowids[j].second = indices;
      }
      comm_->BroadcastRowSparse(key, local, grouped_val_rowids[i], false, priority);
    }
  }

 protected:
  KVStoreLocal() : KVStore() {}
  /**
   * \brief set the key type of the kvstore if haven't already.
   * If the key type is already defined, check if it matches the provided key type
   */
  void SetKeyType(const KeyType key_type) {
    if (key_type_ == kUndefinedKey) key_type_ = key_type;
    CHECK_EQ(key_type_, key_type) << "Mixed key types are not allowed";
  }

  /**
   * \brief group values on keys for push
   */
  virtual void GroupKVPairsPush(const std::vector<int>& keys,
                                const std::vector<NDArray>& values,
                                std::vector<int> *uniq_keys,
                                std::vector<std::vector<NDArray>> *grouped_vals) {
    // check if the storage type of a value is valid
    auto validator = [this](const int key, const NDArray& nd) -> bool {
      auto stype = nd.storage_type();
      // valid NDArray
      if (stype == kDefaultStorage || stype == kRowSparseStorage) return true;
      // invalid NDArray, abort
      LOG(FATAL) << "Unexpected storage type detected during kvstore push: " << stype;
      return false;
    };
    GroupKVPairs(keys, values, uniq_keys, grouped_vals, validator);
  }
  /**
   * \brief group values on keys for pull
   */
  virtual void GroupKVPairsPull(const std::vector<int>& keys,
                                const std::vector<NDArray*>& values,
                                std::vector<int> *uniq_keys,
                                std::vector<std::vector<NDArray*>> *grouped_vals) {
    // check if the storage type of a value is valid
    auto validator = [this](const int key, const NDArray* nd) -> bool {
      // valid
      if (nd->storage_type() == kDefaultStorage) return true;
      // invalid, print warning messages once
      if (this->warnings_printed_.find(key) == this->warnings_printed_.end()) {
        LOG(INFO) << "Warning: non-default weights detected during kvstore pull. "
                  << "This call has been ignored. "
                  << "Please make sure to use row_sparse_pull with row_ids.";
        this->warnings_printed_.insert(key);
      }
      return false;
    };
    GroupKVPairs(keys, values, uniq_keys, grouped_vals, validator);
  }

  typedef std::pair<NDArray*, NDArray> RSPVal;
  /**
   * \brief group values on keys for row_sparse_pull
   */
  virtual void GroupKVPairsPullRsp(const std::vector<int>& keys,
                                   const std::vector<RSPVal>& values,
                                   std::vector<int> *uniq_keys,
                                   std::vector<std::vector<RSPVal>> *grouped_vals) {
    // check if the storage type of a value is valid
    auto validator = [this](const int key, const RSPVal& val_rowid) -> bool {
      auto val_stype = val_rowid.first->storage_type();
      auto rowid_stype = val_rowid.second.storage_type();
      // check storage types
      CHECK_EQ(val_stype, kRowSparseStorage) << "Expected row_sparse storage type for "
              << "row_sparse_pull values, but detected storage type " << val_stype;
      CHECK_EQ(rowid_stype, kDefaultStorage) << "Expected default storage type for "
              << "row_sparse_pull rowids, but detected storage type " << rowid_stype;
      return true;
    };
    GroupKVPairs(keys, values, uniq_keys, grouped_vals, validator);
  }

  /**
   * \brief group values on keys with validation.
   * A value `v` is not included in the result if is_valid(v) returns false.
   */
  template <typename V, typename FValidate>
  void GroupKVPairs(const std::vector<int>& keys,
                    const std::vector<V>& values,
                    std::vector<int>* uniq_keys,
                    std::vector<std::vector<V> >* grouped_vals,
                    const FValidate& is_valid) {
    CHECK_EQ(keys.size(), values.size());
    // TODO(mli) check if already sorted as an optimization
    using Idx = std::pair<int, int>;
    std::vector<Idx> idx(keys.size());
    for (size_t i = 0; i < keys.size(); ++i) {
      idx[i].first = keys[i]; idx[i].second = i;
    }
    std::sort(idx.begin(), idx.end(), [](const Idx& a, const Idx& b) {
        return a.first < b.first;
      });

    int pre_key = idx[0].first - 1;
    for (auto i : idx) {
      if (is_valid(i.first, values[i.second])) {
        if (i.first != pre_key) {
          uniq_keys->push_back(i.first);
          grouped_vals->push_back({values[i.second]});
          pre_key = i.first;
        } else {
          grouped_vals->back().push_back(values[i.second]);
        }
      }
    }
  }

  void LookupKeys(const std::vector<std::string>& str_keys,
                  std::vector<int> *keys) {
    for (size_t i = 0; i < str_keys.size(); ++i) {
      auto &str_key = str_keys[i];
      CHECK(str_key_dict_.find(str_key) != str_key_dict_.end())
            << "key " << str_key << " doesn't exist. Did you init?";
      keys->at(i) = str_key_dict_[str_key];
    }
  }

  /**
   * \brief sort and get unique values.
   */
  void Unique(NDArray *out, int priority) {
    Resource rsc = ResourceManager::Get()->Request(out->ctx(),
      ResourceRequest(ResourceRequest::kTempSpace));
    Engine::Get()->PushAsync(
      [rsc, out](RunContext rctx, Engine::CallbackOnComplete on_complete) {
        NDArray *output = out;
        CHECK_EQ(out->shape().ndim(), 1) << "Unique expects 1D inputs";
        nnvm::dim_t size = out->shape()[0];
        switch (out->ctx().dev_mask()) {
          case cpu::kDevMask: {
            mshadow::Stream<cpu> *s = rctx.get_stream<cpu>();
            UniqueImpl(rsc, s, output, size);
            break;
          }
  #if MXNET_USE_CUDA
          case gpu::kDevMask: {
            mshadow::Stream<gpu> *s = rctx.get_stream<gpu>();
            UniqueImpl(rsc, s, output, size);
            // wait for GPU operations to complete
            s->Wait();
            break;
          }
  #endif
          default:
            LOG(FATAL) << "GPU not enabled.";
        }
        on_complete();
      }, out->ctx(), {}, {out->var(), rsc.var},
      FnProperty::kNormal, priority, PROFILER_MESSAGE("KVStoreUnique"));
    out->WaitToRead();
  }


  /// reducer and broadcaster
  Comm* comm_;
  /// pinned context
  Context pinned_ctx_;
  /// \brief buffer for storing local values
  std::unordered_map<int, NDArray> local_;
  /// key mapping for string -> integer
  std::unordered_map<std::string, int> str_key_dict_;
  /// reverse key mapping for integer -> string
  std::unordered_map<int, std::string> reverse_str_key_dict_;
  /// the next available integer for string->int key mapping
  int next_str_key_ = 0;
  /// whether printed warning due to mismatch stype in each key
  std::unordered_set<int> warnings_printed_;
  /// whether int or string is used for keys
  KeyType key_type_ = kUndefinedKey;
};
}  // namespace kvstore
}  // namespace mxnet
#endif  // MXNET_KVSTORE_KVSTORE_LOCAL_H_
