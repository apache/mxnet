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
#include "./comm_tree.h"
#include "./kvstore_utils.h"
#include "../ndarray/ndarray_function.h"

namespace mxnet {
namespace kvstore {
/*!
 * \brief Splits a string into smaller strings using char as delimiter
 * Example: "a,b,c,,d" is split into ["a","b","c","","d"]
 * \param s string to split
 * \param delim char to split string around
 * \param result container for tokens extracted after splitting
 */
template<typename Out>
void split(const std::string &s, const char delim, Out result) {
  std::stringstream ss;
  ss.str(s);
  std::string item;
  while (std::getline(ss, item, delim)) {
    *(result++) = item;
  }
}

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
      bool tree = dmlc::GetEnv("MXNET_KVSTORE_USETREE", 0) & MXNET_USE_CUDA;
      if (tree) {
        comm_ = new CommDeviceTree();
      } else {
        comm_ = new CommDevice();
      }
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
            int priority,
            bool ignore_sparse) override {
    SetKeyType(kIntKey);
    PullImpl(keys, values, priority, ignore_sparse);
  }

  void Broadcast(const std::vector<int>& vkeys,
                 const std::vector<int>& okeys,
                 const std::vector<NDArray>& values,
                 const std::vector<NDArray*>& outs,
                 int priority) override {
    SetKeyType(kIntKey);
    BroadcastImpl(vkeys, okeys, values, outs, priority);
  }

  void PushPull(const std::vector<int>& vkeys,
                const std::vector<int>& okeys,
                const std::vector<NDArray>& values,
                const std::vector<NDArray*>& outs,
                int priority) override {
    SetKeyType(kIntKey);
    PushPullImpl(vkeys, okeys, values, outs, priority);
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
            int priority,
            bool ignore_sparse) override {
    SetKeyType(kStringKey);
    std::vector<int> keys(str_keys.size());
    LookupKeys(str_keys, &keys);
    PullImpl(keys, values, priority, ignore_sparse);
  }

  void Broadcast(const std::vector<std::string>& str_vkeys,
                 const std::vector<std::string>& str_okeys,
                 const std::vector<NDArray>& values,
                 const std::vector<NDArray*>& outs,
                 int priority) override {
    SetKeyType(kStringKey);
    std::vector<int> vkeys(str_vkeys.size());
    std::vector<int> okeys(str_okeys.size());
    for (size_t i = 0; i < str_vkeys.size(); ++i) {
      auto &str_key = str_vkeys[i];
      CHECK(str_key_dict_.find(str_key) == str_key_dict_.end())
            << "duplicate init of key " << str_key;
      auto key = next_str_key_++;
      str_key_dict_[str_key] = key;
      // record reverse mapping from int to string
      reverse_str_key_dict_[key] = str_key;
      vkeys[i] = key;
    }
    LookupKeys(str_okeys, &okeys);
    BroadcastImpl(vkeys, okeys, values, outs, priority);
  }

  void PushPull(const std::vector<std::string>& str_vkeys,
                const std::vector<std::string>& str_okeys,
                const std::vector<NDArray>& values,
                const std::vector<NDArray*>& outs,
                int priority) override {
    SetKeyType(kStringKey);
    std::vector<int> vkeys(str_vkeys.size());
    std::vector<int> okeys(str_okeys.size());
    LookupKeys(str_vkeys, &vkeys);
    LookupKeys(str_okeys, &okeys);
    PushPullImpl(vkeys, okeys, values, outs, priority);
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
          << "duplicate init of key " << keys[i]
          << ". Please double check if you called kv.init or kv.broadcast with this key "
          << "multiple times";
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
    GroupKVPairsPush(keys, values, &uniq_keys, &grouped_vals, false);
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
                        int priority,
                        bool ignore_sparse) {
    std::vector<int> uniq_keys;
    std::vector<std::vector<NDArray*> > grouped_vals;
    GroupKVPairsPull(keys, values, &uniq_keys, &grouped_vals, ignore_sparse);

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
    GroupKVPairsPullRsp(keys, val_rowids, &uniq_keys, &grouped_val_rowids, false);
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
        target_val_rowids[j].second = Unique(row_id, local.ctx(), 0);
      }
      comm_->BroadcastRowSparse(key, local, grouped_val_rowids[i], priority);
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

  virtual void BroadcastImpl(const std::vector<int>& vkeys,
                             const std::vector<int>& okeys,
                             const std::vector<NDArray>& values,
                             const std::vector<NDArray*>& outs,
                             int priority) {
    InitImpl(vkeys, values);
    PullImpl(okeys, outs, priority, true);
  }

  virtual void PushPullImpl(const std::vector<int>& vkeys,
                            const std::vector<int>& okeys,
                            const std::vector<NDArray>& values,
                            const std::vector<NDArray*>& outs,
                            int priority) {
    PushImpl(vkeys, values, priority);
    PullImpl(okeys, outs, priority, true);
  }

  /**
   * \brief group values on keys for push
   */
  virtual void GroupKVPairsPush(const std::vector<int>& keys,
                                const std::vector<NDArray>& values,
                                std::vector<int> *uniq_keys,
                                std::vector<std::vector<NDArray>> *grouped_vals,
                                bool ignore_sparse) {
    // check if the storage type of a value is valid
    auto validator = [](const int key, const NDArray& nd, bool ignore_sparse) -> bool {
      CHECK(!ignore_sparse) << "Cannot ignore sparse arrays for push";
      auto stype = nd.storage_type();
      // valid NDArray
      if (stype == kDefaultStorage || stype == kRowSparseStorage) return true;
      // invalid NDArray, abort
      LOG(FATAL) << "Unexpected storage type detected during kvstore push: " << stype;
      return false;
    };
    GroupKVPairs(keys, values, uniq_keys, grouped_vals, validator, ignore_sparse);
  }
  /**
   * \brief group values on keys for pull
   */
  virtual void GroupKVPairsPull(const std::vector<int>& keys,
                                const std::vector<NDArray*>& values,
                                std::vector<int> *uniq_keys,
                                std::vector<std::vector<NDArray*>> *grouped_vals,
                                bool ignore_sparse) {
    // check if the storage type of a value is valid
    auto validator = [this](const int key, const NDArray* nd, bool ignore_sparse) -> bool {
      // valid
      if (nd->storage_type() == kDefaultStorage || !ignore_sparse) return true;
      // invalid, print warning messages once
      if (this->warnings_printed_.find(key) == this->warnings_printed_.end()) {
        LOG(INFO) << "Warning: non-default weights detected during kvstore pull. "
                     "This call has been ignored. Please make sure to use "
                     "kv.row_sparse_pull() or module.prepare() with row_ids.";
        this->warnings_printed_.insert(key);
      }
      return false;
    };
    GroupKVPairs(keys, values, uniq_keys, grouped_vals, validator, ignore_sparse);
  }

  typedef std::pair<NDArray*, NDArray> RSPVal;
  /**
   * \brief group values on keys for row_sparse_pull
   */
  virtual void GroupKVPairsPullRsp(const std::vector<int>& keys,
                                   const std::vector<RSPVal>& values,
                                   std::vector<int> *uniq_keys,
                                   std::vector<std::vector<RSPVal>> *grouped_vals,
                                   bool ignore_sparse) {
    // check if the storage type of a value is valid
    auto validator = [](const int key, const RSPVal& val_rowid, bool ignore_sparse) -> bool {
      CHECK(!ignore_sparse) << "Cannot ignore sparse arrays in row_sparse_pull";
      auto val_stype = val_rowid.first->storage_type();
      auto rowid_stype = val_rowid.second.storage_type();
      // check storage types
      CHECK_EQ(val_stype, kRowSparseStorage) << "Expected row_sparse storage type for "
              << "row_sparse_pull values, but detected storage type " << val_stype;
      CHECK_EQ(rowid_stype, kDefaultStorage) << "Expected default storage type for "
              << "row_sparse_pull rowids, but detected storage type " << rowid_stype;
      return true;
    };
    GroupKVPairs(keys, values, uniq_keys, grouped_vals, validator, ignore_sparse);
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
                    const FValidate& is_valid,
                    bool ignore_sparse) {
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
      if (is_valid(i.first, values[i.second], ignore_sparse)) {
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

  /*
   * \brief Compute the unique values in data and store them in ascending order
   * in an int64_t row_sparse ndarray on ctx. The opeartion is async. The result
   * row_sparse ndarray stores the unique values in out.data(). The aux_data()
   * contains values that are not necessarily meaningful and should be ignored.
   * \param data the input data
   * \param ctx the target context
   * \param priority the priority of the operation
   */
  NDArray Unique(const NDArray &data, Context ctx, int priority) {
    // create kRowSparseStorage output ndarray
    const size_t num_elements = data.shape().Size();
    NDArray out(kRowSparseStorage, mshadow::Shape2(num_elements, 1),
                ctx, true, mshadow::kInt64);
    bool diff_ctx = data.ctx() != ctx;
    NDArray data_in_ctx = diff_ctx ? NDArray(data.shape(), ctx, true, data.dtype()) : data;
    // if data == data_in_ctx, CopyFromTo is smart enough to skip the copy
    CopyFromTo(data, &data_in_ctx, priority);
    // GPU requires temp resources
    bool is_gpu = out.ctx().dev_mask() == gpu::kDevMask;
    Engine::Get()->PushAsync(
      [=](RunContext rctx, Engine::CallbackOnComplete on_complete) {
        // copy data.data() to out.data()
        out.CheckAndAlloc({mshadow::Shape1(num_elements)});
        TBlob out_data = out.data();
        NDArray workspace;
        switch (out.ctx().dev_mask()) {
          case cpu::kDevMask: {
            mshadow::Stream<cpu> *s = rctx.get_stream<cpu>();
            ndarray::Copy<cpu, cpu>(data_in_ctx.data(), &out_data,
                                    ctx, ctx, rctx);
            UniqueImpl(&workspace, s, out);
            break;
          }
  #if MXNET_USE_CUDA
          case gpu::kDevMask: {
            mshadow::Stream<gpu> *s = rctx.get_stream<gpu>();
            ndarray::Copy<gpu, gpu>(data_in_ctx.data(), &out_data,
                                    ctx, ctx, rctx);
            UniqueImpl(&workspace, s, out);
            // wait for GPU operations to complete
            s->Wait();
            break;
          }
  #endif
          default:
            LOG(FATAL) << MXNET_GPU_NOT_ENABLED_ERROR;
        }
        on_complete();
      }, out.ctx(), {data_in_ctx.var()}, {out.var()},
      is_gpu ? FnProperty::kGPUPrioritized : FnProperty::kCPUPrioritized,
      priority, "KVStoreUnique");
    return out;
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
