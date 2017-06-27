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
#include <algorithm>
#include "./comm.h"

namespace mxnet {
namespace kvstore {
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
  }

  virtual ~KVStoreLocal() {
    delete comm_;
  }

  void Init(const std::vector<int>& keys,
            const std::vector<NDArray>& values) override {
    for (size_t i = 0; i < keys.size(); ++i) {
      CHECK(local_.find(keys[i]) == local_.end())
          << "duplicate init of key " << keys[i];
      local_[keys[i]] = values[i].Copy(pinned_ctx_);
      comm_->Init(keys[i], values[i].shape(), values[i].dtype());
    }
  }

  void Init(const std::vector<std::string>& str_keys,
            const std::vector<NDArray>& values) override {
    std::vector<int> keys(str_keys.size());
    for (size_t i = 0; i < str_keys.size(); ++i) {
      auto &str_key = str_keys[i];
      CHECK(str_key_dict_.find(str_key) == str_key_dict_.end())
            << "duplicate init of key " << str_key;
      auto key = next_str_key_++;
      str_key_dict_[str_key] = key;
      keys[i] = key;
    }
    Init(keys, values);
  }

  void Push(const std::vector<int>& keys,
            const std::vector<NDArray>& values,
            int priority) override {
    std::vector<int> uniq_keys;
    std::vector<std::vector<NDArray> > grouped_vals;
    GroupKVPairs(keys, values, &uniq_keys, &grouped_vals);

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
        updater_(key, merged,  &local);
      } else {
        local = merged;
      }
    }
  }

  void Pull(const std::vector<int>& keys,
            const std::vector<NDArray*>& values,
            int priority) override {
    std::vector<int> uniq_keys;
    std::vector<std::vector<NDArray*> > grouped_vals;
    GroupKVPairs(keys, values, &uniq_keys, &grouped_vals);

    for (size_t i = 0; i < uniq_keys.size(); ++i) {
      int key = uniq_keys[i];
      const NDArray& local = local_[key];
      CHECK(!local.is_none()) << "key " << key << " has not been inited";
      comm_->Broadcast(key, local, grouped_vals[i], priority);
    }
  }

  void Push(const std::vector<std::string>& str_keys,
            const std::vector<NDArray>& values,
            int priority) override {
    std::vector<int> keys(str_keys.size());
    LookupKeys(str_keys, &keys);
    Push(keys, values, priority);
  }

  void Pull(const std::vector<std::string>& str_keys,
            const std::vector<NDArray*>& values,
            int priority) override {
    std::vector<int> keys(str_keys.size());
    LookupKeys(str_keys, &keys);
    Pull(keys, values, priority);
  }

 protected:
  /**
   * \brief group values on keys
   */
  template <typename V>
  void GroupKVPairs(const std::vector<int>& keys,
                    const std::vector<V>& values,
                    std::vector<int>* uniq_keys,
                    std::vector<std::vector<V> >* grouped_vals) {
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
      if (i.first != pre_key) {
        uniq_keys->push_back(i.first);
        grouped_vals->push_back({values[i.second]});
        pre_key = i.first;;
      } else {
        grouped_vals->back().push_back(values[i.second]);
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

  /// reducer and broadcaster
  Comm* comm_;
  /// pinned context
  Context pinned_ctx_;
  /// \brief buffer for storing local values
  std::unordered_map<int, NDArray> local_;
  /// key mapping for string -> integer
  std::unordered_map<std::string, int> str_key_dict_;
  /// the next available integer for string->int key mapping
  int next_str_key_ = 0;
};
}  // namespace kvstore
}  // namespace mxnet
#endif  // MXNET_KVSTORE_KVSTORE_LOCAL_H_
