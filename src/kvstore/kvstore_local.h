/**
 * Copyright (c) 2015 by Contributors
 * @file   kvstore_local.h
 * @brief  local implementation
 */
#ifndef MXNET_KVSTORE_KVSTORE_LOCAL_H_
#define MXNET_KVSTORE_KVSTORE_LOCAL_H_
#include <unordered_map>
#include <bitset>
#include <vector>
#include <utility>
#include <algorithm>
#include "mxnet/kvstore.h"

namespace mxnet {

/**
 * \brief store data in local machine
 */
class KVStoreLocal : public KVStore {
 public:
  KVStoreLocal() {
#if MXNET_USE_CUDA
    pinned_ctx_ = Context(cpu::kDevMask, Context::kPinnedMemoryID);
#else
    pinned_ctx_ = Context(cpu::kDevMask, 0);
#endif
    Clear();
  }

  virtual ~KVStoreLocal() { Clear(); }

  virtual void Start() { }

  virtual void Stop() { Clear(); }

  virtual void set_updater(const Updater& updater) {
    updater_ = updater;
  }

  virtual void set_aggregator(bool aggregator) { }

  virtual int get_rank() const { return 0; }

  virtual int get_group_size() const { return 1; }

  virtual void Init(const std::vector<int>& keys,
                    const std::vector<NDArray>& values) {
    for (size_t i = 0; i < keys.size(); ++i) {
      CHECK(local_.find(keys[i]) == local_.end())
          << "duplicate init of key " << keys[i];
      local_.insert({keys[i], values[i].Copy(pinned_ctx_)});
    }
  }

  virtual void Push(const std::vector<int>& keys,
                    const std::vector<NDArray>& values) {
    std::vector<int> uniq_keys;
    std::vector<std::vector<NDArray> > grouped_vals;
    GroupKVPairs(keys, values, &uniq_keys, &grouped_vals);

    CHECK(updater_) << "invalid updater";
    for (size_t i = 0; i < uniq_keys.size(); ++i) {
      int key = uniq_keys[i];
      auto it = local_.find(key);
      CHECK(it != local_.end()) << "key " << key << " has not been inited";
      updater_(key, MergePushValue(key, grouped_vals[i]), &it->second);
    }
  }

  virtual void Pull(const std::vector<int>& keys,
                    const std::vector<NDArray*>& values) {
    std::vector<int> uniq_keys;
    std::vector<std::vector<NDArray*> > grouped_vals;
    GroupKVPairs(keys, values, &uniq_keys, &grouped_vals);

    for (size_t i = 0; i < uniq_keys.size(); ++i) {
      int key = uniq_keys[i];
      auto it = local_.find(key);
      CHECK(it != local_.end()) << "key " << key << " has not been inited";
      for (NDArray* v : grouped_vals[i])
        CopyFromTo(it->second, v);
    }
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

  /**
   * \brief returns the aggregated push value
   */
  NDArray MergePushValue(int key, const std::vector<NDArray>& val) {
    CHECK(val.size());
    auto& buf = merge_buf_[key];
    if (buf.merged.is_none()) {
      buf.merged = val[0].Copy(pinned_ctx_);
    } else {
      CopyFromTo(val[0], &buf.merged);
    }

    for (size_t i = 1; i < val.size(); ++i) {
      const auto& v = val[i];
      if (v.ctx().dev_mask == cpu::kDevMask) {
        buf.merged += v;
      } else {
        int id = v.ctx().dev_id;
        // first copy to the pinned memory
        if (buf.gpu_buf.size() <= (size_t)id) {
          buf.gpu_buf.resize(id + 2);
        }
        if (buf.gpu_buf[id].is_none()) {
          buf.gpu_buf[id] = NDArray(v.shape(), pinned_ctx_);
        }
        CopyFromTo(v, &buf.gpu_buf[id]);
        buf.merged += buf.gpu_buf[id];
      }
    }
    return buf.merged;
  }

 private:
  void Clear() {
    updater_ = DefaultUpdater();
    merge_buf_.clear();
    local_.clear();
  }

  /// \brief temperal space for pushing value
  struct MergeBuf {
    /// \brief the cpu buffer for gpu data
    std::vector<NDArray> gpu_buf;
    /// \brief merged data in cpu
    NDArray merged;
  };

  /// \brief buffer for merging push value
  std::unordered_map<int, MergeBuf> merge_buf_;

  /// \brief local storage
  std::unordered_map<int, NDArray> local_;

  Context pinned_ctx_;

  Updater updater_;
};

}  // namespace mxnet
#endif  // MXNET_KVSTORE_KVSTORE_LOCAL_H_
