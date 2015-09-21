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
    pinned_ctx_ = (MXNET_USE_CUDA != 0) ?
        Context::CPUPinned(0) : Context::CPU();
    set_updater(DefaultUpdater());
  }

  void set_updater(Updater updater) override {
    updater_ = updater;
  }

  void Init(const std::vector<int>& keys,
            const std::vector<NDArray>& values) override {
    for (size_t i = 0; i < keys.size(); ++i) {
      CHECK(local_.find(keys[i]) == local_.end())
          << "duplicate init of key " << keys[i];
      local_[keys[i]] = values[i].Copy(pinned_ctx_);
    }
  }

  void Push(const std::vector<int>& keys,
            const std::vector<NDArray>& values) override {
    std::vector<int> uniq_keys;
    std::vector<std::vector<NDArray> > grouped_vals;
    GroupKVPairs(keys, values, &uniq_keys, &grouped_vals);
    CHECK(updater_) << "invalid updater";
    for (size_t i = 0; i < uniq_keys.size(); ++i) {
      int key = uniq_keys[i];
      auto it = local_.find(key);
      CHECK(it != local_.end()) << "key " << key << " has not been inited";
      updater_(key, MergePushValue(key, grouped_vals[i]), &(it->second));
    }
  }

  void Pull(const std::vector<int>& keys,
            const std::vector<NDArray*>& values) override {
    std::vector<int> uniq_keys;
    std::vector<std::vector<NDArray*> > grouped_vals;
    GroupKVPairs(keys, values, &uniq_keys, &grouped_vals);

    for (size_t i = 0; i < uniq_keys.size(); ++i) {
      int key = uniq_keys[i];
      auto it = local_.find(key);
      CHECK(it != local_.end()) << "key " << key << " has not been inited";
      for (NDArray* v : grouped_vals[i]) {
        CopyFromTo(it->second, v);
      }
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
  /*!
   * \brief returns the aggregated push value
   */
  const NDArray &MergePushValue(int key, const std::vector<NDArray>& val) {
    CHECK(val.size());
    auto& buf = merge_buf_[key];

    if (buf.merged.is_none()) {
      buf.merged = NDArray(val[0].shape(), pinned_ctx_);
    }
    CopyFromTo(val[0], &buf.merged);

    for (size_t i = 1; i < val.size(); ++i) {
      const auto& v = val[i];
      Context ctx = v.ctx();
      if (v.ctx().dev_mask() == cpu::kDevMask) {
        buf.merged += v;
      } else {
        CHECK_EQ(ctx.dev_mask(), gpu::kDevMask);
        NDArray *copy_buf = buf.AllocCopyBuf(ctx.dev_id, val[0].shape());
        CopyFromTo(val[0], copy_buf);
        buf.merged += *copy_buf;
      }
    }
    return buf.merged;
  }

 private:
  /// \brief temperal space for pushing and pull
  struct BufferEntry {
    /// \brief the cpu buffer for gpu data
    std::vector<NDArray> copy_buf;
    /// \brief merged data in cpu
    NDArray merged;
    // allocate copy buffer, if it has not been allocated
    inline NDArray *AllocCopyBuf(uint32_t dev_id, const TShape& shape) {
      if (dev_id >= copy_buf.size()) copy_buf.resize(dev_id + 1);
      if (copy_buf[dev_id].is_none()) {
        copy_buf[dev_id] = NDArray(shape, Context::CPUPinned(dev_id));
      }
      return &copy_buf[dev_id];
    }
  };
  /// \brief buffer for merging push value
  std::unordered_map<int, BufferEntry> merge_buf_;
  /// \brief buffer for storing local values
  std::unordered_map<int, NDArray> local_;
  // pinned context
  Context pinned_ctx_;
  // updater
  Updater updater_;
};

}  // namespace mxnet
#endif  // MXNET_KVSTORE_KVSTORE_LOCAL_H_
