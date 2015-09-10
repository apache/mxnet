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
#include "mxnet/kvstore.h"

namespace mxnet {

/**
 * \brief store data in local machine
 */
class KVStoreLocal : public KVStore {
 public:
  KVStoreLocal() : pinned_ctx_(cpu::kDevMask, Context::kPinnedMemoryID) {
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

  virtual void Init(int key, const NArray& val) {
    CHECK(local_.find(key) == local_.end()) << "duplicate init of key " << key;
    local_.insert({key, val.Copy(local_ctx_)});
  }

  virtual void Push(int key, const std::vector<NArray>& val) {
    auto it = local_.find(key);
    CHECK(it != local_.end()) << "key " << key << " has not been inited";

    CHECK(updater_) << "invalid updater";
    updater_(MergePushValue(key, val), &it->second);
  }

  virtual void Pull(int key, NArray* val) {
    auto it = local_.find(key);
    CHECK(it != local_.end()) << "key " << key << " has not been inited";
    CopyFromTo(it->second, val);
  }

 protected:
  /**
   * \brief returns the aggregated push value
   */
  NArray MergePushValue(int key, const std::vector<NArray>& val) {
    CHECK(val.size());
    auto& buf = merge_buf_[key];
    if (buf.merged.is_none()) {
      buf.merged = val[0].Copy(local_ctx_);
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
          buf.gpu_buf[id] = NArray(v.shape(), pinned_ctx_);
        }
        CopyFromTo(v, &buf.gpu_buf[id]);
        buf.merged += buf.gpu_buf[id];
      }
    }
    return buf.merged;
  }

  void Clear() {
    updater_ = DefaultUpdater();
    merge_buf_.clear();
    local_.clear();
  }

  Updater updater_;

  /// \brief temperal space for pushing value
  struct MergeBuf {
    /// \brief the cpu buffer for gpu data
    std::vector<NArray> gpu_buf;
    /// \brief merged data in cpu
    NArray merged;
  };

 private:
  /// \brief buffer for merging push value
  std::unordered_map<int, MergeBuf> merge_buf_;

  /// \brief local storage
  std::unordered_map<int, NArray> local_;

  Context local_ctx_;
  Context pinned_ctx_;
};

}  // namespace mxnet
#endif  // MXNET_KVSTORE_KVSTORE_LOCAL_H_
