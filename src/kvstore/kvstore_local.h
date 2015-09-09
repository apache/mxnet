/**
 * Copyright (c) 2015 by Contributors
 * @file   kvstore_local.h
 * @brief  local implementation
 */
#ifndef MXNET_KVSTORE_LOCAL_H_
#define MXNET_KVSTORE_LOCAL_H_
#include <unordered_map>
#include <bitset>
#include "mxnet/kvstore.h"

namespace mxnet {

/**
 * \brief store data in local machine
 */
class KVStoreLocal : public KVStore {
 public:
  KVStoreLocal() { Clear(); }
  virtual ~KVStoreLocal() { Clear(); }

  virtual void InitDevices(const std::vector<Context>& devices) {
    num_devs_ = 0;
    for (auto d : devices) devs_[d.UID()] = num_devs_ ++;
  }

  virtual void Init(int key, const NArray& val) {
    CHECK(local_.find(key) == local_.end()) << "duplicate init of key " << key;
    Value lc_v(num_devs_, val.Copy(local_ctx_));
    local_.insert({key, lc_v});
  }

  virtual void Push(int key, const NArray& val) {
    auto it = local_.find(key);
    CHECK(it != local_.end()) << "key " << key << " has not been inited";
    auto& lc_v = it->second;
    CHECK_EQ(lc_v.val.shape(), val.shape())
        << "shape mismatch: " << lc_v.val.shape() << ", " << val.shape();

    if (aggregator_) {
      int dix = GetDevIdx(val.ctx());
      CHECK(!lc_v.pending_push[dix])
          << "duplicate push on key " << key << "from " << val.ctx().Name();
      lc_v.pending_push[dix] = true;
      lc_v.num_pending_push ++;

      if (lc_v.agg_buf.is_none()) {
        lc_v.agg_buf = NArray(lc_v.val.shape(), local_ctx_);
      }
      if (val.ctx().dev_mask == cpu::kDevMask) {
        lc_v.agg_buf += val;
      } else {
        // copy to pinned memory
        LOG(FATAL) << "TODO";
      }

      if (lc_v.num_pending_push == num_devs_) {
        // apply updater
        if (updater_) updater_(lc_v.agg_buf, &lc_v.val);

        // clean
        lc_v.agg_buf = 0.0;
        lc_v.pending_push.flip();
        lc_v.num_pending_push = 0;


        // issue blocked pull
        for (auto& w : lc_v.pending_pull_val) {
          CopyFromTo(lc_v.val, &w);
        }
        lc_v.pending_pull_val.clear();
      }
    } else {
      LOG(FATAL) << "TODO";
    }
  }

  virtual void Pull(int key, NArray* val) {
    auto it = local_.find(key);
    CHECK(it != local_.end()) << "key " << key << " has not been inited";
    auto& lc_v = it->second;
    CHECK_EQ(lc_v.val.shape(), val->shape())
        << "shape mismatch: " << lc_v.val.shape() << ", " << val->shape();

    if (aggregator_) {
      int dix = GetDevIdx(val->ctx());
      if (lc_v.pending_push[dix]) {
        lc_v.pending_pull_val.push_back(*val);
        return;
      }
      CopyFromTo(lc_v.val, val);
    } else {
      LOG(FATAL) << "TODO";
    }
  }

  virtual void Stop() { Clear(); }

 protected:
  void Clear() {
    num_devs_ = 0;
    devs_.clear();
    local_.clear();
  }

  /// get the continous device index starting from 0
  inline int GetDevIdx(const Context& ctx) {
    auto it = devs_.find(ctx.UID());
    CHECK(it != devs_.end()) << "unknow device " << ctx.Name();
    return it->second;
  }
  size_t num_devs_;
  /// map a device into an index
  std::unordered_map<uint64_t, int> devs_;

  /// internal storage of a value
  struct Value {
    Value() {}
    Value(int num_devs, NArray data)
        : pending_push(num_devs, false), num_pending_push(0) {
      val = data;
    }
    std::vector<bool> pending_push;
    std::vector<NArray> pending_push_val, pending_pull_val;
    size_t num_pending_push;
    NArray val, agg_buf;
  };
  std::unordered_map<int, Value> local_;
  Context local_ctx_;
};

}  // namespace mxnet
#endif  // MXNET_KVSTORE_LOCAL_H_
