/*!
 * Copyright (c) 2015 by Contributors
 * \file kvstore_device.h
 * \brief Device implementation of KVStore that do reduction on GPU reduction.
 */
#ifndef MXNET_KVSTORE_KVSTORE_DEVICE_H_
#define MXNET_KVSTORE_KVSTORE_DEVICE_H_

#include <mxnet/kvstore.h>
#include <unordered_map>
#include <vector>
#include <utility>
#include <algorithm>
#include <limits>
#include "./kvstore_local.h"
#include "../common/utils.h"

namespace mxnet {
namespace kvstore {
/*!
 * \brief Device implementation of KVStore that do reduction on GPU reduction.
 */
class KVStoreDevice : public KVStoreLocal {
 public:
  explicit KVStoreDevice(bool device_mode)
      : device_mode_(device_mode) {}

 protected:
  using KeyShape = std::pair<int, TShape>;
  void Init(const std::vector<int>& keys,
            const std::vector<NDArray>& values) override {
    KVStoreLocal::Init(keys, values);

    for (size_t i = 0; i < keys.size(); ++i) {
      sorted_key_shape_.push_back(std::make_pair(keys[i], values[i].shape()));
    }
  }

  void InitMergeBuffers(const std::vector<NDArray>& val) {
    std::sort(sorted_key_shape_.begin(), sorted_key_shape_.end(), [](
              const KeyShape& a, const KeyShape& b) {
      return a.second.Size() > b.second.Size();
    });

    CHECK(!val.empty());
    std::unordered_map<int, std::pair<Context, size_t>> ctx_info;
    for (size_t i = 0; i < val.size(); ++i) {
      int32_t dev_id = val[i].ctx().dev_id;
      ctx_info[dev_id] = std::make_pair(val[i].ctx(), 0);
    }
    for (size_t i = 0; i < sorted_key_shape_.size(); ++i) {
      int k = sorted_key_shape_[i].first;
      TShape s = sorted_key_shape_[i].second;
      auto& tm_buf = merge_buf_[k];
      size_t min_size = std::numeric_limits<size_t>::max();
      for (auto it = ctx_info.begin(); it != ctx_info.end(); ++it) {
        size_t tm_size = it->second.second;
        if (tm_size <= min_size) {
          tm_buf.ctx = it->second.first;
          min_size = tm_size;
        }
      }

      tm_buf.merged = NDArray(s, Context::CPUPinned(tm_buf.ctx.dev_id));
      tm_buf.merged_device = NDArray(s, tm_buf.ctx);
      ctx_info[tm_buf.ctx.dev_id].second += s.Size();
    }
  }

  const NDArray& MergePushValue(
      int key, const std::vector<NDArray>& val, int priority) override {
    if (!device_mode_) {
      return KVStoreLocal::MergePushValue(key, val, priority);
    }
    if (!buf_initialized_) {
      InitMergeBuffers(val);
      buf_initialized_ = true;
    }

    auto& buf = merge_buf_[key];
    std::vector<NDArray> reduce(val.size());
    CHECK(!buf.merged_device.is_none());
    CopyFromTo(val[0], &(buf.merged_device), priority);
    reduce[0] = buf.merged_device;

    for (size_t i = 1; i < val.size(); ++i) {
      NDArray *copy_buf = buf.AllocCopyBuf(
          i, buf.ctx, val[0].shape());
      CopyFromTo(val[i], copy_buf, priority);
      reduce[i] = *copy_buf;
    }
    ElementwiseSum(reduce, &buf.merged_device);

    if (updater_ != nullptr) {
      CopyFromTo(buf.merged_device, &(buf.merged));
      return buf.merged;
    } else {
      return buf.merged_device;
    }
  }

  void ScatterPullValue(
      int key,
      const NDArray& src,
      const std::vector<NDArray*>& vals,
      int priority) override {
    if (!device_mode_) {
      KVStoreLocal::ScatterPullValue(key, src, vals, priority);
      return;
    }
    auto it = merge_buf_.find(key);
    if (it != merge_buf_.end() && it->first == key) {
      auto& buf = it->second;
      if (!buf.merged_device.is_none()) {
        CopyFromTo(src, &(buf.merged_device));
        for (auto* vptr : vals) {
          CopyFromTo(buf.merged_device, vptr, priority);
        }
        return;
      }
    }
    // default, copy back
    for (auto* vptr : vals) {
      CopyFromTo(src, vptr, priority);
    }
  }

 private:
  bool device_mode_;
  bool buf_initialized_{false};
  std::vector<KeyShape> sorted_key_shape_;
};
}  // namespace kvstore
}  // namespace mxnet
#endif  // MXNET_KVSTORE_KVSTORE_DEVICE_H_
