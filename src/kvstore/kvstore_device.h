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

      tm_buf.merged = NDArray(s, tm_buf.ctx);
      ctx_info[tm_buf.ctx.dev_id].second += s.Size();
    }
  }

  const NDArray& MergePushValue(
      int key, const std::vector<NDArray>& val, int priority) override {
    if (updater_ != nullptr) {
      // fall back to CPU based update if updater presents
      return KVStoreLocal::MergePushValue(key, val, priority);
    }

    if (merge_buf_.empty()) {
      InitMergeBuffers(val);
    }

    auto& buf = merge_buf_[key];
    std::vector<NDArray> reduce(val.size());
    CHECK(!buf.merged.is_none());
    CopyFromTo(val[0], &(buf.merged), priority);
    reduce[0] = buf.merged;

    for (size_t i = 1; i < val.size(); ++i) {
      NDArray *copy_buf = buf.AllocCopyBuf(
          i, buf.ctx, val[0].shape());
      CopyFromTo(val[i], copy_buf, priority);
      reduce[i] = *copy_buf;
    }
    ElementwiseSum(reduce, &buf.merged);
    return buf.merged;
  }

 private:
  std::vector<KeyShape> sorted_key_shape_;
};
}  // namespace kvstore
}  // namespace mxnet
#endif  // MXNET_KVSTORE_KVSTORE_DEVICE_H_
