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
#include <random>
#include "./kvstore_local.h"
#include "../common/utils.h"

namespace mxnet {
namespace kvstore {
/*!
 * \brief Device implementation of KVStore that do reduction on GPU reduction.
 */
class KVStoreDevice : public KVStoreLocal {
 protected:
  const NDArray& MergePushValue(
      int key, const std::vector<NDArray>& val, int priority) override {
    if (updater_ != nullptr) {
      // fall back to CPU based update if updater presents
      return KVStoreLocal::MergePushValue(key, val, priority);
    }
    auto& buf = merge_buf_[key];
    std::vector<NDArray> reduce(val.size());
    if (buf.merged.is_none()) {
      // round robin to assign the reduction device.
      size_t round_robin_index =
          std::uniform_int_distribution<size_t>(0, val.size()-1)(rnd_);
      buf.ctx = val[round_robin_index].ctx();
      buf.merged = NDArray(val[0].shape(), buf.ctx);
    }
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
  common::RANDOM_ENGINE rnd_{0};
};
}  // namespace kvstore
}  // namespace mxnet
#endif  // MXNET_KVSTORE_KVSTORE_DEVICE_H_
