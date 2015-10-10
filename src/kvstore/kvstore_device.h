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
	  size_t min_size = size_t(-1);	
	  buf.ctx = val[0].ctx();
	  for (size_t i = 0; i < val.size(); ++i) {
		  int32_t device_id = val[i].ctx().dev_id;
		  if (total_device_buf_[device_id] < min_size) {
			  min_size = total_device_buf_[device_id];
			  buf.ctx = val[i].ctx();
		  }
	  }

      buf.merged = NDArray(val[0].shape(), buf.ctx);
	  total_device_buf_[buf.ctx.dev_id] += val[0].shape().Size();
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
  std::unordered_map<int, size_t> total_device_buf_;
  common::RANDOM_ENGINE rnd_{0};
};
}  // namespace kvstore
}  // namespace mxnet
#endif  // MXNET_KVSTORE_KVSTORE_DEVICE_H_
