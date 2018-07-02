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
 * Copyright (c) 2018 by Contributors
 * @file   kvstore_dist_sync_allreduce.h
 * @brief  distributed implementation based on allreduce
 */
#ifndef MXNET_KVSTORE_KVSTORE_DIST_SYNC_ALLREDUCE_H_
#define MXNET_KVSTORE_KVSTORE_DIST_SYNC_ALLREDUCE_H_

#include <mxnet/kvstore.h>
#include <unordered_map>
#include <bitset>
#include <vector>
#include <string>
#include <utility>
#include <functional>
#include <algorithm>
#include "./comm.h"
#include "./kvstore_utils.h"
#include "./kvstore_local.h"

#if MXNET_USE_ALLREDUCE_DIST_KVSTORE
#include "collectives/include/collectives.h"

namespace mxnet {
namespace kvstore {

/**
 * \brief store data in local machine
 */
class KVStoreDistSyncAllReduce : public KVStoreLocal {
 public:
  explicit KVStoreDistSyncAllReduce(bool use_device_comm)
  : KVStoreLocal(use_device_comm) {
    int ret = MXCOLLIBInit(comm_);
    if (ret != 0) {
      LOG(FATAL) << "kvstore with type [" << type_ << "] failed with collective library init";
    }
  }

  virtual ~KVStoreDistSyncAllReduce() {
  }

  void Push(const std::vector<int>& keys,
            const std::vector<NDArray>& values,
            int priority) override {
    LOG(FATAL) << "Not supported in KVStore with type " << type_ << ".";
  }

  void Pull(const std::vector<int>& keys,
            const std::vector<NDArray*>& values,
            int priority) override {
    LOG(FATAL) << "Not supported in KVStore with type " << type_ << ".";
  }

  void PullRowSparse(const std::vector<int>& keys,
                     const std::vector<std::pair<NDArray*, NDArray>>& val_rowids,
                     int priority = 0) override {
    LOG(FATAL) << "Not supported in KVStore with type " << type_ << ".";
  }

  void Push(const std::vector<std::string>& str_keys,
            const std::vector<NDArray>& values,
            int priority) override {
    LOG(FATAL) << "Not supported in KVStore with type " << type_ << ".";
  }

  void Pull(const std::vector<std::string>& str_keys,
            const std::vector<NDArray*>& values,
            int priority) override {
    LOG(FATAL) << "Not supported in KVStore with type " << type_ << ".";
  }

  void PullRowSparse(const std::vector<std::string>& str_keys,
                     const std::vector<std::pair<NDArray*, NDArray>>& val_rowids,
                     int priority = 0) override {
    LOG(FATAL) << "Not supported in KVStore with type " << type_ << ".";
  }

  void SetGradientCompression(const std::vector<std::pair<std::string, std::string> >
                              & kwargs) override {
    LOG(FATAL) << "Not supported in KVStore with type " << type_ << ".";
  }

  void PushPull(const std::vector<int> &keys,
                const std::vector<NDArray> &in_values,
                const std::vector<NDArray*> &out_values,
                int priority) override {
    SetKeyType(kIntKey);
    PushPullImpl(keys, in_values, out_values, priority);
  }

  void PushPull(const std::vector<std::string> &str_keys,
                const std::vector<NDArray> &in_values,
                const std::vector<NDArray*> &out_values,
                int priority) override {
    SetKeyType(kStringKey);
    std::vector<int> keys(str_keys.size());
    LookupKeys(str_keys, &keys);
    PushPullImpl(keys, in_values, out_values, priority);
  }

  void Broadcast(const std::vector<int> &keys,
                 const std::vector<NDArray*> &values,
                 int root_rank,
                 int priority) override {
    SetKeyType(kIntKey);
    BroadcastImpl(keys, values, root_rank, priority);
  }

  void Broadcast(const std::vector<std::string> &str_keys,
                 const std::vector<NDArray*> &values,
                 int root_rank,
                 int priority) override {
    SetKeyType(kStringKey);
    std::vector<int> keys(str_keys.size());
    LookupKeys(str_keys, &keys);
    BroadcastImpl(keys, values, root_rank, priority);
  }

  void Barrier() override {
    int ret = MXBarrier();
    if (ret != 0) {
      LOG(FATAL) << "MXBarrier is not successful. ret: " << ret;
    }
  }

  int get_rank() const override {
    int ret, rank;
    ret = MXGetMpiRank(&rank);
    if (ret != 0) {
      LOG(FATAL) << "MXGetMpiRank is not successful. ret: " << ret;
      rank = -1;
    }
    return rank;
  }

  int get_group_size() const override {
    int ret, size;
    ret = MXGetMpiSize(&size);
    if (ret != 0) {
      LOG(FATAL) << "MXGetMpiSize is not successful. ret: " << ret;
      size = -1;
    }
    return size;
  }

 private:
  void InitImpl(const std::vector<int>& keys,
                const std::vector<NDArray>& values) override {
    CheckUnique(keys);
    for (size_t i = 0; i < keys.size(); ++i) {
      comm_->Init(keys[i], values[i].storage_type(), values[i].shape(), values[i].dtype());
    }
  }

  void PushPullImpl(const std::vector<int> &keys,
                    const std::vector<NDArray> &in_values,
                    const std::vector<NDArray*> &out_values,
                    int priority) {
    std::vector<int> uniq_keys;
    std::vector<std::vector<NDArray> > grouped_invals;
    std::vector<std::vector<NDArray*> > grouped_outvals;

    CHECK_EQ(in_values.size(), out_values.size());
    GroupKVPairsPush(keys, in_values, &uniq_keys, &grouped_invals);
    uniq_keys.clear();
    GroupKVPairsPull(keys, out_values, &uniq_keys, &grouped_outvals);

    for (size_t i = 0; i < uniq_keys.size(); ++i) {
      // reduce over devices
      int key = uniq_keys[i];
      const auto& invals = grouped_invals[i];
      NDArray reduced = comm_->Reduce(key, invals, priority);
      const auto storage_type = reduced.storage_type();
      auto &comm_buf = comm_buf_[key];
      if (reduced.ctx().dev_mask() == cpu::kDevMask) {
        comm_buf = reduced;  // avoid memory copy
      } else {
         if (comm_buf.is_none()) {
          if (storage_type == kDefaultStorage) {
            comm_buf = NDArray(reduced.shape(), pinned_ctx_, true, reduced.dtype());
          } else {
            comm_buf = NDArray(storage_type, reduced.shape(), pinned_ctx_, true, reduced.dtype());
          }
        }
        CopyFromTo(reduced, &comm_buf);
      }
      int ret = MXAllReduce(key, &comm_buf, &comm_buf, priority);
      if (ret != 0) {
        LOG(FATAL) << "MXAllReduce is not successful. ret:" << ret;
      }
      comm_->Broadcast(key, comm_buf, grouped_outvals[i], priority);
    }
  }

  void BroadcastImpl(const std::vector<int> &keys,
                     const std::vector<NDArray*> &values,
                     int root_rank,
                     int priority) {
    std::vector<int> uniq_keys;
    std::vector<std::vector<NDArray*> > grouped_vals;
    GroupKVPairsPull(keys, values, &uniq_keys, &grouped_vals);

    for (size_t i = 0; i < uniq_keys.size(); ++i) {
      int key = uniq_keys[i];
      auto& comm_buf = comm_buf_[key];
      const auto storage_type = grouped_vals[i][0]->storage_type();
      CHECK_EQ(storage_type, kDefaultStorage)
              << "Expected stype of value to be kDefaultStorage";
      if (comm_buf.is_none()) {
        comm_buf = NDArray(grouped_vals[i][0]->shape(), pinned_ctx_,
                          true, grouped_vals[i][0]->dtype());
      }

      if (get_rank() == 0) {
        CopyFromTo(*grouped_vals[i][0], &comm_buf);
      }
      int ret = MXBroadcast(key, &comm_buf, root_rank, priority);
      if (ret != 0) {
        LOG(FATAL) << "MXBroadcast is not successful. ret:" << ret;
      }
      comm_->Broadcast(key, comm_buf, grouped_vals[i], priority);
    }
  }

  /**
   * \brief check if the keys are all unique
   */
  void CheckUnique(const std::vector<int>& keys) {
    auto keys_copy = keys;
    auto last = std::unique(keys_copy.begin(), keys_copy.end());
    CHECK_EQ(static_cast<size_t>(std::distance(keys_copy.begin(), last)),
             static_cast<size_t>(keys.size()));
  }

 private:
  std::unordered_map<int, NDArray> comm_buf_;
};
}  // namespace kvstore
}  // namespace mxnet

#endif  // MXNET USE ALLREDUCE DIST KVSTORE
#endif  // MXNET_KVSTORE_KVSTORE_DIST_SYNC_ALLREDUCE_H_
