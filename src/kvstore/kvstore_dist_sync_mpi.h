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
 * @file   kvstore_dist_sync_mpi.h
 * @brief  distributed implementation based on mpi
 */
#ifndef MXNET_KVSTORE_KVSTORE_DIST_SYNC_MPI_H_
#define MXNET_KVSTORE_KVSTORE_DIST_SYNC_MPI_H_

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

#if MXNET_USE_MPI_DIST_KVSTORE
#include "../mpi_collectives/include/mpi_collectives.h"

namespace mxnet {
namespace kvstore {

/**
 * \brief store data in local machine
 */
class KVStoreDistSyncMPI : public KVStore {
 public:
  KVStoreDistSyncMPI() : KVStore() {
    int ret = MXMPIInit();
    if (ret != 0) {
      LOG(FATAL) << "kvstore with type [" << type_ << "] failed with mpi init";
    }
  }

  virtual ~KVStoreDistSyncMPI() {
  }

  void Init(const std::vector<int>& keys,
            const std::vector<NDArray>& values) override {
    // Init does nothing in kvstore with type dist_sync_mpi
  }

  void Init(const std::vector<std::string>& str_keys,
            const std::vector<NDArray>& values) override {
    // Init does nothing in kvstore with type dist_sync_mpi
  }

  void Push(const std::vector<int>& keys,
            const std::vector<NDArray>& values,
            int priority) override {
    LOG(WARNING) << "Not supported in KVStore with type " << type_ << ".";
  }

  void Pull(const std::vector<int>& keys,
            const std::vector<NDArray*>& values,
            int priority) override {
    LOG(WARNING) << "Not supported in KVStore with type " << type_ << ".";
  }

  void PullRowSparse(const std::vector<int>& keys,
                     const std::vector<std::pair<NDArray*, NDArray>>& val_rowids,
                     int priority = 0) override {
    LOG(WARNING) << "Not supported in KVStore with type " << type_ << ".";
  }

  void Push(const std::vector<std::string>& str_keys,
            const std::vector<NDArray>& values,
            int priority) override {
    LOG(WARNING) << "Not supported in KVStore with type " << type_ << ".";
  }

  void Pull(const std::vector<std::string>& str_keys,
            const std::vector<NDArray*>& values,
            int priority) override {
    LOG(WARNING) << "Not supported in KVStore with type " << type_ << ".";
  }

  void PullRowSparse(const std::vector<std::string>& str_keys,
                     const std::vector<std::pair<NDArray*, NDArray>>& val_rowids,
                     int priority = 0) override {
    LOG(WARNING) << "Not supported in KVStore with type " << type_ << ".";
  }

  void SetGradientCompression(const std::vector<std::pair<std::string, std::string> >
                              & kwargs) override {
    LOG(WARNING) << "Not supported in KVStore with type " << type_ << ".";
  }

  void PushPull(const std::vector<int> &keys,
                const std::vector<NDArray*> &in_values,
                const std::vector<NDArray*> &out_values,
                int priority) override {
    int ret = MXMPIAllReduce(keys, in_values, out_values, priority);
    if (ret != 0) {
      LOG(WARNING) << "MXMPIAllReduce is not successful. ret: " << ret;
    }
  }

  void PushPull(const std::vector<std::string> &str_keys,
                const std::vector<NDArray*> &in_values,
                const std::vector<NDArray*> &out_values,
                int priority) override {
    int ret = MXMPIAllReduceEx(str_keys, in_values, out_values, priority);
    if (ret != 0) {
      LOG(WARNING) << "MXMPIAllReduceEx is not successful. ret: " << ret;
    }
  }

  void Broadcast(const std::vector<int> &keys,
                 const std::vector<NDArray*> &values,
                 int root_rank,
                 int priority) override {
    int ret = MXMPIBroadcast(keys, values, root_rank, priority);
    if (ret != 0) {
      LOG(WARNING) << "MXMPIBroadCast is not successful. ret: " << ret;
    }
  }

  void Broadcast(const std::vector<std::string> &str_keys,
                 const std::vector<NDArray*> &values,
                 int root_rank,
                 int priority) override {
    int ret = MXMPIBroadcastEx(str_keys, values, root_rank, priority);
    if (ret != 0) {
      LOG(WARNING) << "MXMPIBroadCastEx is not successful. ret: " << ret;
    }
  }

  int get_rank() const override {
    int ret, rank;
    ret = MXMPIGetMpiRank(&rank);
    if (ret != 0) {
      LOG(WARNING) << "MXMPIGetMpiRank is not successful. ret: " << ret;
      rank = -1;
    }
    return rank;
  }

  int get_group_size() const override {
    int ret, size;
    ret = MXMPIGetMpiSize(&size);
    if (ret != 0) {
      LOG(WARNING) << "MXMPIGetMpiSize is not successful. ret: " << ret;
      size = -1;
    }
    return size;
  }
};
}  // namespace kvstore
}  // namespace mxnet

#endif  // MXNET USE MPI DIST KVSTORE
#endif  // MXNET_KVSTORE_KVSTORE_DIST_SYNC_MPI_H_
