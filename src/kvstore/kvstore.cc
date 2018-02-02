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

/*!
 * Copyright (c) 2015 by Contributors
 * \file kvstore.cc
 * \brief implement kv_store
 */
#include <mxnet/kvstore.h>
#include <stdlib.h>
#include <dmlc/logging.h>
#include "./kvstore_local.h"

#if MXNET_USE_DIST_KVSTORE
#include "./kvstore_dist.h"
std::atomic<int> mxnet::kvstore::KVStoreDist::customer_id_{0};
#endif  // MXNET_USE_DIST_KVSTORE
#if MXNET_USE_NCCL
#include "./kvstore_nccl.h"
#endif  // MXNET_USE_NCCL

namespace mxnet {

KVStore* KVStore::Create(const char *type_name) {
  std::string tname = type_name;
  std::transform(tname.begin(), tname.end(), tname.begin(), ::tolower);
  KVStore* kv = nullptr;
  bool use_device_comm = false;
  auto has = [tname](const std::string& pattern) {
    return tname.find(pattern) != std::string::npos;
  };
  if (has("device")) {
    use_device_comm = true;
  }

  if (has("dist")) {
#if MXNET_USE_DIST_KVSTORE
    kv = new kvstore::KVStoreDist(use_device_comm);
    if (!has("_async") && kv->IsWorkerNode() && kv->get_rank() == 0) {
      // configure the server to be the sync mode
      kv->SendCommandToServers(static_cast<int>(kvstore::CommandType::kSyncMode), "");
    }
#else
    LOG(FATAL) << "compile with USE_DIST_KVSTORE=1 to use " << tname;
    return nullptr;
#endif  // MXNET_USE_DIST_KVSTORE
  } else {
    if (has("nccl")) {
#if MXNET_USE_NCCL
      kv = new kvstore::KVStoreNCCL();
#else
      LOG(FATAL) << "compile with USE_NCCL=1 to use " << tname;
      return nullptr;
#endif
    } else {
      kv =  new kvstore::KVStoreLocal(use_device_comm);
    }
  }
  kv->type_ = tname;
  return kv;
}

}  // namespace mxnet
