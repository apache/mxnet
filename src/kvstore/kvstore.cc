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
#endif  // MXNET_USE_DIST_KVSTORE

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
      kv->SendCommandToServers(kvstore::kSyncMode, "");
    }
#else
    LOG(FATAL) << "compile with USE_DIST_KVSTORE=1 to use " << tname;
    return nullptr;
#endif  // MXNET_USE_DIST_KVSTORE
  } else {
    kv =  new kvstore::KVStoreLocal(use_device_comm);
  }
  kv->type_ = tname;
  return kv;
}

}  // namespace mxnet
