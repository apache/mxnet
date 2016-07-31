/*!
 * Copyright (c) 2015 by Contributors
 * \file kvstore.cc
 * \brief implement kv_store
 */
#include <mxnet/kvstore.h>
#include <stdlib.h>
#include <dmlc/logging.h>
#include "./kvstore_local.h"
#include "./kvstore_device.h"
#if MXNET_USE_DIST_KVSTORE
#include "./kvstore_dist.h"
#endif  // MXNET_USE_DIST_KVSTORE

namespace mxnet {

KVStore* KVStore::Create(const char *type_name) {
  std::string tname = type_name;
  std::transform(tname.begin(), tname.end(), tname.begin(), ::tolower);
  KVStore* kv = nullptr;
  if (tname == "local" ||
      tname == "local_update_cpu" ||
      tname == "local_allreduce_cpu") {
    kv =  new kvstore::KVStoreLocal();
  } else if (tname == "device" ||
             tname == "local_update_device" ||
             tname == "local_allreduce_device") {
    kv = new kvstore::KVStoreDevice(true);
  } else if (tname == "dist_async" ||
             tname == "dist_sync" ||
             tname == "dist_sync_device" ||
             tname == "dist") {
#if MXNET_USE_DIST_KVSTORE
    kv = new kvstore::KVStoreDist(
        tname.find("device") != std::string::npos);
    if (tname == "dist_sync" &&
        kv->IsWorkerNode() &&
        kv->get_rank() == 0) {
      // configure the server to be the sync mode
      kv->SendCommandToServers(kvstore::kSyncMode, "");
    }
#else
    LOG(FATAL) << "compile with USE_DIST_KVSTORE=1 to use " << tname;
    return nullptr;
#endif  // MXNET_USE_DIST_KVSTORE
  } else {
    LOG(FATAL) << "Unknown KVStore type \"" << tname << "\"";
  }
  kv->type_ = tname;
  return kv;
}

}  // namespace mxnet
