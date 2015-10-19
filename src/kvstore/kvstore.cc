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
#include "./mxnet_ps_node.h"
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
             tname == "local_allreduce_device") {
    tname = "local_allreduce_device";
    kv = new kvstore::KVStoreDevice();
  } else if (tname == "dist_async" ||
             tname == "dist_sync" ||
             tname == "dist") {
#if MXNET_USE_DIST_KVSTORE
    kv = new kvstore::KVStoreDist();
    if (tname == "dist_sync" &&
        kv->IsWorkerNode() &&
        kv->get_rank() == 0) {
      // configure the server to be the sync mode
      kv->SendCommandToServers(kvstore::CommandID::kSyncMode, "");
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

#if MXNET_USE_DIST_KVSTORE

namespace ps {

App* App::Create(int argc, char *argv[]) {
  NodeInfo n;
  if (n.IsWorker()) {
    return new ::mxnet::kvstore::MXNetWorker();
  } else if (n.IsServer()) {
    return new ::mxnet::kvstore::MXNetServer();
  } else if (n.IsScheduler()) {
    return new ::mxnet::kvstore::MXNetScheduler();
  } else {
    LOG(FATAL) << "unknown node";
  }
  return NULL;
}

}  // namespace ps

#endif  // MXNET_USE_DIST_KVSTORE
