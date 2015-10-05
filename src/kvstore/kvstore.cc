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
#include "./mxnet_node.h"
#endif  // MXNET_USE_DIST_KVSTORE

namespace mxnet {

KVStore* KVStore::Create(const char *type_name) {
  std::string tname = type_name;
  if (tname == "local") {
    return new kvstore::KVStoreLocal();
  } else if (tname == "device") {
    return new kvstore::KVStoreDevice();
  } else if (tname == "dist") {
#if MXNET_USE_DIST_KVSTORE
    return new kvstore::KVStoreDist();
#else
    LOG(FATAL) << "compile with USE_DIST_KVSTORE=1";
    return nullptr;
#endif // MXNET_USE_DIST_KVSTORE
  }
  LOG(FATAL) << "Unknown KVStore type \"" << type_name << "\"";
  return nullptr;
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
