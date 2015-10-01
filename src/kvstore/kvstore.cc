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

namespace mxnet {

KVStore* KVStore::Create(const char *type_name) {
  std::string tname = type_name;
  if (tname == "local") {
    return new kvstore::KVStoreLocal();
  } else if (tname == "device") {
    return new kvstore::KVStoreDevice();
  } else {
      LOG(FATAL) << "Unknown KVStore type \"" << type_name << "\"";
    return nullptr;
  }
}
}  // namespace mxnet
