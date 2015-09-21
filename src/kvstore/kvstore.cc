/*!
 * Copyright (c) 2015 by Contributors
 * \file kvstore.cc
 * \brief implement kv_store
 */
#include <mxnet/kvstore.h>
#include <stdlib.h>
#include <dmlc/logging.h>
#include "./kvstore_local.h"

namespace mxnet {

KVStore* KVStore::Create(const char *type_name) {
  std::string tname = type_name;
  CHECK_EQ(tname, "local")
      << "Only support local KVStore currently.";
  return new KVStoreLocal();
}
}  // namespace mxnet
