/*!
 * Copyright (c) 2015 by Contributors
 * \file kvstore.cc
 * \brief implement kv_store
 */
#include "mxnet/kvstore.h"
#include "kvstore_base.h"
#include <stdlib.h>
#include "dmlc/logging.h"

namespace mxnet {

void KVStore::InitDevices(const std::vector<Context>& devices) {
  char* num_worker = getenv("DMLC_NUM_WORKER");
  if (num_worker == NULL || atoi(num_worker) == 1) {
    // local model
    store_ = new KVStoreBase();
  } else {
    LOG(FATAL) << "not implemented yet";
  }
  store_->InitDevices(devices);
}

void KVStore::Init(int key, const NArray& value) {
  CHECK(store_ != NULL) << "call InitDevices first";
  store_->Push(key, value, true);
}

void KVStore::Push(int key, const NArray& value) {
  CHECK(store_ != NULL) << "call InitDevices first";
  store_->Push(key, value, false);
}

void KVStore::Pull(int key, NArray* value) {
  CHECK(store_ != NULL) << "call InitDevices first";
  store_->Pull(key, value);
}

void KVStore::Clear() {
  if (store_) store_->Clear();
}

int KVStore::GetRank() { return store_->GetRank(); }
int KVStore::GetGroupSize() { return store_->GetGroupSize(); }

void KVStore::SetUpdater(const Updater& updt) {
  CHECK(store_ != NULL) << "call InitDevices first";
  store_->SetUpdater(updt);
}

void KVStore::SetAggregator(bool aggregator) {
  CHECK(store_ != NULL) << "call InitDevices first";
  store_->SetAggregator(aggregator);
}

KVStore::~KVStore() { delete store_; }

}  // namespace mxnet
