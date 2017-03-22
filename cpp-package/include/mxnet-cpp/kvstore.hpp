/*!
 *  Copyright (c) 2016 by Contributors
 * \file kvstore.hpp
 * \brief implementation of kvstore
 * \author Xin Li
 */

#include <algorithm>
#include <map>
#include <numeric>
#include <string>
#include <vector>

#include "mxnet-cpp/kvstore.h"
#include "mxnet-cpp/optimizer.h"

#ifndef KVSTORE_HPP
#define KVSTORE_HPP

namespace mxnet {
namespace cpp {

namespace private_ {
  KVStore *kvstore = nullptr;

  extern "C"
  void controller(int head, const char* body, void * controller_handle) {
    if (kvstore == nullptr) {
      return;
    }
    if (head == 0) {
      std::map<std::string, std::string> params;
      std::istringstream sin(body);
      std::string line;
      while (getline(sin, line)) {
        size_t n = line.find('=');
        params.emplace(line.substr(0, n), line.substr(n+1));
      }
      std::unique_ptr<Optimizer> opt(OptimizerRegistry::Find(params.at("opt_type")));
      params.erase("opt_type");
      for (const auto& pair : params) {
        opt->SetParam(pair.first, pair.second);
      }
      kvstore->SetOptimizer(std::move(opt), true);
    }
  }
}  // namespace private_

KVStore::KVStore(const std::string& name) {
  CHECK_EQ(MXKVStoreCreate(name.c_str(), &handle_), 0);
}

KVStore::KVStore(KVStore &&kv) {
  optimizer_ = std::move(kv.optimizer_);
  handle_ = kv.handle_;
  kv.handle_ = nullptr;
}

void KVStore::RunServer() {
  CHECK_NE(GetRole(), "worker");
  private_::kvstore = this;
  CHECK_EQ(MXKVStoreRunServer(handle_, &private_::controller, 0), 0);
}

void KVStore::Init(int key, const NDArray& val) {
  NDArrayHandle val_handle = val.GetHandle();
  CHECK_EQ(MXKVStoreInit(handle_, 1, &key, &val_handle), 0);
}

void KVStore::Init(const std::vector<int>& keys, const std::vector<NDArray>& vals) {
  CHECK_EQ(keys.size(), vals.size());
  std::vector<NDArrayHandle> val_handles(vals.size());
  std::transform(vals.cbegin(), vals.cend(), val_handles.begin(),
      [](const NDArray& val) {
        return val.GetHandle();
      });

  CHECK_EQ(MXKVStoreInit(handle_, keys.size(), keys.data(),
      val_handles.data()), 0);
}

void KVStore::Push(int key, const NDArray& val, int priority) {
  NDArrayHandle val_handle = val.GetHandle();
  CHECK_EQ(MXKVStorePush(handle_, 1, &key, &val_handle, priority), 0);
}

void KVStore::Push(const std::vector<int>& keys,
                   const std::vector<NDArray>& vals,
                   int priority) {
  CHECK_EQ(keys.size(), vals.size());
  std::vector<NDArrayHandle> val_handles(vals.size());
  std::transform(vals.cbegin(), vals.cend(), val_handles.begin(),
      [](const NDArray& val) {
        return val.GetHandle();
      });

  CHECK_EQ(MXKVStorePush(handle_, keys.size(), keys.data(),
      val_handles.data(), priority), 0);
}

void KVStore::Pull(int key, NDArray* out, int priority) {
  NDArrayHandle out_handle = out->GetHandle();
  CHECK_EQ(MXKVStorePull(handle_, 1, &key, &out_handle, priority), 0);
}

void KVStore::Pull(const std::vector<int>& keys, std::vector<NDArray>* outs, int priority) {
  CHECK_EQ(keys.size(), outs->size());

  std::vector<NDArrayHandle> out_handles(keys.size());
  std::transform(outs->cbegin(), outs->cend(), out_handles.begin(),
      [](const NDArray& val) {
        return val.GetHandle();
      });

  CHECK_EQ(MXKVStorePull(handle_, keys.size(), keys.data(),
      out_handles.data(), priority), 0);
}

namespace private_ {
  extern "C"
  void updater(int key, NDArrayHandle recv, NDArrayHandle local,
      void* handle_) {
    Optimizer *opt = static_cast<Optimizer*>(handle_);
    opt->Update(key, NDArray(local), NDArray(recv));
  }
}

void KVStore::SetOptimizer(std::unique_ptr<Optimizer> optimizer, bool local) {
  if (local) {
    optimizer_ = std::move(optimizer);
    CHECK_EQ(MXKVStoreSetUpdater(handle_, &private_::updater, optimizer_.get()), 0);
  } else {
    CHECK_EQ(MXKVStoreSendCommmandToServers(handle_, 0, (*optimizer).Serialize().c_str()), 0);
  }
}

std::string KVStore::GetType() const {
  const char *type;
  CHECK_EQ(MXKVStoreGetType(handle_, &type), 0);
  // type is managed by handle_, no need to free its memory.
  return type;
}

int KVStore::GetRank() const {
  int rank;
  CHECK_EQ(MXKVStoreGetRank(handle_, &rank), 0);
  return rank;
}

int KVStore::GetNumWorkers() const {
  int num_workers;
  CHECK_EQ(MXKVStoreGetGroupSize(handle_, &num_workers), 0);
  return num_workers;
}

void KVStore::Barrier() const {
  CHECK_EQ(MXKVStoreBarrier(handle_), 0);
}

std::string KVStore::GetRole() const {
  int ret;
  CHECK_EQ(MXKVStoreIsSchedulerNode(&ret), 0);
  if (ret) {
    return "scheduler";
  }
  CHECK_EQ(MXKVStoreIsServerNode(&ret), 0);
  if (ret) {
    return "server";
  }
  CHECK_EQ(MXKVStoreIsWorkerNode(&ret), 0);
  CHECK(ret);
  return "worker";
}

}  // namespace cpp
}  // namespace mxnet

#endif  // KVSTORE_HPP
