/*!
*  Copyright (c) 2016 by Contributors
* \file kvstore.h
* \brief definition of kvstore
* \author Chuntao Hong
*/

#ifndef CPP_PACKAGE_INCLUDE_MXNET_CPP_KVSTORE_H_
#define CPP_PACKAGE_INCLUDE_MXNET_CPP_KVSTORE_H_

#include <string>
#include <vector>
#include "mxnet-cpp/ndarray.h"

namespace mxnet {
namespace cpp {

class KVStore {
 public:
  static void SetType(const std::string& type);
  static void RunServer();
  static void Init(int key, const NDArray& val);
  static void Init(const std::vector<int>& keys, const std::vector<NDArray>& vals);
  static void Push(int key, const NDArray& val, int priority = 0);
  static void Push(const std::vector<int>& keys,
      const std::vector<NDArray>& vals, int priority = 0);
  static void Pull(int key, NDArray* out, int priority = 0);
  static void Pull(const std::vector<int>& keys, std::vector<NDArray>* outs, int priority = 0);
  // TODO(lx): put lr in optimizer or not?
  static void SetOptimizer(std::unique_ptr<Optimizer> optimizer, bool local = false);
  static std::string GetType();
  static int GetRank();
  static int GetNumWorkers();
  static void Barrier();
  static std::string GetRole();

 private:
  KVStore();
  static KVStoreHandle& get_handle();
  static std::unique_ptr<Optimizer>& get_optimizer();
  static KVStore*& get_kvstore();
  static void Controller(int head, const char* body, void* controller_handle);
  static void Updater(int key, NDArrayHandle recv, NDArrayHandle local, void* handle_);
};

}  // namespace cpp
}  // namespace mxnet

#endif  // CPP_PACKAGE_INCLUDE_MXNET_CPP_KVSTORE_H_
