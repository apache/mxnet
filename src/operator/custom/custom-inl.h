/*!
 * Copyright (c) 2015 by Contributors
 * \file native_op-inl.h
 * \brief
 * \author Junyuan Xie
*/

#ifndef MXNET_OPERATOR_CUSTOM_CUSTOM_INL_H_
#define MXNET_OPERATOR_CUSTOM_CUSTOM_INL_H_
#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <mxnet/operator.h>
#include <mxnet/c_api.h>
#include <map>
#include <vector>
#include <string>
#include <utility>
#include <sstream>
#include <thread>
#include <mutex>
#include <functional>
#include <condition_variable>
#include <queue>
#include "../operator_common.h"

namespace mxnet {
namespace op {
namespace custom {

class Registry {
 public:
  void Register(const std::string &op_type, CustomOpPropCreator creator) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (registry_.find(op_type) != registry_.end()) {
      LOG(WARNING) << "New registration is overriding existing custom operator " << op_type;
    }
    registry_[op_type] = creator;
  }

  CustomOpPropCreator Find(const std::string &op_type) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = registry_.find(op_type);
    if (it != registry_.end()) return it->second;
    return nullptr;
  }

  static Registry* Get();
 private:
  Registry() {}
  std::mutex mutex_;
  std::map<std::string, CustomOpPropCreator> registry_;
};

}  // namespace custom
}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_CUSTOM_CUSTOM_INL_H_
