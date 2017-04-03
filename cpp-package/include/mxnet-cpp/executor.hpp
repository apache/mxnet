/*!
 *  Copyright (c) 2016 by Contributors
 * \file executor.hpp
 * \brief implementation of the executor
 * \author Zhang Chen, Chuntao Hong
 */

#ifndef CPP_PACKAGE_INCLUDE_MXNET_CPP_EXECUTOR_HPP_
#define CPP_PACKAGE_INCLUDE_MXNET_CPP_EXECUTOR_HPP_

#include <vector>
#include <map>
#include <string>
#include "mxnet-cpp/executor.h"
#include "mxnet-cpp/optimizer.h"

namespace mxnet {
namespace cpp {
inline Executor::Executor(const Symbol &symbol, Context context,
                          const std::vector<NDArray> &arg_arrays,
                          const std::vector<NDArray> &grad_arrays,
                          const std::vector<OpReqType> &grad_reqs,
                          const std::vector<NDArray> &aux_arrays,
                          const std::map<std::string, Context> &group_to_ctx,
                          Executor *shared_exec) {
  this->arg_arrays = arg_arrays;
  this->grad_arrays = grad_arrays;
  this->aux_arrays = aux_arrays;
  this->symbol_ = symbol;

  std::vector<NDArrayHandle> arg_handles;
  std::vector<NDArrayHandle> grad_handles;
  std::vector<NDArrayHandle> aux_handles;

  for (const auto &array : arg_arrays) {
    arg_handles.push_back(array.GetHandle());
  }
  for (const auto &array : grad_arrays) {
    grad_handles.push_back(array.GetHandle());
  }
  for (const auto &array : aux_arrays) {
    aux_handles.push_back(array.GetHandle());
  }

  std::vector<mx_uint> grad_reqs_uint;
  for (auto s : grad_reqs) grad_reqs_uint.push_back(s);

  std::vector<const char *> map_keys;
  std::vector<int> dev_types, dev_ids;
  for (const auto &s : group_to_ctx) {
    map_keys.push_back(s.first.c_str());
    dev_types.push_back(s.second.GetDeviceType());
    dev_ids.push_back(s.second.GetDeviceId());
  }

  ExecutorHandle *shared_exec_handle =
      shared_exec == nullptr ? nullptr : &shared_exec->handle_;

  CHECK_EQ(MXExecutorBindEX(symbol.GetHandle(), context.GetDeviceType(),
                            context.GetDeviceId(), group_to_ctx.size(),
                            map_keys.data(), dev_types.data(), dev_ids.data(),
                            arg_handles.size(), arg_handles.data(),
                            grad_handles.data(), grad_reqs_uint.data(),
                            aux_handles.size(), aux_handles.data(),
                            shared_exec_handle, &handle_),
           0);

  mx_uint out_size;
  NDArrayHandle *out_array;
  CHECK_EQ(MXExecutorOutputs(handle_, &out_size, &out_array), 0);
  for (mx_uint i = 0; i < out_size; ++i) {
    outputs.push_back(NDArray(out_array[i]));
  }
}

inline std::string Executor::DebugStr() {
  const char *output;
  MXExecutorPrint(handle_, &output);
  return std::string(output);
}

inline void Executor::UpdateAll(Optimizer *opt, float lr, float wd,
                                int arg_update_begin, int arg_update_end) {
  arg_update_end = arg_update_end < 0 ? arg_arrays.size() - 1 : arg_update_end;
  for (int i = arg_update_begin; i < arg_update_end; ++i) {
    opt->Update(i, arg_arrays[i], grad_arrays[i], lr, wd);
  }
}
}  // namespace cpp
}  // namespace mxnet

#endif  // CPP_PACKAGE_INCLUDE_MXNET_CPP_EXECUTOR_HPP_
