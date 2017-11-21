/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
*  Copyright (c) 2016 by Contributors
* \file executor.h
* \brief executor definition
* \author Chuntao Hong, Zhang Chen
*/

#ifndef MXNET_CPP_EXECUTOR_H_
#define MXNET_CPP_EXECUTOR_H_

#include <vector>
#include <map>
#include <set>
#include <string>
#include "mxnet-cpp/base.h"
#include "mxnet-cpp/symbol.h"

namespace mxnet {
namespace cpp {

class Optimizer;

/*!
* \brief Executor interface
*/
class Executor {
  friend class Monitor;
 public:
  Executor(const Symbol &symbol, Context context,
           const std::vector<NDArray> &arg_arrays,
           const std::vector<NDArray> &grad_arrays,
           const std::vector<OpReqType> &grad_reqs,
           const std::vector<NDArray> &aux_arrays,
           const std::map<std::string, Context> &group_to_ctx =
               std::map<std::string, Context>(),
           Executor *shared_exec = nullptr);
  explicit Executor(const ExecutorHandle &h) { handle_ = h; }
  /*!
  * \brief Perform a Forward operation of Operator
  *  After this operation, user can get the result by using function head.
  */
  void Forward(bool is_train) {
    MXExecutorForward(handle_, is_train ? 1 : 0);
    mx_uint out_size;
    NDArrayHandle *out_array;
    CHECK_EQ(MXExecutorOutputs(handle_, &out_size, &out_array), 0);
    for (mx_uint i = 0; i < out_size; ++i) {
      outputs[i] = NDArray(out_array[i]);
    }
  }
  /*!
  * \brief Perform a Backward operation of the Operator.
  *  This must be called after Forward.
  *  After this operation, NDArrays specified by grad_in_args_store will be
  *updated accordingly.
  *  User is allowed to pass in an empty Array if the head node is
  *  loss function and head gradeitn is not needed.
  *
  * \param head_grads the gradient of head nodes to be backproped.
  */
  void Backward(const std::vector<NDArray> &head_grads =
                    std::vector<NDArray>()) {
    std::vector<NDArrayHandle> head_grads_;
    for (auto d : head_grads) {
      head_grads_.push_back(d.GetHandle());
    }
    if (head_grads_.size() > 0) {
      MXExecutorBackward(handle_, head_grads_.size(), head_grads_.data());
    } else {
      MXExecutorBackward(handle_, 0, nullptr);
    }
  }
  // TODO(zhangchen-qinyinghua)
  // To implement reshape function
  void Reshape();
  /*!
  * \brief update the arguments with given learning rate and optimizer
  * \return the SymbolHandle
  */
  std::string DebugStr();
  /*!
  * \brief destructor, free the handle
  */
  ~Executor() { MXExecutorFree(handle_); }
  std::vector<NDArray> arg_arrays;
  std::vector<NDArray> grad_arrays;
  std::vector<NDArray> aux_arrays;
  /*!
  * \brief arrays store the outputs of forward
  */
  std::vector<NDArray> outputs;
  std::map<std::string, NDArray> arg_dict() {
    return GetDict(symbol_.ListArguments(), arg_arrays);
  }
  std::map<std::string, NDArray> grad_dict() {
    return GetDict(symbol_.ListArguments(), grad_arrays);
  }
  std::map<std::string, NDArray> aux_dict() {
    return GetDict(symbol_.ListAuxiliaryStates(), aux_arrays);
  }

 private:
  Executor(const Executor &e);
  Executor &operator=(const Executor &e);
  ExecutorHandle handle_;
  Symbol symbol_;
  std::map<std::string, NDArray> GetDict(const std::vector<std::string> &names,
                                         const std::vector<NDArray> &arrays) {
    std::map<std::string, NDArray> ret;
    std::set<std::string> name_set;
    for (const auto &s : names) {
      CHECK(name_set.find(s) == name_set.end()) << "Duplicate names detected, "
                                                << s;
      name_set.insert(s);
    }
    CHECK_EQ(name_set.size(), arrays.size())
        << "names size not equal to arrays size";
    for (size_t i = 0; i < names.size(); ++i) {
      ret[names[i]] = arrays[i];
    }
    return ret;
  }
};
}  // namespace cpp
}  // namespace mxnet
#endif  // MXNET_CPP_EXECUTOR_H_
