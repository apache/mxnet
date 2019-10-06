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
 * Copyright (c) 2019 by Contributors
 * \file op_module.h
 * \brief Invoke registered TVM operators.
 * \author Yizhi Liu
 */
#ifndef MXNET_OPERATOR_TVMOP_OP_MODULE_H_
#define MXNET_OPERATOR_TVMOP_OP_MODULE_H_

#if MXNET_USE_TVM_OP
#include <mxnet/base.h>
#include <mxnet/op_attr_types.h>
#include <mutex>
#include <string>
#include <vector>

namespace tvm {
namespace runtime {

class TVMArgs;
class Module;
class TVMOpModule {
 public:
  // Load TVM operators binary
  void Load(const std::string& filepath);

  void Call(const std::string& func_name,
            const mxnet::OpContext& ctx,
            const std::vector<mxnet::TBlob>& args) const;

  void CallEx(const std::string &func_name,
              const mxnet::OpContext& ctx,
              const std::vector<mxnet::TBlob>& tblobs,
              TVMArgs tvm_args) const;

  static TVMOpModule *Get() {
    static TVMOpModule inst;
    return &inst;
  }

 private:
  std::mutex mutex_;
  std::shared_ptr<Module> module_ptr_;
};

}  // namespace runtime
}  // namespace tvm

#endif  // MXNET_USE_TVM_OP
#endif  // MXNET_OPERATOR_TVMOP_OP_MODULE_H_
