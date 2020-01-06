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
#include <map>

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

  /*!
   * \brief Launch operator kernels which have been pre-compiled into a lib file
   * by TVM compiler.
   * \param func_name Function name that corresponds to the operator kernel
   * \param ctx Operator context that includes device and stream information.
   * \param tblobs Tensor blobs whose dtype and shape information are extracted
   * to construct the function name. Each configuration of dtype and shape has
   * a unique kernel.
   * \param tvm_args Arguments to be passed to kernel function.
   */
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

class OtherOptionEntity {
 public:
  explicit OtherOptionEntity(int val): val_(val) {}
  OtherOptionEntity(): val_(0) {}
  inline int get_val() const {
    return val_;
  }
 private:
  int val_;
};

class OtherOptionSpace {
 public:
  explicit OtherOptionSpace(const std::vector<int>& entities) {
    int size = entities.size();
    for (int i = 0; i < size; ++i) {
      this->entities_.push_back(OtherOptionEntity(entities[i]));
    }
  }

  OtherOptionSpace() {}

  inline OtherOptionEntity &operator[] (int idx) {
    return entities_[idx];
  }

  inline const OtherOptionEntity &operator[] (int idx) const {
    return entities_[idx];
  }

  inline int size() const {
    return entities_.size();
  }

 private:
  std::vector<OtherOptionEntity> entities_;
};

class TVMOpConfig {
 public:
  std::string name;

  inline TVMOpConfig& add_space(const std::string& name, const std::vector<int>& val) {
    int size = val.size();
    space_map_[name] = OtherOptionSpace(val);
    weight_map_[name] = weight_acc_;
    weight_acc_ *= size;
    return *this;
  }
  inline TVMOpConfig& add_entity(const std::string& name, const int val) {
    entity_map_[name] = OtherOptionEntity(val);
    return *this;
  }

  TVMOpConfig(): weight_acc_(1) {}

  inline const OtherOptionSpace& get_space(const std::string& name) const {
    return space_map_.at(name);
  }

  inline int get_weight(const std::string& name) const {
    return weight_map_.at(name);
  }

 private:
  std::map<std::string, OtherOptionEntity> entity_map_;
  std::map<std::string, OtherOptionSpace> space_map_;
  std::map<std::string, int> weight_map_;
  int weight_acc_;
};

const TVMOpConfig& GetOpConfig(const std::string& name);

}  // namespace runtime
}  // namespace tvm

#endif  // MXNET_USE_TVM_OP
#endif  // MXNET_OPERATOR_TVMOP_OP_MODULE_H_
