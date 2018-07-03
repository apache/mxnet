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
* \file symbol.h
* \brief definition of symbol
* \author Chuntao Hong, Zhang Chen
*/

#ifndef MXNET_LITE_SYMBOL_BLOCK_H_
#define MXNET_LITE_SYMBOL_BLOCK_H_

#include <map>
#include <unordered_map>
#include <string>
#include <vector>

#include <mxnet-lite/base.h>
#include <mxnet-lite/ndarray.h>
#include <mxnet-lite/symbol.h>

namespace mxnet {
namespace lite {

class SymbolBlock {
 public:
  SymbolBlock(const Context& context,
              const std::vector<std::pair<std::string, std::string> >& flags,
              const Symbol& output,
              const std::vector<std::string>& data_names,
              const std::map<std::string, NDArray>& params) {
    auto input_names = output.ListInputs();
    num_inputs_ = input_names.size();
    std::unordered_map<std::string, size_t> name_to_idx;
    for (size_t i = 0; i < input_names.size(); ++i) {
      name_to_idx[input_names[i]] = i;
    }
    for (const auto& i : data_names) {
      const auto iter = name_to_idx.find(i);
      CHECK(iter != name_to_idx.end()) << "Invalid input name " << i;
      data_indices_.push_back(iter->second);
    }
    for (const auto& i : params) {
      std::string name = i.first;
      if (name.compare(0, 4, "arg:") || name.compare(0, 4, "aux:")) {
        name = name.substr(4);
      }
      const auto iter = name_to_idx.find(name);
      CHECK(iter != name_to_idx.end()) << "Invalid parameter name " << i.first;
      param_indices_.push_back(iter->second);
      if (i.second.GetContext() != context) {
        params_.push_back(i.second.Copy(context));
      } else {
        params_.push_back(i.second);
      }
    }
    std::vector<const char*> keys, vals;
    for (const auto& i : flags) {
      keys.push_back(i.first.c_str());
      vals.push_back(i.second.c_str());
    }
    std::stringstream stream_data_indices, stream_param_indices;
    stream_data_indices << "[";
    for (const auto& i : data_indices_) stream_data_indices << i << ", ";
    stream_data_indices << "]";
    stream_param_indices << "[";
    for (const auto& i : param_indices_) stream_param_indices << i << ", ";
    stream_param_indices << "]";
    std::string str_data_indices = stream_data_indices.str();
    std::string str_param_indices = stream_param_indices.str();
    keys.push_back("data_indices");
    vals.push_back(str_data_indices.c_str());
    keys.push_back("param_indices");
    vals.push_back(str_param_indices.c_str());

    CachedOpHandle handle;
    MXNET_CALL(MXCreateCachedOpEx(
        output.GetHandle(), keys.size(), keys.data(), vals.data(), &handle));
    blob_ptr_ = std::make_shared<OpBlob>(handle);
  }

  std::vector<NDArray> Forward(const std::vector<NDArray>& data) {
    std::vector<NDArrayHandle> inputs(num_inputs_);
    CHECK_EQ(data.size(), data_indices_.size());
    for (size_t i = 0; i < data.size(); ++i) {
      inputs[data_indices_[i]] = data[i].GetHandle();
    }
    for (size_t i = 0; i < params_.size(); ++i) {
      inputs[param_indices_[i]] = params_[i].GetHandle();
    }
    int num_outputs = 0;
    NDArrayHandle *outputs = nullptr;
    MXNET_CALL(MXInvokeCachedOp(
        blob_ptr_->handle, num_inputs_, inputs.data(), &num_outputs,
        &outputs));
    std::vector<NDArray> ret;
    for (int i = 0; i < num_outputs; ++i) {
      ret.push_back(NDArray(outputs[i]));
    }
    return ret;
  }

 private:
  /*!
  * \brief struct to store CachedOpHandle
  */
  struct OpBlob {
    /*!
    * \brief default constructor
    */
    OpBlob() : handle(nullptr) {}
    /*!
    * \brief construct with CachedOpHandle to store
    */
    explicit OpBlob(CachedOpHandle handle_) : handle(handle_) {}
    /*!
    * \brief destructor, free the CachedOpHandle
    */
    ~OpBlob() { MXFreeCachedOp(handle); }
    /*!
    * \brief the CachedOpHandle to store
    */
    CachedOpHandle handle;
  };
  std::shared_ptr<OpBlob> blob_ptr_;
  size_t num_inputs_;
  std::vector<size_t> data_indices_;
  std::vector<size_t> param_indices_;
  std::vector<NDArray> params_;
};

}  // namespace lite
}  // namespace mxnet
#endif  // MXNET_LITE_SYMBOL_BLOCK_H_
