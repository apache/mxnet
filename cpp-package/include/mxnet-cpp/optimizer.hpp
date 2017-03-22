/*!
*  Copyright (c) 2016 by Contributors
* \file optimizer.hpp
* \brief implementation of optimizer
* \author Chuntao Hong, Zhang Chen
*/

#ifndef MXNETCPP_OPTIMIZER_HPP
#define MXNETCPP_OPTIMIZER_HPP

#include <algorithm>
#include <utility>
#include <numeric>
#include <map>
#include <string>
#include <vector>
#include "mxnet-cpp/optimizer.h"
#include "mxnet-cpp/op.h"
#include "mxnet-cpp/op_map.h"

namespace mxnet {
namespace cpp {

OpMap* Optimizer::op_map_ = new OpMap();

std::map<std::string, OptimizerCreator> OptimizerRegistry::cmap_;

MXNETCPP_REGISTER_OPTIMIZER(sgd, SGDOptimizer);
MXNETCPP_REGISTER_OPTIMIZER(ccsgd, SGDOptimizer);  // For backward compatibility

Optimizer::~Optimizer() {}

void Optimizer::Update(int index, NDArray weight, NDArray grad, mx_float lr,
                       mx_float wd) {
  params_["lr"] = std::to_string(lr);
  params_["wd"] = std::to_string(wd);
  Update(index, weight, grad);
}

std::string Optimizer::Serialize() const {
  using ValueType = std::map<std::string, std::string>::value_type;
  auto params = params_;
  params.emplace("opt_type", GetType());
  return std::accumulate(params.cbegin(), params.cend(), std::string(""),
    [](const std::string& sum, const ValueType& i) {
      return sum + '\n' + i.first + '=' + i.second;
    }).substr(1);
}

const std::vector<const char*> Optimizer::GetParamKeys_() const {
  std::vector<const char*> keys;
  for (auto& iter : params_)
    keys.push_back(iter.first.c_str());
  return keys;
}

const std::vector<const char*> Optimizer::GetParamValues_() const {
  std::vector<const char*> values;
  for (auto& iter : params_)
    values.push_back(iter.second.c_str());
  return values;
}

Optimizer* OptimizerRegistry::Find(const std::string& name) {
  auto it = cmap_.find(name);
  if (it == cmap_.end())
    return nullptr;
  return it->second();
}

int OptimizerRegistry::__REGISTER__(const std::string& name, OptimizerCreator creator) {
  CHECK_EQ(cmap_.count(name), 0) << name << " already registered";
  cmap_.emplace(name, std::move(creator));
  return 0;
}

std::string SGDOptimizer::GetType() const {
  return "sgd";
}

SGDOptimizer::SGDOptimizer() {
  update_handle_ = op_map_->GetSymbolCreator("sgd_update");
  mom_update_handle_ = op_map_->GetSymbolCreator("sgd_mom_update");
}

SGDOptimizer::~SGDOptimizer() {
  for (auto &it : states_) {
    delete it.second;
  }
}

void SGDOptimizer::Update(int index, NDArray weight, NDArray grad) {
  if (states_.count(index) == 0) {
    CreateState_(index, weight);
  }

  auto keys = GetParamKeys_();
  auto values = GetParamValues_();
  CHECK_EQ(keys.size(), values.size());

  NDArrayHandle inputs[3];
  inputs[0] = weight.GetHandle();
  inputs[1] = grad.GetHandle();

  int num_outputs = 1;
  NDArrayHandle output = weight.GetHandle();
  NDArrayHandle *outputs = &output;

  if (states_[index] == nullptr) {
    MXImperativeInvoke(update_handle_, 2, inputs,
        &num_outputs, &outputs,
        keys.size(), keys.data(), values.data());
  } else {
    inputs[2] = states_[index]->GetHandle();
    MXImperativeInvoke(mom_update_handle_, 3, inputs,
        &num_outputs, &outputs,
        keys.size(), keys.data(), values.data());
  }
}

void SGDOptimizer::CreateState_(int index, NDArray weight) {
  if (params_.count("momentum") == 0) {
    states_[index] = nullptr;
  } else {
    states_[index] = new NDArray(weight.GetShape(), weight.GetContext());
    *states_[index] = 0;
  }
}


}  // namespace cpp
}  // namespace mxnet

#endif  // MXNETCPP_OPTIMIZER_HPP
