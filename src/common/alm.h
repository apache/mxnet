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
 * \file alm.h
 * \brief Automatic Layout Manager
 * \author Dawid Tracz
 */

#ifndef MXNET_COMMON_ALM_H_
#define MXNET_COMMON_ALM_H_

#include <mxnet/base.h>
#include <nnvm/graph.h>
#include <nnvm/node.h>
#include <functional>
#include <string>
#include <unordered_map>
#include <vector>

namespace mxnet {
namespace alm {

/*!
 *  \brief Class connects MXInitializeALM function from c_api and ALM.
 *         It provides simple interface for adding new targets
 *         and stores this data to be used later, while running ALM.
 */
class ALMParams {
 public:
  using ALMTargetsT = std::unordered_map<std::string, std::string>;

  static ALMParams& getALMParams() {
    static ALMParams alm;
    return alm;
  }

  bool empty() const {
    return targets_.empty();
  }

  void addTarget(std::string opname, std::string targetLayout) {
    targets_[opname] = targetLayout;
  }

  int setTargets(ALMTargetsT new_targets) {
    targets_.clear();
    targets_ = new_targets;
    return targets_.size();
  }

  const ALMTargetsT& getTargets() const {
    return targets_;
  }

  ALMParams(ALMParams&) = delete;
  ALMParams operator=(ALMParams&) = delete;

 private:
  ALMTargetsT targets_;

  ALMParams() { }
};  // class ALM_params

/*!
 * \bried Top-level function to run layout optimization.
 */
nnvm::Graph OptimizeLayout(nnvm::Graph&& g,
                           const std::unordered_map<std::string, std::string>& layout_targets);

/*!
 * \brief Transpose, represented by permutation of axes.
 */
using Transpose = std::vector<size_t>;

bool IsIdentity(const Transpose& t);
Transpose Reverse(const Transpose& axes);

/*!
 * \bried Compose 2 transposes. Not commutative: a * b means b is applied first, then a.
 */
Transpose Compose(const Transpose& lhs, const Transpose& rhs);

mshadow::LayoutFlag ApplyTranspose(mshadow::LayoutFlag layout, const Transpose& axes);
std::string ApplyTranspose(const std::string& layout, const Transpose& axes);

Transpose FromTShape(const mxnet::TShape& s);

/*!
 * \brief May change operator's layout. Used in LayoutOptimization.
 *
 * \param target_layout The target layout to change to, or kUNKNOWN. In the latter case the target
 * layout is calculated based on in_axes, with a goal to cancel them out (at least some, ideally -
 * all).
 * \param in_axes (in/out) On input - pending inputs' transposes. On output - inputs' transposes,
 * required by the new layout.
 * \param out_axes (out) Outputs' transposes, required to convert to the original layouts.
 * \return true if attrs changed and params need to be reparsed.
 */
using FChangeLayout =
    std::function<bool(nnvm::NodeAttrs*, mshadow::LayoutFlag target_layout,
                       std::vector<Transpose>* in_axes, std::vector<Transpose>* out_axes)>;

/*!
 * \brief Factors out and returns a common transpose, or default-constructed Transpose if all
 * axes (in/out parameter) are empty.
 */
Transpose FactorCommonTranspose(std::vector<Transpose>* axes);

}  // namespace alm
}  // namespace mxnet

#endif  // MXNET_COMMON_ALM_H_
