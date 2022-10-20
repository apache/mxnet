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

#ifndef MXNET_OPERATOR_SUBGRAPH_DNNL_DNNL_TRANSFORMER_QK_COMMON_H_
#define MXNET_OPERATOR_SUBGRAPH_DNNL_DNNL_TRANSFORMER_QK_COMMON_H_

#if MXNET_USE_ONEDNN == 1

#include <string>
#include <vector>

#include "operator/contrib/transformer-inl.h"
#include "operator/numpy/np_matrix_op-inl.h"
#include "operator/tensor/matrix_op-inl.h"
#include "operator/subgraph/common.h"
#include "dnnl_common.h"
#include "dnnl_subgraph_base-inl.h"
#include "dnnl_transformer-inl.h"

namespace mxnet {
namespace op {
namespace qk_common {

enum SelectStatusTransformerQK {
  kFail = 0,
  kStart,
  kFirstSwapAx,
  kSecondSwapAx,
  kFirstReshape,
  kSecondReshape,
  kSuccess
};

// /*
// kStart ---> kFirstSwapAx ---> kSecondSwapAx ---> kFirstReshape ---> kSecondReshape ---> kSuccess
// OR
// kStart ---> kFirstSwapAx ---> kSecondSwapAx ---> kFirstReshape ---> kSuccess
// each status except kStart is connected with kFail
// */

inline bool CheckSwapAxisConditionsQK(const BiDirectedNode& input_node) {
  if (input_node.outputs.size() != 1)
    return false;
  return CheckSwapAxisConditions(*input_node.node);
}

inline bool CheckReshapeConditionsQK(const BiDirectedNode& input_node, const index_t out_index) {
  if (input_node.outputs.size() != 1)
    return false;
  return CheckReshapeConditions(*input_node.node, out_index);
}

inline bool CheckSplitConditions(const std::vector<const BiDirectedNode*>& matched_list,
                                 const BiDirectedNode& node) {
  const SplitParam& param = dmlc::get<SplitParam>(node.node->attrs.parsed);

  if (param.axis != -1 || param.sections != 3 || param.squeeze_axis)
    return false;

  const auto first_reshape  = (*(matched_list.end() - 2))->node;
  const auto second_reshape = (*(matched_list.end() - 1))->node;
  if (first_reshape->op() != Op::Get("_npx_reshape") ||
      second_reshape->op() != Op::Get("_npx_reshape")) {
    return false;
  }
  // 3 sections - ensure that every output is used only once
  if (node.outputs.size() == 3 && node.outputs.count(first_reshape) &&
      node.outputs.count(second_reshape)) {
    return true;
  }

  return false;
}

inline bool Select(SelectStatusTransformerQK* status,
                   std::vector<const BiDirectedNode*>* matched_list,
                   const BiDirectedNode& seed_node,
                   const std::shared_ptr<NodeAttr>& node_attr) {
  if (seed_node.node->op() == Op::Get("batch_dot")) {
    *status = kStart;
    matched_list->clear();
    matched_list->push_back(&seed_node);
    return true;
  }
  return false;
}

template <bool with_split>
bool SelectInput(SelectStatusTransformerQK* status,
                 std::vector<const BiDirectedNode*>* matched_list,
                 const BiDirectedNode& n,
                 const BiDirectedNode& input_node) {
  if (*status == kFail || *status == kSuccess || input_node.node->is_variable())
    return false;
  const auto& raw_input_node = *input_node.node;
  switch (*status) {
    case kStart:
      if (raw_input_node.op() == Op::Get("SwapAxis")) {
        if (CheckSwapAxisConditionsQK(input_node)) {
          *status = kFirstSwapAx;
          matched_list->push_back(&input_node);
          return true;
        }
      }
      break;
    case kFirstSwapAx:
      if (raw_input_node.op() == Op::Get("SwapAxis")) {
        if (CheckSwapAxisConditionsQK(input_node)) {
          *status = kSecondSwapAx;
          matched_list->push_back(&input_node);
          return true;
        }
      }
      break;
    case kSecondSwapAx:
      if (raw_input_node.op() == Op::Get("_npx_reshape")) {
        // input to reshape must be first or second output from split
        if (CheckReshapeConditionsQK(input_node, 0) || CheckReshapeConditionsQK(input_node, 1)) {
          *status = kFirstReshape;
          matched_list->push_back(&input_node);
          return true;
        }
      }
      break;
    case kFirstReshape:
      if (raw_input_node.op() == Op::Get("_npx_reshape")) {
        if (CheckReshapeConditionsQK(input_node, 0) || CheckReshapeConditionsQK(input_node, 1)) {
          if constexpr (with_split) {
            *status = kSecondReshape;
          } else {
            *status = kSuccess;
          }
          matched_list->push_back(&input_node);
          return true;
        }
      }
      break;
    case kSecondReshape:
      if (raw_input_node.op() == Op::Get("_split_v2") &&
          CheckSplitConditions(*matched_list, input_node)) {
        *status = kSuccess;
        return true;
      }
      break;
    default:
      *status = kFail;
      return false;
  }
  return false;
}

inline std::vector<BiDirectedNode*> Filter(const SelectStatusTransformerQK& status,
                                           const std::vector<const BiDirectedNode*>& matched_list,
                                           const std::vector<BiDirectedNode*>& candidates) {
  if (status != kSuccess) {
    return std::vector<BiDirectedNode*>(0);
  } else {
    std::vector<BiDirectedNode*> ret;
    for (auto i : matched_list) {
      auto non_const_i = const_cast<BiDirectedNode*>(i);
      if (std::find(candidates.begin(), candidates.end(), non_const_i) != candidates.end()) {
        ret.push_back(non_const_i);
      }
    }
    return ret;
  }
}

template <bool with_split>
nnvm::ObjectPtr CreateSubgraphNode(const nnvm::Symbol& sym, const int subgraph_id = 0) {
  std::string op_name;
  if constexpr (with_split) {
    op_name = "_sg_onednn_selfatt_qk_split";
  } else {
    op_name = "_sg_onednn_selfatt_qk";
  }
  nnvm::ObjectPtr n = nnvm::Node::Create();
  // this op has single output, remove duplicated
  auto last_node = sym.outputs[0].node;
  nnvm::Symbol new_sym;
  new_sym.outputs.emplace_back(last_node);
  std::ostringstream node_name;

  DFSVisit(new_sym.outputs, [&](const nnvm::ObjectPtr& node) {
    if ((node->op() == Op::Get("_npx_reshape"))) {
      auto const& reshape_param = nnvm::get<NumpyXReshapeParam>(node->attrs.parsed);
      // set heads attribute - all necessary conditions are checked before
      n->attrs.dict["heads"] = std::to_string(reshape_param.newshape[2]);
    }
  });

  node_name << op_name << subgraph_id;
  n->attrs.name = node_name.str();
  n->attrs.op   = Op::Get(op_name);
  CHECK(n->attrs.op);
  n->op()->attr_parser(&(n->attrs));
  return n;
}

inline void ConnectSubgraphOutputs(const nnvm::ObjectPtr n,
                                   std::vector<nnvm::NodeEntry*>* output_entries) {
  // connect all extern output entries to output[0]
  for (size_t i = 0; i < output_entries->size(); ++i) {
    auto entry_ptr = output_entries->at(i);
    *entry_ptr     = nnvm::NodeEntry{n, entry_ptr->index, 0};
  }
}

}  // namespace qk_common
}  // namespace op
}  // namespace mxnet

#endif  // if MXNET_USE_ONEDNN == 1
#endif  // MXNET_OPERATOR_SUBGRAPH_DNNL_DNNL_TRANSFORMER_QK_COMMON_H_
