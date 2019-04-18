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
#include <sstream>
#include "common.h"
#include "subgraph_property.h"
#include "../nn/fully_connected-inl.h"
#include "../nn/activation-inl.h"
#include "../nn/concat-inl.h"
#include "../tensor/slice_split_embedding.h"
#include "../tensor/indexing_op.h"
#include "../tensor/matrix_op-inl.h"
#include "../slice_channel-inl.h"
namespace mxnet {
namespace op {
#define EMBEDDING_NODE_NAME "Embedding"
#define CONCAT_NODE_NAME "Concat"
class SgWideAndDeepInputFuseSelector : public SubgraphSelector {
 public:
  /*! \brief pattern match status */
  enum SelectStatus {
    kFail = 0,
    kSuccess,
  };

 private:
  bool disable_all;
  SelectStatus status;

 public:
  explicit SgWideAndDeepInputFuseSelector(int dis_all)
      : disable_all(dis_all), status(kFail) {}
  bool Select(const nnvm::Node &n) override {
    if (disable_all)
        return false;
    if (n.op() && n.op()->name == CONCAT_NODE_NAME) {
      status = kSuccess;
      return true;
    }
    return false;
  }
  bool SelectInput(const nnvm::Node &n, const nnvm::Node &new_node) override {
    if (disable_all) return false;
    if (new_node.is_variable() )
        return false;
    return true;
  }
  bool SelectOutput(const nnvm::Node &n, const nnvm::Node &new_node) override {
      return false;
  }
  std::vector<nnvm::Node *> Filter(
      const std::vector<nnvm::Node *> &candidates) override {
    if (status != kSuccess) {
      return std::vector<nnvm::Node *>(0);
    } else {
      return candidates;
    }
  }
};
template <typename T>
static std::string int_vector_to_attr(T v) {
  std::stringstream ss;
  ss << "[";
  size_t i = 0;
  for (; i < v.size()-1; i++) {
      ss << v[i] << ",";
  }
  ss << v[i];
  ss << "]";
  return ss.str();
}
static std::string int_vector_to_tuple_attr(mxnet::Tuple<dmlc::optional<int>> v) {
  std::stringstream ss;
  ss << "(";
  index_t i = 0;
  for (; i < v.ndim()-1; ++i) {
      ss << v[i] << ",";
  }
  ss << v[i];
  ss << ")";
  return ss.str();
}
static std::string get_value_from_op_prop(
  std::unordered_map<std::string, std::string> op_dict, std::string key) {
  std::unordered_map<std::string, std::string>::const_iterator got = op_dict.find(key);
  if (got == op_dict.end())
      return "";
  else
      return got->second;
}
class SgWideAndDeepInputFuseProperty : public SubgraphProperty {
 public:
  SgWideAndDeepInputFuseProperty() {
    disable_all = dmlc::GetEnv("MXNET_DISABLE_WIDE_DEEP_OPT", 0);
    if (disable_all) {
      LOG(INFO) << "Wide And Deep Input Fuse is disabled.";
    } else {
      LOG(INFO) << "Start to execute Wide And Deep Input Fuse optimization pass.";
    }
  }
  static SubgraphPropertyPtr Create() {
    return std::make_shared<SgWideAndDeepInputFuseProperty>();
  }
  nnvm::NodePtr CreateSubgraphNode(const nnvm::Symbol &sym,
                                  const int subgraph_id = 0) const override {
    std::vector<nnvm::NodePtr> emb_nodes;
    std::vector<nnvm::NodePtr> slice_nodes;
    nnvm::NodePtr split_nodes;
    nnvm::NodePtr concat_node = nullptr;
    DFSVisit(sym.outputs, [&](const nnvm::NodePtr &node) {
      if (node->is_variable()) return;
      auto &op_name = node->op()->name;

      // The Assumption is only base on W&D which all
      // embedding occur at the beginning and output to 1 concat node
      if (op_name == EMBEDDING_NODE_NAME) {
          emb_nodes.push_back(node);
      }
      if (op_name == CONCAT_NODE_NAME) {
          concat_node = node;
      }
      if (op_name == "slice") {
          slice_nodes.push_back(node);
      }
      if (op_name == "SliceChannel") {
          split_nodes = node;
      }
    });
    nnvm::NodePtr pe = nnvm::Node::Create();
    pe->attrs.name = "SliceSplitEmbeddingConcatFuse_0";
    pe->attrs.op = Op::Get("SliceSplitEmbeddingConcatFuse");
    CHECK(pe->attrs.op);
    std::vector<int> v_in_dims;
    std::vector<int> v_out_dims;
    std::vector<int> v_types;
    std::vector<bool> v_sparse_grads;
    std::vector<nnvm::NodePtr>::iterator it;
    for (it = emb_nodes.begin(); it != emb_nodes.end(); it++) {
        nnvm::NodePtr em_node = *it;
        const EmbeddingParam &param = nnvm::get<EmbeddingParam>(em_node->attrs.parsed);
        v_in_dims.push_back(param.input_dim);
        v_out_dims.push_back(param.output_dim);
        v_types.push_back(param.dtype);
        v_sparse_grads.push_back(param.sparse_grad);
    }
    pe->attrs.dict["input_dims"] =
        int_vector_to_attr<std::vector<int>>(v_in_dims);
    pe->attrs.dict["output_dims"] =
        int_vector_to_attr<std::vector<int>>(v_out_dims);
    const ConcatParam& concat_param =
        nnvm::get<ConcatParam>(concat_node->attrs.parsed);
    pe->attrs.dict["concat_dim"] = std::to_string(concat_param.dim);
    pe->attrs.dict["num_outputs"] =
        get_value_from_op_prop(split_nodes->attrs.dict, "num_outputs");
    pe->attrs.dict["squeeze_axis"] =
        get_value_from_op_prop(split_nodes->attrs.dict, "squeeze_axis");
    const SliceParam& sclie_1_param =
        nnvm::get<SliceParam>(slice_nodes[0]->attrs.parsed);
    pe->attrs.dict["cont_begin"] = int_vector_to_tuple_attr(sclie_1_param.begin);
    pe->attrs.dict["cont_end"] = int_vector_to_tuple_attr(sclie_1_param.end);
    const SliceParam& sclie_0_param =
        nnvm::get<SliceParam>(slice_nodes[1]->attrs.parsed);
    pe->attrs.dict["embed_begin"] = int_vector_to_tuple_attr(sclie_0_param.begin);
    pe->attrs.dict["embed_end"] = int_vector_to_tuple_attr(sclie_0_param.end);

    pe->op()->attr_parser(&(pe->attrs));
    return pe;
  }
  SubgraphSelectorPtr CreateSubgraphSelector() const override {
    auto selector =
        std::make_shared<SgWideAndDeepInputFuseSelector>(disable_all);
    return selector;
  }
  void ConnectSubgraphOutputs(
      const nnvm::NodePtr n,
      std::vector<nnvm::NodeEntry *> *output_entries) const override {
    for (size_t i = 0; i < output_entries->size(); ++i) {
      auto entry_ptr = output_entries->at(i);
      *entry_ptr = nnvm::NodeEntry{n, entry_ptr->index, 0};
    }
  }
  void ConnectSubgraphInputs(
      const nnvm::NodePtr n, std::vector<nnvm::NodeEntry *> *input_entries,
      std::vector<nnvm::NodeEntry> *orig_input_entries) const override {
    std::unordered_map<std::string, int> name_count_map;
    for (size_t i = 0; i < orig_input_entries->size(); ++i) {
        nnvm::NodeEntry &entry = (*orig_input_entries)[i];
        if (entry.node->is_variable()) {
            auto var_name = entry.node->attrs.name;
            auto it = name_count_map.find(var_name);
            if (name_count_map.end() == it) {
                name_count_map.emplace(var_name, 0);
                n->inputs.push_back(entry);
            }
        } else {
            n->inputs.push_back(entry);
        }
    }
  }

 private:
  int disable_all;
};

MXNET_REGISTER_SUBGRAPH_PROPERTY(WIDE_AND_DEEP_INPUT_FUSE,
                                 SgWideAndDeepInputFuseProperty);
}  // namespace op
}  // namespace mxnet
