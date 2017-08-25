/*!
 *  Copyright (c) 2016 by Contributors
 * \file legacy_json_util.cc
 * \brief Utility upgrade symbol from previous versions
 */
#include <dmlc/base.h>
#include <mxnet/base.h>
#include <mxnet/operator.h>
#include <mxnet/op_attr_types.h>
#include <mxnet/ndarray.h>
#include <nnvm/node.h>
#include <nnvm/graph.h>
#include <nnvm/pass.h>
#include <nnvm/op_attr_types.h>
#include <memory>
#include <functional>
#include "../c_api/c_api_common.h"

namespace mxnet {
using nnvm::Graph;
using nnvm::Op;
using nnvm::Node;
using nnvm::NodePtr;
using nnvm::NodeAttrs;
using nnvm::NodeEntry;
using nnvm::Symbol;
using nnvm::FListInputNames;

// First fix things that prevent attr_parser success.
Graph UpgradeJSON_FixParsing(Graph g) {
  nnvm::DFSVisit(g.outputs, [](const std::shared_ptr<Node>& n) {
      static auto& flist_inputs = Op::GetAttr<FListInputNames>("FListInputNames");

      // hold keys that should be converted to hidden keys
      std::vector<std::pair<std::string, std::string> > hidden_keys;

      // remove attrs that prevent parsing
      for (auto it = n->attrs.dict.begin(); it != n->attrs.dict.end();) {
        bool erase = false;
        // remove hidden keys
        for (const auto key : kHiddenKeys) {
          size_t pos = it->first.rfind(key);
          if (pos == 0 || (pos != std::string::npos && pos == it->first.length() - key.length())) {
            hidden_keys.push_back(*it);
            erase = true;
            break;
          }
        }

        auto tmp = it;
        ++it;
        if (erase) n->attrs.dict.erase(tmp);
      }

      // parse
      if (n->op() != nullptr && n->op()->attr_parser != nullptr)
        n->op()->attr_parser(&(n->attrs));

      // add back removed hidden keys
      for (const auto &kv : hidden_keys) {
        bool flag = false;
        for (const auto &key : kHiddenKeys) {
          size_t pos = kv.first.rfind(key);
          if (pos == 0 && key.length() == kv.first.length()) {
            n->attrs.dict["__"+key+"__"] = kv.second;
            flag = true;
            break;
          } else if (pos != std::string::npos && pos > 1
                     && pos == kv.first.length() - key.length()) {
            if (n->is_variable()) break;
            FListInputNames fn = flist_inputs.get(n->op(), nullptr);
            if (fn == nullptr) break;
            auto arg_names = fn(n->attrs);
            auto name = kv.first.substr(0, pos-1);
            auto it = std::find(arg_names.begin(), arg_names.end(), name);
            if (it != arg_names.end()) {
              int idx = it - arg_names.begin();
              if (n->inputs[idx].node->is_variable()) {
                n->inputs[idx].node->attrs.dict["__"+key+"__"] = kv.second;
                flag = true;
              }
            }
            break;
          }
        }
        if (!flag) n->attrs.dict[kv.first] = kv.second;
      }
    });
  return g;
}

Graph UpgradeJSON_Parse(Graph g) {
  nnvm::DFSVisit(g.outputs, [](const std::shared_ptr<Node>& n) {
      if (n->op() != nullptr) {
        if (n->op()->attr_parser != nullptr)
          n->op()->attr_parser(&(n->attrs));
      } else {
        // ugly workaround due to VariableParam is not exposed.
        n->attrs.parsed =
          nnvm::Symbol::CreateVariable(n->attrs.name).outputs[0].node->attrs.parsed;
      }
    });
  return g;
}

inline std::string DefaultVarName(const std::string &op_name,
                                  const std::string &arg_name) {
  if (op_name.length() == 0) {
    return arg_name;
  } else {
    return op_name + '_' + arg_name;
  }
}

// aux variables are not stored in json before 0.9.0. Add them here.
Graph UpgradeJSON_000800_000900(Graph g) {
  nnvm::DFSVisit(g.outputs, [](const std::shared_ptr<Node>& n) {
      static auto& flist_inputs = Op::GetAttr<FListInputNames>("FListInputNames");
      if (n->inputs.size() < n->num_inputs()) {
        FListInputNames fn = flist_inputs.get(n->op(), nullptr);
        if (fn == nullptr) return;

        auto arg_names = fn(n->attrs);
        for (size_t i = n->inputs.size(); i < n->num_inputs(); ++i) {
          auto var = Symbol::CreateVariable(
                      DefaultVarName(n->attrs.name, arg_names[i])).outputs[0];
          var.node->attrs.dict = n->attrs.dict;
          n->inputs.push_back(var);
        }
      }
    });
  return g;
}

// Refactor initializer in v0.9.2
Graph UpgradeJSON_000903_000904(Graph g) {
  nnvm::DFSVisit(g.outputs, [](const std::shared_ptr<Node>& n) {
      static auto& fset_attrs =
        Op::GetAttr<nnvm::FSetInputVarAttrOnCompose>("FSetInputVarAttrOnCompose");

      if (n->op() != nullptr) {
        nnvm::FSetInputVarAttrOnCompose fn = fset_attrs.get(n->op(), nullptr);
        if (fn != nullptr) {
          for (size_t i = 0; i < n->inputs.size(); ++i) {
            if (n->inputs[i].node->is_variable()) {
              fn(n->attrs, n->inputs[i].node, i);
            }
          }
        }
      }
    });
  return g;
}

// ReduceAxisParam: int axis -> optional<int> axis
Graph UpgradeJSON_000904_000905(Graph g) {
  nnvm::DFSVisit(g.outputs, [](const std::shared_ptr<Node>& n) {
      if (n->op() == nullptr) return;
      if (n->op()->name != "argmin" && n->op()->name != "argmax") return;
      if (n->attrs.dict.find("axis") == n->attrs.dict.end() || n->attrs.dict["axis"] != "-1")
        return;
      n->attrs.dict.erase("axis");
      n->op()->attr_parser(&(n->attrs));
    });
  return g;
}

static std::vector<std::pair<int, std::function<Graph(Graph)> > > upgrader_list = {
  {MXNET_VERSION, UpgradeJSON_FixParsing},
  {MXNET_MAKE_VERSION(100, 0, 0), UpgradeJSON_Parse},
  {MXNET_MAKE_VERSION(0, 9, 0), UpgradeJSON_000800_000900},
  {MXNET_MAKE_VERSION(0, 9, 4), UpgradeJSON_000903_000904},
  {MXNET_MAKE_VERSION(0, 9, 5), UpgradeJSON_000904_000905},
};

Graph LoadLegacyJSONPass(Graph g) {
  g.attrs["load_json_no_parse"] = std::make_shared<nnvm::any>(true);
  Graph load = nnvm::ApplyPass(g, "LoadJSON");
  int version = MXNET_MAKE_VERSION(0, 8, 0);
  if (load.attrs.find("mxnet_version") != load.attrs.end()) {
    version = nnvm::get<int>(*load.attrs["mxnet_version"]);
  }
  bool upgrading = false;
  if (version > MXNET_VERSION) {
    LOG(INFO) << "Warning: loading symbol saved by MXNet version " << version
              << " with lower version of MXNet v" << MXNET_VERSION
              << ". May cause undefined behavior. "
              << "Please update MXNet if you encounter any issue";
  } else if (version < MXNET_VERSION) {
    LOG(INFO) << "Loading symbol saved by previous version v"
              << version/10000 << "." << (version/100)%100 << "." << version%100
              << ". Attempting to upgrade...";
    upgrading = true;
  }
  for (auto it = upgrader_list.begin(); it != upgrader_list.end(); ++it) {
    if (it->first > version) load = it->second(load);
  }
  if (upgrading) LOG(INFO) << "Symbol successfully upgraded!";
  return load;
}

// register pass
NNVM_REGISTER_PASS(LoadLegacyJSON)
.describe("Return a new Graph, loaded from src.attrs[\"json\"] and upgraded to current version")
.set_body(LoadLegacyJSONPass)
.set_change_graph(true)
.depend_graph_attr("json");

}  // namespace mxnet
