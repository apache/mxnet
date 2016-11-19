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
#include <memory>
#include <functional>


namespace mxnet {
using nnvm::Graph;
using nnvm::Op;
using nnvm::Node;
using nnvm::NodePtr;
using nnvm::NodeAttrs;
using nnvm::NodeEntry;

inline std::string DefaultVarName(const std::string &op_name,
                                  const std::string &arg_name) {
  if (op_name.length() == 0) {
    return arg_name;
  } else {
    return op_name + '_' + arg_name;
  }
}

Graph UpgradeJSON_000800_000900(Graph g) {
  using nnvm::Symbol;
  using nnvm::FListInputNames;

  nnvm::DFSVisit(g.outputs, [](const std::shared_ptr<Node>& n) {
      static auto& flist_inputs = Op::GetAttr<FListInputNames>("FListInputNames");
      if (n->inputs.size() < n->num_inputs()) {
        FListInputNames fn = flist_inputs.get(n->op(), nullptr);
        if (fn == nullptr) return;

        auto arg_names = fn(n->attrs);
        for (size_t i = n->inputs.size(); i < n->num_inputs(); ++i) {
          n->inputs.push_back(
              Symbol::CreateVariable(
                  DefaultVarName(n->attrs.name, arg_names[i])).outputs[0]);
        }
      }
    });
  return g;
}

static std::vector<std::pair<int, std::function<Graph(Graph)> > > upgrader_list = {
  {MXNET_MAKE_VERSION(0, 9, 0), UpgradeJSON_000800_000900}
};

Graph LoadLegacyJSONPass(Graph g) {
  Graph load = nnvm::ApplyPass(g, "LoadJSON");
  int version = MXNET_MAKE_VERSION(0, 8, 0);
  if (load.attrs.find("mxnet_version") != load.attrs.end()) {
    version = nnvm::get<int>(*load.attrs["mxnet_version"]);
  }
  auto it = upgrader_list.begin();
  while (it != upgrader_list.end() && it->first <= version) ++it;
  while (it != upgrader_list.end()) {
    load = it->second(load);
    ++it;
  }
  return load;
}

// register pass
NNVM_REGISTER_PASS(LoadLegacyJSON)
.describe("Return a new Graph, loaded from src.attrs[\"json\"] and upgraded to current version")
.set_body(LoadLegacyJSONPass)
.set_change_graph(true)
.depend_graph_attr("json");

}  // namespace mxnet
