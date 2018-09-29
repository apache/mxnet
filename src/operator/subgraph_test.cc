#include "./subgraph/common.h"
#include "./subgraph/subgraph_property.h"

namespace mxnet {
namespace op {

class SgSelector : public SubgraphSelector {
 public:
  SgSelector() {
    find_sg = false;
  }
  bool Select(const nnvm::Node &n) override {
    return n.op() && n.op()->name == "Convolution";
  }
  bool SelectInput(const nnvm::Node &n, const nnvm::Node &new_node) override {
    return false;
  }
  bool SelectOutput(const nnvm::Node &n, const nnvm::Node &new_node) override {
    if (new_node.op() && new_node.op()->name == "Activation" && !find_sg) {
      find_sg = true;
      return true;
    } else {
      return false;
    }
  }
  std::vector<nnvm::Node *> Filter(const std::vector<nnvm::Node *> &candidates) override {
    if (find_sg)
      return candidates;
    else
      return std::vector<nnvm::Node *>();
  }
 private:
  bool find_sg;
};

class SgProperty : public SubgraphProperty {
 public:
  static SubgraphPropertyPtr Create() {
    return std::make_shared<SgProperty>();
  }
  nnvm::NodePtr CreateSubgraphNode(
    const nnvm::Symbol &sym, const int subgraph_id = 0) const override {
      nnvm::NodePtr n = nnvm::Node::Create();
      n->attrs.op = Op::Get("_CachedOp");
      n->attrs.name = "ConvBN" + std::to_string(subgraph_id);
      n->attrs.subgraphs.push_back(std::make_shared<nnvm::Symbol>(sym));
      return n;
  }
  SubgraphSelectorPtr CreateSubgraphSelector() const override {
    return std::make_shared<SgSelector>();
  }
};

MXNET_REGISTER_SUBGRAPH_PROPERTY(SgTest, SgProperty);

}
}
