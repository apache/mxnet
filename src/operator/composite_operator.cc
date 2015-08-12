/*!
 * Copyright (c) 2015 by Contributors
 * \file composite_operator.cc
 * \brief composite operator of mxnet
 * \author Bing Xu
*/
#include <mxnet/base.h>
#include <mxnet/symbolic.h>
#include <mxnet/operator.h>
#include <string>
#include <vector>
#include <unordered_map>

namespace mxnet {
/*!
 * \brief composite_operator interface
 * composite operator is a combination of static operator from static graph
 */
class CompositeOperator : public Operator {
 public:
  /*! \brief destructor */
  virtual ~CompositeOperator() {}
  /*!
   * \brief describe property of op
   * \return a bit map in int
   */
  virtual int DescribeProperty() const {
    // default most of layer only conatin internal state
    return kContainInteralState;
  }
  /*! \brief Make operator by using graph
   *  \param ctx ctx context of the created operator
   *  \param in input narray
   *  \param grad gradient narray
   *  \param req gradient request
   */
  void Bind(Context ctx,
            const std::vector<NArray> &in,
            const std::vector<NArray> &grad
            const std::vector<GradReqType> &req) {
    ctx_ = ctx;
    // infer shape
    // build dict
    // alloc nodes
    // alloc feature map
    UpdateConnection(in, grad, req);
  }
  /*!
   * \brief Update connections data in/after bind
   *  \param in input narray
   *  \param grad gradient narray
   *  \param req gradient request
   */
  void UpdateConnection(const std::vector<NArray> &in,
                        const std::vector<NArray> &grad,
                        const std::vector<GradReqType> &req) {
    CHECK_EQ(in.size() == nodes_.size());
    CHECK_EQ(grad.size() == nodes_.size());
    CHECK_EQ(req.size() == nodes_.size());
  }
  /*!
   * \brief perform a forward operation of operator (no change to binded NArray)
   * \param opt option on Forward such as whether this is training phase
   */
  virtual void Forward(Option opt) {
    for (auto nid : topo_order_) {
      if (nodes_[nid].is_variable) continue;
      nodes_[nid].op->Forward(opt,
                              ctx_,
                              nodes_[nid].inputs,
                              nodes_[nid].outputs);
    }
  }
  /*!
   * \brief perform a backward operation of the operator to get the gradient
   *        No change to Binded NArray
   */
  virtual void Backward() {
    for (auto it = topo_order_.rbegin(); it < topo_order_.rend(); ++it) {
      if (nodes_[*it].is_variable) continue;
      nodes_[*it].op->Backward(ctx_,
                               nodes_[*it].outputs,
                               nodes_[*it].inputs,
                               nodes_[*it].outputs_grad,
                               nodes_[*it].req);
    }
  }
  /*!
   * \brief perform an extraction operation to get outputs
   * \param name of symbol need to be extracted
   * \return empty narray for invalid name or narray of the feature map
   */
  virtual std::vector<NArray> Extract(const std::string &symbol_name) {
    auto it = name_dict_.find(symbol_name);
    if (it == name_dict_.end()) return {};
    return nodes_[it->second].outputs;
  }
 private:
  /*!
   * \brief Structure for OpNode
  */
  struct OpNode {
    /*! \brief Static Operator */
    std::unique_ptr<Operator> op;
    /*! \brief inputs (init after setting output correctly) */
    std::vector<NArray> inputs;
    /*! \brief outputs */
    std::vector<NArray> outputs;
    /*! \brief gradient for output */
    std::vector<NArray> outputs_grad;
    /*! \brief gradient req for grad */
    std::vector<GradReqType> req;
    /*! \brief is variable */
    bool is_variable;
  };
  /*! \brief connections */
  std::vector<OpNode> nodes_;
  /*! \brief topo order of connections */
  std::vector<uint_32> topo_order_;
  /*! \brief static graph */
  StaticGraph graph_;
  /*! \brief running context */
  RunContext ctx_;
  /*! \brief name id dictionary */
  std::unordered_map<std::string, uint_32> name_dict_;
};  // class CompositeOperator
}  // namespace mxnet
