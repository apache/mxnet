/*!
 * Copyright (c) 2015 by Contributors
 * \file composite_operator.h
 * \brief composite operator of mxnet
 * \author Bing Xu
*/
#ifndef MXNET_OPERATOR_COMPOSITE_OPERATOR_H_
#define MXNET_OPERATOR_COMPOSITE_OPERATOR_H_
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
            const std::vector<GradReqType> &req);
  /*!
   * \brief perform a forward operation of operator, save the output to NArray
   *        This method only pushes an execution request to the DAG engine, and
   *        return immediately. Actual execution is conducted by the DAG engine.
   * \param opt option on Forward such as whether this is training phase
   * \param ctx runtime context
   * \param in_data array of input data, it is const
   * \param out_data array of output data,
   *        the space of NArray in out_data must be pre-allocated with InferShape
   * \sa NArray
   */
  virtual void Forward(Option opt,
                       RunContext ctx,
                       const std::vector<NArray> &in_data,
                       const std::vector<NArray> &out_data);
  /*!
   * \brief perform a forward operation of operator (no change to binded NArray)
   * \param opt option on Forward such as whether this is training phase
   */
  virtual void Forward(Option opt);
  /*!
   * \brief perform a backward operation of the operator to get the gradient
   *        This method only pushes an execution request to the DAG engine, and
   *        return immediately. Actual execution is conducted by the DAG engine.
   * \param ctx runtime context
   * \param grad_next the gradient value of the output of the operator, used by chain rule.
   * \param in_data the array of input data
   * \param out_data the array of output data
   * \param out_grad array of output gradient
   * \param req request types of the gradient saving operation
   *                  only inplace will change input data
   * \sa GradReqType, NArray
   */
  virtual void Backward(RunContext ctx,
                        const std::vector<NArray> &grad_next,
                        const std::vector<NArray> &in_data,
                        const std::vector<NArray> &out_data,
                        const std::vector<NArray> &out_grad,
                        const std::vector<GradReqType> &req);
  /*!
   * \brief perform a backward operation of the operator to get the gradient
   *        No change to Binded NArray
   */
  virtual void Backward();
  /*!
   * \brief perform an extraction operation to get feature map
   * \param name of symbol need to be extracted
   * \return empty narray for invalid name or narray of the feature map
   */
  virtual NArray Extract(const std::string &symbol_name);

 private:
  /*!
   * \brief Update connections data in/after bind
   *  \param in input narray
   *  \param grad gradient narray
   *  \param req gradient request
   */
  void UpdateConnection(const std::vector<NArray> &in,
                        const std::vector<NArray> &grad,
                        const std::vector<GradReqType> &req);
  /*!
   * \brief Allocate each op node
   */
  void AllocateNodes(RunContext ctx);
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
};  // class CompositeOperator
}  // namespace mxnet
#endif  // MXNET_OPERATOR_COMPOSITE_OPERATOR_H_
