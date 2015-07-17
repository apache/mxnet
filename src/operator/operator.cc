/*!
 *  Copyright (c) 2015 by Contributors
 * \file operator.cc
 * \brief the implementation of narray operator
 * \author Naiyan Wang
 */
#include <mxnet/operator.h>

namespace mxnet {

  Operator::Operator(StaticOperator* op, Context ctx) {
    this->op = op;
    this->global_ctx = ctx;
  }
  /*!
   * \brief get types of input argument of this oeprator
   * \return a vector corresponding to type of each argument
   *  this order is same as the order of inputs in Forward, InferShape and Backward
   */
  std::vector<ArgType> Operator::DescribeArgs() const {
    // default most of layers only have one data argument
    return op->DescribeArgs();
  }
  /*!
   * \brief describe property of op
   * \return a bit map in int
   */
  int Operator::DescribeProperty() const {
    // default most of layer only conatin internal state
    return op->DescribeProperty();
  }
  /*!
   * \brief set param for the operator from string
   * \param name parameter name
   * \param val string for configuration
   */
  void Operator::SetParam(const char *name, const char *val) {
    op->SetParam(name, val);
  }
  /*!
   * \brief inter the shapes of outputs and unknown input arguments
   * \param in_shape the shape of input arguments of the operator
   *     this should be of same length as the vector returned by DescribeArgs
   *     in_shape allows unknown elements, which are checked by shape.ndim() == 0.
   *     For unknown shapes, InferShape will try to fill in the correct Shape in in_shape
   *     For known shapes, InferShape will check shape consistency
   *
   *     common practice: set the shape of data input, and usually weight's shape can be infered
   *
   * \param out_shape the shape of outputs of the operator
   *     InferShape will modify the vector to fill output TShape
   */
  void Operator::InferShape(std::vector<TShape> *in_shape,
                          std::vector<TShape> *out_shape) {
    op->InferShape(in_shape, out_shape);
  }
  /*!
   * \brief perform a forward operation of operator, save the output to TBlob
   * \param opt option on Forward such as whether this is training phase
   * \param ctx runtime context
   * \param in_data array of input data, it is const
   * \param out_data array of output data,
   *        the space of TBlob in out_data must be pre-allocated with InferShape
   */
  void Operator::Forward(Option opt,
                       RunContext ctx,
                       const std::vector<NArray> &in_data,
                       const std::vector<NArray> &out_data) {
    std::vector<DAGEngine::Variable> used_var;
    std::vector<DAGEngine::Variable> mutate_var;
    std::vector<TBlob> in;
    std::vector<TBlob> out;
    for (size_t i = 0; i < in_data.size(); ++i) {
      used_var.push_back(in_data[i].Var());
      in.push_back(in_data[i].data());
    }
    for (size_t i = 0; i < out_data.size(); ++i) {
      mutate_var.push_back(out_data[i].Var());
      out.push_back(out_data[i].data());
    }
    DAGEngine::Get()->Push([this, opt, ctx, in, out](RunContext ctx) {
      op->Forward(opt, ctx, in, out);
      }, global_ctx, used_var, mutate_var);
  }
  /*!
   * \brief perform a backward operation of the operator to get the gradient
   * \param ctx runtime context
   * \param grad_next the gradient value we get from output of the operator
   * \param in_data the array of input data
   * \param out_grad array of output gradient, there could be three possible TBlob
   *  in the each element in the array
   * \param req request types of the gradient saving operation
   *                  only inplace will change input data
   * \sa GradReqType
   */
  void Operator::Backward(RunContext ctx,
                        const std::vector<NArray> &grad_next,
                        const std::vector<NArray> &in_data,
                        const std::vector<NArray> &out_grad,
                        const std::vector<GradReqType> &req) {
    std::vector<DAGEngine::Variable> used_var;
    std::vector<DAGEngine::Variable> mutate_var;
    std::vector<TBlob> grad_in;
    std::vector<TBlob> grad_out;
    std::vector<TBlob> data;
    for (size_t i = 0; i < grad_next.size(); ++i) {
      used_var.push_back(grad_next[i].Var());
      grad_in.push_back(grad_next[i].data());
    }
    for (size_t i = 0; i < in_data.size(); ++i) {
      used_var.push_back(in_data[i].Var());
      data.push_back(in_data[i].data());
    }
    for (size_t i = 0; i < out_grad.size(); ++i) {
      mutate_var.push_back(out_grad[i].Var());
      grad_out.push_back(out_grad[i].data());
    }
    DAGEngine::Get()->Push([this, ctx, grad_in, grad_out, data, req](RunContext ctx) {
      op->Backward(ctx, grad_in, data, grad_out, req);
      }, global_ctx, used_var, mutate_var);
  }

  void Operator::SetContext(Context ctx) {
    this->global_ctx = ctx;
  }

}  // namespace mxnet
