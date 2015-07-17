/*!
 *  Copyright (c) 2015 by Contributors
 * \file operator.h
 * \brief operator interface of mxnet
 * \author Naiyan Wang
 */
#ifndef MXNET_OPERATOR_H_
#define MXNET_OPERATOR_H_
// this file will be seen by cuda, no c++11 for now
#include <dmlc/base.h>
#include <vector>
#include "./base.h"
#include "./tensor_blob.h"
#include "./static_operator.h"
#include "./narray.h"
#include "./dag_engine.h"

namespace mxnet {
/*!
 * \brief static operator interface (current interface have not yet todo with scheduler),
 *  operator is a stateful object that can be used to call forward and backprop
 *
 *  This interface relies on pre-allocated memory in TBlob, the caller need to set
 *  the memory region in TBlob correctly before calling Forward and Backward
 *
 * \sa Operator
 */
class Operator {
 public:
  /*!
   * \brief construct Operator from StaticOperator and Context
   * \param op StaticOperator to wrap
   * \param ctx Context of the Operator
   */
  Operator(StaticOperator* op, Context ctx);
  /*!
   * \brief get types of input argument of this oeprator
   * \return a vector corresponding to type of each argument
   *  this order is same as the order of inputs in Forward, InferShape and Backward
   */
  virtual std::vector<ArgType> DescribeArgs() const;
  /*!
   * \brief describe property of op
   * \return a bit map in int
   */
  virtual int DescribeProperty() const;
  /*!
   * \brief set param for the operator from string
   * \param name parameter name
   * \param val string for configuration
   */
  virtual void SetParam(const char *name, const char *val);
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
  virtual void InferShape(std::vector<TShape> *in_shape,
                          std::vector<TShape> *out_shape);

  /*!
   * \brief set the context of the Operator
   * \param ctx the context to be set to
   */
  virtual void SetContext(Context ctx);
  /*!
   * \brief perform a forward operation of operator, save the output to TBlob
   * \param opt option on Forward such as whether this is training phase
   * \param ctx runtime context
   * \param in_data array of input data, it is const
   * \param out_data array of output data,
   *        the space of TBlob in out_data must be pre-allocated with InferShape
   */
  virtual void Forward(Option opt,
                       RunContext ctx,
                       const std::vector<NArray> &in_data,
                       const std::vector<NArray> &out_data);
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
  virtual void Backward(RunContext ctx,
                        const std::vector<NArray> &grad_next,
                        const std::vector<NArray> &in_data,
                        const std::vector<NArray> &out_grad,
                        const std::vector<GradReqType> &req);

 private:
  /* \brief the static operator */
  StaticOperator* op;
  Context global_ctx;
};
}  // namespace mxnet
#endif  // MXNET_OPERATOR_H_
