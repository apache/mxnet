/*!
 *  Copyright (c) 2015 by Contributors
 * \file static_operator.h
 * \brief static operator interface of mxnet
 */
#ifndef MXNET_STATIC_OPERATOR_H_
#define MXNET_STATIC_OPERATOR_H_
// this file will be seen by cuda, no c++11 for now
#include <dmlc/base.h>
#include <vector>
#include "./base.h"
#include "./tensor_blob.h"

namespace mxnet {
/*!
 * \brief static StaticOperator interface (current interface have not yet todo with scheduler),
 *  StaticOperator is a stateful object that can be used to call forward and backprop
 *
 *  This interface relies on pre-allocated memory in TBlob, the caller need to set
 *  the memory region in TBlob correctly before calling Forward and Backward
 *
 * \sa TBlob, TShape
 */
class StaticOperator {
 public:
  /*!
   * \brief describe property of op
   * \return a bit map in int
   */
  virtual int DescribeProperty() const {
    // default most of layer only conatin internal state
    return kContainInteralState;
  }
  /*!
   * \brief perform a forward operation of StaticOperator, save the output to TBlob
   * \param opt option on Forward such as whether this is training phase
   * \param ctx runtime context
   * \param in_data array of input data, it is const
   * \param out_data array of output data,
   *        the space of TBlob in out_data must be pre-allocated with InferShape
   */
  virtual void Forward(Option opt,
                       RunContext ctx,
                       const std::vector<TBlob> &in_data,
                       const std::vector<TBlob> &out_data) = 0;
  /*!
   * \brief perform a backward operation of the StaticOperator to get the gradient
   * \param ctx runtime context
   * \param grad_next the gradient value we get from output of the StaticOperator
   * \param in_data the array of input data
   * \param out_grad array of output gradient, there could be three possible TBlob
   *  in the each element in the array
   * \param req request types of the gradient saving operation
   *                  only inplace will change input data
   * \sa GradReqType
   */
  virtual void Backward(RunContext ctx,
                        const std::vector<TBlob> &grad_next,
                        const std::vector<TBlob> &in_data,
                        const std::vector<TBlob> &out_grad,
                        const std::vector<GradReqType> &req) = 0;
  /*!
   * \brief factory function, create a new StaticOperator
   * \param type the type of StaticOperator
   * \param ctx the context device type of StaticOperator
   * \return a pointer of StaticOperator object
   */
  static StaticOperator *Create(const char *type, Context ctx);
};
}  // namespace mxnet
#endif  // MXNET_STATIC_OPERATOR_H_
