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
 * \brief operator interface
 *  operator is an object can be scheduled by DAG engine directly.
 *
 *  This interface relies on NArray. The user should prepare the input NArray and
 *  output NArray by themselves.
 * \sa Operator
 */
class Operator {
 public:
  /*!
   * \brief describe property of op
   * \return a bit map in int
   */
  virtual int DescribeProperty() const;
  /*!
   * \brief set the global context of the Operator
   * \param ctx the context to be set to
   */
  virtual void SetContext(Context ctx);
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
                       const std::vector<NArray> &out_data) = 0;
  /*!
   * \brief perform a backward operation of the operator to get the gradient
   *        This method only pushes an execution request to the DAG engine, and
   *        return immediately. Actual execution is conducted by the DAG engine.
   * \param ctx runtime context
   * \param grad_next the gradient value of the output of the operator, used by chain rule.
   * \param in_data the array of input data
   * \param out_grad array of output gradient
   * \param req request types of the gradient saving operation
   *                  only inplace will change input data
   * \sa GradReqType, NArray
   */
  virtual void Backward(RunContext ctx,
                        const std::vector<NArray> &grad_next,
                        const std::vector<NArray> &in_data,
                        const std::vector<NArray> &out_grad,
                        const std::vector<GradReqType> &req) = 0;

 protected:
  /**
   * \brief the global context denots the device info.
   */
  Context global_ctx;
};  // class operator
}  // namespace mxnet
#endif  // MXNET_OPERATOR_H_
