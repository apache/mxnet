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
#if DMLC_USE_CXX11 == 1
#include "./narray.h"
#include "./dag_engine.h"
#endif
namespace mxnet {
/*!
 * \brief StaticOperator interface
 *  StaticOperator is a stateful object that can be used to call forward and backprop
 *
 *  This interface relies on pre-allocated memory in TBlob, the caller need to set
 *  the memory region in TBlob correctly before calling Forward and Backward
 *
 * \sa TBlob, TShape
 */
class StaticOperator {
 public:
  /*! \brief destructor */
  virtual ~StaticOperator() {}
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

#if DMLC_USE_CXX11 == 1
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
  /*! \brief destructor */
  virtual ~Operator() {}
  /*!
   * \brief describe property of op
   * \return a bit map in int
   */
  virtual int DescribeProperty() const = 0;
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
  /*!
   * \brief Create a wrapper of static operator to wrap it into Operator.
   *  This function takes ownership of op
   * \param op static operator to wrap from
   * \param ctx context of the created operator
   * \return a wrapper operator
   */
  static Operator *CreateWrapper(StaticOperator *op, Context ctx);
};  // class operator
#endif
}  // namespace mxnet
#endif  // MXNET_OPERATOR_H_
