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
#if DMLC_USE_CXX11
#include "./narray.h"
#include "./dag_engine.h"
#endif
#include "./symbolic.h"

namespace mxnet {
/*! \brief option to pass into the forward function */
struct Option {
  /*! \brief whether it is training phase*/
  int is_train;
};

/*! \brief operation request type to Forward and Backward */
enum OpReqType {
  /*! \brief no operation, do not write anything */
  kNullOp,
  /*! \brief write gradient to provided space */
  kWriteTo,
  /*!
   * \brief perform an inplace write,
   * Target shares memory with one of input arguments.
   * This option only happen when
   */
  kWriteInplace,
  /*! \brief add to the provided space */
  kAddTo
};
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
   * \brief perform a forward operation of StaticOperator, save the output to TBlob.
   * \param opt option on Forward such as whether this is training phase.
   * \param ctx runtime context
   * \param in_data array of input data, it is const
   * \param req the request types of saving operation, can only be kWriteTo or kWriteInplace.
   * \param out_data array of output data, pointer is used to indicate that this is holder
   *        the space of TBlob in out_data must be pre-allocated with InferShape
   * \sa OpReqType
   */
  virtual void Forward(Option opt,
                       RunContext ctx,
                       const std::vector<TBlob> &in_data,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &out_data) = 0;
  /*!
   * \brief Perform a backward Operation, write gradient to the in_grad.
   * \param ctx runtime context
   * \param out_grad the gradient value we get from output of the StaticOperator
   * \param in_data the array of input data.
   * \param out_data the array of output data.
   * \param req request types of the saving operation, can be all types.
   * \param in_grad the array of gradient we need to write to.
   * \sa OpReqType
   */
  virtual void Backward(RunContext ctx,
                        const std::vector<TBlob> &out_grad,
                        const std::vector<TBlob> &in_data,
                        const std::vector<TBlob> &out_data,
                        const std::vector<OpReqType> &req,
                        const std::vector<TBlob> &in_grad) = 0;
};

#if DMLC_USE_CXX11
                        const std::vector<TBlob> &out_grad,
                        const std::vector<TBlob> &in_data,
                        const std::vector<TBlob> &out_data,
                        const std::vector<OpReqType> &req,
                        const std::vector<TBlob> &in_grad) = 0;
};

#if DMLC_USE_CXX11
/*!
 * \brief Operator interface.
 *  Operator is an object can have Forward and Backward function.
 *
 *  It can be created from
 */
class Operator {
 public:
  /*! \brief destructor */
  virtual ~Operator() {}
  /*!
   * \brief Perform a Forward operation of Operator
   *  After this operation, user can get the result by using function head.
   */
  virtual void Forward() = 0;
  /*!
   * \brief Perform a Backward operation of the Operator.
   * This must be called after Forward.
   * After this operation, NArrays specified by grad_in_args_store will be updated accordingly.
   */
  virtual void Backward() = 0;
  /*! \return get array of heads in the operator */
  virtual const std::vector<NArray> &head() const = 0;
  /*!
   * \brief Create an operator by bind symbol with context and arguments.
   *  If user do not want to compute the gradients of i-th argument, grad_req_type[i] can be kNullOp.
   *
   * \param ctx the context of binding.
   * \param symbol the symbol that specifies the output of Forward pass.
   * \param in_args the NArray that stores the input arguments to the symbol.
   * \param grad_in_args_store NArray that is used to store the gradient output of the input arguments.
   * \param grad_req_type requirment type of gradient saving. Can only be in {kNullOp, kAddTo, kWriteTo}.
   */
  static Operator *Bind(Symbol symbol,
                        Context ctx,
                        const std::vector<NArray> &in_args,
                        const std::vector<NArray> &grad_in_args_store,
                        const std::vector<OpReqType> &grad_req_type);
};  // class operator
#endif
}  // namespace mxnet
#endif  // MXNET_OPERATOR_H_
