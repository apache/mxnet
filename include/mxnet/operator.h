/*!
 *  Copyright (c) 2015 by Contributors
 * \file operator.h
 * \brief operator interface of mxnet
 */
#ifndef MXNET_OPERATOR_H_
#define MXNET_OPERATOR_H_
#include <dmlc/base.h>
#include <mshadow/tensor.h>
#include "./base.h"
#include "./narray.h"

namespace mxnet {
/*!
 * \brief static operator interface (current interface have not yet todo with scheduler),
 *  operator is a stateful object that can be used to call forward and backprop
 *
 *  This interface relies on pre-allocated memory in TBlob, the caller need to set
 *  the memory region in TBlob correctly before calling Forward and Backward
 *
 * \sa mshadow::TBlob, mshadow::TShape
 */
class Operator {
 public:
  /*! \brief context to pass into the forward function */
  struct Context {
    /*! \brief whether it is training phase*/
    int is_train;
  };
  /*!
   * \brief set param for the operator from string
   * \param name parameter name
   * \param val string for configuration
   */
  virtual void SetParam(const char *name, const char *val) {}  
  /*!
   * \brief inter the shape of output given the input data
   * \param in_shape the shape of input arguments of the operator
   * \param out_shape the shape of outputs of the operator
   */
  virtual void InferShape(const std::vector<mshadow::TShape> &in_shape,
                          std::vector<mshadow::TShape> *out_shape) = 0;
  /*!
   * \brief perform a forward operation of operator, save the output to TBlob
   * \param ctx context on Forward such as whether this is training phase
   * \param in_data array of input data
   * \param out_data array of output data,
   *        the space of TBlob in out_data must be pre-allocated with InferShape
   */
  virtual void Forward(Context ctx,
                       const std::vector<mshadow::TBlob> &in_data,
                       const std::vector<mshadow::TBlob> &out_data) = 0;
  /*!
   * \brief perform a backward operation of the operator to get the gradient
   * \param grad_next the gradient value we get from output of the operator
   * \param in_data the array of input data
   * \param out_grad array of output gradient, there could be three possible TBlob
   *  in the each element in the array
   * \param req_types request types of the gradient saving operation
   *
   *  out_grad[i].dptr_ == NULL: means we do not need the gradient
   *  out_grad[i].dptr_ == in_data[i].dptr_: means the gradient need can be saved inplace
   *  out_grad[i].dptr_ != in_data[i].dptr_: means the gradient space is newly allocated
   */
  virtual void Backward(const std::vector<mshadow::TBlob> &grad_next,
                        const std::vector<mshadow::TBlob> &in_data,
                        const std::vector<mshadow::TBlob> &out_grad,
                        const std::vector<int> req);
};
}  // namespace mxnet
#endif  // MXNET_OPERATOR_H_
