/*!
 *  Copyright (c) 2015 by Contributors
 * \file operator.h
 * \brief static operator interface of mxnet
 */
#ifndef MXNET_OPERATOR_H_
#define MXNET_OPERATOR_H_
// this file will be seen by cuda, no c++11 for now
#include <dmlc/base.h>
#include "./base.h"
#include "./tensor_blob.h"

namespace mxnet {
/*!
 * \brief static operator interface (current interface have not yet todo with scheduler),
 *  operator is a stateful object that can be used to call forward and backprop
 *
 *  This interface relies on pre-allocated memory in TBlob, the caller need to set
 *  the memory region in TBlob correctly before calling Forward and Backward
 *
 * \sa TBlob, TShape
 */
class Operator {
 public:
  /*! \brief option to pass into the forward function */
  struct Option {
    /*! \brief whether it is training phase*/
    int is_train;
  };
  /*! \briref gradient request type the request can have */
  enum GradReqType {
    /*! \brief no operation, do not write gradient */
    kNullOp = 0,
    /*! \brief write gradient to provided space */
    kWriteTo = 1,
    /*! \brief same as kWriteTo, but provided space is same as space of input-data */
    kWriteInplace = 2,
    /*! \brief add to the provided space */
    kAddTo = 3
  };
  /*! \brief input argument type of the operator have */
  enum ArgType {
    /*! \brief data argument */
    kDataArg = 0,
    /*! \brief weight argument */
    kWeightArg = 1,
    /*! \brief bias argument */
    kBiasArg = 2
  };
  /*!
   * \brief get types of input argument of this oeprator
   * \return a vector corresponding to type of each argument
   *  this order is same as the order of inputs in Forward, InferShape and Backward
   */
  virtual std::vector<ArgType> DescribeArgs() const {
    // default most of layers only have one data argument
    return std::vector<ArgType>(1, kDataArg);
  }
  /*!
   * \brief set param for the operator from string
   * \param name parameter name
   * \param val string for configuration
   */
  virtual void SetParam(const char *name, const char *val) {}
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
                          std::vector<TShape> *out_shape) = 0;
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
                       const std::vector<TBlob> &in_data,
                       const std::vector<TBlob> &out_data) = 0;
  /*!
   * \brief perform a backward operation of the operator to get the gradient
   * \param ctx runtime context
   * \param grad_next the gradient value we get from output of the operator
   * \param in_data the array of input data
   * \param out_grad array of output gradient, there could be three possible TBlob
   *  in the each element in the array
   * \param req_types request types of the gradient saving operation
   *                  only inplace will change input data
   * \sa GradReqType
   */
  virtual void Backward(RunContext ctx,
                        const std::vector<TBlob> &grad_next,
                        const std::vector<TBlob> &in_data,
                        const std::vector<TBlob> &out_grad,
                        const std::vector<GradReqType> &req);

  /*!
   * \brief factory unction, create a new operator
   * \param type the type of operator
   * \param ctx the context device type of operator
   */
  static Operator *Create(const char *type, Context ctx);
};
}  // namespace mxnet
#endif  // MXNET_OPERATOR_H_
