/*!
 *  Copyright (c) 2015 by Contributors
 * \file operator_util.h
 * \brief Utility functions and registries to help quickly build new operators.
 *
 *  Use the register functions in this file when possible to simplify operator creations.
 *  Operators registred in this file will be exposed to both NDArray API and symbolic API.
 *
 * \author Tianqi Chen
 */
#ifndef MXNET_OPERATOR_UTIL_H_
#define MXNET_OPERATOR_UTIL_H_

#include <dmlc/registry.h>
#include <dmlc/parameter.h>
#include <map>
#include <vector>
#include <string>
#include <utility>
#include "./base.h"
#include "./operator.h"

#if DMLC_USE_CXX11
#include <functional>
#endif

namespace mxnet {
/*! \brief namespace of arguments */
namespace op {
/*! \brief super class of all gradient function argument */
struct GradFunctionArgument {
  /*! \brief The real data */
  TBlob data;
};

/*! \brief First input to the function */
struct Input0 : GradFunctionArgument {};
/*! \brief Second input to the function */
struct Input1 : GradFunctionArgument {};

/*! \brief Ouput value of the function to the function */
struct OutputValue : GradFunctionArgument {};
/*! \brief Gradient of output value */
struct OutputGrad : GradFunctionArgument {};

/*!
 * \brief Environment arguments that is used by the function.
 * These can be things like scalar arguments when add a value with scalar.
 */
struct EnvArguments {
  /*! \brief scalar argument, if enabled */
  real_t scalar;
  /*! \brief keyword arguments */
  std::vector<std::pair<std::string, std::string> > kwargs;
  /*! \brief pointer to the resources requested */
  std::vector<Resource> resource;
};

/*!
 * \brief Unary function that takes a src and save result to ret.
 *  The result container is pre-allocated with the correct shape.
 * \param src The source data.
 * \param env The Environment arguments.
 * \param ret The containter to store return value.
 * \param req The requirement to stroe the ret.
 * \param ctx Runtime context to execute the function.
 */
typedef void (*UnaryFunction)(const TBlob& src,
                              const EnvArguments& env,
                              TBlob* ret,
                              OpReqType req,
                              RunContext ctx);
/*!
 * \brief Shape inference function to get the correct shape given source.
 * \param src The source shape
 * \param env The Environment arguments.
 * \return The inferred result shape.
 */
typedef TShape (*UnaryShapeFunction)(const TShape& src,
                                     const EnvArguments& env);

/*!
 * \brief Gradient function that takes output value of function and computes gradient wrt to input.
 * \param out_grad the gradient wrt to output of the function.
 * \param env The Environment arguments.
 * \param in_grad The container to store result input gradient.
 * \param req The requirement to store the ret value.
 * \param ctx Runtime context to execute the function.
 */
typedef void (*UnaryGradFunctionT0)(const OutputGrad& out_grad,
                                    const EnvArguments& env,
                                    TBlob* in_grad,
                                    OpReqType req,
                                    RunContext ctx);
/*!
 * \brief Gradient function that takes output value of function and computes gradient wrt to input.
 * \param out_grad the gradient wrt to output of the function.
 * \param out_value the value of the function.
 * \param env The Environment arguments.
 * \param in_grad The container to store result input gradient.
 * \param req The requirement to store the ret value.
 * \param ctx Runtime context to execute the function.
 */
typedef void (*UnaryGradFunctionT1)(const OutputGrad& out_grad,
                                    const OutputValue& out_value,
                                    const EnvArguments& env,
                                    TBlob* in_grad,
                                    OpReqType req,
                                    RunContext ctx);
/*!
 * \brief Gradient function that takes input value of function and computes gradient wrt to input.
 * \param out_grad the gradient wrt to output of the function.
 * \param in_data0 the input value of the function.
 * \param env The Environment arguments.
 * \param in_grad The container to store result input gradient.
 * \param req The requirement to store the ret value.
 * \param ctx Runtime context to execute the function.
 */
typedef void (*UnaryGradFunctionT2)(const OutputGrad& out_grad,
                                    const Input0& in_data0,
                                    const EnvArguments& env,
                                    TBlob* in_grad,
                                    OpReqType req,
                                    RunContext ctx);
/*!
 * \brief Binary function that takes lhs, rhs and save result to ret.
 *  The result container is pre-allocated with the correct shape.
 * \param lhs The left operand
 * \param rhs The right operand
 * \param env The Environment arguments.
 * \param ret The containter to store return value.
 * \param req The requirement to stroe the ret.
 * \param ctx Runtime context to execute the function.
 */
typedef void (*BinaryFunction)(const TBlob& lhs,
                               const TBlob& rhs,
                               const EnvArguments& env,
                               TBlob* ret,
                               OpReqType req,
                               RunContext ctx);

/*!
 * \brief Shape inference function to get the correct shape given source shapes.
 * \param lhs The shape of left operand.
 * \param rhs The shape of right operand.
 * \param env The Environment arguments.
 * \return The inferred result shape.
 */
typedef TShape (*BinaryShapeFunction)(const TShape& lhs,
                                      const TShape& rhs,
                                      const EnvArguments& env);
/*!
 * \brief Gradient function that takes only output gradient and computes gradient wrt to input.
 *  We support total gradient as a whole to make it easy to combine a few ops.
 * \param out_grad the gradient wrt to output of the function.
 * \param env The Environment arguments.
 * \param lhs_grad The container to store result of lhs gradient.
 * \param rhs_grad The container to store result of lhs gradient.
 * \param req_lhs_grad The requirement to store the lhs_grad
 * \param req_rhs_grad The requirement to store the rhs_grad
 * \param ctx Runtime context to execute the function.
 */
typedef void (*BinaryGradFunctionT0)(const OutputGrad& out_grad,
                                     const EnvArguments& env,
                                     TBlob* lhs_grad,
                                     TBlob* rhs_grad,
                                     OpReqType req_lhs_grad,
                                     OpReqType req_rhs_grad,
                                     RunContext ctx);
/*!
 * \brief Gradient function that takes inputs of function anod computes gradient wrt to input.
 * \param out_grad the gradient wrt to output of the function.
 * \param lhs The left operand to the function.
 * \param rhs The right operand to the function.
 * \param env The Environment arguments.
 * \param lhs_grad The container to store result of lhs gradient.
 * \param rhs_grad The container to store result of lhs gradient.
 * \param req_lhs_grad The requirement to store the lhs_grad
 * \param req_rhs_grad The requirement to store the rhs_grad
 * \param ctx Runtime context to execute the function.
 */
typedef void (*BinaryGradFunctionT1)(const OutputGrad& out_grad,
                                     const Input0& lhs,
                                     const Input1& rhs,
                                     const EnvArguments& env,
                                     TBlob* lhs_grad,
                                     TBlob* rhs_grad,
                                     OpReqType req_lhs_grad,
                                     OpReqType req_rhs_grad,
                                     RunContext ctx);

/*! \brief options in the registry to set inplace of operator */
enum SimpleOpInplaceOption {
  /*! \brief do not allow inplace in arguments */
  kNoInplace,
  /*! \brief in unary forward, allow inplace in with out */
  kInplaceInOut,
  /*! \brief in unary backward, allow inplace out_grad with in_grad */
  kInplaceOutIn,
  /*! \brief in binary forward, allow inplace left operand with out */
  kInplaceLhsOut,
  /*! \brief in binary backward, allow inplace out_grad with lhs_grad */
  kInplaceOutLhs
};

/*! \brief options in the registry to set symbolic registration */
enum SimpleOpScalarOption {
  kScalarBeforeArray,
  kArrayBeforeScalar
};

/*! \brief options in the registry to set symbolic registration */
enum SimpleOpRegOption {
  kNotRegisterSymbolic,
  kRegisterSymbolic
};

/*! \brief registry entry to register simple operators via functions. */
class SimpleOpRegEntry {
 public:
  /*! \brief declare self type */
  typedef SimpleOpRegEntry TSelf;
  /*! \brief name of the operator */
  std::string name;
  /*!
   * \brief set a seperate name for symbol
   *  This must be called before set_function.
   *  Default: this is set to be same as the name of operator.
   * \param symbol_name the name of symbolic operator.
   */
  virtual TSelf& set_symbol_op_name(char const* symbol_name) = 0;
  /*!
   * \brief set number of scalar arguments needed to be passed in env
   *  A function cannot have both kwargs and scalar arguments.
   *  Default: this is set to false
   * \param enable_scalar whether to enable scalar argument
   * \param type_mask the position of the scalar argument.
   */
  virtual TSelf& set_enable_scalar(
      bool enable_scalar,
      SimpleOpScalarOption type_mask = kArrayBeforeScalar) = 0;
  /*!
   * \brief set whether to enable kwargs
   *  A function cannot have both kwargs and scalar arguments.
   *  Default: this is set to false
   * \param enable_kwargs whether to enable kwargs
   */
  virtual TSelf& set_enable_kwargs(bool enable_kwargs) = 0;
  /*!
   * \brief set resource request
   *  By default there is no resource request.
   *  The resource will be presented in both forward and backward.
   * \param reqs the request.
   */
  virtual TSelf& set_resource_request(
      const std::vector<ResourceRequest>& reqs) = 0;
  /*!
   * \brief set resource request
   *  By default there is no resource request.
   *  The resource will be presented in both forward and backward.
   * \param req the request.
   */
  virtual TSelf& set_resource_request(ResourceRequest req) = 0;
  /*!
   * \brief set shape inference function.
   *  Default: out_shape = in_shape
   * \param fshapeinfer The unary function that peforms the operation.
   */
  virtual TSelf& set_shape_function(UnaryShapeFunction fshapeinfer) = 0;
  /*!
   * \brief set shape inference function to be the binary inference function
   *  Default: out_shape = lhs_shape, and lhs_shape must equal rhs_shape.
   * \param fshapeinfer The binary function that peforms the operation.
   */
  virtual TSelf& set_shape_function(BinaryShapeFunction fshapeinfer) = 0;
  /*!
   * \brief set function of the function to be funary
   * \param dev_mask The device mask of the function can act on.
   * \param funary The unary function that peforms the operation.
   * \param inplace_in_out Whether do inplace optimization on in and out.
   * \param register_symbolic Whether register a symbolic operator as well.
   */
  virtual TSelf& set_function(
      int dev_mask,
      UnaryFunction funary,
      SimpleOpInplaceOption inplace_in_out,
      SimpleOpRegOption register_symbolic = kRegisterSymbolic) = 0;
  /*!
   * \brief set function of the function to be funary
   * \param dev_mask The device mask of the function can act on.
   * \param fbinary The binary function that peforms the operation.
   * \param inplace_lhs_out Whether do inplace optimization on lhs and out.
   * \param register_symbolic Whether register a symbolic operator as well.
   */
  virtual TSelf& set_function(
      int dev_mask,
      BinaryFunction fbinary,
      SimpleOpInplaceOption inplace_lhs_out,
      SimpleOpRegOption register_symbolic = kRegisterSymbolic) = 0;
  /*!
   * \brief set gradient of the function of this function.
   * \param dev_mask The device mask of the function can act on.
   * \param fgrad The gradient function to be set.
   * \param inplace_out_in_grad whether out_grad and in_grad can share memory.
   */
  virtual TSelf& set_gradient(int dev_mask,
                              UnaryGradFunctionT0 fgrad,
                              SimpleOpInplaceOption inplace_out_in_grad) = 0;
  /*!
   * \brief set gradient of the function of this function.
   * \param dev_mask The device mask of the function can act on.
   * \param fgrad The gradient function to be set.
   * \param inplace_out_in_grad whether out_grad and in_grad can share memory.
   */
  virtual TSelf& set_gradient(int dev_mask,
                              UnaryGradFunctionT1 fgrad,
                              SimpleOpInplaceOption inplace_out_in_grad) = 0;
  /*!
   * \brief set gradient of the function of this function.
   * \param dev_mask The device mask of the function can act on.
   * \param fgrad The gradient function to be set.
   * \param inplace_out_in_grad whether out_grad and in_grad can share memory.
   */
  virtual TSelf& set_gradient(int dev_mask,
                              UnaryGradFunctionT2 fgrad,
                              SimpleOpInplaceOption inplace_out_in_grad) = 0;
  /*!
   * \brief set gradient of the function of this function.
   * \param dev_mask The device mask of the function can act on.
   * \param fgrad The gradient function to be set.
   * \param inplace_out_lhs_grad whether out_grad and lhs_grad can share memory.
   */
  virtual TSelf& set_gradient(int dev_mask,
                              BinaryGradFunctionT0 fgrad,
                              SimpleOpInplaceOption inplace_out_lhs_grad) = 0;
  /*!
   * \brief set gradient of the function of this function.
   * \param dev_mask The device mask of the function can act on.
   * \param fgrad The gradient function to be set.
   * \param inplace_out_lhs_grad whether out_grad and lhs_grad can share memory.
   */
  virtual TSelf& set_gradient(int dev_mask,
                              BinaryGradFunctionT1 fgrad,
                              SimpleOpInplaceOption inplace_out_lhs_grad) = 0;
  /*!
   * \brief Describe the function.
   * \param description The description of the function.
   * \return reference to self.
   */
  virtual TSelf& describe(const std::string &description) = 0;
  /*!
   * \brief Describe the function.
   * \param args argument information.
   *  Add addtional arguments to the function.
   * \return reference to self.
   */
  virtual TSelf& add_arguments(const std::vector<dmlc::ParamFieldInfo> &args) = 0;
  /*! \brief virtual destructor */
  virtual ~SimpleOpRegEntry() {}
};

/*! \brief registry for TBlob functions */
class SimpleOpRegistry {
 public:
  /*!
   * \brief Internal function to register a name function under name.
   * \param name name of the function
   * \return ref to the registered entry, used to set properties
   */
  SimpleOpRegEntry &__REGISTER_OR_FIND__(char const* name);
  /*!
   * \brief Find the entry with corresponding name.
   * \param name name of the function
   * \return the corresponding function, can be NULL
   */
  inline static const SimpleOpRegEntry *Find(const std::string &name) {
    return Get()->fmap_.at(name);
  }
  /*! \return global singleton of the registry */
  static SimpleOpRegistry* Get();

 private:
  // destructor
  ~SimpleOpRegistry();
  /*! \brief internal registry map */
  std::map<std::string, SimpleOpRegEntry*> fmap_;
};

/*!
 * \brief assign the expression to out according to request
 * \param out the data to be assigned
 * \param req the assignment request
 * \param exp the expression
 * \tparam OType output type
 * \tparam Exp expression type
 */
#define ASSIGN_DISPATCH(out, req, exp)  \
  {                                     \
    switch (req) {                      \
      case kNullOp:                     \
        break;                          \
      case kWriteTo:                    \
      case kWriteInplace:               \
        (out) = (exp);                  \
        break;                          \
      case kAddTo:                      \
        (out) += (exp);                 \
        break;                          \
      default:                          \
        LOG(FATAL) << "not reached";    \
    }                                   \
  }

//--------------------------------------------------------------
// The following part are API Registration of Simple Operators
//--------------------------------------------------------------
/*!
 * \brief Macro to register simple operator to both imperative and symbolic API.
 *
 * see src/operator/elementwise_unary_op-inl.h for example
 *
 * \code
 * // example of registering a sigmoid operator on GPU
 * // MySigmoid is of type UnaryFunction,
 * // MySigmoidGrad is of type UnaryGradFunctionT2
 *
 * MXNET_REGISTER_SIMPLE_OP(sigmoid, cpu)
 * .set_function(MySigmoid<gpu>, true)
 * .set_gradient(MySigmoidGrad<gpu>, true)
 * .describe("Sigmoid function");
 *
 * \endcode
 */
#define MXNET_REGISTER_SIMPLE_OP(Name, DEV)                             \
  static ::mxnet::op::SimpleOpRegEntry &                                \
  __make_ ## SimpleOpRegEntry ## _ ## Name ## __ ## DEV ##__ =          \
      ::mxnet::op::SimpleOpRegistry::Get()->__REGISTER_OR_FIND__(#Name)

}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_UTIL_H_
