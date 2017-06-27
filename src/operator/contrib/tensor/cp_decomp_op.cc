/*!
 *  Copyright (c) 2016 by Contributors
 * \file cp_decomp_op.cc
 * \brief Register CPU implementation of CP Decomposition
 * \author Jencir Lee
 */
#include "./cp_decomp_op.h"

namespace mxnet {
namespace op {

#define FLOAT_DOUBLE_SWITCH(type, DType, ...)       \
  switch (type) {                                   \
  case mshadow::kFloat32:                           \
    {                                               \
      typedef float DType;                          \
      {__VA_ARGS__}                                 \
    }                                               \
    break;                                          \
  case mshadow::kFloat64:                           \
    {                                               \
      typedef double DType;                         \
      {__VA_ARGS__}                                 \
    }                                               \
    break;                                          \
  default:                                          \
    LOG(FATAL) << "Unknown type enum " << type;     \
  }

// DO_BIND_DISPATCH comes from operator_common.h
template <int order>
Operator *CPDecompProp<order>::CreateOperatorEx
  (Context ctx,
  std::vector<TShape> *in_shape,
  std::vector<int> *in_type) const {
  std::vector<TShape> out_shape, aux_shape;
  std::vector<int> out_type, aux_type;
  CHECK(InferType(in_type, &out_type, &aux_type));
  CHECK(InferShape(in_shape, &out_shape, &aux_shape));

  if (ctx.dev_mask() != cpu::kDevMask) {
    LOG(FATAL) << "GPU is not enabled";
    return nullptr;
  }

  Operator *op = nullptr;
  FLOAT_DOUBLE_SWITCH(in_type->at(0), DType,
    { op = new CPDecompOp<cpu, order, DType>(param_); });
  return op;
}

DMLC_REGISTER_PARAMETER(CPDecompParam);

MXNET_REGISTER_OP_PROPERTY(CPDecomp3D, CPDecompProp<3>)
.describe(R"code(Performs CANDECOMP/PARAFAC Decomposition on 3D tensors

Examples::

  r = CPDecomp3D(data=t, k=5)

r[0] is the eigen-value vector, r[1] .. r[3] are the transposed factor matrices.

Internally it uses an iterative algorithm with random initial matrices and may not converge to the same solution from run to run.)code")
.add_argument("data", "Symbol", "Input tensor to CPDecomp3D.")
.add_arguments(CPDecompParam::__FIELDS__());

MXNET_REGISTER_OP_PROPERTY(CPDecomp4D, CPDecompProp<4>)
.describe(R"code(Performs CANDECOMP/PARAFAC Decomposition on 4D tensors

Examples::

  r = CPDecomp4D(data=t, k=5)

r[0] is the eigen-value vector, r[1] .. r[4] are the transposed factor matrices.

Internally it uses an iterative algorithm with random initial matrices and may not converge to the same solution from run to run.)code")
.add_argument("data", "Symbol", "Input tensor to CPDecomp4D.")
.add_arguments(CPDecompParam::__FIELDS__());

MXNET_REGISTER_OP_PROPERTY(CPDecomp5D, CPDecompProp<5>)
.describe(R"code(Performs CANDECOMP/PARAFAC Decomposition on 5D tensors

Examples::

  r = CPDecomp5D(data=t, k=5)

r[0] is the eigen-value vector, r[1] .. r[5] are the transposed factor matrices.

Internally it uses an iterative algorithm with random initial matrices and may not converge to the same solution from run to run.)code")
.add_argument("data", "Symbol", "Input tensor to CPDecomp5D.")
.add_arguments(CPDecompParam::__FIELDS__());

MXNET_REGISTER_OP_PROPERTY(CPDecomp6D, CPDecompProp<6>)
.describe(R"code(Performs CANDECOMP/PARAFAC Decomposition on 6D tensors

Examples::

  r = CPDecomp6D(data=t, k=5)

r[0] is the eigen-value vector, r[1] .. r[6] are the transposed factor matrices.

Internally it uses an iterative algorithm with random initial matrices and may not converge to the same solution from run to run.)code")
.add_argument("data", "Symbol", "Input tensor to CPDecomp6D.")
.add_arguments(CPDecompParam::__FIELDS__());

}  // namespace op
}  // namespace mxnet
