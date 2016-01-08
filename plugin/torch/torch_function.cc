/*!
 * Copyright (c) 2016 by Contributors
 * \file torch_base.cc
 * \brief torch_state
 * \author Junyuan Xie
*/
#include "./torch_function.h"

namespace mxnet {

// Element-wise Mathematical Operations
MXNET_REGISTER_TORCH_UNARY_FUN(_th_abs, abs);
MXNET_REGISTER_TORCH_UNARY_FUN(_th_sign, sign);
MXNET_REGISTER_TORCH_UNARY_FUN(_th_acos, acos);
MXNET_REGISTER_TORCH_UNARY_FUN(_th_asin, asin);
MXNET_REGISTER_TORCH_UNARY_FUN(_th_atan, atan);
MXNET_REGISTER_TORCH_UNARY_FUN(_th_ceil, ceil);
MXNET_REGISTER_TORCH_UNARY_FUN(_th_cos, cos);
MXNET_REGISTER_TORCH_UNARY_FUN(_th_cosh, cosh);
MXNET_REGISTER_TORCH_UNARY_FUN(_th_exp, exp);
MXNET_REGISTER_TORCH_UNARY_FUN(_th_floor, floor);
MXNET_REGISTER_TORCH_UNARY_FUN(_th_log, log);
MXNET_REGISTER_TORCH_UNARY_FUN(_th_log1p, log1p);
MXNET_REGISTER_TORCH_UNARY_FUN(_th_pow, pow);
MXNET_REGISTER_TORCH_UNARY_FUN(_th_round, round);
MXNET_REGISTER_TORCH_UNARY_FUN(_th_sin, sin);
MXNET_REGISTER_TORCH_UNARY_FUN(_th_sinh, sinh);
MXNET_REGISTER_TORCH_UNARY_FUN(_th_sqrt, sqrt);
MXNET_REGISTER_TORCH_UNARY_FUN(_th_tan, tan);
MXNET_REGISTER_TORCH_UNARY_FUN(_th_tanh, tanh);

// Basic operations
MXNET_REGISTER_TORCH_UNARY_FUN(_th_add_scalar, add);


}  // namespace mxnet
