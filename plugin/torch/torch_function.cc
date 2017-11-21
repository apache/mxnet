/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * Copyright (c) 2016 by Contributors
 * \file torch_base.cc
 * \brief torch_state
 * \author Junyuan Xie
*/
#include "./torch_function.h"

namespace mxnet {

// Construction or extraction functions
MXNET_REGISTER_TORCH_CONSTRUCTOR_FUN(_th_eye, eye);
MXNET_REGISTER_TORCH_CONSTRUCTOR_FUN(_th_ones, ones);
MXNET_REGISTER_TORCH_CONSTRUCTOR_FUN(_th_rand, rand);
MXNET_REGISTER_TORCH_CONSTRUCTOR_FUN(_th_randn, randn);
MXNET_REGISTER_TORCH_CONSTRUCTOR_FUN(_th_randperm, randperm);
MXNET_REGISTER_TORCH_UNARY_FUN(_th_tril, tril);
MXNET_REGISTER_TORCH_UNARY_FUN(_th_triu, triu);
MXNET_REGISTER_TORCH_CONSTRUCTOR_FUN(_th_zeros, zeros);

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
MXNET_REGISTER_TORCH_UNARY_FUN(_th_pow, pow)
.add_argument("n", "float", "pow(x, n) returns x^n, element-wise. "
  "pow(n, x) returns n^x, element-wise.");
MXNET_REGISTER_TORCH_UNARY_FUN(_th_round, round);
MXNET_REGISTER_TORCH_UNARY_FUN(_th_sin, sin);
MXNET_REGISTER_TORCH_UNARY_FUN(_th_sinh, sinh);
MXNET_REGISTER_TORCH_UNARY_FUN(_th_sqrt, sqrt);
MXNET_REGISTER_TORCH_UNARY_FUN(_th_tan, tan);
MXNET_REGISTER_TORCH_UNARY_FUN(_th_tanh, tanh);

// Basic operations
MXNET_REGISTER_TORCH_UNARY_FUN(_th_add_scalar, add)
.add_argument("value", "float", "Add value to all elements in x");
MXNET_REGISTER_TORCH_BINARY_FUN_WITH_ARG(_th_add, add);
MXNET_REGISTER_TORCH_BINARY_FUN(_th_add_axpy, add);

// MXNET_REGISTER_TORCH_UNARY_FUN(_th_csub_scalar, csub);
// MXNET_REGISTER_TORCH_BINARY_FUN_WITH_ARG(_th_csub, csub);

MXNET_REGISTER_TORCH_UNARY_FUN(_th_mul_scalar, mul)
.add_argument("value", "float", "Multiply value to all elements in x");
MXNET_REGISTER_TORCH_BINARY_FUN_WITH_ARG(_th_cmul, cmul);

MXNET_REGISTER_TORCH_UNARY_FUN(_th_clamp, clamp);
MXNET_REGISTER_TORCH_BINARY_FUN_WITH_ARG(_th_cpow, cpow);
MXNET_REGISTER_TORCH_TENARY_FUN(_th_addcmul, addcmul);

MXNET_REGISTER_TORCH_UNARY_FUN(_th_div_scalar, div)
.add_argument("value", "float", "Divide all elements in x by value");
MXNET_REGISTER_TORCH_BINARY_FUN_WITH_ARG(_th_cdiv, cdiv);
MXNET_REGISTER_TORCH_TENARY_FUN(_th_addcdiv, addcdiv);

MXNET_REGISTER_TORCH_TENARY_FUN(_th_addmv, addmv);
MXNET_REGISTER_TORCH_TENARY_FUN(_th_addr, addr);
MXNET_REGISTER_TORCH_TENARY_FUN(_th_addmm, addmm);
MXNET_REGISTER_TORCH_TENARY_FUN(_th_addbmm, addbmm);
MXNET_REGISTER_TORCH_TENARY_FUN(_th_baddbmm, baddbmm);

struct TorchMMShape {
  static std::vector<mshadow::TShape> GetShape(NDArray **u,
    const std::map<std::string, std::string>& param) {
    CHECK_EQ(u[0]->shape().ndim(), 2);
    CHECK_EQ(u[1]->shape().ndim(), 2);
    CHECK_EQ(u[0]->shape()[1], u[1]->shape()[0]);
    index_t shape[] = {u[0]->shape()[0], u[1]->shape()[1]};
    mshadow::TShape tshape(shape, shape+2);
    return {tshape};
  }
  static constexpr const char* fname = "mm";
  static const int num_inputs = 2;
  static const int num_outputs = 1;
};
MXNET_REGISTER_TORCH_FUN(_th_mm, TorchMMShape);

struct TorchMVShape {
  static std::vector<mshadow::TShape> GetShape(NDArray **u,
    const std::map<std::string, std::string>& param) {
    CHECK_EQ(u[0]->shape().ndim(), 2);
    CHECK_EQ(u[1]->shape().ndim(), 1);
    CHECK_EQ(u[0]->shape()[1], u[1]->shape()[0]);
    index_t shape[] = {u[0]->shape()[0]};
    mshadow::TShape tshape(shape, shape+1);
    return {tshape};
  }
  static constexpr const char* fname = "mv";
  static const int num_inputs = 2;
  static const int num_outputs = 1;
};
MXNET_REGISTER_TORCH_FUN(_th_mv, TorchMVShape);


struct TorchBMMShape {
  static std::vector<mshadow::TShape> GetShape(NDArray **u,
    const std::map<std::string, std::string>& param) {
    CHECK_EQ(u[0]->shape().ndim(), 3);
    CHECK_EQ(u[1]->shape().ndim(), 3);
    CHECK_EQ(u[0]->shape()[0], u[1]->shape()[0]);
    CHECK_EQ(u[0]->shape()[2], u[1]->shape()[1]);
    index_t shape[] = {u[0]->shape()[1], u[1]->shape()[2]};
    mshadow::TShape tshape(shape, shape+2);
    return {tshape};
  }
  static constexpr const char* fname = "bmm";
  static const int num_inputs = 2;
  static const int num_outputs = 1;
};
MXNET_REGISTER_TORCH_FUN(_th_bmm, TorchBMMShape);

struct TorchGERShape {
  static std::vector<mshadow::TShape> GetShape(NDArray **u,
    const std::map<std::string, std::string>& param) {
    CHECK_EQ(u[0]->shape().ndim(), 1);
    CHECK_EQ(u[1]->shape().ndim(), 1);
    index_t shape[] = {u[0]->shape()[0], u[1]->shape()[0]};
    mshadow::TShape tshape(shape, shape+2);
    return {tshape};
  }
  static constexpr const char* fname = "ger";
  static const int num_inputs = 2;
  static const int num_outputs = 1;
};
MXNET_REGISTER_TORCH_FUN(_th_ger, TorchGERShape);

}  // namespace mxnet
