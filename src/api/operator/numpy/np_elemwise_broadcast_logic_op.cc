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
 * \file np_elemwise_broadcast_logic_op.cc
 * \brief Implementation of the API of functions in src/operator/numpy/np_elemwise_broadcast_logic_op.cc
 */
#include <mxnet/api_registry.h>
#include <mxnet/runtime/packed_func.h>
#include "../utils.h"
#include "../ufunc_helper.h"

namespace mxnet {

MXNET_REGISTER_API("_npi.equal")
.set_body([](runtime::MXNetArgs args, runtime::MXNetRetValue* ret) {
  using namespace runtime;
  const nnvm::Op* op = Op::Get("_npi_equal");
  const nnvm::Op* op_scalar = Op::Get("_npi_equal_scalar");
  UFuncHelper(args, ret, op, op_scalar, nullptr);
});

MXNET_REGISTER_API("_npi.not_equal")
.set_body([](runtime::MXNetArgs args, runtime::MXNetRetValue* ret) {
  using namespace runtime;
  const nnvm::Op* op = Op::Get("_npi_not_equal");
  const nnvm::Op* op_scalar = Op::Get("_npi_not_equal_scalar");
  UFuncHelper(args, ret, op, op_scalar, nullptr);
});

void set_UFuncHelper(runtime::MXNetArgs args, runtime::MXNetRetValue* ret,
                     const nnvm::Op* op, const nnvm::Op* op_scalar,
                     const nnvm::Op* op_rscalar) {
  int result = 0;
  if (args[1].type_code() == kNDArrayHandle) {
    result++;
    result <<= 1;
  }
  if (args[0].type_code() == kNDArrayHandle) {
    result++;
  }

  switch(result) {
    case 0 :
      UFuncHelper(args, ret, op, nullptr, nullptr);
      break;
    case 1 :
      UFuncHelper(args, ret, nullptr, op_scalar, nullptr);
      break;
    case 2 :
      UFuncHelper(args, ret, nullptr, nullptr, op_rscalar);
      break;
  }
}

MXNET_REGISTER_API("_npi.less")
.set_body([](runtime::MXNetArgs args, runtime::MXNetRetValue* ret) {
  using namespace runtime;
  const nnvm::Op* op = Op::Get("_npi_less");
  const nnvm::Op* op_scalar = Op::Get("_npi_less_scalar");
  const nnvm::Op* op_rscalar = Op::Get("_npi_greater_scalar");
  set_UFuncHelper(args, ret, op, op_scalar, op_rscalar);
});

MXNET_REGISTER_API("_npi.greater_equal")
.set_body([](runtime::MXNetArgs args, runtime::MXNetRetValue* ret) {
  using namespace runtime;
  const nnvm::Op* op = Op::Get("_npi_greater_equal");
  const nnvm::Op* op_scalar = Op::Get("_npi_greater_equal_scalar");
  const nnvm::Op* op_rscalar = Op::Get("_npi_less_equal_scalar");
  set_UFuncHelper(args, ret, op, op_scalar, op_rscalar);
});

MXNET_REGISTER_API("_npi.less_equal")
.set_body([](runtime::MXNetArgs args, runtime::MXNetRetValue* ret) {
  using namespace runtime;
  const nnvm::Op* op = Op::Get("_npi_less_equal");
  const nnvm::Op* op_scalar = Op::Get("_npi_less_equal_scalar");
  const nnvm::Op* op_rscalar = Op::Get("_npi_greater_equal_scalar");
  set_UFuncHelper(args, ret, op, op_scalar, op_rscalar);
});

}  // namespace mxnet
