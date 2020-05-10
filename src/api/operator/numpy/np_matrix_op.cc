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
 * \file np_matrix_op.cc
 * \brief Implementation of the API of functions in src/operator/numpy/np_matrix_op.cc
 */
#include <mxnet/api_registry.h>
#include <mxnet/runtime/packed_func.h>
#include <vector>
#include "../utils.h"
#include "../../../operator/nn/concat-inl.h"
#include "../../../operator/tensor/matrix_op-inl.h"
#include "../../../operator/numpy/np_matrix_op-inl.h"

namespace mxnet {

MXNET_REGISTER_API("_npi.transpose")
.set_body([](runtime::MXNetArgs args, runtime::MXNetRetValue* ret) {
  using namespace runtime;
  static const nnvm::Op* op = Op::Get("_npi_transpose");
  nnvm::NodeAttrs attrs;
  op::NumpyTransposeParam param;
  if (args[1].type_code() == kNull) {
    param.axes = TShape(-1, 0);
  } else if (args[1].type_code() == kDLInt) {
    param.axes = TShape(1, args[1].operator int64_t());
  } else {
    param.axes = TShape(args[1].operator ObjectRef());
  }
  attrs.parsed = std::move(param);
  attrs.op = op;
  SetAttrDict<op::NumpyTransposeParam>(&attrs);
  NDArray* inputs[] = {args[0].operator mxnet::NDArray*()};
  int num_inputs = 1;
  int num_outputs = 0;
  auto ndoutputs = Invoke(op, &attrs, num_inputs, inputs, &num_outputs, nullptr);
  *ret = ndoutputs[0];
});

MXNET_REGISTER_API("_npi.expand_dims")
.set_body([](runtime::MXNetArgs args, runtime::MXNetRetValue* ret) {
  using namespace runtime;
  const nnvm::Op* op = Op::Get("_npi_expand_dims");
  nnvm::NodeAttrs attrs;
  op::ExpandDimParam param;
  param.axis = args[1].operator int();

  // we directly copy ExpandDimParam, which is trivially-copyable
  attrs.parsed = param;
  attrs.op = op;
  SetAttrDict<op::ExpandDimParam>(&attrs);

  int num_outputs = 0;
  NDArray* inputs[] = {args[0].operator mxnet::NDArray*()};
  int num_inputs = 1;
  auto ndoutputs = Invoke(op, &attrs, num_inputs, inputs, &num_outputs, nullptr);
  *ret = ndoutputs[0];
});

MXNET_REGISTER_API("_npi.concatenate")
.set_body([](runtime::MXNetArgs args, runtime::MXNetRetValue* ret) {
  using namespace runtime;
  const nnvm::Op* op = Op::Get("_npi_concatenate");
  nnvm::NodeAttrs attrs;
  op::NumpyConcatenateParam param;
  int arg_size = args.num_args;
  param.num_args = arg_size - 2;
  if (args[arg_size - 2].type_code() == kNull) {
    param.axis = dmlc::nullopt;
  } else {
    param.axis = args[arg_size - 2].operator int();
  }
  attrs.parsed = std::move(param);
  attrs.op = op;
  SetAttrDict<op::NumpyConcatenateParam>(&attrs);
  int num_inputs = arg_size - 2;
  std::vector<NDArray*> inputs;
  for (int i = 0; i < num_inputs; ++i) {
    inputs.push_back(args[i].operator mxnet::NDArray*());
  }
  NDArray* out = args[arg_size - 1].operator mxnet::NDArray*();
  NDArray** outputs = out == nullptr ? nullptr : &out;
  int num_outputs = out != nullptr;
  auto ndoutputs = Invoke(op, &attrs, num_inputs, inputs.data(), &num_outputs, outputs);
  if (out) {
    *ret = PythonArg(arg_size - 1);
  } else {
    *ret = ndoutputs[0];
  }
});

MXNET_REGISTER_API("_npi.dstack")
.set_body([](runtime::MXNetArgs args, runtime::MXNetRetValue* ret) {
  using namespace runtime;
  const nnvm::Op* op = Op::Get("_npi_dstack");
  nnvm::NodeAttrs attrs;
  op::ConcatParam param;
  int args_size = args.size();
  // param.num_args
  param.num_args = args_size;
  attrs.parsed = param;
  attrs.op = op;
  SetAttrDict<op::ConcatParam>(&attrs);
  // inputs
  int num_inputs = args_size;
  std::vector<NDArray*> inputs_vec(args_size, nullptr);
  for (int i = 0; i < args_size; ++i) {
    inputs_vec[i] = args[i].operator mxnet::NDArray*();
  }
  NDArray** inputs = inputs_vec.data();
  // outputs
  int num_outputs = 0;
  auto ndoutputs = Invoke(op, &attrs, num_inputs, inputs, &num_outputs, nullptr);
  *ret = ndoutputs[0];
});

MXNET_REGISTER_API("_npi.split")
.set_body([](runtime::MXNetArgs args, runtime::MXNetRetValue* ret) {
  using namespace runtime;
  const nnvm::Op* op = Op::Get("_npi_split");
  int num_inputs = 1;
  NDArray* inputs[] = {args[0].operator mxnet::NDArray*()};
  nnvm::NodeAttrs attrs;
  op::SplitParam param;
  param.axis = args[2].operator int();
  param.squeeze_axis = false;
  if (args[1].type_code() == kDLInt) {
    param.indices = TShape(0, 0);
    param.sections = args[1].operator int();
    int index = param.axis >= 0 ? param.axis :
                                  param.axis + inputs[0]->shape().ndim();
    CHECK_GE(index, 0) << "IndexError: tuple index out of range";
    CHECK_GT(param.sections, 0)
      << "ValueError: number sections must be larger than 0";
    CHECK_EQ(inputs[0]->shape()[index] % param.sections, 0)
      << "ValueError: array split does not result in an equal division";
  } else {
    TShape t = TShape(args[1].operator ObjectRef());
    param.indices = TShape(t.ndim() + 1, 0);
    for (int i = 0; i < t.ndim(); ++i) {
      param.indices[i + 1] = t[i];
    }
    param.sections = 0;
  }
  attrs.parsed = std::move(param);
  attrs.op = op;
  SetAttrDict<op::SplitParam>(&attrs);

  int num_outputs = 0;
  auto ndoutputs = Invoke(op, &attrs, num_inputs, inputs, &num_outputs, nullptr);
  std::vector<NDArrayHandle> ndarray_handles;
  ndarray_handles.reserve(num_outputs);
  for (int i = 0; i < num_outputs; ++i) {
    ndarray_handles.emplace_back(ndoutputs[i]);
  }
  *ret = ADT(0, ndarray_handles.begin(), ndarray_handles.end());
});

MXNET_REGISTER_API("_npi.roll")
.set_body([](runtime::MXNetArgs args, runtime::MXNetRetValue* ret) {
  using namespace runtime;
  static const nnvm::Op* op = Op::Get("_npi_roll");
  nnvm::NodeAttrs attrs;
  op::NumpyRollParam param;
  if (args[1].type_code() == kNull) {
    param.shift = dmlc::nullopt;
  } else if (args[1].type_code() == kDLInt) {
    param.shift = TShape(1, args[1].operator int64_t());
  } else {
    param.shift = TShape(args[1].operator ObjectRef());
  }
  if (args[2].type_code() == kNull) {
    param.axis = dmlc::nullopt;
  } else if (args[2].type_code() == kDLInt) {
    param.axis = TShape(1, args[2].operator int64_t());
  } else {
    param.axis = TShape(args[2].operator ObjectRef());
  }
  attrs.parsed = std::move(param);
  attrs.op = op;
  SetAttrDict<op::NumpyRollParam>(&attrs);
  NDArray* inputs[] = {args[0].operator mxnet::NDArray*()};
  int num_inputs = 1;
  int num_outputs = 0;
  auto ndoutputs = Invoke(op, &attrs, num_inputs, inputs, &num_outputs, nullptr);
  *ret = ndoutputs[0];
});

MXNET_REGISTER_API("_npi.rot90")
.set_body([](runtime::MXNetArgs args, runtime::MXNetRetValue* ret) {
  using namespace runtime;
  static const nnvm::Op* op = Op::Get("_npi_rot90");
  nnvm::NodeAttrs attrs;
  op::NumpyRot90Param param;
  param.k = args[1].operator int();
  if (args[2].type_code() == kNull) {
    param.axes = dmlc::nullopt;
  } else if (args[2].type_code() == kDLInt) {
    param.axes = TShape(1, args[2].operator int64_t());
  } else {
    param.axes = TShape(args[2].operator ObjectRef());
  }
  attrs.parsed = std::move(param);
  attrs.op = op;
  SetAttrDict<op::NumpyRot90Param>(&attrs);
  NDArray* inputs[] = {args[0].operator mxnet::NDArray*()};
  int num_inputs = 1;
  int num_outputs = 0;
  auto ndoutputs = Invoke(op, &attrs, num_inputs, inputs, &num_outputs, nullptr);
  *ret = ndoutputs[0];
});

MXNET_REGISTER_API("_npi.column_stack")
.set_body([](runtime::MXNetArgs args, runtime::MXNetRetValue* ret) {
  using namespace runtime;
  const nnvm::Op* op = Op::Get("_npi_column_stack");
  nnvm::NodeAttrs attrs;
  op::NumpyColumnStackParam param;
  param.num_args = args.size();

  attrs.parsed = param;
  attrs.op = op;
  SetAttrDict<op::NumpyColumnStackParam>(&attrs);
  int num_outputs = 0;
  std::vector<NDArray*> inputs;
  for (int i = 0; i < param.num_args; ++i) {
    inputs.push_back(args[i].operator mxnet::NDArray*());
  }
  auto ndoutputs = Invoke(op, &attrs, param.num_args, &inputs[0], &num_outputs, nullptr);
  *ret = ndoutputs[0];
});

MXNET_REGISTER_API("_npi.hstack")
.set_body([](runtime::MXNetArgs args, runtime::MXNetRetValue* ret) {
  using namespace runtime;
  const nnvm::Op* op = Op::Get("_npi_hstack");
  nnvm::NodeAttrs attrs;
  op::ConcatParam param;
  param.num_args = args.size();

  attrs.parsed = param;
  attrs.op = op;
  SetAttrDict<op::ConcatParam>(&attrs);
  int num_outputs = 0;
  std::vector<NDArray*> inputs;
  for (int i = 0; i < param.num_args; ++i) {
    inputs.push_back(args[i].operator mxnet::NDArray*());
  }
  auto ndoutputs = Invoke(op, &attrs, param.num_args, &inputs[0], &num_outputs, nullptr);
  *ret = ndoutputs[0];
});

MXNET_REGISTER_API("_npi.array_split")
.set_body([](runtime::MXNetArgs args, runtime::MXNetRetValue* ret) {
  using namespace runtime;
  static const nnvm::Op* op = Op::Get("_npi_array_split");
  nnvm::NodeAttrs attrs;
  op::SplitParam param;
  param.axis = args[2].operator int();
  param.squeeze_axis = false;
  if (args[1].type_code() == kDLInt) {
    param.indices = TShape(0, 0);
    param.sections = args[1].operator int();
    CHECK_GT(param.sections, 0)
      << "ValueError: number sections must be larger than 0";
  } else {
    TShape t = TShape(args[1].operator ObjectRef());
    param.indices = TShape(t.ndim() + 1, 0);
    for (int i = 0; i < t.ndim(); ++i) {
      param.indices[i + 1] = t[i];
    }
    param.sections = 0;
  }
  attrs.parsed = std::move(param);
  attrs.op = op;
  SetAttrDict<op::SplitParam>(&attrs);
  NDArray* inputs[] = {args[0].operator mxnet::NDArray*()};
  int num_inputs = 1;
  int num_outputs = 0;
  auto ndoutputs = Invoke(op, &attrs, num_inputs, inputs, &num_outputs, nullptr);
  std::vector<NDArrayHandle> ndarray_handles;
  ndarray_handles.reserve(num_outputs);
  for (int i = 0; i < num_outputs; ++i) {
    ndarray_handles.emplace_back(ndoutputs[i]);
  }
  *ret = ADT(0, ndarray_handles.begin(), ndarray_handles.end());
});

MXNET_REGISTER_API("_npi.dsplit")
.set_body([](runtime::MXNetArgs args, runtime::MXNetRetValue* ret) {
  using namespace runtime;
  static const nnvm::Op* op = Op::Get("_npi_split");
  int num_inputs = 1;
  NDArray* inputs[] = {args[0].operator mxnet::NDArray*()};
  CHECK_GE(inputs[0]->shape().ndim(), 3)
      << "ValueError: dsplit only works on arrays of 3 or more dimensions";
  nnvm::NodeAttrs attrs;
  op::SplitParam param;
  param.axis = 2;
  param.squeeze_axis = false;
  if (args[1].type_code() == kDLInt) {
    param.indices = TShape(0, 0);
    param.sections = args[1].operator int();
    CHECK_EQ(inputs[0]->shape()[2] % param.sections, 0)
      << "ValueError: array split does not result in an equal division";
    CHECK_GT(param.sections, 0)
      << "ValueError: number sections must be larger than 0";
  } else {
    TShape t = TShape(args[1].operator ObjectRef());
    param.indices = TShape(t.ndim() + 1, 0);
    for (int i = 0; i < t.ndim(); ++i) {
      param.indices[i + 1] = t[i];
    }
    param.sections = 0;
  }
  attrs.parsed = std::move(param);
  attrs.op = op;
  SetAttrDict<op::SplitParam>(&attrs);
  int num_outputs = 0;
  auto ndoutputs = Invoke(op, &attrs, num_inputs, inputs, &num_outputs, nullptr);
  std::vector<NDArrayHandle> ndarray_handles;
  ndarray_handles.reserve(num_outputs);
  for (int i = 0; i < num_outputs; ++i) {
    ndarray_handles.emplace_back(ndoutputs[i]);
  }
  *ret = ADT(0, ndarray_handles.begin(), ndarray_handles.end());
});

MXNET_REGISTER_API("_npi.hsplit")
.set_body([](runtime::MXNetArgs args, runtime::MXNetRetValue* ret) {
  using namespace runtime;
  static const nnvm::Op* op = Op::Get("_npi_hsplit");
  int num_inputs = 1;
  NDArray* inputs[] = {args[0].operator mxnet::NDArray*()};
  CHECK_GE(inputs[0]->shape().ndim(), 1)
    << "ValueError: hsplit only works on arrays of 1 or more dimensions";
  nnvm::NodeAttrs attrs;
  op::SplitParam param;
  param.axis = 0;
  param.squeeze_axis = false;
  if (args[1].type_code() == kDLInt) {
    param.indices = TShape(0, 0);
    param.sections = args[1].operator int();
    CHECK_GT(param.sections, 0)
      << "ValueError: number sections must be larger than 0";
  } else {
    TShape t = TShape(args[1].operator ObjectRef());
    param.indices = TShape(t.ndim() + 1, 0);
    for (int i = 0; i < t.ndim(); ++i) {
      param.indices[i + 1] = t[i];
    }
    param.sections = 0;
  }
  attrs.parsed = std::move(param);
  attrs.op = op;
  SetAttrDict<op::SplitParam>(&attrs);
  int num_outputs = 0;
  auto ndoutputs = Invoke(op, &attrs, num_inputs, inputs, &num_outputs, nullptr);
  std::vector<NDArrayHandle> ndarray_handles;
  ndarray_handles.reserve(num_outputs);
  for (int i = 0; i < num_outputs; ++i) {
    ndarray_handles.emplace_back(ndoutputs[i]);
  }
  *ret = ADT(0, ndarray_handles.begin(), ndarray_handles.end());
});

MXNET_REGISTER_API("_npi.vsplit")
.set_body([](runtime::MXNetArgs args, runtime::MXNetRetValue* ret) {
  using namespace runtime;
  static const nnvm::Op* op = Op::Get("_npi_split");
  int num_inputs = 1;
  NDArray* inputs[] = {args[0].operator mxnet::NDArray*()};
  CHECK_GE(inputs[0]->shape().ndim(), 2)
      << "ValueError: vsplit only works on arrays of 2 or more dimensions";
  nnvm::NodeAttrs attrs;
  op::SplitParam param;
  param.axis = 0;
  param.squeeze_axis = false;
  if (args[1].type_code() == kDLInt) {
    param.indices = TShape(0, 0);
    param.sections = args[1].operator int();
    CHECK_EQ(inputs[0]->shape()[0] % param.sections, 0)
      << "ValueError: array split does not result in an equal division";
    CHECK_GT(param.sections, 0)
      << "ValueError: number sections must be larger than 0";
  } else {
    TShape t = TShape(args[1].operator ObjectRef());
    param.indices = TShape(t.ndim() + 1, 0);
    for (int i = 0; i < t.ndim(); ++i) {
      param.indices[i + 1] = t[i];
    }
    param.sections = 0;
  }
  attrs.parsed = std::move(param);
  attrs.op = op;
  SetAttrDict<op::SplitParam>(&attrs);
  int num_outputs = 0;
  auto ndoutputs = Invoke(op, &attrs, num_inputs, inputs, &num_outputs, nullptr);
  std::vector<NDArrayHandle> ndarray_handles;
  ndarray_handles.reserve(num_outputs);
  for (int i = 0; i < num_outputs; ++i) {
    ndarray_handles.emplace_back(ndoutputs[i]);
  }
  *ret = ADT(0, ndarray_handles.begin(), ndarray_handles.end());
});

MXNET_REGISTER_API("_npi.diag")
.set_body([](runtime::MXNetArgs args, runtime::MXNetRetValue* ret) {
  using namespace runtime;
  const nnvm::Op* op = Op::Get("_npi_diag");
  nnvm::NodeAttrs attrs;
  op::NumpyDiagParam param;
  param.k = args[1].operator int();
  attrs.parsed = param;
  attrs.op = op;
  SetAttrDict<op::NumpyDiagParam>(&attrs);
  NDArray* inputs[] = {args[0].operator mxnet::NDArray*()};
  int num_inputs = 1;
  int num_outputs = 0;
  auto ndoutputs = Invoke(op, &attrs, num_inputs, inputs, &num_outputs, nullptr);
  *ret = ndoutputs[0];
});

MXNET_REGISTER_API("_npi.diagonal")
.set_body([](runtime::MXNetArgs args, runtime::MXNetRetValue* ret) {
  using namespace runtime;
  const nnvm::Op* op = Op::Get("_npi_diagonal");
  nnvm::NodeAttrs attrs;
  op::NumpyDiagonalParam param;
  param.offset = args[1].operator int();
  param.axis1 = args[2].operator int();
  param.axis2 = args[3].operator int();
  attrs.parsed = param;
  attrs.op = op;
  SetAttrDict<op::NumpyDiagonalParam>(&attrs);
  NDArray* inputs[] = {args[0].operator mxnet::NDArray*()};
  int num_inputs = 1;
  int num_outputs = 0;
  auto ndoutputs = Invoke(op, &attrs, num_inputs, inputs, &num_outputs, nullptr);
  *ret = ndoutputs[0];
});

MXNET_REGISTER_API("_npi.diag_indices_from")
.set_body([](runtime::MXNetArgs args, runtime::MXNetRetValue* ret) {
  using namespace runtime;
  const nnvm::Op* op = Op::Get("_npi_diag_indices_from");
  nnvm::NodeAttrs attrs;
  attrs.op = op;
  NDArray* inputs[] = {args[0].operator mxnet::NDArray*()};
  int num_inputs = 1;
  int num_outputs = 0;
  auto ndoutputs = Invoke(op, &attrs, num_inputs, inputs, &num_outputs, nullptr);
  *ret = ndoutputs[0];
});

}  // namespace mxnet
