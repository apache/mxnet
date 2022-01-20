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
 * \file dnnl_quantized_reshape-inl.h
 * \author: Adam Grabowski, adam.grabowski@intel.com
 */

#ifndef MXNET_OPERATOR_QUANTIZATION_DNNL_DNNL_QUANTIZED_RESHAPE_INL_H_
#define MXNET_OPERATOR_QUANTIZATION_DNNL_DNNL_QUANTIZED_RESHAPE_INL_H_

#if MXNET_USE_ONEDNN == 1
#include <string>
#include <vector>
#include "../../tensor/matrix_op-inl.h"
#include "../../numpy/np_matrix_op-inl.h"
#include "../../nn/dnnl/dnnl_ops-inl.h"

namespace mxnet {
namespace op {

struct QuantizedReshapeParam : public dmlc::Parameter<QuantizedReshapeParam> {
  mxnet::TShape newshape;
  mxnet::Tuple<int> shape;
  bool reverse, keep_highest, is_numpy_op;
  std::string order;

  DMLC_DECLARE_PARAMETER(QuantizedReshapeParam) {
    DMLC_DECLARE_FIELD(newshape).set_default(mxnet::TShape(0, -1));
    DMLC_DECLARE_FIELD(shape).set_default(mxnet::Tuple<int>());
    DMLC_DECLARE_FIELD(reverse).set_default(false);
    DMLC_DECLARE_FIELD(order).set_default("C");
    DMLC_DECLARE_FIELD(keep_highest).set_default(false);
    DMLC_DECLARE_FIELD(is_numpy_op).set_default(true);
  }

  void SetAttrDict(std::unordered_map<std::string, std::string>* dict) {
    std::ostringstream newshape_s, shape_s, reverse_s, order_s, keep_highest_s, is_numpy_op_s;
    newshape_s << newshape;
    shape_s << shape;
    reverse_s << reverse;
    order_s << order;
    keep_highest_s << keep_highest;
    is_numpy_op_s << is_numpy_op;
    (*dict)["newshape"]     = newshape_s.str();
    (*dict)["shape"]        = shape_s.str();
    (*dict)["reverse"]      = reverse_s.str();
    (*dict)["order"]        = order_s.str();
    (*dict)["keep_highest"] = keep_highest_s.str();
    (*dict)["is_numpy_op"]  = is_numpy_op_s.str();
  }
};

bool QuantizedReshapeInferShape(const nnvm::NodeAttrs& attrs,
                                mxnet::ShapeVector* in_attrs,
                                mxnet::ShapeVector* out_attrs) {
  const QuantizedReshapeParam& param = nnvm::get<QuantizedReshapeParam>(attrs.parsed);
  CHECK_EQ(in_attrs->size(), 3U);
  CHECK_EQ(out_attrs->size(), 3U);
  mxnet::ShapeVector input  = {in_attrs->at(0)};
  mxnet::ShapeVector output = {out_attrs->at(0)};
  nnvm::NodeAttrs _attrs;
  bool ret;

  if (param.is_numpy_op) {
    NumpyXReshapeParam _param;
    _param.newshape = param.newshape;
    _param.reverse  = param.reverse;
    _param.order    = param.order;
    _attrs.parsed   = _param;
    ret             = NumpyXReshapeShape(_attrs, &input, &output);
  } else {
    ReshapeParam _param;
    _param.shape        = param.shape;
    _param.keep_highest = param.keep_highest;
    _param.reverse      = param.reverse;
    _attrs.parsed       = _param;
    ret                 = ReshapeShape(_attrs, &input, &output);
  }
  SHAPE_ASSIGN_CHECK(*in_attrs, 1, mxnet::TShape{1});
  SHAPE_ASSIGN_CHECK(*in_attrs, 2, mxnet::TShape{1});
  SHAPE_ASSIGN_CHECK(*out_attrs, 0, output[0]);
  SHAPE_ASSIGN_CHECK(*out_attrs, 1, mxnet::TShape{1});
  SHAPE_ASSIGN_CHECK(*out_attrs, 2, mxnet::TShape{1});

  return ret;
}

bool QuantizedReshapeStorageType(const nnvm::NodeAttrs& attrs,
                                 const int dev_mask,
                                 DispatchMode* dispatch_mode,
                                 std::vector<int>* in_attrs,
                                 std::vector<int>* out_attrs) {
  CHECK_EQ(in_attrs->size(), 3U);
  CHECK_EQ(out_attrs->size(), 3U);
  return DNNLStorageType(attrs, dev_mask, true, dispatch_mode, in_attrs, out_attrs);
}

bool QuantizedReshapeType(const nnvm::NodeAttrs& attrs,
                          std::vector<int>* in_attrs,
                          std::vector<int>* out_attrs) {
  CHECK_EQ(in_attrs->size(), 3U);
  CHECK_EQ(out_attrs->size(), 3U);
  TYPE_ASSIGN_CHECK(*in_attrs, 1, mshadow::kFloat32);
  TYPE_ASSIGN_CHECK(*in_attrs, 2, mshadow::kFloat32);
  TYPE_ASSIGN_CHECK(*out_attrs, 0, (*in_attrs)[0]);
  TYPE_ASSIGN_CHECK(*out_attrs, 1, mshadow::kFloat32);
  TYPE_ASSIGN_CHECK(*out_attrs, 2, mshadow::kFloat32);
  return (*in_attrs)[0] != -1;
}

static void DNNLQuantizedReshapeForward(const nnvm::NodeAttrs& attrs,
                                        const OpContext& ctx,
                                        const std::vector<NDArray>& inputs,
                                        const std::vector<OpReqType>& req,
                                        const std::vector<NDArray>& outputs) {
  CHECK(inputs[0].dtype() == mshadow::kUint8 || inputs[0].dtype() == mshadow::kInt8)
      << "dnnl_quantized_reshape op only supports uint8 and int8 as input type";

  if (SupportDNNLReshape(inputs[0], outputs[0])) {
    OpReqType reqType;
    if (inputs[0].GetDNNLData()->get_data_handle() != outputs[0].GetDNNLData()->get_data_handle())
      reqType = kWriteTo;
    else
      reqType = req[0];
    DNNLRun(DNNLReshapeForward, attrs, ctx, inputs[0], reqType, outputs[0]);
  } else {
    FallBackCompute(UnaryOp::IdentityCompute<cpu>, attrs, ctx, inputs, req, outputs);
  }
  *outputs[1].data().dptr<float>() = *inputs[1].data().dptr<float>();
  *outputs[2].data().dptr<float>() = *inputs[2].data().dptr<float>();
}

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_USE_ONEDNN == 1
#endif  // MXNET_OPERATOR_QUANTIZATION_DNNL_DNNL_QUANTIZED_RESHAPE_INL_H_
