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
 * \file dnnl_batch_dot.cc
 * \brief DNNL (Quantized) batch_dot operator based on subgraph
 */

#if MXNET_USE_ONEDNN == 1

#include <string>
#include <utility>
#include <vector>

#include "operator/nn/dnnl/dnnl_base-inl.h"
#include "operator/nn/dnnl/dnnl_batch_dot-inl.h"
#include "operator/quantization/quantization_utils.h"
#include "operator/tensor/matrix_op-inl.h"
#include "operator/subgraph/common.h"

namespace mxnet {
namespace op {

bool DNNLBatchDotShape(const nnvm::NodeAttrs& attrs,
                       mxnet::ShapeVector* in_shapes,
                       mxnet::ShapeVector* out_shapes) {
  const DNNLDotParam& param = nnvm::get<DNNLDotParam>(attrs.parsed);
  mxnet::ShapeVector base_in_shapes;
  mxnet::ShapeVector base_out_shapes;
  const size_t base_num_inputs = 2;

  base_out_shapes.push_back(out_shapes->at(DotOut::out));
  for (int i = 0; i < base_num_inputs; ++i) {
    base_in_shapes.push_back(in_shapes->at(i));
  }
  BatchDotShape<DNNLDotParam>(attrs, &base_in_shapes, &base_out_shapes);

  for (size_t i = 0; i < in_shapes->size(); ++i) {
    if (i < base_in_shapes.size()) {
      in_shapes->at(i) = base_in_shapes[i];
    } else {
      SHAPE_ASSIGN_CHECK(*in_shapes, i, mshadow::Shape1(1));
    }
  }

  out_shapes->at(DotOut::out) = base_out_shapes[DotOut::out];
  if (param.quantized && !param.enabled_float_output.has_value()) {
    SHAPE_ASSIGN_CHECK(*out_shapes, DotOut::out_min, mshadow::Shape1(1));
    SHAPE_ASSIGN_CHECK(*out_shapes, DotOut::out_max, mshadow::Shape1(1));
  }

  return true;
}

bool DNNLBatchDotType(const nnvm::NodeAttrs& attrs,
                      std::vector<int>* in_types,
                      std::vector<int>* out_types) {
  const DNNLDotParam& param    = nnvm::get<DNNLDotParam>(attrs.parsed);
  const size_t base_num_inputs = 2;
  if (param.quantized) {
    if (in_types->at(DotIn::lhs) == mshadow::kBfloat16 ||
        in_types->at(DotIn::rhs) == mshadow::kBfloat16) {
      return false;
    }

    CHECK(in_types->at(DotIn::lhs) == mshadow::kInt8 || in_types->at(DotIn::lhs) == mshadow::kUint8)
        << "Quantized batch-dot lhs only supports int8/uint8 input, while "
        << in_types->at(DotIn::lhs) << " is given.";
    CHECK(in_types->at(DotIn::rhs) == mshadow::kInt8 || in_types->at(DotIn::rhs) == mshadow::kUint8)
        << "Quantized batch-dot rhs only supports int8 input, while " << in_types->at(DotIn::rhs)
        << " is given.";

    for (size_t i = base_num_inputs; i < in_types->size(); ++i) {
      TYPE_ASSIGN_CHECK(*in_types, i, mshadow::kFloat32);
    }

    if (param.enabled_float_output.has_value()) {
      TYPE_ASSIGN_CHECK(*out_types, DotOut::out, param.enabled_float_output.value());
    } else {
      if (param.min_calib_range.has_value() && param.max_calib_range.has_value()) {
        TYPE_ASSIGN_CHECK(*out_types, DotOut::out, mshadow::kInt8);
      } else {
        TYPE_ASSIGN_CHECK(*out_types, DotOut::out, mshadow::kInt32);
      }
      TYPE_ASSIGN_CHECK(*out_types, DotOut::out_min, mshadow::kFloat32);
      TYPE_ASSIGN_CHECK(*out_types, DotOut::out_max, mshadow::kFloat32);
    }
  } else {
    if ((*in_types)[DotIn::lhs] == mshadow::kBfloat16 ||
        (*in_types)[DotIn::rhs] == mshadow::kBfloat16) {
      TYPE_ASSIGN_CHECK(*in_types, DotIn::lhs, mshadow::kBfloat16);
      TYPE_ASSIGN_CHECK(*in_types, DotIn::rhs, mshadow::kBfloat16);
      if (param.enabled_float_output.has_value()) {
        CHECK_EQ(param.enabled_float_output.value(), mshadow::kFloat32);
        TYPE_ASSIGN_CHECK(*out_types, DotOut::out, mshadow::kFloat32);
      } else {
        TYPE_ASSIGN_CHECK(*out_types, DotOut::out, mshadow::kBfloat16);
      }
    } else {
      CHECK(!param.enabled_float_output.has_value());
      TYPE_ASSIGN_CHECK(*in_types, DotIn::lhs, mshadow::kFloat32);
      TYPE_ASSIGN_CHECK(*in_types, DotIn::rhs, mshadow::kFloat32);
      TYPE_ASSIGN_CHECK(*out_types, DotOut::out, mshadow::kFloat32);
    }
  }

  return true;
}

inline static bool DNNLBatchDotStorageType(const nnvm::NodeAttrs& attrs,
                                           const int dev_mask,
                                           DispatchMode* dispatch_mode,
                                           std::vector<int>* in_attrs,
                                           std::vector<int>* out_attrs) {
  return DNNLStorageType(attrs, dev_mask, true, dispatch_mode, in_attrs, out_attrs);
}

NNVM_REGISTER_OP(_sg_onednn_batch_dot)
    .describe(R"code(_sg_onednn_batch_dot)code" ADD_FILELINE)
    .set_num_inputs([](const NodeAttrs& attrs) {
      auto const& param = nnvm::get<DNNLDotParam>(attrs.parsed);
      // two normal inputs + min/max for quantized version
      return param.quantized ? 6 : 2;
    })
    .set_num_outputs([](const NodeAttrs& attrs) {
      auto const& param = nnvm::get<DNNLDotParam>(attrs.parsed);
      return (param.quantized && !param.enabled_float_output.has_value()) ? 3 : 1;
    })
    .set_attr_parser(ParamParser<DNNLDotParam>)
    .set_attr<nnvm::FListInputNames>(
        "FListInputNames",
        [](const NodeAttrs& attrs) {
          auto const& param = nnvm::get<DNNLDotParam>(attrs.parsed);
          if (param.quantized) {
            return std::vector<std::string>{
                "lhs", "rhs", "min_lhs", "max_lhs", "min_rhs", "max_rhs"};
          } else {
            return std::vector<std::string>{"lhs", "rhs"};
          }
        })
    .set_attr<nnvm::FListOutputNames>(
        "FListOutputNames",
        [](const NodeAttrs& attrs) {
          auto const& param = nnvm::get<DNNLDotParam>(attrs.parsed);
          if (param.quantized && !param.enabled_float_output.has_value()) {
            return std::vector<std::string>{"output", "min_output", "max_output"};
          } else {
            return std::vector<std::string>{"output"};
          }
        })
    .set_attr<mxnet::FInferShape>("FInferShape", DNNLBatchDotShape)
    .set_attr<nnvm::FInferType>("FInferType", DNNLBatchDotType)
    .set_attr<FInferStorageType>("FInferStorageType", DNNLBatchDotStorageType)
    .set_attr<FComputeEx>("FComputeEx<cpu>", DNNLBatchDotForward<true>)
    .set_attr<bool>("TIsDNNL", true)
    .set_attr<nnvm::FGradient>("FGradient", MakeZeroGradNodes)
    .set_attr<FQuantizable>("FQuantizable",
                            [](const NodeAttrs& attrs) { return QuantizeType::kMust; })
    .set_attr<FQuantizedOp>("FQuantizedOp",
                            [](const NodeAttrs& attrs) {
                              nnvm::ObjectPtr node          = nnvm::Node::Create();
                              node->attrs.op                = Op::Get("_sg_onednn_batch_dot");
                              node->attrs.name              = "quantized_" + attrs.name;
                              node->attrs.dict              = attrs.dict;
                              node->attrs.dict["quantized"] = "True";

                              if (node->op()->attr_parser != nullptr) {
                                node->op()->attr_parser(&(node->attrs));
                              }
                              return node;
                            })
    .set_attr<FNeedRequantize>("FNeedRequantize", [](const NodeAttrs& attrs) { return true; });

}  // namespace op
}  // namespace mxnet

#endif  // if MXNET_USE_ONEDNN == 1
