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
 * \file intgemm_fully_connected_op.cc
 * \brief Operator wrapping intgemm's Multiply routine
 */

#include <mxnet/operator_util.h>
#include <vector>
#include <cstdlib>
#include "../../mshadow_op.h"
#include "../../mxnet_op.h"
#include "../../operator_common.h"
#include "../../tensor/init_op.h"

#include "intgemm/intgemm.h"

namespace mxnet {
namespace op {

struct IntgemmFullyConnectedParam : public dmlc::Parameter<IntgemmFullyConnectedParam> {
  int out_type;
  int num_hidden;
  bool no_bias;
  bool flatten;
  DMLC_DECLARE_PARAMETER(IntgemmFullyConnectedParam) {
    // This part os a copy of the FullyConnected parameters.
    DMLC_DECLARE_FIELD(num_hidden).set_lower_bound(1)
    .describe("Number of hidden nodes of the output.");
    DMLC_DECLARE_FIELD(no_bias).set_default(false)
    .describe("Whether to disable bias parameter.");
    DMLC_DECLARE_FIELD(flatten).set_default(true)
    .describe("Whether to collapse all but the first axis of the input data tensor.");

    DMLC_DECLARE_FIELD(out_type)
    .add_enum("float32", mshadow::kFloat32)
    .add_enum("int32", mshadow::kInt32)
    .set_default(mshadow::kFloat32)
    .describe("Output data type.");
  }
};
DMLC_REGISTER_PARAMETER(IntgemmFullyConnectedParam);

namespace {
// Parse the above fields into indices for parameters.
// The order is: data weight [scaling] [bias].
struct ParameterIndices {
  explicit ParameterIndices(const IntgemmFullyConnectedParam& param) :
    data(0),
    weight(1),
    scaling(param.out_type == mshadow::kFloat32 ? 2 : kInvalid),
    bias(param.no_bias ? kInvalid : (HaveScaling() ? 3 : 2)),
    count(2U + HaveScaling() + HaveBias()) {}
  bool HaveScaling() const { return scaling != kInvalid; }
  bool HaveBias() const { return bias != kInvalid; }
  const unsigned int data;
  const unsigned int weight;
  const unsigned int scaling;
  const unsigned int bias;
  const unsigned int count;
  static const unsigned int kInvalid = std::numeric_limits<unsigned int>::max();
};
template<class T> ParameterIndices Sanity(const nnvm::NodeAttrs& attrs,
                                          T* in,
                                          T* out) {
  // 3-4 parameters: A, B, scaling, and optional bias
  ParameterIndices ret(nnvm::get<IntgemmFullyConnectedParam>(attrs.parsed));
  CHECK_EQ(in->size(), ret.count);
  CHECK_EQ(out->size(), 1U);
  return ret;
}
}  // namespace

inline bool IntgemmFullyConnectedOpShape(const nnvm::NodeAttrs& attrs,
                             mxnet::ShapeVector* in_shape,
                             mxnet::ShapeVector* out_shape) {
  const ParameterIndices indices(Sanity(attrs, in_shape, out_shape));
  const IntgemmFullyConnectedParam& param = nnvm::get<IntgemmFullyConnectedParam>(attrs.parsed);
  // This follows FullyConnectedShape except for scaling.
  using namespace mshadow;
  mxnet::TShape dshape = (*in_shape)[indices.data];
  mxnet::TShape oshape = (*out_shape)[0];
  // require data to be known
  if (!mxnet::ndim_is_known(dshape)) return false;

  index_t num_input;
  if (!param.flatten) {
    num_input = dshape[dshape.ndim()-1];
  } else {
    num_input = dshape.ProdShape(1, dshape.ndim());
  }
  SHAPE_ASSIGN_CHECK(*in_shape, indices.weight, Shape2(param.num_hidden, num_input));
  if (indices.HaveScaling()) {
    SHAPE_ASSIGN_CHECK(*in_shape, indices.scaling, mxnet::TShape(1, 1));
  }
  if (indices.HaveBias()) {
    if (!shape_assign(&(*in_shape)[indices.bias], Shape1(param.num_hidden)) &&
        !shape_assign(&(*in_shape)[indices.bias], Shape2(param.num_hidden, 1))) {
      LOG(FATAL) << "Unexpected shape for bias " << (*in_shape)[indices.bias];
    }
  }

  if (!param.flatten) {
    mxnet::TShape result_shape(dshape);
    result_shape[dshape.ndim()-1] = param.num_hidden;
    SHAPE_ASSIGN_CHECK(*out_shape, 0, result_shape);
  } else {
    SHAPE_ASSIGN_CHECK(*out_shape, 0, Shape2(dshape[0], param.num_hidden));
  }
  if (oshape.ndim() > 0) {
    dshape[0] = oshape[0];
    SHAPE_ASSIGN_CHECK(*in_shape, indices.data, dshape);
  }
  return true;
}

bool IntgemmFullyConnectedOpType(const nnvm::NodeAttrs& attrs,
                            std::vector<int>* in_attrs,
                            std::vector<int>* out_attrs) {
  const ParameterIndices indices(Sanity(attrs, in_attrs, out_attrs));
  const IntgemmFullyConnectedParam& param = nnvm::get<IntgemmFullyConnectedParam>(attrs.parsed);

  // Match the configuration for output.
  TYPE_ASSIGN_CHECK(*out_attrs, 0, param.out_type);
  if (indices.HaveBias()) {
    // Bias has same type as output.
    TYPE_ASSIGN_CHECK(*in_attrs, indices.bias, (*out_attrs)[0]);
    TYPE_ASSIGN_CHECK(*out_attrs, 0, (*in_attrs)[indices.bias]);
  }
  // Scaling is float32.
  if (indices.HaveScaling()) {
    TYPE_ASSIGN_CHECK(*in_attrs, indices.scaling, mshadow::kFloat32);
  }
  // Users have to prepare B. It wasn't intended to be efficient.
  TYPE_ASSIGN_CHECK(*in_attrs, indices.weight, mshadow::kInt8);
  // A can be a float (in which case it is automatically quantized) or int8.
  if (type_is_none((*in_attrs)[indices.data])) {
    return false;
  }
  return ((*in_attrs)[indices.data] == mshadow::kInt8 ||
      (*in_attrs)[indices.data] == mshadow::kFloat32);
}

void IntgemmFullyConnectedOpForwardCPU(const nnvm::NodeAttrs& attrs,
                          const OpContext& ctx,
                          const std::vector<TBlob>& inputs,
                          const std::vector<OpReqType>& req,
                          const std::vector<TBlob>& outputs) {
  const ParameterIndices indices(Sanity(attrs, &inputs, &outputs));
  const IntgemmFullyConnectedParam& param = nnvm::get<IntgemmFullyConnectedParam>(attrs.parsed);
  CHECK_EQ(req.size(), 1U);
  CHECK_EQ(req[0], kWriteTo) << "TODO: doing more than overwriting for intgemm.";

  const TBlob &A = inputs[indices.data], &B = inputs[indices.weight], &C = outputs[0];

  CHECK(A.type_flag_ == mshadow::kInt8 || A.type_flag_ == mshadow::kFloat32);
  CHECK_EQ(B.type_flag_, mshadow::kInt8);
  CHECK(C.type_flag_ == mshadow::kInt32 || C.type_flag_ == mshadow::kFloat32);
  CHECK(A.CheckContiguous());
  CHECK(B.CheckContiguous());
  CHECK(C.CheckContiguous());
  CHECK_GE(A.shape_.ndim(), 1);
  CHECK_GE(B.shape_.ndim(), 2);
  size_t A_rows = A.shape_.ProdShape(0, A.shape_.ndim() - 1);
  size_t inner = A.shape_[A.shape_.ndim() - 1];
  CHECK_EQ(B.shape_[B.shape_.ndim() - 1], inner);
  size_t B_cols = B.shape_.ProdShape(0, B.shape_.ndim() - 1);

  CHECK_EQ(C.shape_.Size(), A_rows * B_cols);

  bool bias = !param.no_bias;
  if (bias) {
    CHECK_EQ(inputs[indices.bias].type_flag_, C.type_flag_);
    CHECK_EQ(inputs[indices.bias].shape_.Size(), param.num_hidden);
  }
  CHECK_EQ(inner % ::intgemm::Int8::tile_info.b_rows, 0) <<
    "intgemm requires the inner dimension be a multiple of " << ::intgemm::Int8::tile_info.b_rows;
  CHECK_EQ(B_cols % ::intgemm::Int8::tile_info.b_cols, 0) <<
    "intgemm requires B have a multiple of " << ::intgemm::Int8::tile_info.b_cols <<
    " columns in the equation C = AB.";

  float out_float_multiplier;
  if (indices.HaveScaling()) {
    out_float_multiplier = *inputs[indices.scaling].dptr<float>();
  } else {
    out_float_multiplier = 0.0;  // Unused; stop compiler from complaining.
  }

  int8_t *A_quant;
  mshadow::Tensor<cpu, 1, int8_t> A_quant_store;
  if (A.type_flag_ == mshadow::kFloat32) {
    const float *A_raw = A.dptr<float>();
    // Quantize A for the user.
    // Future: allow scale to be passed in? Should the induced scale be an output?
    float scale = 127.0 / ::intgemm::MaxAbsolute(A_raw, A_raw + A.shape_.Size());
    out_float_multiplier /= scale;
    A_quant_store = ctx.requested[0].get_space_typed<cpu, 1, int8_t>(
        mshadow::Shape1(A.shape_.Size()),
        ctx.get_stream<cpu>());
    A_quant = A_quant_store.dptr_;
    ::intgemm::Int8::PrepareA(A_raw, A_quant, scale, A_rows, inner);
  } else {
    CHECK_EQ(A.type_flag_, mshadow::kInt8);
    A_quant = A.dptr<int8_t>();
  }
  const int8_t *B_quant = B.dptr<int8_t>();
  CHECK_EQ(reinterpret_cast<intptr_t>(A_quant) % 64, 0) <<
    "Pointers should be aligned to a multiple of 64.";
  CHECK_EQ(reinterpret_cast<intptr_t>(B_quant) % 64, 0) <<
    "Pointers should be aligned to a multiple of 64.";
  if (C.type_flag_ == mshadow::kFloat32) {
    CHECK_EQ(reinterpret_cast<intptr_t>(C.dptr<float>()) % 64, 0) <<
      "Pointers should be aligned to a multiple of 64.";
  } else {
    CHECK_EQ(reinterpret_cast<intptr_t>(C.dptr<int32_t>()) % 64, 0) <<
      "Pointers should be aligned to a multiple of 64.";
  }

  if (bias) {
    if (C.type_flag_ == mshadow::kFloat32) {
      CHECK_EQ(reinterpret_cast<intptr_t>(inputs[indices.bias].dptr<float>()) % 64, 0) <<
        "Pointers should be aligned to a multiple of 64.";
      ::intgemm::callbacks::UnquantizeAndAddBiasAndWrite cb(
          out_float_multiplier,
          inputs[indices.bias].dptr<float>(),
          C.dptr<float>());
      ::intgemm::Int8::Multiply(A_quant, B_quant, A_rows, inner, B_cols, cb);
    } else {
      // int32
      CHECK_EQ(reinterpret_cast<intptr_t>(inputs[indices.bias].dptr<int32_t>()) % 64, 0) <<
        "Pointers should be aligned to a multiple of 64.";
      ::intgemm::callbacks::AddBiasAndWrite cb(
          inputs[indices.bias].dptr<int32_t>(),
          C.dptr<int32_t>());
      ::intgemm::Int8::Multiply(A_quant, B_quant, A_rows, inner, B_cols, cb);
    }
  } else {
    if (C.type_flag_ == mshadow::kFloat32) {
      ::intgemm::callbacks::UnquantizeAndWrite cb(out_float_multiplier, C.dptr<float>());
      ::intgemm::Int8::Multiply(A_quant, B_quant, A_rows, inner, B_cols, cb);
    } else {
      // int32
      ::intgemm::callbacks::Write<int32_t> cb(C.dptr<int32_t>());
      ::intgemm::Int8::Multiply(A_quant, B_quant, A_rows, inner, B_cols, cb);
    }
  }
}

NNVM_REGISTER_OP(_contrib_intgemm_fully_connected)
.add_alias("_npx_intgemm_fully_connected")
.describe(R"code(Multiply matrices using 8-bit integers.  data * weight.

Input tensor arguments are: data weight [scaling] [bias]

data: either float32 or prepared using intgemm_prepare_data (in which case it is int8).

weight: must be prepared using intgemm_prepare_weight.

scaling: present if and only if out_type is float32. If so this is multiplied by the result before adding bias. Typically:
scaling = (max passed to intgemm_prepare_weight)/127.0 if data is in float32
scaling = (max_passed to intgemm_prepare_data)/127.0 * (max passed to intgemm_prepare_weight)/127.0 if data is in int8

bias: present if and only if !no_bias. This is added to the output after scaling and has the same number of columns as the output.

out_type: type of the output.
)code" ADD_FILELINE)
.set_attr_parser(ParamParser<IntgemmFullyConnectedParam>)
.set_num_inputs([](const NodeAttrs& attrs) {
  return ParameterIndices(nnvm::get<IntgemmFullyConnectedParam>(attrs.parsed)).count;
})
.set_num_outputs(1)
.set_attr<nnvm::FListInputNames>("FListInputNames",
  [](const NodeAttrs& attrs) {
    std::vector<std::string> ret{"data", "weight"};
    ParameterIndices indices(nnvm::get<IntgemmFullyConnectedParam>(attrs.parsed));
    if (indices.HaveScaling()) {
      ret.emplace_back("scaling");
    }
    if (indices.HaveBias()) {
      ret.emplace_back("bias");
    }
    return ret;
  })
.set_attr<FResourceRequest>("FResourceRequest",
  [](const NodeAttrs& attrs) {
    return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
  })
.set_attr<mxnet::FInferShape>("FInferShape", IntgemmFullyConnectedOpShape)
.set_attr<nnvm::FInferType>("FInferType", IntgemmFullyConnectedOpType)
.set_attr<FCompute>("FCompute<cpu>", IntgemmFullyConnectedOpForwardCPU)
.add_argument(
    "data",
    "NDArray-or-Symbol",
    "First argument to multiplication. Tensor of float32 (quantized on the fly) or int8 from "
      "intgemm_prepare_data. If you use a different quantizer, be sure to ban -128. The last "
      "dimension must be a multiple of 64.")
.add_argument(
    "weight",
    "NDArray-or-Symbol",
    "Second argument to multiplication. Tensor of int8 from intgemm_prepare_weight. The last "
      "dimension must be a multiple of 64.  The product of non-last dimensions must be a multiple "
      "of 8.")
.add_argument("scaling", "NDArray-or-Symbol", "Scaling factor to apply if output type is float32.")
.add_argument("bias", "NDArray-or-Symbol", "Bias term.")
// TODO(Xinyu): a temp solution to enable GluonCV INT8 flow,
// will be reverted after the improvement of CachedOP is done.
.set_attr<nnvm::FGradient>("FGradient", MakeZeroGradNodes)
.add_arguments(IntgemmFullyConnectedParam::__FIELDS__());

}  // namespace op
}  // namespace mxnet
