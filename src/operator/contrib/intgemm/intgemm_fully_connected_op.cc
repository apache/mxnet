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

#include "../../../../3rdparty/intgemm/aligned.h"
#include "../../../../3rdparty/intgemm/intgemm.h"

namespace mxnet {
namespace op {

struct IntgemmFullyConnectedParam : public dmlc::Parameter<IntgemmFullyConnectedParam> {
  float out_float_multiplier;
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
template<class T> void IntgemmFullyConnectedSanity(const nnvm::NodeAttrs& attrs, T* in, T* out) {
  // 3-4 parameters: A, B, scaling, and optional bias
  const IntgemmFullyConnectedParam& param = nnvm::get<IntgemmFullyConnectedParam>(attrs.parsed);
  CHECK_EQ(in->size(), param.no_bias ? 3U : 4U);
  CHECK_EQ(out->size(), 1U);
}
} // namespace

inline bool IntgemmFullyConnectedOpShape(const nnvm::NodeAttrs& attrs,
                             mxnet::ShapeVector* in_shape,
                             mxnet::ShapeVector* out_shape) {
  IntgemmFullyConnectedSanity(attrs, in_shape, out_shape);
  // This follows FullyConnectedShape except there's no option to flatten and the bias is implied.
  const IntgemmFullyConnectedParam& param = nnvm::get<IntgemmFullyConnectedParam>(attrs.parsed);

  // The rest is copied from FullyConnected.
  using namespace mshadow;
  if (!param.no_bias) {
    CHECK_EQ(in_shape->size(), 4U) << "Input:[data, weight, scaling_factor, bias]";
  } else {
    CHECK_EQ(in_shape->size(), 3U) << "Input:[data, weight, scaling_factor]";
  }
  CHECK_EQ(out_shape->size(), 1U);
  mxnet::TShape dshape = (*in_shape)[0];
  mxnet::TShape oshape = (*out_shape)[0];
  // require data to be known
  if (!mxnet::ndim_is_known(dshape)) return false;

  index_t num_input;
  if (!param.flatten) {
    num_input = dshape[dshape.ndim()-1];
  } else {
    num_input = dshape.ProdShape(1, dshape.ndim());
  }
  SHAPE_ASSIGN_CHECK(*in_shape, 1, Shape2(param.num_hidden, num_input));
  SHAPE_ASSIGN_CHECK(*in_shape, 2, mxnet::TShape(1, 1));
  if (!param.no_bias) {
    if (!shape_assign(&(*in_shape)[3], Shape1(param.num_hidden)) &&
        !shape_assign(&(*in_shape)[3], Shape2(param.num_hidden, 1))) {
      LOG(FATAL) << "Unexpected shape for bias " << (*in_shape)[3];
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
    SHAPE_ASSIGN_CHECK(*in_shape, 0, dshape);
  }
  return true;
}

bool IntgemmFullyConnectedOpType(const nnvm::NodeAttrs& attrs,
                            std::vector<int>* in_attrs,
                            std::vector<int>* out_attrs) {
  IntgemmFullyConnectedSanity(attrs, in_attrs, out_attrs);
  const IntgemmFullyConnectedParam& param = nnvm::get<IntgemmFullyConnectedParam>(attrs.parsed);

  // Match the configuration for output.
  TYPE_ASSIGN_CHECK(*out_attrs, 0, param.out_type);
  if (!param.no_bias) {
    // Bias has same type as output.
    TYPE_ASSIGN_CHECK(*in_attrs, 3, (*out_attrs)[0]);
    TYPE_ASSIGN_CHECK(*out_attrs, 0, (*in_attrs)[3]);
  }
  // Scaling is float32.
  TYPE_ASSIGN_CHECK(*in_attrs, 2, mshadow::kFloat32);
  // Users have to prepare B.
  TYPE_ASSIGN_CHECK(*in_attrs, 1, mshadow::kInt8);
  // A can be a float (in which case it is automatically quantized) or int8.
  if (type_is_none((*in_attrs)[0])) {
    return false;
  }
  return ((*in_attrs)[0] == mshadow::kInt8 || (*in_attrs)[0] == mshadow::kFloat32);
}

namespace {

// TODO: amend AlignedVector to allow a reset.
class FreeMe {
  public:
    FreeMe() : mem_(nullptr) {}
    ~FreeMe() { std::free(mem_); }
    void Reset(int8_t *with) {
      std::free(mem_);
      mem_ = with;
    }
  private:
    int8_t *mem_;
};

} // namespace

void IntgemmFullyConnectedOpForwardCPU(const nnvm::NodeAttrs& attrs,
                          const OpContext& ctx,
                          const std::vector<TBlob>& inputs,
                          const std::vector<OpReqType>& req,
                          const std::vector<TBlob>& outputs) {
  IntgemmFullyConnectedSanity(attrs, &inputs, &outputs);
  const IntgemmFullyConnectedParam& param = nnvm::get<IntgemmFullyConnectedParam>(attrs.parsed);
  CHECK_EQ(req.size(), 1U);
  CHECK_EQ(req[0], kWriteTo) << "TODO: doing more than overwriting for intgemm.  Note: kWriteInplace = " << kWriteInplace << " kWriteTo = " << kWriteTo;

  const TBlob &A = inputs[0], &B = inputs[1], &C = outputs[0];

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
    CHECK_EQ(inputs[3].type_flag_, mshadow::kFloat32);
    CHECK_EQ(C.type_flag_, mshadow::kFloat32);
    CHECK_EQ(inputs[3].shape_.Size(), param.num_hidden);
  }
  CHECK_EQ(inner % ::intgemm::Int8::kBTileRow, 0) << "intgemm requires the inner dimension be a multiple of " << ::intgemm::Int8::kBTileRow;
  CHECK_EQ(B_cols % ::intgemm::Int8::kBTileCol, 0) << "intgemm requires B have a multiple of " << ::intgemm::Int8::kBTileCol << " columns inthe equation C = AB.";

  float out_float_multiplier = *inputs[2].dptr<float>();

  int8_t *A_quant;
  // TODO report this memory consumption?
  FreeMe A_quant_store;
  if (A.type_flag_ == mshadow::kFloat32) {
    const float *A_raw = A.dptr<float>();
    // Quantize A for the user.  TODO: allow scale to be passed in.  Should the induced scale be an output?
    float scale = 127.0 / ::intgemm::MaxAbsolute(A_raw, A_raw + A.shape_.Size());
    out_float_multiplier /= scale;
    // TODO report this memory consumption to mxnet?
    A_quant = (int8_t*)aligned_alloc(64, A.shape_.Size());
    CHECK(A_quant);
    A_quant_store.Reset(A_quant);
    ::intgemm::Int8::PrepareA(A_raw, A_quant, scale, A_rows, inner);
  } else {
    CHECK_EQ(A.type_flag_, mshadow::kInt8);
    A_quant = A.dptr<int8_t>();
  }
  const int8_t *B_quant = B.dptr<int8_t>();

  if (bias) {
    if (C.type_flag_ == mshadow::kFloat32) {
      ::intgemm::callbacks::UnquantizeAndAddBiasAndWrite cb(out_float_multiplier, inputs[3].dptr<float>(), C.dptr<float>());
      ::intgemm::Int8::Multiply(A_quant, B_quant, A_rows, inner, B_cols, cb);
    } else {
      // int32
      ::intgemm::callbacks::AddBiasAndWrite cb(inputs[3].dptr<int32_t>(), C.dptr<int32_t>());
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
.describe(R"code(Multiply matrices using 8-bit integers.

The data argument can be either float32 or prepared using intgemm_prepare_data.

The weight argument must be prepared using intgemm_prepare_weight.

If out_type is float32, then a scaling factor is applied before bias.  Typically this is 1/the scaling factor you provided to prepare_weight/the scaling factor you provided to prepare_data (if data is quantized).

The out_type can be int32 or float32.  Bias must have the same type.
)code" ADD_FILELINE)
.set_attr_parser(ParamParser<IntgemmFullyConnectedParam>)
.set_num_inputs([](const NodeAttrs& attrs) {
  const IntgemmFullyConnectedParam& params = nnvm::get<IntgemmFullyConnectedParam>(attrs.parsed);
  return params.no_bias ? 3 : 4;
})
.set_num_outputs(1)
.set_attr<nnvm::FListInputNames>("FListInputNames",
  [](const NodeAttrs& attrs) {
    const IntgemmFullyConnectedParam& params = nnvm::get<IntgemmFullyConnectedParam>(attrs.parsed);
    return params.no_bias ? std::vector<std::string>{"data", "weight", "scaling"} : std::vector<std::string>{"data", "weight", "scaling", "bias"};
  })
.set_attr<mxnet::FInferShape>("FInferShape", IntgemmFullyConnectedOpShape)
.set_attr<nnvm::FInferType>("FInferType", IntgemmFullyConnectedOpType)
.set_attr<FCompute>("FCompute<cpu>", IntgemmFullyConnectedOpForwardCPU)
.add_argument("data", "NDArray-or-Symbol", "First argument to multiplication. Tensor of float32 (quantized on the fly) or int8 from intgemm_prepare_data. If you use a different quantizer, be sure to ban -128. The last dimension must be a multiple of 64.")
.add_argument("weight", "NDArray-or-Symbol", "Second argument to multiplication. Tensor of int8 from intgemm_prepare_weight. The last dimension must be a multiple of 64.  The product of non-last dimensions must be a multiple of 8.")
.add_argument("scaling", "NDArray-or-Symbol", "Scaling factor to apply if output type is float32.")
.add_argument("bias", "NDArray-or-Symbol", "Bias term.")
// TODO(Xinyu): a temp solution to enable GluonCV INT8 flow,
// will be reverted after the improvement of CachedOP is done.
.set_attr<nnvm::FGradient>("FGradient", MakeZeroGradNodes)
.add_arguments(IntgemmFullyConnectedParam::__FIELDS__());

}  // namespace op
}  // namespace mxnet
