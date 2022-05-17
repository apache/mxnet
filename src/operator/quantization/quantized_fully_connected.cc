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
 * \file quantized_fully_connected.cc
 * \brief
 * \author Ziheng Jiang, Jun Wu
 */
#include <vector>
#include "quantization_utils.h"
#include "../nn/fully_connected-inl.h"
#if MXNET_USE_ONEDNN == 1
#include "../nn/dnnl/dnnl_fully_connected-inl.h"
#include "dnnl/dnnl_quantized_ops-inl.h"
#endif

namespace mxnet {
namespace op {

namespace quantized_fc {
enum QuantizedfcOpResource { kTempSpace };
}

bool QuantizedFullyConnectedShape(const nnvm::NodeAttrs& attrs,
                                  mxnet::ShapeVector* in_shape,
                                  mxnet::ShapeVector* out_shape) {
  const FullyConnectedParam& param = nnvm::get<FullyConnectedParam>(attrs.parsed);
  using namespace mshadow;
  uint32_t num_inputs = param.no_bias ? 2 : 3;
  CHECK_EQ(in_shape->size(), num_inputs * 3);
  CHECK_EQ(out_shape->size(), 3U);

  mxnet::TShape dshape = (*in_shape)[0];
  // require data ndim to be known
  if (!mxnet::ndim_is_known(dshape))
    return false;

  index_t num_input;
  if (!param.flatten) {
    num_input = dshape[dshape.ndim() - 1];
  } else {
    num_input = dshape.ProdShape(1, dshape.ndim());
  }

  mxnet::TShape wshape = Shape2(param.num_hidden, num_input);
  SHAPE_ASSIGN_CHECK(*in_shape, 1, wshape);
  if (!param.no_bias) {
    mxnet::TShape bshape = Shape1(param.num_hidden);
    SHAPE_ASSIGN_CHECK(*in_shape, 2, bshape);
  }

  for (size_t i = num_inputs; i < 3 * num_inputs; ++i) {
    SHAPE_ASSIGN_CHECK(*in_shape, i, mxnet::TShape(1, 1));
  }

  if (!param.flatten) {
    mxnet::TShape result_shape(dshape);
    result_shape[dshape.ndim() - 1] = param.num_hidden;
    SHAPE_ASSIGN_CHECK(*out_shape, 0, result_shape);
  } else {
    SHAPE_ASSIGN_CHECK(*out_shape, 0, Shape2(dshape[0], param.num_hidden));
  }
  SHAPE_ASSIGN_CHECK(*out_shape, 1, mxnet::TShape(1, 1));
  SHAPE_ASSIGN_CHECK(*out_shape, 2, mxnet::TShape(1, 1));

  if ((*out_shape)[0].ndim() > 0) {
    dshape[0] = ((*out_shape)[0])[0];
    SHAPE_ASSIGN_CHECK(*in_shape, 0, dshape);
  }
  return true;
}

bool QuantizedFullyConnectedType(const nnvm::NodeAttrs& attrs,
                                 std::vector<int>* in_type,
                                 std::vector<int>* out_type) {
  const FullyConnectedParam& param = nnvm::get<FullyConnectedParam>(attrs.parsed);
  uint32_t num_inputs              = param.no_bias ? 2 : 3;
  CHECK_EQ(in_type->size(), num_inputs * 3);
  CHECK_EQ(out_type->size(), 3U);

#if MXNET_USE_ONEDNN == 1
  CHECK(in_type->at(0) == mshadow::kInt8 || in_type->at(0) == mshadow::kUint8)
      << "QuantizedFullyConnected only supports int8/uint8 input, while " << in_type->at(0)
      << " is given.";
#else
  TYPE_ASSIGN_CHECK(*in_type, 0, mshadow::kInt8);
#endif
  for (size_t i = 1; i < num_inputs; ++i) {
    TYPE_ASSIGN_CHECK(*in_type, i, mshadow::kInt8);
  }
  for (size_t i = num_inputs; i < 3 * num_inputs; ++i) {
    TYPE_ASSIGN_CHECK(*in_type, i, mshadow::kFloat32);
  }

  TYPE_ASSIGN_CHECK(*out_type, 0, mshadow::kInt32);
  TYPE_ASSIGN_CHECK(*out_type, 1, mshadow::kFloat32);
  TYPE_ASSIGN_CHECK(*out_type, 2, mshadow::kFloat32);
  return true;
}

bool QuantizedFullyConnectedStorageType(const nnvm::NodeAttrs& attrs,
                                        const int dev_mask,
                                        DispatchMode* dispatch_mode,
                                        std::vector<int>* in_attrs,
                                        std::vector<int>* out_attrs) {
  const FullyConnectedParam& param = nnvm::get<FullyConnectedParam>(attrs.parsed);
  uint32_t num_inputs              = param.no_bias ? 2 : 3;
  CHECK_EQ(in_attrs->size(), num_inputs * 3);
  CHECK_EQ(out_attrs->size(), 3U);

#if MXNET_USE_ONEDNN == 1
  return DNNLStorageType(attrs, dev_mask, true, dispatch_mode, in_attrs, out_attrs);
#else
  *dispatch_mode = DispatchMode::kFCompute;

  for (auto& v : *out_attrs) {
    v = kDefaultStorage;
    if (common::stype_string(v).compare("unknown") == 0) {
      return false;
    }
  }

  for (auto& v : *in_attrs) {
    v = kDefaultStorage;
    if (common::stype_string(v).compare("unknown") == 0) {
      return false;
    }
  }
  return true;
#endif
}

struct QuantizedSumInitKernelWithBias {
  //  init sum data with bias for matrix b (n)
  MSHADOW_XINLINE static void Map(int i,
                                  int32_t* out,
                                  const int8_t* bias,
                                  const float* min_out,
                                  const float* max_out,
                                  const float* min_bias,
                                  const float* max_bias) {
    typedef int32_t T1;
    using T2 = int8_t;
    using mshadow::red::limits::MaxValue;
    using mshadow::red::limits::MinValue;
    float float_for_one_out_quant =
        MaxAbs(*min_out, *max_out) / static_cast<double>(MaxValue<T1>());
    float float_for_one_bias_quant =
        MaxAbs(*min_bias, *max_bias) / static_cast<double>(MaxValue<T2>());
    if (float_for_one_out_quant != 0) {
      out[i] = bias[i] * float_for_one_bias_quant / float_for_one_out_quant;
    } else {
      LOG(INFO) << "float_for_one_out_quant is 0,"
                << " need to check the why MaxAbs(*min_out, *max_out) of out_data is 0!";
      out[i] = 0;
    }
  }
};

void QuantizedFullyConnectedForwardCPU(const nnvm::NodeAttrs& attrs,
                                       const OpContext& ctx,
                                       const std::vector<TBlob>& in_data,
                                       const std::vector<OpReqType>& req,
                                       const std::vector<TBlob>& out_data) {
#if MSHADOW_USE_MKL == 1
  const FullyConnectedParam& param = nnvm::get<FullyConnectedParam>(attrs.parsed);
  using namespace mshadow;
  using namespace mxnet_op;
  Stream<cpu>* s    = ctx.get_stream<cpu>();
  size_t num_inputs = param.no_bias ? 2 : 3;
  CHECK_EQ(in_data.size(), num_inputs * 3);
  CHECK_EQ(out_data.size(), 3U);

  const mxnet::TShape& dshape = in_data[fullc::kData].shape_;
  const mxnet::TShape& wshape = in_data[fullc::kWeight].shape_;
  const mxnet::TShape& oshape = out_data[fullc::kOut].shape_;

  CHECK(in_data[fullc::kData].type_flag_ == mshadow::kInt8)
      << "QuantizedFullyConnectedForwardCPU Op only supports int8 for now, but got "
      << mxnet::op::type_string(in_data[fullc::kData].type_flag_);

  if (dshape.ndim() != 2)
    CHECK(param.flatten) << "QuantizedFullyConnectedForwardCPU only supports flatten=true "
                         << "when dshape.ndim() != 2 for now.";

  Tensor<cpu, 2, int8_t> weight = in_data[fullc::kWeight].get<cpu, 2, int8_t>(s);
  Tensor<cpu, 2, int8_t> data   = in_data[fullc::kData].get_with_shape<cpu, 2, int8_t>(
      Shape2(dshape[0], dshape.ProdShape(1, dshape.ndim())), s);
  Tensor<cpu, 2, int32_t> out = out_data[fullc::kOut].get_with_shape<cpu, 2, int32_t>(
      Shape2(oshape[0], oshape.ProdShape(1, oshape.ndim())), s);

  auto data_temp             = data.dptr_;
  auto weight_temp           = weight.dptr_;
  auto output_temp           = out.dptr_;
  const int omp_threads      = mxnet::engine::OpenMP::Get()->GetRecommendedOMPThreadCount();
  const float alpha          = 1.0f;
  const float beta           = 1.0f;
  const CBLAS_OFFSET offsetc = CblasFixOffset;
  const MKL_INT8 oa          = 0;
  const MKL_INT8 ob          = 0;
  MKL_INT32 oc               = 0;
  const int m = dshape[0], n = wshape[0], k = dshape.ProdShape(1, dshape.ndim());
  //  cblas_gemm_s8u8s32 required first matrix must be uint8
  //  shift data from int8(from -128 to 127) to uint8 (from 0 to 255)
  int shift = 128;
  Tensor<cpu, 1, uint8_t> shiftdata =
      ctx.requested[quantized_fc::kTempSpace].get_space_typed<cpu, 1, uint8_t>(Shape1(m * k), s);
#pragma omp parallel for num_threads(omp_threads)
  for (int i = 0; i < m * k; ++i) {
    shiftdata.dptr_[i] = data_temp[i] + shift;
  }

  Tensor<cpu, 1, float> min_output = out_data[quantized_fullc::kOutMin].get<cpu, 1, float>(s);
  Tensor<cpu, 1, float> max_output = out_data[quantized_fullc::kOutMax].get<cpu, 1, float>(s);
  Tensor<cpu, 1, float> min_data =
      in_data[num_inputs + quantized_fullc::kDataMin].get<cpu, 1, float>(s);
  Tensor<cpu, 1, float> max_data =
      in_data[num_inputs + quantized_fullc::kDataMax].get<cpu, 1, float>(s);
  Tensor<cpu, 1, float> min_weight =
      in_data[num_inputs + quantized_fullc::kWeightMin].get<cpu, 1, float>(s);
  Tensor<cpu, 1, float> max_weight =
      in_data[num_inputs + quantized_fullc::kWeightMax].get<cpu, 1, float>(s);

  Kernel<QuantizationRangeForS8S8MultiplicationStruct, cpu>::Launch(s,
                                                                    1,
                                                                    min_output.dptr_,
                                                                    max_output.dptr_,
                                                                    min_data.dptr_,
                                                                    max_data.dptr_,
                                                                    min_weight.dptr_,
                                                                    max_weight.dptr_);
  if (!param.no_bias) {
    Tensor<cpu, 1, int8_t> bias =
        in_data[fullc::kBias].get_with_shape<cpu, 1, int8_t>(Shape1(wshape[0]), s);
    Tensor<cpu, 1, float> min_bias =
        in_data[num_inputs + quantized_fullc::kBiasMin].get<cpu, 1, float>(s);
    Tensor<cpu, 1, float> max_bias =
        in_data[num_inputs + quantized_fullc::kBiasMax].get<cpu, 1, float>(s);

    Kernel<QuantizedSumInitKernelWithBias, cpu>::Launch(s,
                                                        n,
                                                        out.dptr_,
                                                        bias.dptr_,
                                                        min_output.dptr_,
                                                        max_output.dptr_,
                                                        min_bias.dptr_,
                                                        max_bias.dptr_);
  } else {
#pragma omp parallel for num_threads(omp_threads)
    for (int i = 0; i < m * n; ++i) {
      output_temp[i] = 0;
    }
  }
#pragma omp parallel for num_threads(omp_threads)
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < k; ++j) {
      output_temp[i] -= shift * weight_temp[i * k + j];
    }
  }
#pragma omp parallel for num_threads(omp_threads)
  for (int i = n; i < m * n; ++i) {
    output_temp[i] = output_temp[i % n];
  }
  cblas_gemm_s8u8s32(CblasRowMajor,
                     CblasNoTrans,
                     CblasTrans,
                     offsetc,
                     m,
                     n,
                     k,
                     alpha,
                     shiftdata.dptr_,
                     k,
                     oa,
                     weight.dptr_,
                     k,
                     ob,
                     beta,
                     out.dptr_,
                     n,
                     &oc);
#else
  LOG(FATAL) << "Quantized fully connected operator relies on cblas_gemm_s8u8s32"
             << " which is only supported by MKL BLAS."
             << " Please build MXNet with USE_BLAS=mkl to leverage this operator.";
#endif
}

#if MXNET_USE_ONEDNN == 1
void QuantizedFullyConnectedForwardExCPU(const nnvm::NodeAttrs& attrs,
                                         const OpContext& ctx,
                                         const std::vector<NDArray>& inputs,
                                         const std::vector<OpReqType>& req,
                                         const std::vector<NDArray>& outputs) {
  DNNL_OPCHECK_INIT(false, outputs.size(), inputs, outputs);
  DNNLRun(DNNLQuantizedFullyConnectedForward, attrs, ctx, inputs, req, outputs);
  DNNL_OPCHECK_RUN(QuantizedFullyConnectedForwardCPU, attrs, ctx, inputs, req, outputs);
}
#endif

NNVM_REGISTER_OP(_contrib_quantized_fully_connected)
    .add_alias("_npx_quantized_fully_connected")
    .describe(R"code(Fully Connected operator for input, weight and bias data type of int8,
and accumulates in type int32 for the output. For each argument, two more arguments of type
float32 must be provided representing the thresholds of quantizing argument from data
type float32 to int8. The final outputs contain the convolution result in int32, and min
and max thresholds representing the threholds for quantizing the float32 output into int32.

.. Note::
    This operator only supports forward propogation. DO NOT use it in training.)code" ADD_FILELINE)
    .set_num_inputs([](const NodeAttrs& attrs) {
      const FullyConnectedParam& param = nnvm::get<FullyConnectedParam>(attrs.parsed);
      return param.no_bias ? 6 : 9;
    })
    .set_num_outputs(3)
    .set_attr_parser(ParamParser<FullyConnectedParam>)
    .set_attr<nnvm::FListInputNames>(
        "FListInputNames",
        [](const NodeAttrs& attrs) {
          const FullyConnectedParam& param = nnvm::get<FullyConnectedParam>(attrs.parsed);
          if (param.no_bias) {
            return std::vector<std::string>{
                "data", "weight", "min_data", "max_data", "min_weight", "max_weight"};
          } else {
            return std::vector<std::string>{"data",
                                            "weight",
                                            "bias",
                                            "min_data",
                                            "max_data",
                                            "min_weight",
                                            "max_weight",
                                            "min_bias",
                                            "max_bias"};
          }
        })
    .set_attr<nnvm::FListOutputNames>(
        "FListOutputNames",
        [](const NodeAttrs& attrs) {
          return std::vector<std::string>{"output", "min_output", "max_output"};
        })
    .set_attr<mxnet::FInferShape>("FInferShape", QuantizedFullyConnectedShape)
    .set_attr<nnvm::FInferType>("FInferType", QuantizedFullyConnectedType)
    .set_attr<FInferStorageType>("FInferStorageType", QuantizedFullyConnectedStorageType)
    // TODO(Xinyu): a temp solution to enable GluonCV INT8 flow,
    // will be reverted after the improvement of CachedOP is done.
    .set_attr<nnvm::FGradient>("FGradient", MakeZeroGradNodes)
    .set_attr<FNeedRequantize>("FNeedRequantize", [](const NodeAttrs& attrs) { return true; })
    .set_attr<FCompute>("FCompute<cpu>", QuantizedFullyConnectedForwardCPU)
#if MXNET_USE_ONEDNN == 1
    .set_attr<bool>("TIsDNNL", true)
    .set_attr<FComputeEx>("FComputeEx<cpu>", QuantizedFullyConnectedForwardExCPU)
#endif
    .set_attr<FResourceRequest>("FResourceRequest",
                                [](const NodeAttrs& attrs) {
                                  return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
                                })
    .add_argument("data", "NDArray-or-Symbol", "Input data.")
    .add_argument("weight", "NDArray-or-Symbol", "weight.")
    .add_argument("bias", "NDArray-or-Symbol", "bias.")
    .add_argument("min_data", "NDArray-or-Symbol", "Minimum value of data.")
    .add_argument("max_data", "NDArray-or-Symbol", "Maximum value of data.")
    .add_argument("min_weight", "NDArray-or-Symbol", "Minimum value of weight.")
    .add_argument("max_weight", "NDArray-or-Symbol", "Maximum value of weight.")
    .add_argument("min_bias", "NDArray-or-Symbol", "Minimum value of bias.")
    .add_argument("max_bias", "NDArray-or-Symbol", "Maximum value of bias.")
    .add_arguments(FullyConnectedParam::__FIELDS__());

NNVM_REGISTER_OP(FullyConnected)
    .set_attr<FQuantizable>("FQuantizable",
                            [](const NodeAttrs& attrs) { return QuantizeType::kMust; })
    .set_attr<FQuantizedOp>("FQuantizedOp", [](const NodeAttrs& attrs) {
      nnvm::ObjectPtr node = nnvm::Node::Create();
      node->attrs.op       = Op::Get("_contrib_quantized_fully_connected");
      node->attrs.name     = "quantized_" + attrs.name;
      node->attrs.dict     = attrs.dict;
      if (node->op()->attr_parser != nullptr) {
        node->op()->attr_parser(&(node->attrs));
      }
      return node;
    });
}  // namespace op
}  // namespace mxnet
