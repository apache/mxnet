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
 * Copyright (c) 2017 by Contributors
 * \file quantized_fully_connected.cc
 * \brief
 * \author Ziheng Jiang, Jun Wu
*/
#include <vector>
#include "quantization_utils.h"
#include "../nn/fully_connected-inl.h"

namespace mxnet {
namespace op {

namespace quantized_fc {
enum QuantizedfcOpResource {kTempSpace};
}

bool QuantizedFullyConnectedShape(const nnvm::NodeAttrs& attrs,
                                  std::vector<TShape> *in_shape,
                                  std::vector<TShape> *out_shape) {
  const FullyConnectedParam& param = nnvm::get<FullyConnectedParam>(attrs.parsed);
  CHECK(param.flatten) << "QuantizedFullyConnectedOp only supports flatten=true for now";
  using namespace mshadow;
  uint32_t num_inputs = param.no_bias ? 2 : 3;
  CHECK_EQ(in_shape->size(), num_inputs * 3);
  CHECK_EQ(out_shape->size(), 3U);

  CHECK(!shape_is_none(in_shape->at(0)))
    << "QuantizedFullyConnectedOp input data shape must be given";
  const TShape& dshape = in_shape->at(0);
  TShape wshape = Shape2(param.num_hidden, dshape.ProdShape(1, dshape.ndim()));
  SHAPE_ASSIGN_CHECK(*in_shape, 1, wshape);
  if (!param.no_bias) {
    TShape bshape = Shape1(param.num_hidden);
    SHAPE_ASSIGN_CHECK(*in_shape, 2, bshape);
  }

  for (size_t i = num_inputs; i < 3 * num_inputs; ++i) {
    SHAPE_ASSIGN_CHECK(*in_shape, i, TShape{1});
  }

  SHAPE_ASSIGN_CHECK(*out_shape, 0, TShape({dshape[0], wshape[0]}));
  SHAPE_ASSIGN_CHECK(*out_shape, 1, TShape({1}));
  SHAPE_ASSIGN_CHECK(*out_shape, 2, TShape({1}));
  return true;
}

bool QuantizedFullyConnectedType(const nnvm::NodeAttrs& attrs,
                                 std::vector<int> *in_type,
                                 std::vector<int> *out_type) {
  const FullyConnectedParam& param = nnvm::get<FullyConnectedParam>(attrs.parsed);
  uint32_t num_inputs = param.no_bias ? 2 : 3;
  CHECK_EQ(in_type->size(), num_inputs * 3);
  CHECK_EQ(out_type->size(), 3U);

  for (size_t i = 0; i < num_inputs; ++i) {
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
                                        std::vector<int> *in_attrs,
                                        std::vector<int> *out_attrs) {
  *dispatch_mode = DispatchMode::kFCompute;
  if (dev_mask == mshadow::cpu::kDevMask) {
    *dispatch_mode = DispatchMode::kFComputeEx;
  }

  for (auto &v : *out_attrs) {
    v = kDefaultStorage;
    if (common::stype_string(v).compare("unknown") == 0) {
      return false;
    }
  }

  for (auto &v : *in_attrs) {
    v = kDefaultStorage;
    if (common::stype_string(v).compare("unknown") == 0) {
      return false;
    }
  }
  return true;
}

struct QuantizedSumInitKernelWithBias {
  //  init sum data with bias for matrix b (n)
  MSHADOW_XINLINE static void Map(int i, int32_t *out,
                                  const int8_t *bias, const float *min_out,
                                  const float *max_out, const float *min_bias,
                                  const float *max_bias) {
    typedef int32_t T1;
    typedef int8_t  T2;
    using mshadow::red::limits::MinValue;
    using mshadow::red::limits::MaxValue;
    float float_for_one_out_quant  =
        MaxAbs(*min_out, *max_out) / static_cast<double>(MaxValue<T1>());
    float float_for_one_bias_quant =
        MaxAbs(*min_bias, *max_bias) / static_cast<double>(MaxValue<T2>());
    if (float_for_one_out_quant != 0) {
      out[i] = bias[i] * float_for_one_bias_quant /
          float_for_one_out_quant;
    } else {
      LOG(INFO) << "float_for_one_out_quant is 0,"
                << " need to check the why MaxAbs(*min_out, *max_out) of out_data is 0!";
      out[i] = 0;
    }
  }
};


template<typename SrcType>
void QuantizedFullyConnectedForward(const nnvm::NodeAttrs& attrs,
                                    const OpContext &ctx,
                                    const std::vector<NDArray> &in_data,
                                    const std::vector<OpReqType> &req,
                                    const std::vector<NDArray> &out_data) {
#if MSHADOW_USE_MKL == 1
  const FullyConnectedParam& param = nnvm::get<FullyConnectedParam>(attrs.parsed);
  using namespace mshadow;
  using namespace mxnet_op;
  size_t num_inputs = param.no_bias ? 2 : 3;
  CHECK_EQ(in_data.size(),  num_inputs * 3);
  CHECK_EQ(out_data.size(), 3U);
  const NDArray& data = in_data[0];
  const NDArray& weight = in_data[1];
  const NDArray& out = out_data[0];
  TShape dshape = data.shape();
  TShape wshape = weight.shape();
  TShape oshape = out.shape();
  auto output_temp = out.data().dptr<int32_t>();
  auto weight_temp = weight.data().dptr<SrcType>();
  auto data_temp = data.data().dptr<SrcType>();
  const int omp_threads = mxnet::engine::OpenMP::Get()->GetRecommendedOMPThreadCount();
  const float alpha = 1.0f;
  const float beta  = 1.0f;
  const CBLAS_OFFSET offsetc = CblasFixOffset;
  const MKL_INT8 oa = 0;
  const MKL_INT8 ob = 0;
  MKL_INT32 oc = 0;
  const int m = dshape[0], n = wshape[0], k = dshape.ProdShape(1, dshape.ndim());
  Stream<cpu> *s = ctx.get_stream<cpu>();
  //  cblas_gemm_s8u8s32 required first matrix must be uint8
  //  shift data from int8(from -128 to 127) to uint8 (from 0 to 255)
  int shift = 128;
  Tensor<cpu, 1, uint8_t> shiftdata =
    ctx.requested[quantized_fc::kTempSpace].get_space_typed<cpu, 1, uint8_t>(
      Shape1(m * k), s);
  #pragma omp parallel for num_threads(omp_threads)
  for (int i = 0; i < m * k; ++i) {
    shiftdata.dptr_[i] = data_temp[i] + shift;
  }

  Kernel<QuantizationRangeForMultiplicationStruct, cpu>::Launch(s, 1,
      out_data[1].data().dptr<float>(), out_data[2].data().dptr<float>(),
      in_data[num_inputs].data().dptr<float>(), in_data[num_inputs+1].data().dptr<float>(),
      in_data[num_inputs+2].data().dptr<float>(), in_data[num_inputs+3].data().dptr<float>());
  if (!param.no_bias) {
    const NDArray& bias = in_data[2];
    Kernel<QuantizedSumInitKernelWithBias, cpu>::Launch(s, n, out.data().dptr<int32_t>(),
        bias.data().dptr<int8_t>(), out_data[1].data().dptr<float>(),
        out_data[2].data().dptr<float>(), in_data[7].data().dptr<float>(),
        in_data[8].data().dptr<float>());
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
                     weight.data().dptr<SrcType>(),
                     k,
                     ob,
                     beta,
                     out.data().dptr<int32_t>(),
                     n,
                     &oc);
#else
  LOG(FATAL) << "Quantized fully connected operator relies on cblas_gemm_s8u8s32"
             << " which is only supported by MKL BLAS."
             << " Please build MXNet with USE_BLAS=mkl to leverage this operator.";
#endif
}

NNVM_REGISTER_OP(_contrib_quantized_fully_connected)
.describe(R"code(Fully Connected operator for input, weight and bias data type of int8,
and accumulates in type int32 for the output. For each argument, two more arguments of type
float32 must be provided representing the thresholds of quantizing argument from data
type float32 to int8. The final outputs contain the convolution result in int32, and min
and max thresholds representing the threholds for quantizing the float32 output into int32.

.. Note::
    This operator only supports forward propogation. DO NOT use it in training.)code" ADD_FILELINE)
.set_num_inputs(
  [](const NodeAttrs& attrs) {
    const FullyConnectedParam& param = nnvm::get<FullyConnectedParam>(attrs.parsed);
    return param.no_bias? 6 : 9;
  })
.set_num_outputs(3)
.set_attr_parser(ParamParser<FullyConnectedParam>)
.set_attr<nnvm::FListInputNames>("FListInputNames",
  [](const NodeAttrs& attrs) {
    const FullyConnectedParam& param = nnvm::get<FullyConnectedParam>(attrs.parsed);
    if (param.no_bias) {
      return std::vector<std::string>{"data", "weight", "min_data", "max_data",
                                      "min_weight", "max_weight"};
    } else {
      return std::vector<std::string>{"data", "weight", "bias", "min_data", "max_data",
                                      "min_weight", "max_weight", "min_bias", "max_bias"};
    }
  })
.set_attr<nnvm::FListOutputNames>("FListOutputNames",
  [](const NodeAttrs& attrs) {
    return std::vector<std::string>{"output", "min_output", "max_output"};
  })
.set_attr<nnvm::FInferShape>("FInferShape", QuantizedFullyConnectedShape)
.set_attr<nnvm::FInferType>("FInferType", QuantizedFullyConnectedType)
.set_attr<FInferStorageType>("FInferStorageType", QuantizedFullyConnectedStorageType)
.set_attr<FNeedRequantize>("FNeedRequantize", [](const NodeAttrs& attrs) { return true; })
.set_attr<FComputeEx>("FComputeEx<cpu>",
    QuantizedFullyConnectedForward<int8_t>)
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
.set_attr<FQuantizedOp>("FQuantizedOp", [](const NodeAttrs& attrs) {
    nnvm::NodePtr node = nnvm::Node::Create();
    node->attrs.op = Op::Get("_contrib_quantized_fully_connected");
    node->attrs.name = "quantized_" + attrs.name;
    node->attrs.dict = attrs.dict;
    if (node->op()->attr_parser != nullptr) {
      node->op()->attr_parser(&(node->attrs));
    }
    return node;
  });
}  // namespace op
}  // namespace mxnet
