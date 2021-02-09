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

#if MXNET_USE_MKLDNN == 1

#include <mxnet/base.h>
#include "../common.h"
#include "./mkldnn_transformer-inl.h"
#include "../../contrib/transformer-inl.h"
#include "../../quantization/quantization_utils.h"

namespace mxnet {
namespace op {

DMLC_REGISTER_PARAMETER(MKLDNNInterleavedMatMulParam);

static bool MKLDNNInterleavedMatMulSelfAttQKShape(const NodeAttrs& attrs,
                                                  mxnet::ShapeVector* in_shape,
                                                  mxnet::ShapeVector* out_shape) {
  const auto& param = nnvm::get<MKLDNNInterleavedMatMulParam>(attrs.parsed);
  auto qkv_shape = in_shape->at(0);

  CHECK_EQ(qkv_shape.ndim(), 3U)
    << "Input queries_keys_values should be 3D in seq_length-batch-proj_dim, "
    << "currently is: " << qkv_shape.ndim() << "D"; 

  if (param.quantized) {
    // CHECK_EQ(in_shape->size(), 3U) << "Input:[queries_keys_values, min_qkv, max_qkv] currently have, "
    //                               << in_shape->size() << " inputs";
    if (!param.enable_float_output) {
      out_shape->resize(3);
      SHAPE_ASSIGN_CHECK(*out_shape, 1, mxnet::TShape({1}));
      SHAPE_ASSIGN_CHECK(*out_shape, 2, mxnet::TShape({1}));
    } else {
      out_shape->resize(1);
    }
  } else {
    
    // CHECK_EQ(in_shape->size(), 1U) << "Input:[queries_keys_values] currently have, "
    //                               << in_shape->size() << " inputs";
    out_shape->resize(1);
  }

  SHAPE_ASSIGN_CHECK(*out_shape, 0,
    mxnet::TShape({param.heads * qkv_shape[1], qkv_shape[0], qkv_shape[0]}));
  return true;
}

static bool MKLDNNInterleavedMatMulSelfAttQKInferType(const nnvm::NodeAttrs &attrs,
                                                      std::vector<int> *in_types,
                                                      std::vector<int> *out_types) {
  const auto& param = nnvm::get<MKLDNNInterleavedMatMulParam>(attrs.parsed);
  if (param.quantized) {
    TYPE_ASSIGN_CHECK(*in_types, 0, mshadow::kInt8);
    TYPE_ASSIGN_CHECK(*in_types, 1, mshadow::kFloat32);
    TYPE_ASSIGN_CHECK(*in_types, 2, mshadow::kFloat32);
    if (param.with_mask) {
      TYPE_ASSIGN_CHECK(*in_types, 3, mshadow::kFloat32);
    }

    if (param.enable_float_output) {
      TYPE_ASSIGN_CHECK(*out_types, 0, mshadow::kFloat32);
    } else {
      if (param.min_calib_range.has_value() && param.max_calib_range.has_value()) {
        TYPE_ASSIGN_CHECK(*out_types, 0, mshadow::kInt8);
      } else {
        TYPE_ASSIGN_CHECK(*out_types, 0, mshadow::kInt32);
      }
      TYPE_ASSIGN_CHECK(*out_types, 1, mshadow::kFloat32);
      TYPE_ASSIGN_CHECK(*out_types, 2, mshadow::kFloat32);
    }
    return true;
  } else {
    return DefaultSubgraphOpType(attrs, in_types, out_types);
  }
}

static bool MKLDNNInterleavedMatMulSelfAttQKStorageType(const nnvm::NodeAttrs &attrs,
                                                        const int dev_mask,
                                                        DispatchMode *dispatch_mode,
                                                        std::vector<int> *in_attrs,
                                                        std::vector<int> *out_attrs) {
  auto const &param = nnvm::get<MKLDNNInterleavedMatMulParam>(attrs.parsed);
  if (param.quantized) {
    type_assign(&in_attrs->at(0), mxnet::kDefaultStorage);
    type_assign(&in_attrs->at(1), mxnet::kDefaultStorage);
    type_assign(&in_attrs->at(2), mxnet::kDefaultStorage);
    if (param.with_mask) {
      type_assign(&in_attrs->at(3), mxnet::kDefaultStorage);
    }

    type_assign(&out_attrs->at(0), mxnet::kDefaultStorage);
    if (!param.enable_float_output) {
      type_assign(&out_attrs->at(1), mxnet::kDefaultStorage);
      type_assign(&out_attrs->at(2), mxnet::kDefaultStorage);
    }
    std::vector<int> base_in_attrs{in_attrs->at(0)};
    std::vector<int> base_out_attrs{out_attrs->at(0)};
    return DefaultSubgraphOpStorageType(attrs, dev_mask, dispatch_mode,
                                        &base_in_attrs, &base_out_attrs);;
  } else {
    return DefaultSubgraphOpStorageType(attrs, dev_mask, dispatch_mode,
                                        in_attrs, out_attrs);
  }
}

static OpStatePtr CreateMKLDNNInterleavedMatMulSelfAttQKState(const nnvm::NodeAttrs &attrs,
                                                              Context ctx,
                                                              const mxnet::ShapeVector &in_shapes,
                                                              const std::vector<int> &in_types) {
  return OpStatePtr::Create<MKLDNNInterleavedMatMulSelfAttQKOp>(attrs);
}

static void MKLDNNInterleavedMatMulSelfAttQKForward(const OpStatePtr &state_pointer,
                                                    const OpContext &ctx,
                                                    const std::vector<NDArray> &inputs,
                                                    const std::vector<OpReqType> &req,
                                                    const std::vector<NDArray> &outputs) {
  MKLDNNInterleavedMatMulSelfAttQKOp &op = state_pointer.get_state<MKLDNNInterleavedMatMulSelfAttQKOp>();
  op.Forward(ctx, inputs, req, outputs);
}

void MKLDNNInterleavedMatMulSelfAttQKOp::Forward(
                                   const OpContext &ctx,
                                   const std::vector<NDArray> &inputs,
                                   const std::vector<OpReqType> &req,
                                   const std::vector<NDArray> &outputs) {
  const dnnl::memory::dim HEADS = param_.heads;
  const dnnl::memory::dim BS = inputs[0].shape()[1];
  const dnnl::memory::dim SEQ_LEN = inputs[0].shape()[0];
  const dnnl::memory::dim EMBED = inputs[0].shape()[2];

  if (!initialized_) {
    const auto engine = CpuEngine::Get()->get_engine();

    dnnl::memory::dims src1_dims = {HEADS*BS, SEQ_LEN, EMBED/HEADS/3};
    dnnl::memory::dims src2_dims = {HEADS*BS, EMBED/HEADS/3, SEQ_LEN};
    dnnl::memory::dims dst_dims  = {HEADS*BS, SEQ_LEN, SEQ_LEN};

    dnnl::memory::dims src1_strides = {3*(EMBED/HEADS/3), EMBED*BS, 1};
    dnnl::memory::dims src2_strides = {3*(EMBED/HEADS/3), 1, EMBED*BS};

    auto src1_md = param_.quantized ? dnnl::memory::desc(src1_dims, dnnl::memory::data_type::s8, src1_strides) :
                                      dnnl::memory::desc(src1_dims, dnnl::memory::data_type::f32, src1_strides);
    auto src2_md = param_.quantized ? dnnl::memory::desc(src2_dims, dnnl::memory::data_type::s8, src2_strides) :
                                      dnnl::memory::desc(src2_dims, dnnl::memory::data_type::f32, src2_strides);

    dnnl::memory::desc dst_md;
    float out_scale = 1.0f;
    if (param_.quantized) {
      const float min_data = inputs[1].data().dptr<float>()[0];
      const float max_data = inputs[2].data().dptr<float>()[0];
      const float data_scale = GetQuantizeScale(mshadow::kInt8, min_data, max_data);
      
      if (param_.min_calib_range.has_value() && param_.max_calib_range.has_value()) {
        cached_min_output_ = param_.min_calib_range.value();
        cached_max_output_ = param_.max_calib_range.value();

        out_scale = GetQuantizeScale(mshadow::kInt8, cached_min_output_, cached_max_output_) / (data_scale * data_scale);
        dst_md = dnnl::memory::desc(dst_dims, dnnl::memory::data_type::s8, dnnl::memory::format_tag::abc);
      } else if (param_.enable_float_output) {
        out_scale = 1.0f / (data_scale * data_scale);
        dst_md = dnnl::memory::desc(dst_dims, dnnl::memory::data_type::f32, dnnl::memory::format_tag::abc);
      } else {
        mshadow::Stream<cpu> *s = ctx.get_stream<cpu>();
        mxnet_op::Kernel<QuantizationRangeForS8S8MultiplicationStruct, cpu>::Launch(
              s, 1, &cached_min_output_, &cached_max_output_, &min_data, &max_data, &min_data,
              &max_data);
        dst_md = dnnl::memory::desc(dst_dims, dnnl::memory::data_type::s32, dnnl::memory::format_tag::abc);
      }
    } else {
      dst_md = dnnl::memory::desc(dst_dims, dnnl::memory::data_type::f32, dnnl::memory::format_tag::abc);
    }
    out_scale /= sqrt(static_cast<float>(EMBED/HEADS/3));

    MSHADOW_TYPE_SWITCH(inputs[0].dtype(), DType, {
      cached_data1_mem_ = std::make_shared<dnnl::memory>(src1_md, engine, inputs[0].data().dptr<DType>());
      cached_data2_mem_ = std::make_shared<dnnl::memory>(src2_md, engine, inputs[0].data().dptr<DType>() + (EMBED/HEADS/3));
    });
    MSHADOW_TYPE_SWITCH(outputs[0].dtype(), DType, {
      cached_out_mem_ = std::make_shared<dnnl::memory>(dst_md, engine, outputs[0].data().dptr<DType>());
    });

    dnnl::primitive_attr attr;
    if (param_.with_mask) {
      const int mask_index = param_.quantized ? 3 : 1;
      cached_mask_mem_ = std::make_shared<dnnl::memory>(dst_md, engine, inputs[mask_index].data().dptr<float>());
      args_[DNNL_ARG_ATTR_MULTIPLE_POST_OP(0) | DNNL_ARG_SRC_1] = *cached_mask_mem_;

      dnnl::post_ops post_op;
      post_op.append_binary(dnnl::algorithm::binary_add, dst_md);
      attr.set_post_ops(post_op);
    }
    attr.set_output_scales(0, {out_scale});
    auto matmul_d = dnnl::matmul::desc(src1_md, src2_md, dst_md);
    auto matmul_pd = dnnl::matmul::primitive_desc(matmul_d, attr, engine);

    fwd_ = std::make_shared<dnnl::matmul>(matmul_pd);

    args_[DNNL_ARG_SRC]     = *cached_data1_mem_;
    args_[DNNL_ARG_WEIGHTS] = *cached_data2_mem_;
    args_[DNNL_ARG_DST]     = *cached_out_mem_;
    initialized_ = true;
  } else {
    if (param_.with_mask) {
      const int mask_index = param_.quantized ? 3 : 1;
      cached_mask_mem_->set_data_handle(reinterpret_cast<void*>(inputs[mask_index].data().dptr<float>()));
    }
    MSHADOW_TYPE_SWITCH(inputs[0].dtype(), DType, {
      cached_data1_mem_->set_data_handle(reinterpret_cast<void*>(inputs[0].data().dptr<DType>()));
      cached_data2_mem_->set_data_handle(reinterpret_cast<void*>(inputs[0].data().dptr<DType>() + (EMBED/HEADS/3)));
    });
    MSHADOW_TYPE_SWITCH(outputs[0].dtype(), DType, {
      cached_out_mem_->set_data_handle(reinterpret_cast<void*>(outputs[0].data().dptr<DType>()));
    });
  }

  MKLDNNStream::Get()->RegisterPrimArgs(*fwd_, args_);
  MKLDNNStream::Get()->Submit();

  if (param_.quantized && !param_.enable_float_output) {
    float* min_output = outputs[1].data().dptr<float>();
    float* max_output = outputs[2].data().dptr<float>();

    *min_output = cached_min_output_;
    *max_output = cached_max_output_;
  }
}

nnvm::ObjectPtr SgMKLDNNInterleavedMatMulSelfAttQKQuantizedOp(const NodeAttrs& attrs) {
  nnvm::ObjectPtr node = nnvm::Node::Create();
  auto const &param = nnvm::get<MKLDNNInterleavedMatMulParam>(attrs.parsed);
  node->attrs.op = Op::Get("_sg_mkldnn_contrib_interleaved_matmul_selfatt_qk");
  node->attrs.name = "quantized_" + attrs.name;
  node->attrs.dict = attrs.dict;
  node->attrs.dict["heads"] = std::to_string(param.heads);
  node->attrs.dict["quantized"] = "True";
  node->attrs.subgraphs.reserve(attrs.subgraphs.size());
  for (auto sub : attrs.subgraphs) {
    node->attrs.subgraphs.push_back(sub);
  }
  node->op()->attr_parser(&(node->attrs));
  return node;
}

NNVM_REGISTER_OP(_sg_mkldnn_contrib_interleaved_matmul_selfatt_qk)
.describe(R"code(_sg_mkldnn_contrib_interleaved_matmul_selfatt_qk)code" ADD_FILELINE)
.set_num_inputs([](const NodeAttrs& attrs) {
  auto const& param = nnvm::get<MKLDNNInterleavedMatMulParam>(attrs.parsed);
  const int base_inputs = param.quantized ? 3 : 1;
  return base_inputs + param.with_mask;
})
.set_num_outputs([](const NodeAttrs& attrs) {
  auto const& param = nnvm::get<MKLDNNInterleavedMatMulParam>(attrs.parsed);
  return (param.quantized && !param.enable_float_output) ? 3 : 1;
})
.set_attr_parser(ParamParser<MKLDNNInterleavedMatMulParam>)
.set_attr<nnvm::FListInputNames>("FListInputNames", [](const NodeAttrs& attrs) {
  auto const& param = nnvm::get<MKLDNNInterleavedMatMulParam>(attrs.parsed);
  std::vector<std::string> input_names {"queries_keys_values"};
  if (param.quantized) {
    input_names.emplace_back("min_qkv");
    input_names.emplace_back("max_qkv");
  }
  if (param.with_mask) {
    input_names.emplace_back("mask");
  }
  return input_names;
})
.set_attr<nnvm::FListOutputNames>("FListOutputNames", [](const NodeAttrs& attrs) {
  auto const& param = nnvm::get<MKLDNNInterleavedMatMulParam>(attrs.parsed);
  std::vector<std::string> output_names {"output"};
  if (param.quantized && !param.enable_float_output) {
    output_names.emplace_back("min_output");
    output_names.emplace_back("max_output");
  }
  return output_names;
})
.set_attr<mxnet::FInferShape>("FInferShape", MKLDNNInterleavedMatMulSelfAttQKShape)
.set_attr<nnvm::FInferType>("FInferType", MKLDNNInterleavedMatMulSelfAttQKInferType)
.set_attr<FInferStorageType>("FInferStorageType", MKLDNNInterleavedMatMulSelfAttQKStorageType)
.set_attr<FCreateOpState>("FCreateOpState", CreateMKLDNNInterleavedMatMulSelfAttQKState)
.set_attr<FStatefulComputeEx>("FStatefulComputeEx<cpu>", MKLDNNInterleavedMatMulSelfAttQKForward)
.set_attr<bool>("TIsMKLDNN", true)
.set_attr<nnvm::FGradient>("FGradient", MakeZeroGradNodes)
.set_attr<FQuantizable>("FQuantizable", [](const NodeAttrs& attrs) {
    return QuantizeType::kMust;
})
.set_attr<FQuantizedOp>("FQuantizedOp", SgMKLDNNInterleavedMatMulSelfAttQKQuantizedOp)
.set_attr<FNeedRequantize>("FNeedRequantize", [](const NodeAttrs& attrs) { return true; })
.add_argument("queries_keys_values", "NDArray-or-Symbol", "Interleaved queries, keys and values")
.add_arguments(MKLDNNInterleavedMatMulParam::__FIELDS__());

/********************************************************************************************************/

static bool MKLDNNInterleavedMatMulSelfAttValAttShape(const NodeAttrs& attrs,
                                                mxnet::ShapeVector* in_shape,
                                                mxnet::ShapeVector* out_shape) {
  const auto& param = nnvm::get<MKLDNNInterleavedMatMulParam>(attrs.parsed);
  auto qkv_shape = in_shape->at(0);
  auto att_shape = in_shape->at(1);

  CHECK_EQ(qkv_shape.ndim(), 3U)
    << "Input queries_keys_values should be 3D in seq_length-batch-3*proj_dim, "
    << "currently is: " << qkv_shape.ndim() << "D";
  CHECK_EQ(att_shape.ndim(), 3U)
    << "Input attention should be 3D in batch-seq_length-seq_length, "
    << "currently is: " << att_shape.ndim() << "D";
  CHECK_EQ(qkv_shape[0], att_shape[1])
    << "queries_keys_values.shape[0] and attention.shape[1] should be the same, "
    << "currently are " << qkv_shape[0] << " and " << att_shape[1];
  CHECK_EQ(qkv_shape[0], att_shape[2])
    << "queries_keys_values.shape[0] and attention.shape[2] should be the same, "
    << "currently are " << qkv_shape[0] << " and " << att_shape[2];
  CHECK_EQ(qkv_shape[2] % 3, 0)
    << "queries_keys_values.shape[2] should be a multiple of 3, "
    << "currently is " << qkv_shape[2];
  
  if (param.quantized) {
    CHECK_EQ(in_shape->size(), 6U) << "Input:[queries_keys_values, attention, min_qkv, max_qkv, "
                                      "min_att, max_att] currently have, "
                                 << in_shape->size() << " inputs";
    if (!param.enable_float_output) {
      out_shape->resize(3);
      SHAPE_ASSIGN_CHECK(*out_shape, 1, mxnet::TShape({1}));
      SHAPE_ASSIGN_CHECK(*out_shape, 2, mxnet::TShape({1}));
    } else {
      out_shape->resize(1);
    }
  } else {
    CHECK_EQ(in_shape->size(), 2U) << "Input:[queries_keys_values, attention] currently have, "
                                 << in_shape->size() << " inputs";
    out_shape->resize(1);
  }
  SHAPE_ASSIGN_CHECK(*out_shape, 0,
    mxnet::TShape({qkv_shape[0], qkv_shape[1], qkv_shape[2] / 3}));
  return true;
}

static bool MKLDNNInterleavedMatMulSelfAttValAttInferType(const nnvm::NodeAttrs &attrs,
                                std::vector<int> *in_types,
                                std::vector<int> *out_types) {
  const auto& param = nnvm::get<MKLDNNInterleavedMatMulParam>(attrs.parsed);
  if (param.quantized) {
    TYPE_ASSIGN_CHECK(*in_types, 0, mshadow::kInt8);
    TYPE_ASSIGN_CHECK(*in_types, 1, mshadow::kUint8);

    TYPE_ASSIGN_CHECK(*in_types, 2, mshadow::kFloat32);
    TYPE_ASSIGN_CHECK(*in_types, 3, mshadow::kFloat32);
    TYPE_ASSIGN_CHECK(*in_types, 4, mshadow::kFloat32);
    TYPE_ASSIGN_CHECK(*in_types, 5, mshadow::kFloat32);

    if (param.enable_float_output) {
      TYPE_ASSIGN_CHECK(*out_types, 0, mshadow::kFloat32);
    } else {
      if (param.min_calib_range.has_value() && param.max_calib_range.has_value()) {
        TYPE_ASSIGN_CHECK(*out_types, 0, mshadow::kInt8);
      } else {
        TYPE_ASSIGN_CHECK(*out_types, 0, mshadow::kInt32);
      }
      TYPE_ASSIGN_CHECK(*out_types, 1, mshadow::kFloat32);
      TYPE_ASSIGN_CHECK(*out_types, 2, mshadow::kFloat32);
    }
    return true;
  } else {
    return DefaultSubgraphOpType(attrs, in_types, out_types);
  }
}

static bool MKLDNNInterleavedMatMulSelfAttValAttStorageType(const nnvm::NodeAttrs &attrs,
                                                          const int dev_mask,
                                                          DispatchMode *dispatch_mode,
                                                          std::vector<int> *in_attrs,
                                                          std::vector<int> *out_attrs) {
  auto const &param = nnvm::get<MKLDNNInterleavedMatMulParam>(attrs.parsed);
  if (param.quantized) {
    type_assign(&in_attrs->at(0), mxnet::kDefaultStorage);
    type_assign(&in_attrs->at(1), mxnet::kDefaultStorage);
    type_assign(&in_attrs->at(2), mxnet::kDefaultStorage);
    type_assign(&in_attrs->at(3), mxnet::kDefaultStorage);
    type_assign(&in_attrs->at(4), mxnet::kDefaultStorage);
    type_assign(&in_attrs->at(5), mxnet::kDefaultStorage);

    type_assign(&out_attrs->at(0), mxnet::kDefaultStorage);
    if (!param.enable_float_output) {
      type_assign(&out_attrs->at(1), mxnet::kDefaultStorage);
      type_assign(&out_attrs->at(2), mxnet::kDefaultStorage);
    }
    std::vector<int> base_in_attrs{in_attrs->at(0), in_attrs->at(1)};
    std::vector<int> base_out_attrs{out_attrs->at(0)};
    return DefaultSubgraphOpStorageType(attrs, dev_mask, dispatch_mode,
                                        &base_in_attrs, &base_out_attrs);;
  } else {
    return DefaultSubgraphOpStorageType(attrs, dev_mask, dispatch_mode,
                                        in_attrs, out_attrs);
  }
}

nnvm::ObjectPtr SgMKLDNNInterleavedMatMulSelfAttValAttQuantizedOp(const NodeAttrs& attrs) {
  nnvm::ObjectPtr node = nnvm::Node::Create();
  auto const &param = nnvm::get<MKLDNNInterleavedMatMulParam>(attrs.parsed);
  node->attrs.op = Op::Get("_sg_mkldnn_contrib_interleaved_matmul_selfatt_valatt");
  node->attrs.name = "quantized_" + attrs.name;
  node->attrs.dict = attrs.dict;
  node->attrs.dict["heads"] = std::to_string(param.heads);
  node->attrs.dict["quantized"] = "True";
  node->attrs.subgraphs.reserve(attrs.subgraphs.size());
  for (auto sub : attrs.subgraphs) {
    node->attrs.subgraphs.push_back(sub);
  }
  node->op()->attr_parser(&(node->attrs));
  return node;
}

static OpStatePtr CreateMKLDNNInterleavedMatMulSelfAttValAttState(const nnvm::NodeAttrs &attrs,
                                                              Context ctx,
                                                              const mxnet::ShapeVector &in_shapes,
                                                              const std::vector<int> &in_types) {
  return OpStatePtr::Create<MKLDNNInterleavedMatMulSelfAttValAttOp>(attrs);
}

static void MKLDNNInterleavedMatMulSelfAttValAttForward(const OpStatePtr &state_pointer,
                              const OpContext &ctx,
                              const std::vector<NDArray> &inputs,
                              const std::vector<OpReqType> &req,
                              const std::vector<NDArray> &outputs) {
  MKLDNNInterleavedMatMulSelfAttValAttOp &op = state_pointer.get_state<MKLDNNInterleavedMatMulSelfAttValAttOp>();
  op.Forward(ctx, inputs, req, outputs);
}

void MKLDNNInterleavedMatMulSelfAttValAttOp::Forward(
                                   const OpContext &ctx,
                                   const std::vector<NDArray> &inputs,
                                   const std::vector<OpReqType> &req,
                                   const std::vector<NDArray> &outputs) {
  const dnnl::memory::dim HEADS = param_.heads;
  const dnnl::memory::dim BS = inputs[0].shape()[1];
  const dnnl::memory::dim SEQ_LEN = inputs[0].shape()[0];
  const dnnl::memory::dim EMBED = inputs[0].shape()[2];

  if (!initialized_) {
    const auto engine = CpuEngine::Get()->get_engine();

    dnnl::memory::dims src1_dims = {BS*HEADS, SEQ_LEN, SEQ_LEN};
    dnnl::memory::dims src2_dims = {BS*HEADS, SEQ_LEN, EMBED/HEADS/3};
    dnnl::memory::dims dst_dims = {BS*HEADS, SEQ_LEN, EMBED/HEADS/3};

    dnnl::memory::dims src1_strides = {SEQ_LEN*SEQ_LEN, SEQ_LEN, 1};
    dnnl::memory::dims src2_strides = {3*(EMBED/HEADS/3), EMBED*BS, 1};

    auto src1_md = param_.quantized ? dnnl::memory::desc(src1_dims, dnnl::memory::data_type::u8, src1_strides) :
                                      dnnl::memory::desc(src1_dims, dnnl::memory::data_type::f32, src1_strides);
    auto src2_md = param_.quantized ? dnnl::memory::desc(src2_dims, dnnl::memory::data_type::s8, src2_strides) :
                                      dnnl::memory::desc(src2_dims, dnnl::memory::data_type::f32, src2_strides);

    dnnl::memory::desc dst_md;
    
    float out_scale = 1.0f;
    if (param_.quantized) {
      const float min_qkv = inputs[2].data().dptr<float>()[0];
      const float max_qkv = inputs[3].data().dptr<float>()[0];
      const float min_att = inputs[4].data().dptr<float>()[0];
      const float max_att = inputs[5].data().dptr<float>()[0];

      const float qkv_scale = GetQuantizeScale(mshadow::kInt8, min_qkv, max_qkv);
      const float att_scale = GetQuantizeScale(mshadow::kUint8, min_att, max_att);

      if (param_.min_calib_range.has_value() &&
          param_.max_calib_range.has_value()) {
        cached_min_output_ = param_.min_calib_range.value();
        cached_max_output_ = param_.max_calib_range.value();

        out_scale = GetQuantizeScale(mshadow::kInt8, cached_min_output_, cached_max_output_)  / (qkv_scale * att_scale);
        dst_md = dnnl::memory::desc(dst_dims, dnnl::memory::data_type::s8, dnnl::memory::format_tag::bac);
      } else if (param_.enable_float_output) {
        out_scale = 1.0f / (qkv_scale * att_scale);
        dst_md = dnnl::memory::desc(dst_dims, dnnl::memory::data_type::f32, dnnl::memory::format_tag::bac);
      } else {
        mshadow::Stream<cpu> *s = ctx.get_stream<cpu>();
        mxnet_op::Kernel<QuantizationRangeForS8U8MultiplicationStruct, cpu>::Launch(
                s, 1, &cached_min_output_, &cached_max_output_, &min_qkv, &max_qkv, &min_att,
                &max_att);
        dst_md = dnnl::memory::desc(dst_dims, dnnl::memory::data_type::s32, dnnl::memory::format_tag::bac);
      }
    } else {
      dst_md = dnnl::memory::desc(dst_dims, dnnl::memory::data_type::f32, dnnl::memory::format_tag::bac);
    }

    dnnl::primitive_attr attr;
    attr.set_output_scales(0, {out_scale});
    auto matmul_d = dnnl::matmul::desc(src1_md, src2_md, dst_md);
    auto matmul_pd = dnnl::matmul::primitive_desc(matmul_d, attr, engine);

    fwd_ = std::make_shared<dnnl::matmul>(matmul_pd);

    MSHADOW_TYPE_SWITCH(inputs[1].dtype(), DType, {
      cached_data1_mem_ = std::make_shared<dnnl::memory>(src1_md, engine, inputs[1].data().dptr<DType>());
    });
    MSHADOW_TYPE_SWITCH(inputs[0].dtype(), DType, {
      cached_data2_mem_ = std::make_shared<dnnl::memory>(src2_md, engine, inputs[0].data().dptr<DType>() + 2*(EMBED/HEADS/3));
    });
    MSHADOW_TYPE_SWITCH(outputs[0].dtype(), DType, {
      cached_out_mem_ = std::make_shared<dnnl::memory>(dst_md, engine, outputs[0].data().dptr<DType>());
    });

    args_[DNNL_ARG_SRC]     = *cached_data1_mem_;
    args_[DNNL_ARG_WEIGHTS] = *cached_data2_mem_;
    args_[DNNL_ARG_DST]     = *cached_out_mem_;
    initialized_ = true;
  } else {
    MSHADOW_TYPE_SWITCH(inputs[1].dtype(), DType, {
      cached_data1_mem_->set_data_handle(reinterpret_cast<void*>(inputs[1].data().dptr<DType>()));
    });
    MSHADOW_TYPE_SWITCH(inputs[0].dtype(), DType, {
      cached_data2_mem_->set_data_handle(reinterpret_cast<void*>(inputs[0].data().dptr<DType>() + 2*(EMBED/HEADS/3)));
    });
    MSHADOW_TYPE_SWITCH(outputs[0].dtype(), DType, {
      cached_out_mem_->set_data_handle(reinterpret_cast<void*>(outputs[0].data().dptr<DType>()));
    });
  }
  MKLDNNStream::Get()->RegisterPrimArgs(*fwd_, args_);
  MKLDNNStream::Get()->Submit();

  if (param_.quantized && !param_.enable_float_output) {
    float* min_output = outputs[1].data().dptr<float>();
    float* max_output = outputs[2].data().dptr<float>();

    *min_output = cached_min_output_;
    *max_output = cached_max_output_;
  }
}

NNVM_REGISTER_OP(_sg_mkldnn_contrib_interleaved_matmul_selfatt_valatt)
.describe(R"code(_sg_mkldnn_contrib_interleaved_matmul_selfatt_valatt)code" ADD_FILELINE)
.set_num_inputs([](const NodeAttrs& attrs) {
  auto const& param = nnvm::get<MKLDNNInterleavedMatMulParam>(attrs.parsed);
  return param.quantized ? 6 : 2;
})
.set_num_outputs([](const NodeAttrs& attrs) {
  auto const& param = nnvm::get<MKLDNNInterleavedMatMulParam>(attrs.parsed);
  return (param.quantized && !param.enable_float_output) ? 3 : 1;
})
.set_attr_parser(ParamParser<MKLDNNInterleavedMatMulParam>)
.set_attr<nnvm::FListInputNames>("FListInputNames", [](const NodeAttrs& attrs) {
  auto const& param = nnvm::get<MKLDNNInterleavedMatMulParam>(attrs.parsed);
  std::vector<std::string> input_names {"queries_keys_values", "attention"};
  if (param.quantized) {
    input_names.emplace_back("min_qkv");
    input_names.emplace_back("max_qkv");
    input_names.emplace_back("min_att");
    input_names.emplace_back("max_att");
  }
  return input_names;
})
.set_attr<nnvm::FListOutputNames>("FListOutputNames", [](const NodeAttrs& attrs) {
  auto const& param = nnvm::get<MKLDNNInterleavedMatMulParam>(attrs.parsed);
  std::vector<std::string> output_names {"output"};
  if (param.quantized && !param.enable_float_output) {
    output_names.emplace_back("min_output");
    output_names.emplace_back("max_output");
  }
  return output_names;
})
.set_attr<mxnet::FInferShape>("FInferShape", MKLDNNInterleavedMatMulSelfAttValAttShape)
.set_attr<nnvm::FInferType>("FInferType", MKLDNNInterleavedMatMulSelfAttValAttInferType)
.set_attr<FInferStorageType>("FInferStorageType", MKLDNNInterleavedMatMulSelfAttValAttStorageType)
.set_attr<FCreateOpState>("FCreateOpState", CreateMKLDNNInterleavedMatMulSelfAttValAttState)
.set_attr<FStatefulComputeEx>("FStatefulComputeEx<cpu>", MKLDNNInterleavedMatMulSelfAttValAttForward)
.set_attr<bool>("TIsMKLDNN", true)
.set_attr<nnvm::FGradient>("FGradient", MakeZeroGradNodes)
.set_attr<FQuantizable>("FQuantizable", [](const NodeAttrs& attrs) {
    return QuantizeType::kMust;
})
.set_attr<FQuantizedOp>("FQuantizedOp", SgMKLDNNInterleavedMatMulSelfAttValAttQuantizedOp)
.set_attr<FNeedRequantize>("FNeedRequantize", [](const NodeAttrs& attrs) { return true; })
.add_argument("queries_keys_values", "NDArray-or-Symbol", "Queries, keys and values interleaved")
.add_argument("attention", "NDArray-or-Symbol", "Attention maps")
.add_arguments(MKLDNNInterleavedMatMulParam::__FIELDS__());

}  // namespace op
}  // namespace mxnet

#endif