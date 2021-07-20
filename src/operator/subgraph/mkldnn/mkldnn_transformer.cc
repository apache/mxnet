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

#include <string>
#include <utility>
#include <vector>

#include "../../contrib/transformer-inl.h"
#include "../../quantization/quantization_utils.h"
#include "../../tensor/elemwise_unary_op.h"
#include "../common.h"
#include "./mkldnn_transformer-inl.h"

namespace mxnet {
namespace op {

DMLC_REGISTER_PARAMETER(MKLDNNSelfAttParam);

template <int base_num_inputs>
static bool SgMKLDNNSelfAttShape(const NodeAttrs& attrs,
                                 mxnet::ShapeVector* in_shapes,
                                 mxnet::ShapeVector* out_shapes) {
  const auto& param = nnvm::get<MKLDNNSelfAttParam>(attrs.parsed);
  if (param.quantized) {
    mxnet::ShapeVector base_in_shapes;
    mxnet::ShapeVector base_out_shapes = {out_shapes->at(0)};

    for (int i = 0; i < base_num_inputs; i++) {
      base_in_shapes.emplace_back(in_shapes->at(i));
    }
    bool ret = DefaultSubgraphOpShape(attrs, &base_in_shapes, &base_out_shapes);

    for (size_t i = 0; i < in_shapes->size(); ++i) {
      if (i < base_in_shapes.size())
        in_shapes->at(i) = base_in_shapes[i];
      else
        SHAPE_ASSIGN_CHECK(*in_shapes, i, mxnet::TShape({1}));
    }
    out_shapes->resize(3);
    out_shapes->at(0) = base_out_shapes[0];
    if (!param.enable_float_output) {
      SHAPE_ASSIGN_CHECK(*out_shapes, 1, mxnet::TShape({1}));  // min output
      SHAPE_ASSIGN_CHECK(*out_shapes, 2, mxnet::TShape({1}));  // max output
    }

    return ret;
  } else {
    return DefaultSubgraphOpShape(attrs, in_shapes, out_shapes);
  }
}

static bool SgMKLDNNSelfAttQKInferType(const nnvm::NodeAttrs& attrs,
                                       std::vector<int>* in_types,
                                       std::vector<int>* out_types) {
  const auto& param = nnvm::get<MKLDNNSelfAttParam>(attrs.parsed);
  if (param.quantized) {
    CHECK(in_types->at(0) == mshadow::kInt8)
        << "QuantizedInterleavedMatMulSelfAttQK only supports int8 input, "
           "while "
        << in_types->at(0) << " is given.";

    TYPE_ASSIGN_CHECK(*in_types, 1, mshadow::kFloat32);  // min value
    TYPE_ASSIGN_CHECK(*in_types, 2, mshadow::kFloat32);  // max value

    if (param.enable_float_output) {
      TYPE_ASSIGN_CHECK(*out_types, 0, mshadow::kFloat32);  // output
    } else {
      if (param.min_calib_range.has_value() && param.max_calib_range.has_value()) {
        TYPE_ASSIGN_CHECK(*out_types, 0, mshadow::kInt8);  // output
      } else {
        TYPE_ASSIGN_CHECK(*out_types, 0, mshadow::kInt32);  // output
      }
      TYPE_ASSIGN_CHECK(*out_types, 1, mshadow::kFloat32);  // min output
      TYPE_ASSIGN_CHECK(*out_types, 2, mshadow::kFloat32);  // max output
    }
    return true;
  } else {
    return DefaultSubgraphOpType(attrs, in_types, out_types);
  }
}

template <int base_num_inputs>
static bool SgMKLDNNSelfAttStorageType(const nnvm::NodeAttrs& attrs,
                                       const int dev_mask,
                                       DispatchMode* dispatch_mode,
                                       std::vector<int>* in_attrs,
                                       std::vector<int>* out_attrs) {
  auto const& param = nnvm::get<MKLDNNSelfAttParam>(attrs.parsed);
  if (param.quantized) {
    std::vector<int> base_in_attrs;
    std::vector<int> base_out_attrs{out_attrs->at(0)};

    for (int i = 0; i < base_num_inputs; i++) {
      base_in_attrs.emplace_back(in_attrs->at(i));
    }
    bool ret = DefaultSubgraphOpStorageType(
        attrs, dev_mask, dispatch_mode, &base_in_attrs, &base_out_attrs);

    for (size_t i = 0; i < in_attrs->size(); ++i) {
      if (i < base_in_attrs.size())
        in_attrs->at(i) = base_in_attrs[i];
      else
        type_assign(&in_attrs->at(i), mxnet::kDefaultStorage);
    }

    out_attrs->at(0) = base_out_attrs[0];
    if (!param.enable_float_output) {
      type_assign(&out_attrs->at(1), mxnet::kDefaultStorage);
      type_assign(&out_attrs->at(2), mxnet::kDefaultStorage);
    }
    return ret;
  } else {
    return DefaultSubgraphOpStorageType(attrs, dev_mask, dispatch_mode, in_attrs, out_attrs);
  }
}

class SgMKLDNNSelfAttQKOp {
 public:
  explicit SgMKLDNNSelfAttQKOp(const nnvm::NodeAttrs& attrs)
      : param_(nnvm::get<MKLDNNSelfAttParam>(attrs.parsed)) {}

  void Forward(const OpContext& ctx,
               const std::vector<NDArray>& inputs,
               const std::vector<OpReqType>& req,
               const std::vector<NDArray>& outputs);

  void Backward(const OpContext& ctx,
                const std::vector<NDArray>& inputs,
                const std::vector<OpReqType>& req,
                const std::vector<NDArray>& outputs) {
    LOG(FATAL) << "Not implemented: subgraph mkldnn fully connected only supports "
                  "inference computation.";
  }

  void Initialize(const OpContext& ctx,
                  const std::vector<NDArray>& inputs,
                  const std::vector<OpReqType>& req,
                  const std::vector<NDArray>& outputs);

  bool IsInitialized() {
    return initialized_;
  }

 private:
  bool initialized_{false};
  MKLDNNSelfAttParam param_;
  mkldnn_args_map_t args_;
  std::shared_ptr<dnnl::matmul> fwd_;
  std::shared_ptr<dnnl::memory> cached_query_mem_;
  std::shared_ptr<dnnl::memory> cached_key_mem_;
  std::shared_ptr<dnnl::memory> cached_out_mem_;
  float min_data_;
  float max_data_;
  float min_output_;
  float max_output_;
  float data_scale_{0.0f};
};

static OpStatePtr CreateSgMKLDNNSelfAttQKState(const nnvm::NodeAttrs& attrs,
                                               Context ctx,
                                               const mxnet::ShapeVector& in_shapes,
                                               const std::vector<int>& in_types) {
  return OpStatePtr::Create<SgMKLDNNSelfAttQKOp>(attrs);
}

static void SgMKLDNNSelfAttQKForward(const OpStatePtr& state_pointer,
                                     const OpContext& ctx,
                                     const std::vector<NDArray>& inputs,
                                     const std::vector<OpReqType>& req,
                                     const std::vector<NDArray>& outputs) {
  SgMKLDNNSelfAttQKOp& op = state_pointer.get_state<SgMKLDNNSelfAttQKOp>();
  if (!op.IsInitialized()) {
    op.Initialize(ctx, inputs, req, outputs);
  }
  op.Forward(ctx, inputs, req, outputs);
}

void SgMKLDNNSelfAttQKOp::Initialize(const OpContext& ctx,
                                     const std::vector<NDArray>& inputs,
                                     const std::vector<OpReqType>& req,
                                     const std::vector<NDArray>& outputs) {
  using namespace mkldnn;
  const auto qkv_tensor = inputs[0];
  const auto out_tensor = outputs[0];
  const auto qkv_dtype  = get_mkldnn_type(qkv_tensor.dtype());

  const memory::dim heads          = param_.heads;
  const memory::dim sequences      = inputs[0].shape()[1];
  const memory::dim qkv_seq_len    = inputs[0].shape()[0];
  const memory::dim output_lin_dim = inputs[0].shape()[2];
  const memory::dim embed_dim      = output_lin_dim / 3;
  const memory::dim head_dim       = embed_dim / heads;
  const memory::dim attn_batches   = heads * sequences;
  const memory::dim lead_dim       = attn_batches * 3 * head_dim;
  const memory::dim batch_stride   = 3 * head_dim;

  float min_data = 0.0f;
  float max_data = 0.0f;

  if (param_.quantized) {
    min_data_ = inputs[1].data().dptr<float>()[0];
    max_data_ = inputs[2].data().dptr<float>()[0];
  }

  const auto engine = CpuEngine::Get()->get_engine();

  memory::dims query_dims = {attn_batches, qkv_seq_len, head_dim};
  memory::dims key_dims   = {attn_batches, head_dim, qkv_seq_len};
  memory::dims out_dims   = {attn_batches, qkv_seq_len, qkv_seq_len};

  memory::dims query_strides = {batch_stride, lead_dim, 1};
  memory::dims key_strides   = {batch_stride, 1, lead_dim};

  auto query_md = memory::desc(query_dims, qkv_dtype, query_strides);
  auto key_md   = memory::desc(key_dims, qkv_dtype, key_strides);

  memory::desc out_md;

  float oscale = 1.0f;
  if (param_.quantized) {
    data_scale_ = GetQuantizeScale(qkv_tensor.dtype(), min_data_, max_data_);

    if (param_.min_calib_range.has_value() && param_.max_calib_range.has_value()) {
      min_output_ = param_.min_calib_range.value();
      max_output_ = param_.max_calib_range.value();
      oscale      = GetQuantizeScale(out_tensor.dtype(), min_output_, max_output_) /
               (data_scale_ * data_scale_);
      out_md = memory::desc(out_dims, memory::data_type::s8, memory::format_tag::abc);
    } else if (param_.enable_float_output) {
      oscale = 1.0f / (data_scale_ * data_scale_);
      out_md = dnnl::memory::desc(out_dims, memory::data_type::f32, memory::format_tag::abc);
    } else {
      mshadow::Stream<cpu>* s = ctx.get_stream<cpu>();
      mxnet_op::Kernel<QuantizationRangeForS8S8MultiplicationStruct, cpu>::Launch(
          s, 1, &min_output_, &max_output_, &min_data, &max_data, &min_data, &max_data);
      out_md = dnnl::memory::desc(out_dims, memory::data_type::s32, memory::format_tag::abc);
    }
  } else {
    out_md = dnnl::memory::desc(out_dims, memory::data_type::f32, memory::format_tag::abc);
  }
  oscale /= sqrt(static_cast<float>(head_dim));  // combine quantized scale and sqrt(head_dim)

  dnnl::primitive_attr attr;
  attr.set_output_scales(0, {oscale});
  auto matmul_d  = matmul::desc(query_md, key_md, out_md);
  auto matmul_pd = matmul::primitive_desc(matmul_d, attr, engine);

  fwd_ = std::make_shared<matmul>(matmul_pd);

  MSHADOW_TYPE_SWITCH(inputs[0].dtype(), DType, {
    DType* query_mem_ptr = inputs[0].data().dptr<DType>();
    DType* key_mem_ptr   = query_mem_ptr + head_dim;
    cached_query_mem_    = std::make_shared<memory>(query_md, engine, query_mem_ptr);
    cached_key_mem_      = std::make_shared<memory>(key_md, engine, key_mem_ptr);
  });
  MSHADOW_TYPE_SWITCH(outputs[0].dtype(), DType, {
    cached_out_mem_ = std::make_shared<memory>(out_md, engine, outputs[0].data().dptr<DType>());
  });

  args_[DNNL_ARG_SRC]     = *cached_query_mem_;
  args_[DNNL_ARG_WEIGHTS] = *cached_key_mem_;
  args_[DNNL_ARG_DST]     = *cached_out_mem_;
  initialized_            = true;
}

void SgMKLDNNSelfAttQKOp::Forward(const OpContext& ctx,
                                  const std::vector<NDArray>& inputs,
                                  const std::vector<OpReqType>& req,
                                  const std::vector<NDArray>& outputs) {
  const size_t head_dim = inputs[0].shape()[2] / 3 / param_.heads;

  MSHADOW_TYPE_SWITCH(inputs[0].dtype(), DType, {
    DType* query_mem_ptr = inputs[0].data().dptr<DType>();
    DType* key_mem_ptr   = query_mem_ptr + head_dim;
    cached_query_mem_->set_data_handle(query_mem_ptr);
    cached_key_mem_->set_data_handle(key_mem_ptr);
  });

  MSHADOW_TYPE_SWITCH(outputs[0].dtype(), DType, {
    cached_out_mem_->set_data_handle(outputs[0].data().dptr<DType>());
  });

  MKLDNNStream::Get()->RegisterPrimArgs(*fwd_, args_);
  MKLDNNStream::Get()->Submit();

  if (param_.quantized && !param_.enable_float_output) {
    float* output_min = outputs[1].data().dptr<float>();
    float* output_max = outputs[2].data().dptr<float>();

    *output_min = min_output_;
    *output_max = max_output_;
  }
}

nnvm::ObjectPtr SgMKLDNNSelfAttQKQuantizedOp(const NodeAttrs& attrs) {
  nnvm::ObjectPtr node          = nnvm::Node::Create();
  auto const& param             = nnvm::get<MKLDNNSelfAttParam>(attrs.parsed);
  node->attrs.op                = Op::Get("_sg_mkldnn_selfatt_qk");
  node->attrs.name              = "quantized_" + attrs.name;
  node->attrs.dict              = attrs.dict;
  node->attrs.dict["heads"]     = std::to_string(param.heads);
  node->attrs.dict["quantized"] = "True";
  node->attrs.subgraphs.reserve(attrs.subgraphs.size());
  for (auto sub : attrs.subgraphs) {
    node->attrs.subgraphs.push_back(sub);
  }
  node->op()->attr_parser(&(node->attrs));
  return node;
}

NNVM_REGISTER_OP(_sg_mkldnn_selfatt_qk)
    .describe(R"code(_sg_mkldnn_selfatt_qk)code" ADD_FILELINE)
    .set_num_inputs([](const NodeAttrs& attrs) {
      auto const& param = nnvm::get<MKLDNNSelfAttParam>(attrs.parsed);
      if (param.quantized) {
        return 3;
      } else {
        return 1;
      }
    })
    .set_num_outputs([](const NodeAttrs& attrs) {
      auto const& param = nnvm::get<MKLDNNSelfAttParam>(attrs.parsed);
      if (param.quantized && !param.enable_float_output) {
        return 3;
      } else {
        return 1;
      }
    })
    .set_attr_parser(ParamParser<MKLDNNSelfAttParam>)
    .set_attr<nnvm::FListInputNames>("FListInputNames",
                                     [](const NodeAttrs& attrs) {
                                       auto const& param =
                                           nnvm::get<MKLDNNSelfAttParam>(attrs.parsed);
                                       std::vector<std::string> input_names{"queries_keys_values"};
                                       if (param.quantized) {
                                         input_names.emplace_back("min_qkv");
                                         input_names.emplace_back("max_qkv");
                                       }
                                       return input_names;
                                     })
    .set_attr<nnvm::FListOutputNames>("FListOutputNames",
                                      [](const NodeAttrs& attrs) {
                                        auto const& param =
                                            nnvm::get<MKLDNNSelfAttParam>(attrs.parsed);
                                        std::vector<std::string> output_names{"output"};
                                        if (param.quantized && !param.enable_float_output) {
                                          output_names.emplace_back("min_output");
                                          output_names.emplace_back("max_output");
                                        }
                                        return output_names;
                                      })
    .set_attr<mxnet::FInferShape>("FInferShape", SgMKLDNNSelfAttShape<1>)
    .set_attr<nnvm::FInferType>("FInferType", SgMKLDNNSelfAttQKInferType)
    .set_attr<FInferStorageType>("FInferStorageType", SgMKLDNNSelfAttStorageType<1>)
    .set_attr<FCreateOpState>("FCreateOpState", CreateSgMKLDNNSelfAttQKState)
    .set_attr<FStatefulComputeEx>("FStatefulComputeEx<cpu>", SgMKLDNNSelfAttQKForward)
    .set_attr<bool>("TIsMKLDNN", true)
    .set_attr<nnvm::FGradient>("FGradient", MakeZeroGradNodes)
    .set_attr<FQuantizable>("FQuantizable",
                            [](const NodeAttrs& attrs) { return QuantizeType::kMust; })
    .set_attr<FQuantizedOp>("FQuantizedOp", SgMKLDNNSelfAttQKQuantizedOp)
    .set_attr<FNeedRequantize>("FNeedRequantize", [](const NodeAttrs& attrs) { return true; })
    .add_argument("queries_keys_values",
                  "NDArray-or-Symbol",
                  "Interleaved queries, keys and values")
    .add_arguments(MKLDNNSelfAttParam::__FIELDS__());

/**********************************_sg_mkldnn_selfatt_valatt**********************************/

static bool SgMKLDNNSelfAttValAttInferType(const nnvm::NodeAttrs& attrs,
                                           std::vector<int>* in_types,
                                           std::vector<int>* out_types) {
  const auto& param = nnvm::get<MKLDNNSelfAttParam>(attrs.parsed);
  if (param.quantized) {
    TYPE_ASSIGN_CHECK(*in_types, 0, mshadow::kInt8);   // qkv input
    TYPE_ASSIGN_CHECK(*in_types, 1, mshadow::kUint8);  // att input

    // min qkv, max qkv, min att, max att
    for (size_t i = 2; i < in_types->size(); ++i) {
      TYPE_ASSIGN_CHECK(*in_types, i, mshadow::kFloat32);
    }

    if (param.enable_float_output) {
      TYPE_ASSIGN_CHECK(*out_types, 0, mshadow::kFloat32);  // output
    } else {
      if (param.min_calib_range.has_value() && param.max_calib_range.has_value()) {
        TYPE_ASSIGN_CHECK(*out_types, 0, mshadow::kInt8);  // output
      } else {
        TYPE_ASSIGN_CHECK(*out_types, 0, mshadow::kInt32);  // output
      }
      TYPE_ASSIGN_CHECK(*out_types, 1, mshadow::kFloat32);  // min output
      TYPE_ASSIGN_CHECK(*out_types, 2, mshadow::kFloat32);  // max output
    }
    return true;
  } else {
    return DefaultSubgraphOpType(attrs, in_types, out_types);
  }
}

nnvm::ObjectPtr SgMKLDNNSelfAttValAttQuantizedOp(const NodeAttrs& attrs) {
  nnvm::ObjectPtr node          = nnvm::Node::Create();
  auto const& param             = nnvm::get<MKLDNNSelfAttParam>(attrs.parsed);
  node->attrs.op                = Op::Get("_sg_mkldnn_selfatt_valatt");
  node->attrs.name              = "quantized_" + attrs.name;
  node->attrs.dict              = attrs.dict;
  node->attrs.dict["heads"]     = std::to_string(param.heads);
  node->attrs.dict["quantized"] = "True";
  node->attrs.subgraphs.reserve(attrs.subgraphs.size());
  for (auto sub : attrs.subgraphs) {
    node->attrs.subgraphs.push_back(sub);
  }
  node->op()->attr_parser(&(node->attrs));
  return node;
}

class MKLDNNSelfAttValAttOp {
 public:
  explicit MKLDNNSelfAttValAttOp(const nnvm::NodeAttrs& attrs)
      : param_(nnvm::get<MKLDNNSelfAttParam>(attrs.parsed)) {}

  void Forward(const OpContext& ctx,
               const std::vector<NDArray>& inputs,
               const std::vector<OpReqType>& req,
               const std::vector<NDArray>& outputs);

  void Backward(const OpContext& ctx,
                const std::vector<NDArray>& inputs,
                const std::vector<OpReqType>& req,
                const std::vector<NDArray>& outputs) {
    LOG(FATAL) << "Not implemented: subgraph mkldnn fully connected only supports "
                  "inference computation.";
  }

  void Initialize(const OpContext& ctx,
                  const std::vector<NDArray>& inputs,
                  const std::vector<OpReqType>& req,
                  const std::vector<NDArray>& outputs);

  bool IsInitialized() {
    return initialized_;
  }

 private:
  bool initialized_{false};
  MKLDNNSelfAttParam param_;
  mkldnn_args_map_t args_;
  std::shared_ptr<dnnl::matmul> fwd_;
  std::shared_ptr<dnnl::memory> cached_att_mem_;
  std::shared_ptr<dnnl::memory> cached_qkv_mem_;
  std::shared_ptr<dnnl::memory> cached_out_mem_;
  float min_qkv_;
  float max_qkv_;
  float min_att_;
  float max_att_;
  float min_output_;
  float max_output_;
  float qkv_scale_{0.0f};
  float att_scale_{0.0f};
};

static OpStatePtr CreateMKLDNNSelfAttValAttState(const nnvm::NodeAttrs& attrs,
                                                 Context ctx,
                                                 const mxnet::ShapeVector& in_shapes,
                                                 const std::vector<int>& in_types) {
  return OpStatePtr::Create<MKLDNNSelfAttValAttOp>(attrs);
}

static void MKLDNNSelfAttValAttForward(const OpStatePtr& state_pointer,
                                       const OpContext& ctx,
                                       const std::vector<NDArray>& inputs,
                                       const std::vector<OpReqType>& req,
                                       const std::vector<NDArray>& outputs) {
  MKLDNNSelfAttValAttOp& op = state_pointer.get_state<MKLDNNSelfAttValAttOp>();
  if (!op.IsInitialized()) {
    op.Initialize(ctx, inputs, req, outputs);
  }
  op.Forward(ctx, inputs, req, outputs);
}

void MKLDNNSelfAttValAttOp::Initialize(const OpContext& ctx,
                                       const std::vector<NDArray>& inputs,
                                       const std::vector<OpReqType>& req,
                                       const std::vector<NDArray>& outputs) {
  const dnnl::memory::dim qkv_seq_len    = inputs[0].shape()[0];
  const dnnl::memory::dim sequences      = inputs[0].shape()[1];
  const dnnl::memory::dim output_lin_dim = inputs[0].shape()[2];
  const dnnl::memory::dim embed_dim      = output_lin_dim / 3;
  const dnnl::memory::dim head_dim       = embed_dim / param_.heads;
  const dnnl::memory::dim attn_batches   = param_.heads * sequences;
  const dnnl::memory::dim lead_dim       = attn_batches * 3 * head_dim;
  const dnnl::memory::dim batch_stride   = 3 * head_dim;

  dnnl::memory::dims att_dims = {attn_batches, qkv_seq_len, qkv_seq_len};
  dnnl::memory::dims qkv_dims = {attn_batches, qkv_seq_len, head_dim};
  dnnl::memory::dims dst_dims = {attn_batches, qkv_seq_len, head_dim};

  dnnl::memory::dims att_strides = {qkv_seq_len * qkv_seq_len, qkv_seq_len, 1};
  dnnl::memory::dims qkv_strides = {batch_stride, lead_dim, 1};

  auto att_dtype = inputs[1].dtype();
  auto qkv_dtype = inputs[0].dtype();
  auto out_dtype = outputs[0].dtype();
  auto att_md    = dnnl::memory::desc(att_dims, get_mkldnn_type(att_dtype), att_strides);
  auto qkv_md    = dnnl::memory::desc(qkv_dims, get_mkldnn_type(qkv_dtype), qkv_strides);

  dnnl::memory::desc out_md;
  dnnl::primitive_attr attr;

  float oscale = 1.0f;
  if (param_.quantized) {
    min_qkv_   = inputs[2].data().dptr<float>()[0];
    max_qkv_   = inputs[3].data().dptr<float>()[0];
    min_att_   = inputs[4].data().dptr<float>()[0];
    max_att_   = inputs[5].data().dptr<float>()[0];
    qkv_scale_ = GetQuantizeScale(qkv_dtype, min_qkv_, max_qkv_);
    att_scale_ = GetQuantizeScale(att_dtype, min_att_, max_att_);

    if (param_.min_calib_range.has_value() && param_.max_calib_range.has_value()) {
      min_output_ = param_.min_calib_range.value();
      max_output_ = param_.max_calib_range.value();

      oscale = GetQuantizeScale(out_dtype, min_output_, max_output_) / (qkv_scale_ * att_scale_);
      attr.set_output_scales(0, {oscale});
    } else if (param_.enable_float_output) {
      oscale = 1.0f / (qkv_scale_ * att_scale_);
      attr.set_output_scales(0, {oscale});
    } else {
      mshadow::Stream<cpu>* s = ctx.get_stream<cpu>();
      mxnet_op::Kernel<QuantizationRangeForS8U8MultiplicationStruct, cpu>::Launch(
          s, 1, &min_output_, &max_output_, &min_qkv_, &max_qkv_, &min_att_, &max_att_);
    }
  }
  out_md = dnnl::memory::desc(dst_dims, get_mkldnn_type(out_dtype), dnnl::memory::format_tag::bac);

  const auto engine = CpuEngine::Get()->get_engine();
  auto matmul_d     = dnnl::matmul::desc(att_md, qkv_md, out_md);
  auto matmul_pd    = dnnl::matmul::primitive_desc(matmul_d, attr, engine);

  fwd_ = std::make_shared<dnnl::matmul>(matmul_pd);

  MSHADOW_TYPE_SWITCH(att_dtype, DType, {
    DType* att_ptr  = inputs[1].data().dptr<DType>();
    cached_att_mem_ = std::make_shared<dnnl::memory>(att_md, engine, att_ptr);
  });
  MSHADOW_TYPE_SWITCH(qkv_dtype, DType, {
    DType* value_ptr = inputs[0].data().dptr<DType>() + 2 * head_dim;
    cached_qkv_mem_  = std::make_shared<dnnl::memory>(qkv_md, engine, value_ptr);
  });
  MSHADOW_TYPE_SWITCH(out_dtype, DType, {
    DType* out_ptr  = outputs[0].data().dptr<DType>();
    cached_out_mem_ = std::make_shared<dnnl::memory>(out_md, engine, out_ptr);
  });

  args_[DNNL_ARG_SRC]     = *cached_att_mem_;
  args_[DNNL_ARG_WEIGHTS] = *cached_qkv_mem_;
  args_[DNNL_ARG_DST]     = *cached_out_mem_;
  initialized_            = true;
}

void MKLDNNSelfAttValAttOp::Forward(const OpContext& ctx,
                                    const std::vector<NDArray>& inputs,
                                    const std::vector<OpReqType>& req,
                                    const std::vector<NDArray>& outputs) {
  const auto engine     = CpuEngine::Get()->get_engine();
  const size_t head_dim = inputs[0].shape()[2] / param_.heads / 3;
  MSHADOW_TYPE_SWITCH(inputs[1].dtype(), DType, {
    DType* att_ptr = inputs[1].data().dptr<DType>();
    cached_att_mem_->set_data_handle(att_ptr);
  });
  MSHADOW_TYPE_SWITCH(inputs[0].dtype(), DType, {
    DType* value_ptr = inputs[0].data().dptr<DType>() + 2 * head_dim;
    cached_qkv_mem_->set_data_handle(value_ptr);
  });
  MSHADOW_TYPE_SWITCH(outputs[0].dtype(), DType, {
    DType* out_ptr = outputs[0].data().dptr<DType>();
    cached_out_mem_->set_data_handle(out_ptr);
  });

  MKLDNNStream::Get()->RegisterPrimArgs(*fwd_, args_);
  MKLDNNStream::Get()->Submit();

  if (param_.quantized && !param_.enable_float_output) {
    float* output_min = outputs[1].data().dptr<float>();
    float* output_max = outputs[2].data().dptr<float>();

    *output_min = min_output_;
    *output_max = max_output_;
  }
}

NNVM_REGISTER_OP(_sg_mkldnn_selfatt_valatt)
    .describe(R"code(_sg_mkldnn_selfatt_valatt)code" ADD_FILELINE)
    .set_num_inputs([](const NodeAttrs& attrs) {
      auto const& param = nnvm::get<MKLDNNSelfAttParam>(attrs.parsed);
      if (param.quantized) {
        return 6;
      } else {
        return 2;
      }
    })
    .set_num_outputs([](const NodeAttrs& attrs) {
      auto const& param = nnvm::get<MKLDNNSelfAttParam>(attrs.parsed);
      if (param.quantized && !param.enable_float_output) {
        return 3;
      } else {
        return 1;
      }
    })
    .set_attr_parser(ParamParser<MKLDNNSelfAttParam>)
    .set_attr<nnvm::FListInputNames>(
        "FListInputNames",
        [](const NodeAttrs& attrs) {
          auto const& param = nnvm::get<MKLDNNSelfAttParam>(attrs.parsed);
          std::vector<std::string> input_names{"queries_keys_values", "attention"};
          if (param.quantized) {
            input_names.emplace_back("min_qkv");
            input_names.emplace_back("max_qkv");

            input_names.emplace_back("min_attention");
            input_names.emplace_back("max_attention");
          }
          return input_names;
        })
    .set_attr<nnvm::FListOutputNames>("FListOutputNames",
                                      [](const NodeAttrs& attrs) {
                                        auto const& param =
                                            nnvm::get<MKLDNNSelfAttParam>(attrs.parsed);
                                        std::vector<std::string> output_names{"output"};
                                        if (param.quantized && !param.enable_float_output) {
                                          output_names.emplace_back("min_output");
                                          output_names.emplace_back("max_output");
                                        }
                                        return output_names;
                                      })
    .set_attr<mxnet::FInferShape>("FInferShape", SgMKLDNNSelfAttShape<2>)
    .set_attr<nnvm::FInferType>("FInferType", SgMKLDNNSelfAttValAttInferType)
    .set_attr<FInferStorageType>("FInferStorageType", SgMKLDNNSelfAttStorageType<2>)
    .set_attr<FCreateOpState>("FCreateOpState", CreateMKLDNNSelfAttValAttState)
    .set_attr<FStatefulComputeEx>("FStatefulComputeEx<cpu>", MKLDNNSelfAttValAttForward)
    .set_attr<bool>("TIsMKLDNN", true)
    .set_attr<nnvm::FGradient>("FGradient", MakeZeroGradNodes)
    .set_attr<FQuantizable>("FQuantizable",
                            [](const NodeAttrs& attrs) { return QuantizeType::kMust; })
    .set_attr<FQuantizedOp>("FQuantizedOp", SgMKLDNNSelfAttValAttQuantizedOp)
    .set_attr<FNeedRequantize>("FNeedRequantize", [](const NodeAttrs& attrs) { return true; })
    .add_argument("queries_keys_values",
                  "NDArray-or-Symbol",
                  "Queries, keys and values interleaved")
    .add_argument("attention", "NDArray-or-Symbol", "Attention maps")
    .add_arguments(MKLDNNSelfAttParam::__FIELDS__());

}  // namespace op
}  // namespace mxnet

#endif
