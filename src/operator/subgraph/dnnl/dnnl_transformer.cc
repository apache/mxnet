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

#if MXNET_USE_ONEDNN == 1

#include <string>
#include <utility>
#include <vector>

#include "operator/contrib/transformer-inl.h"
#include "operator/quantization/quantization_utils.h"
#include "operator/tensor/elemwise_unary_op.h"
#include "operator/subgraph/common.h"
#include "dnnl_transformer-inl.h"

// 3 tensors within one (queries keys values)
#define QKV_NUM 3

namespace mxnet {
namespace op {

DMLC_REGISTER_PARAMETER(DNNLSelfAttParam);

template <bool with_split>
static bool SgDNNLSelfAttShape(const NodeAttrs& attrs,
                               mxnet::ShapeVector* in_shape,
                               mxnet::ShapeVector* out_shape) {
  const auto& params        = nnvm::get<DNNLSelfAttParam>(attrs.parsed);
  unsigned int in_shape_num = 1;
  auto in_shape_0           = in_shape->at(0);
  auto in_shape_1           = in_shape_0;  // with split there is only one input
  CHECK_EQ(in_shape_0.ndim(), 3U)
      << "Input queries_keys_values should be 3D in batch-seq_length-proj_dim, "
      << "but the given tensor is " << in_shape_0.ndim() << "D";

  if constexpr (!with_split) {
    in_shape_1 = in_shape->at(1);  // without split we need to consider 2nd input
    CHECK_EQ(in_shape_1.ndim(), 3U)
        << "Input queries_keys_values should be 3D in batch-seq_length-proj_dim, "
        << "but the given tensor is " << in_shape_1.ndim() << "D";
    CHECK_EQ(in_shape_0[0], in_shape_1[0]);
    CHECK_EQ(in_shape_0[2], in_shape_1[2]);
    in_shape_num = 2;
  }

  if (params.quantized) {
    CHECK_EQ(in_shape->size(), 3 * in_shape_num)
        << "Input: [queries_keys_values, min_qkv, max_qkv] "
        << "- currently have " << in_shape->size() << " inputs";
    if constexpr (with_split) {
      SHAPE_ASSIGN_CHECK(*in_shape, 1, mxnet::TShape({1}));
      SHAPE_ASSIGN_CHECK(*in_shape, 2, mxnet::TShape({1}));
    } else {
      SHAPE_ASSIGN_CHECK(*in_shape, 2, mxnet::TShape({1}));
      SHAPE_ASSIGN_CHECK(*in_shape, 3, mxnet::TShape({1}));
      SHAPE_ASSIGN_CHECK(*in_shape, 4, mxnet::TShape({1}));
      SHAPE_ASSIGN_CHECK(*in_shape, 5, mxnet::TShape({1}));
    }

    out_shape->resize(3);
    if (!params.enabled_float_output.has_value()) {
      SHAPE_ASSIGN_CHECK(*out_shape, 1, mxnet::TShape({1}));  // min output
      SHAPE_ASSIGN_CHECK(*out_shape, 2, mxnet::TShape({1}));  // max output
    }
  } else {
    CHECK_EQ(in_shape->size(), in_shape_num)
        << "Input:[queries_keys_values] - currently have " << in_shape->size() << " inputs";
    out_shape->resize(1);
  }

  SHAPE_ASSIGN_CHECK(
      *out_shape, 0, mxnet::TShape({in_shape_0[0], params.heads, in_shape_0[1], in_shape_1[1]}));
  return true;
}

template <bool with_split>
static bool SgDNNLSelfAttQKInferType(const nnvm::NodeAttrs& attrs,
                                     std::vector<int>* in_types,
                                     std::vector<int>* out_types) {
  const auto& params        = nnvm::get<DNNLSelfAttParam>(attrs.parsed);
  unsigned int in_shape_num = 1;
  if constexpr (!with_split) {
    CHECK_EQ(in_types->at(0), in_types->at(1));
    in_shape_num = 2;
  }
  if (params.quantized) {
    CHECK_EQ(in_types->size(), 3 * in_shape_num);

    if (in_types->at(0) == mshadow::kBfloat16) {
      return false;
    }

    CHECK(in_types->at(0) == mshadow::kInt8)
        << "QuantizedSelfAttentionQK only supports int8 input, while " << in_types->at(0)
        << " is given.";

    if constexpr (with_split) {
      TYPE_ASSIGN_CHECK(*in_types, 1, mshadow::kFloat32);
      TYPE_ASSIGN_CHECK(*in_types, 2, mshadow::kFloat32);
    } else {
      TYPE_ASSIGN_CHECK(*in_types, 2, mshadow::kFloat32);
      TYPE_ASSIGN_CHECK(*in_types, 3, mshadow::kFloat32);
      TYPE_ASSIGN_CHECK(*in_types, 4, mshadow::kFloat32);
      TYPE_ASSIGN_CHECK(*in_types, 5, mshadow::kFloat32);
    }

    if (params.enabled_float_output.has_value()) {
      CHECK_EQ(out_types->size(), 1U);
      TYPE_ASSIGN_CHECK(*out_types, 0, params.enabled_float_output.value());
    } else {
      CHECK_EQ(out_types->size(), 3U);
      if (params.min_calib_range.has_value() && params.max_calib_range.has_value()) {
        TYPE_ASSIGN_CHECK(*out_types, 0, mshadow::kInt8);
      } else {
        TYPE_ASSIGN_CHECK(*out_types, 0, mshadow::kInt32);
      }
      TYPE_ASSIGN_CHECK(*out_types, 1, mshadow::kFloat32);
      TYPE_ASSIGN_CHECK(*out_types, 2, mshadow::kFloat32);
    }
  } else {
    CHECK_EQ(in_types->size(), in_shape_num);
    CHECK_EQ(out_types->size(), 1U);
    if (in_types->at(0) == mshadow::kFloat32) {
      TYPE_ASSIGN_CHECK(*in_types, 0, mshadow::kFloat32);
      if constexpr (!with_split) {
        TYPE_ASSIGN_CHECK(*in_types, 1, mshadow::kFloat32);
      }
      TYPE_ASSIGN_CHECK(*out_types, 0, mshadow::kFloat32);
    } else if (in_types->at(0) == mshadow::kBfloat16) {
      if constexpr (!with_split) {
        TYPE_ASSIGN_CHECK(*in_types, 1, mshadow::kBfloat16);
      }
      if (params.enabled_float_output.has_value()) {
        TYPE_ASSIGN_CHECK(*out_types, 0, params.enabled_float_output.value());
      } else {
        TYPE_ASSIGN_CHECK(*out_types, 0, mshadow::kBfloat16);
      }
    } else {
      CHECK_EQ(in_types->at(0), -1);
      return false;
    }
  }

  return true;
}

class SgDNNLSelfAttQKOp {
 public:
  explicit SgDNNLSelfAttQKOp(const nnvm::NodeAttrs& attrs)
      : param_(nnvm::get<DNNLSelfAttParam>(attrs.parsed)) {}

  template <bool with_split>
  void Forward(const OpContext& ctx,
               const std::vector<NDArray>& inputs,
               const std::vector<OpReqType>& req,
               const std::vector<NDArray>& outputs,
               bool already_prepared);

  void Backward(const OpContext& ctx,
                const std::vector<NDArray>& inputs,
                const std::vector<OpReqType>& req,
                const std::vector<NDArray>& outputs) {
    LOG(FATAL) << "Not implemented: subgraph oneDNN self attention qk only supports "
                  "inference computation.";
  }

  template <bool with_split>
  void Initialize(const OpContext& ctx,
                  const std::vector<NDArray>& inputs,
                  const std::vector<OpReqType>& req,
                  const std::vector<NDArray>& outputs);

  bool IsInitialized() {
    return initialized_;
  }

 private:
  bool initialized_{false};
  DNNLSelfAttParam param_;
  dnnl_args_map_t args_;
  std::shared_ptr<dnnl::matmul> fwd_;
  std::shared_ptr<dnnl::memory> cached_query_mem_;
  std::shared_ptr<dnnl::memory> cached_key_mem_;
  std::shared_ptr<dnnl::memory> cached_out_mem_;
  float min_data_0_;
  float max_data_0_;
  float min_data_1_;
  float max_data_1_;
  float min_output_;
  float max_output_;
  float data_scale_0_{0.0f};
  float data_scale_1_{0.0f};
};

static OpStatePtr CreateSgDNNLSelfAttQKState(const nnvm::NodeAttrs& attrs,
                                             Context ctx,
                                             const mxnet::ShapeVector& in_shapes,
                                             const std::vector<int>& in_types) {
  return OpStatePtr::Create<SgDNNLSelfAttQKOp>(attrs);
}

template <bool with_split>
void SgDNNLSelfAttQKForward(const OpStatePtr& state_pointer,
                            const OpContext& ctx,
                            const std::vector<NDArray>& inputs,
                            const std::vector<OpReqType>& req,
                            const std::vector<NDArray>& outputs) {
  SgDNNLSelfAttQKOp& op = state_pointer.get_state<SgDNNLSelfAttQKOp>();
  bool already_prepared = false;
  if (!op.IsInitialized()) {
    op.Initialize<with_split>(ctx, inputs, req, outputs);
    already_prepared = true;
  }
  op.Forward<with_split>(ctx, inputs, req, outputs, already_prepared);
}

static bool SgDNNLSelfAttStorageType(const nnvm::NodeAttrs& attrs,
                                     const int dev_mask,
                                     DispatchMode* dispatch_mode,
                                     std::vector<int>* in_attrs,
                                     std::vector<int>* out_attrs) {
  return DNNLStorageType(attrs, dev_mask, true, dispatch_mode, in_attrs, out_attrs);
}

template <bool with_split>
void SgDNNLSelfAttQKOp::Initialize(const OpContext& ctx,
                                   const std::vector<NDArray>& inputs,
                                   const std::vector<OpReqType>& req,
                                   const std::vector<NDArray>& outputs) {
  using namespace dnnl;

  const auto in_tensor_0 = inputs[0];
  auto in_tensor_1       = in_tensor_0;  // with split there is only one input
  const auto out_tensor  = outputs[0];

  const auto in_dtype = get_dnnl_type(in_tensor_0.dtype());

  const memory::dim heads          = param_.heads;
  const memory::dim sequences      = in_tensor_0.shape()[0];
  const memory::dim qkv_seq_len_0  = in_tensor_0.shape()[1];
  const memory::dim output_lin_dim = in_tensor_0.shape()[2];
  memory::dim embed_dim            = output_lin_dim;
  if constexpr (with_split) {
    embed_dim /= QKV_NUM;
  } else {
    in_tensor_1 = inputs[1];  // without split we need to consider 2nd input
  }
  const memory::dim qkv_seq_len_1  = in_tensor_1.shape()[1];
  const memory::dim head_dim       = embed_dim / heads;
  const memory::dim batch_stride_0 = output_lin_dim * qkv_seq_len_0;
  const memory::dim batch_stride_1 = output_lin_dim * qkv_seq_len_1;

  float min_data = 0.0f;
  float max_data = 0.0f;

  const auto engine = CpuEngine::Get()->get_engine();

  memory::dims query_dims    = {sequences, heads, qkv_seq_len_0, head_dim};
  memory::dims key_dims      = {sequences, heads, head_dim, qkv_seq_len_1};
  memory::dims query_strides = {batch_stride_0, head_dim, output_lin_dim, 1};
  memory::dims key_strides   = {batch_stride_1, head_dim, 1, output_lin_dim};

  auto query_md = memory::desc(query_dims, in_dtype, query_strides);
  auto key_md   = memory::desc(key_dims, in_dtype, key_strides);

  float oscale = 1.0f;
  if (param_.quantized) {
    if constexpr (with_split) {
      min_data_0_   = inputs[1].data().dptr<float>()[0];
      max_data_0_   = inputs[2].data().dptr<float>()[0];
      data_scale_0_ = data_scale_1_ =
          GetQuantizeScale(in_tensor_0.dtype(), min_data_0_, max_data_0_);
    } else {
      min_data_0_   = inputs[2].data().dptr<float>()[0];
      max_data_0_   = inputs[3].data().dptr<float>()[0];
      min_data_1_   = inputs[4].data().dptr<float>()[0];
      max_data_1_   = inputs[5].data().dptr<float>()[0];
      data_scale_0_ = GetQuantizeScale(in_tensor_0.dtype(), min_data_0_, max_data_0_);
      data_scale_1_ = GetQuantizeScale(in_tensor_1.dtype(), min_data_1_, max_data_1_);
    }

    if (param_.min_calib_range.has_value() && param_.max_calib_range.has_value()) {
      min_output_ = param_.min_calib_range.value();
      max_output_ = param_.max_calib_range.value();
      oscale      = GetQuantizeScale(out_tensor.dtype(), min_output_, max_output_) /
               (data_scale_0_ * data_scale_1_);
    } else if (param_.enabled_float_output.has_value()) {
      oscale = 1.0f / (data_scale_0_ * data_scale_1_);
    } else {
      mshadow::Stream<cpu>* s = ctx.get_stream<cpu>();
      mxnet_op::Kernel<QuantizationRangeForS8S8MultiplicationStruct, cpu>::Launch(
          s, 1, &min_output_, &max_output_, &min_data, &max_data, &min_data, &max_data);
    }
  }

  dnnl::primitive_attr attr;
  attr.set_output_scales(0, {oscale});
  auto matmul_d  = matmul::desc(query_md, key_md, GetMemDesc(out_tensor));
  auto matmul_pd = matmul::primitive_desc(matmul_d, attr, engine);
  fwd_           = std::make_shared<matmul>(matmul_pd);

  MSHADOW_TYPE_SWITCH(inputs[0].dtype(), DType, {
    DType* query_mem_ptr = inputs[0].data().dptr<DType>();
    DType* key_mem_ptr;
    if constexpr (with_split) {
      key_mem_ptr = query_mem_ptr + embed_dim;
    } else {
      key_mem_ptr = inputs[1].data().dptr<DType>();
    }
    cached_query_mem_ = std::make_shared<memory>(query_md, engine, query_mem_ptr);
    cached_key_mem_   = std::make_shared<memory>(key_md, engine, key_mem_ptr);
  });

  MSHADOW_TYPE_SWITCH(out_tensor.dtype(), DType, {
    cached_out_mem_ =
        std::make_shared<memory>(matmul_pd.dst_desc(), engine, out_tensor.data().dptr<DType>());
  });

  args_[DNNL_ARG_SRC]     = *cached_query_mem_;
  args_[DNNL_ARG_WEIGHTS] = *cached_key_mem_;
  args_[DNNL_ARG_DST]     = *cached_out_mem_;
  initialized_            = true;
}

template <bool with_split>
void SgDNNLSelfAttQKOp::Forward(const OpContext& ctx,
                                const std::vector<NDArray>& inputs,
                                const std::vector<OpReqType>& req,
                                const std::vector<NDArray>& outputs,
                                bool already_prepared) {
  if (!already_prepared) {
    const size_t output_lin_dim = inputs[0].shape()[2];
    const size_t embed_dim      = output_lin_dim / QKV_NUM;

    MSHADOW_TYPE_SWITCH(inputs[0].dtype(), DType, {
      DType* query_mem_ptr = inputs[0].data().dptr<DType>();
      DType* key_mem_ptr;
      if constexpr (with_split) {
        key_mem_ptr = query_mem_ptr + embed_dim;
      } else {
        key_mem_ptr = inputs[1].data().dptr<DType>();
      }
      cached_query_mem_->set_data_handle(query_mem_ptr);
      cached_key_mem_->set_data_handle(key_mem_ptr);
    });

    MSHADOW_TYPE_SWITCH(outputs[0].dtype(), DType, {
      cached_out_mem_->set_data_handle(outputs[0].data().dptr<DType>());
    });
  }
  DNNLStream::Get()->RegisterPrimArgs(*fwd_, args_);
  DNNLStream::Get()->Submit();

  if (param_.quantized && !param_.enabled_float_output.has_value()) {
    float* output_min = outputs[1].data().dptr<float>();
    float* output_max = outputs[2].data().dptr<float>();
    *output_min       = min_output_;
    *output_max       = max_output_;
  }
}

template <bool with_split>
nnvm::ObjectPtr SgDNNLSelfAttQKQuantizedOp(const NodeAttrs& attrs) {
  nnvm::ObjectPtr node = nnvm::Node::Create();
  auto const& param    = nnvm::get<DNNLSelfAttParam>(attrs.parsed);
  if constexpr (with_split) {
    node->attrs.op = Op::Get("_sg_onednn_selfatt_qk_split");
  } else {
    node->attrs.op = Op::Get("_sg_onednn_selfatt_qk");
  }
  node->attrs.name              = "quantized_" + attrs.name;
  node->attrs.dict              = attrs.dict;
  node->attrs.dict["heads"]     = std::to_string(param.heads);
  node->attrs.dict["quantized"] = "True";
  node->attrs.subgraphs.reserve(attrs.subgraphs.size());
  node->attrs.subgraphs = attrs.subgraphs;
  node->op()->attr_parser(&(node->attrs));
  return node;
}

#define MXNET_OPERATOR_REGISTER_SELFATT_QK(name)                                                 \
  NNVM_REGISTER_OP(name)                                                                         \
      .set_num_outputs([](const NodeAttrs& attrs) {                                              \
        auto const& param = nnvm::get<DNNLSelfAttParam>(attrs.parsed);                           \
        if (param.quantized && !param.enabled_float_output.has_value()) {                        \
          return 3;                                                                              \
        } else {                                                                                 \
          return 1;                                                                              \
        }                                                                                        \
      })                                                                                         \
      .set_attr<nnvm::FListOutputNames>(                                                         \
          "FListOutputNames",                                                                    \
          [](const NodeAttrs& attrs) {                                                           \
            auto const& param = nnvm::get<DNNLSelfAttParam>(attrs.parsed);                       \
            std::vector<std::string> output_names{"output"};                                     \
            if (param.quantized && !param.enabled_float_output.has_value()) {                    \
              output_names.emplace_back("min_output");                                           \
              output_names.emplace_back("max_output");                                           \
            }                                                                                    \
            return output_names;                                                                 \
          })                                                                                     \
      .set_attr_parser(ParamParser<DNNLSelfAttParam>)                                            \
      .set_attr<FInferStorageType>("FInferStorageType", SgDNNLSelfAttStorageType)                \
      .set_attr<FCreateOpState>("FCreateOpState", CreateSgDNNLSelfAttQKState)                    \
      .set_attr<bool>("TIsDNNL", true)                                                           \
      .set_attr<nnvm::FGradient>("FGradient", MakeZeroGradNodes)                                 \
      .set_attr<FQuantizable>("FQuantizable",                                                    \
                              [](const NodeAttrs& attrs) { return QuantizeType::kMust; })        \
      .set_attr<FNeedRequantize>("FNeedRequantize", [](const NodeAttrs& attrs) { return true; }) \
      .add_arguments(DNNLSelfAttParam::__FIELDS__())

MXNET_OPERATOR_REGISTER_SELFATT_QK(_sg_onednn_selfatt_qk)
    .describe(R"code(_sg_onednn_selfatt_qk)code" ADD_FILELINE)
    .set_num_inputs([](const NodeAttrs& attrs) {
      auto const& param = nnvm::get<DNNLSelfAttParam>(attrs.parsed);
      if (param.quantized) {
        return 6;
      } else {
        return 2;
      }
    })
    .set_attr<nnvm::FListInputNames>("FListInputNames",
                                     [](const NodeAttrs& attrs) {
                                       auto const& param =
                                           nnvm::get<DNNLSelfAttParam>(attrs.parsed);
                                       std::vector<std::string> input_names{"queries"};
                                       input_names.emplace_back("keys");
                                       if (param.quantized) {
                                         input_names.emplace_back("min_q");
                                         input_names.emplace_back("max_q");
                                         input_names.emplace_back("min_k");
                                         input_names.emplace_back("max_k");
                                       }
                                       return input_names;
                                     })
    .set_attr<mxnet::FInferShape>("FInferShape", SgDNNLSelfAttShape<false>)
    .set_attr<nnvm::FInferType>("FInferType", SgDNNLSelfAttQKInferType<false>)
    .set_attr<FStatefulComputeEx>("FStatefulComputeEx<cpu>", SgDNNLSelfAttQKForward<false>)
    .set_attr<FQuantizedOp>("FQuantizedOp", SgDNNLSelfAttQKQuantizedOp<false>)
    .add_argument("queries", "NDArray-or-Symbol", "Interleaved queries, keys and values")
    .add_argument("keys", "NDArray-or-Symbol", "Interleaved queries, keys and values");

MXNET_OPERATOR_REGISTER_SELFATT_QK(_sg_onednn_selfatt_qk_split)
    .add_alias("_sg_mkldnn_selfatt_qk")
    .describe(R"code(_sg_onednn_selfatt_qk_split)code" ADD_FILELINE)
    .set_num_inputs([](const NodeAttrs& attrs) {
      auto const& param = nnvm::get<DNNLSelfAttParam>(attrs.parsed);
      if (param.quantized) {
        return 3;
      } else {
        return 1;
      }
    })
    .set_attr<nnvm::FListInputNames>("FListInputNames",
                                     [](const NodeAttrs& attrs) {
                                       auto const& param =
                                           nnvm::get<DNNLSelfAttParam>(attrs.parsed);
                                       std::vector<std::string> input_names{"queries_keys_values"};
                                       if (param.quantized) {
                                         input_names.emplace_back("min_qkv");
                                         input_names.emplace_back("max_qkv");
                                       }
                                       return input_names;
                                     })
    .set_attr<mxnet::FInferShape>("FInferShape", SgDNNLSelfAttShape<true>)
    .set_attr<nnvm::FInferType>("FInferType", SgDNNLSelfAttQKInferType<true>)
    .set_attr<FStatefulComputeEx>("FStatefulComputeEx<cpu>", SgDNNLSelfAttQKForward<true>)
    .set_attr<FQuantizedOp>("FQuantizedOp", SgDNNLSelfAttQKQuantizedOp<true>)
    .add_argument("query_keys_values", "NDArray-or-Symbol", "Interleaved queries, keys and values");

/**********************************_sg_onednn_selfatt_valatt**********************************/

static bool SgDNNLSelfAttValShape(const NodeAttrs& attrs,
                                  mxnet::ShapeVector* in_shape,
                                  mxnet::ShapeVector* out_shape) {
  const auto& params = nnvm::get<DNNLSelfAttParam>(attrs.parsed);
  auto att_shape     = in_shape->at(0);
  auto qkv_shape     = in_shape->at(1);

  CHECK_EQ(att_shape.ndim(), 4U)
      << "Attention maps should be 4D in batch-heads-seq_length-seq_length, "
      << "but the given tensor is " << att_shape.ndim() << "D";

  CHECK_EQ(qkv_shape.ndim(), 3U)
      << "Input queries_keys_values should be 3D in batch-seq_length-proj_dim, "
      << "but the given tensor is " << qkv_shape.ndim() << "D";

  if (params.quantized) {
    CHECK_EQ(in_shape->size(), 6U) << "Input:[attention, queries_keys_values, "
                                   << "attn_min, attn_max, qkv_min, qkv_max] - currently have "
                                   << in_shape->size() << " inputs";
    for (int i = 2; i < 6; i++) {
      SHAPE_ASSIGN_CHECK(*in_shape, i, mxnet::TShape({1}));
    }

    out_shape->resize(3);
    SHAPE_ASSIGN_CHECK(
        *out_shape,
        0,
        mxnet::TShape(
            {att_shape[0], att_shape[2], att_shape[1] * qkv_shape[2] / params.heads / QKV_NUM}));
    if (!params.enabled_float_output.has_value()) {
      SHAPE_ASSIGN_CHECK(*out_shape, 1, mxnet::TShape({1}));  // min output
      SHAPE_ASSIGN_CHECK(*out_shape, 2, mxnet::TShape({1}));  // max output
    }
  } else {
    CHECK_EQ(in_shape->size(), 2U) << "Inputs: [queries_keys_values, attention] - currently have "
                                   << in_shape->size() << " inputs";
    auto qkv_shape = in_shape->at(1);
    auto att_shape = in_shape->at(0);
    CHECK_EQ(qkv_shape.ndim(), 3U)
        << "Input queries_keys_values should be 3D in batch-seq_length-proj_dim, "
        << "but the given tensor is " << qkv_shape.ndim() << "D";
    out_shape->resize(1);
    SHAPE_ASSIGN_CHECK(
        *out_shape,
        0,
        mxnet::TShape(
            {att_shape[0], att_shape[2], att_shape[1] * qkv_shape[2] / params.heads / QKV_NUM}));
    return true;
  }

  return true;
}

static bool SgDNNLSelfAttValInferType(const nnvm::NodeAttrs& attrs,
                                      std::vector<int>* in_types,
                                      std::vector<int>* out_types) {
  const auto& params = nnvm::get<DNNLSelfAttParam>(attrs.parsed);

  if (params.quantized) {
    if (in_types->at(0) == mshadow::kBfloat16 || in_types->at(1) == mshadow::kBfloat16) {
      return false;
    }

    CHECK_EQ(in_types->size(), 6U) << "Input:[attention, queries_keys_values, min_att, max_att, "
                                      "min_qkv, max_qkv] - currently have "
                                   << in_types->size() << " inputs";

    CHECK(in_types->at(0) == mshadow::kUint8)
        << "QuantizedSelfAttentionQK only supports int8/uint8 input, while " << in_types->at(0)
        << " is given.";
    CHECK(in_types->at(1) == mshadow::kInt8 || in_types->at(1) == mshadow::kUint8)
        << "QuantizedSelfAttentionQK only supports int8/uint8 input, while " << in_types->at(1)
        << " is given.";
    for (int i = 2; i < 6; i++) {
      TYPE_ASSIGN_CHECK(*in_types, i, mshadow::kFloat32);
    }

    if (params.enabled_float_output.has_value()) {
      CHECK_EQ(out_types->size(), 1U);
      TYPE_ASSIGN_CHECK(*out_types, 0, params.enabled_float_output.value());
    } else {
      CHECK_EQ(out_types->size(), 3U);
      if (params.min_calib_range.has_value() && params.max_calib_range.has_value()) {
        TYPE_ASSIGN_CHECK(*out_types, 0, mshadow::kInt8);
      } else {
        TYPE_ASSIGN_CHECK(*out_types, 0, mshadow::kInt32);
      }
      TYPE_ASSIGN_CHECK(*out_types, 1, mshadow::kFloat32);
      TYPE_ASSIGN_CHECK(*out_types, 2, mshadow::kFloat32);
    }
  } else {
    CHECK_EQ(in_types->size(), 2U);
    CHECK_EQ(out_types->size(), 1U);
    if (in_types->at(0) == mshadow::kFloat32 || in_types->at(1) == mshadow::kFloat32) {
      TYPE_ASSIGN_CHECK(*in_types, 0, mshadow::kFloat32);
      TYPE_ASSIGN_CHECK(*in_types, 1, mshadow::kFloat32);
      TYPE_ASSIGN_CHECK(*out_types, 0, mshadow::kFloat32);
    } else if (in_types->at(0) == mshadow::kBfloat16 || in_types->at(1) == mshadow::kBfloat16) {
      TYPE_ASSIGN_CHECK(*in_types, 0, mshadow::kBfloat16);
      TYPE_ASSIGN_CHECK(*in_types, 1, mshadow::kBfloat16);
      if (params.enabled_float_output.has_value()) {
        CHECK_EQ(params.enabled_float_output.value(), mshadow::kFloat32);
        TYPE_ASSIGN_CHECK(*out_types, 0, mshadow::kFloat32);
      } else {
        TYPE_ASSIGN_CHECK(*out_types, 0, mshadow::kBfloat16);
      }
    } else {
      return false;
    }
  }

  return true;
}

nnvm::ObjectPtr SgDNNLSelfAttValAttQuantizedOp(const NodeAttrs& attrs) {
  nnvm::ObjectPtr node          = nnvm::Node::Create();
  auto const& param             = nnvm::get<DNNLSelfAttParam>(attrs.parsed);
  node->attrs.op                = Op::Get("_sg_onednn_selfatt_valatt");
  node->attrs.name              = "quantized_" + attrs.name;
  node->attrs.dict              = attrs.dict;
  node->attrs.dict["heads"]     = std::to_string(param.heads);
  node->attrs.dict["quantized"] = "True";
  node->attrs.subgraphs.reserve(attrs.subgraphs.size());
  node->attrs.subgraphs = attrs.subgraphs;
  node->op()->attr_parser(&(node->attrs));
  return node;
}

class DNNLSelfAttValAttOp {
 public:
  explicit DNNLSelfAttValAttOp(const nnvm::NodeAttrs& attrs)
      : param_(nnvm::get<DNNLSelfAttParam>(attrs.parsed)) {}

  void Forward(const OpContext& ctx,
               const std::vector<NDArray>& inputs,
               const std::vector<OpReqType>& req,
               const std::vector<NDArray>& outputs,
               bool already_prepared);

  void Backward(const OpContext& ctx,
                const std::vector<NDArray>& inputs,
                const std::vector<OpReqType>& req,
                const std::vector<NDArray>& outputs) {
    LOG(FATAL) << "Not implemented: subgraph oneDNN self attention val only supports "
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
  DNNLSelfAttParam param_;
  dnnl_args_map_t args_;
  dnnl_args_map_t reorder_args;
  std::shared_ptr<dnnl::matmul> fwd_;
  std::shared_ptr<dnnl::reorder> reorder_;
  std::shared_ptr<dnnl::memory> cached_att_mem_;
  std::shared_ptr<dnnl::memory> cached_value_mem_;
  std::shared_ptr<dnnl::memory> cached_result_mem_;
  std::shared_ptr<dnnl::memory> cached_tmp_mem_;
  std::shared_ptr<dnnl::memory> cached_transposed_mem_;  // op output
  float min_qkv_;
  float max_qkv_;
  float min_att_;
  float max_att_;
  float min_output_;
  float max_output_;
  float qkv_scale_{0.0f};
  float att_scale_{0.0f};
};

static OpStatePtr CreateDNNLSelfAttValAttState(const nnvm::NodeAttrs& attrs,
                                               Context ctx,
                                               const mxnet::ShapeVector& in_shapes,
                                               const std::vector<int>& in_types) {
  return OpStatePtr::Create<DNNLSelfAttValAttOp>(attrs);
}

static void DNNLSelfAttValAttForward(const OpStatePtr& state_pointer,
                                     const OpContext& ctx,
                                     const std::vector<NDArray>& inputs,
                                     const std::vector<OpReqType>& req,
                                     const std::vector<NDArray>& outputs) {
  DNNLSelfAttValAttOp& op = state_pointer.get_state<DNNLSelfAttValAttOp>();
  bool already_prepared   = false;
  if (!op.IsInitialized()) {
    op.Initialize(ctx, inputs, req, outputs);
    already_prepared = true;
  }
  op.Forward(ctx, inputs, req, outputs, already_prepared);
}

void DNNLSelfAttValAttOp::Initialize(const OpContext& ctx,
                                     const std::vector<NDArray>& inputs,
                                     const std::vector<OpReqType>& req,
                                     const std::vector<NDArray>& outputs) {
  using namespace dnnl;

  const auto attn_tensor = inputs[0].Reorder2Default();
  const auto qkv_tensor  = inputs[1].Reorder2Default();
  const auto out_tensor  = outputs[0];

  const auto qkv_dtype  = get_dnnl_type(qkv_tensor.dtype());
  const auto attn_dtype = get_dnnl_type(attn_tensor.dtype());

  const memory::dim heads          = param_.heads;
  const memory::dim sequences      = qkv_tensor.shape()[0];
  const memory::dim qkv_seq_len    = qkv_tensor.shape()[1];
  const memory::dim output_lin_dim = qkv_tensor.shape()[2];
  const memory::dim embed_dim      = output_lin_dim / QKV_NUM;
  const memory::dim head_dim       = embed_dim / heads;
  const memory::dim batch_stride   = output_lin_dim * qkv_seq_len;

  const auto engine = CpuEngine::Get()->get_engine();

  memory::dims attn_dims  = {sequences, heads, qkv_seq_len, qkv_seq_len};
  memory::dims value_dims = {sequences, heads, qkv_seq_len, head_dim};
  memory::dims out_dims   = {sequences, heads, qkv_seq_len, head_dim};

  // needed to make transpose on 2nd and 3rd axis with oneDNN
  memory::dims transpose_dims = {sequences, heads, qkv_seq_len, head_dim, 1};

  memory::dims value_strides = {batch_stride, head_dim, output_lin_dim, 1};

  // for attention tensor just use normal data layout,
  // for value tensor we need to use strides as input tensor consists of queries, keys and values
  const auto attn_md  = memory::desc(attn_dims, attn_dtype, memory::format_tag::abcd);
  const auto value_md = memory::desc(value_dims, qkv_dtype, value_strides);

  // result = attn * value
  // tmp = result + artificial dimension (1) - same memory ptr as result
  // transpose = transposed tmp - output
  memory::desc result_md, tmp_md, transpose_md;

  float oscale = 1.0f;
  if (param_.quantized) {
    min_att_ = inputs[2].data().dptr<float>()[0];
    max_att_ = inputs[3].data().dptr<float>()[0];
    min_qkv_ = inputs[4].data().dptr<float>()[0];
    max_qkv_ = inputs[5].data().dptr<float>()[0];

    att_scale_ = GetQuantizeScale(mshadow::kUint8, min_att_, max_att_);
    qkv_scale_ = GetQuantizeScale(mshadow::kInt8, min_qkv_, max_qkv_);

    if (param_.min_calib_range.has_value() && param_.max_calib_range.has_value()) {
      min_output_ = param_.min_calib_range.value();
      max_output_ = param_.max_calib_range.value();
      oscale      = GetQuantizeScale(out_tensor.dtype(), min_output_, max_output_) /
               (att_scale_ * qkv_scale_);
    } else if (param_.enabled_float_output.has_value()) {
      oscale = 1.0f / (att_scale_ * qkv_scale_);
    } else {
      mshadow::Stream<cpu>* s = ctx.get_stream<cpu>();
      mxnet_op::Kernel<QuantizationRangeForS8S8MultiplicationStruct, cpu>::Launch(
          s, 1, &min_output_, &max_output_, &min_att_, &max_att_, &min_qkv_, &max_qkv_);
    }
  }
  memory::data_type result_dnnl_dtype = get_dnnl_type(out_tensor.dtype());

  result_md    = memory::desc(out_dims, result_dnnl_dtype, memory::format_tag::abcd);
  tmp_md       = memory::desc(transpose_dims, result_dnnl_dtype, memory::format_tag::abcde);
  transpose_md = memory::desc(transpose_dims, result_dnnl_dtype, memory::format_tag::acbde);

  // multiply by 2 as we need to skip query and key
  const size_t value_offset = inputs[1].shape()[2] / QKV_NUM * 2;
  auto att_buffer           = inputs[0];
  if (att_buffer.IsDNNLData())
    att_buffer = att_buffer.Reorder2Default();

  MSHADOW_TYPE_SWITCH(att_buffer.dtype(), DType, {
    DType* attention_ptr = att_buffer.data().dptr<DType>();
    cached_att_mem_      = std::make_shared<memory>(attn_md, engine, attention_ptr);
  });

  MSHADOW_TYPE_SWITCH(inputs[1].dtype(), DType, {
    DType* value_mem_ptr = inputs[1].data().dptr<DType>() + value_offset;
    cached_value_mem_    = std::make_shared<memory>(value_md, engine, value_mem_ptr);
  });

  MSHADOW_TYPE_SWITCH(outputs[0].dtype(), DType, {
    cached_result_mem_ = std::make_shared<memory>(result_md, engine);
    DType* orig_buf    = reinterpret_cast<DType*>(cached_result_mem_->get_data_handle());
    cached_tmp_mem_    = std::make_shared<dnnl::memory>(tmp_md, engine, orig_buf);
    cached_transposed_mem_ =
        std::make_shared<dnnl::memory>(transpose_md, engine, outputs[0].data().dptr<DType>());
  });

  dnnl::primitive_attr attr;
  attr.set_output_scales(0, {oscale});
  auto matmul_d           = matmul::desc(attn_md, value_md, result_md);
  auto matmul_pd          = matmul::primitive_desc(matmul_d, attr, engine);
  fwd_                    = std::make_shared<matmul>(matmul_pd);
  args_[DNNL_ARG_SRC]     = *cached_att_mem_;
  args_[DNNL_ARG_WEIGHTS] = *cached_value_mem_;
  args_[DNNL_ARG_DST]     = *cached_result_mem_;

  auto reorder_pd            = dnnl::reorder::primitive_desc(engine, tmp_md, engine, transpose_md);
  reorder_                   = std::make_shared<dnnl::reorder>(reorder_pd);
  reorder_args[DNNL_ARG_SRC] = *cached_tmp_mem_;
  reorder_args[DNNL_ARG_DST] = *cached_transposed_mem_;

  initialized_ = true;
}

void DNNLSelfAttValAttOp::Forward(const OpContext& ctx,
                                  const std::vector<NDArray>& inputs,
                                  const std::vector<OpReqType>& req,
                                  const std::vector<NDArray>& outputs,
                                  bool already_prepared) {
  if (!already_prepared) {
    // multiply by 2 as we need to skip queries and keys
    const size_t value_offset = inputs[1].shape()[2] / QKV_NUM * 2;

    auto att_buffer = inputs[0];
    if (att_buffer.IsDNNLData())
      att_buffer = att_buffer.Reorder2Default();

    MSHADOW_TYPE_SWITCH(att_buffer.dtype(), DType, {
      DType* attention_ptr = att_buffer.data().dptr<DType>();
      cached_att_mem_->set_data_handle(attention_ptr);
    });

    MSHADOW_TYPE_SWITCH(inputs[1].dtype(), DType, {
      DType* qkv_ptr       = inputs[1].data().dptr<DType>();
      DType* value_mem_ptr = qkv_ptr + value_offset;
      cached_value_mem_->set_data_handle(value_mem_ptr);
    });

    MSHADOW_TYPE_SWITCH(outputs[0].dtype(), DType, {
      cached_transposed_mem_->set_data_handle(outputs[0].data().dptr<DType>());
    });
  }
  DNNLStream::Get()->RegisterPrimArgs(*fwd_, args_);
  DNNLStream::Get()->RegisterPrimArgs(*reorder_, reorder_args);
  DNNLStream::Get()->Submit();

  if (param_.quantized && !param_.enabled_float_output.has_value()) {
    float* output_min = outputs[1].data().dptr<float>();
    float* output_max = outputs[2].data().dptr<float>();

    *output_min = min_output_;
    *output_max = max_output_;
  }
}

NNVM_REGISTER_OP(_sg_onednn_selfatt_valatt)
    .add_alias("_sg_mkldnn_selfatt_valatt")
    .describe(R"code(_sg_onednn_selfatt_valatt)code" ADD_FILELINE)
    .set_num_inputs([](const NodeAttrs& attrs) {
      auto const& param = nnvm::get<DNNLSelfAttParam>(attrs.parsed);
      if (param.quantized) {
        return 6;
      } else {
        return 2;
      }
    })
    .set_num_outputs([](const NodeAttrs& attrs) {
      auto const& param = nnvm::get<DNNLSelfAttParam>(attrs.parsed);
      if (param.quantized && !param.enabled_float_output.has_value()) {
        return 3;
      } else {
        return 1;
      }
    })
    .set_attr_parser(ParamParser<DNNLSelfAttParam>)
    .set_attr<nnvm::FListInputNames>(
        "FListInputNames",
        [](const NodeAttrs& attrs) {
          auto const& param = nnvm::get<DNNLSelfAttParam>(attrs.parsed);
          std::vector<std::string> input_names{"attention", "queries_keys_values"};
          if (param.quantized) {
            input_names.emplace_back("min_attention");
            input_names.emplace_back("max_attention");

            input_names.emplace_back("min_qkv");
            input_names.emplace_back("max_qkv");
          }
          return input_names;
        })
    .set_attr<nnvm::FListOutputNames>("FListOutputNames",
                                      [](const NodeAttrs& attrs) {
                                        auto const& param =
                                            nnvm::get<DNNLSelfAttParam>(attrs.parsed);
                                        std::vector<std::string> output_names{"output"};
                                        if (param.quantized &&
                                            !param.enabled_float_output.has_value()) {
                                          output_names.emplace_back("min_output");
                                          output_names.emplace_back("max_output");
                                        }
                                        return output_names;
                                      })
    .set_attr<mxnet::FInferShape>("FInferShape", SgDNNLSelfAttValShape)
    .set_attr<nnvm::FInferType>("FInferType", SgDNNLSelfAttValInferType)
    .set_attr<FInferStorageType>("FInferStorageType", SgDNNLSelfAttStorageType)
    .set_attr<FCreateOpState>("FCreateOpState", CreateDNNLSelfAttValAttState)
    .set_attr<FStatefulComputeEx>("FStatefulComputeEx<cpu>", DNNLSelfAttValAttForward)
    .set_attr<bool>("TIsDNNL", true)
    .set_attr<nnvm::FGradient>("FGradient", MakeZeroGradNodes)
    .set_attr<FQuantizable>("FQuantizable",
                            [](const NodeAttrs& attrs) { return QuantizeType::kMust; })
    .set_attr<FQuantizedOp>("FQuantizedOp", SgDNNLSelfAttValAttQuantizedOp)
    .set_attr<FNeedRequantize>("FNeedRequantize", [](const NodeAttrs& attrs) { return true; })
    .add_argument("attention", "NDArray-or-Symbol", "Attention maps")
    .add_argument("queries_keys_values",
                  "NDArray-or-Symbol",
                  "Queries, keys and values interleaved")
    .add_arguments(DNNLSelfAttParam::__FIELDS__());

}  // namespace op
}  // namespace mxnet

#endif
