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

#include <utility>
#include <vector>
#include <string>
#include <mxnet/base.h>
#include "../common.h"
#include "./mkldnn_transformer-inl.h"
#include "../../contrib/transformer-inl.h"
#include "../../tensor/elemwise_unary_op.h"

#include "../../quantization/quantization_utils.h"

namespace mxnet {
namespace op {

DMLC_REGISTER_PARAMETER(MKLDNNInterleavedMatMulParam);

static bool SgMKLDNNSelfAttQKShape(const NodeAttrs& attrs,
                                   mxnet::ShapeVector* in_shapes,
                                   mxnet::ShapeVector* out_shapes) {
  const auto& param = nnvm::get<MKLDNNInterleavedMatMulParam>(attrs.parsed);
  if (param.quantized) {
    mxnet::ShapeVector base_in_shapes  = {in_shapes->at(0)};
    mxnet::ShapeVector base_out_shapes = {out_shapes->at(0)};
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
      SHAPE_ASSIGN_CHECK(*out_shapes, 1, mxnet::TShape({1}));      // min output
      SHAPE_ASSIGN_CHECK(*out_shapes, 2, mxnet::TShape({1}));      // max output
    }

    return ret;
  } else {
    return DefaultSubgraphOpShape(attrs, in_shapes, out_shapes);
  }
}

static bool SgMKLDNNSelfAttQKInferType(const nnvm::NodeAttrs &attrs,
                                       std::vector<int> *in_types,
                                       std::vector<int> *out_types) {
  const auto& param = nnvm::get<MKLDNNInterleavedMatMulParam>(attrs.parsed);
  if (param.quantized) {
     CHECK(in_types->at(0) == mshadow::kInt8)
        << "QuantizedInterleavedMatMulSelfAttQK only supports int8 input, while "
        << in_types->at(0) << " is given.";

    TYPE_ASSIGN_CHECK(*in_types, 1, mshadow::kFloat32); // min value
    TYPE_ASSIGN_CHECK(*in_types, 2, mshadow::kFloat32); // max value

    if (param.enable_float_output) {
      TYPE_ASSIGN_CHECK(*out_types, 0, mshadow::kFloat32);     // output
    } else {
      if (param.min_calib_range.has_value() && param.max_calib_range.has_value()) {
        TYPE_ASSIGN_CHECK(*out_types, 0, mshadow::kInt8);      // output
      } else {
        TYPE_ASSIGN_CHECK(*out_types, 0, mshadow::kInt32);     // output
      }
      TYPE_ASSIGN_CHECK(*out_types, 1, mshadow::kFloat32);     // min output
      TYPE_ASSIGN_CHECK(*out_types, 2, mshadow::kFloat32);     // max output
    }
    return true;
  } else {
    return DefaultSubgraphOpType(attrs, in_types, out_types);
  }
}

static bool SgMKLDNNSelfAttQKStorageType(const nnvm::NodeAttrs &attrs,
                                         const int dev_mask,
                                         DispatchMode *dispatch_mode,
                                         std::vector<int> *in_attrs,
                                         std::vector<int> *out_attrs) {
  auto const &param = nnvm::get<MKLDNNInterleavedMatMulParam>(attrs.parsed);
  if (param.quantized) {
    std::vector<int> base_in_attrs{in_attrs->at(0)};
    std::vector<int> base_out_attrs{out_attrs->at(0)};
    bool ret = DefaultSubgraphOpStorageType(attrs, dev_mask, dispatch_mode,
                                            &base_in_attrs, &base_out_attrs);

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
    return DefaultSubgraphOpStorageType(attrs, dev_mask, dispatch_mode,
                                        in_attrs, out_attrs);
  }
}

class SgMKLDNNSelfAttQKOp {
 public:
  explicit SgMKLDNNSelfAttQKOp(const nnvm::NodeAttrs &attrs) :
    param_(nnvm::get<MKLDNNInterleavedMatMulParam>(attrs.parsed)) {}

  void Forward(const OpContext &ctx,
               const std::vector<NDArray> &inputs,
               const std::vector<OpReqType> &req,
               const std::vector<NDArray> &outputs);

  void Backward(const OpContext &ctx,
                const std::vector<NDArray> &inputs,
                const std::vector<OpReqType> &req,
                const std::vector<NDArray> &outputs) {
    LOG(FATAL) << "Not implemented: subgraph mkldnn fully connected only supports "
                  "inference computation.";
  }

  void Initialize(const OpContext &ctx,
                  const std::vector<NDArray> &inputs,
                  const std::vector<OpReqType> &req,
                  const std::vector<NDArray> &outputs);

  bool IsInitialized() {
    return initialized_;
  }

 private:
  bool initialized_{false};
  MKLDNNInterleavedMatMulParam param_;
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

static OpStatePtr CreateSgMKLDNNSelfAttQKState(const nnvm::NodeAttrs &attrs,
                                               Context ctx,
                                               const mxnet::ShapeVector &in_shapes,
                                               const std::vector<int> &in_types) {
  return OpStatePtr::Create<SgMKLDNNSelfAttQKOp>(attrs);
}

static void SgMKLDNNSelfAttQKForward(const OpStatePtr &state_pointer,
                                     const OpContext &ctx,
                                     const std::vector<NDArray> &inputs,
                                     const std::vector<OpReqType> &req,
                                     const std::vector<NDArray> &outputs) {
  SgMKLDNNSelfAttQKOp &op = state_pointer.get_state<SgMKLDNNSelfAttQKOp>();
  if(!op.IsInitialized()) {
    op.Initialize(ctx, inputs, req, outputs);
  }
  op.Forward(ctx, inputs, req, outputs);
}

void SgMKLDNNSelfAttQKOp::Initialize(const OpContext &ctx,
                                     const std::vector<NDArray> &inputs,
                                     const std::vector<OpReqType> &req,
                                     const std::vector<NDArray> &outputs) {
    using namespace mkldnn;
    const auto qkv_tensor = inputs[0];
    const auto out_tensor = outputs[0];
    const auto qkv_dtype = get_mkldnn_type(qkv_tensor.dtype());

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

    memory::dims query_dims    = {attn_batches, qkv_seq_len, head_dim};
    memory::dims key_dims      = {attn_batches, head_dim, qkv_seq_len};
    memory::dims out_dims      = {attn_batches, qkv_seq_len, qkv_seq_len};

    memory::dims query_strides = {batch_stride, lead_dim, 1};
    memory::dims key_strides   = {batch_stride, 1, lead_dim};
    
    auto query_md = memory::desc(query_dims, qkv_dtype, query_strides);
    auto key_md   = memory::desc(key_dims, qkv_dtype, key_strides);

    memory::desc out_md;

    float oscale = 1.0f;
    if (param_.quantized) {
      data_scale_ = GetQuantizeScale(qkv_tensor.dtype(), min_data_, max_data_);

      if (param_.min_calib_range.has_value() &&
          param_.max_calib_range.has_value()) {
        min_output_ = param_.min_calib_range.value();
        max_output_ = param_.max_calib_range.value();
        oscale = 
            GetQuantizeScale(out_tensor.dtype(), min_output_, max_output_) /
            (data_scale_ * data_scale_);
        out_md = memory::desc(out_dims, memory::data_type::s8, memory::format_tag::abc);
      } else if (param_.enable_float_output) {
        oscale = 1.0f / (data_scale_ * data_scale_);
        out_md = dnnl::memory::desc(out_dims, memory::data_type::f32, memory::format_tag::abc);
      } else {
        mshadow::Stream<cpu> *s = ctx.get_stream<cpu>();
        mxnet_op::Kernel<QuantizationRangeForS8S8MultiplicationStruct, cpu>::Launch(
              s, 1, &min_output_, &max_output_, &min_data, &max_data, &min_data,
              &max_data);
        out_md = dnnl::memory::desc(out_dims, memory::data_type::s32, memory::format_tag::abc);
      }
    } else {
      out_md = dnnl::memory::desc(out_dims, memory::data_type::f32, memory::format_tag::abc);
    }
    oscale /= sqrt(static_cast<float>(head_dim)); // combine quantized scale and sqrt(head_dim)

    dnnl::primitive_attr attr;
    attr.set_output_scales(0, {oscale});
    auto matmul_d = matmul::desc(query_md, key_md, out_md);
    auto matmul_pd = matmul::primitive_desc(matmul_d, attr, engine);

    fwd_ = std::make_shared<matmul>(matmul_pd);

    MSHADOW_TYPE_SWITCH(inputs[0].dtype(), DType, {
      DType* query_mem_ptr = inputs[0].data().dptr<DType>();
      DType* key_mem_ptr   = query_mem_ptr + head_dim;
      cached_query_mem_ = std::make_shared<memory>(query_md, engine, query_mem_ptr);
      cached_key_mem_ = std::make_shared<memory>(key_md, engine, key_mem_ptr);
    });
    MSHADOW_TYPE_SWITCH(outputs[0].dtype(), DType, {
      cached_out_mem_ = std::make_shared<memory>(out_md, engine, outputs[0].data().dptr<DType>());
    });

    args_[DNNL_ARG_SRC]     = *cached_query_mem_;
    args_[DNNL_ARG_WEIGHTS] = *cached_key_mem_;
    args_[DNNL_ARG_DST]     = *cached_out_mem_;
    initialized_ = true;
}


void SgMKLDNNSelfAttQKOp::Forward(const OpContext &ctx,
                                  const std::vector<NDArray> &inputs,
                                  const std::vector<OpReqType> &req,
                                  const std::vector<NDArray> &outputs) {

    const size_t head_dim = inputs[0].shape()[2] / 3 / param_.heads;

    MSHADOW_TYPE_SWITCH(inputs[0].dtype(), DType, {
      DType* query_mem_ptr = inputs[0].data().dptr<DType>();
      DType* key_mem_ptr   = query_mem_ptr + head_dim;
      cached_query_mem_->set_data_handle(query_mem_ptr);
      cached_key_mem_->set_data_handle(key_mem_ptr);
    });

    MSHADOW_TYPE_SWITCH(outputs[0].dtype(), DType, {
      cached_out_mem_->set_data_handle(reinterpret_cast<void*>(outputs[0].data().dptr<DType>()));
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
  nnvm::ObjectPtr node = nnvm::Node::Create();
  auto const &param = nnvm::get<MKLDNNInterleavedMatMulParam>(attrs.parsed);
  node->attrs.op = Op::Get("_sg_mkldnn_selfatt_qk");
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

NNVM_REGISTER_OP(_sg_mkldnn_selfatt_qk)
.describe(R"code(_sg_mkldnn_selfatt_qk)code" ADD_FILELINE)
.set_num_inputs([](const NodeAttrs& attrs) {
  auto const& param = nnvm::get<MKLDNNInterleavedMatMulParam>(attrs.parsed);
  if (param.quantized) {
    return 3;
  } else {
    return 1;
  }
})
.set_num_outputs([](const NodeAttrs& attrs) {
  auto const& param = nnvm::get<MKLDNNInterleavedMatMulParam>(attrs.parsed);
  if (param.quantized && !param.enable_float_output) {
    return 3;
  } else {
    return 1;
  }
})
.set_attr_parser(ParamParser<MKLDNNInterleavedMatMulParam>)
.set_attr<nnvm::FListInputNames>("FListInputNames", [](const NodeAttrs& attrs) {
  auto const& param = nnvm::get<MKLDNNInterleavedMatMulParam>(attrs.parsed);
  std::vector<std::string> input_names {"queries_keys_values"};
  if (param.quantized) {
    input_names.emplace_back("min_qkv");
    input_names.emplace_back("max_qkv");
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
.set_attr<mxnet::FInferShape>("FInferShape", SgMKLDNNSelfAttQKShape)
.set_attr<nnvm::FInferType>("FInferType", SgMKLDNNSelfAttQKInferType)
.set_attr<FInferStorageType>("FInferStorageType", SgMKLDNNSelfAttQKStorageType)
.set_attr<FCreateOpState>("FCreateOpState", CreateSgMKLDNNSelfAttQKState)
.set_attr<FStatefulComputeEx>("FStatefulComputeEx<cpu>", SgMKLDNNSelfAttQKForward)
.set_attr<bool>("TIsMKLDNN", true)
.set_attr<nnvm::FGradient>("FGradient", MakeZeroGradNodes)
.set_attr<FQuantizable>("FQuantizable", [](const NodeAttrs& attrs) {
    return QuantizeType::kMust;
})
.set_attr<FQuantizedOp>("FQuantizedOp", SgMKLDNNSelfAttQKQuantizedOp)
.set_attr<FNeedRequantize>("FNeedRequantize", [](const NodeAttrs& attrs) { return true; })
.add_argument("queries_keys_values", "NDArray-or-Symbol", "Interleaved queries, keys and values")
.add_arguments(MKLDNNInterleavedMatMulParam::__FIELDS__());

/********************************************************************************************************/

static bool MKLDNNInterleavedMatMulSelfAttValAttShape(const NodeAttrs& attrs,
                                                mxnet::ShapeVector* in_shape,
                                                mxnet::ShapeVector* out_shape) {
  const auto& param = nnvm::get<MKLDNNInterleavedMatMulParam>(attrs.parsed);
  if (param.quantized) {
    auto qkv_shape = in_shape->at(0);
    
    out_shape->resize(3);
    SHAPE_ASSIGN_CHECK(*out_shape, 0,
      mxnet::TShape({qkv_shape[0], qkv_shape[1], qkv_shape[2] / 3}));
    if (!param.enable_float_output) {
      SHAPE_ASSIGN_CHECK(*out_shape, 1, mxnet::TShape({1}));                     // min output
      SHAPE_ASSIGN_CHECK(*out_shape, 2, mxnet::TShape({1}));                     // max output
    }
    
    return true;
  } else {
    CHECK_EQ(in_shape->size(), 2U) << "Input:[queries_keys_values, attention] currently have, "
                                 << in_shape->size() << " inputs";
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
    SHAPE_ASSIGN_CHECK(*out_shape, 0,
      mxnet::TShape({qkv_shape[0], qkv_shape[1], qkv_shape[2] / 3}));
    return true;
  }
}

static bool MKLDNNInterleavedMatMulSelfAttValAttInferType(const nnvm::NodeAttrs &attrs,
                                std::vector<int> *in_types,
                                std::vector<int> *out_types) {
  const auto& param = nnvm::get<MKLDNNInterleavedMatMulParam>(attrs.parsed);
  if (param.quantized) {
    TYPE_ASSIGN_CHECK(*in_types, 0, mshadow::kInt8);    // qkv input
    TYPE_ASSIGN_CHECK(*in_types, 1, mshadow::kUint8);    // att input

    TYPE_ASSIGN_CHECK(*in_types, 2, mshadow::kFloat32); // min qkv
    TYPE_ASSIGN_CHECK(*in_types, 3, mshadow::kFloat32); // max qkv

    TYPE_ASSIGN_CHECK(*in_types, 4, mshadow::kFloat32); // min att
    TYPE_ASSIGN_CHECK(*in_types, 5, mshadow::kFloat32); // max att

    if (param.enable_float_output) {
      TYPE_ASSIGN_CHECK(*out_types, 0, mshadow::kFloat32);     // output
    } else {
      if (param.min_calib_range.has_value() && param.max_calib_range.has_value()) {
        TYPE_ASSIGN_CHECK(*out_types, 0, mshadow::kInt8);      // output
      } else {
        TYPE_ASSIGN_CHECK(*out_types, 0, mshadow::kInt32);     // output
      }
      TYPE_ASSIGN_CHECK(*out_types, 1, mshadow::kFloat32);     // min output
      TYPE_ASSIGN_CHECK(*out_types, 2, mshadow::kFloat32);     // max output
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
  node->attrs.op = Op::Get("_sg_mkldnn_interleaved_matmul_selfatt_valatt");
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

class MKLDNNInterleavedMatMulSelfAttValAttOp {
 public:
  explicit MKLDNNInterleavedMatMulSelfAttValAttOp(const nnvm::NodeAttrs &attrs) :
    param_(nnvm::get<MKLDNNInterleavedMatMulParam>(attrs.parsed)) {}

  void Forward(const OpContext &ctx,
               const std::vector<NDArray> &inputs,
               const std::vector<OpReqType> &req,
               const std::vector<NDArray> &outputs);

  void Backward(const OpContext &ctx,
                const std::vector<NDArray> &inputs,
                const std::vector<OpReqType> &req,
                const std::vector<NDArray> &outputs) {
    LOG(FATAL) << "Not implemented: subgraph mkldnn fully connected only supports "
                  "inference computation.";
  }

 private:
  bool initialized_{false};
  MKLDNNInterleavedMatMulParam param_;
  mkldnn_args_map_t args_;
  std::shared_ptr<dnnl::matmul> fwd_;
  std::shared_ptr<dnnl::memory> cached_data1_mem_;
  std::shared_ptr<dnnl::memory> cached_data2_mem_;
  std::shared_ptr<dnnl::memory> cached_out_mem_;
  float cached_min_qkv_;
  float cached_max_qkv_;
  float cached_min_att_;
  float cached_max_att_;
  float cached_min_output_;
  float cached_max_output_;
  float qkv_scale_{0.0f};
  float att_scale_{0.0f};
};

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

  float min_qkv = 0.0f;
  float max_qkv = 0.0f;
  float min_att = 0.0f;
  float max_att = 0.0f;

  if (param_.quantized) {
    min_qkv = inputs[2].data().dptr<float>()[0];
    max_qkv = inputs[3].data().dptr<float>()[0];
    min_att = inputs[4].data().dptr<float>()[0];
    max_att = inputs[5].data().dptr<float>()[0];
  }

  const dnnl::memory::dim HEADS = param_.heads;
  const dnnl::memory::dim BS = inputs[0].shape()[1];
  const dnnl::memory::dim SEQ_LEN = inputs[0].shape()[0];
  const dnnl::memory::dim EMBED = inputs[0].shape()[2];

  if (!initialized_) {
    const auto engine = CpuEngine::Get()->get_engine();

    cached_min_qkv_ = min_qkv;
    cached_max_qkv_ = max_qkv;
    cached_min_att_ = min_att;
    cached_max_att_ = max_att;

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
    
    float tmp_scale = 1.0f;
    if (param_.quantized) {
      qkv_scale_ = GetQuantizeScale(mshadow::kInt8, min_qkv, max_qkv);
      att_scale_ = GetQuantizeScale(mshadow::kUint8, min_att, max_att);

      if (param_.min_calib_range.has_value() &&
          param_.max_calib_range.has_value()) {
        cached_min_output_ = param_.min_calib_range.value();
        cached_max_output_ = param_.max_calib_range.value();

        tmp_scale = GetQuantizeScale(mshadow::kInt8, cached_min_output_, cached_max_output_)  / (qkv_scale_ * att_scale_);
        dst_md = dnnl::memory::desc(dst_dims, dnnl::memory::data_type::s8, dnnl::memory::format_tag::bac);
      } else if (param_.enable_float_output) {
        tmp_scale = 1.0f / (qkv_scale_ * att_scale_);
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
    attr.set_output_scales(0, {tmp_scale});
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

    args_[DNNL_ARG_SRC] = *cached_data1_mem_;
    args_[DNNL_ARG_WEIGHTS] = *cached_data2_mem_;
    args_[DNNL_ARG_DST] = *cached_out_mem_;
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

// void MKLDNNInterleavedMatMulSelfAttValAttCPU(const nnvm::NodeAttrs& attrs,
//                                        const OpContext &ctx,
//                                        const std::vector<TBlob> &inputs,
//                                        const std::vector<OpReqType> &req,
//                                        const std::vector<TBlob> &outputs) {
//   const auto& param = nnvm::get<MKLDNNInterleavedMatMulParam>(attrs.parsed);

//   if (param.quantized) {
//     if (param.enable_float_output) {
//       const dnnl::memory::dim HEADS = param.heads;
//       const dnnl::memory::dim BS = inputs[0].shape_[1];
//       const dnnl::memory::dim SEQ_LEN = inputs[0].shape_[0];
//       const dnnl::memory::dim EMBED = inputs[0].shape_[2];

//       dnnl::engine engine(dnnl::engine::kind::cpu, 0);
//       dnnl::stream engine_stream(engine);

//       dnnl::memory::dims src1_dims = {BS*HEADS, SEQ_LEN, SEQ_LEN};
//       dnnl::memory::dims src2_dims = {BS*HEADS, SEQ_LEN, EMBED/HEADS/3};
//       dnnl::memory::dims dst_dims = {BS*HEADS, SEQ_LEN, EMBED/HEADS/3};

//       // dnnl::memory::dims src1_strides = {SEQ_LEN*SEQ_LEN, SEQ_LEN, 1};
//       dnnl::memory::dims src2_strides = {3*(EMBED/HEADS/3), EMBED*BS, 1};

//       auto src1_md = dnnl::memory::desc(src1_dims, dnnl::memory::data_type::u8, dnnl::memory::format_tag::abc); // CHECK IF IT IS U8 FOR SURE
//       auto src2_md = dnnl::memory::desc(src2_dims, dnnl::memory::data_type::s8, src2_strides);
//       auto dst_md = dnnl::memory::desc(dst_dims, dnnl::memory::data_type::f32, dnnl::memory::format_tag::bac);

//       float min_qkv = inputs[2].dptr<float>()[0];
//       float max_qkv = inputs[3].dptr<float>()[0];
//       float min_att = inputs[4].dptr<float>()[0];
//       float max_att = inputs[5].dptr<float>()[0];

//       float qkv_scale = GetQuantizeScale(mshadow::kInt8, min_qkv, max_qkv);
//       float att_scale = GetQuantizeScale(mshadow::kUint8, min_att, max_att);

//       const float scale = 1.0f / (qkv_scale * att_scale);

//       dnnl::primitive_attr attr;
//       attr.set_output_scales(0, {scale});

//       auto matmul_d = dnnl::matmul::desc(src1_md, src2_md, dst_md);
//       auto matmul_pd = dnnl::matmul::primitive_desc(matmul_d, attr, engine);

//       auto matmul_prim = dnnl::matmul(matmul_pd);

//       mshadow::Stream<cpu>* s = ctx.get_stream<cpu>();
//       int8_t* queries_keys_values = inputs[0].FlatTo2D<cpu, int8_t>(s).dptr_;
//       uint8_t* attention_maps     = inputs[1].FlatTo2D<cpu, uint8_t>(s).dptr_;
//       float* output               = outputs[0].FlatTo2D<cpu, float>(s).dptr_;

//       auto src1_mem = dnnl::memory(src1_md, engine, attention_maps);
//       auto src2_mem = dnnl::memory(src2_md, engine, queries_keys_values+2*(EMBED/HEADS/3));
//       auto dst_mem = dnnl::memory(dst_md, engine, output);

//       std::unordered_map<int, dnnl::memory> matmul_args;
//       matmul_args.insert({DNNL_ARG_SRC, src1_mem});
//       matmul_args.insert({DNNL_ARG_WEIGHTS, src2_mem});
//       matmul_args.insert({DNNL_ARG_DST, dst_mem});

//       matmul_prim.execute(engine_stream, matmul_args);
//       engine_stream.wait();
//     } else if (param.min_calib_range.has_value() && param.max_calib_range.has_value()) {
//       const dnnl::memory::dim HEADS = param.heads;
//       const dnnl::memory::dim BS = inputs[0].shape_[1];
//       const dnnl::memory::dim SEQ_LEN = inputs[0].shape_[0];
//       const dnnl::memory::dim EMBED = inputs[0].shape_[2];

//       dnnl::engine engine(dnnl::engine::kind::cpu, 0);
//       dnnl::stream engine_stream(engine);

//       dnnl::memory::dims src1_dims = {BS*HEADS, SEQ_LEN, SEQ_LEN};
//       dnnl::memory::dims src2_dims = {BS*HEADS, SEQ_LEN, EMBED/HEADS/3};
//       dnnl::memory::dims dst_dims = {BS*HEADS, SEQ_LEN, EMBED/HEADS/3};

//       // dnnl::memory::dims src1_strides = {SEQ_LEN*SEQ_LEN, SEQ_LEN, 1};
//       dnnl::memory::dims src2_strides = {3*(EMBED/HEADS/3), EMBED*BS, 1};

//       auto src1_md = dnnl::memory::desc(src1_dims, dnnl::memory::data_type::u8, dnnl::memory::format_tag::abc); // CHECK IF IT IS U8 FOR SURE
//       auto src2_md = dnnl::memory::desc(src2_dims, dnnl::memory::data_type::s8, src2_strides);
//       auto dst_md = dnnl::memory::desc(dst_dims, dnnl::memory::data_type::s8, dnnl::memory::format_tag::bac);

//       float min_qkv = inputs[2].dptr<float>()[0];
//       float max_qkv = inputs[3].dptr<float>()[0];
//       float min_att = inputs[4].dptr<float>()[0];
//       float max_att = inputs[5].dptr<float>()[0];

//       float qkv_scale = GetQuantizeScale(mshadow::kInt8, min_qkv, max_qkv);
//       float att_scale = GetQuantizeScale(mshadow::kUint8, min_att, max_att);

//       const float scale = GetQuantizeScale(mshadow::kInt8, param.min_calib_range.value(), param.max_calib_range.value()) 
//                           / (qkv_scale * att_scale);

//       dnnl::primitive_attr attr;
//       attr.set_output_scales(0, {scale});

//       auto matmul_d = dnnl::matmul::desc(src1_md, src2_md, dst_md);
//       auto matmul_pd = dnnl::matmul::primitive_desc(matmul_d, attr, engine);

//       auto matmul_prim = dnnl::matmul(matmul_pd);

//       mshadow::Stream<cpu>* s = ctx.get_stream<cpu>();
//       int8_t* queries_keys_values = inputs[0].FlatTo2D<cpu, int8_t>(s).dptr_;
//       uint8_t* attention_maps     = inputs[1].FlatTo2D<cpu, uint8_t>(s).dptr_;
//       int8_t* output              = outputs[0].FlatTo2D<cpu, int8_t>(s).dptr_;

//       float* min_output = outputs[1].dptr<float>();
//       float* max_output = outputs[2].dptr<float>();
//       min_output[0] = param.min_calib_range.value();
//       max_output[0] = param.max_calib_range.value();

//       auto src1_mem = dnnl::memory(src1_md, engine, attention_maps);
//       auto src2_mem = dnnl::memory(src2_md, engine, queries_keys_values+2*(EMBED/HEADS/3));
//       auto dst_mem = dnnl::memory(dst_md, engine, output);

//       std::unordered_map<int, dnnl::memory> matmul_args;
//       matmul_args.insert({DNNL_ARG_SRC, src1_mem});
//       matmul_args.insert({DNNL_ARG_WEIGHTS, src2_mem});
//       matmul_args.insert({DNNL_ARG_DST, dst_mem});

//       matmul_prim.execute(engine_stream, matmul_args);
//       engine_stream.wait();
//     } else {
//       const dnnl::memory::dim HEADS = param.heads;
//       const dnnl::memory::dim BS = inputs[0].shape_[1];
//       const dnnl::memory::dim SEQ_LEN = inputs[0].shape_[0];
//       const dnnl::memory::dim EMBED = inputs[0].shape_[2];

//       dnnl::engine engine(dnnl::engine::kind::cpu, 0);
//       dnnl::stream engine_stream(engine);

//       dnnl::memory::dims src1_dims = {BS*HEADS, SEQ_LEN, SEQ_LEN};
//       dnnl::memory::dims src2_dims = {BS*HEADS, SEQ_LEN, EMBED/HEADS/3};
//       dnnl::memory::dims dst_dims = {BS*HEADS, SEQ_LEN, EMBED/HEADS/3};

//       // dnnl::memory::dims src1_strides = {SEQ_LEN*SEQ_LEN, SEQ_LEN, 1};
//       dnnl::memory::dims src2_strides = {3*(EMBED/HEADS/3), EMBED*BS, 1};

//       auto src1_md = dnnl::memory::desc(src1_dims, dnnl::memory::data_type::u8, dnnl::memory::format_tag::abc); // CHECK IF IT IS U8 FOR SURE
//       auto src2_md = dnnl::memory::desc(src2_dims, dnnl::memory::data_type::s8, src2_strides);
//       auto dst_md = dnnl::memory::desc(dst_dims, dnnl::memory::data_type::s32, dnnl::memory::format_tag::bac);

//       float min_qkv = inputs[2].dptr<float>()[0];
//       float max_qkv = inputs[3].dptr<float>()[0];
//       float min_att = inputs[4].dptr<float>()[0];
//       float max_att = inputs[5].dptr<float>()[0];

//       // float qkv_scale = GetQuantizeScale(mshadow::kInt8, min_qkv, max_qkv);
//       // float att_scale = GetQuantizeScale(mshadow::kUint8, min_att, max_att);

//       // const float scale = 1.0f / (qkv_scale * att_scale);

//       // dnnl::primitive_attr attr;
//       // attr.set_output_scales(0, {scale});

//       auto matmul_d = dnnl::matmul::desc(src1_md, src2_md, dst_md);
//       auto matmul_pd = dnnl::matmul::primitive_desc(matmul_d, engine);

//       auto matmul_prim = dnnl::matmul(matmul_pd);

//       mshadow::Stream<cpu>* s = ctx.get_stream<cpu>();
//       int8_t* queries_keys_values = inputs[0].FlatTo2D<cpu, int8_t>(s).dptr_;
//       uint8_t* attention_maps     = inputs[1].FlatTo2D<cpu, uint8_t>(s).dptr_;
//       int32_t* output             = outputs[0].FlatTo2D<cpu, int32_t>(s).dptr_;

//       float* min_output = outputs[1].dptr<float>();
//       float* max_output = outputs[2].dptr<float>();

//       mxnet_op::Kernel<QuantizationRangeForS8U8MultiplicationStruct, cpu>::Launch(
//                 s, 1, min_output, max_output, &min_qkv, &max_qkv, &min_att,
//                 &max_att);

//       auto src1_mem = dnnl::memory(src1_md, engine, attention_maps);
//       auto src2_mem = dnnl::memory(src2_md, engine, queries_keys_values+2*(EMBED/HEADS/3));
//       auto dst_mem = dnnl::memory(dst_md, engine, output);

//       std::unordered_map<int, dnnl::memory> matmul_args;
//       matmul_args.insert({DNNL_ARG_SRC, src1_mem});
//       matmul_args.insert({DNNL_ARG_WEIGHTS, src2_mem});
//       matmul_args.insert({DNNL_ARG_DST, dst_mem});

//       matmul_prim.execute(engine_stream, matmul_args);
//       engine_stream.wait();
//     }
//   } else {
//     const dnnl::memory::dim HEADS = param.heads;
//     const dnnl::memory::dim BS = inputs[0].shape_[1];
//     const dnnl::memory::dim SEQ_LEN = inputs[0].shape_[0];
//     const dnnl::memory::dim EMBED = inputs[0].shape_[2];

//     dnnl::engine engine(dnnl::engine::kind::cpu, 0);
//     dnnl::stream engine_stream(engine);

//     dnnl::memory::dims src1_dims = {BS*HEADS, SEQ_LEN, SEQ_LEN};
//     dnnl::memory::dims src2_dims = {BS*HEADS, SEQ_LEN, EMBED/HEADS/3};
//     dnnl::memory::dims dst_dims = {BS*HEADS, SEQ_LEN, EMBED/HEADS/3};

//     // dnnl::memory::dims src1_strides = {SEQ_LEN*SEQ_LEN, SEQ_LEN, 1};
//     dnnl::memory::dims src2_strides = {3*(EMBED/HEADS/3), EMBED*BS, 1};

//     auto src1_md = dnnl::memory::desc(src1_dims, dnnl::memory::data_type::f32, dnnl::memory::format_tag::abc);
//     auto src2_md = dnnl::memory::desc(src2_dims, dnnl::memory::data_type::f32, src2_strides);
//     auto dst_md = dnnl::memory::desc(dst_dims, dnnl::memory::data_type::f32, dnnl::memory::format_tag::bac);

//     auto matmul_d = dnnl::matmul::desc(src1_md, src2_md, dst_md);
//     auto matmul_pd = dnnl::matmul::primitive_desc(matmul_d, engine);

//     auto matmul_prim = dnnl::matmul(matmul_pd);

//     mshadow::Stream<cpu>* s = ctx.get_stream<cpu>();
//     float* queries_keys_values = inputs[0].FlatTo2D<cpu, float>(s).dptr_;
//     float* attention_maps      = inputs[1].FlatTo2D<cpu, float>(s).dptr_;
//     float* output              = outputs[0].FlatTo2D<cpu, float>(s).dptr_;

//     auto src1_mem = dnnl::memory(src1_md, engine, attention_maps);
//     auto src2_mem = dnnl::memory(src2_md, engine, queries_keys_values+2*(EMBED/HEADS/3));
//     auto dst_mem = dnnl::memory(dst_md, engine, output);

//     std::unordered_map<int, dnnl::memory> matmul_args;
//     matmul_args.insert({DNNL_ARG_SRC, src1_mem});
//     matmul_args.insert({DNNL_ARG_WEIGHTS, src2_mem});
//     matmul_args.insert({DNNL_ARG_DST, dst_mem});

//     matmul_prim.execute(engine_stream, matmul_args);
//     engine_stream.wait();
//   }
// }

// NNVM_REGISTER_OP(_sg_mkldnn_interleaved_matmul_selfatt_valatt)
// .describe(R"code(_sg_mkldnn_interleaved_matmul_selfatt_valatt)code" ADD_FILELINE)
// .set_num_inputs([](const NodeAttrs& attrs) {
//   auto const& param = nnvm::get<MKLDNNInterleavedMatMulParam>(attrs.parsed);
//   if (param.quantized) {
//     return 6;
//   } else {
//     return 2;
//   }
// })
// .set_num_outputs([](const NodeAttrs& attrs) {
//   auto const& param = nnvm::get<MKLDNNInterleavedMatMulParam>(attrs.parsed);
//   if (param.quantized && !param.enable_float_output) {
//     return 3;
//   } else {
//     return 1;
//   }
// })
// .set_attr_parser(ParamParser<MKLDNNInterleavedMatMulParam>)
// .set_attr<nnvm::FListInputNames>("FListInputNames", [](const NodeAttrs& attrs) {
//   auto const& param = nnvm::get<MKLDNNInterleavedMatMulParam>(attrs.parsed);
//   std::vector<std::string> input_names {"queries_keys_values", "attention"};
//   if (param.quantized) {
//     input_names.emplace_back("min_qkv");
//     input_names.emplace_back("max_qkv");

//     input_names.emplace_back("min_attention");
//     input_names.emplace_back("max_attention");
//   }
//   return input_names;
// })
// .set_attr<nnvm::FListOutputNames>("FListOutputNames", [](const NodeAttrs& attrs) {
//   auto const& param = nnvm::get<MKLDNNInterleavedMatMulParam>(attrs.parsed);
//   std::vector<std::string> output_names {"output"};
//   if (param.quantized && !param.enable_float_output) {
//     output_names.emplace_back("min_output");
//     output_names.emplace_back("max_output");
//   }
//   return output_names;
// })
// .set_attr<mxnet::FInferShape>("FInferShape", MKLDNNInterleavedMatMulSelfAttValAttShape)
// .set_attr<nnvm::FInferType>("FInferType", MKLDNNInterleavedMatMulSelfAttValAttInferType)
// // .set_attr<FCompute>("FCompute<cpu>", MKLDNNInterleavedMatMulSelfAttValAttCPU)
// .set_attr<FInferStorageType>("FInferStorageType", MKLDNNInterleavedMatMulSelfAttValAttStorageType)
// .set_attr<FCreateOpState>("FCreateOpState", CreateMKLDNNInterleavedMatMulSelfAttValAttState)
// .set_attr<FStatefulComputeEx>("FStatefulComputeEx<cpu>", MKLDNNInterleavedMatMulSelfAttValAttForward)
// .set_attr<bool>("TIsMKLDNN", true)
// .set_attr<nnvm::FGradient>("FGradient", MakeZeroGradNodes)
// .set_attr<FQuantizable>("FQuantizable", [](const NodeAttrs& attrs) {
//     return QuantizeType::kMust;
// })
// .set_attr<FQuantizedOp>("FQuantizedOp", SgMKLDNNInterleavedMatMulSelfAttValAttQuantizedOp)
// .set_attr<FNeedRequantize>("FNeedRequantize", [](const NodeAttrs& attrs) { return true; })
// .add_argument("queries_keys_values", "NDArray-or-Symbol", "Queries, keys and values interleaved")
// .add_argument("attention", "NDArray-or-Symbol", "Attention maps")
// .add_arguments(MKLDNNInterleavedMatMulParam::__FIELDS__());

}  // namespace op
}  // namespace mxnet

#endif