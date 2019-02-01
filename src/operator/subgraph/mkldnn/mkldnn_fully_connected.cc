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
#include "../common.h"
#include "../../nn/mkldnn/mkldnn_base-inl.h"
#include "../../nn/mkldnn/mkldnn_ops-inl.h"
#include "../../nn/mkldnn/mkldnn_fully_connected-inl.h"
#include "../../quantization/quantization_utils.h"

namespace mxnet {
namespace op {

class SgMKLDNNFCOp {
  public:
    explicit SgMKLDNNFCOp(const nnvm::NodeAttrs &attrs)
      : initialized_(false),
        subgraph_sym_(*attrs.subgraphs[0]),
        full_param_(nnvm::get<MKLDNNFCFullParam>(attrs.parsed)) {}

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
    bool initialized_;
    nnvm::Symbol subgraph_sym_;
    MKLDNNFCFullParam full_param_;
    std::shared_ptr<MKLDNNFullyConnectForward> fwd_;
    NDArray cached_weight_;
    NDArray cached_bias_;
    float cached_min_data_;
    float cached_max_data_;
    float cached_min_weight_;
    float cached_max_weight_;
    float cached_min_bias_;
    float cached_max_bias_;
};

void SgMKLDNNFCOp::Forward((const OpContext &ctx,
                            const std::vector<NDArray> &in_data,
                            const std::vector<OpReqType> &req,
                            const std::vector<NDArray> &out_data) {
//TODO         
  auto &mkldnn_param = full_param_.mkldnn_param;
  auto &default_param = full_param_.default_param;
  bool has_bias = !default_param.no_bias;
  size_t base_num_inputs = has_bias ? 3 : 2;
  size_t total_num_inputs = base_num_inputs;
  size_t base_num_outputs = 1;
  size_t total_num_outputs = base_num_outputs;

  float min_data = 0.0;
  float max_data = 0.0;
  float min_weight = 0.0;
  float max_weight = 0.0;
  float min_bias = 0.0;
  float max_bias = 0.0;
  float *min_output_ptr = nullptr;
  float *max_output_ptr = nullptr;
  
  if (mkldnn_param.quantized) {
    total_num_inputs = base_num_inputs * 3;
    min_data = in_data[num_inputs].data().dptr<float>()[0];
    max_data = in_data[num_inputs + 1].data().dptr<float>()[0];
    min_weight = in_data[num_inputs + 2].data().dptr<float>()[0];
    max_weight = in_data[num_inputs + 3].data().dptr<float>()[0];
    if (has_bias) {
      min_bias = in_data[num_inputs + 4].data().dptr<float>()[0];
      max_bias = in_data[num_inputs + 5].data().dptr<float>()[0];
    }
    if (!mkldnn_param.fuse_dequantize) {
      total_num_outputs = base_num_outputs * 3; 
      min_output_ptr = out_data[1].data().dptr<float>();
      max_output_ptr = out_data[2].data().dptr<float>();
    }
  }
  CHECK_EQ(in_data.size(), total_num_inputs);
  CHECK_EQ(out_data.size(), total_num_outputs);

  NDArray data = in_data[fullc::kData];
  NDArray weight = in_data[fullc::kWeight];
  NDArray output = out_data[fullc::kOut];
  const TShape &ishae = data.shape();
  if (ishape.ndim())

  if (initialized_ && mkldnn_param.quantized) {
    if (cached_min_data_ != min_data || cached_max_data_ != max_data ||
        cached_min_weight_ != min_weight || cached_max_weight_ != max_weight ||
        (has_bias && (cached_min_bias_ != min_bias || cached_max_bias_ != max_bias))) {
          initialized_ = false;
        }
  }

  if (!initialized_) {
    if (mkldnn_param.quantized) {
      CHECK(data.dtype() == mshadow::kInt8 || data.dtype() == mshadow::kUint8);
      auto data_range = (data.dtype() == mshadow::kInt8) ? kInt8Range : kUint8Range;
      float data_scale  = data_range / MaxAbs(min_data, max_data);
      float weight_scale = kInt8Range / MaxAbs(min_weight, max_weight);

      if (has_bias) {
        NDArray bias = in_data[fullc::kBias];
        float bias_int32_rescale = data_scale * weight_scale * MaxAbs(min_bias, max_bias) / kInt8Range;

        cached_bias_ = NDArray(bias.storage_type(), bias.shape(),
                               bias.ctx(), true, mshadow::kInt32);
        int8_t *bias_ptr = bias.data().dptr<int8_t>();
        int32_t *quantized_bias_ptr = cached_bias_.data().dptr<int32_t>();
        size_t bias_size = bias.shape().Size();
        #pragma omp parallel for num_threads(engine::OpenMP::Get()->GetRecommendedOMPThreadCount())
        for (size_t i = 0; i < bias_size; ++i) {
          quantized_bias_ptr[i] = bias_ptr[i] * bias_int32_rescale;
        }
      }

      if (mkldnn_param.fuse_dequantze) {
        full_param.output_scales[0] = 1.0 / data_scale / weight_scale;
        full_param.requantize_scales.resize(0);
      } else if (mkldnn_param.fuse_requantize) {
        full_param.output_scales.resize(0);
        if (mkldnn_param.min_calib_range.has_value() && 
            mkldnn_param.max_calib_range.has_value()) {
              *min_output_ptr = mkldnn_param.min_calib_range.value();
              *max_output_ptr = mkldnn_param.max_calib_range.value();
              mkldnn_param.requantize_scales[0] = kInt8Range / MaxAbs(*min_output_ptr, *max_output_ptr) / data_scale / weight_scale;
        } else {
          LOG(FATAL) << "Failed to fuse requantize due to no min_calib_range and max_calib_range found."
        }
      } else {
        Stream<cpu> *s = ctx.get_stream<cpu>();
        mxnet_op::Kernel<QuantizationRangeForMultiplicationStruct, cpu>::Launch(s, 1,
          min_output_ptr, max_output_ptr, &min_data, &max_data, &min_weight, &max_weight);
      }
    } else {
      cached_bias_ = in_data[fullc::kBias];
    }

    fwd_.reset(new MKLDNNFCForward(full_param_, ctx.is_train, data, weight, 
      has_bias ? cached_bias_ : nullptr, output));
    initialized_ = true;
  }
  std::vector<NDArray> new_inputs;
  std::vector<OpReqType> new_req;
  if (has_bias) {
    new_inputs = {data, weight, cached_bias_};
    new_req = {req[fullc::kData], req[fullc::kWeight], req[fullc::kBias]};
  } else {
    new_inputs = {data, weight};
    new_req = {req[fullc::kData], req[fullc::kWeight]};
  }

  MKLDNNFCForwardFullFeature(full_param_, ctx, fwd_.get(), new_inputs, new_req, out_data);
}

static void sgMKLDNNFCParamParser(nnvm::NodeAttrs *attrs) {
  MKLDNNFCFullParam full_param;
  try {
    full_param.mkldnn_param.Init(attrs->dict);
  } catch (const dmlc::ParamError &e) {
    std::ostringstream os;
    os << e.what();
    os << ", in operator " << attrs->op->name << "("
       << "name=\"" << attrs->name << "\"";
    for (const auto &k : attrs->dict) {
      os << ", " << k.first << "=\"" << k.second << "\"";
    }
    os << ")";
    throw dmlc::ParamError(os.str());
  }
  auto subgraph_sym = attrs->subgraphs[0];
  DFSVisit(subgraph_sym->outputs, [&](const nnvm::NodePtr &node) {
    if (node->is_variable()) return;
    auto &node_name = node->op()->name;
    if (node_name == "FullyConnected") {
      full_param.default_param =
          nnvm::get<FullyConnectedParam>(node->attrs.parsed);
    }
  });
  attrs->parsed = std::move(full_param);
}

static std::vector<std::string> SgMKLDNNFCListInputNames(const NodeAttrs &attrs) {
  auto const &full_param = nnvm::get<MKLDNNFCFullParam>(attrs.parsed);
  std::vector<std::string> input_names = DefaultSubgraphOpListInputs(attrs);
  if (full_param.mkldnn_param.quantized) {
    inputs_names.emplace_back("min_data");
    inputs_names.emplace_back("max_data");
    inputs_names.emplace_back("min_weight");
    inputs_names.emplace_back("max_weight");
    if (!full_param.default_param.no_bias) {
      inputs_names.emplace_back("min_bias");
      inputs_names.emplace_back("max_bias");
    }
  }
  return input_names;
}

static std::vector<std::string> SgMKLDNNFCListOutputNames(const NodeAttrs &attrs) {
  auto const &full_param = nnvm::get<MKLDNNFCFullParam>(attrs.parsed);
  if (full_param.mkldnn_param.quantized) {
    if (full_param.mkldnn_param.fuse_dequantize)
      return std::vector<std::string>{"output"};
    else
      return std::vector<std::string>>{"output", "min_output", "max_output"};

  } else {
    return std::vector<std::string>{"output"};
  }
}

static bool SgMKLDNNFCInferShape(const nnvm::NodeAttrs &attrs,
                                 std::vector<Tshape> *in_shapes,
                                 std::vector<Tshape> *out_shapes) {
  auto const &full_param = nnvm::get<MKLDNNFCFullParam>(attrs.parsed);
  if (full_param.mkldnn_param.quantized) {
    std::vector<TShape> base_in_shapes;
    std::vector<TShape> base_out_shapes;
    bool ret = DefaultSubgraphOpShape(attrs, &base_in_shapes, &base_out_shapes);

    auto base_num_inputs = full_param.default_param.no_bias ? 2 : 3;
    auto total_num_inputs = base_num_inputs * 3;
    for (size_t i = 0; i < total_num_inputs; ++i) {
      if (i < base_num_inputs)
        in_shapes->at(i) = base_in_shapes[i];
      else
        SHAPE_ASSIGN_CHECK(*in_shapes, i, Shape1(1));
    }
    
    out_shapes->at(0) = base_out_shapes[0];
    if (!full_param.mkldnn_param.fuse_dequantize) {
      SHAPE_ASSIGN_CHECK(*out_shapes, 1, Shape1(1));
      SHAPE_ASSIGN_CHECK(*out_shapes, 2, Shape1(1));
    }
    return ret;
  } else {
    return DefaultSubgraphOpShape(attrs, in_shapes, out_shapes);
  }
}

static bool SgMKLDNNFCInferType(const nnvm::NodeAttrs &attrs,
                                std::vector<int> *in_types,
                                std::vector<int> *out_types) {
  auto const &full_param = nnvm::get<MKLDNNFCFullParam>(attrs.parsed);
  if (full_param.mkldnn_param.quantized) {
    std::vector<TShape> base_in_types;
    std::vector<TShape> base_out_types;
    bool ret = DefaultSubgraphOpType(attrs, &base_in_types, &base_out_types);

    auto base_num_inputs = full_param.default_param.no_bias ? 2 : 3;
    auto total_num_inputs = base_num_inputs * 3;
    for (size_t i = 0; i < total_num_inputs; ++i) {
      if (i < base_num_inputs)
        in_types->at(i) = base_in_types[i];
      else
        TYPE_ASSIGN_CHECK(*in_types, i, mshadow::kFloat32);
    }
    
    if (full_param.mkldnn_param.fuse_dequantize) {
      TYPE_ASSIGN_CHECK(*out_types, 0, mshadow::kFloat32);
    } else {
      if (full_param.mkldnn_param.fuse_requantize) {
        TYPE_ASSIGN_CHECK(*out_types, 0, mshadow::kInt8); //with_relu=True, this should be Uint8.
      } else {
        TYPE_ASSIGN_CHECK(*out_types, 0, mshadow::kInt32);
      }
      TYPE_ASSIGN_CHECK(*out_types, 1, mshadow::kFloat32);
      TYPE_ASSIGN_CHECK(*out_types, 2, mshadow::kFloat32);
    }
    return ret;
  } else {
    return DefaultSubgraphOpType(attrs, in_types, out_types);
  }                              
}

static bool SgMKLDNNFCStorageType(const nnvm::NodeAttrs &attrs,
                                  const int dev_mask,
                                  DispatchMode *dispatch_mode,
                                  std::vector<int> *in_attrs,
                                  std::vector<int> *out_attrs) {
  auto const &full_param = nnvm::get<MKLDNNFCFullParam>(attrs.parsed);
  if (full_param.mkldnn_param.quantized) {
    std::vector<TShape> base_in_attrs;
    std::vector<TShape> base_out_attrs;
    bool ret = DefaultSubgraphOpStorageType(attrs, dev_mask, dispatch_mode,
                                            &base_in_attrs, &base_out_attrs);

    auto base_num_inputs = full_param.default_param.no_bias ? 2 : 3;
    auto total_num_inputs = base_num_inputs * 3;
    for (size_t i = 0; i < total_num_inputs; ++i) {
      if (i < base_num_inputs)
        in_attrs->at(i) = base_in_attrs[i];
      else
        type_assign(&in_attrs->at(i), mxnet::kDefaultStorage);
    }
    
    out_attrs->at(0) = base_out_attrs[0];
    if (!full_param.mkldnn_param.fuse_dequantize) {
      type_assign(&out_attrs->at(1), mxnet::kDefaultStorage);
      type_assign(&out_attrs->at(2), mxnet::kDefaultStorage);
    }
    return ret;
  } else {
    return DefaultSubgraphOpStorageType(attrs, dev_mask, disaptch_mode, 
                                        in_attrs, out_attrs);
  }
}

static OpStatePtr CreateSgMKLDNNFCState(const nnvm::NodeAttrs &attrs,
                                        Context ctx,
                                        const std::vector<TShape> &in_shapes,
                                        const std::vector<int> &in_types) {
  return OpStatePtr::Create<SgMKLDNNFCOp>(attrs);
}

static void SgMKLDNNFCForward(const OpStatePtr &state_pointer,
                              const OpContext &ctx,
                              const std::vector<NDArray> &inputs,
                              const std::vector<OpReqType> &req,
                              const std::vector<NDArray> &outputs) {
  SgMKLDNNFCOp &op = state_pointer.get_state<SgMKLDNNFCOp>();
  op.Forward(ctx, inputs, req, outputs);                              
}                                

nnvm::NodePtr SgMKLDNNFCQuantizedOp(const NodeAttrs& attrs) {
  auto const &param = nnvm::get<MKLDNNFCFullParam>(attrs.parsed);
  nnvm::NodePtr node = nnvm::Node::Create();
  node->attrs.op = Op::Get("_sg_mkldnn_fully_connected");
  node->attrs.name = "quantized_" + attrs.name;
  node->attrs.dict = attrs.dict;
  node->attrs.dict["quantized"] = "true";
  node->attrs.subgraphs.reserve(attrs.subgraphs.size());
  for (auto sub : attrs.subgraphs) {
    node->attrs.subgraphs.push_back(sub);
  }
  node->op()->attr_parser(&(node->attrs));
  return node;
}

NNVM_REGISTER_OP(_sg_mkldnn_fully_connected)
.describe(R"code(_sg_mkldnn_fully_connected)code" ADD_FILELINE)
.set_num_inputs([](const NodeAttrs& attrs) {
  auto const &full_param = nnvm::get<MKLDNNFCFullParam>(attrs.parsed);
  num_inputs = full_param.default_param.no_bias ? 2 : 3;
  if (full_param.mkldnn_param.quantized)
    return num_inputs * 3;
  else
    return num_inputs;
})
.set_num_outputs([](const NodeAttrs& attrs) {
  auto const &full_param = nnvm::get<MKLDNNFCFullParam>(attrs.parsed);
  return (full_param.mkldnn_param.quantized && full_param.mkldnn_param.fuse_dequantized) ? 1 : 3;
})
.set_attr_parser(SgMKLDNNFCParamParser)
.set_attr<nnvm::FListInputNames>("FListInputNames", SgMKLDNNFCListInputNames)
.set_attr<nnvm::FListOutputNames>("FListOutputNames", SgMKLDNNFCListOutputNames)
.set_attr<nnvm::FInferShape>("FInferShape", SgMKLDNNFCInferShape)
.set_attr<nnvm::FInferType>("FInferType", SgMKLDNNFCInferType)
.set_attr<FInferStorageType>("FInferStorageType", SgMKLDNNFCStorageType)
.set_attr<FCreateOpState>("FCreateOpState", CreateSgMKLDNNFCState)
.set_attr<FStatefulComputeEx>("FStatefulComputeEx<cpu>", SgMKLDNNFCForward)
.set_attr<bool>("TIsMKLDNN", true)
.set_attr<FResourceRequest>("FResourceRequest", [](const NodeAttrs& n) {
  return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
})
.set_attr<nnvm::FMutateInputs>("FMutateInputs",
                                DefaultSubgraphOpMutableInputs)
.set_attr<std::string>("key_var_num_args", "num_args")
.set_attr<FQuantizedOp>("FQuantizedOp", SgMKLDNNFCQuantizedOp)
.set_attr<FNeedRequantize>("FNeedRequantize", [](const NodeAttrs& attrs) { return true; })

}  // namespace op
}  // namespace mxnet

#endif  // if MXNET_USE_MKLDNN == 1
