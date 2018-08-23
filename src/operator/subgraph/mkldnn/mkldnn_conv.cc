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
#include <nnvm/graph.h>
#include <mshadow/base.h>
#include "./mkldnn_conv.h"
#include "../../nn/mkldnn/mkldnn_ops-inl.h"
#include "../../../imperative/imperative_utils.h"
#include "../../../imperative/cached_op.h"
#include "../../nn/convolution-inl.h"
#include "../../nn/batch_norm-inl.h"
namespace mxnet {
namespace op {

#define SUBGRAPH_DEBUG 0

template <typename DType>
static void UpdateConvWeightBias(const NDArray &weight, const NDArray *bias,
                                 const NDArray &gamma, const NDArray &beta,
                                 const NDArray &mean,
                                 const NDArray &variance,
                                 std::shared_ptr<NDArray> update_weight,
                                 std::shared_ptr<NDArray> update_bias,
                                 const BatchNormParam &param) {
#if SUBGRAPH_DEBUG
  printf("input weight: %f %f %f %f \n", weight.data().dptr<float>()[0],
                                         weight.data().dptr<float>()[1],
                                         weight.data().dptr<float>()[2],
                                         weight.data().dptr<float>()[3]);
  printf("bn param eps: %f \n", param.eps);
  printf("bn param fix_gamma: %d \n", param.fix_gamma);
  printf("bn param use_global_stats: %d \n", param.use_global_stats);
  printf("bn param output_mean_var: %d \n", param.output_mean_var);
  printf("bn param axis: %d \n", param.axis);
#endif
  DType *weight_ptr = weight.Reorder2Default().data().dptr<DType>();
  DType *bias_ptr = bias ? bias->Reorder2Default().data().dptr<DType>() : nullptr;
  DType *gamma_ptr = gamma.Reorder2Default().data().dptr<DType>();
  DType *beta_ptr = beta.Reorder2Default().data().dptr<DType>();
  DType *mean_ptr = mean.Reorder2Default().data().dptr<DType>();
  DType *var_ptr = variance.Reorder2Default().data().dptr<DType>();
  DType *update_weight_ptr = update_weight->data().dptr<DType>();
  DType *update_bias_ptr = update_bias->data().dptr<DType>();
  size_t channel = gamma.shape()[0];
  size_t offset = weight.shape()[1] * weight.shape()[2] * weight.shape()[3];
#pragma omp parallel for
  for (size_t c = 0; c < channel; ++c) {
    DType *p1 = reinterpret_cast<DType *>(weight_ptr + c * offset);
    DType *p2 = reinterpret_cast<DType *>(update_weight_ptr + c * offset);
    DType alpha = (param.fix_gamma ? static_cast<DType>(1.0f) : gamma_ptr[c]) /
                  sqrt(var_ptr[c] + param.eps);

    if (bias_ptr)
      update_bias_ptr[c] = beta_ptr[c] + alpha * (bias_ptr[c] - mean_ptr[c]);
    else
      update_bias_ptr[c] = beta_ptr[c] - alpha * mean_ptr[c];

    for (size_t k = 0; k < offset; ++k) {
      p2[k] = p1[k] * alpha;
    }
  }
#if SUBGRAPH_DEBUG
  printf("update weight: %f %f %f %f \n", update_weight->data().dptr<float>()[0],
                                          update_weight->data().dptr<float>()[1],
                                          update_weight->data().dptr<float>()[2],
                                          update_weight->data().dptr<float>()[3]);
#endif
}

static void ConvFusionFallBackCompute() {
  LOG(FATAL) << "Don't know how to do ConvFusionFallBackCompute!";
}

static void ConvolutionFusionComputeExCPU(const nnvm::NodeAttrs &conv_attrs,
                                          const OpContext &ctx,
                                          const std::vector<NDArray> &inputs,
                                          const std::vector<OpReqType> &req,
                                          const std::vector<NDArray> &outputs) {
  const ConvolutionParam &params =
      nnvm::get<ConvolutionParam>(conv_attrs.parsed);
  if (SupportMKLDNNConv(params, inputs[0])) {
    // MKLDNN_OPCHECK_INIT(false, outputs.size(), inputs, outputs);
    MKLDNNConvolutionForward(conv_attrs, ctx, inputs, req, outputs);
    // MKLDNN_OPCHECK_RUN(ConvolutionCompute<cpu>, attrs, ctx, inputs, req,
    // outputs);
    return;
  }
  ConvFusionFallBackCompute();
}

class SgMKLDNNConvOperator {
 public:
  explicit SgMKLDNNConvOperator(const nnvm::NodeAttrs &attrs)
      : subgraph_sym_(nnvm::get<Symbol>(attrs.parsed)),
        // subgraph_exec_(nullptr),
        cached_weight_(nullptr),
        cached_bias_(nullptr),
        bn_attrs_(nullptr),
        conv_attrs_(nullptr),
        in_sum_at_begin(false),
        with_bn(false),
        with_relu(false),
        with_sum(false),
        with_postsum_relu(false) {
    // subgraph_exec_.reset(new CachedOp(subgraph_sym_, {{"static_alloc", "true"}}));
    auto it = attrs.dict.find("in_sum_at_begin");
    if (it != attrs.dict.end())
      in_sum_at_begin = (it->second == "true");
    it = attrs.dict.find("with_bn");
    if (it != attrs.dict.end())
      with_bn = (it->second == "true");
    it = attrs.dict.find("with_relu");
    if (it != attrs.dict.end())
      with_relu = (it->second == "true");
    it = attrs.dict.find("with_sum");
    if (it != attrs.dict.end())
      with_sum = (it->second == "true");
    it = attrs.dict.find("with_postsum_relu");
    if (it != attrs.dict.end())
      with_postsum_relu = (it->second == "true");

    DFSVisit(subgraph_sym_.outputs, [&](const nnvm::NodePtr &node) {
      if (node->is_variable()) return;
      auto &node_name = node->op()->name;
      if (node_name == "BatchNorm") {
        CHECK(bn_attrs_.get() == nullptr);
        CHECK_EQ(with_bn, true);
        bn_attrs_ = std::make_shared<nnvm::NodeAttrs>(node->attrs);
      } else if (node_name == "Convolution") {
        CHECK(conv_attrs_.get() == nullptr);
        conv_attrs_ = std::make_shared<nnvm::NodeAttrs>(node->attrs);
      }
    });
    CHECK(conv_attrs_.get());
    conv_attrs_->dict["with_bn"] = with_bn ? "true" : "false";
    conv_attrs_->dict["with_relu"] = with_relu ? "true" : "false";
    conv_attrs_->dict["with_sum"] = with_sum ? "true" : "false";
    conv_attrs_->dict["with_postsum_relu"] = with_postsum_relu ? "true" : "false";
  }

  void Forward(const OpContext &ctx,
               const std::vector<NDArray> &inputs,
               const std::vector<OpReqType> &req,
               const std::vector<NDArray> &outputs);

  void Backward(const OpContext &ctx,
                const std::vector<NDArray> &inputs,
                const std::vector<OpReqType> &req,
                const std::vector<NDArray> &outputs) {
    LOG(FATAL) << "Not implemented: subgraph mkldnn Conv only supports inference computation";
  }

 private:
  nnvm::Symbol subgraph_sym_;
  // CachedOpPtr subgraph_exec_; // Used for fallback compute
  std::shared_ptr<NDArray> cached_weight_;
  std::shared_ptr<NDArray> cached_bias_;
  std::shared_ptr<nnvm::NodeAttrs> bn_attrs_;
  std::shared_ptr<nnvm::NodeAttrs> conv_attrs_;
  bool in_sum_at_begin;
  bool with_bn;
  bool with_relu;
  bool with_sum;
  bool with_postsum_relu;
};

void SgMKLDNNConvOperator::Forward(const OpContext &ctx,
                                   const std::vector<NDArray> &inputs,
                                   const std::vector<OpReqType> &req,
                                   const std::vector<NDArray> &outputs) {
  const ConvolutionParam &conv_params = nnvm::get<ConvolutionParam>(conv_attrs_->parsed);
#if SUBGRAPH_DEBUG
  LOG(INFO) << "Conv inputs size: " << inputs.size();
  LOG(INFO) << "Conv outputs size: " << outputs.size();
  LOG(INFO) << "Conv req size: " << req.size();
  for (size_t k = 0; k < inputs.size(); ++k) {
    auto input = inputs[k];
    printf("input %ld :", k);
    for (size_t i = 0; i < input.shape().ndim(); ++i) {
      printf("%ld ", input.shape()[i]);
    }
    printf("\n");
  }
  CHECK_EQ(ctx.is_train, false);
  printf("output:");
    for (size_t i = 0; i < outputs[0].shape().ndim(); ++i) {
      printf("%ld ", outputs[0].shape()[i]);
    }
    printf("\n");
#endif
    size_t input_size = 2 + (conv_params.no_bias ? 0 : 1) + (with_bn ? 4 : 0) +
                        (with_sum ? 1 : 0);
    CHECK_EQ(inputs.size(), input_size);
    size_t idx = 0;
    auto in_sum = in_sum_at_begin ? (idx++) : 0;
    auto in_data = idx++;
    auto in_weight = idx++;
    auto in_bias = conv_params.no_bias ? 0 : (idx++);
    auto in_gamma = with_bn ? (idx++) : 0;
    auto in_beta = with_bn ? (idx++) : 0;
    auto in_mean = with_bn ? (idx++) : 0;
    auto in_var = with_bn ? (idx++) : 0;
    in_sum = ((!in_sum_at_begin) && with_sum) ? (idx++) : 0;
    auto output = outputs[0];
    CHECK_EQ(input_size, idx);

    if (with_bn && (nullptr == cached_weight_ || nullptr == cached_bias_)) {
      CHECK_EQ(inputs[in_weight].dtype(), inputs[in_gamma].dtype());
      CHECK_EQ(inputs[in_weight].dtype(), inputs[in_beta].dtype());
      CHECK_EQ(inputs[in_weight].dtype(), inputs[in_var].dtype());
      const BatchNormParam &bn_param =
          nnvm::get<BatchNormParam>(bn_attrs_->parsed);
      cached_weight_ = std::make_shared<NDArray>(
          inputs[in_weight].storage_type(), inputs[in_weight].shape(),
          inputs[in_weight].ctx(), true, inputs[in_weight].dtype());
      cached_bias_ = std::make_shared<NDArray>(
          inputs[in_beta].storage_type(), inputs[in_beta].shape(),
          inputs[in_beta].ctx(), true, inputs[in_beta].dtype());
      MSHADOW_REAL_TYPE_SWITCH(inputs[in_weight].dtype(), DType, {
        UpdateConvWeightBias<DType>(
            inputs[in_weight], conv_params.no_bias ? nullptr : &inputs[in_bias],
            inputs[in_gamma], inputs[in_beta], inputs[in_mean], inputs[in_var],
            cached_weight_, cached_bias_, bn_param);
      });
    }
    std::vector<NDArray> new_inputs;
    std::vector<OpReqType> new_req;
    std::vector<NDArray> new_outputs;
    if (with_bn) {
      new_inputs = {inputs[in_data], *cached_weight_, *cached_bias_};
      new_req = {req[in_data], req[in_weight], req[in_beta]};
    } else {
      if (conv_params.no_bias) {
        new_inputs = {inputs[in_data], inputs[in_weight]};
        new_req = {req[in_data], req[in_weight]};
      } else {
        new_inputs = {inputs[in_data], inputs[in_weight], inputs[in_bias]};
        new_req = {req[in_data], req[in_weight], req[in_bias]};
      }
    }
    if (with_sum)
      new_outputs = {inputs[in_sum]};
    else
      new_outputs = {output};
    ConvolutionFusionComputeExCPU(*conv_attrs_, ctx, new_inputs, new_req,
                                  new_outputs);
  }

  OpStatePtr CreateSgMKLDNNConvOpState(const nnvm::NodeAttrs &attrs, Context ctx,
                                       const std::vector<TShape> &in_shapes,
                                       const std::vector<int> &in_types) {
    return OpStatePtr::Create<SgMKLDNNConvOperator>(attrs);
  }

  void SgMKLDNNConvOpForward(const OpStatePtr &state_ptr, const OpContext &ctx,
                             const std::vector<NDArray> &inputs,
                             const std::vector<OpReqType> &req,
                             const std::vector<NDArray> &outputs) {
    SgMKLDNNConvOperator &op = state_ptr.get_state<SgMKLDNNConvOperator>();
    op.Forward(ctx, inputs, req, outputs);
  }

NNVM_REGISTER_OP(_sg_mkldnn_conv)
.describe(R"code(_sg_mkldnn_conv)code" ADD_FILELINE)
.set_num_inputs(DefaultSubgraphOpNumInputs)
.set_num_outputs(DefaultSubgraphOpNumOutputs)
.set_attr<nnvm::FListInputNames>("FListInputNames",
                                  DefaultSubgraphOpListInputs)
.set_attr<nnvm::FListOutputNames>("FListOutputNames",
  [](const NodeAttrs& attrs) {
  return std::vector<std::string>{"output"};
})
.set_attr<FCreateOpState>("FCreateOpState", CreateSgMKLDNNConvOpState)
.set_attr<nnvm::FInferShape>("FInferShape", DefaultSubgraphOpShape)
.set_attr<nnvm::FInferType>("FInferType", DefaultSubgraphOpType)
.set_attr<FInferStorageType>("FInferStorageType",
                              DefaultSubgraphOpStorageType)
.set_attr<FStatefulComputeEx>("FStatefulComputeEx<cpu>",
                              SgMKLDNNConvOpForward)
.set_attr<nnvm::FMutateInputs>("FMutateInputs",
                                DefaultSubgraphOpMutableInputs)
.set_attr<FResourceRequest>("FResourceRequest",
                            DefaultSubgraphOpResourceRequest)
.set_attr<std::string>("key_var_num_args", "num_args")
.set_attr<nnvm::FInplaceOption>("FInplaceOption", [](const nnvm::NodeAttrs
                                                          &attrs) {
  auto it = attrs.dict.find("with_sum");
  if (it != attrs.dict.end() && it->second == "true") {
    it = attrs.dict.find("in_sum_at_begin");
    if (it != attrs.dict.end() && it->second == "true") {
      return std::vector<std::pair<int, int>>{std::pair<int, int>{0, 0}};
    } else {
      it = attrs.dict.find("no_bias");
      CHECK(it != attrs.dict.end());
      bool no_bias = it->second == "true";
      it = attrs.dict.find("with_bn");
      bool with_bn = (it != attrs.dict.end()) ? it->second == "true" : false;
      int idx = 2 + (no_bias ? 0 : 1) + (with_bn ? 4 : 0);
      return std::vector<std::pair<int, int>>{std::pair<int, int>{idx, 0}};
    }
  } else {
    return std::vector<std::pair<int, int>>();
  }
});
}  // namespace op
}  // namespace mxnet

#endif  // MXNET_USE_MKLDNN == 1
