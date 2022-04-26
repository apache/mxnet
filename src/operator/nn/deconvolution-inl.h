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
 * \file deconvolution-inl.h
 * \brief
 * \author Wei Wu, Da Zheng
 */
#ifndef MXNET_OPERATOR_NN_DECONVOLUTION_INL_H_
#define MXNET_OPERATOR_NN_DECONVOLUTION_INL_H_

#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <mxnet/operator.h>
#include <algorithm>
#include <map>
#include <vector>
#include <string>
#include <utility>
#include "../operator_common.h"
#include "../linalg.h"
#include "convolution-inl.h"

namespace mxnet {
namespace op {

namespace deconv {
enum DeconvolutionOpInputs { kData, kWeight, kBias };
enum DeconvolutionOpOutputs { kOut };
enum DeconvolutionOpResource { kTempSpace };
enum DeconvolutionOpCudnnTune { kOff, kLimited, kFastest };
}  // namespace deconv

struct DeconvolutionParam : public dmlc::Parameter<DeconvolutionParam> {
  mxnet::TShape kernel;
  mxnet::TShape stride;
  mxnet::TShape dilate;
  mxnet::TShape pad;
  mxnet::TShape adj;
  mxnet::TShape target_shape;
  uint32_t num_filter;
  uint32_t num_group;
  uint64_t workspace;
  bool no_bias;
  dmlc::optional<int> cudnn_tune;
  bool cudnn_off;
  dmlc::optional<int> layout;
  DMLC_DECLARE_PARAMETER(DeconvolutionParam) {
    DMLC_DECLARE_FIELD(kernel).describe(
        "Deconvolution kernel size: (w,), (h, w) or (d, h, w). "
        "This is same as the kernel size used for the corresponding convolution");
    DMLC_DECLARE_FIELD(stride)
        .set_default(mxnet::TShape(0, 0))
        .describe(
            "The stride used for the corresponding convolution: (w,), (h, w) or (d, h, w). "
            "Defaults to 1 for each dimension.");
    DMLC_DECLARE_FIELD(dilate)
        .set_default(mxnet::TShape(0, 0))
        .describe(
            "Dilation factor for each dimension of the input: (w,), (h, w) or (d, h, w). "
            "Defaults to 1 for each dimension.");
    DMLC_DECLARE_FIELD(pad)
        .set_default(mxnet::TShape(0, 0))
        .describe(
            "The amount of implicit zero padding added during convolution for each "
            "dimension of the input: "
            "(w,), (h, w) or (d, h, w). "
            "``(kernel-1)/2`` is usually a good choice. "
            "If `target_shape` is set, "
            "`pad` will be ignored and a padding that will generate the target shape "
            "will be used. Defaults to no padding.");
    DMLC_DECLARE_FIELD(adj)
        .set_default(mxnet::TShape(0, 0))
        .describe(
            "Adjustment for output shape: (w,), (h, w) or (d, h, w). "
            "If `target_shape` is set, "
            "`adj` will be ignored and computed accordingly.");
    DMLC_DECLARE_FIELD(target_shape)
        .set_default(mxnet::TShape(0, 0))
        .describe("Shape of the output tensor: (w,), (h, w) or (d, h, w).");
    DMLC_DECLARE_FIELD(num_filter).set_lower_bound(1).describe("Number of output filters.");
    DMLC_DECLARE_FIELD(num_group).set_default(1).describe("Number of groups partition.");
    DMLC_DECLARE_FIELD(workspace).set_default(1024).set_lower_bound(0).describe(
        "Maximum temporary workspace allowed (MB) in deconvolution."
        "This parameter has two usages. When CUDNN is not used, it determines the "
        "effective batch size of the deconvolution kernel. When CUDNN is used, "
        "it controls the maximum temporary storage used for tuning "
        "the best CUDNN kernel when `limited_workspace` strategy is used.");
    DMLC_DECLARE_FIELD(no_bias).set_default(true).describe("Whether to disable bias parameter.");
    DMLC_DECLARE_FIELD(cudnn_tune)
        .add_enum("off", deconv::kOff)
        .add_enum("limited_workspace", deconv::kLimited)
        .add_enum("fastest", deconv::kFastest)
        .set_default(dmlc::optional<int>())
        .describe("Whether to pick convolution algorithm by running performance test.");
    DMLC_DECLARE_FIELD(cudnn_off).set_default(false).describe("Turn off cudnn for this layer.");
    DMLC_DECLARE_FIELD(layout)
        .add_enum("NCW", mshadow::kNCW)
        .add_enum("NCHW", mshadow::kNCHW)
        .add_enum("NCDHW", mshadow::kNCDHW)
        .add_enum("NHWC", mshadow::kNHWC)
        .add_enum("NDHWC", mshadow::kNDHWC)
        .set_default(dmlc::optional<int>())
        .describe(
            "Set layout for input, output and weight. Empty for "
            "default layout, NCW for 1d, NCHW for 2d and NCDHW for 3d."
            "NHWC and NDHWC are only supported on GPU.");
  }

  template <size_t ndim>
  void InferPad(mxnet::TShape input, index_t (&o_pad)[ndim], index_t (&o_adj)[ndim]) const {
    // Modified by Li.bs
    // Use tag to control the calculation of pad
    bool bCal = false;
    if (target_shape.ndim() != 0) {
      for (index_t i = 0; i < target_shape.ndim(); i++) {
        if (target_shape[i] != 0)
          bCal = true;
      }
    }

    if (bCal) {
      size_t input_ndim = input.ndim();

      for (size_t i = 0; i < ndim; i++) {
        // input.ndim() can be larger than ndim, in case that the complete input
        // shape was passed and not only the ndim last ones
        if (mxnet::dim_size_is_known(input, input_ndim - ndim + i)) {
          o_pad[i] = stride[i] * (input[(input_ndim - ndim) + i] - 1) + DilatedKernelSize(i);
          CHECK_GE(o_pad[i], target_shape[i]) << "too big target shape";
          o_pad[i] -= target_shape[i];
          o_adj[i] = o_pad[i] % 2;
          o_pad[i] = (o_pad[i] + 1) / 2;
        }
      }
    } else {
      for (int i = 0; i < static_cast<int>(ndim); i++) {
        o_pad[i] = i < pad.ndim() ? pad[i] : 0;
        o_adj[i] = i < adj.ndim() ? adj[i] : 0;
      }
    }
  }

  // Adjusts kernel size for effects of dilation in the dimension `dim`.
  index_t DilatedKernelSize(int dim) const {
    return 1 + (kernel[dim] - 1) * dilate[dim];
  }

  bool operator==(const DeconvolutionParam& other) const {
    return this->kernel == other.kernel && this->stride == other.stride &&
           this->dilate == other.dilate && this->pad == other.pad && this->adj == other.adj &&
           this->target_shape == other.target_shape && this->num_filter == other.num_filter &&
           this->num_group == other.num_group && this->workspace == other.workspace &&
           this->no_bias == other.no_bias && this->cudnn_tune == other.cudnn_tune &&
           this->cudnn_off == other.cudnn_off && this->layout == other.layout;
  }

  std::string CudnnTune2String(int cudnn_tune) {
    switch (cudnn_tune) {
      case deconv::kOff:
        return "off";
      case deconv::kLimited:
        return "limited_workspace";
      case deconv::kFastest:
        return "fastest";
      default:
        LOG(FATAL) << "Unknown cudnn_tune enum " << cudnn_tune;
    }
    LOG(FATAL) << "should not reach here ";
    return "";
  }
  std::string Layout2String(int layout) {
    switch (layout) {
      case mshadow::kNCW:
        return "NCW";
      case mshadow::kNCHW:
        return "NCHW";
      case mshadow::kNCDHW:
        return "NCDHW";
      case mshadow::kNHWC:
        return "NHWC";
      case mshadow::kNDHWC:
        return "NDHWC";
      default:
        LOG(FATAL) << "Unknown layout enum " << layout;
    }
    LOG(FATAL) << "should not reach here ";
    return "";
  }

  void SetAttrDict(std::unordered_map<std::string, std::string>* dict) {
    std::ostringstream kernel_s, stride_s, dilate_s, pad_s, adj_s, target_shape_s, num_filter_s,
        num_group_s, workspace_s, no_bias_s, cudnn_tune_s, cudnn_off_s, layout_s;
    kernel_s << kernel;
    stride_s << stride;
    dilate_s << dilate;
    pad_s << pad;
    adj_s << adj;
    target_shape_s << target_shape;
    num_filter_s << num_filter;
    num_group_s << num_group;
    workspace_s << workspace;
    no_bias_s << no_bias;
    cudnn_tune_s << cudnn_tune;
    cudnn_off_s << cudnn_off;
    layout_s << layout;
    (*dict)["kernel"]       = kernel_s.str();
    (*dict)["stride"]       = stride_s.str();
    (*dict)["dilate"]       = dilate_s.str();
    (*dict)["pad"]          = pad_s.str();
    (*dict)["adj"]          = adj_s.str();
    (*dict)["target_shape"] = target_shape_s.str();
    (*dict)["num_filter"]   = num_filter_s.str();
    (*dict)["num_group"]    = num_group_s.str();
    (*dict)["workspace"]    = workspace_s.str();
    (*dict)["no_bias"]      = no_bias_s.str();
    if (cudnn_tune.has_value()) {
      (*dict)["cudnn_tune"] = CudnnTune2String(cudnn_tune.value());
    } else {
      (*dict)["cudnn_tune"] = cudnn_tune_s.str();
    }
    (*dict)["cudnn_off"] = cudnn_off_s.str();
    if (layout.has_value()) {
      (*dict)["layout"] = Layout2String(layout.value());
    } else {
      (*dict)["layout"] = layout_s.str();
    }
  }
};

typedef ParamOpSign<DeconvolutionParam> DeconvSignature;

}  // namespace op
}  // namespace mxnet

namespace std {
template <>
struct hash<mxnet::op::DeconvolutionParam> {
  size_t operator()(const mxnet::op::DeconvolutionParam& val) {
    size_t ret = 0;
    ret        = dmlc::HashCombine(ret, val.kernel);
    ret        = dmlc::HashCombine(ret, val.stride);
    ret        = dmlc::HashCombine(ret, val.dilate);
    ret        = dmlc::HashCombine(ret, val.pad);
    ret        = dmlc::HashCombine(ret, val.adj);
    ret        = dmlc::HashCombine(ret, val.target_shape);
    ret        = dmlc::HashCombine(ret, val.num_filter);
    ret        = dmlc::HashCombine(ret, val.num_group);
    ret        = dmlc::HashCombine(ret, val.workspace);
    ret        = dmlc::HashCombine(ret, val.no_bias);
    ret        = dmlc::HashCombine(ret, val.cudnn_tune);
    ret        = dmlc::HashCombine(ret, val.cudnn_off);
    ret        = dmlc::HashCombine(ret, val.layout);
    return ret;
  }
};
}  // namespace std

namespace mxnet {
namespace op {

template <typename xpu, typename DType>
class DeconvolutionOp {
 public:
  void Init(DeconvolutionParam dp) {
    param_           = dp;
    param_.workspace = (param_.workspace << 20) / sizeof(DType);
  }

  void Forward(const OpContext& ctx,
               const std::vector<TBlob>& in_data,
               const std::vector<OpReqType>& req,
               const std::vector<TBlob>& out_data) {
    size_t expected = param_.no_bias ? 2 : 3;
    CHECK_EQ(req[deconv::kOut], kWriteTo);
    CHECK_EQ(in_data.size(), expected);
    CHECK_EQ(out_data.size(), 1U);

    if (need_init_conv)
      InitConv(in_data[deconv::kData]);

    conv_op._BackwardData(ctx,
                          in_data[deconv::kData],
                          in_data[deconv::kWeight],
                          req[deconv::kOut],
                          out_data[deconv::kOut]);

    if (!param_.no_bias) {
      Stream<xpu>* s                  = ctx.get_stream<xpu>();
      const TShape& out_shape         = out_data[deconv::kOut].shape_;
      Tensor<xpu, 1, DType> bias      = in_data[deconv::kBias].get<xpu, 1, DType>(s);
      Tensor<xpu, 3, DType> output_3d = out_data[deconv::kOut].get_with_shape<xpu, 3, DType>(
          Shape3(out_shape[0], out_shape[1], out_shape.ProdShape(2, out_shape.ndim())), s);
      // broadcast bias to the same shape of output_3d in channel dim
      output_3d += mshadow::expr::broadcast<1>(bias, output_3d.shape_);
    }
  }

  void Backward(const OpContext& ctx,
                const std::vector<TBlob>& out_grad,
                const std::vector<TBlob>& in_data,
                const std::vector<OpReqType>& req,
                const std::vector<TBlob>& in_grad) {
    using namespace mshadow;
    using namespace mshadow::expr;

    const size_t expected = param_.no_bias == 0 ? 3 : 2;
    CHECK_EQ(out_grad.size(), 1U);
    CHECK_EQ(in_data.size(), expected);
    CHECK_EQ(in_grad.size(), expected);
    CHECK_EQ(req.size(), expected);

    if (need_init_conv)
      InitConv(in_data[deconv::kData]);

    // data gradient
    auto workspace = conv_op._Forward(ctx,
                                      out_grad[deconv::kOut],
                                      in_data[deconv::kWeight],
                                      nullptr,
                                      req[deconv::kData],
                                      in_grad[deconv::kData]);
    // weights gradient
    conv_op._BackwardWeightsBias(workspace,
                                 ctx,
                                 in_data[deconv::kData],
                                 out_grad[deconv::kOut],
                                 req[deconv::kWeight],
                                 in_grad[deconv::kWeight],
                                 OpReqType(),
                                 nullptr);
    // bias gradient
    if (!param_.no_bias) {
      Stream<xpu>* s              = ctx.get_stream<xpu>();
      const TShape& out_shape     = out_grad[deconv::kOut].shape_;
      Tensor<xpu, 1, DType> dbias = in_grad[deconv::kBias].get<xpu, 1, DType>(s);
      Tensor<xpu, 3, DType> dout  = out_grad[deconv::kOut].get_with_shape<xpu, 3, DType>(
          Shape3(out_shape[0], out_shape[1], out_shape.ProdShape(2, out_shape.ndim())), s);
      ASSIGN_DISPATCH(dbias, req[deconv::kBias], sumall_except_dim<1>(dout));
    }
  }

 private:
  void InitConv(const TBlob& in_data) {
    ConvolutionParam cp;
    cp.kernel     = param_.kernel;
    cp.stride     = param_.stride;
    cp.dilate     = param_.dilate;
    cp.pad        = param_.pad;
    cp.num_filter = in_data.shape_[1];
    cp.num_group  = param_.num_group;
    cp.workspace  = (param_.workspace * sizeof(DType)) >> 20;
    cp.no_bias    = true;
    cp.cudnn_tune = param_.cudnn_tune;
    cp.cudnn_off  = param_.cudnn_off;
    cp.layout     = param_.layout;
    conv_op.Init(cp);
    need_init_conv = false;
  }

  bool need_init_conv = true;
  DeconvolutionParam param_;
  ConvolutionOp<xpu, DType> conv_op;
};  // class DeconvolutionOp

template <typename xpu>
void _DeconvolutionCompute(const DeconvolutionParam& param,
                           const OpContext& ctx,
                           const std::vector<TBlob>& inputs,
                           const std::vector<OpReqType>& req,
                           const std::vector<TBlob>& outputs) {
  MSHADOW_REAL_TYPE_SWITCH(inputs[deconv::kData].type_flag_, DType, {
    DeconvolutionOp<xpu, DType> op;
    op.Init(param);
    op.Forward(ctx, inputs, req, outputs);
  });
}

template <typename xpu>
void DeconvolutionCompute(const nnvm::NodeAttrs& attrs,
                          const OpContext& ctx,
                          const std::vector<TBlob>& inputs,
                          const std::vector<OpReqType>& req,
                          const std::vector<TBlob>& outputs) {
  const DeconvolutionParam& param = nnvm::get<DeconvolutionParam>(attrs.parsed);
  _DeconvolutionCompute<xpu>(param, ctx, inputs, req, outputs);
}

template <typename xpu>
void _DeconvolutionGradCompute(const DeconvolutionParam& param,
                               const OpContext& ctx,
                               const std::vector<TBlob>& inputs,
                               const std::vector<OpReqType>& req,
                               const std::vector<TBlob>& outputs) {
  std::vector<TBlob> in_data(inputs.begin() + 1, inputs.end());
  const TBlob& out_grad             = inputs[0];
  const std::vector<TBlob>& in_grad = outputs;

  MSHADOW_REAL_TYPE_SWITCH(out_grad.type_flag_, DType, {
    DeconvolutionOp<xpu, DType> op;
    op.Init(param);
    op.Backward(ctx, std::vector<TBlob>{out_grad}, in_data, req, in_grad);
  });
}

template <typename xpu>
void DeconvolutionGradCompute(const nnvm::NodeAttrs& attrs,
                              const OpContext& ctx,
                              const std::vector<TBlob>& inputs,
                              const std::vector<OpReqType>& req,
                              const std::vector<TBlob>& outputs) {
  const DeconvolutionParam& param = nnvm::get<DeconvolutionParam>(attrs.parsed);
  _DeconvolutionGradCompute<xpu>(param, ctx, inputs, req, outputs);
}
}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_NN_DECONVOLUTION_INL_H_
