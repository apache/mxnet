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
 * \file convolution-inl.h
 * \brief
 * \ref: https://github.com/Yangqing/caffe/wiki/Convolution-in-Caffe:-a-memo
 * \author Bing Xu, Jun Wu, Da Zheng
*/
#ifndef MXNET_OPERATOR_NN_CONVOLUTION_INL_H_
#define MXNET_OPERATOR_NN_CONVOLUTION_INL_H_

#include <mxnet/io.h>
#include <mxnet/base.h>
#include <mxnet/ndarray.h>
#include <mxnet/operator.h>
#include <mxnet/operator_util.h>
#include <dmlc/logging.h>
#include <dmlc/optional.h>
#include <algorithm>
#include <map>
#include <vector>
#include <string>
#include <utility>
#include "../operator_common.h"
#include "../linalg.h"
#include "./im2col.h"


namespace mxnet {
namespace op {

namespace conv {
enum ConvolutionOpInputs {kData, kWeight, kBias};
enum ConvolutionOpOutputs {kOut};
enum ConvolutionOpResource {kTempSpace};
enum ConvolutionOpCudnnTune {kOff, kLimited, kFastest};
}

struct ConvolutionParam : public dmlc::Parameter<ConvolutionParam> {
  TShape kernel;
  TShape stride;
  TShape dilate;
  TShape pad;
  uint32_t num_filter;
  uint32_t num_group;
  uint64_t workspace;
  bool no_bias;
  dmlc::optional<int> cudnn_tune;
  bool cudnn_off;
  dmlc::optional<int> layout;
  DMLC_DECLARE_PARAMETER(ConvolutionParam) {
    DMLC_DECLARE_FIELD(kernel).describe("Convolution kernel size: (w,), (h, w) or (d, h, w)");
    DMLC_DECLARE_FIELD(stride).set_default(TShape())
    .describe("Convolution stride: (w,), (h, w) or (d, h, w). Defaults to 1 for each dimension.");
    DMLC_DECLARE_FIELD(dilate).set_default(TShape())
    .describe("Convolution dilate: (w,), (h, w) or (d, h, w). Defaults to 1 for each dimension.");
    DMLC_DECLARE_FIELD(pad).set_default(TShape())
    .describe("Zero pad for convolution: (w,), (h, w) or (d, h, w). Defaults to no padding.");
    DMLC_DECLARE_FIELD(num_filter).set_range(1, 100000)
    .describe("Convolution filter(channel) number");
    DMLC_DECLARE_FIELD(num_group).set_default(1)
    .describe("Number of group partitions.");
    DMLC_DECLARE_FIELD(workspace).set_default(1024).set_range(0, 8192)
    .describe("Maximum temporary workspace allowed for convolution (MB).");
    DMLC_DECLARE_FIELD(no_bias).set_default(false)
    .describe("Whether to disable bias parameter.");
    DMLC_DECLARE_FIELD(cudnn_tune)
    .add_enum("off", conv::kOff)
    .add_enum("limited_workspace", conv::kLimited)
    .add_enum("fastest", conv::kFastest)
    .set_default(dmlc::optional<int>())
        .describe("Whether to pick convolution algo by running performance test.");
    DMLC_DECLARE_FIELD(cudnn_off).set_default(false)
    .describe("Turn off cudnn for this layer.");
    DMLC_DECLARE_FIELD(layout)
    .add_enum("NCW", mshadow::kNCW)
    .add_enum("NCHW", mshadow::kNCHW)
    .add_enum("NCDHW", mshadow::kNCDHW)
    .add_enum("NHWC", mshadow::kNHWC)
    .add_enum("NDHWC", mshadow::kNDHWC)
    .set_default(dmlc::optional<int>())
    .describe("Set layout for input, output and weight. Empty for\n    "
              "default layout: NCW for 1d, NCHW for 2d and NCDHW for 3d.");
  }
  // Adjusts kernel size for effects of dilation in the dimension `dim`.
  index_t DilatedKernelSize(int dim) const {
    return 1 + (kernel[dim] - 1) * dilate[dim];
  }

  bool operator==(const ConvolutionParam& other) const {
    return this->kernel == other.kernel &&
           this->stride == other.stride &&
           this->dilate == other.dilate &&
           this->pad == other.pad &&
           this->num_filter == other.num_filter &&
           this->num_group == other.num_group &&
           this->workspace == other.workspace &&
           this->no_bias == other.no_bias &&
           this->cudnn_tune == other.cudnn_tune &&
           this->cudnn_off == other.cudnn_off &&
           this->layout == other.layout;
  }
};

}  // namespace op
}  // namespace mxnet

namespace std {
template<>
struct hash<mxnet::op::ConvolutionParam> {
  size_t operator()(const mxnet::op::ConvolutionParam& val) {
    size_t ret = 0;
    ret = dmlc::HashCombine(ret, val.kernel);
    ret = dmlc::HashCombine(ret, val.stride);
    ret = dmlc::HashCombine(ret, val.dilate);
    ret = dmlc::HashCombine(ret, val.pad);
    ret = dmlc::HashCombine(ret, val.num_filter);
    ret = dmlc::HashCombine(ret, val.num_group);
    ret = dmlc::HashCombine(ret, val.workspace);
    ret = dmlc::HashCombine(ret, val.no_bias);
    ret = dmlc::HashCombine(ret, val.cudnn_tune);
    ret = dmlc::HashCombine(ret, val.cudnn_off);
    ret = dmlc::HashCombine(ret, val.layout);
    return ret;
  }
};
}  // namespace std

namespace mxnet {
namespace op {

template<typename xpu, typename DType>
class ConvolutionOp {
 public:
  void Init(ConvolutionParam p) {
    this->param_ = p;
    // convert MBytes first to Bytes and then to elements.
    param_.workspace = (param_.workspace << 20) / sizeof(DType);
    CHECK(param_.layout.value() == mshadow::kNCW ||
          param_.layout.value() == mshadow::kNCHW ||
          param_.layout.value() == mshadow::kNCDHW)
      << "Only support NCW, NCHW and NCDHW layout";
  }

  void Forward(const OpContext &ctx,
               const std::vector<TBlob> &in_data,
               const std::vector<OpReqType> &req,
               const std::vector<TBlob> &out_data) {
    using namespace mshadow;
    using namespace mshadow::expr;
    CHECK_EQ(req[conv::kOut], kWriteTo);
    size_t expected = param_.no_bias ? 2 : 3;
    CHECK_EQ(in_data.size(), expected);
    CHECK_EQ(out_data.size(), 1U);
    CHECK_EQ(req[conv::kOut], kWriteTo);
    LayerSetUp(in_data[conv::kData].shape_, out_data[conv::kOut].shape_);
    Stream<xpu>* s = ctx.get_stream<xpu>();

    // initialize weight and col_buffer 3D tensors for using gemm
    index_t M = conv_out_channels_ / group_;
    index_t N = conv_out_spatial_dim_;
    index_t K = kernel_dim_;
    Tensor<xpu, 3, DType> weight_3d = in_data[conv::kWeight].get_with_shape<xpu, 3, DType>(
      Shape3(group_, M, K), s);
    Tensor<xpu, 4, DType> output_4d = out_data[conv::kOut].get_with_shape<xpu, 4, DType>(
      Shape4(num_, group_, M, N), s);

    // no need to allocating memory and reordering in memory
    if (is_1x1_) {
      Tensor<xpu, 4, DType> input_4d = in_data[conv::kData].get_with_shape<xpu, 4, DType>(
        Shape4(num_, group_, K, N), s);
      for (index_t n = 0; n < num_; ++n) {
        Tensor<xpu, 3, DType> input_3d = input_4d[n];
        Tensor<xpu, 3, DType> output_3d = output_4d[n];
        for (index_t g = 0; g < group_; ++g) {
          linalg_gemm(weight_3d[g], input_3d[g], output_3d[g], false, false, s, req[conv::kOut]);
        }
      }
    } else {
      // allocate workspace for col_buffer
      Tensor<xpu, 1, DType> workspace = ctx.requested[conv::kTempSpace]
        .get_space_typed<xpu, 1, DType>(Shape1(col_buffer_size_), s);
      // calculate the shape of col_buffer
      TShape col_buffer_shape(num_spatial_axes_ + 1);
      col_buffer_shape[0] = conv_in_channels_ * param_.kernel.Size();
      for (index_t i = 1; i < col_buffer_shape.ndim(); ++i) {
        col_buffer_shape[i] = out_data[0].shape_[i+1];
      }
      // create a column buffer using workspace and col_buffer_shape
      TBlob col_buffer(workspace.dptr_, col_buffer_shape, xpu::kDevMask, DataType<DType>::kFlag);
      Tensor<xpu, 3, DType> col_buffer_3d = col_buffer.get_with_shape<xpu, 3, DType>(
        Shape3(group_, K, N), s);
      for (index_t n = 0; n < num_; ++n) {
        // transform image to col_buffer in order to use gemm
        im2col(s, in_data[conv::kData].dptr<DType>()+n*input_dim_, in_data[conv::kData].shape_,
               col_buffer.shape_, param_.kernel, param_.pad, param_.stride, param_.dilate,
               col_buffer.dptr<DType>());
        Tensor<xpu, 3, DType> output_3d = output_4d[n];
        for (index_t g = 0; g < group_; ++g) {
          // Legacy approach shown here for comparison:
          //   Assign(output_3d[g], req[conv::kOut], dot(weight_3d[g], col_buffer_3d[g]));
          linalg_gemm(weight_3d[g], col_buffer_3d[g], output_3d[g], false, false, s,
            req[conv::kOut]);
        }
      }
    }

    if (bias_term_) {
      Tensor<xpu, 1, DType> bias = in_data[conv::kBias].get<xpu, 1, DType>(s);
      Tensor<xpu, 3, DType> output_3d = out_data[conv::kOut].get_with_shape<xpu, 3, DType>(
        Shape3(num_, conv_out_channels_, conv_out_spatial_dim_), s);
      // has bias term, broadcast it to the same shape of output_3d in channel dim
      output_3d += mshadow::expr::broadcast<1>(bias, output_3d.shape_);
    }
  }

  void Backward(const OpContext &ctx,
                const std::vector<TBlob>& out_grad,
                const std::vector<TBlob>& in_data,
                const std::vector<OpReqType>& req,
                const std::vector<TBlob>& in_grad) {
    using namespace mshadow;
    using namespace mshadow::expr;
    CHECK_EQ(out_grad.size(), 1U);
    // We expect 2 inputs: in data and weight. We don't need bias for
    // computing gradient.
    size_t expected = param_.no_bias == 0 ? 3 : 2;
    CHECK_EQ(in_data.size(), expected);
    CHECK_EQ(in_grad.size(), expected);
    CHECK_EQ(req.size(), expected);
    CHECK_EQ(in_data[conv::kWeight].CheckContiguous(), true);
    LayerSetUp(in_grad[conv::kData].shape_, out_grad[conv::kOut].shape_);
    Stream<xpu> *s = ctx.get_stream<xpu>();

    // initialize weight and col_buffer 3D tensors for using gemm
    // For computing dLoss/d(in_data[kData])
    index_t M = kernel_dim_;
    index_t N = conv_out_spatial_dim_;
    index_t K = conv_out_channels_ / group_;
    Tensor<xpu, 3, DType> weight_3d = in_data[conv::kWeight].get_with_shape<xpu, 3, DType>(
      Shape3(group_, K, M), s);
    Tensor<xpu, 4, DType> out_grad_4d = out_grad[conv::kOut].get_with_shape<xpu, 4, DType>(
      Shape4(num_, group_, K, N), s);
    // For computing dLoss/dWeight
    Tensor<xpu, 3, DType> dweight_3d = in_grad[conv::kWeight].get_with_shape<xpu, 3, DType>(
      Shape3(group_, K, M), s);

    // no need to allocating memory and reordering in memory
    if (is_1x1_) {
      Tensor<xpu, 4, DType> input_4d = in_data[conv::kData].get_with_shape<xpu, 4, DType>(
        Shape4(num_, group_, M, N), s);
      Tensor<xpu, 4, DType> in_grad_4d = in_grad[conv::kData].get_with_shape<xpu, 4, DType>(
        Shape4(num_, group_, M, N), s);
      for (index_t n = 0; n < num_; ++n) {
        Tensor<xpu, 3, DType> input_3d = input_4d[n];
        Tensor<xpu, 3, DType> in_grad_3d = in_grad_4d[n];
        Tensor<xpu, 3, DType> out_grad_3d = out_grad_4d[n];
        // gradient w.r.t. input data
        for (index_t g = 0; g < group_; ++g) {
          linalg_gemm(weight_3d[g], out_grad_3d[g], in_grad_3d[g], true, false, s);
          auto request = (n == 0) ? req[conv::kWeight] : kAddTo;
          linalg_gemm(out_grad_3d[g], input_3d[g], dweight_3d[g], false, true, s, request);
        }
      }
    } else {
      // allocate workspace for col_buffer
      Tensor<xpu, 1, DType> workspace = ctx.requested[conv::kTempSpace]
        .get_space_typed<xpu, 1, DType>(Shape1(col_buffer_size_), s);
      // calculate the shape of col_buffer
      TShape col_buffer_shape(num_spatial_axes_ + 1);
      col_buffer_shape[0] = conv_in_channels_ * param_.kernel.Size();
      for (index_t i = 1; i < col_buffer_shape.ndim(); ++i) {
        col_buffer_shape[i] = out_grad[conv::kData].shape_[i+1];
      }
      // create a column buffer using workspace and col_buffer_shape
      TBlob col_buffer(workspace.dptr_, col_buffer_shape, xpu::kDevMask, DataType<DType>::kFlag);
      Tensor<xpu, 3, DType> col_buffer_3d = col_buffer.get_with_shape<xpu, 3, DType>(
        Shape3(group_, M, N), s);
      for (index_t n = 0; n < num_; ++n) {
        Tensor<xpu, 3, DType> out_grad_3d = out_grad_4d[n];
        // gradient w.r.t. input data
        for (index_t g = 0; g < group_; ++g) {
          // Legacy approach shown here for comparison:
          //   col_buffer_3d[g] = dot(weight_3d[g].T(), out_grad_3d[g]);
          linalg_gemm(weight_3d[g], out_grad_3d[g], col_buffer_3d[g], true, false, s);
        }
        col2im(s, col_buffer.dptr<DType>(), in_grad[conv::kData].shape_, col_buffer.shape_,
               param_.kernel, param_.pad, param_.stride, param_.dilate,
               in_grad[conv::kData].dptr<DType>()+n*input_dim_, req[conv::kData]);

        // gradient w.r.t. weight, dWeight should accumulate across the batch and group
        im2col(s, in_data[conv::kData].dptr<DType>()+n*input_dim_, in_data[conv::kData].shape_,
               col_buffer.shape_, param_.kernel, param_.pad, param_.stride, param_.dilate,
               col_buffer.dptr<DType>());
        for (index_t g = 0; g < group_; ++g) {
          auto request = (n == 0) ? req[conv::kWeight] : kAddTo;
          // Legacy approach shown here for comparison:
          //   Assign(dweight_3d[g], request, dot(out_grad_3d[g], col_buffer_3d[g].T()));
          linalg_gemm(out_grad_3d[g], col_buffer_3d[g], dweight_3d[g], false, true, s, request);
        }
      }
    }

    // gradient w.r.t bias
    if (bias_term_) {
      Tensor<xpu, 1, DType> dbias = in_grad[conv::kBias].get<xpu, 1, DType>(s);
      Tensor<xpu, 3, DType> dout = out_grad[conv::kOut].get_with_shape<xpu, 3, DType>(
          Shape3(num_, conv_out_channels_, conv_out_spatial_dim_), s);
      ASSIGN_DISPATCH(dbias, req[conv::kBias], sumall_except_dim<1>(dout));
    }
  }

 private:
  void LayerSetUp(const TShape& ishape, const TShape& oshape) {
    channel_axis_ = 1;  // hard code channel axis
    const index_t first_spatial_axis = channel_axis_ + 1;
    const index_t num_axes = param_.kernel.ndim() + 2;
    num_spatial_axes_ = num_axes - first_spatial_axis;
    is_1x1_ = true;
    for (index_t i = 0; i < param_.kernel.ndim(); ++i) {
      is_1x1_ &= param_.kernel[i] == 1 && param_.stride[i] == 1 && param_.pad[i] == 0;
      if (!is_1x1_) break;
    }

    // batch size
    num_ = ishape[0];
    // number of input channels
    channels_ = ishape[1];
    group_ = param_.num_group;
    conv_out_channels_ = param_.num_filter;
    conv_in_channels_ = channels_;
    bias_term_ = !param_.no_bias;
    kernel_dim_ = conv_in_channels_ / group_ * param_.kernel.Size();
    weight_offset_ = conv_out_channels_ * kernel_dim_ / group_;
    conv_out_spatial_dim_ = oshape.ProdShape(2, oshape.ndim());
    col_offset_ = kernel_dim_ * conv_out_spatial_dim_;
    output_offset_ = conv_out_channels_ * conv_out_spatial_dim_ / group_;
    // size of the column buffer used for storing im2col-ed pixels
    col_buffer_size_ = kernel_dim_ * group_ * conv_out_spatial_dim_;
    // input/output image size (#channels * height * width)
    input_dim_ = ishape.ProdShape(1, ishape.ndim());
    output_dim_ = oshape.ProdShape(1, oshape.ndim());
    num_kernels_im2col_ = conv_in_channels_ * conv_out_spatial_dim_;
    num_kernels_col2im_ = input_dim_;
  }

 private:
  ConvolutionParam param_;
  index_t channel_axis_;  // channel axis of the input
  index_t channels_;  // number of channels of input image
  index_t num_spatial_axes_;  // number of spatial axes
  index_t num_;  // batch size
  index_t group_;  // number of groups
  index_t conv_out_channels_;  // number of output channels (num_filter)
  index_t conv_out_spatial_dim_;  // number of pixels of output images per channel
  index_t conv_in_channels_;  // number of input channels
  index_t kernel_dim_;  // number of input channels per group * kernel size
  index_t weight_offset_;  // number of output channels per group * kernel_dim_
  index_t col_offset_;
  index_t output_offset_;
  index_t col_buffer_size_;
  index_t input_dim_;
  index_t output_dim_;
  index_t num_kernels_im2col_;
  index_t num_kernels_col2im_;
  bool bias_term_;  // has bias term?
  bool is_1x1_;
};  // class ConvolutionOp

template<typename xpu>
void ConvolutionCompute(const nnvm::NodeAttrs& attrs,
                        const OpContext& ctx, const std::vector<TBlob>& inputs,
                        const std::vector<OpReqType>& req,
                        const std::vector<TBlob>& outputs) {
  const ConvolutionParam& param = nnvm::get<ConvolutionParam>(attrs.parsed);
  MSHADOW_REAL_TYPE_SWITCH(inputs[conv::kData].type_flag_, DType, {
    ConvolutionOp<xpu, DType> op;
    op.Init(param);
    op.Forward(ctx, inputs, req, outputs);
  });
}

template<typename xpu>
void ConvolutionGradCompute(const nnvm::NodeAttrs& attrs,
                            const OpContext& ctx, const std::vector<TBlob>& inputs,
                            const std::vector<OpReqType>& req,
                            const std::vector<TBlob>& outputs) {
  const ConvolutionParam& param = nnvm::get<ConvolutionParam>(attrs.parsed);
  std::vector<TBlob> in_data(inputs.begin() + 1, inputs.end());
  const TBlob &out_grad = inputs[0];
  const std::vector<TBlob> &in_grad = outputs;

  MSHADOW_REAL_TYPE_SWITCH(out_grad.type_flag_, DType, {
    ConvolutionOp<xpu, DType> op;
    op.Init(param);
    op.Backward(ctx, std::vector<TBlob>{out_grad}, in_data, req, in_grad);
  });
}

}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_NN_CONVOLUTION_INL_H_
