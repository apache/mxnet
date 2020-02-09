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
 * Copyright (c) 2018 Microsoft
 * Licensed under The MIT License [see LICENSE for details]
 * \file modulated_deformable_convolution-inl.h
 * \brief
 * \ref: https://github.com/Yangqing/caffe/wiki/Convolution-in-Caffe:-a-memo
 * \ref: https://arxiv.org/abs/1811.11168
 * \author Yuwen Xiong, Haozhi Qi, Jifeng Dai, Xizhou Zhu, Han Hu
*/
#ifndef MXNET_OPERATOR_CONTRIB_MODULATED_DEFORMABLE_CONVOLUTION_INL_H_
#define MXNET_OPERATOR_CONTRIB_MODULATED_DEFORMABLE_CONVOLUTION_INL_H_

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
#include "../nn/im2col.h"
#include "./nn/modulated_deformable_im2col.h"
#include "../linalg.h"


namespace mxnet {
namespace op {

namespace dmconv {
  enum ModulatedDeformableConvolutionOpInputs { kData, kOffset, kMask, kWeight, kBias };
  enum ModulatedDeformableConvolutionOpOutputs { kOut };
  enum ModulatedDeformableConvolutionOpResource { kTempSpace };
}

struct ModulatedDeformableConvolutionParam
  : public dmlc::Parameter<ModulatedDeformableConvolutionParam> {
  mxnet::TShape kernel;
  mxnet::TShape stride;
  mxnet::TShape dilate;
  mxnet::TShape pad;
  uint32_t num_filter;
  uint32_t num_group;
  uint32_t num_deformable_group;
  uint64_t workspace;
  bool no_bias;
  uint32_t im2col_step;
  dmlc::optional<int> layout;
  DMLC_DECLARE_PARAMETER(ModulatedDeformableConvolutionParam) {
    DMLC_DECLARE_FIELD(kernel).describe("Convolution kernel size: (h, w) or (d, h, w)");
    DMLC_DECLARE_FIELD(stride).set_default(mxnet::TShape(0, -1))
      .describe("Convolution stride: (h, w) or (d, h, w). Defaults to 1 for each dimension.");
    DMLC_DECLARE_FIELD(dilate).set_default(mxnet::TShape(0, -1))
      .describe("Convolution dilate: (h, w) or (d, h, w). Defaults to 1 for each dimension.");
    DMLC_DECLARE_FIELD(pad).set_default(mxnet::TShape(0, -1))
      .describe("Zero pad for convolution: (h, w) or (d, h, w). Defaults to no padding.");
    DMLC_DECLARE_FIELD(num_filter).set_range(1, 100000)
      .describe("Convolution filter(channel) number");
    DMLC_DECLARE_FIELD(num_group).set_default(1)
      .describe("Number of group partitions.");
    DMLC_DECLARE_FIELD(num_deformable_group).set_default(1)
      .describe("Number of deformable group partitions.");
    DMLC_DECLARE_FIELD(workspace).set_default(1024).set_range(0, 8192)
      .describe("Maximum temperal workspace allowed for convolution (MB).");
    DMLC_DECLARE_FIELD(no_bias).set_default(false)
      .describe("Whether to disable bias parameter.");
    DMLC_DECLARE_FIELD(im2col_step).set_default(64)
      .describe("Maximum number of images per im2col computation; "
                "The total batch size should be divisable by this value or "
                "smaller than this value; if you face out of memory problem, "
                "you can try to use a smaller value here.");
    DMLC_DECLARE_FIELD(layout)
      .add_enum("NCW", mshadow::kNCW)
      .add_enum("NCHW", mshadow::kNCHW)
      .add_enum("NCDHW", mshadow::kNCDHW)
      .set_default(dmlc::optional<int>())
      .describe("Set layout for input, output and weight. Empty for\n    "
        "default layout: NCW for 1d, NCHW for 2d and NCDHW for 3d.");
  }
};

template<typename xpu, typename DType>
class ModulatedDeformableConvolutionOp : public Operator {
 public:
  explicit ModulatedDeformableConvolutionOp(ModulatedDeformableConvolutionParam p) {
    this->param_ = p;
    // convert MBytes first to Bytes and then to elements.
    param_.workspace = (param_.workspace << 20) / sizeof(DType);
    CHECK(param_.layout.value() == mshadow::kNCW ||
      param_.layout.value() == mshadow::kNCHW ||
      param_.layout.value() == mshadow::kNCDHW)
      << "Only support NCW, NCHW and NCDHW layout";
  }

  virtual void Forward(const OpContext &ctx,
    const std::vector<TBlob> &in_data,
    const std::vector<OpReqType> &req,
    const std::vector<TBlob> &out_data,
    const std::vector<TBlob> &aux_args) {
    using namespace mshadow;
    using namespace mshadow::expr;
    CHECK_EQ(req[dmconv::kOut], kWriteTo);
    size_t expected = param_.no_bias ? 4 : 5;
    CHECK_EQ(in_data.size(), expected);
    CHECK_EQ(out_data.size(), 1U);
    LayerSetUp(in_data[dmconv::kData].shape_,
               in_data[dmconv::kOffset].shape_,
               in_data[dmconv::kMask].shape_,
               out_data[dmconv::kOut].shape_);
    Stream<xpu>* s = ctx.get_stream<xpu>();
    // allocate workspace for col_buffer
    Tensor<xpu, 1, DType> workspace = ctx.requested[dmconv::kTempSpace]
      .get_space_typed<xpu, 1, DType>(Shape1(col_buffer_size_ + num_*output_dim_), s);
    // calculate the shape of col_buffer
    mxnet::TShape col_buffer_shape(num_spatial_axes_ + 2, -1);
    col_buffer_shape[0] = conv_in_channels_ * param_.kernel.Size();
    //  for (index_t i = 1; i < col_buffer_shape.ndim(); ++i) {
    //  col_buffer_shape[i] = out_data[0].shape_[i + 1];
    col_buffer_shape[1] = im2col_step_;
    for (index_t i = 2; i < col_buffer_shape.ndim(); ++i) {
      col_buffer_shape[i] = out_data[0].shape_[i];
    }
    // create a column buffer using workspace and col_buffer_shape
    TBlob col_buffer(workspace.dptr_, col_buffer_shape, xpu::kDevMask, DataType<DType>::kFlag);
    mxnet::TShape output_buffer_shape(1, -1);
    output_buffer_shape[0] = num_*output_dim_;
    TBlob output_buffer(workspace.dptr_ + col_buffer_size_, output_buffer_shape,
                        xpu::kDevMask, DataType<DType>::kFlag);

    // initialize weight and col_buffer 3D tensors for using gemm
    index_t M = conv_out_channels_ / group_;
    index_t N = im2col_step_ * conv_out_spatial_dim_;
    index_t K = kernel_dim_;
    Tensor<xpu, 3, DType> weight_3d = in_data[dmconv::kWeight].get_with_shape<xpu, 3, DType>(
      Shape3(group_, M, K), s);
    Tensor<xpu, 3, DType> col_buffer_3d = col_buffer.get_with_shape<xpu, 3, DType>(
      Shape3(group_, K, N), s);
    Tensor<xpu, 4, DType> output_4d = output_buffer.get_with_shape<xpu, 4, DType>(
                Shape4(num_ / im2col_step_, group_, M, N), s);
    for (index_t n = 0; n < num_ / im2col_step_; ++n) {
      // transform image to col_buffer in order to use gemm
      modulated_deformable_im2col(s,
              in_data[dmconv::kData].dptr<DType>() + n*im2col_step_*input_dim_,
              in_data[dmconv::kOffset].dptr<DType>() + n*im2col_step_*input_offset_dim_,
              in_data[dmconv::kMask].dptr<DType>() + n*im2col_step_ * input_mask_dim_,
              in_data[dmconv::kData].shape_,
              col_buffer.shape_, param_.kernel, param_.pad, param_.stride, param_.dilate,
              param_.num_deformable_group, col_buffer.dptr<DType>());
      Tensor<xpu, 3, DType> output_3d = output_4d[n];
      for (index_t g = 0; g < group_; ++g) {
        // Legacy approach shown here for comparison:
        //   Assign(output_3d[g], req[dmconv::kOut], dot(weight_3d[g], col_buffer_3d[g]));
        linalg_gemm(weight_3d[g], col_buffer_3d[g], output_3d[g], false, false, s, kWriteTo);
      }
    }
    Tensor<xpu, 4, DType> trans_output_4d = output_buffer.get_with_shape<xpu, 4, DType>(
        Shape4(num_ / im2col_step_, conv_out_channels_, im2col_step_, conv_out_spatial_dim_), s);
    Tensor<xpu, 4, DType> original_output_4d = out_data[dmconv::kOut].get_with_shape<xpu, 4, DType>(
        Shape4(num_ / im2col_step_, im2col_step_, conv_out_channels_, conv_out_spatial_dim_), s);
    original_output_4d = swapaxis<2, 1>(trans_output_4d);

    if (bias_term_) {
      Tensor<xpu, 1, DType> bias = in_data[dmconv::kBias].get<xpu, 1, DType>(s);
      Tensor<xpu, 3, DType> output_3d = out_data[dmconv::kOut].get_with_shape<xpu, 3, DType>(
        Shape3(num_, conv_out_channels_, conv_out_spatial_dim_), s);
      // has bias term, broadcast it to the same shape of output_3d in channel dim
      output_3d += mshadow::expr::broadcast<1>(bias, output_3d.shape_);
    }
  }

  virtual void Backward(const OpContext &ctx,
    const std::vector<TBlob>& out_grad,
    const std::vector<TBlob>& in_data,
    const std::vector<TBlob>& out_data,
    const std::vector<OpReqType>& req,
    const std::vector<TBlob>& in_grad,
    const std::vector<TBlob>& aux_args) {
    using namespace mshadow;
    using namespace mshadow::expr;
    CHECK_EQ(out_grad.size(), 1U);
    size_t expected = param_.no_bias == 0 ? 5 : 4;
    CHECK(in_data.size() == expected && in_grad.size() == expected);
    CHECK_EQ(req.size(), expected);
    CHECK_EQ(in_data[dmconv::kWeight].CheckContiguous(), true);
    LayerSetUp(in_grad[dmconv::kData].shape_,
               in_grad[dmconv::kOffset].shape_,
               in_grad[dmconv::kMask].shape_,
               out_grad[dmconv::kOut].shape_);
    Stream<xpu> *s = ctx.get_stream<xpu>();
    // allocate workspace for col_buffer
    Tensor<xpu, 1, DType> workspace = ctx.requested[dmconv::kTempSpace]
      .get_space_typed<xpu, 1, DType>(Shape1(col_buffer_size_ + num_*output_dim_), s);
    // calculate the shape of col_buffer
    mxnet::TShape col_buffer_shape(num_spatial_axes_ + 2, -1);
    col_buffer_shape[0] = conv_in_channels_ * param_.kernel.Size();
    col_buffer_shape[1] = im2col_step_;
    for (index_t i = 2; i < col_buffer_shape.ndim(); ++i) {
      col_buffer_shape[i] = out_grad[dmconv::kData].shape_[i];
    }
    // create a column buffer using workspace and col_buffer_shape
    TBlob col_buffer(workspace.dptr_, col_buffer_shape, xpu::kDevMask, DataType<DType>::kFlag);
    mxnet::TShape output_buffer_shape(1, -1);
    output_buffer_shape[0] = num_*output_dim_;
    TBlob output_buffer(workspace.dptr_ + col_buffer_size_,
      output_buffer_shape, xpu::kDevMask, DataType<DType>::kFlag);

    Tensor<xpu, 4, DType> trans_output_4d = output_buffer.get_with_shape<xpu, 4, DType>(
        Shape4(num_ / im2col_step_, conv_out_channels_, im2col_step_, conv_out_spatial_dim_), s);
    Tensor<xpu, 4, DType> original_output_4d = out_grad[dmconv::kOut].get_with_shape<xpu, 4, DType>(
        Shape4(num_ / im2col_step_, im2col_step_, conv_out_channels_, conv_out_spatial_dim_), s);
    trans_output_4d = swapaxis<2, 1>(original_output_4d);

    // initialize weight and col_buffer 3D tensors for using gemm
    // For computing dLoss/d(in_data[kData])
    index_t M = kernel_dim_;
    index_t N = im2col_step_ * conv_out_spatial_dim_;
    index_t K = conv_out_channels_ / group_;
    Tensor<xpu, 3, DType> weight_3d = in_data[dmconv::kWeight].get_with_shape<xpu, 3, DType>(
      Shape3(group_, K, M), s);
    Tensor<xpu, 4, DType> out_grad_4d = output_buffer.get_with_shape<xpu, 4, DType>(
          Shape4(num_ / im2col_step_, group_, K, N), s);
    Tensor<xpu, 3, DType> col_buffer_3d = col_buffer.get_with_shape<xpu, 3, DType>(
      Shape3(group_, M, N), s);
    // For computing dLoss/dWeight
    Tensor<xpu, 3, DType> dweight_3d = in_grad[dmconv::kWeight].get_with_shape<xpu, 3, DType>(
      Shape3(group_, K, M), s);

    Tensor<xpu, 1, DType> data_grad = in_grad[dmconv::kData].FlatTo1D<xpu, DType>(s);
    if (req[dmconv::kData] == kWriteTo)
        data_grad = 0;


    for (index_t n = 0; n < num_ / im2col_step_; ++n) {
      Tensor<xpu, 3, DType> out_grad_3d = out_grad_4d[n];
      for (index_t g = 0; g < group_; ++g) {
        // Legacy approach shown here for comparison:
        //   col_buffer_3d[g] = dot(weight_3d[g].T(), out_grad_3d[g]);
        linalg_gemm(weight_3d[g], out_grad_3d[g], col_buffer_3d[g], true, false, s);
      }

      // gradient w.r.t. input coordinate data
      modulated_deformable_col2im_coord(s, col_buffer.dptr<DType>(),
        in_data[dmconv::kData].dptr<DType>() + n*im2col_step_*input_dim_,
        in_data[dmconv::kOffset].dptr<DType>() + n*im2col_step_*input_offset_dim_,
        in_data[dmconv::kMask].dptr<DType>() + n*im2col_step_*input_mask_dim_,
        in_grad[dmconv::kData].shape_, col_buffer.shape_,
        param_.kernel, param_.pad, param_.stride, param_.dilate, param_.num_deformable_group,
        in_grad[dmconv::kOffset].dptr<DType>() + n*im2col_step_*input_offset_dim_,
        in_grad[dmconv::kMask].dptr<DType>() + n*im2col_step_*input_mask_dim_,
        req[dmconv::kOffset], req[dmconv::kMask]);

      // gradient w.r.t. input data
      modulated_deformable_col2im(s, col_buffer.dptr<DType>(),
        in_data[dmconv::kOffset].dptr<DType>() + n*im2col_step_*input_offset_dim_,
        in_data[dmconv::kMask].dptr<DType>() + n*im2col_step_*input_mask_dim_,
        in_grad[dmconv::kData].shape_, col_buffer.shape_,
        param_.kernel, param_.pad, param_.stride, param_.dilate, param_.num_deformable_group,
        in_grad[dmconv::kData].dptr<DType>() + n*im2col_step_*input_dim_,
        req[dmconv::kData]);

      // gradient w.r.t. weight, dWeight should accumulate across the batch and group
      modulated_deformable_im2col(s,
        in_data[dmconv::kData].dptr<DType>() + n*im2col_step_*input_dim_,
        in_data[dmconv::kOffset].dptr<DType>() + n*im2col_step_*input_offset_dim_,
        in_data[dmconv::kMask].dptr<DType>() + n*im2col_step_*input_mask_dim_,
        in_data[dmconv::kData].shape_,
        col_buffer.shape_, param_.kernel, param_.pad, param_.stride, param_.dilate,
        param_.num_deformable_group, col_buffer.dptr<DType>());

      for (index_t g = 0; g < group_; ++g) {
        auto request = (n == 0) ? req[dmconv::kWeight] : kAddTo;
        // Legacy approach shown here for comparison:
        //   Assign(dweight_3d[g], request, dot(out_grad_3d[g], col_buffer_3d[g].T()));
        linalg_gemm(out_grad_3d[g], col_buffer_3d[g], dweight_3d[g], false, true, s, request);
      }
    }

    // gradient w.r.t bias
    if (bias_term_) {
      Tensor<xpu, 1, DType> dbias = in_grad[dmconv::kBias].get<xpu, 1, DType>(s);
      Tensor<xpu, 3, DType> dout = out_grad[dmconv::kOut].get_with_shape<xpu, 3, DType>(
        Shape3(num_, conv_out_channels_, conv_out_spatial_dim_), s);
      ASSIGN_DISPATCH(dbias, req[dmconv::kBias], sumall_except_dim<1>(dout));
    }
  }

 private:
  void LayerSetUp(const mxnet::TShape& ishape, const mxnet::TShape& offset_shape,
                  const mxnet::TShape& mask_shape, const mxnet::TShape& oshape) {
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
    im2col_step_ = std::min(param_.im2col_step, static_cast<uint32_t>(num_));
    col_buffer_size_ = kernel_dim_ * group_ * im2col_step_ * conv_out_spatial_dim_;
    // input/output image size (#channels * height * width)
    input_dim_ = ishape.ProdShape(1, ishape.ndim());
    input_offset_dim_ = offset_shape.ProdShape(1, offset_shape.ndim());
    input_mask_dim_ = mask_shape.ProdShape(1, mask_shape.ndim());
    output_dim_ = oshape.ProdShape(1, oshape.ndim());
    num_kernels_im2col_ = conv_in_channels_ * conv_out_spatial_dim_;
    num_kernels_col2im_ = input_dim_;
  }

 private:
  ModulatedDeformableConvolutionParam param_;
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
  index_t input_offset_dim_;
  index_t input_mask_dim_;
  index_t output_dim_;
  index_t num_kernels_im2col_;
  index_t num_kernels_col2im_;
  index_t im2col_step_;
  bool bias_term_;  // has bias term?
  bool is_1x1_;
};  // class ConvolutionOp

template<typename xpu>
Operator* CreateOp(ModulatedDeformableConvolutionParam param, int dtype,
  std::vector<mxnet::TShape> *in_shape,
  std::vector<mxnet::TShape> *out_shape,
  Context ctx);

#if DMLC_USE_CXX11
class ModulatedDeformableConvolutionProp : public OperatorProperty {
 public:
  std::vector<std::string> ListArguments() const override {
    if (!param_.no_bias) {
      return{ "data", "offset", "mask", "weight", "bias" };
    } else {
      return{ "data", "offset", "mask", "weight" };
    }
  }

  void Init(const std::vector<std::pair<std::string, std::string> >& kwargs) override {
    using namespace mshadow;
    param_.Init(kwargs);
    if (param_.kernel.ndim() == 2) {
      param_.layout = param_.layout ? param_.layout.value() : mshadow::kNCHW;
      if (param_.stride.ndim() == 0) param_.stride = Shape2(1, 1);
      if (param_.dilate.ndim() == 0) param_.dilate = Shape2(1, 1);
      if (param_.pad.ndim() == 0) param_.pad = Shape2(0, 0);
    } else {
      LOG(FATAL) << "not implemented";
    }
  }

  std::map<std::string, std::string> GetParams() const override {
    return param_.__DICT__();
  }

  bool InferShape(std::vector<mxnet::TShape> *in_shape,
    std::vector<mxnet::TShape> *out_shape,
    std::vector<mxnet::TShape> *aux_shape) const override {
    using namespace mshadow;
    if (!param_.no_bias) {
      CHECK_EQ(in_shape->size(), 5U) << "Input:[data, offset, mask, weight, bias]";
    } else {
      CHECK_EQ(in_shape->size(), 4U) << "Input:[data, offset, mask, weight]";
    }
    out_shape->resize(1, mxnet::TShape());
    const mxnet::TShape &dshp = (*in_shape)[dmconv::kData];
    const mxnet::TShape &oshp = (*in_shape)[dmconv::kOffset];
    const mxnet::TShape &mshp = (*in_shape)[dmconv::kMask];
    if (dshp.ndim() == 0) return false;
    if (param_.kernel.ndim() == 2) {
      // 2d dmconv
      CHECK_EQ(dshp.ndim(), 4U) \
        << "Input data should be 4D in batch-num_filter-y-x";
      CHECK_EQ(oshp.ndim(), 4U) \
        << "Input offset should be 4D in batch-num_filter-y-x";
      CHECK_EQ(mshp.ndim(), 4U) \
        << "Input offset should be 4D in batch-num_filter-y-x";
      Shape<4> dshape = ConvertLayout(dshp.get<4>(), param_.layout.value(), kNCHW);
      Shape<4> offsetshape = ConvertLayout(oshp.get<4>(), param_.layout.value(), kNCHW);
      Shape<4> maskshape = ConvertLayout(mshp.get<4>(), param_.layout.value(), kNCHW);
      Shape<4> wshape = Shape4(param_.num_filter / param_.num_group, dshape[1] / param_.num_group,
        param_.kernel[0], param_.kernel[1]);
      wshape = ConvertLayout(wshape, kNCHW, param_.layout.value());
      wshape[0] *= param_.num_group;
      SHAPE_ASSIGN_CHECK(*in_shape, dmconv::kWeight, wshape);
      if (!param_.no_bias) {
        SHAPE_ASSIGN_CHECK(*in_shape, dmconv::kBias, Shape1(param_.num_filter));
      }

      const index_t ksize_y = static_cast<index_t>(param_.kernel[0]);
      const index_t ksize_x = static_cast<index_t>(param_.kernel[1]);
      if (dshape[0] > static_cast<index_t>(param_.im2col_step)) {
           CHECK_EQ(dshape[0] % param_.im2col_step, 0U) \
           << "input batchsize must be smaller than or divide im2col_step";
      }
      CHECK_EQ(dshape[1] % param_.num_group, 0U) \
        << "input num_filter must divide group size";
      CHECK_EQ(dshape[1] % param_.num_deformable_group, 0U) \
        << "input num_filter must divide deformable group size";
      CHECK_EQ(param_.num_filter % param_.num_group, 0U) \
        << "output num_filter must divide group size";
      CHECK_GT(param_.kernel.Size(), 0U) \
        << "incorrect kernel size: " << param_.kernel;
      CHECK_GT(param_.stride.Size(), 0U) \
        << "incorrect stride size: " << param_.stride;
      CHECK_GT(param_.dilate.Size(), 0U) \
        << "incorrect dilate size: " << param_.dilate;
      Shape<4> oshape;
      oshape[0] = dshape[0];
      oshape[1] = param_.num_filter;
      oshape[2] = (dshape[2] + 2 * param_.pad[0] -
        (param_.dilate[0] * (ksize_y - 1) + 1)) / param_.stride[0] + 1;
      oshape[3] = (dshape[3] + 2 * param_.pad[1] -
        (param_.dilate[1] * (ksize_x - 1) + 1)) / param_.stride[1] + 1;
      SHAPE_ASSIGN_CHECK(*out_shape, 0, ConvertLayout(oshape, kNCHW, param_.layout.value()));
      CHECK_EQ(oshape[1] % param_.num_deformable_group, 0U) \
        << "output num_filter must divide deformable group size";
      CHECK_EQ(oshape[2], offsetshape[2]) \
        << "output height must equal to offset map height";
      CHECK_EQ(oshape[3], offsetshape[3]) \
        << "output width must equal to offset map width";
      CHECK_EQ(offsetshape[1] % (param_.kernel[0] * param_.kernel[1]), 0U) \
        << "offset filter must divide deformable group size";
      CHECK_EQ(offsetshape[1] / (2 * param_.kernel[0] * param_.kernel[1]), \
               param_.num_deformable_group) \
        << "offset filter must divide deformable group size";
      CHECK_EQ(oshape[2], maskshape[2]) \
        << "output height must equal to mask map height";
      CHECK_EQ(oshape[3], maskshape[3]) \
        << "output width must equal to mask map width";
      CHECK_EQ(maskshape[1] % (param_.kernel[0] * param_.kernel[1]), 0U) \
        << "offset filter must divide deformable group size";
      CHECK_EQ(maskshape[1] / (param_.kernel[0] * param_.kernel[1]), \
               param_.num_deformable_group) \
        << "offset filter must divide deformable group size";
      // Perform incomplete shape inference. Fill in the missing values in data shape.
      // 1) We can always fill in the batch_size.
      // 2) We can back-calculate the input height/width if the corresponding stride is 1.
      oshape = ConvertLayout((*out_shape)[0].get<4>(), param_.layout.value(), kNCHW);
      dshape[0] = oshape[0];
      if (param_.stride[0] == 1) {
        dshape[2] = oshape[2] + param_.dilate[0] * (ksize_y - 1) - 2 * param_.pad[0];
      }
      if (param_.stride[1] == 1) {
        dshape[3] = oshape[3] + param_.dilate[1] * (ksize_x - 1) - 2 * param_.pad[1];
      }
      SHAPE_ASSIGN_CHECK(*in_shape, dmconv::kData,
        ConvertLayout(dshape, kNCHW, param_.layout.value()));
      // Check whether the kernel sizes are valid
      if (dshape[2] != 0) {
        CHECK_LE(ksize_y, dshape[2] + 2 * param_.pad[0]) << "kernel size exceed input";
      }
      if (dshape[3] != 0) {
        CHECK_LE(ksize_x, dshape[3] + 2 * param_.pad[1]) << "kernel size exceed input";
      }
      return true;
    } else {
      LOG(FATAL) << "not implemented";
      return false;
    }
  }

  bool InferType(std::vector<int> *in_type,
    std::vector<int> *out_type,
    std::vector<int> *aux_type) const override {
    CHECK_GE(in_type->size(), 1U);
    int dtype = (*in_type)[0];
    CHECK_NE(dtype, -1) << "First input must have specified type";
    for (std::size_t i = 0; i < in_type->size(); ++i) {
      if ((*in_type)[i] == -1) {
        (*in_type)[i] = dtype;
      } else {
        UNIFORM_TYPE_CHECK((*in_type)[i], dtype, ListArguments()[i]);
      }
    }
    out_type->clear();
    out_type->push_back(dtype);
    return true;
  }

  OperatorProperty* Copy() const override {
    auto ptr = new ModulatedDeformableConvolutionProp();
    ptr->param_ = param_;
    return ptr;
  }

  std::string TypeString() const override {
    return "_contrib_ModulatedDeformableConvolution";
  }

  std::vector<int> DeclareBackwardDependency(
    const std::vector<int> &out_grad,
    const std::vector<int> &in_data,
    const std::vector<int> &out_data) const override {
    return{ out_grad[dmconv::kOut], in_data[dmconv::kData],
            in_data[dmconv::kOffset], in_data[dmconv::kMask],
            in_data[dmconv::kWeight] };
  }

  std::vector<ResourceRequest> ForwardResource(
    const std::vector<mxnet::TShape> &in_shape) const override {
    return{ ResourceRequest::kTempSpace };
  }

  std::vector<ResourceRequest> BackwardResource(
    const std::vector<mxnet::TShape> &in_shape) const override {
    return{ ResourceRequest::kTempSpace };
  }

  Operator* CreateOperator(Context ctx) const override {
    LOG(FATAL) << "Not Implemented.";
    return nullptr;
  }

  Operator* CreateOperatorEx(Context ctx, std::vector<mxnet::TShape> *in_shape,
    std::vector<int> *in_type) const override;

 private:
  ModulatedDeformableConvolutionParam param_;
};  // class ConvolutionProp
#endif  // DMLC_USE_CXX11
}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_CONTRIB_MODULATED_DEFORMABLE_CONVOLUTION_INL_H_
