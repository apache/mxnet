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
 * Copyright (c) 2015 by Contributors
 * \file deconvolution-inl.h
 * \brief
 * \author Wei Wu
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


namespace mxnet {
namespace op {

namespace deconv {
  enum DeconvolutionOpInputs {kData, kWeight, kBias};
  enum DeconvolutionOpOutputs {kOut};
  enum DeconvolutionOpResource {kTempSpace};
  enum DeconvolutionOpCudnnTune {kOff, kLimited, kFastest};
}

struct DeconvolutionParam : public dmlc::Parameter<DeconvolutionParam> {
  TShape kernel;
  TShape stride;
  TShape dilate;
  TShape pad;
  TShape adj;
  TShape target_shape;
  uint32_t num_filter;
  uint32_t num_group;
  uint64_t workspace;
  bool no_bias;
  dmlc::optional<int> cudnn_tune;
  bool cudnn_off;
  dmlc::optional<int> layout;
  DMLC_DECLARE_PARAMETER(DeconvolutionParam) {
    DMLC_DECLARE_FIELD(kernel).describe("Deconvolution kernel size: (w,), (h, w) or (d, h, w). "
                  "This is same as the kernel size used for the corresponding convolution");
    DMLC_DECLARE_FIELD(stride).set_default(TShape())
        .describe("The stride used for the corresponding convolution: (w,), (h, w) or (d, h, w). "
                  "Defaults to 1 for each dimension.");
    DMLC_DECLARE_FIELD(dilate).set_default(TShape())
        .describe("Dilation factor for each dimension of the input: (w,), (h, w) or (d, h, w). "
                  "Defaults to 1 for each dimension.");
    DMLC_DECLARE_FIELD(pad).set_default(TShape())
        .describe("The amount of implicit zero padding added during convolution for each "
                  "dimension of the input: "
                  "(w,), (h, w) or (d, h, w). "
                  "``(kernel-1)/2`` is usually a good choice. "
                  "If `target_shape` is set, "
                  "`pad` will be ignored and a padding that will generate the target shape "
                  "will be used. Defaults to no padding.");
    DMLC_DECLARE_FIELD(adj).set_default(TShape())
        .describe("Adjustment for output shape: (w,), (h, w) or (d, h, w). "
                  "If `target_shape` is set, "
                  "`adj` will be ignored and computed accordingly.");
    DMLC_DECLARE_FIELD(target_shape).set_default(TShape())
        .describe("Shape of the output tensor: (w,), (h, w) or (d, h, w).");
    DMLC_DECLARE_FIELD(num_filter).set_range(1, 100000)
        .describe("Number of output filters.");
    DMLC_DECLARE_FIELD(num_group).set_default(1)
        .describe("Number of groups partition.");
    DMLC_DECLARE_FIELD(workspace).set_default(512).set_range(0, 8192)
      .describe("Maximum temporal workspace allowed for deconvolution (MB).");
    DMLC_DECLARE_FIELD(no_bias).set_default(true)
        .describe("Whether to disable bias parameter.");
    DMLC_DECLARE_FIELD(cudnn_tune)
      .add_enum("off", deconv::kOff)
      .add_enum("limited_workspace", deconv::kLimited)
      .add_enum("fastest", deconv::kFastest)
      .set_default(dmlc::optional<int>())
      .describe("Whether to pick convolution algorithm by running performance test.");
    DMLC_DECLARE_FIELD(cudnn_off).set_default(false)
    .describe("Turn off cudnn for this layer.");
    DMLC_DECLARE_FIELD(layout)
      .add_enum("NCW", mshadow::kNCW)
      .add_enum("NCHW", mshadow::kNCHW)
      .add_enum("NCDHW", mshadow::kNCDHW)
      .add_enum("NHWC", mshadow::kNHWC)
      .add_enum("NDHWC", mshadow::kNDHWC)
      .set_default(dmlc::optional<int>())
      .describe("Set layout for input, output and weight. Empty for "
                "default layout, NCW for 1d, NCHW for 2d and NCDHW for 3d.");
  }

  template<size_t ndim>
  void InferPad(TShape input, index_t (&o_pad)[ndim], index_t (&o_adj)[ndim] ) const {
    // Modified by Li.bs
    // Use tag to control the calculation of pad
    bool bCal = false;
    if (target_shape.ndim() != 0) {
      for (index_t i = 0; i < target_shape.ndim(); i++) {
        if (target_shape[i] != 0) bCal = true;
      }
    }

    if (bCal) {
      size_t input_ndim = input.ndim();

      for (index_t i = 0; i < ndim; i++) {
        // input.ndim() can be larger than ndim, in case that the complete input
        // shape was passed and not only the ndim last ones
        o_pad[i] = stride[i] * (input[(input_ndim - ndim) + i] - 1) + DilatedKernelSize(i);
        CHECK_GE(o_pad[i], target_shape[i]) << "too big target shape";
        o_pad[i] -= target_shape[i];
        o_adj[i] = o_pad[i] % 2;
        o_pad[i] = (o_pad[i] + 1) / 2;
      }
    } else {
      for (index_t i = 0; i < ndim; i++) {
        o_pad[i] = pad[i];
        o_adj[i] = adj[i];
      }
    }
  }

  // Adjusts kernel size for effects of dilation in the dimension `dim`.
  index_t DilatedKernelSize(int dim) const {
    return 1 + (kernel[dim] - 1) * dilate[dim];
  }

  bool operator==(const DeconvolutionParam& other) const {
    return this->kernel == other.kernel &&
           this->stride == other.stride &&
           this->dilate == other.dilate &&
           this->pad == other.pad &&
           this->adj == other.adj &&
           this->target_shape == other.target_shape &&
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
struct hash<mxnet::op::DeconvolutionParam> {
  size_t operator()(const mxnet::op::DeconvolutionParam& val) {
    size_t ret = 0;
    ret = dmlc::HashCombine(ret, val.kernel);
    ret = dmlc::HashCombine(ret, val.stride);
    ret = dmlc::HashCombine(ret, val.dilate);
    ret = dmlc::HashCombine(ret, val.pad);
    ret = dmlc::HashCombine(ret, val.adj);
    ret = dmlc::HashCombine(ret, val.target_shape);
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
class DeconvolutionOp : public Operator {
 public:
  explicit DeconvolutionOp(DeconvolutionParam p) {
    this->param_ = p;
    // convert MBytes first to Bytes and then to elements.
    param_.workspace = (param_.workspace << 20) / sizeof(real_t);
  }

  virtual void Forward(const OpContext &ctx,
                       const std::vector<TBlob> &in_data,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &out_data,
                       const std::vector<TBlob> &aux_args) {
    using namespace mshadow;
    using namespace mshadow::expr;

    if (param_.kernel.ndim() > 2) {
      LOG(FATAL) << "If not using CUDNN, only 1D or 2D Deconvolution is supported";
    }

    CHECK_EQ(req[deconv::kOut], kWriteTo);
    size_t expected = param_.no_bias ? 2 : 3;
    CHECK_EQ(in_data.size(), expected);
    CHECK_EQ(out_data.size(), 1U);
    Stream<xpu> *s = ctx.get_stream<xpu>();
    auto in_data_shape = in_data[deconv::kData].shape_;
    Tensor<xpu, 4, DType> data = TBlobTo4DTensor(in_data[deconv::kData], s);
    Tensor<xpu, 4, DType> out = TBlobTo4DTensor(out_data[deconv::kOut], s);
    index_t o_pad[2], o_adj[2];
    if (param_.kernel.ndim() == 2) {
      param_.InferPad(TShape({in_data_shape[2], in_data_shape[3]}), o_pad, o_adj);
    } else {
      index_t o_pad_1D[1], o_adj_1D[1];
      param_.InferPad({in_data_shape[2]}, o_pad_1D, o_adj_1D);
      o_pad[0] = 0;
      o_pad[1] = o_pad_1D[0];
      o_adj[0] = 0;
      o_adj[1] = o_adj_1D[0];
    }
    auto stride = param_.kernel.ndim() == 2 ? param_.stride : TShape({1, param_.stride[0]});
    auto dilate = param_.kernel.ndim() == 2 ? param_.dilate : TShape({1, param_.dilate[0]});
    auto kernel = param_.kernel.ndim() == 2 ? param_.kernel : TShape({1, param_.kernel[0]});
    auto kernel_size = kernel.Size();

    Shape<3> wmat_shape =
        Shape3(param_.num_group,
               data.shape_[1] / param_.num_group,
               param_.num_filter / param_.num_group * kernel_size);
    Tensor<xpu, 3, DType> wmat =
        in_data[deconv::kWeight].get_with_shape<xpu, 3, DType>(wmat_shape, s);
#if defined(__CUDACC__)
    CHECK_EQ(s->blas_handle_ownership_, Stream<xpu>::OwnHandle)
        << "Must init CuBLAS handle in stream";
#endif
    const index_t nbatch = data.size(0);
    Tensor<xpu, 1, DType> workspace =
        ctx.requested[deconv::kTempSpace].get_space_typed<xpu, 1, DType>(
            Shape1(this->InitTemp(out.shape_, data.shape_)), s);
    for (index_t i = 0; i < nbatch; i += nstep_) {
      const index_t step = std::min(nstep_, nbatch - i);
      Tensor<xpu, 2, DType> temp_col = Tensor<xpu, 2, DType>(
                                            workspace.dptr_,
                                            Shape2(shape_colunit_[0],
                                            shape_colunit_[1] * step), s);
      Tensor<xpu, 3, DType> temp_dst = Tensor<xpu, 3, DType>(
                                           workspace.dptr_ + temp_col.shape_.Size(),
                                           Shape3(shape_dstunit_[0],
                                           shape_dstunit_[1],
                                           shape_dstunit_[2] * step), s);
      temp_dst = reshape(swapaxis<1, 0>(data.Slice(i, i + step)), temp_dst.shape_);
      if (o_pad[0] == 0 && o_pad[1] == 0) {
        temp_col = unpack_patch2col(out.Slice(i, i + step),
                                    kernel[0],
                                    kernel[1],
                                    stride[0],
                                    stride[1],
                                    dilate[0],
                                    dilate[1]);
      } else {
        temp_col = unpack_patch2col(pad(out.Slice(i, i + step),
                                        o_pad[0], o_pad[1]),
                                    kernel[0],
                                    kernel[1],
                                    stride[0],
                                    stride[1],
                                    dilate[0],
                                    dilate[1]);
      }
      const index_t gstride = temp_col.size(0) / param_.num_group;
      for (uint32_t gid = 0; gid < param_.num_group; ++gid) {
        mshadow::Tensor<xpu, 2, DType> tmpc = temp_col.Slice(gstride * gid,
                                              gstride * (gid + 1));
        // Legacy approach shown here for comparison:
        //   tmpc = dot(wmat[gid].T(), temp_dst[gid]);
        linalg_gemm(wmat[gid], temp_dst[gid], tmpc, true, false, s);
      }
      if (o_pad[0] == 0 && o_pad[1] == 0) {
        out.Slice(i, i + step) = pack_col2patch(temp_col,
                                   out.Slice(i, i + step).shape_,
                                   kernel[0],
                                   kernel[1],
                                   stride[0],
                                   stride[1],
                                   dilate[0],
                                   dilate[1]);
      } else {
        Shape<4> pshape = out.Slice(i, i + step).shape_;
        pshape[2] += 2 * o_pad[0];
        pshape[3] += 2 * o_pad[1];
        out.Slice(i, i + step) = crop(pack_col2patch(temp_col,
                                        pshape,
                                        kernel[0],
                                        kernel[1],
                                        stride[0],
                                        stride[1],
                                        dilate[0],
                                        dilate[1]),
                                        out[i][0].shape_);
      }
    }
    if (!param_.no_bias) {
      // add bias, broadcast bias to dim 1: channel
      Tensor<xpu, 1, DType> bias = in_data[deconv::kBias].get<xpu, 1, DType>(s);
      out += mshadow::expr::broadcast<1>(bias, out.shape_);
    }
  }

  virtual void Backward(const OpContext &ctx,
                        const std::vector<TBlob> &out_grad,
                        const std::vector<TBlob> &in_data,
                        const std::vector<TBlob> &out_data,
                        const std::vector<OpReqType> &req,
                        const std::vector<TBlob> &in_grad,
                        const std::vector<TBlob> &aux_args) {
    using namespace mshadow;
    using namespace mshadow::expr;
    // TODO(bing): check the BLAS Handle, be careful
    CHECK_EQ(out_grad.size(), 1U);
    size_t expected = param_.no_bias == 0 ? 3 : 2;
    CHECK(in_data.size() == expected && in_grad.size() == expected);
    CHECK_EQ(req.size(), expected);
    CHECK_EQ(in_data[deconv::kWeight].CheckContiguous(), true);
    // get data
    Stream<xpu> *s = ctx.get_stream<xpu>();
    auto in_data_shape = in_data[deconv::kData].shape_;
    Tensor<xpu, 4, DType> data = TBlobTo4DTensor(in_data[deconv::kData], s);
    Tensor<xpu, 4, DType> grad = TBlobTo4DTensor(out_grad[deconv::kOut], s);
    Tensor<xpu, 4, DType> gdata = TBlobTo4DTensor(in_grad[deconv::kData], s);

    index_t o_pad[2], o_adj[2];
    if (param_.kernel.ndim() == 2) {
      param_.InferPad(TShape({in_data_shape[2], in_data_shape[3]}), o_pad, o_adj);
    } else {
      index_t o_pad_1D[1], o_adj_1D[1];
      param_.InferPad({in_data_shape[2]}, o_pad_1D, o_adj_1D);
      o_pad[0] = 0;
      o_pad[1] = o_pad_1D[0];
      o_adj[0] = 0;
      o_adj[1] = o_adj_1D[0];
    }
    auto stride = param_.kernel.ndim() == 2 ? param_.stride : TShape({1, param_.stride[0]});
    auto dilate = param_.kernel.ndim() == 2 ? param_.dilate : TShape({1, param_.dilate[0]});
    auto kernel = param_.kernel.ndim() == 2 ? param_.kernel : TShape({1, param_.kernel[0]});
    auto kernel_size = kernel.Size();

    Shape<3> wmat_shape =
        Shape3(param_.num_group,
               data.shape_[1] / param_.num_group,
               param_.num_filter / param_.num_group * kernel_size);
    Tensor<xpu, 3, DType> wmat =
        in_data[deconv::kWeight].get_with_shape<xpu, 3, DType>(wmat_shape, s);
    Tensor<xpu, 3, DType> gwmat =
        in_grad[deconv::kWeight].get_with_shape<xpu, 3, DType>(wmat_shape, s);
#if defined(__CUDACC__)
    CHECK_EQ(s->blas_handle_ownership_, Stream<xpu>::OwnHandle)
        << "Must init CuBLAS handle in stream";
#endif

    const index_t nbatch = data.size(0);
    Tensor<xpu, 1, DType> workspace =
        ctx.requested[deconv::kTempSpace].get_space_typed<xpu, 1, DType>(
            Shape1(this->InitTemp(grad.shape_, data.shape_)), s);
    for (index_t i = 0; i < nbatch; i += nstep_) {
      const index_t step = std::min(nstep_, nbatch - i);
      Tensor<xpu, 2, DType> temp_col = Tensor<xpu, 2, DType>(
                                           workspace.dptr_,
                                           Shape2(shape_colunit_[0],
                                           shape_colunit_[1] * step), s);
      Tensor<xpu, 3, DType> temp_dst = Tensor<xpu, 3, DType>(
                                           workspace.dptr_ + temp_col.shape_.Size(),
                                           Shape3(shape_dstunit_[0],
                                           shape_dstunit_[1],
                                           shape_dstunit_[2] * step), s);
      temp_dst = reshape(swapaxis<1, 0>(data.Slice(i, i + step)), temp_dst.shape_);
      if (o_pad[0] == 0 && o_pad[1] == 0) {
        temp_col = unpack_patch2col(grad.Slice(i, i + step),
                                     kernel[0],
                                     kernel[1],
                                     stride[0],
                                     stride[1],
                                     dilate[0],
                                     dilate[1]);
      } else {
        temp_col = unpack_patch2col(pad(grad.Slice(i, i + step), o_pad[0], o_pad[1]),
                                     kernel[0],
                                     kernel[1],
                                     stride[0],
                                     stride[1],
                                     dilate[0],
                                     dilate[1]);
      }
      const index_t gstride = temp_col.size(0) / param_.num_group;
      for (uint32_t gid = 0; gid < param_.num_group; ++gid) {
        Tensor<xpu, 2, DType> tmpc = temp_col.Slice(gstride * gid, gstride * (gid + 1));
        if (i == 0) {
          Tensor<xpu, 2, DType> tmp_gwmat = gwmat[gid];
          // Legacy approach shown here for comparison:
          //   Assign(tmp_gwmat, req[deconv::kWeight], dot(temp_dst[gid], tmpc.T()));
          linalg_gemm(temp_dst[gid], tmpc, tmp_gwmat, false, true, s, req[deconv::kWeight]);
        } else {
          // Legacy approach shown here for comparison:
          //   gwmat[gid] += dot(temp_dst[gid], tmpc.T());
          linalg_gemm(temp_dst[gid], tmpc, gwmat[gid], false, true, s, kAddTo);
        }
      }
      if (req[deconv::kData] == kWriteTo ||
          req[deconv::kData] == kWriteInplace ||
          req[deconv::kData] == kAddTo) {
        for (uint32_t gid = 0; gid < param_.num_group; ++gid) {
          Tensor<xpu, 2, DType> tmpc = temp_col.Slice(gstride * gid, gstride * (gid + 1));
          // Legacy approach shown here for comparison:
          //   temp_dst[gid] = dot(wmat[gid], tmpc);
          linalg_gemm(wmat[gid], tmpc, temp_dst[gid], false, false, s);
        }
        Assign(gdata.Slice(i, i + step),
               req[deconv::kData],
               (swapaxis<1, 0>(reshape(temp_dst,
                                      mshadow::Shape4(gdata.shape_[1],
                                                      step,
                                                      gdata.size(2),
                                                      gdata.size(3))))));
      }
    }
    if (!param_.no_bias) {
      Tensor<xpu, 1, DType> gbias = in_grad[deconv::kBias].get<xpu, 1, DType>(s);
      Assign(gbias, req[deconv::kBias], sumall_except_dim<1>(grad));
    }
  }

 private:
  inline index_t InitTemp(const mshadow::Shape<4> &ishape,
                          const mshadow::Shape<4> &oshape) {
    const int ksize = param_.kernel.Size();
    shape_colunit_ = mshadow::Shape2(ishape[1] * ksize,
                                     oshape[2] * oshape[3]);
    shape_dstunit_ = mshadow::Shape3(param_.num_group,
                                     oshape[1] / param_.num_group,
                                     oshape[2] * oshape[3]);
    // See convolution for workspace calculations
    nstep_ = std::max(
        std::min(
            static_cast<index_t>(
                param_.workspace / (shape_colunit_.Size() + shape_dstunit_.Size())),
            ishape[0]),
        1U);

    mshadow::Shape<2> scol = mshadow::Shape2(shape_colunit_[0],
                                             shape_colunit_[1] * nstep_);
    mshadow::Shape<3> sdst = mshadow::Shape3(shape_dstunit_[0],
                                             shape_dstunit_[1],
                                             shape_dstunit_[2] * nstep_);
    index_t required_size = scol.Size() + sdst.Size();
    CHECK_GE(param_.workspace, required_size)
      << "\nMinimum workspace size: " << required_size * sizeof(DType) << " Bytes\n"
      << "Given: " << param_.workspace * sizeof(DType);
    return required_size;
  }

  inline Tensor<xpu, 4, DType> TBlobTo4DTensor(const TBlob &tb, Stream<xpu> *s) {
    using namespace mshadow;
    if (param_.kernel.ndim() == 2)
      return tb.get<xpu, 4, DType>(s);
    else
      return tb.get_with_shape<xpu, 4, DType>(
          Shape4(tb.shape_[0], tb.shape_[1], 1, tb.shape_[2]), s);
  }

  DeconvolutionParam param_;
  mshadow::Shape<2> shape_colunit_;
  mshadow::Shape<3> shape_dstunit_;
  index_t nstep_;
};  // class DeconvolutionOp

template<typename xpu>
Operator* CreateOp(DeconvolutionParam param, int dtype,
                   std::vector<TShape> *in_shape,
                   std::vector<TShape> *out_shape,
                   Context ctx);

#if DMLC_USE_CXX11
class DeconvolutionProp : public OperatorProperty {
 public:
  std::vector<std::string> ListArguments() const override {
    if (!param_.no_bias) {
      return {"data", "weight", "bias"};
    } else {
      return {"data", "weight"};
    }
  }

  void Init(const std::vector<std::pair<std::string, std::string> >& kwargs) override {
    using namespace mshadow;
    param_.Init(kwargs);
    if (param_.kernel.ndim() == 1) {
      param_.layout = param_.layout? param_.layout.value() : mshadow::kNCW;
      if (param_.stride.ndim() == 0) param_.stride = Shape1(1);
      if (param_.dilate.ndim() == 0) param_.dilate = Shape1(1);
      if (param_.pad.ndim() == 0) param_.pad = Shape1(0);
      if (param_.adj.ndim() == 0) param_.adj = Shape1(0);
    } else if (param_.kernel.ndim() == 2) {
      param_.layout = param_.layout ? param_.layout.value() : mshadow::kNCHW;
      if (param_.stride.ndim() == 0) param_.stride = Shape2(1, 1);
      if (param_.dilate.ndim() == 0) param_.dilate = Shape2(1, 1);
      if (param_.pad.ndim() == 0) param_.pad = Shape2(0, 0);
      if (param_.adj.ndim() == 0) param_.adj = Shape2(0, 0);
    } else {
      CHECK_EQ(param_.kernel.ndim(), 3U) << param_.kernel.ndim() << "D deconvolution not supported";
      param_.layout = param_.layout ? param_.layout.value(): mshadow::kNCDHW;
      if (param_.stride.ndim() == 0) param_.stride = Shape3(1, 1, 1);
      if (param_.dilate.ndim() == 0) param_.dilate = Shape3(1, 1, 1);
      if (param_.pad.ndim() == 0) param_.pad = Shape3(0, 0, 0);
      if (param_.adj.ndim() == 0) param_.adj = Shape3(0, 0, 0);
    }
  }

  std::map<std::string, std::string> GetParams() const override {
    return param_.__DICT__();
  }

  bool InferShape(std::vector<TShape> *in_shape,
                  std::vector<TShape> *out_shape,
                  std::vector<TShape> *aux_shape) const override {
#if MXNET_USE_CUDNN == 0
    if (param_.kernel.ndim() > 2) {
      LOG(FATAL) << "If not using CUDNN, only 1D or 2D Deconvolution is supported";
      return false;
    }
#endif  // CUDNN

    using namespace mshadow;
    if (!param_.no_bias) {
      CHECK_EQ(in_shape->size(), 3U) << "Input:[data, weight, bias]";
    } else {
      CHECK_EQ(in_shape->size(), 2U) << "Input:[data, weight]";
    }
    out_shape->resize(1, TShape());
    const TShape &dshape = (*in_shape)[deconv::kData];
    if (dshape.ndim() ==  0) return false;

    if (param_.kernel.ndim() == 1) {
      // 1d conv
      CHECK_EQ(dshape.ndim(), 3U) << "Input data should be 3D in batch-num_filter-x";
      Shape<3> dshape_ncw = ConvertLayout(dshape.get<3>(), param_.layout.value(), kNCW);
      Shape<3> wshape = Shape3(dshape_ncw[1], param_.num_filter / param_.num_group,
                               param_.kernel[0]);
      wshape = ConvertLayout(wshape, kNCW, param_.layout.value());
      SHAPE_ASSIGN_CHECK(*in_shape, deconv::kWeight, wshape);
      if (!param_.no_bias) {
        SHAPE_ASSIGN_CHECK(*in_shape, deconv::kBias, Shape1(param_.num_filter));
      }

      const index_t dilated_ksize_x = param_.DilatedKernelSize(0);

      index_t o_pad[1];
      index_t o_adj[1];
      param_.InferPad(dshape_ncw, o_pad, o_adj);

      CHECK_EQ(dshape_ncw[1] % param_.num_group, 0U) \
        << "input num_filter must divide group size";
      CHECK_EQ(param_.num_filter % param_.num_group, 0U) \
        << "output num_filter must divide group size";
      CHECK_GT(param_.kernel.Size(), 0U) \
        << "incorrect kernel size: " << param_.kernel;
      CHECK_GT(param_.stride.Size(), 0U) \
        << "incorrect stride size: " << param_.stride;
      CHECK_GT(param_.dilate.Size(), 0U) \
        << "incorrect dilate size: " << param_.dilate;

      CHECK_GE(param_.stride[0]-1, o_adj[0]) << "adj(x) must be samller than stride[0]";

      Shape<3> oshape;
      oshape[0] = dshape_ncw[0];
      oshape[1] = param_.num_filter;
      oshape[2] = param_.stride[0] * (dshape_ncw[2] - 1) +
        dilated_ksize_x - 2 * o_pad[0] + o_adj[0];

      if (param_.target_shape[0] > 0) {
        CHECK_EQ(param_.target_shape[0], oshape[2]) \
          << "param_.target_shape[0] was not reasonable, please set it carefully";
      }

      SHAPE_ASSIGN_CHECK(*out_shape, 0, ConvertLayout(oshape, kNCW, param_.layout.value()));

      return true;
    } else if (param_.kernel.ndim() == 2) {
      // 2d conv
      CHECK_EQ(dshape.ndim(), 4U) \
        << "Input data should be 4D in batch-num_filter-y-x";
      Shape<4> dshape_nchw = ConvertLayout(dshape.get<4>(), param_.layout.value(), kNCHW);
      Shape<4> wshape = Shape4(dshape_nchw[1],
                               param_.num_filter / param_.num_group,
                               param_.kernel[0], param_.kernel[1]);
      wshape = ConvertLayout(wshape, kNCHW, param_.layout.value());
      SHAPE_ASSIGN_CHECK(*in_shape, deconv::kWeight, wshape);
      if (!param_.no_bias) {
        SHAPE_ASSIGN_CHECK(*in_shape, deconv::kBias, Shape1(param_.num_filter));
      }

      const index_t dilated_ksize_y = param_.DilatedKernelSize(0);
      const index_t dilated_ksize_x = param_.DilatedKernelSize(1);

      index_t o_pad[2];
      index_t o_adj[2];
      param_.InferPad(dshape_nchw, o_pad, o_adj);

      CHECK_EQ(dshape_nchw[1] % param_.num_group, 0U) \
        << "input num_filter must divide group size";
      CHECK_EQ(param_.num_filter % param_.num_group, 0U) \
        << "output num_filter must divide group size";
      CHECK_GT(param_.kernel.Size(), 0U) \
        << "incorrect kernel size: " << param_.kernel;
      CHECK_GT(param_.stride.Size(), 0U) \
        << "incorrect stride size: " << param_.stride;
      CHECK_GT(param_.dilate.Size(), 0U) \
          << "incorrect dilate size: " << param_.dilate;

      CHECK_GE(param_.stride[0]-1, o_adj[0]) << "adj(y) must be samller than stride[0]";
      CHECK_GE(param_.stride[1]-1, o_adj[1]) << "adj(x) must be samller than stride[1]";

      Shape<4> oshape;
      oshape[0] = dshape_nchw[0];
      oshape[1] = param_.num_filter;
      oshape[2] = param_.stride[0] * (dshape_nchw[2] - 1) +
        dilated_ksize_y - 2 * o_pad[0] + o_adj[0];
      oshape[3] = param_.stride[1] * (dshape_nchw[3] - 1) +
        dilated_ksize_x - 2 * o_pad[1] + o_adj[1];

      if (param_.target_shape[0] > 0) {
        CHECK_EQ(param_.target_shape[0], oshape[2]) \
          << "param_.target_shape[0] was not reasonable, please set it carefully";
      }
      if (param_.target_shape[1] > 0) {
        CHECK_EQ(param_.target_shape[1], oshape[3]) \
          << "param_.target_shape[1] was not reasonable, please set it carefully";
      }

      SHAPE_ASSIGN_CHECK(*out_shape, 0, ConvertLayout(oshape, kNCHW, param_.layout.value()));

      return true;
    } else if (param_.kernel.ndim() == 3) {
      // 3d conv
      CHECK_EQ(dshape.ndim(), 5U) \
        << "Input data should be 5D in batch-num_filter-depth-y-x";
      Shape<5> dshape_ncdhw = ConvertLayout(dshape.get<5>(), param_.layout.value(), kNCDHW);
      Shape<5> wshape = Shape5(dshape_ncdhw[1], param_.num_filter / param_.num_group,
                               param_.kernel[0], param_.kernel[1], param_.kernel[2]);
      wshape = ConvertLayout(wshape, kNCDHW, param_.layout.value());
      SHAPE_ASSIGN_CHECK(*in_shape, deconv::kWeight, wshape);
      if (!param_.no_bias) {
        SHAPE_ASSIGN_CHECK(*in_shape, deconv::kBias, Shape1(param_.num_filter));
      }

      // Note: 3D dilation currently not supported.
      // Calculations below done to preserve symmetry with 1D/2D code.
      const index_t dilated_ksize_d = param_.DilatedKernelSize(0);
      const index_t dilated_ksize_y = param_.DilatedKernelSize(1);
      const index_t dilated_ksize_x = param_.DilatedKernelSize(2);

      index_t o_pad[3];
      index_t o_adj[3];
      param_.InferPad(dshape_ncdhw, o_pad, o_adj);

      CHECK_EQ(dshape_ncdhw[1] % param_.num_group, 0U) \
        << "input num_filter must divide group size";
      CHECK_EQ(param_.num_filter % param_.num_group, 0U) \
        << "output num_filter must divide group size";
      CHECK_GT(param_.kernel.Size(), 0U) \
        << "incorrect kernel size: " << param_.kernel;
      CHECK_GT(param_.stride.Size(), 0U) \
        << "incorrect stride size: " << param_.stride;
      CHECK_GT(param_.dilate.Size(), 0U) \
        << "incorrect dilate size: " << param_.dilate;
      CHECK_EQ(param_.dilate.Size(), 1U)
        << "Dilate is not supported in 3d deconvolution";

      CHECK_GE(param_.stride[0]-1, o_adj[0]) << "adj(d) must be samller than stride[0]";
      CHECK_GE(param_.stride[1]-1, o_adj[1]) << "adj(y) must be samller than stride[1]";
      CHECK_GE(param_.stride[2]-1, o_adj[2]) << "adj(x) must be samller than stride[2]";

      Shape<5> oshape;
      oshape[0] = dshape_ncdhw[0];
      oshape[1] = param_.num_filter;
      oshape[2] = param_.stride[0] * (dshape_ncdhw[2] - 1) +
        dilated_ksize_d - 2 * o_pad[0] + o_adj[0];
      oshape[3] = param_.stride[1] * (dshape_ncdhw[3] - 1) +
        dilated_ksize_y - 2 * o_pad[1] + o_adj[1];
      oshape[4] = param_.stride[2] * (dshape_ncdhw[4] - 1) +
        dilated_ksize_x - 2 * o_pad[2] + o_adj[2];

      if (param_.target_shape[0] > 0) {
        CHECK_EQ(param_.target_shape[0], oshape[2]) \
          << "param_.target_shape[0] was not reasonable, please it carefully";
      }
      if (param_.target_shape[1] > 0) {
        CHECK_EQ(param_.target_shape[1], oshape[3]) \
          << "param_.target_shape[1] was not reasonable, please set it carefully";
      }
      if (param_.target_shape[2] > 0) {
        CHECK_EQ(param_.target_shape[2], oshape[4]) \
          << "param_.target_shape[2] was not reasonable, please set it carefully";
      }

      SHAPE_ASSIGN_CHECK(*out_shape, 0, ConvertLayout(oshape, kNCDHW, param_.layout.value()));

      return true;
    } else {
      LOG(FATAL) << "Unknown convolution type";
      return false;
    }
  }

  bool InferType(std::vector<int> *in_type,
                 std::vector<int> *out_type,
                 std::vector<int> *aux_type) const override {
    CHECK_GE(in_type->size(), 1U);
    int dtype = (*in_type)[0];
    CHECK_NE(dtype, -1) << "First input must have specified type";
    for (index_t i = 0; i < in_type->size(); ++i) {
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
    auto ptr = new DeconvolutionProp();
    ptr->param_ = param_;
    return ptr;
  }

  std::string TypeString() const override {
    return "Deconvolution";
  }

  std::vector<int> DeclareBackwardDependency(
    const std::vector<int> &out_grad,
    const std::vector<int> &in_data,
    const std::vector<int> &out_data) const override {
    return {out_grad[deconv::kOut], in_data[deconv::kData], in_data[deconv::kWeight]};
  }

  std::vector<ResourceRequest> ForwardResource(
      const std::vector<TShape> &in_shape) const override {
    return {ResourceRequest::kTempSpace};
  }

  std::vector<ResourceRequest> BackwardResource(
      const std::vector<TShape> &in_shape) const override {
    return {ResourceRequest::kTempSpace};
  }

  Operator* CreateOperator(Context ctx) const override {
    LOG(FATAL) << "Not Implemented";
    return NULL;
  }

  Operator* CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
                             std::vector<int> *in_type) const override;

 private:
  DeconvolutionParam param_;
};  // class DeconvolutionProp
#endif  // DMLC_USE_CXX11
}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_NN_DECONVOLUTION_INL_H_
