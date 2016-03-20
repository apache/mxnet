/*!
 * Copyright (c) 2015 by Contributors
 * \file convolution-inl.h
 * \brief
 * \author Bing Xu
*/
#ifndef MXNET_OPERATOR_CONVOLUTION_INL_H_
#define MXNET_OPERATOR_CONVOLUTION_INL_H_

#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <mxnet/operator.h>
#include <algorithm>
#include <map>
#include <vector>
#include <string>
#include <utility>
#include "./operator_common.h"


namespace mxnet {
namespace op {

namespace conv {
enum ConvolutionOpInputs {kData, kWeight, kBias};
enum ConvolutionOpOutputs {kOut};
enum ConvolutionOpResource {kTempSpace};
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
  DMLC_DECLARE_PARAMETER(ConvolutionParam) {
    int shape[] = {1, 1};
    DMLC_DECLARE_FIELD(kernel).describe("convolution kernel size: (y, x)");
    DMLC_DECLARE_FIELD(stride).set_default(TShape(shape, shape + 2))
    .describe("convolution stride: (y, x)");
    DMLC_DECLARE_FIELD(dilate).set_default(TShape(shape, shape + 2))
    .describe("convolution dilate: (y, x)");
    shape[0] = shape[1] = 0;
    DMLC_DECLARE_FIELD(pad).set_default(TShape(shape, shape + 2))
    .describe("pad for convolution: (y, x)");
    DMLC_DECLARE_FIELD(num_filter).set_range(1, 100000)
    .describe("convolution filter(channel) number");
    DMLC_DECLARE_FIELD(num_group).set_default(1)
    .describe("Number of groups partition. "
              "This option is not supported by CuDNN, you can use SliceChannel to num_group,"
              "apply convolution and concat instead to achieve the same need.");
    DMLC_DECLARE_FIELD(workspace).set_default(512).set_range(0, 4096)
    .describe("Tmp workspace for convolution (MB).");
    DMLC_DECLARE_FIELD(no_bias).set_default(false)
    .describe("Whether to disable bias parameter.");
  }
};

template<typename xpu>
class ConvolutionOp : public Operator {
 public:
  explicit ConvolutionOp(ConvolutionParam p) {
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
    CHECK_EQ(req[conv::kOut], kWriteTo);
    size_t expected = param_.no_bias ? 2 : 3;
    CHECK_EQ(in_data.size(), expected);
    CHECK_EQ(out_data.size(), 1);
    Stream<xpu> *s = ctx.get_stream<xpu>();
    Tensor<xpu, 4> data = in_data[conv::kData].get<xpu, 4, real_t>(s);
    Shape<3> wmat_shape =
        Shape3(param_.num_group,
               param_.num_filter / param_.num_group,
               data.shape_[1] / param_.num_group * param_.kernel[0] * param_.kernel[1]);
    Tensor<xpu, 3> wmat = in_data[conv::kWeight].get_with_shape<xpu, 3, real_t>(wmat_shape, s);
    Tensor<xpu, 4> out = out_data[conv::kOut].get<xpu, 4, real_t>(s);
#if defined(__CUDACC__)
    CHECK_EQ(s->blas_handle_ownership_, Stream<xpu>::OwnHandle)
        << "Must init CuBLAS handle in stream";
#endif
    const index_t nbatch = data.size(0);
    Tensor<xpu, 1> workspace = ctx.requested[conv::kTempSpace].get_space<xpu>(
        Shape1(this->InitTemp(data.shape_, out.shape_)), s);
    for (index_t i = 0; i < nbatch; i += nstep_) {
      const index_t step = std::min(nstep_, nbatch - i);
      Tensor<xpu, 2> temp_col = Tensor<xpu, 2>(workspace.dptr_,
                                               Shape2(shape_colunit_[0],
                                                      shape_colunit_[1] * step), s);
      Tensor<xpu, 3> temp_dst = Tensor<xpu, 3>(workspace.dptr_ + temp_col.shape_.Size(),
                                               Shape3(shape_dstunit_[0],
                                                      shape_dstunit_[1],
                                                      shape_dstunit_[2] * step), s);
      if (param_.pad[0] == 0 && param_.pad[1] == 0) {
        temp_col = unpack_patch2col(data.Slice(i, i + step),
                                    param_.kernel[0],
                                    param_.kernel[1],
                                    param_.stride[0],
                                    param_.stride[1],
                                    param_.dilate[0],
                                    param_.dilate[1]);
      } else {
        temp_col = unpack_patch2col(pad(data.Slice(i, i + step),
                                        param_.pad[0], param_.pad[1]),
                                    param_.kernel[0],
                                    param_.kernel[1],
                                    param_.stride[0],
                                    param_.stride[1],
                                    param_.dilate[0],
                                    param_.dilate[1]);
      }
      const index_t gstride = temp_col.size(0) / param_.num_group;
      for (uint32_t gid = 0; gid < param_.num_group; ++gid) {
        mshadow::Tensor<xpu, 2> tmpc = temp_col.Slice(gstride * gid,
                                       gstride * (gid + 1));
        temp_dst[gid] = dot(wmat[gid], tmpc);
      }
      out.Slice(i, i + step) = swapaxis<1, 0>(reshape(temp_dst,
                                              mshadow::Shape4(param_.num_filter,
                                                  step,
                                                  out.size(2),
                                                  out.size(3))));
    }
    if (!param_.no_bias) {
      // add bias, broadcast bias to dim 1: channel
      Tensor<xpu, 1> bias = in_data[conv::kBias].get<xpu, 1, real_t>(s);
      out += broadcast<1>(bias, out.shape_);
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
    CHECK_EQ(out_grad.size(), 1);
    size_t expected = param_.no_bias == 0 ? 3 : 2;
    CHECK(in_data.size() == expected && in_grad.size() == expected);
    CHECK_EQ(req.size(), expected);
    CHECK_EQ(in_data[conv::kWeight].CheckContiguous(), true);
    // get data
    Stream<xpu> *s = ctx.get_stream<xpu>();
    Tensor<xpu, 4> data = in_data[conv::kData].get<xpu, 4, real_t>(s);
    Shape<3> wmat_shape =
        Shape3(param_.num_group,
               param_.num_filter / param_.num_group,
               data.shape_[1] / param_.num_group * param_.kernel[0] * param_.kernel[1]);
    Tensor<xpu, 3> wmat = in_data[conv::kWeight].get_with_shape<xpu, 3, real_t>(wmat_shape, s);
    Tensor<xpu, 4> grad = out_grad[conv::kOut].get<xpu, 4, real_t>(s);
    Tensor<xpu, 4> gdata = in_grad[conv::kData].get<xpu, 4, real_t>(s);
    Tensor<xpu, 3> gwmat = in_grad[conv::kWeight].get_with_shape<xpu, 3, real_t>(wmat_shape, s);
#if defined(__CUDACC__)
    CHECK_EQ(s->blas_handle_ownership_, Stream<xpu>::OwnHandle)
        << "Must init CuBLAS handle in stream";
#endif
    const index_t nbatch = data.size(0);
    Tensor<xpu, 1> workspace = ctx.requested[conv::kTempSpace].get_space<xpu>(
              Shape1(this->InitTemp(data.shape_, grad.shape_)), s);
    for (index_t i = 0; i < nbatch; i += nstep_) {
      const index_t step = std::min(nstep_, nbatch - i);
      Tensor<xpu, 2> temp_col = Tensor<xpu, 2>(workspace.dptr_,
                                               Shape2(shape_colunit_[0],
                                                      shape_colunit_[1] * step), s);
      Tensor<xpu, 3> temp_dst = Tensor<xpu, 3>(workspace.dptr_ + temp_col.shape_.Size(),
                                               Shape3(shape_dstunit_[0],
                                                      shape_dstunit_[1],
                                                      shape_dstunit_[2] * step), s);
      temp_dst = reshape(swapaxis<1, 0>(grad.Slice(i, i + step)), temp_dst.shape_);
      if (param_.pad[0] == 0 && param_.pad[1] == 0) {
        temp_col = unpack_patch2col(data.Slice(i, i + step),
                                     param_.kernel[0],
                                     param_.kernel[1],
                                     param_.stride[0],
                                     param_.stride[1],
                                     param_.dilate[0],
                                     param_.dilate[1]);
      } else {
        temp_col = unpack_patch2col(pad(data.Slice(i, i + step), param_.pad[0], param_.pad[1]),
                                     param_.kernel[0],
                                     param_.kernel[1],
                                     param_.stride[0],
                                     param_.stride[1],
                                     param_.dilate[0],
                                     param_.dilate[1]);
      }
      const index_t gstride = temp_col.size(0) / param_.num_group;
      for (uint32_t gid = 0; gid < param_.num_group; ++gid) {
        Tensor<xpu, 2> tmpc = temp_col.Slice(gstride * gid, gstride * (gid + 1));
        if (i == 0) {
          Tensor<xpu, 2> tmp_gwmat = gwmat[gid];
          Assign(tmp_gwmat, req[conv::kWeight], dot(temp_dst[gid], tmpc.T()));
        } else {
          gwmat[gid] += dot(temp_dst[gid], tmpc.T());
        }
      }
      if (req[conv::kData] == kWriteTo || req[conv::kData] == kWriteInplace) {
        for (uint32_t gid = 0; gid < param_.num_group; ++gid) {
          Tensor<xpu, 2> tmpc = temp_col.Slice(gstride * gid, gstride * (gid + 1));
          tmpc = dot(wmat[gid].T(), temp_dst[gid]);
        }
        if (param_.pad[0] == 0 && param_.pad[1] == 0) {
          gdata.Slice(i, i + step) = pack_col2patch(temp_col,
                                     data.Slice(i, i + step).shape_,
                                     param_.kernel[0],
                                     param_.kernel[1],
                                     param_.stride[0],
                                     param_.dilate[0]);
        } else {
          Shape<4> pshape = data.Slice(i, i + step).shape_;
          pshape[2] += 2 * param_.pad[0];
          pshape[3] += 2 * param_.pad[1];
          gdata.Slice(i, i + step) = crop(pack_col2patch(temp_col,
                                          pshape,
                                          param_.kernel[0],
                                          param_.kernel[1],
                                          param_.stride[0],
                                          param_.dilate[0]),
                                          gdata[i][0].shape_);
        }
      }
    }
    if (!param_.no_bias) {
      Tensor<xpu, 1> gbias = in_grad[conv::kBias].get<xpu, 1, real_t>(s);
      Assign(gbias, req[conv::kBias], sumall_except_dim<1>(grad));
    }
  }

 private:
  inline index_t InitTemp(const mshadow::Shape<4> &ishape,
                          const mshadow::Shape<4> &oshape) {
    const int ksize_y = param_.kernel[0];
    const int ksize_x = param_.kernel[1];
    shape_colunit_ = mshadow::Shape2(ishape[1] * ksize_y * ksize_x,
                                     oshape[2] * oshape[3]);
    shape_dstunit_ = mshadow::Shape3(param_.num_group,
                                     param_.num_filter / param_.num_group,
                                     oshape[2] * oshape[3]);
    // param_.workspace is in elements of sizeof(real_t)
    // if param_.workspace is set to zero the nstep_ equals ishape[0] (batch)
    nstep_ = std::max(
        std::min(
          static_cast<index_t>(param_.workspace / (shape_colunit_.Size() + shape_dstunit_.Size())),
          ishape[0]),
        1U);

    mshadow::Shape<2> scol = mshadow::Shape2(shape_colunit_[0],
                                             shape_colunit_[1] * nstep_);
    mshadow::Shape<3> sdst = mshadow::Shape3(shape_dstunit_[0],
                                             shape_dstunit_[1],
                                             shape_dstunit_[2] * nstep_);
    index_t required_size = scol.Size() + sdst.Size();
    CHECK_GE(param_.workspace, required_size)
      << "\nMinimum workspace size: " << required_size * sizeof(real_t) << " Bytes\n"
      << "Given: " << param_.workspace * sizeof(real_t) << " Bytes";
    return required_size;
  }

  ConvolutionParam param_;
  mshadow::Shape<2> shape_colunit_;
  mshadow::Shape<3> shape_dstunit_;
  index_t nstep_;
};  // class ConvolutionOp

template<typename xpu>
Operator* CreateOp(ConvolutionParam param);

#if DMLC_USE_CXX11
class ConvolutionProp : public OperatorProperty {
 public:
  std::vector<std::string> ListArguments() const override {
    if (!param_.no_bias) {
      return {"data", "weight", "bias"};
    } else {
      return {"data", "weight"};
    }
  }

  void Init(const std::vector<std::pair<std::string, std::string> >& kwargs) override {
    param_.Init(kwargs);
  }

  std::map<std::string, std::string> GetParams() const override {
    return param_.__DICT__();
  }

  bool InferShape(std::vector<TShape> *in_shape,
                  std::vector<TShape> *out_shape,
                  std::vector<TShape> *aux_shape) const override {
    using namespace mshadow;
    if (!param_.no_bias) {
      CHECK_EQ(in_shape->size(), 3) << "Input:[data, weight, bias]";
    } else {
      CHECK_EQ(in_shape->size(), 2) << "Input:[data, weight]";
    }
    const TShape &dshape = (*in_shape)[conv::kData];
    if (dshape.ndim() ==  0) return false;
    CHECK_EQ(dshape.ndim(), 4) \
        << "Input data should be 4D in batch-num_filter-y-x";
    SHAPE_ASSIGN_CHECK(*in_shape,
                       conv::kWeight,
                       Shape4(param_.num_filter, dshape[1] / param_.num_group,
                              param_.kernel[0], param_.kernel[1]));
    if (!param_.no_bias) {
      SHAPE_ASSIGN_CHECK(*in_shape, conv::kBias, Shape1(param_.num_filter));
    }
    out_shape->clear();
    out_shape->push_back(dshape);
    const index_t ksize_y = static_cast<index_t>(param_.kernel[0]);
    const index_t ksize_x = static_cast<index_t>(param_.kernel[1]);
    CHECK_EQ(dshape[1] % param_.num_group, 0) \
        << "input num_filter must divide group size";
    CHECK_EQ(param_.num_filter % param_.num_group, 0) \
        << "output num_filter must divide group size";
    CHECK_GE(param_.kernel.Size(), 0) \
        << "incorrect kernel size: " << param_.kernel;
    CHECK_GE(param_.stride.Size(), 0) \
        << "incorrect stride size: " << param_.stride;
    CHECK_GE(param_.dilate.Size(), 0) \
        << "incorrect dilate size: " << param_.dilate;
    CHECK(ksize_x <= dshape[3] && ksize_y <= dshape[2])
        << "kernel size exceed input";
    (*out_shape)[conv::kOut][1] = param_.num_filter;
    (*out_shape)[conv::kOut][2] = (dshape[2] + 2 * param_.pad[0] -
        (param_.dilate[0] == 1 ? ksize_y : ksize_y * param_.dilate[0] - 1)) / param_.stride[0] + 1;
    (*out_shape)[conv::kOut][3] = (dshape[3] + 2 * param_.pad[1] -
        (param_.dilate[1] == 1 ? ksize_x : ksize_x * param_.dilate[1] - 1)) / param_.stride[1] + 1;
    return true;
  }

  OperatorProperty* Copy() const override {
    auto ptr = new ConvolutionProp();
    ptr->param_ = param_;
    return ptr;
  }

  std::string TypeString() const override {
    return "Convolution";
  }

  std::vector<int> DeclareBackwardDependency(
    const std::vector<int> &out_grad,
    const std::vector<int> &in_data,
    const std::vector<int> &out_data) const override {
    return {out_grad[conv::kOut], in_data[conv::kData], in_data[conv::kWeight]};
  }

  std::vector<ResourceRequest> ForwardResource(
      const std::vector<TShape> &in_shape) const override {
    return {ResourceRequest::kTempSpace};
  }

  std::vector<ResourceRequest> BackwardResource(
      const std::vector<TShape> &in_shape) const override {
    return {ResourceRequest::kTempSpace};
  }

  Operator* CreateOperator(Context ctx) const override;

 private:
  ConvolutionParam param_;
};  // class ConvolutionProp
#endif  // DMLC_USE_CXX11
}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_CONVOLUTION_INL_H_
