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

enum FullyConnectedOpInputs {kData, kWeight, kBias};
enum FullyConnectedOpOutputs {kOut};

struct ConvolutionParam : public dmlc::Parameter<ConvolutionParam> {
  TShape kernel;
  TShape stride;
  TShape pad;
  int nb_filter;
  int nb_group;
  uint32_t nstep;
  bool no_bias;
  DMLC_DECLARE_PARAMETER(ConvolutionParam) {
    DMLC_DECLARE_FIELD(kernel)
      .set_expect_ndim(2).enforce_nonzero()
      .describe("convolution kernel size: (y, x)");

    int stride_shape[] = {1, 1};
    DMLC_DECLARE_FIELD(stride)
      .set_expect_ndim(2).enforce_nonzero()
      .set_default(TShape(stride_shape, stride_shape + 2))
      .describe("convolution stride: (y, x)");

    int pad_shape[] = {1, 1};
    DMLC_DECLARE_FIELD(pad)
      .set_expect_ndim(2)
      .set_default(TShape(pad_shape, pad_shape + 2))
      .describe("pad for convolution: (y, x)");

    DMLC_DECLARE_FIELD(nb_filter)
      .set_lower_bound(1)
      .describe("convolution filter(channel) number");

    DMLC_DECLARE_FIELD(nb_group).set_default(1)
      .describe("number of groups partition");

    DMLC_DECLARE_FIELD(nstep)
      .set_default(2).set_range(1, 10000)
      .describe("process n images once");

    DMLC_DECLARE_FIELD(no_bias).set_default(false)
      .describe("Whether to disable bias parameter.");
  }
};

template<typename xpu>
class ConvolutionOp : public Operator {
 public:
  explicit ConvolutionOp(ConvolutionParam p) {
    this->param_ = p;
  }

  virtual void Forward(const OpContext &ctx,
                       const std::vector<TBlob> &in_data,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &out_data) {
    using namespace mshadow;
    using namespace mshadow::expr;
    CHECK_EQ(req[kOut], kWriteTo);
    size_t expected = param_.no_bias ? 2 : 3;
    CHECK_EQ(in_data.size(), expected);
    CHECK_EQ(out_data.size(), 1);
    // TODO(bing): check the BLAS Handle, be careful
    Stream<xpu> *s = ctx.get_stream<xpu>();
    Tensor<xpu, 4> data = in_data[kData].get<xpu, 4, real_t>(s);
    Tensor<xpu, 3> wmat = in_data[kWeight].get<xpu, 3, real_t>(s);
    Tensor<xpu, 4> out = out_data[kOut].get<xpu, 4, real_t>(s);
    this->InitTemp(data.shape_, out.shape_);
    const index_t nbatch = data.size(0);
    for (index_t i = 0; i < nbatch; i += param_.nstep) {
      const index_t step = std::min(param_.nstep, nbatch - i);
      temp_col_.Resize(mshadow::Shape2(shape_colunit_[0],
                                       shape_colunit_[1] * step));
      temp_dst_.Resize(mshadow::Shape3(shape_dstunit_[0],
                                       shape_dstunit_[1],
                                       shape_dstunit_[2] * step));
      if (param_.pad[0] == 0 && param_.pad[1] == 0) {
        temp_col_ = unpack_patch2col(data.Slice(i, i + step),
                                     param_.kernel[0],
                                     param_.kernel[1],
                                     param_.stride[0]);
        // TODO(bing): make mshadow support dual stride
      } else {
        temp_col_ = unpack_patch2col(pad(data.Slice(i, i + step),
                                         param_.pad[0], param_.pad[1]),
                                     param_.kernel[0],
                                     param_.kernel[1],
                                     param_.stride[0]);
        // TODO(bing): make mshadow support dual stride
      }
      const index_t gstride = temp_col_.size(0) / param_.nb_group;
      for (int gid = 0; gid < param_.nb_group; ++gid) {
        mshadow::Tensor<xpu, 2> tmpc = temp_col_.Slice(gstride * gid,
                                                       gstride * (gid + 1));
        temp_dst_[gid] = dot(wmat[gid], tmpc);
      }
      out.Slice(i, i + step) = swapaxis<1, 0>(reshape(temp_dst_,
                                                      mshadow::Shape4(param_.nb_filter,
                                                      step,
                                                      out.size(2),
                                                      out.size(3))));
    }
    if (!param_.no_bias) {
      // add bias, broadcast bias to dim 1: channel
      Tensor<xpu, 1> bias = in_data[kBias].get<xpu, 1, real_t>(s);
      out += broadcast<1>(bias, out.shape_);
    }
  }

  virtual void Backward(const OpContext &ctx,
                        const std::vector<TBlob> &out_grad,
                        const std::vector<TBlob> &in_data,
                        const std::vector<TBlob> &out_data,
                        const std::vector<OpReqType> &req,
                        const std::vector<TBlob> &in_grad) {
    using namespace mshadow;
    using namespace mshadow::expr;
    // TODO(bing): check the BLAS Handle, be careful
    CHECK_EQ(out_grad.size(), 1);
    size_t expected = param_.no_bias == 0 ? 3 : 2;
    CHECK(in_data.size() == expected && in_grad.size() == expected);
    CHECK_EQ(req.size(), expected);
    // get data
    Stream<xpu> *s = ctx.get_stream<xpu>();
    Tensor<xpu, 4> data = in_data[kData].get<xpu, 4, real_t>(s);
    Tensor<xpu, 3> wmat = in_data[kWeight].get<xpu, 3, real_t>(s);
    Tensor<xpu, 4> grad = out_grad[kOut].get<xpu, 4, real_t>(s);
    Tensor<xpu, 4> gdata = in_grad[kData].get<xpu, 4, real_t>(s);
    Tensor<xpu, 3> gwmat = in_grad[kWeight].get<xpu, 3, real_t>(s);
    this->InitTemp(data.shape_, grad.shape_);
    const index_t nbatch = data.size(0);
    for (index_t i = 0; i < nbatch; i += param_.nstep) {
      const index_t step = std::min(param_.nstep, nbatch - i);
      temp_col_.Resize(mshadow::Shape2(shape_colunit_[0], \
                                       shape_colunit_[1] * step));
      temp_dst_.Resize(mshadow::Shape3(shape_dstunit_[0], \
                                       shape_dstunit_[1], shape_dstunit_[2] * step));
      temp_dst_ = reshape(swapaxis<1, 0>(grad.Slice(i, i + step)), temp_dst_.shape_);
      if (param_.pad[0] == 0 && param_.pad[1] == 0) {
        // TODO(bing): dual stride
        temp_col_ = unpack_patch2col(data.Slice(i, i + step), \
                                     param_.kernel[0], \
                                     param_.kernel[1], \
                                     param_.stride[0]);
      } else {
        // TODO(bing): dual stride
        temp_col_ = unpack_patch2col(pad(data.Slice(i, i + step), param_.pad[0], param_.pad[1]), \
                                     param_.kernel[0], \
                                     param_.kernel[1], \
                                     param_.stride[0]);
      }
      const index_t gstride = temp_col_.size(0) / param_.nb_group;
      for (int gid = 0; gid < param_.nb_group; ++gid) {
        mshadow::Tensor<xpu, 2> tmpc = temp_col_.Slice(gstride * gid, gstride * (gid + 1));
        gwmat[gid] += dot(temp_dst_[gid], tmpc.T());
      }
      if (req[kData] == kWriteTo) {
        for (int gid = 0; gid < param_.nb_group; ++gid) {
          mshadow::Tensor<xpu, 2> tmpc = temp_col_.Slice(gstride * gid, gstride * (gid + 1));
          tmpc = dot(wmat[gid].T(), temp_dst_[gid]);
        }
        if (param_.pad[0] == 0 && param_.pad[1] == 0) {
          gdata.Slice(i, i + step) = pack_col2patch(temp_col_, \
                                                    data.Slice(i, i + step).shape_, \
                                                    param_.kernel[0], \
                                                    param_.kernel[1], \
                                                    param_.stride[0]);
        } else {
          mshadow::Shape<4> pshape = data.Slice(i, i + step).shape_;
          pshape[2] += 2 * param_.pad[0];
          pshape[3] += 2 * param_.pad[1];
          gdata.Slice(i, i + step) = crop(pack_col2patch(temp_col_, \
                                                         pshape, \
                                                         param_.kernel[0], \
                                                         param_.kernel[1], \
                                                         param_.stride[0]), \
                                          gdata[i][0].shape_);
        }
      }
    }
    if (!param_.no_bias) {
      Tensor<xpu, 1> gbias = in_grad[kBias].get<xpu, 1, real_t>(s);
      // Assign(gbias, req[kBias], sumall_except_dim<1>(grad);
      gbias += sumall_except_dim<1>(grad);
    }
  }

 private:
  // TODO(bing): use global resource allocator
  inline void InitTemp(const mshadow::Shape<4> &ishape,
                       const mshadow::Shape<4> &oshape) {
    const int ksize_y = param_.kernel[0];
    const int ksize_x = param_.kernel[1];
    shape_colunit_ = mshadow::Shape2(ishape[1] * ksize_y * ksize_x,
                                     oshape[2] * oshape[3]);
    shape_dstunit_ = mshadow::Shape3(param_.nb_group,
                                     param_.nb_filter / param_.nb_group,
                                     oshape[2] * oshape[3]);
    int nop = (ishape[0] + param_.nstep - 1) / param_.nstep;
    param_.nstep = (ishape[0] + nop - 1) / nop;
    temp_col_.Resize(mshadow::Shape2(shape_colunit_[0],
                                     shape_colunit_[1] * param_.nstep));
    temp_dst_.Resize(mshadow::Shape3(shape_dstunit_[0],
                                     shape_dstunit_[1],
                                     shape_dstunit_[2] * param_.nstep));
  }

  ConvolutionParam param_;
  // TODO(bing): use global resource allocator
  mshadow::TensorContainer<xpu, 2> temp_col_;
  mshadow::TensorContainer<xpu, 3> temp_dst_;
  mshadow::Shape<2> shape_colunit_;
  mshadow::Shape<3> shape_dstunit_;
};  // class ConvolutionOp

template<typename xpu>
Operator* CreateOp(ConvolutionParam param);

#if DMLC_USE_CXX11
class ConvolutionProp : public OperatorProperty {
 public:
  virtual std::vector<std::string> ListArguments() const {
    if (!param_.no_bias) {
      return {"data", "weight", "bias"};
    } else {
      return {"data", "weight"};
    }
  }

  virtual void Init(const std::vector<std::pair<std::string, std::string> >& kwargs) {
    param_.Init(kwargs);
  }

  virtual bool InferShape(std::vector<TShape> *in_shape,
                          std::vector<TShape> *out_shape) const {
    using namespace mshadow;
    if (!param_.no_bias) {
      CHECK_EQ(in_shape->size(), 3) << "Input:[data, weight, bias]";
    } else {
      CHECK_EQ(in_shape->size(), 2) << "Input:[data, weight]";
    }
    const TShape &dshape = (*in_shape)[kData];
    if (dshape.ndim() ==  0) return false;
    CHECK_EQ(dshape.ndim(), 4) \
      << "Input data should be 4D in batch-nb_filter-y-x";
    SHAPE_ASSIGN_CHECK(*in_shape, \
                       kWeight, \
                       Shape3(param_.nb_group, \
                              param_.nb_filter / param_.nb_group, \
                              dshape[1] / param_.nb_group * param_.kernel[0] * param_.kernel[1]));
    if (!param_.no_bias) {
      SHAPE_ASSIGN_CHECK(*in_shape, kBias, Shape1(param_.nb_filter));
    }
    out_shape->clear();
    out_shape->push_back(dshape);
    const index_t ksize_y = static_cast<index_t>(param_.kernel[0]);
    const index_t ksize_x = static_cast<index_t>(param_.kernel[1]);
    const index_t kstride = static_cast<index_t>(param_.stride[0]);
    // TODO(bing) : support dual stride
    CHECK_EQ(dshape[1] % param_.nb_group, 0) \
      << "input nb_filter must divide group size";
    CHECK_EQ(param_.nb_filter % param_.nb_group, 0) \
      << "output nb_filter must divide group size";
    CHECK_GE(param_.kernel.Size(), 0) \
      << "incorrect kernel size: " << param_.kernel;
    CHECK_GE(param_.stride.Size(), 0) \
      << "incorrect stride size: " << param_.stride;
    CHECK(ksize_x <= dshape[3] && ksize_y <= dshape[2])
      << "kernel size exceed input";
    (*out_shape)[kOut][1] = param_.nb_filter;
    (*out_shape)[kOut][2] = (dshape[2] + 2 * param_.pad[0] - ksize_y) / kstride + 1;
    (*out_shape)[kOut][3] = (dshape[3] + 2 * param_.pad[1] - ksize_x) / kstride + 1;
    return true;
  }

  virtual OperatorProperty* Copy() const {
    auto ptr = new ConvolutionProp();
    ptr->param_ = param_;
    return ptr;
  }

  virtual std::string TypeString() const {
    return "Convolution";
  }

  virtual std::vector<int> DeclareBackwardDependency(
      const std::vector<int> &out_grad,
      const std::vector<int> &in_data,
      const std::vector<int> &out_data) const {
    return {out_grad[kOut], in_data[kData], in_data[kWeight]};
  }

  virtual std::vector<std::pair<int, void*> > BackwardInplaceOption(
      const std::vector<int> &out_grad,
      const std::vector<int> &in_data,
      const std::vector<int> &out_data,
      const std::vector<void*> &in_grad) const {
    return {{in_data[kData], in_grad[kData]}};
  }

  Operator* CreateOperator(Context ctx) const;

 private:
  ConvolutionParam param_;
};  // class ConvolutionProp
#endif  // DMLC_USE_CXX11
}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_CONVOLUTION_INL_H_
