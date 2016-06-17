/*!
 * Copyright (c) 2015 by Contributors
 * \file deconvolution-inl.h
 * \brief
 * \author Wei Wu
*/
#ifndef MXNET_OPERATOR_DECONVOLUTION_INL_H_
#define MXNET_OPERATOR_DECONVOLUTION_INL_H_

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

namespace deconv {
  enum DeconvolutionOpInputs {kData, kWeight, kBias};
  enum DeconvolutionOpOutputs {kOut};
  enum DeconvolutionOpResource {kTempSpace};
}

struct DeconvolutionParam : public dmlc::Parameter<DeconvolutionParam> {
  TShape kernel;
  TShape stride;
  TShape pad;
  TShape adj;
  TShape target_shape;
  uint32_t num_filter;
  uint32_t num_group;
  uint64_t workspace;
  bool no_bias;
  DMLC_DECLARE_PARAMETER(DeconvolutionParam) {
    int shape[] = {1, 1};
    DMLC_DECLARE_FIELD(kernel).describe("deconvolution kernel size: (y, x)");
    DMLC_DECLARE_FIELD(stride).set_default(TShape(shape, shape + 2))
        .describe("deconvolution stride: (y, x)");
    shape[0] = shape[1] = 0;
    DMLC_DECLARE_FIELD(pad).set_default(TShape(shape, shape + 2))
        .describe("pad for deconvolution: (y, x), a good number is : (kernel-1)/2, "
                  "if target_shape set, pad will be ignored and will be computed "
                  "automatically");
    DMLC_DECLARE_FIELD(adj).set_default(TShape(shape, shape + 2))
        .describe("adjustment for output shape: (y, x), if target_shape set, adj "
                  "will be ignored and will be computed automatically");
    DMLC_DECLARE_FIELD(target_shape).set_default(TShape(shape, shape + 2))
        .describe("output shape with targe shape : (y, x)");
    DMLC_DECLARE_FIELD(num_filter).set_range(1, 100000)
        .describe("deconvolution filter(channel) number");
    DMLC_DECLARE_FIELD(num_group).set_default(1)
        .describe("number of groups partition");
    DMLC_DECLARE_FIELD(workspace).set_default(512).set_range(0, 8192)
        .describe("Tmp workspace for deconvolution (MB)");
    DMLC_DECLARE_FIELD(no_bias).set_default(true)
        .describe("Whether to disable bias parameter.");
  }

  inline void InferPad(index_t input_y, index_t input_x,
                       index_t* o_pad_y, index_t* o_pad_x,
                       index_t* o_adj_y, index_t* o_adj_x) const {
    index_t& pad_y = *o_pad_y;
    index_t& pad_x = *o_pad_x;
    index_t& adj_y = *o_adj_y;
    index_t& adj_x = *o_adj_x;
    if (target_shape[0] != 0 || target_shape[1] != 0) {
      pad_y = stride[0] * (input_y - 1) + kernel[0];
      pad_x = stride[1] * (input_x - 1) + kernel[1];
      CHECK_GE(pad_y, target_shape[0])
          << "too big target shape";
      CHECK_GE(pad_x, target_shape[1])
          << "too big target shape";
      pad_y -= target_shape[0];
      pad_x -= target_shape[1];
      adj_y = pad_y % 2; pad_y = (pad_y + 1) / 2;
      adj_x = pad_x % 2; pad_x = (pad_x + 1) / 2;
    } else {
      pad_y = pad[0];
      pad_x = pad[1];
      adj_y = adj[0];
      adj_x = adj[1];
    }
  }
};

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
    CHECK_EQ(req[deconv::kOut], kWriteTo);
    size_t expected = param_.no_bias ? 2 : 3;
    CHECK_EQ(in_data.size(), expected);
    CHECK_EQ(out_data.size(), 1);
    Stream<xpu> *s = ctx.get_stream<xpu>();
    Tensor<xpu, 4, DType> data = in_data[deconv::kData].get<xpu, 4, DType>(s);
    Tensor<xpu, 4, DType> out = out_data[deconv::kOut].get<xpu, 4, DType>(s);

    index_t pad_y, pad_x, adj_y, adj_x;
    param_.InferPad(data.size(2), data.size(3), &pad_y, &pad_x, &adj_y, &adj_x);

    Shape<3> wmat_shape =
        Shape3(param_.num_group,
               data.shape_[1] / param_.num_group,
               param_.num_filter / param_.num_group * param_.kernel[0] * param_.kernel[1]);
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
      if (pad_y == 0 && pad_x == 0) {
        temp_col = unpack_patch2col(out.Slice(i, i + step),
                                    param_.kernel[0],
                                    param_.kernel[1],
                                    param_.stride[0],
                                    param_.stride[1],
                                    1, 1);  // Deconvolution only support dilate equals 1
      } else {
        temp_col = unpack_patch2col(pad(out.Slice(i, i + step),
                                        pad_y, pad_x),
                                    param_.kernel[0],
                                    param_.kernel[1],
                                    param_.stride[0],
                                    param_.stride[1],
                                    1, 1);  // Deconvolution only support dilate equals 1
      }
      const index_t gstride = temp_col.size(0) / param_.num_group;
      for (uint32_t gid = 0; gid < param_.num_group; ++gid) {
        mshadow::Tensor<xpu, 2, DType> tmpc = temp_col.Slice(gstride * gid,
                                              gstride * (gid + 1));
        tmpc = dot(wmat[gid].T(), temp_dst[gid]);
      }
      if (pad_y == 0 && pad_x == 0) {
        out.Slice(i, i + step) = pack_col2patch(temp_col,
                                   out.Slice(i, i + step).shape_,
                                   param_.kernel[0],
                                   param_.kernel[1],
                                   param_.stride[0],
                                   1);  // Deconvolution only support dilate equals 1
      } else {
        Shape<4> pshape = out.Slice(i, i + step).shape_;
        pshape[2] += 2 * pad_y;
        pshape[3] += 2 * pad_x;
        out.Slice(i, i + step) = crop(pack_col2patch(temp_col,
                                        pshape,
                                        param_.kernel[0],
                                        param_.kernel[1],
                                        param_.stride[0],
                                        1),  // Deconvolution only support dilate equals 1
                                        out[i][0].shape_);
      }
    }
    if (!param_.no_bias) {
      // add bias, broadcast bias to dim 1: channel
      Tensor<xpu, 1, DType> bias = in_data[deconv::kBias].get<xpu, 1, DType>(s);
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
    CHECK_EQ(in_data[deconv::kWeight].CheckContiguous(), true);
    // get data
    Stream<xpu> *s = ctx.get_stream<xpu>();
    Tensor<xpu, 4, DType> data = in_data[deconv::kData].get<xpu, 4, DType>(s);
    Tensor<xpu, 4, DType> grad = out_grad[deconv::kOut].get<xpu, 4, DType>(s);
    Tensor<xpu, 4, DType> gdata = in_grad[deconv::kData].get<xpu, 4, DType>(s);
    Shape<3> wmat_shape =
        Shape3(param_.num_group,
               data.shape_[1] / param_.num_group,
               param_.num_filter / param_.num_group * param_.kernel[0] * param_.kernel[1]);
    Tensor<xpu, 3, DType> wmat =
        in_data[deconv::kWeight].get_with_shape<xpu, 3, DType>(wmat_shape, s);
    Tensor<xpu, 3, DType> gwmat =
        in_grad[deconv::kWeight].get_with_shape<xpu, 3, DType>(wmat_shape, s);
#if defined(__CUDACC__)
    CHECK_EQ(s->blas_handle_ownership_, Stream<xpu>::OwnHandle)
        << "Must init CuBLAS handle in stream";
#endif
    index_t pad_y, pad_x, adj_y, adj_x;
    param_.InferPad(data.size(2), data.size(3), &pad_y, &pad_x, &adj_y, &adj_x);

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
      if (pad_y == 0 && pad_x == 0) {
        temp_col = unpack_patch2col(grad.Slice(i, i + step),
                                     param_.kernel[0],
                                     param_.kernel[1],
                                     param_.stride[0],
                                     param_.stride[1],
                                     1, 1);  // Deconvolution only support dilate equals 1
      } else {
        temp_col = unpack_patch2col(pad(grad.Slice(i, i + step), pad_y, pad_x),
                                     param_.kernel[0],
                                     param_.kernel[1],
                                     param_.stride[0],
                                     param_.stride[1],
                                     1, 1);  // Deconvolution only support dilate equals 1
      }
      const index_t gstride = temp_col.size(0) / param_.num_group;
      for (uint32_t gid = 0; gid < param_.num_group; ++gid) {
        Tensor<xpu, 2, DType> tmpc = temp_col.Slice(gstride * gid, gstride * (gid + 1));
        if (i == 0) {
          Tensor<xpu, 2, DType> tmp_gwmat = gwmat[gid];
          Assign(tmp_gwmat, req[deconv::kWeight], dot(temp_dst[gid], tmpc.T()));
        } else {
          gwmat[gid] += dot(temp_dst[gid], tmpc.T());
        }
      }
      if (req[deconv::kData] == kWriteTo || req[deconv::kData] == kWriteInplace) {
        for (uint32_t gid = 0; gid < param_.num_group; ++gid) {
          Tensor<xpu, 2, DType> tmpc = temp_col.Slice(gstride * gid, gstride * (gid + 1));
          temp_dst[gid] = dot(wmat[gid], tmpc);
        }
        gdata.Slice(i, i + step) = swapaxis<1, 0>(reshape(temp_dst,
                                                    mshadow::Shape4(gdata.shape_[1],
                                                    step,
                                                    gdata.size(2),
                                                    gdata.size(3))));
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
    const int ksize_y = param_.kernel[0];
    const int ksize_x = param_.kernel[1];
    shape_colunit_ = mshadow::Shape2(ishape[1] * ksize_y * ksize_x,
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

  DeconvolutionParam param_;
  mshadow::Shape<2> shape_colunit_;
  mshadow::Shape<3> shape_dstunit_;
  index_t nstep_;
};  // class DeconvolutionOp

template<typename xpu>
Operator* CreateOp(DeconvolutionParam param, int dtype);

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
    const TShape &dshape = (*in_shape)[deconv::kData];
    if (dshape.ndim() ==  0) return false;
    CHECK_EQ(dshape.ndim(), 4) \
        << "Input data should be 4D in batch-num_filter-y-x";
    SHAPE_ASSIGN_CHECK(*in_shape,
                       deconv::kWeight,
                       Shape4(dshape[1], param_.num_filter / param_.num_group,
                              param_.kernel[0], param_.kernel[1]));
    if (!param_.no_bias) {
      SHAPE_ASSIGN_CHECK(*in_shape, deconv::kBias, Shape1(param_.num_filter));
    }
    out_shape->clear();
    out_shape->push_back(dshape);
    // osize = stride * (isize - 1) + ksize - 2 * pad + adj
    const index_t ksize_y = static_cast<index_t>(param_.kernel[0]);
    const index_t ksize_x = static_cast<index_t>(param_.kernel[1]);
    index_t pad_y, pad_x, adj_y, adj_x;
    param_.InferPad(dshape[2], dshape[3], &pad_y, &pad_x, &adj_y, &adj_x);
    CHECK_EQ(dshape[1] % param_.num_group, 0) \
        << "input num_filter must divide group size";
    CHECK_EQ(param_.num_filter % param_.num_group, 0) \
        << "output num_filter must divide group size";
    CHECK_GT(param_.kernel.Size(), 0) \
        << "incorrect kernel size: " << param_.kernel;
    CHECK_GT(param_.stride.Size(), 0) \
        << "incorrect stride size: " << param_.stride;
    CHECK_GE(ksize_y-1, adj_y) << "adj(y) must be samller than kernel(h)";
    CHECK_GE(ksize_x-1, adj_x) << "adj(x) must be samller than kernel(w)";
    (*out_shape)[deconv::kOut][1] = param_.num_filter;
    (*out_shape)[deconv::kOut][2] = param_.stride[0] * (dshape[2] - 1) +
        ksize_y - 2 * pad_y + adj_y;
    (*out_shape)[deconv::kOut][3] = param_.stride[1] * (dshape[3] - 1) +
        ksize_x - 2 * pad_x + adj_x;
    if (param_.target_shape[0] > 0) {
      CHECK_EQ(param_.target_shape[0], (*out_shape)[deconv::kOut][2]) \
          << "param_.target_shape[0] was not reasonable, pelase set it carefully";
    }
    if (param_.target_shape[1] > 0) {
      CHECK_EQ(param_.target_shape[1], (*out_shape)[deconv::kOut][3]) \
          << "param_.target_shape[1] was not reasonable, pelase set it carefully";
    }
    return true;
  }

  bool InferType(std::vector<int> *in_type,
                 std::vector<int> *out_type,
                 std::vector<int> *aux_type) const override {
    CHECK_GE(in_type->size(), 1);
    int dtype = (*in_type)[0];
    CHECK_NE(dtype, -1) << "First input must have specified type";
    for (index_t i = 0; i < in_type->size(); ++i) {
      if ((*in_type)[i] == -1) {
        (*in_type)[i] = dtype;
      } else {
        CHECK_EQ((*in_type)[i], dtype) << "This layer requires uniform type. "
                                       << "Expected " << dtype << " v.s. given "
                                       << (*in_type)[i] << " at " << ListArguments()[i];
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
#endif  // MXNET_OPERATOR_DECONVOLUTION_INL_H_
