/*!
 * Copyright (c) 2015 by Contributors
 * \file convolution_op-inl.h
 * \brief convolution op
 * \author Bing Xu
*/
#ifndef MXNET_STATIC_OPERATOR_CONVOLUTION_OP_INL_H_
#define MXNET_STATIC_OPERATOR_CONVOLUTION_OP_INL_H_

#include <mxnet/static_operator.h>
#include <vector>
#include <algorithm>
#include "./static_operator_common.h"
#include "./param.h"

namespace mxnet {
namespace op {
template<typename xpu>
class ConvolutionOp : public StaticOperator {
 public:
  virtual std::vector<ArgType> DescribeArgs() const {
    ArgType ret[] = {kDataArg, kWeightArg, kBiasArg};
    if (param_.no_bias == 0) {
      return std::vector<ArgType>(ret, ret + 3);
    } else {
      return std::vector<ArgType>(ret, ret + 2);
    }
  }
  virtual void SetParam(const char *name, const char *val) {
    param_.SetParam(name, val);
  }
  virtual void InferShape(std::vector<TShape> *in_shape,
                          std::vector<TShape> *out_shape) {
    using namespace mshadow;
    if (param_.no_bias == 0) {
      CHECK_EQ(in_shape->size(), 3) << "Input:[data, weight, bias]";
    } else {
      CHECK_EQ(in_shape->size(), 2) << "Input:[data, weight]";
    }
    CHECK_GT(param_.num_channel, 0);
    const TShape &dshape = (*in_shape)[0];
    CHECK_EQ(dshape.ndim(), 4) << \
                         "Input data should be 4D in batch-channel-y-x";
    ShapeAssignCheck((*in_shape)[1], Shape4(param_.num_channel,
                                            dshape[1],
                                            param_.kernel_y,
                                            param_.kernel_x));
    if (param_.no_bias == 0) {
      ShapeAssignCheck((*in_shape)[2], Shape1(param_.num_channel));
    }
    out_shape->clear();
    out_shape->push_back(dshape);
    const index_t ksize_y = static_cast<index_t>(param_.kernel_y);
    const index_t ksize_x = static_cast<index_t>(param_.kernel_x);
    const index_t kstride = static_cast<index_t>(param_.stride_y);
    // todo : support dual stride
    mshadow::Shape<4> ishape = in_shape->at(0).get<4>();
    CHECK_EQ(ishape[1] % param_.num_group, 0) << \
      "input channels must divide group size";
    CHECK_EQ(param_.num_channel % param_.num_group, 0) << \
      "output channels must divide group size";
    CHECK(ksize_y > 0 && ksize_x > 0) << \
      "incorrect kernel size";
    CHECK(ksize_x <= ishape[3] && ksize_y <= ishape[2]) << \
      "kernel size exceed input";
    (*out_shape)[0][1] = param_.num_channel;
    (*out_shape)[0][2] = (ishape[2] + 2 * param_.pad_y - ksize_y) / kstride + 1;
    (*out_shape)[0][3] = (ishape[3] + 2 * param_.pad_x - ksize_x) / kstride + 1;
  }
  virtual void Forward(Option opt,
                       RunContext ctx,
                       const std::vector<TBlob> &in_data,
                       const std::vector<TBlob> &out_data) {
    using namespace mshadow;
    using namespace mshadow::expr;
    // TODO(bing): check the BLAS Handle, be careful
    // maybe need blas handle from context
    size_t expected = param_.no_bias == 0 ? 3 : 2;
    CHECK_EQ(in_data.size(), expected);
    CHECK_EQ(out_data.size(), 1);
    // weight shape with group
    TShape ws;
    ShapeAssignCheck(ws, Shape3(param_.num_group,
                       param_.num_channel / param_.num_group,
                       param_.num_input_channel / param_.num_group *
                       param_.kernel_y * param_.kernel_x));
    Stream<xpu> *s = static_cast<Stream<xpu> *>(ctx.stream);
    Tensor<xpu, 4> data = in_data[0].get<xpu, 4, real_t>(s);
    Tensor<xpu, 3> wmat = in_data[1].get_with_shape<xpu, 3, real_t>(ws, s);
    Tensor<xpu, 4> out = out_data[0].get<xpu, 4, real_t>(s);
    this->InitTemp(data.shape_, out.shape_);
    const index_t nbatch = data.size(0);
    for (index_t i = 0; i < nbatch; i += nstep_) {
      // resize, incase last batch is smaller
      const index_t step = std::min(nstep_, nbatch - i);
      temp_col_.Resize(mshadow::Shape2(shape_colunit_[0],
                                       shape_colunit_[1] * step));
      temp_dst_.Resize(mshadow::Shape3(shape_dstunit_[0],
                                       shape_dstunit_[1],
                                       shape_dstunit_[2] * step));

      if (param_.pad_x == 0 && param_.pad_y == 0) {
        temp_col_ = unpack_patch2col(data.Slice(i, i+step),
                                     param_.kernel_y,
                                     param_.kernel_x,
                                     param_.stride_y);
        // TODO(bing): make mshadow support dual stride
      } else {
        temp_col_ = unpack_patch2col(pad(data.Slice(i, i+step),
                                         param_.pad_y, param_.pad_x),
                                     param_.kernel_y,
                                     param_.kernel_x,
                                     param_.stride_y);
        // TODO(bing): make mshadow support dual stride
      }
      const index_t gstride = temp_col_.size(0) / param_.num_group;
      for (int gid = 0; gid < param_.num_group; ++gid) {
        mshadow::Tensor<xpu, 2> tmpc = temp_col_.Slice(gstride * gid,
                                                      gstride * (gid + 1));
        temp_dst_[gid] = dot(wmat[gid], tmpc);
      }
      out.Slice(i, i + step) = swapaxis<1, 0>(reshape(temp_dst_,
                                      mshadow::Shape4(param_.num_channel,
                                                      step,
                                                      out.size(2),
                                                      out.size(3))));
    }
    if (param_.no_bias == 0) {
      // add bias, broadcast bias to dim 1: channel
      Tensor<xpu, 1> bias = in_data[2].get<xpu, 1, real_t>(s);
      out += broadcast<1>(bias, out.shape_);
    }
  }
  virtual void Backward(RunContext ctx,
                        const std::vector<TBlob> &grad_next,
                        const std::vector<TBlob> &in_data,
                        const std::vector<TBlob> &out_grad,
                        const std::vector<GradReqType> &req) {
    using namespace mshadow;
    using namespace mshadow::expr;
    // TODO(bing): check the BLAS Handle, be careful
    // maybe need blas handle from context
    CHECK_EQ(grad_next.size(), 1);
    size_t expected = param_.no_bias == 0 ? 3 : 2;
    CHECK(in_data.size() == expected && out_grad.size() == expected);
    CHECK_EQ(req.size(), expected);
    TShape ws;
    ShapeAssignCheck(ws, Shape3(param_.num_group,
                       param_.num_channel / param_.num_group,
                       param_.num_input_channel / param_.num_group *
                       param_.kernel_y * param_.kernel_x));
    Stream<xpu> *s = static_cast<Stream<xpu> *>(ctx.stream);
    Tensor<xpu, 4> data = in_data[0].get<xpu, 4, real_t>(s);
    Tensor<xpu, 3> wmat = in_data[1].get_with_shape<xpu, 3, real_t>(ws, s);
    Tensor<xpu, 4> grad = grad_next[0].get<xpu, 4, real_t>(s);
    Tensor<xpu, 4> gdata = out_grad[0].get<xpu, 4, real_t>(s);
    Tensor<xpu, 3> gwmat = out_grad[0].get_with_shape<xpu, 3, real_t>(ws, s);
    this->InitTemp(data.shape_, grad.shape_);
    const index_t nbatch = data.size(0);
    for (index_t i = 0; i < nbatch; i += nstep_) {
      const index_t step = std::min(nstep_, nbatch-i);
      temp_col_.Resize(mshadow::Shape2(shape_colunit_[0],
                                       shape_colunit_[1] * step));
      temp_dst_.Resize(mshadow::Shape3(shape_dstunit_[0],
                                       shape_dstunit_[1],
                                       shape_dstunit_[2] * step));
      temp_dst_ = reshape(swapaxis<1, 0>(grad.Slice(i, i + step)),
                          temp_dst_.shape_);
      if (param_.pad_x == 0 && param_.pad_y == 0) {
        temp_col_ = unpack_patch2col(data.Slice(i, i + step),
                                     param_.kernel_y,
                                     param_.kernel_x,
                                     param_.stride_y);
        // TODO(bing): dual stride
      } else {
        temp_col_ = unpack_patch2col(pad(data.Slice(i, i + step),
                                         param_.pad_y, param_.pad_x),
                                     param_.kernel_y,
                                     param_.kernel_x,
                                     param_.stride_y);
        // TODO(bing): dual stride
      }
      const index_t gstride = temp_col_.size(0) / param_.num_group;
      for (int gid = 0; gid < param_.num_group; ++gid) {
        mshadow::Tensor<xpu, 2> tmpc = temp_col_.Slice(gstride * gid,
                                                      gstride * (gid + 1));
        gwmat[gid] += dot(temp_dst_[gid], tmpc.T());
      }
      if (req[0] != kNullOp) {
        for (int gid = 0; gid < param_.num_group; ++gid) {
          mshadow::Tensor<xpu, 2> tmpc = temp_col_.Slice(gstride * gid,
                                                        gstride * (gid+1));
          tmpc = dot(wmat[gid].T(), temp_dst_[gid]);
        }

        if (param_.pad_x == 0 && param_.pad_y == 0) {
          Tensor<xpu, 4> gdata_tmp = gdata.Slice(i, i + step);
          Assign(gdata_tmp,
                 req[0],
                 pack_col2patch(temp_col_,
                                data.Slice(i, i + step).shape_,
                                param_.kernel_y,
                                param_.kernel_x,
                                param_.stride_y));
        // TODO(bing): dual stride
        } else {
          mshadow::Shape<4> pshape = data.Slice(i, i + step).shape_;
          pshape[2] += 2 * param_.pad_y; pshape[3] += 2 * param_.pad_x;
          Tensor<xpu, 4> gdata_tmp = gdata.Slice(i, i + step);
          Assign(gdata_tmp,
                 req[0],
                 crop(pack_col2patch(temp_col_,
                                     pshape,
                                     param_.kernel_y,
                                     param_.kernel_x,
                                     param_.stride_y),
                      data[i][0].shape_));
        // TODO(bing): dual stride
        }
      }
    }
    if (param_.no_bias == 0) {
      Tensor<xpu, 1> gbias = out_grad[2].get<xpu, 1, real_t>(s);
      Assign(gbias, req[2], sumall_except_dim<1>(grad));
    }
  }

 private:
  /*! \brief Alloc temp space for pack/unpack */
  inline void InitTemp(mshadow::Shape<4> ishape, mshadow::Shape<4> oshape) {
    const index_t ksize_y = static_cast<index_t>(param_.kernel_y);
    const index_t ksize_x = static_cast<index_t>(param_.kernel_x);
    // this is the unit size of each temp structure
    shape_colunit_ = mshadow::Shape2(ishape[1] * ksize_y * ksize_x,
                                     oshape[2] * oshape[3]);
    shape_dstunit_ = mshadow::Shape3(param_.num_group,
                                     param_.num_channel/param_.num_group,
                                     oshape[2] * oshape[3]);
    nstep_ = std::max(std::min((index_t)(param_.temp_col_max /
                                         shape_colunit_.Size()),
                               ishape[0]), 1U);
    // make nstep more balanced,
    // nstep will use exactly same number of operations to finish,
    index_t nop = (ishape[0]+nstep_-1) / nstep_;
    nstep_ = (ishape[0] + nop - 1)/ nop;
    CHECK_GT(nstep_, 0);
    // helper structure
    temp_col_.Resize(mshadow::Shape2(shape_colunit_[0],
                                     shape_colunit_[1] * nstep_));
    temp_dst_.Resize(mshadow::Shape3(shape_dstunit_[0],
                                     shape_dstunit_[1],
                                     shape_dstunit_[2] * nstep_));
  }
  /*! \brief parameters that potentially be useful */
  Param param_;
  /*! \brief temporary data structure to store patches */
  mshadow::TensorContainer<xpu, 2> temp_col_;
  /*! \brief temporary data structure to store results */
  mshadow::TensorContainer<xpu, 3> temp_dst_;
  /*! \brief shape of column unit */
  mshadow::Shape<2> shape_colunit_;
  /*! \brief shape of dst unit */
  mshadow::Shape<3> shape_dstunit_;
  /*! \brief how many number of batches to be unpacked together */
  mshadow::index_t nstep_;
};  // class ConvolutionOp
}  // namespace op
}  // namespace mxnet
#endif  // MXNET_STATIC_OPERATOR_CONVOLUTION_OP_INL_H_
