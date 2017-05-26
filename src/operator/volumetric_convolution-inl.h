/*!
 * Copyright (c) 2015 by Contributors
 * \file convolution-inl.h
 * \brief
 * \author Bing Xu
*/
#ifndef MXNET_OPERATOR_VOLUMETRIC_CONVOLUTION_INL_H_
#define MXNET_OPERATOR_VOLUMETRIC_CONVOLUTION_INL_H_

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
            enum VolumetricConvolutionOpInputs {
                kData, kWeight, kBias
            };
            enum VolumetricConvolutionOpOutputs {
                kOut
            };
            enum VolumetricConvolutionOpResource {
                kTempSpace
            };
        }

        struct VolumetricConvolutionParam : public dmlc::Parameter<VolumetricConvolutionParam> {
            TShape kernel;
            TShape stride;
            TShape pad;
            uint32_t num_filter;
            uint32_t num_group;
            uint64_t workspace;
            bool no_bias;
            DMLC_DECLARE_PARAMETER(VolumetricConvolutionParam) {
                    int shape[] = { 1, 1, 1 };
                    DMLC_DECLARE_FIELD(kernel).describe("convolution kernel size: (z, y, x)");
                    DMLC_DECLARE_FIELD(stride).set_default(TShape(shape, shape + 3))
                    .describe("convolution stride: (z, y, x)");
                    shape[0] = shape[1] = shape[2] = 0;
                    DMLC_DECLARE_FIELD(pad).set_default(TShape(shape, shape + 3))
                    .describe("pad for convolution: (z, y, x)");
                    DMLC_DECLARE_FIELD(num_filter).set_range(1, 100000)
                    .describe("convolution filter(channel) number");
                    DMLC_DECLARE_FIELD(num_group).set_default(1)
                    .describe("Number of groups partition. "
                    "This option is not supported by CuDNN, you can use SliceChannel to num_group,"
                    "apply convolution and concat instead to achieve the same need.");
                    DMLC_DECLARE_FIELD(workspace).set_default(512).set_range(128, 4096)
                    .describe("Tmp workspace for convolution (MB)");
                    DMLC_DECLARE_FIELD(no_bias).set_default(false)
                    .describe("Whether to disable bias parameter.");
            }
        };

        template<typename xpu>
        class VolumetricConvolutionOp : public Operator {
        public:
            explicit VolumetricConvolutionOp(VolumetricConvolutionParam p) {
                LOG(FATAL) << "NOT IMPLEMENTED";
            }

            virtual void Forward(const OpContext &ctx,
                                 const std::vector <TBlob> &in_data,
                                 const std::vector <OpReqType> &req,
                                 const std::vector <TBlob> &out_data,
                                 const std::vector <TBlob> &aux_args) {
                LOG(FATAL) << "NOT IMPLEMENTED";
            }

            virtual void Backward(const OpContext &ctx,
                                  const std::vector <TBlob> &out_grad,
                                  const std::vector <TBlob> &in_data,
                                  const std::vector <TBlob> &out_data,
                                  const std::vector <OpReqType> &req,
                                  const std::vector <TBlob> &in_grad,
                                  const std::vector <TBlob> &aux_args) {
                LOG(FATAL) << "NOT IMPLEMENTED";
            }
        };  // class ConvolutionOp

        template<typename xpu>
        Operator *CreateOp(VolumetricConvolutionParam param);

#if DMLC_USE_CXX11
class VolumetricConvolutionProp : public OperatorProperty {
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
    CHECK_EQ(dshape.ndim(), 5) \
        << "Input data should be 5D in batch-num_filter-z-y-x";

    Shape<5> weightShape;
    weightShape[0] = param_.num_filter;
    weightShape[1] = dshape[1];
    weightShape[2] = param_.kernel[0];
    weightShape[3] = param_.kernel[1];
    weightShape[4] = param_.kernel[2];

    SHAPE_ASSIGN_CHECK(*in_shape,
                       conv::kWeight,
                       weightShape);
    if (!param_.no_bias) {
      SHAPE_ASSIGN_CHECK(*in_shape, conv::kBias, Shape1(param_.num_filter));
    }
    out_shape->clear();
    out_shape->push_back(dshape);
    const index_t ksize_z = static_cast<index_t>(param_.kernel[0]);
    const index_t ksize_y = static_cast<index_t>(param_.kernel[1]);
    const index_t ksize_x = static_cast<index_t>(param_.kernel[2]);
    CHECK_EQ(dshape[1] % param_.num_group, 0) \
        << "input num_filter must divide group size";
    CHECK_EQ(param_.num_filter % param_.num_group, 0) \
        << "output num_filter must divide group size";
    CHECK_GE(param_.kernel.Size(), 0) \
        << "incorrect kernel size: " << param_.kernel;
    CHECK_GE(param_.stride.Size(), 0) \
        << "incorrect stride size: " << param_.stride;
    CHECK(ksize_x <= dshape[4] && ksize_y <= dshape[3] && ksize_z <= dshape[2])
        << "kernel size exceed input";
    (*out_shape)[conv::kOut][1] = param_.num_filter;
    (*out_shape)[conv::kOut][2] = (dshape[2] + 2 * param_.pad[0] - ksize_z) / param_.stride[0] + 1;
    (*out_shape)[conv::kOut][3] = (dshape[3] + 2 * param_.pad[1] - ksize_y) / param_.stride[1] + 1;
    (*out_shape)[conv::kOut][4] = (dshape[4] + 2 * param_.pad[2] - ksize_x) / param_.stride[2] + 1;
    return true;
  }

  OperatorProperty* Copy() const override {
    auto ptr = new VolumetricConvolutionProp();
    ptr->param_ = param_;
    return ptr;
  }

  std::string TypeString() const override {
    return "VolumetricConvolution";
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
  VolumetricConvolutionParam param_;
};  // class ConvolutionProp
#endif  // DMLC_USE_CXX11
    }  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_CONVOLUTION_INL_H_
