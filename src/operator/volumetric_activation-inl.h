/*!
 * Copyright (c) 2015 by Contributors
 * \file activation-inl.h
 * \brief Activation operator
 * \author Bing Xu
*/
#ifndef MXNET_OPERATOR_VOLUMETRIC_ACTIVATION_INL_H_
#define MXNET_OPERATOR_VOLUMETRIC_ACTIVATION_INL_H_

#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <mxnet/operator.h>
#include <cstring>
#include <map>
#include <string>
#include <vector>
#include <utility>
#include "./operator_common.h"

namespace mxnet {
    namespace op {
// Declare enumeration of input order to make code more intuitive.
// // These enums are only visible within this header
        namespace activation {
            enum VolumetricActivationOpInputs {
                kData
            };
            enum VolumetricActivationOpOutputs {
                kOut
            };
            enum VolumetricActivationOpType {
                kReLU, kSigmoid, kTanh
            };
        }  // activation

        struct VolumetricActivationParam : public dmlc::Parameter<VolumetricActivationParam> {
            // use int for enumeration
            int act_type;
            DMLC_DECLARE_PARAMETER(VolumetricActivationParam) {
                    DMLC_DECLARE_FIELD(act_type)
                            .add_enum("relu", activation::kReLU)
                            .add_enum("sigmoid", activation::kSigmoid)
                            .add_enum("tanh", activation::kTanh)
                            .describe("Activation function to be applied.");
            }
        };

/**
 * \brief This is the implementation of activation operator.
 * \tparam xpu The device that the op will be executed on.
 */
        template<typename xpu, typename ForwardOp, typename BackwardOp>
        class VolumetricActivationOp : public Operator {
        public:
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
        };  // class ActivationOp

// Decalre Factory function, used for dispatch specialization
        template<typename xpu>
        Operator *CreateOp(VolumetricActivationParam type);

#if DMLC_USE_CXX11
class VolumetricActivationProp : public OperatorProperty {
 public:
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
    CHECK_EQ(in_shape->size(), 1) << "Input:[data]";
    const TShape &dshape = in_shape->at(activation::kData);
    if (dshape.ndim() == 0) return false;
    out_shape->clear();
    out_shape->push_back(dshape);
    return true;
  }

  OperatorProperty* Copy() const override {
    auto ptr = new VolumetricActivationProp();
    ptr->param_ = param_;
    return ptr;
  }

  std::string TypeString() const override {
    return "VolumetricActivation";
  }

  // decalre dependency and inplace optimization options
  std::vector<int> DeclareBackwardDependency(
    const std::vector<int> &out_grad,
    const std::vector<int> &in_data,
    const std::vector<int> &out_data) const override {
#if MXNET_USE_CUDNN == 1
    return {out_grad[activation::kOut], out_data[activation::kOut], in_data[activation::kData]};
#else
    return {out_grad[activation::kOut], out_data[activation::kOut]};
#endif  // MXNET_USE_CUDNN
  }

  std::vector<std::pair<int, void*> > BackwardInplaceOption(
    const std::vector<int> &out_grad,
    const std::vector<int> &in_data,
    const std::vector<int> &out_data,
    const std::vector<void*> &in_grad) const override {
    return {{out_grad[activation::kOut], in_grad[activation::kData]}};
  }

  std::vector<std::pair<int, void*> > ForwardInplaceOption(
    const std::vector<int> &in_data,
    const std::vector<void*> &out_data) const override {
    return {{in_data[activation::kData], out_data[activation::kOut]}};
  }

  Operator* CreateOperator(Context ctx) const override;

 private:
  VolumetricActivationParam param_;
};
#endif  // DMLC_USE_CXX11
    }  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_ACTIVATION_INL_H_
