/*!
 * Copyright (c) 2015 by Contributors
 * \file pooling-inl.h
 * \brief
 * \author Bing Xu
*/

#ifndef MXNET_OPERATOR_VOLUMETRIC_POOLING_INL_H_
#define MXNET_OPERATOR_VOLUMETRIC_POOLING_INL_H_

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

        namespace pool_enum {
            enum VolumetricPoolingOpInputs {
                kData
            };
            enum VolumetricPoolingOpOutputs {
                kOut
            };
            enum VolumetricPoolingOpType {
                kMaxPooling, kAvgPooling, kSumPooling
            };
        }  // namespace pool_enum

        struct VolumetricPoolingParam : public dmlc::Parameter<VolumetricPoolingParam> {
            TShape kernel;
            TShape stride;
            TShape pad;
            int pool_type;
            DMLC_DECLARE_PARAMETER(VolumetricPoolingParam) {
                    // TODO(bing) change to only set lower bound
                    DMLC_DECLARE_FIELD(kernel)
                            .set_expect_ndim(3).enforce_nonzero()
                            .describe("pooling kernel size: (z, y, x)");

                    DMLC_DECLARE_FIELD(pool_type)
                    .add_enum("max", pool_enum::kMaxPooling)
                    .add_enum("avg", pool_enum::kAvgPooling)
                    .add_enum("sum", pool_enum::kSumPooling)
                    .describe("Pooling type to be applied.");

                    int stride_shape[] = { 1, 1, 1 };
                    DMLC_DECLARE_FIELD(stride).set_default(TShape(stride_shape, stride_shape + 3))
                    .set_expect_ndim(3).enforce_nonzero()
                    .describe("stride: for pooling (z, y, x)");

                    int pad_shape[] = { 0, 0, 0 };
                    DMLC_DECLARE_FIELD(pad).set_default(TShape(pad_shape, pad_shape + 3))
                    .set_expect_ndim(3)
                    .describe("pad for pooling: (z, y, x)");
            }
        };

        template<typename xpu, typename Reducer>
        class VolumetricPoolingOp : public Operator {
        public:
            explicit VolumetricPoolingOp(VolumetricPoolingParam p) {
                this->param_ = p;
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

        private:
            VolumetricPoolingParam param_;
        };  // class VolumetricPoolingOp

        template<typename xpu>
        Operator *CreateOp(VolumetricPoolingParam param);


#if DMLC_USE_CXX11
        class VolumetricPoolingProp : public OperatorProperty {
        public:
            void Init(const std::vector <std::pair<std::string, std::string>> &kwargs) override {
                param_.Init(kwargs);
            }

            std::map <std::string, std::string> GetParams() const override {
                return param_.__DICT__();
            }

            bool InferShape(std::vector <TShape> *in_shape,
                            std::vector <TShape> *out_shape,
                            std::vector <TShape> *aux_shape) const override {
                CHECK_EQ(in_shape->size(), 1);
                const TShape &dshape = (*in_shape)[0];
                CHECK_EQ(dshape.ndim(), 5) << \
                               "Pooling: Input data should be 5D in (batch, channel, z, y, x)";
                TShape oshape = dshape;
                if (dshape.ndim() == 0) return false;
                oshape[2] = std::min(dshape[2] + 2 * param_.pad[0] - param_.kernel[0] + param_.stride[0] - 1,
                                     dshape[2] + 2 * param_.pad[0] - 1) / param_.stride[0] + 1;
                oshape[3] = std::min(dshape[3] + 2 * param_.pad[1] - param_.kernel[1] + param_.stride[1] - 1,
                                     dshape[3] + 2 * param_.pad[1] - 1) / param_.stride[1] + 1;
                oshape[4] = std::min(dshape[4] + 2 * param_.pad[2] - param_.kernel[2] + param_.stride[2] - 1,
                                     dshape[4] + 2 * param_.pad[2] - 1) / param_.stride[2] + 1;
                CHECK(oshape[2] > 0 && oshape[3] > 0 && oshape[4] > 0) << "Pooling: kernel size exceed input";
                out_shape->clear();
                out_shape->push_back(oshape);
                return true;
            }

            OperatorProperty *Copy() const override {
                VolumetricPoolingProp *prop_sym = new VolumetricPoolingProp();
                prop_sym->param_ = this->param_;
                return prop_sym;
            }

            std::string TypeString() const override {
                return "VolumetricPooling";
            }

            std::vector<int> DeclareBackwardDependency(
                    const std::vector<int> &out_grad,
                    const std::vector<int> &in_data,
                    const std::vector<int> &out_data) const override {
                return {out_grad[pool_enum::kOut], in_data[pool_enum::kData], out_data[pool_enum::kOut]};
            }

            std::vector <std::pair<int, void *>> BackwardInplaceOption(
                    const std::vector<int> &out_grad,
                    const std::vector<int> &in_data,
                    const std::vector<int> &out_data,
                    const std::vector<void *> &in_grad) const override {
#if MXNET_USE_CUDNN == 1
                return {};
#else
                return {{in_data[pool_enum::kData], in_grad[pool_enum::kData]}};
#endif
            }

            Operator *CreateOperator(Context ctx) const override;

        private:
            VolumetricPoolingParam param_;
        };  // class PoolingProp
#endif  // DMLC_USE_CXX11
    }  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_POOLING_INL_H_

