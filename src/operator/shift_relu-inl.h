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
 * \file shift_relu-inl.h
 * \brief shift relu family operator
 * \author Kang Liang
*/
#ifndef MXNET_OPERATOR_SHIFT_RELU_INL_H_
#define MXNET_OPERATOR_SHIFT_RELU_INL_H_

#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <mxnet/operator.h>
#include <algorithm>
#include <cstring>
#include <map>
#include <string>
#include <utility>
#include <vector>
#include "./mshadow_op.h"
#include "./operator_common.h"

namespace mxnet {
    namespace op {

        namespace shiftrelu {
            enum ShiftReLUOpInputs {
                kData
            };
            enum ShiftReLUOpOutputs {
                kOut, kShift
            };
            enum ShiftReLUOpSumType {
                kSum_v1, kSum_v2
            };
            enum ShiftReLUOpActType {
                kAct_v1, kAct_v2
            };
            enum ShiftReLUOpForwardResource {
                kRandom
            };
        }  // namespace shiftrelu

        struct ShiftReLUParam : public dmlc::Parameter<ShiftReLUParam> {
            // use int for enumeration
            int sum_type;
            int act_type;
            float shift;
            float range;
            float slope;

            DMLC_DECLARE_PARAMETER(ShiftReLUParam) {
                DMLC_DECLARE_FIELD(sum_type)
                        .add_enum("sum_v1", shiftrelu::kSum_v1)
                        .add_enum("sum_v2", shiftrelu::kSum_v2)
                        .set_default(shiftrelu::kSum_v1)
                        .describe("Activation function to be applied.");
                DMLC_DECLARE_FIELD(act_type)
                        .add_enum("act_v1", shiftrelu::kAct_v1)
                        .add_enum("act_v2", shiftrelu::kAct_v2)
                        .set_default(shiftrelu::kAct_v1)
                        .describe("Activation function to be applied.");
                DMLC_DECLARE_FIELD(slope)
                        .set_default(0.0f)
                        .describe("Init slope for the activation.");
                DMLC_DECLARE_FIELD(range).set_default(0.0f).describe("range of scale.");
                DMLC_DECLARE_FIELD(shift).set_default(0.0f).describe("shift the value.");
            }
        };

        struct srelu {
            template<typename DType>
            MSHADOW_XINLINE static DType Map(DType a, DType b, DType c) {
                return DType(a > b ? a : DType((a - b) * c + b));
            }
        };

        struct srelu_grad {
            template<typename DType>
            MSHADOW_XINLINE static DType Map(DType a, DType b, DType c) {
                return DType(a > b ? DType(1) : c);
            }
        };


        template<typename xpu, typename DType>
        class ShiftReLUOp : public Operator {
        public:
            explicit ShiftReLUOp(ShiftReLUParam param) { param_ = param; }

            virtual void Forward(const OpContext &ctx, const std::vector<TBlob> &in_data,
                                 const std::vector<OpReqType> &req,
                                 const std::vector<TBlob> &out_data,
                                 const std::vector<TBlob> &aux_args) {
                using namespace mshadow;
                using namespace mshadow::expr;
                CHECK_EQ(in_data.size(), 1U);
                CHECK_EQ(out_data.size(), 2U);
                Stream<xpu> *s = ctx.get_stream<xpu>();
                Tensor<xpu, 2, DType> data;
                Tensor<xpu, 2, DType> out;
                Tensor<xpu, 1, DType> shift;
                LayerSetUp(in_data[shiftrelu::kData].shape_);
                data = in_data[shiftrelu::kData].get_with_shape<xpu, 2, DType>(dshape, s);
                out = out_data[shiftrelu::kOut].get_with_shape<xpu, 2, DType>(dshape, s);
                shift = out_data[shiftrelu::kShift].get<xpu, 1, DType>(s);
                if (ctx.is_train && param_.range > 0.0f) {
                    shift = sumall_except_dim<0>(F<mshadow_op::abs>(data));
                    shift /= (dshape.Size() / shift.shape_.Size());
                    Random<xpu> *prnd =
                            ctx.requested[shiftrelu::kRandom].get_random<xpu, real_t>(s);
                    shift *= tcast<DType>(prnd->uniform(shift.shape_) - 0.5f);
                    shift *= DType(param_.range * 2.0f);
                    shift += DType(param_.shift);
                } else {
                    Assign(shift, req[shiftrelu::kShift],
                           F<mshadow_op::left>(DType(param_.shift), shift));
                }
                switch (param_.act_type) {
                    case shiftrelu::kAct_v1: {
                        // base shift relu
                        Assign(out, req[shiftrelu::kOut], F<srelu>(data, broadcast<0>(shift, data.shape_),
                                                                   F<mshadow_op::left>(DType(param_.slope), data)));
                        break;
                    }
                    case shiftrelu::kAct_v2: {
                        Assign(out, req[shiftrelu::kOut], F<mshadow_op::relu>(data + broadcast<0>(shift, data.shape_)));
                        break;
                    }
                    default:
                        LOG(FATAL) << "Not implmented";
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
                CHECK_EQ(in_data.size(), 1U);
                CHECK_EQ(out_data.size(), 2U);
                Stream<xpu> *s = ctx.get_stream<xpu>();
                Tensor<xpu, 2, DType> data;
                Tensor<xpu, 2, DType> out;
                Tensor<xpu, 2, DType> grad;
                Tensor<xpu, 1, DType> shift;
                data = in_data[shiftrelu::kData].get_with_shape<xpu, 2, DType>(dshape, s);
                out = in_grad[shiftrelu::kData].get_with_shape<xpu, 2, DType>(dshape, s);
                grad = out_grad[shiftrelu::kOut].get_with_shape<xpu, 2, DType>(dshape, s);
                shift = out_data[shiftrelu::kShift].get<xpu, 1, DType>(s);
                switch (param_.act_type) {
                    case shiftrelu::kAct_v1: {
                        // base shift relu
                        Assign(out, req[shiftrelu::kData], F<srelu_grad>(data, broadcast<0>(shift, data.shape_),
                                                                         F<mshadow_op::left>(DType(param_.slope),
                                                                                             data)) * grad);
                        break;
                    }
                    case shiftrelu::kAct_v2: {
                        Assign(out, req[shiftrelu::kData],
                               F<mshadow_op::relu_grad>(data + broadcast<0>(shift, data.shape_)) * grad);
                        break;
                    }
                    default:
                        LOG(FATAL) << "Not implmented";
                }
            }

        private:
            void LayerSetUp(const TShape &shape) {
                int n = shape[0];
                int k = shape[1];
                int d = shape.Size() / n / k;
                switch (param_.sum_type) {
                    case shiftrelu::kSum_v1: {
                        dshape[0] = n;
                        dshape[1] = k * d;
                        break;
                    }
                    case shiftrelu::kSum_v2: {
                        dshape[0] = n * k;
                        dshape[1] = d;
                        break;
                    }
                    default:
                        LOG(FATAL) << "Not implmented";
                }
            }

            mshadow::Shape<2> dshape;
            ShiftReLUParam param_;
        };  // class ShiftReLUOp

        template<typename xpu>
        Operator *CreateOp(ShiftReLUParam param, int dtype, const TShape &shape);

#if DMLC_USE_CXX11

        class ShiftReLUProp : public OperatorProperty {
        public:
            void Init(
                    const std::vector<std::pair<std::string, std::string>> &kwargs) override {
                param_.Init(kwargs);
            }

            std::map<std::string, std::string> GetParams() const override {
                return param_.__DICT__();
            }

            bool InferShape(std::vector<TShape> *in_shape, std::vector<TShape> *out_shape,
                            std::vector<TShape> *aux_shape) const override {
                using namespace mshadow;
                CHECK_EQ(in_shape->size(), 1U) << "Input:[data]";
                const TShape &dshape = in_shape->at(shiftrelu::kData);
                if (dshape.ndim() == 0) return false;
                out_shape->clear();
                out_shape->push_back(dshape);
                switch (param_.sum_type) {
                    case shiftrelu::kSum_v1: {
                        out_shape->push_back(Shape1(dshape[0]));
                        break;
                    }
                    case shiftrelu::kSum_v2: {
                        out_shape->push_back(Shape1(dshape[0] * dshape[1]));
                        break;
                    }
                    default: {
                        LOG(FATAL) << "Not implmented";
                        return false;
                    }
                }
                return true;
            }

            bool InferType(std::vector<int> *in_type, std::vector<int> *out_type,
                           std::vector<int> *aux_type) const override {
                CHECK_EQ(in_type->size(), 1U);
                int dtype = in_type->at(0);
                if (dtype == -1) {
                    LOG(FATAL) << "input type to shiftrelu is not specified.";
                    return false;
                }
                size_t nout = this->ListOutputs().size();
                out_type->clear();
                for (size_t i = 0; i < nout; ++i) out_type->push_back(dtype);
                return true;
            }

            OperatorProperty *Copy() const override {
                auto ptr = new ShiftReLUProp();
                ptr->param_ = param_;
                return ptr;
            }

            std::string TypeString() const override { return "ShiftReLU"; }

            // decalre dependency and inplace optimization options
            std::vector<int> DeclareBackwardDependency(
                    const std::vector<int> &out_grad, const std::vector<int> &in_data,
                    const std::vector<int> &out_data) const override {
                return {out_grad[shiftrelu::kOut], out_data[shiftrelu::kShift], in_data[shiftrelu::kData]};
            }

            std::vector<std::pair<int, void *>> BackwardInplaceOption(
                    const std::vector<int> &out_grad, const std::vector<int> &in_data,
                    const std::vector<int> &out_data,
                    const std::vector<void *> &in_grad) const override {
                return {{out_grad[shiftrelu::kOut], in_grad[shiftrelu::kData]}};
            }

            std::vector<std::pair<int, void *>> ForwardInplaceOption(
                    const std::vector<int> &in_data,
                    const std::vector<void *> &out_data) const override {
                return {{in_data[shiftrelu::kData], out_data[shiftrelu::kOut]}};
            }

            std::vector<ResourceRequest> ForwardResource(
                    const std::vector<TShape> &in_shape) const override {
                return {ResourceRequest::kRandom};
            }

            std::vector<std::string> ListArguments() const override { return {"data"}; }

            std::vector<std::string> ListOutputs() const override {
                return {"output", "shift"};
            }

            int NumOutputs() const override { return 2; }

            int NumVisibleOutputs() const override { return 1; }

            Operator *CreateOperator(Context ctx) const override {
                LOG(FATAL) << "Not Implemented.";
                return NULL;
            }

            Operator *CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
                                       std::vector<int> *in_type) const override;

        private:
            ShiftReLUParam param_;
        };

#endif  // DMLC_USE_CXX11
    }  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_SHIFT_RELU_INL_H_
