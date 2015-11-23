/*!
 * Copyright (c) 2015 by Contributors
 * \file pooling.cc
 * \brief
 * \author Bing Xu
*/
#include "./volumetric_pooling-inl.h"

namespace mxnet {
    namespace op {
        template<>
        Operator *CreateOp<cpu>(VolumetricPoolingParam param) {
            switch (param.pool_type) {
                case pool_enum::kMaxPooling:
                    return new VolumetricPoolingOp<cpu, mshadow::red::maximum>(param);
                case pool_enum::kAvgPooling:
                    return new VolumetricPoolingOp<cpu, mshadow::red::sum>(param);
                case pool_enum::kSumPooling:
                    return new VolumetricPoolingOp<cpu, mshadow::red::sum>(param);
                default:
                    LOG(FATAL) << "unknown activation type";
                    return NULL;
            }
        }

        Operator *VolumetricPoolingProp::CreateOperator(Context ctx) const {
            DO_BIND_DISPATCH(CreateOp, param_);
        }

        DMLC_REGISTER_PARAMETER(VolumetricPoolingParam);

        MXNET_REGISTER_OP_PROPERTY(VolumetricPooling, VolumetricPoolingProp)
        .describe("Perform volumetric pooling on inputs.")
        .add_argument("data", "Symbol", "Input data to the pooling operator.")
        .

        add_arguments(VolumetricPoolingParam::__FIELDS__());

    }  // namespace op
}  // namespace mxnet

