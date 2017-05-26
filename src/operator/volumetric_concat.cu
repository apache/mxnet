/*!
 * Copyright (c) 2015 by Contributors
 * \file concat.cu
 * \brief
 * \author Bing Xu
*/

#include "./volumetric_concat-inl.h"

namespace mxnet {
    namespace op {
        template<>
        Operator *CreateOp<gpu>(VolumetricConcatParam param) {
            return new VolumetricConcatOp<gpu>(param);
        }

    }  // namespace op
}  // namespace mxnet

