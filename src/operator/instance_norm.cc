/*!
 * Copyright (c) 2015 by Contributors
 * \file instance_norm.cc
 * \brief
 * \author Sebastian Bodenstein
*/

#include "./instance_norm-inl.h"

namespace mxnet {
namespace op {
template <>
Operator* CreateOp<cpu>(InstanceNormParam param, int dtype) {
  return new InstanceNormOp<cpu>(param);
}

// DO_BIND_DISPATCH comes from operator_common.h
Operator* InstanceNormProp::CreateOperatorEx(Context ctx,
                                             std::vector<TShape>* in_shape,
                                             std::vector<int>* in_type) const {
  std::vector<TShape> out_shape, aux_shape;
  std::vector<int> out_type, aux_type;
  CHECK(InferType(in_type, &out_type, &aux_type));
  CHECK(InferShape(in_shape, &out_shape, &aux_shape));
  DO_BIND_DISPATCH(CreateOp, param_, (*in_type)[0]);
}

DMLC_REGISTER_PARAMETER(InstanceNormParam);

MXNET_REGISTER_OP_PROPERTY(InstanceNorm, InstanceNormProp)
    .add_argument("data", "NDArray-or-Symbol",
                  "A n-dimensional tensor (n > 2) of the form [batch, "
                  "channel, spatial_dim1, spatial_dim2, ...].")
    .add_argument("gamma", "NDArray-or-Symbol",
                  "A vector of length \'channel\', which multiplies the "
                  "normalized input.")
    .add_argument("beta", "NDArray-or-Symbol",
                  "A vector of length \'channel\', which is added to the "
                  "product of the normalized input and the weight.")
    .add_arguments(InstanceNormParam::__FIELDS__())
    .describe(
        "An operator taking in a n-dimensional input tensor (n > 2), and "
        "normalizing the input by subtracting the mean and variance calculated "
        "over the spatial dimensions. This is an implemention of "
        "the operator described in \"Instance Normalization: The "
        "Missing Ingredient for Fast Stylization\", D. Ulyanov, A. Vedaldi, V. "
        "Lempitsky, 2016 (arXiv:1607.08022v2). This layer is similar to batch "
        "normalization, with two differences: first, the normalization is "
        "carried out per example (\'instance\'), not over a batch. Second, the "
        "same normalization is applied both at test and train time. This "
        "operation is also known as \'contrast normalization\'.");

}  // namespace op
}  // namespace mxnet
