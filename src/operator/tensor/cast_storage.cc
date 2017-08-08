/*!
 *  Copyright (c) 2017 by Contributors
 * \file cast_storage.cc
 * \brief CPU Implementation of cast_storage operator.
 */

#include "./cast_storage-inl.h"
#include "../elemwise_op_common.h"
#include "../tensor/elemwise_unary_op.h"

namespace mxnet {
namespace op {

DMLC_REGISTER_PARAMETER(CastStorageParam);
NNVM_REGISTER_OP(cast_storage)
.describe(R"code(Casts tensor storage type to the new type.
)code" ADD_FILELINE)
.set_num_inputs(1)
.set_num_outputs(1)
.set_attr_parser(ParamParser<CastStorageParam>)
.set_attr<nnvm::FInferShape>("FInferShape", ElemwiseShape<1, 1>)
.set_attr<nnvm::FInferType>("FInferType", ElemwiseType<1, 1>)
.set_attr<FInferStorageType>("FInferStorageType", CastStorageInferStorageType)
.set_attr<FResourceRequest>("FResourceRequest",
  [](const NodeAttrs& attrs) {
    return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
  })
.set_attr<FCompute>("FCompute<cpu>", IdentityCompute<cpu>)
.set_attr<FComputeEx>("FComputeEx<cpu>", CastStorageComputeEx<cpu>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseNone{"_copy"})
.add_argument("data", "NDArray-or-Symbol", "The input.")
.add_arguments(CastStorageParam::__FIELDS__());

}  // namespace op
}  // namespace mxnet
