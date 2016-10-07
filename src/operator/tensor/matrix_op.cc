/*!
 *  Copyright (c) 2015 by Contributors
 * \file matrix_op.cc
 * \brief CPU Implementation of matrix operations
 */
// this will be invoked by gcc and compile CPU version
#include "./matrix_op-inl.h"
#include "./elemwise_unary_op.h"

namespace mxnet {
namespace op {
DMLC_REGISTER_PARAMETER(TransposeParam);
DMLC_REGISTER_PARAMETER(ExpandDimParam);
DMLC_REGISTER_PARAMETER(SimpleCropParam);
DMLC_REGISTER_PARAMETER(SimpleCropAssignScalarParam);
DMLC_REGISTER_PARAMETER(SliceParam);
DMLC_REGISTER_PARAMETER(FlipParam);

NNVM_REGISTER_OP(transpose)
.MXNET_DESCRIBE("Transpose the input tensor and return a new one")
.set_num_inputs(1)
.set_num_outputs(1)
.set_attr_parser(ParamParser<TransposeParam>)
.set_attr<nnvm::FInferShape>("FInferShape", TransposeShape)
.set_attr<nnvm::FInferType>("FInferType", ElemwiseType<1, 1>)
.set_attr<nnvm::FGradient>("FGradient",
  [](const nnvm::NodePtr& n, const std::vector<nnvm::NodeEntry>& ograds) {
    const TransposeParam& param = nnvm::get<TransposeParam>(n->attrs.parsed);
    if (param.axes.ndim() == 0) {
      return MakeGradNode("transpose", n, ograds, {});
    } else {
      TShape axes = TShape(param.axes.ndim());
      for (index_t i = 0; i < axes.ndim(); ++i) {
        axes[param.axes[i]] = i;
      }
      std::string str_axes;
      std::stringstream ss(str_axes);
      ss << axes;
      return MakeGradNode("transpose", n, ograds, {{"axes", str_axes}});
    }
  })
.set_attr<FCompute>("FCompute<cpu>", Transpose<cpu>)
.add_argument("src", "NDArray", "Source input")
.add_arguments(TransposeParam::__FIELDS__());


NNVM_REGISTER_OP(expand_dims)
.MXNET_DESCRIBE("Expand the shape of array by inserting a new axis.")
.set_num_inputs(1)
.set_num_outputs(1)
.set_attr_parser(ParamParser<ExpandDimParam>)
.set_attr<nnvm::FInferShape>("FInferShape", ExpandDimShape)
.set_attr<nnvm::FInferType>("FInferType", ElemwiseType<1, 1>)
.set_attr<nnvm::FInplaceOption>("FInplaceOption",
  [](const NodeAttrs& attrs){
    return std::vector<std::pair<int, int> >{{0, 0}};
  })
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseNone{"_backward_copy"})
.set_attr<FCompute>("FCompute<cpu>", IdentityCompute<cpu>)
.add_argument("src", "NDArray", "Source input")
.add_arguments(ExpandDimParam::__FIELDS__());

NNVM_REGISTER_OP(crop)
.MXNET_DESCRIBE("(Crop the input tensor and return a new one.\n\n"
"Requirements\n"
"------------\n"
"- the input and output (if explicitly given) are of the same data type,\n"
"  and on the same device.\n"
")")
.set_num_inputs(1)
.set_num_outputs(1)
.set_attr_parser(ParamParser<SimpleCropParam>)
.set_attr<nnvm::FInferShape>("FInferShape", CropShape)
.set_attr<nnvm::FInferType>("FInferType", ElemwiseType<1, 1>)
.set_attr<FCompute>("FCompute<cpu>", Crop<cpu>)
.add_argument("src", "NDArray", "Source input")
.add_arguments(SimpleCropParam::__FIELDS__());

NNVM_REGISTER_OP(_crop_assign)
.MXNET_DESCRIBE("(Assign the rhs to a cropped subset of lhs.\n\n"
"Requirements\n"
"------------\n"
"- output should be explicitly given and be the same as lhs.\n"
"- lhs and rhs are of the same data type, and on the same device.\n"
")")
.set_num_inputs(2)
.set_num_outputs(1)
.set_attr_parser(ParamParser<SimpleCropParam>)
.set_attr<nnvm::FInferShape>("FInferShape", CropAssignShape)
.set_attr<nnvm::FInferType>("FInferType", ElemwiseType<2, 1>)
.set_attr<nnvm::FInplaceOption>("FInplaceOption",
  [](const NodeAttrs& attrs){
    return std::vector<std::pair<int, int> >{{0, 0}};
  })
.set_attr<FCompute>("FCompute<cpu>", CropAssign<cpu>)
.add_argument("src", "NDArray", "Source input")
.add_argument("val", "NDArray", "value to assign")
.add_arguments(SimpleCropParam::__FIELDS__());

NNVM_REGISTER_OP(_crop_assign_scalar)
.MXNET_DESCRIBE("(Assign the scalar to a cropped subset of the input.\n\n"
"Requirements\n"
"------------\n"
"- output should be explicitly given and be the same as input\n"
")")
.set_num_inputs(1)
.set_num_outputs(1)
.set_attr_parser(ParamParser<SimpleCropAssignScalarParam>)
.set_attr<nnvm::FInferShape>("FInferShape", CropAssignScalarShape)
.set_attr<nnvm::FInferType>("FInferType", ElemwiseType<1, 1>)
.set_attr<nnvm::FInplaceOption>("FInplaceOption",
  [](const NodeAttrs& attrs){
    return std::vector<std::pair<int, int> >{{0, 0}};
  })
.set_attr<FCompute>("FCompute<cpu>", CropAssignScalar<cpu>)
.add_argument("src", "NDArray", "Source input")
.add_arguments(SimpleCropAssignScalarParam::__FIELDS__());

NNVM_REGISTER_OP(slice_axis)
.MXNET_DESCRIBE("Slice the input along certain axis and return a sliced array.")
.set_num_inputs(1)
.set_num_outputs(1)
.set_attr_parser(ParamParser<SliceParam>)
.set_attr<nnvm::FInferShape>("FInferShape", SliceShape)
.set_attr<nnvm::FInferType>("FInferType", ElemwiseType<1, 1>)
.set_attr<FCompute>("FCompute<cpu>", Slice<cpu>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseNone{"_backward_slice_axis"})
.add_argument("src", "NDArray", "Source input")
.add_arguments(SliceParam::__FIELDS__());

NNVM_REGISTER_OP(_backward_slice_axis)
.set_num_inputs(1)
.set_num_outputs(1)
.set_attr_parser(ParamParser<SliceParam>)
.set_attr<nnvm::FBackwardOutToInIndex>("FBackwardOutToInIndex",
  [](const NodeAttrs& attrs) { return std::vector<uint32_t>{0}; })
.set_attr<nnvm::FBackwardInGradIndex>("FBackwardInGradIndex",
  [](const NodeAttrs& attrs) { return std::vector<uint32_t>{0}; })
.set_attr<FCompute>("FCompute<cpu>", SliceGrad_<cpu>);

NNVM_REGISTER_OP(flip)
.MXNET_DESCRIBE("Flip the input tensor along axis and return a new one.")
.set_num_inputs(1)
.set_num_outputs(1)
.set_attr_parser(ParamParser<FlipParam>)
.set_attr<nnvm::FInferShape>("FInferShape", ElemwiseShape<1, 1>)
.set_attr<nnvm::FInferType>("FInferType", ElemwiseType<1, 1>)
.set_attr<FCompute>("FCompute<cpu>", Flip<cpu>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseNone{"flip"})
.add_argument("src", "NDArray", "Source input")
.add_arguments(FlipParam::__FIELDS__());

NNVM_REGISTER_OP(dot)
.MXNET_DESCRIBE("Calculate dot product of two matrices or two vectors.")
.set_num_inputs(2)
.set_num_outputs(1)
.set_attr<nnvm::FInferShape>("FInferShape", DotShape)
.set_attr<nnvm::FInferType>("FInferType", ElemwiseType<2, 1>)
.set_attr<FCompute>("FCompute<cpu>", DotForward_<cpu>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseIn{"_backward_dot"})
.add_argument("lhs", "NDArray", "Left input")
.add_argument("rhs", "NDArray", "Right input")
.add_arguments(FlipParam::__FIELDS__());

NNVM_REGISTER_OP(_backward_dot)
.set_num_inputs(3)
.set_num_outputs(2)
.set_attr<nnvm::FBackwardOutToInIndex>("FBackwardOutToInIndex",
  [](const NodeAttrs& attrs) { return std::vector<uint32_t>{0, 1}; })
.set_attr<nnvm::FBackwardInGradIndex>("FBackwardInGradIndex",
  [](const NodeAttrs& attrs) { return std::vector<uint32_t>{0}; })
.set_attr<FCompute>("FCompute<cpu>", DotBackward_<cpu>);

NNVM_REGISTER_OP(batch_dot)
.MXNET_DESCRIBE("Calculate batched dot product of two matrices."
                " (batch, M, K) batch_dot (batch, K, N) --> (batch, M, N)")
.set_num_inputs(2)
.set_num_outputs(1)
.set_attr<nnvm::FInferShape>("FInferShape", BatchDotShape)
.set_attr<nnvm::FInferType>("FInferType", ElemwiseType<2, 1>)
.set_attr<FResourceRequest>("FResourceRequest",
  [](const NodeAttrs& attrs) {
    return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
  })
.set_attr<FCompute>("FCompute<cpu>", BatchDotForward_<cpu>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseIn{"_backward_batch_dot"})
.add_argument("lhs", "NDArray", "Left input")
.add_argument("rhs", "NDArray", "Right input")
.add_arguments(FlipParam::__FIELDS__());

NNVM_REGISTER_OP(_backward_batch_dot)
.set_num_inputs(3)
.set_num_outputs(2)
.set_attr<FResourceRequest>("FResourceRequest",
  [](const NodeAttrs& attrs) {
    return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
  })
.set_attr<nnvm::FBackwardOutToInIndex>("FBackwardOutToInIndex",
  [](const NodeAttrs& attrs) { return std::vector<uint32_t>{0, 1}; })
.set_attr<nnvm::FBackwardInGradIndex>("FBackwardInGradIndex",
  [](const NodeAttrs& attrs) { return std::vector<uint32_t>{0}; })
.set_attr<FCompute>("FCompute<cpu>", BatchDotBackward_<cpu>);

}  // namespace op
}  // namespace mxnet
