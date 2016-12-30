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
DMLC_REGISTER_PARAMETER(ReshapeParam);
DMLC_REGISTER_PARAMETER(TransposeParam);
DMLC_REGISTER_PARAMETER(ExpandDimParam);
DMLC_REGISTER_PARAMETER(SimpleCropParam);
DMLC_REGISTER_PARAMETER(SimpleCropAssignScalarParam);
DMLC_REGISTER_PARAMETER(SliceParam);
DMLC_REGISTER_PARAMETER(FlipParam);

NNVM_REGISTER_OP(Reshape)
.MXNET_DESCRIBE("Reshape input according to a target shape spec.\n"
"The target shape is a tuple and can be a simple list of dimensions "
"such as (12,3) or it can incorporate special codes that correspond "
"to contextual operations that refer to the input dimensions.\n"
"The special codes are all expressed as integers less than 1. These "
"codes effectively refer to a machine that pops input dims off the "
"beginning of the input dims list and pushes resulting output dims "
"onto the end of the output dims list, which starts empty. The codes "
"are:\n"
"  0  Copy     Pop one input dim and push it onto the output dims\n"
" -1  Infer    Push a dim that is inferred later from all other output dims\n"
" -2  CopyAll  Pop all remaining input dims and push them onto output dims\n"
" -3  Merge2   Pop two input dims, multiply them, and push result\n"
" -4  Split2   Pop one input dim, and read two next target shape specs,\n"
"              push them both onto output dims (either can be -1 and will\n"
"              be inferred from the other\n"
" The exact mathematical behavior of these codes is given in the "
"description of the 'shape' parameter. All non-codes (positive "
"integers) just pop a dim off the input dims (if any), throw it away, "
"and then push the specified integer onto the output dims.\n"
"Examples:\n"
"Type     Input      Target            Output\n"
"Copy     (2,3,4)    (4,0,2)           (4,3,2)\n"
"Copy     (2,3,4)    (2,0,0)           (2,3,4)\n"
"Infer    (2,3,4)    (6,1,-1)          (6,1,4)\n"
"Infer    (2,3,4)    (3,-1,8)          (3,1,8)\n"
"CopyAll  (9,8,7)    (-2)              (9,8,7)\n"
"CopyAll  (9,8,7)    (9,-2)            (9,8,7)\n"
"CopyAll  (9,8,7)    (-2,1,1)          (9,8,7,1,1)\n"
"Merge2   (3,4)      (-3)              (12)\n"
"Merge2   (3,4,5)    (-3,0)            (12,5)\n"
"Merge2   (3,4,5)    (0,-3)            (3,20)\n"
"Merge2   (3,4,5,6)  (-3,0,0)          (12,5,6)\n"
"Merge2   (3,4,5,6)  (-3,-2)           (12,5,6)\n"
"Split2   (12)       (-4,6,2)          (6,2)\n"
"Split2   (12)       (-4,2,6)          (2,6)\n"
"Split2   (12)       (-4,-1,6)         (2,6)\n"
"Split2   (12,9)     (-4,2,6,0)        (2,6,9)\n"
"Split2   (12,9,9,9) (-4,2,6,-2)       (2,6,9,9,9)\n"
"Split2   (12,12)    (-4,2,-1,-4,-1,2) (2,6,6,2)\n")
.set_num_inputs(1)
.set_num_outputs(1)
.set_attr_parser(ParamParser<ReshapeParam>)
.set_attr<nnvm::FInferShape>("FInferShape", ReshapeShape)
.set_attr<nnvm::FInferType>("FInferType", ElemwiseType<1, 1>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseNone{"_backward_copy"})
.set_attr<FCompute>("FCompute<cpu>", IdentityCompute<cpu>)
.set_attr<nnvm::FInplaceOption>("FInplaceOption",
  [](const NodeAttrs& attrs) {
    return std::vector<std::pair<int, int> >{{0, 0}};
})
.add_argument("data", "NDArray", "Input data to reshape.")
.add_arguments(ReshapeParam::__FIELDS__());


NNVM_REGISTER_OP(Flatten)
.describe(R"(Flatten input into 2D by collapsing all the higher dimensions.
A (d1, d2, ..., dK) tensor is flatten to (d1, d2* ... *dK) matrix.)")
.set_num_inputs(1)
.set_num_outputs(1)
.set_attr<nnvm::FInferShape>("FInferShape", FlattenShape)
.set_attr<nnvm::FInferType>("FInferType", ElemwiseType<1, 1>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseNone{ "_backward_copy" })
.set_attr<FCompute>("FCompute<cpu>", IdentityCompute<cpu>)
.set_attr<nnvm::FInplaceOption>("FInplaceOption",
  [](const NodeAttrs& attrs) {
  return std::vector<std::pair<int, int> >{{0, 0}};
})
.add_argument("data", "NDArray", "Input data to reshape.");

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
      return MakeGradNode("transpose", n, ograds,
                          std::unordered_map<std::string, std::string>());
    } else {
      TShape axes = TShape(param.axes.ndim());
      for (index_t i = 0; i < axes.ndim(); ++i) {
        axes[param.axes[i]] = i;
      }
      std::ostringstream os;
      os << axes;
      return MakeGradNode("transpose", n, ograds, {{"axes", os.str()}});
    }
  })
.set_attr<FCompute>("FCompute<cpu>", Transpose<cpu>)
.add_argument("data", "NDArray", "Source input")
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
.add_argument("data", "NDArray", "Source input")
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
.add_argument("data", "NDArray", "Source input")
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
.set_attr<nnvm::FListInputNames>("FListInputNames",
  [](const NodeAttrs& attrs) {
    return std::vector<std::string>{"lhs", "rhs"};
  })
.set_attr_parser(ParamParser<SimpleCropParam>)
.set_attr<nnvm::FInferShape>("FInferShape", CropAssignShape)
.set_attr<nnvm::FInferType>("FInferType", ElemwiseType<2, 1>)
.set_attr<nnvm::FInplaceOption>("FInplaceOption",
  [](const NodeAttrs& attrs){
    return std::vector<std::pair<int, int> >{{0, 0}};
  })
.set_attr<FCompute>("FCompute<cpu>", CropAssign<cpu>)
.add_argument("lhs", "NDArray", "Source input")
.add_argument("rhs", "NDArray", "value to assign")
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
.add_argument("data", "NDArray", "Source input")
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
.add_argument("data", "NDArray", "Source input")
.add_arguments(SliceParam::__FIELDS__());

NNVM_REGISTER_OP(_backward_slice_axis)
.set_num_inputs(1)
.set_num_outputs(1)
.set_attr_parser(ParamParser<SliceParam>)
.set_attr<nnvm::TIsBackward>("TIsBackward", true)
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
.add_argument("data", "NDArray", "Source input")
.add_arguments(FlipParam::__FIELDS__());

NNVM_REGISTER_OP(dot)
.MXNET_DESCRIBE("Calculate dot product of two matrices or two vectors.")
.set_num_inputs(2)
.set_num_outputs(1)
.set_attr<nnvm::FListInputNames>("FListInputNames",
  [](const NodeAttrs& attrs) {
    return std::vector<std::string>{"lhs", "rhs"};
  })
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
.set_attr<nnvm::TIsBackward>("TIsBackward", true)
.set_attr<FCompute>("FCompute<cpu>", DotBackward_<cpu>);

NNVM_REGISTER_OP(batch_dot)
.MXNET_DESCRIBE("Calculate batched dot product of two matrices."
                " (batch, M, K) batch_dot (batch, K, N) --> (batch, M, N)")
.set_num_inputs(2)
.set_num_outputs(1)
.set_attr<nnvm::FListInputNames>("FListInputNames",
  [](const NodeAttrs& attrs) {
    return std::vector<std::string>{"lhs", "rhs"};
  })
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
.set_attr<nnvm::TIsBackward>("TIsBackward", true)
.set_attr<FCompute>("FCompute<cpu>", BatchDotBackward_<cpu>);

}  // namespace op
}  // namespace mxnet
