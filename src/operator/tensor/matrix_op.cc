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
DMLC_REGISTER_PARAMETER(ClipParam);
DMLC_REGISTER_PARAMETER(SimpleCropAssignScalarParam);
DMLC_REGISTER_PARAMETER(SliceParam);
DMLC_REGISTER_PARAMETER(SliceAxisParam);
DMLC_REGISTER_PARAMETER(DotParam);
DMLC_REGISTER_PARAMETER(RepeatParam);
DMLC_REGISTER_PARAMETER(TileParam);
DMLC_REGISTER_PARAMETER(ReverseParam);

NNVM_REGISTER_OP(Reshape)
.add_alias("reshape")
.describe(R"code(Reshape array into a new shape.

The shape is a tuple of int such as (2,3,4). The new shape should not change the
array size. For example::

   reshape([1,2,3,4], shape=(2,2)) = [[1,2], [3,4]]

In addition, we can use special codes, which are integers less than
1, on some shape dimensions. To inference the output shape, we set it to an
empty tuple at beginning. When continuously pop dimensions from the original
shape starting from the beginning, and then push translated results into the output
shape.

Each special code presents a way of translation.

- ``0`` for copying one. Pop one input dimension and push into the output. For example::

  - input=(2,3,4), shape=(4,0,2), output=(4,3,2)
  - input=(2,3,4), shape=(2,0,0), output=(2,3,4)

- ``-1`` for inference. Push a placeholder into the output whose value will be inferred later::

  - input=(2,3,4), shape=(6,1,-1), output=(6,1,4)
  - input=(2,3,4), shape=(3,-1,8), output=(3,1,8)
  - input=(2,3,4), shape=(-1,), output=(24,)

- ``-2`` for copying all. Pop all remaining input dimensions and push them into
  the output::

  - input=(2,3,4), shape=(-2), output=(9,8,7)
  - input=(2,3,4), shape=(2,-2), output=(2,3,4)
  - input=(2,3,4), shape=(-2,1,1), output=(2,3,4,1,1)

- ``-3`` for merging two dimensions. Pop two input dimensions, compute the product and then
  push into the output::

  - input=(2,3,4), shape=(-3,4), output=(6,4)
  - input=(2,3,4), shape=(0,-3), output=(2,12)
  - input=(2,3,4), shape=(-3,-2), output=(6,4)

- ``-4`` for splitting two dimensions. Pop one input dimensions, next split it
  according to the next two dimensions (can contain one ``-1``) specified after
  this code, then push into the output::

  - input=(2,3,4), shape=(-4,1,2,-2), output=(1,2,3,4)
  - input=(2,3,4), shape=(2,-4,-1,3,-2), output=(2,1,3,4)

If the argument ``reverse`` is set to be true, then translating the input shape
from right to left. For example, with input shape (10, 5, 4) target shape (-1,
0), then the output shape will be (50,4) if ``reverse=1``, otherwise it will be
(40,5).

)code" ADD_FILELINE)
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
.add_argument("data", "ndarray-or-symbol", "Input data to reshape.")
.add_arguments(ReshapeParam::__FIELDS__());


NNVM_REGISTER_OP(Flatten)
.add_alias("flatten")
.describe(R"code(Flatten input into a 2-D array by collapsing the higher dimensions.

Assume the input array has shape ``(d1, d2, ..., dk)``, then ``flatten`` reshapes
the input array into shape ``(d1, d2*...*dk)``.

)code" ADD_FILELINE)
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
.add_argument("data", "ndarray-or-symbol", "Input data to reshape.");

NNVM_REGISTER_OP(transpose)
.describe(R"code(Permute the dimensions of an array.

Examples::

  x = [[ 1, 2],
       [ 3, 4]]

  transpose(x) = [[ 1.,  3.],
                  [ 2.,  4.]]

  x = [[[ 1.,  2.],
        [ 3.,  4.]],

       [[ 5.,  6.],
        [ 7.,  8.]]]

  transpose(x) = [[[ 1.,  5.],
                   [ 3.,  7.]],

                  [[ 2.,  6.],
                   [ 4.,  8.]]]

  transpose(x, axes=(1,0,2)) = [[[ 1.,  2.],
                                 [ 5.,  6.]],

                                [[ 3.,  4.],
                                 [ 7.,  8.]]]
)code" ADD_FILELINE)
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
.add_argument("data", "ndarray-or-symbol", "Source input")
.add_arguments(TransposeParam::__FIELDS__());


NNVM_REGISTER_OP(expand_dims)
.describe(R"code(Insert a new axis with size 1 into the array shape

For example, given ``x`` with shape ``(2,3,4)``, then ``expand_dims(x, axis=1)``
will return a new array with shape ``(2,1,3,4)``.

)code" ADD_FILELINE)
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
.add_argument("data", "ndarray-or-symbol", "Source input")
.add_arguments(ExpandDimParam::__FIELDS__());

NNVM_REGISTER_OP(slice)
.describe(R"code(Crop a continuous region from the array.

Assume the input array has *n* dimensions, given ``begin=(b_1, ..., b_n)`` and
``end=(e_1, ..., e_n)``, then ``crop`` will return a region with shape
``(e_1-b_1, ..., e_n-b_n)``. The result's *k*-th dimension contains elements
from the *k*-th dimension of the input array with the open range ``[b_k, e_k)``.

For example::

  x = [[  1.,   2.,   3.,   4.],
       [  5.,   6.,   7.,   8.],
       [  9.,  10.,  11.,  12.]]

  crop(x, begin=(0,1), end=(2,4)) = [[ 2.,  3.,  4.],
                                     [ 6.,  7.,  8.]]

)code" ADD_FILELINE)
.add_alias("crop")
.set_attr_parser(ParamParser<SliceParam>)
.set_attr<nnvm::FInferShape>("FInferShape", SliceShape)
.set_attr<nnvm::FInferType>("FInferType", ElemwiseType<1, 1>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseNone{"_backward_slice"})
.set_attr<FCompute>("FCompute<cpu>", Slice<cpu>)
.add_argument("data", "ndarray-or-symbol", "Source input")
.add_arguments(SliceParam::__FIELDS__());

NNVM_REGISTER_OP(_backward_slice)
.set_attr_parser(ParamParser<SliceParam>)
.set_attr<nnvm::TIsBackward>("TIsBackward", true)
.set_attr<FCompute>("FCompute<cpu>", SliceBackward<cpu>);

NNVM_REGISTER_OP(_slice_assign)
.add_alias("_crop_assign")
.MXNET_DESCRIBE("Assign the rhs to a cropped subset of lhs.\n\n"
"Requirements\n"
"------------\n"
"- output should be explicitly given and be the same as lhs.\n"
"- lhs and rhs are of the same data type, and on the same device.\n")
.set_num_inputs(2)
.set_num_outputs(1)
.set_attr<nnvm::FListInputNames>("FListInputNames",
  [](const NodeAttrs& attrs) {
    return std::vector<std::string>{"lhs", "rhs"};
  })
.set_attr_parser(ParamParser<SliceParam>)
.set_attr<nnvm::FInferShape>("FInferShape", SliceAssignShape)
.set_attr<nnvm::FInferType>("FInferType", ElemwiseType<2, 1>)
.set_attr<nnvm::FInplaceOption>("FInplaceOption",
  [](const NodeAttrs& attrs){
    return std::vector<std::pair<int, int> >{{0, 0}};
  })
.set_attr<FCompute>("FCompute<cpu>", SliceAssign<cpu>)
.add_argument("lhs", "ndarray-or-symbol", "Source input")
.add_argument("rhs", "ndarray-or-symbol", "value to assign")
.add_arguments(SliceParam::__FIELDS__());

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
.add_argument("data", "ndarray-or-symbol", "Source input")
.add_arguments(SimpleCropAssignScalarParam::__FIELDS__());

NNVM_REGISTER_OP(slice_axis)
.describe(R"code(Slice along a given axis.

Examples:

  x = [[  1.,   2.,   3.,   4.],
       [  5.,   6.,   7.,   8.],
       [  9.,  10.,  11.,  12.]]

  slice_axis(x, axis=0, begin=1, end=3) = [[  5.,   6.,   7.,   8.],
                                           [  9.,  10.,  11.,  12.]]

  slice_axis(x, axis=1, begin=0, end=2) = [[  1.,   2.],
                                           [  5.,   6.],
                                           [  9.,  10.]]

  slice_axis(x, axis=1, begin=-3, end=-1) = [[  2.,   3.],
                                             [  6.,   7.],
                                             [ 10.,  11.]]
)code" ADD_FILELINE)
.set_num_inputs(1)
.set_num_outputs(1)
.set_attr_parser(ParamParser<SliceAxisParam>)
.set_attr<nnvm::FInferShape>("FInferShape", SliceAxisShape)
.set_attr<nnvm::FInferType>("FInferType", ElemwiseType<1, 1>)
.set_attr<FCompute>("FCompute<cpu>", SliceAxis<cpu>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseNone{"_backward_slice_axis"})
.add_argument("data", "ndarray-or-symbol", "Source input")
.add_arguments(SliceAxisParam::__FIELDS__());

NNVM_REGISTER_OP(_backward_slice_axis)
.set_num_inputs(1)
.set_num_outputs(1)
.set_attr_parser(ParamParser<SliceAxisParam>)
.set_attr<nnvm::TIsBackward>("TIsBackward", true)
.set_attr<FCompute>("FCompute<cpu>", SliceAxisGrad_<cpu>);

NNVM_REGISTER_OP(dot)
.describe(R"doc(Dot product of two arrays.

``dot``'s behavior depends on the input array dimensions:

- 1-D arrays: inner product of vectors
- 2-D arrays: matrix multiplication
- N-D arrays: a sum product over the last axis of the first input and the first
  axis of the second input

  For example, given 3-D ``x`` with shape `(n,m,k)` and ``y`` with shape `(k,r,s)`, the
  result array will have shape `(n,m,r,s)`. It is computed by::

    dot(x,y)[i,j,a,b] = sum(x[i,j,:]*y[:,a,b])

)doc" ADD_FILELINE)
.set_num_inputs(2)
.set_num_outputs(1)
.set_attr_parser(ParamParser<DotParam>)
.set_attr<nnvm::FListInputNames>("FListInputNames",
  [](const NodeAttrs& attrs) {
    return std::vector<std::string>{"lhs", "rhs"};
  })
.set_attr<nnvm::FInferShape>("FInferShape", DotShape)
.set_attr<nnvm::FInferType>("FInferType", ElemwiseType<2, 1>)
.set_attr<FCompute>("FCompute<cpu>", DotForward_<cpu>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseIn{"_backward_dot"})
.add_argument("lhs", "ndarray-or-symbol", "The first input")
.add_argument("rhs", "ndarray-or-symbol", "The second input")
.add_arguments(DotParam::__FIELDS__());

NNVM_REGISTER_OP(_backward_dot)
.set_num_inputs(3)
.set_num_outputs(2)
.set_attr_parser(ParamParser<DotParam>)
.set_attr<nnvm::TIsBackward>("TIsBackward", true)
.set_attr<FCompute>("FCompute<cpu>", DotBackward_<cpu>)
.add_arguments(DotParam::__FIELDS__());

NNVM_REGISTER_OP(batch_dot)
.describe(R"doc(Batchwise dot product.

``batch_dot`` is used to compute dot product of ``x`` and ``y`` when ``x`` and
``y`` are data in batch, namely 3D arrays in shape of `(batch_size, :, :)`.

For example, given ``x`` with shape `(batch_size, n, m)` and ``y`` with shape
`(batch_size, m, k)`, the result array will have shape `(batch_size, n, k)`,
which is computed by::

   batch_dot(x,y)[i,:,:] = dot(x[i,:,:], y[i,:,:])

)doc" ADD_FILELINE)
.set_num_inputs(2)
.set_num_outputs(1)
.set_attr_parser(ParamParser<DotParam>)
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
.add_argument("lhs", "ndarray-or-symbol", "The first input")
.add_argument("rhs", "ndarray-or-symbol", "The second input")
.add_arguments(DotParam::__FIELDS__());

NNVM_REGISTER_OP(_backward_batch_dot)
.set_num_inputs(3)
.set_num_outputs(2)
.set_attr_parser(ParamParser<DotParam>)
.set_attr<FResourceRequest>("FResourceRequest",
  [](const NodeAttrs& attrs) {
    return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
  })
.set_attr<nnvm::TIsBackward>("TIsBackward", true)
.set_attr<FCompute>("FCompute<cpu>", BatchDotBackward_<cpu>);

NNVM_REGISTER_OP(clip)
.describe(R"code(Clip (limit) the values in an array, elementwise

Given an interval, values outside the interval are clipped to the interval
edges. That is::

   clip(x) = max(min(x, a_max)), a_min)

)code" ADD_FILELINE)
.set_num_inputs(1)
.set_num_outputs(1)
.set_attr_parser(ParamParser<ClipParam>)
.set_attr<nnvm::FInferShape>("FInferShape", ElemwiseShape<1, 1>)
.set_attr<nnvm::FInferType>("FInferType", ElemwiseType<1, 1>)
.set_attr<FCompute>("FCompute<cpu>", Clip<cpu>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseIn{ "_backward_clip" })
.add_argument("data", "ndarray-or-symbol", "Source input")
.add_arguments(ClipParam::__FIELDS__());

NNVM_REGISTER_OP(_backward_clip)
.set_num_inputs(1)
.set_num_outputs(1)
.set_attr_parser(ParamParser<ClipParam>)
.set_attr<nnvm::TIsBackward>("TIsBackward", true)
.set_attr<FCompute>("FCompute<cpu>", ClipGrad_<cpu>);

NNVM_REGISTER_OP(repeat)
.describe(R"code(Repeat elements of an array.

In default, ``repeat`` flatten the input array into 1-D and then repeat the
elements::

  x = [[ 1, 2],
       [ 3, 4]]

  repeat(x, repeats=2) = [ 1.,  1.,  2.,  2.,  3.,  3.,  4.,  4.]

We can also choose a particular axis to repeat, in which a negative axis is
interpreted counting from the backward::

  repeat(x, repeats=2, axis=1) = [[ 1.,  1.,  2.,  2.],
                                  [ 3.,  3.,  4.,  4.]]

  repeat(x, repeats=2, axis=-1) = [[ 1.,  2.],
                                   [ 1.,  2.],
                                   [ 3.,  4.],
                                   [ 3.,  4.]]
)code" ADD_FILELINE)
.set_num_outputs(1)
.set_num_inputs(1)
.set_attr_parser(ParamParser<RepeatParam>)
.set_attr<nnvm::FListInputNames>("FListInputNames",
  [](const NodeAttrs& attrs) {
    return std::vector<std::string>{"data"};
  })
.set_attr<nnvm::FInferShape>("FInferShape", RepeatOpShape)
.set_attr<nnvm::FInferType>("FInferType", RepeatOpType)
.set_attr<FCompute>("FCompute<cpu>", RepeatOpForward<cpu>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseNone{"_backward_repeat"})
.add_argument("data", "ndarray-or-symbol", "Input data array")
.add_arguments(RepeatParam::__FIELDS__());

NNVM_REGISTER_OP(_backward_repeat)
.set_num_inputs(1)
.set_num_outputs(1)
.set_attr_parser(ParamParser<RepeatParam>)
.set_attr<nnvm::TIsBackward>("TIsBackward", true)
.set_attr<FCompute>("FCompute<cpu>", RepeatOpBackward<cpu>);

NNVM_REGISTER_OP(tile)
.describe(R"code(Repeat the whole array by multiple times.

If ``reps`` has length *d*, and input array has dimension of *n*. There are
there cases:

- **n=d**. Repeat *i*-th dimension of the input by ``reps[i]`` times::

    x = [[1, 2],
         [3, 4]]

    tile(x, reps=(2,3)) = [[ 1.,  2.,  1.,  2.,  1.,  2.],
                           [ 3.,  4.,  3.,  4.,  3.,  4.],
                           [ 1.,  2.,  1.,  2.,  1.,  2.],
                           [ 3.,  4.,  3.,  4.,  3.,  4.]]

- **n>d**. ``reps`` is promoted to length *n* by pre-pending 1’s to it. Thus for
  an input shape ``(2,3)``, ``repos=(2,)`` is treated as ``(1,2)``::


    tile(x, reps=(2,)) = [[ 1.,  2.,  1.,  2.],
                          [ 3.,  4.,  3.,  4.]]

- **n<d**. The input is promoted to be d-dimensional by prepending new axes. So a
  shape ``(2,2)`` array is promoted to ``(1,2,2)`` for 3-D replication::

    tile(x, reps=(2,2,3)) = [[[ 1.,  2.,  1.,  2.,  1.,  2.],
                              [ 3.,  4.,  3.,  4.,  3.,  4.],
                              [ 1.,  2.,  1.,  2.,  1.,  2.],
                              [ 3.,  4.,  3.,  4.,  3.,  4.]],

                             [[ 1.,  2.,  1.,  2.,  1.,  2.],
                              [ 3.,  4.,  3.,  4.,  3.,  4.],
                              [ 1.,  2.,  1.,  2.,  1.,  2.],
                              [ 3.,  4.,  3.,  4.,  3.,  4.]]]
)code" ADD_FILELINE)
.set_num_outputs(1)
.set_num_inputs(1)
.set_attr_parser(ParamParser<TileParam>)
.set_attr<nnvm::FListInputNames>("FListInputNames",
  [](const NodeAttrs& attrs) {
    return std::vector<std::string>{"data"};
  })
.set_attr<nnvm::FInferShape>("FInferShape", TileOpShape)
.set_attr<nnvm::FInferType>("FInferType", TileOpType)
.set_attr<FCompute>("FCompute<cpu>", TileOpForward<cpu>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseNone{"_backward_tile"})
.add_argument("data", "ndarray-or-symbol", "Input data array")
.add_arguments(TileParam::__FIELDS__());

NNVM_REGISTER_OP(_backward_tile)
.set_num_inputs(1)
.set_num_outputs(1)
.set_attr_parser(ParamParser<TileParam>)
.set_attr<nnvm::TIsBackward>("TIsBackward", true)
.set_attr<FCompute>("FCompute<cpu>", TileOpBackward<cpu>);

NNVM_REGISTER_OP(reverse)
.MXNET_DESCRIBE("Reverse elements of an array with axis")
.set_num_outputs(1)
.set_num_inputs(1)
.add_alias("flip")
.set_attr_parser(ParamParser<ReverseParam>)
.set_attr<nnvm::FListInputNames>("FListInputNames",
[](const NodeAttrs& attrs) {
  return std::vector<std::string> {"data"};
})
.set_attr<FResourceRequest>("FResourceRequest",
[](const NodeAttrs& attrs) {
  return std::vector<ResourceRequest> {ResourceRequest::kTempSpace};
})
.set_attr<nnvm::FInferShape>("FInferShape", ElemwiseShape<1, 1>)
.set_attr<nnvm::FInferType>("FInferType", ElemwiseType<1, 1>)
.set_attr<FCompute>("FCompute<cpu>", ReverseOpForward<cpu>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseNone{ "_backward_reverse" })
.add_argument("data", "NDArray", "Input data array")
.add_arguments(ReverseParam::__FIELDS__());

NNVM_REGISTER_OP(_backward_reverse)
.set_num_inputs(1)
.set_num_outputs(1)
.set_attr_parser(ParamParser<ReverseParam>)
.set_attr<nnvm::TIsBackward>("TIsBackward", true)
.set_attr<FResourceRequest>("FResourceRequest",
[](const NodeAttrs& attrs) {
  return std::vector<ResourceRequest> {ResourceRequest::kTempSpace};
})
.set_attr<FCompute>("FCompute<cpu>", ReverseOpForward<cpu>);
}  // namespace op
}  // namespace mxnet
