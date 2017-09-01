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
DMLC_REGISTER_PARAMETER(RepeatParam);
DMLC_REGISTER_PARAMETER(TileParam);
DMLC_REGISTER_PARAMETER(ReverseParam);
DMLC_REGISTER_PARAMETER(StackParam);

NNVM_REGISTER_OP(Reshape)
.add_alias("reshape")
.describe(R"code(Reshapes the input array.

.. note:: ``Reshape`` is deprecated, use ``reshape``

Given an array and a shape, this function returns a copy of the array in the new shape.
The shape is a tuple of integers such as (2,3,4).The size of the new shape should be same as the size of the input array.

Example::

  reshape([1,2,3,4], shape=(2,2)) = [[1,2], [3,4]]

Some dimensions of the shape can take special values from the set {0, -1, -2, -3, -4}. The significance of each is explained below:

- ``0``  copy this dimension from the input to the output shape.

  Example::

  - input shape = (2,3,4), shape = (4,0,2), output shape = (4,3,2)
  - input shape = (2,3,4), shape = (2,0,0), output shape = (2,3,4)

- ``-1`` infers the dimension of the output shape by using the remainder of the input dimensions
  keeping the size of the new array same as that of the input array.
  At most one dimension of shape can be -1.

  Example::

  - input shape = (2,3,4), shape = (6,1,-1), output shape = (6,1,4)
  - input shape = (2,3,4), shape = (3,-1,8), output shape = (3,1,8)
  - input shape = (2,3,4), shape=(-1,), output shape = (24,)

- ``-2`` copy all/remainder of the input dimensions to the output shape.

  Example::

  - input shape = (2,3,4), shape = (-2,), output shape = (2,3,4)
  - input shape = (2,3,4), shape = (2,-2), output shape = (2,3,4)
  - input shape = (2,3,4), shape = (-2,1,1), output shape = (2,3,4,1,1)

- ``-3`` use the product of two consecutive dimensions of the input shape as the output dimension.

  Example::

  - input shape = (2,3,4), shape = (-3,4), output shape = (6,4)
  - input shape = (2,3,4,5), shape = (-3,-3), output shape = (6,20)
  - input shape = (2,3,4), shape = (0,-3), output shape = (2,12)
  - input shape = (2,3,4), shape = (-3,-2), output shape = (6,4)

- ``-4`` split one dimension of the input into two dimensions passed subsequent to -4 in shape (can contain -1).

  Example::

  - input shape = (2,3,4), shape = (-4,1,2,-2), output shape =(1,2,3,4)
  - input shape = (2,3,4), shape = (2,-4,-1,3,-2), output shape = (2,1,3,4)

If the argument `reverse` is set to 1, then the special values are inferred from right to left.

  Example::

  - without reverse=1, for input shape = (10,5,4), shape = (-1,0), output shape would be (40,5)
  - with reverse=1, output shape will be (50,4).

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
.set_attr<nnvm::FInplaceIdentity>("FInplaceIdentity",
  [](const NodeAttrs& attrs){
    return std::vector<bool>{true};
  })
.add_argument("data", "NDArray-or-Symbol", "Input data to reshape.")
.add_arguments(ReshapeParam::__FIELDS__());


NNVM_REGISTER_OP(Flatten)
.add_alias("flatten")
.describe(R"code(Flattens the input array into a 2-D array by collapsing the higher dimensions.

.. note:: `Flatten` is deprecated. Use `flatten` instead.

For an input array with shape ``(d1, d2, ..., dk)``, `flatten` operation reshapes
the input array into an output array of shape ``(d1, d2*...*dk)``.

Example::

    x = [[
        [1,2,3],
        [4,5,6],
        [7,8,9]
    ],
    [    [1,2,3],
        [4,5,6],
        [7,8,9]
    ]],

    flatten(x) = [[ 1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9.],
       [ 1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9.]]

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
.set_attr<nnvm::FInplaceIdentity>("FInplaceIdentity",
  [](const NodeAttrs& attrs){
    return std::vector<bool>{true};
  })
.add_argument("data", "NDArray-or-Symbol", "Input array.");

NNVM_REGISTER_OP(transpose)
.describe(R"code(Permutes the dimensions of an array.

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
      return MakeNonlossGradNode(
          "transpose", n, ograds, {},
          std::unordered_map<std::string, std::string>());
    } else {
      TShape axes = TShape(param.axes.ndim());
      for (index_t i = 0; i < axes.ndim(); ++i) {
        axes[param.axes[i]] = i;
      }
      std::ostringstream os;
      os << axes;
      return MakeNonlossGradNode(
          "transpose", n, ograds,
          {}, {{"axes", os.str()}});
    }
  })
.set_attr<FCompute>("FCompute<cpu>", Transpose<cpu>)
.add_argument("data", "NDArray-or-Symbol", "Source input")
.add_arguments(TransposeParam::__FIELDS__());


NNVM_REGISTER_OP(expand_dims)
.describe(R"code(Inserts a new axis of size 1 into the array shape

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
.set_attr<nnvm::FInplaceIdentity>("FInplaceIdentity",
  [](const NodeAttrs& attrs){
    return std::vector<bool>{true};
  })
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseNone{"_backward_copy"})
.set_attr<FCompute>("FCompute<cpu>", IdentityCompute<cpu>)
.add_argument("data", "NDArray-or-Symbol", "Source input")
.add_arguments(ExpandDimParam::__FIELDS__());

NNVM_REGISTER_OP(slice)
.add_alias("_sparse_slice")
.add_alias("crop")
.describe(R"code(Slices a contiguous region of the array.

.. note:: ``crop`` is deprecated. Use ``slice`` instead.

This function returns a sliced continuous region of the array between the indices given
by `begin` and `end`.

For an input array of `n` dimensions, slice operation with ``begin=(b_0, b_1...b_n-1)`` indices
and ``end=(e_1, e_2, ... e_n)`` indices will result in an array with the shape
``(e_1-b_0, ..., e_n-b_n-1)``.

The resulting array's *k*-th dimension contains elements
from the *k*-th dimension of the input array with the open range ``[b_k, e_k)``.

For an input array of non-default storage type(e.g. `csr` or `row_sparse`), it only supports
slicing on the first dimension.

Example::

  x = [[  1.,   2.,   3.,   4.],
       [  5.,   6.,   7.,   8.],
       [  9.,  10.,  11.,  12.]]

  slice(x, begin=(0,1), end=(2,4)) = [[ 2.,  3.,  4.],
                                     [ 6.,  7.,  8.]]

)code" ADD_FILELINE)
.set_attr_parser(ParamParser<SliceParam>)
.set_attr<nnvm::FInferShape>("FInferShape", SliceShape)
.set_attr<nnvm::FInferType>("FInferType", ElemwiseType<1, 1>)
.set_attr<FInferStorageType>("FInferStorageType", ElemwiseStorageType<1, 1>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseNone{"_backward_slice"})
.set_attr<FCompute>("FCompute<cpu>", Slice<cpu>)
.set_attr<FComputeEx>("FComputeEx<cpu>", SliceEx<cpu>)
.add_argument("data", "NDArray-or-Symbol", "Source input")
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
.add_argument("lhs", "NDArray-or-Symbol", "Source input")
.add_argument("rhs", "NDArray-or-Symbol", "value to assign")
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
.add_argument("data", "NDArray-or-Symbol", "Source input")
.add_arguments(SimpleCropAssignScalarParam::__FIELDS__());

NNVM_REGISTER_OP(slice_axis)
.describe(R"code(Slices along a given axis.

Returns an array slice along a given `axis` starting from the `begin` index
to the `end` index.

Examples::

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
.add_argument("data", "NDArray-or-Symbol", "Source input")
.add_arguments(SliceAxisParam::__FIELDS__());

NNVM_REGISTER_OP(_backward_slice_axis)
.set_num_inputs(1)
.set_num_outputs(1)
.set_attr_parser(ParamParser<SliceAxisParam>)
.set_attr<nnvm::TIsBackward>("TIsBackward", true)
.set_attr<FCompute>("FCompute<cpu>", SliceAxisGrad_<cpu>);

NNVM_REGISTER_OP(clip)
.describe(R"code(Clips (limits) the values in an array.

Given an interval, values outside the interval are clipped to the interval edges.
Clipping ``x`` between `a_min` and `a_x` would be::

   clip(x, a_min, a_max) = max(min(x, a_max), a_min))

Example::

    x = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    clip(x,1,8) = [ 1.,  1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  8.]

)code" ADD_FILELINE)
.set_num_inputs(1)
.set_num_outputs(1)
.set_attr_parser(ParamParser<ClipParam>)
.set_attr<nnvm::FInferShape>("FInferShape", ElemwiseShape<1, 1>)
.set_attr<nnvm::FInferType>("FInferType", ElemwiseType<1, 1>)
.set_attr<FCompute>("FCompute<cpu>", Clip<cpu>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseIn{ "_backward_clip" })
.add_argument("data", "NDArray-or-Symbol", "Input array.")
.add_arguments(ClipParam::__FIELDS__());

NNVM_REGISTER_OP(_backward_clip)
.set_num_inputs(1)
.set_num_outputs(1)
.set_attr_parser(ParamParser<ClipParam>)
.set_attr<nnvm::TIsBackward>("TIsBackward", true)
.set_attr<FCompute>("FCompute<cpu>", ClipGrad_<cpu>);

NNVM_REGISTER_OP(repeat)
.describe(R"code(Repeats elements of an array.

By default, ``repeat`` flattens the input array into 1-D and then repeats the
elements::

  x = [[ 1, 2],
       [ 3, 4]]

  repeat(x, repeats=2) = [ 1.,  1.,  2.,  2.,  3.,  3.,  4.,  4.]

The parameter ``axis`` specifies the axis along which to perform repeat::

  repeat(x, repeats=2, axis=1) = [[ 1.,  1.,  2.,  2.],
                                  [ 3.,  3.,  4.,  4.]]

  repeat(x, repeats=2, axis=0) = [[ 1.,  2.],
                                  [ 1.,  2.],
                                  [ 3.,  4.],
                                  [ 3.,  4.]]

  repeat(x, repeats=2, axis=-1) = [[ 1.,  1.,  2.,  2.],
                                   [ 3.,  3.,  4.,  4.]]

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
.add_argument("data", "NDArray-or-Symbol", "Input data array")
.add_arguments(RepeatParam::__FIELDS__());

NNVM_REGISTER_OP(_backward_repeat)
.set_num_inputs(1)
.set_num_outputs(1)
.set_attr_parser(ParamParser<RepeatParam>)
.set_attr<nnvm::TIsBackward>("TIsBackward", true)
.set_attr<FCompute>("FCompute<cpu>", RepeatOpBackward<cpu>)
.set_attr<FResourceRequest>("FResourceRequest",
[](const NodeAttrs& attrs) {
  return std::vector<ResourceRequest> {ResourceRequest::kTempSpace};
});

NNVM_REGISTER_OP(tile)
.describe(R"code(Repeats the whole array multiple times.

If ``reps`` has length *d*, and input array has dimension of *n*. There are
there cases:

- **n=d**. Repeat *i*-th dimension of the input by ``reps[i]`` times::

    x = [[1, 2],
         [3, 4]]

    tile(x, reps=(2,3)) = [[ 1.,  2.,  1.,  2.,  1.,  2.],
                           [ 3.,  4.,  3.,  4.,  3.,  4.],
                           [ 1.,  2.,  1.,  2.,  1.,  2.],
                           [ 3.,  4.,  3.,  4.,  3.,  4.]]

- **n>d**. ``reps`` is promoted to length *n* by pre-pending 1's to it. Thus for
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
.add_argument("data", "NDArray-or-Symbol", "Input data array")
.add_arguments(TileParam::__FIELDS__());

NNVM_REGISTER_OP(_backward_tile)
.set_num_inputs(1)
.set_num_outputs(1)
.set_attr_parser(ParamParser<TileParam>)
.set_attr<nnvm::TIsBackward>("TIsBackward", true)
.set_attr<FCompute>("FCompute<cpu>", TileOpBackward<cpu>)
.set_attr<FResourceRequest>("FResourceRequest",
[](const NodeAttrs& attrs) {
  return std::vector<ResourceRequest> {ResourceRequest::kTempSpace};
});

NNVM_REGISTER_OP(reverse)
.describe(R"code(Reverses the order of elements along given axis while preserving array shape.

Note: reverse and flip are equivalent. We use reverse in the following examples.

Examples::

  x = [[ 0.,  1.,  2.,  3.,  4.],
       [ 5.,  6.,  7.,  8.,  9.]]

  reverse(x, axis=0) = [[ 5.,  6.,  7.,  8.,  9.],
                        [ 0.,  1.,  2.,  3.,  4.]]

  reverse(x, axis=1) = [[ 4.,  3.,  2.,  1.,  0.],
                        [ 9.,  8.,  7.,  6.,  5.]]
)code" ADD_FILELINE)
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
.add_argument("data", "NDArray-or-Symbol", "Input data array")
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

NNVM_REGISTER_OP(stack)
.describe(R"code(Join a sequence of arrays along a new axis.

The axis parameter specifies the index of the new axis in the dimensions of the
result. For example, if axis=0 it will be the first dimension and if axis=-1 it
will be the last dimension.

Examples::

  x = [1, 2]
  y = [3, 4]

  stack(x, y) = [[1, 2],
                 [3, 4]]
  stack(x, y, axis=1) = [[1, 3],
                         [2, 4]]
)code")
.set_num_inputs([](const nnvm::NodeAttrs& attrs) {
    const StackParam& param = dmlc::get<StackParam>(attrs.parsed);
    return static_cast<uint32_t>(param.num_args);
  })
.set_num_outputs(1)
.set_attr_parser(ParamParser<StackParam>)
.set_attr<nnvm::FListInputNames>("FListInputNames",
  [](const NodeAttrs& attrs) {
    uint32_t num_args = dmlc::get<StackParam>(attrs.parsed).num_args;
    std::vector<std::string> ret;
    for (uint32_t i = 0; i < num_args; ++i) {
      ret.push_back(std::string("arg") + std::to_string(i));
    }
    return ret;
  })
.set_attr<std::string>("key_var_num_args", "num_args")
.set_attr<nnvm::FInferShape>("FInferShape", StackOpShape)
.set_attr<nnvm::FInferType>("FInferType", ElemwiseType<-1, 1>)
.set_attr<FCompute>("FCompute<cpu>", StackOpForward<cpu>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseNone{"_backward_stack"})
.add_argument("data", "NDArray-or-Symbol[]", "List of arrays to stack")
.add_arguments(StackParam::__FIELDS__());

NNVM_REGISTER_OP(_backward_stack)
.set_num_inputs(1)
.set_num_outputs([](const nnvm::NodeAttrs& attrs) {
    const StackParam& param = dmlc::get<StackParam>(attrs.parsed);
    return static_cast<uint32_t>(param.num_args);
  })
.set_attr_parser(ParamParser<StackParam>)
.set_attr<nnvm::TIsBackward>("TIsBackward", true)
.set_attr<FCompute>("FCompute<cpu>", StackOpBackward<cpu>);

}  // namespace op
}  // namespace mxnet
