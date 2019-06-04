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
 *  Copyright (c) 2019 by Contributors
 * \file np_matrix_op.cc
 * \brief CPU Implementation of numpy matrix operations
 */

#include "./np_matrix_op-inl.h"
#include "../nn/concat-inl.h"

namespace mxnet {
namespace op {

DMLC_REGISTER_PARAMETER(NumpyTransposeParam);

bool NumpyTransposeShape(const nnvm::NodeAttrs& attrs,
                         mxnet::ShapeVector *in_attrs,
                         mxnet::ShapeVector *out_attrs) {
  const NumpyTransposeParam& param = nnvm::get<NumpyTransposeParam>(attrs.parsed);
  CHECK_EQ(in_attrs->size(), 1U);
  CHECK_EQ(out_attrs->size(), 1U);
  mxnet::TShape& shp = (*in_attrs)[0];
  CHECK_LE(shp.ndim(), 6) << "Transpose support at most 6 dimensions";
  mxnet::TShape ret(shp.ndim(), -1);
  if (ndim_is_known(param.axes)) {
    CHECK_EQ(shp.ndim(), param.axes.ndim());
    for (int i = 0; i < shp.ndim(); ++i) {
      CHECK(param.axes[i] < static_cast<int64_t>(shp.ndim()));
      ret[i] = shp[param.axes[i]];
    }
  } else {
    for (int i = 0; i < shp.ndim(); ++i) {
      ret[i] = shp[shp.ndim()-1-i];
    }
  }
  SHAPE_ASSIGN_CHECK(*out_attrs, 0, ret);
  return shape_is_known(ret);
}

NNVM_REGISTER_OP(_np_transpose)
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
.set_attr_parser(ParamParser<NumpyTransposeParam>)
.set_attr<mxnet::FInferShape>("FInferShape", NumpyTransposeShape)
.set_attr<nnvm::FInferType>("FInferType", ElemwiseType<1, 1>)
.set_attr<nnvm::FGradient>("FGradient",
  [](const nnvm::NodePtr& n, const std::vector<nnvm::NodeEntry>& ograds) {
    const NumpyTransposeParam& param = nnvm::get<NumpyTransposeParam>(n->attrs.parsed);
    if (ndim_is_known(param.axes)) {
      mxnet::TShape axes = mxnet::TShape(param.axes.ndim(), -1);
      for (int i = 0; i < axes.ndim(); ++i) {
        axes[param.axes[i]] = i;
      }
      std::ostringstream os;
      os << axes;
      return MakeNonlossGradNode("transpose", n, ograds, {}, {{"axes", os.str()}});
    } else {
      return MakeNonlossGradNode("transpose", n, ograds, {},
                                 std::unordered_map<std::string, std::string>());
    }
  })
.set_attr<FCompute>("FCompute<cpu>", NumpyTranspose<cpu>)
.set_attr<nnvm::FListInputNames>("FListInputNames",
  [](const NodeAttrs& attrs) {
    return std::vector<std::string>{"a"};
  })
.add_argument("a", "NDArray-or-Symbol", "Source input")
.add_arguments(NumpyTransposeParam::__FIELDS__());

struct NumpyReshapeParam : public dmlc::Parameter<NumpyReshapeParam> {
  mxnet::TShape newshape;
  std::string order;
  DMLC_DECLARE_PARAMETER(NumpyReshapeParam) {
      DMLC_DECLARE_FIELD(newshape)
          .describe("The new shape should be compatible with the original shape."
                    " If an integer, then the result will be a 1-D array of that length."
                    " One shape dimension can be -1. In this case, the value is inferred"
                    " from the length of the array and remaining dimensions.");
      DMLC_DECLARE_FIELD(order)
      .set_default("C")
      .describe("Read the elements of a using this index order, and place the elements into"
                " the reshaped array using this index order. 'C' means to read/write the elements"
                " using C-like index order, with the last axis index changing fastest, back to the"
                " first axis index changing slowest. Note that currently only C-like order is"
                " supported");
  }
};

DMLC_REGISTER_PARAMETER(NumpyReshapeParam);

bool NumpyReshapeInferShape(const mxnet::TShape& src, mxnet::TShape* dst) {
  if (shape_is_known(src) && shape_is_known(*dst)) {
    CHECK_EQ(src.Size(), dst->Size()) << "Cannot reshape array of size "
                                      << src.Size() << " into shape " << *dst;
    return true;
  } else if (!shape_is_known(src) || !ndim_is_known(*dst)) {
    return false;
  } else {
    int unknown_axis = -1;
    dim_t known_dim_size_prod = 1;
    for (int i = 0; i < dst->ndim(); ++i) {
      if (!dim_size_is_known(*dst, i)) {
        if (unknown_axis == -1) {
          unknown_axis = i;
        } else {
          return false;  // more than one unknown dim
        }
      } else {
        known_dim_size_prod *= (*dst)[i];
      }
    }
    CHECK_NE(known_dim_size_prod, 0) << "Cannot reshape array of size "
                                     << src.Size() << " into shape " << *dst;
    CHECK_EQ(src.Size() % known_dim_size_prod, 0) << "Cannot reshape array of size "
                                                  << src.Size() << " into shape " << *dst;
    (*dst)[unknown_axis] = src.Size() / known_dim_size_prod;
    return true;
  }
}

bool NumpyReshapeShape(const nnvm::NodeAttrs& attrs,
                       mxnet::ShapeVector* in_attrs,
                       mxnet::ShapeVector* out_attrs) {
  CHECK_EQ(in_attrs->size(), 1U) << "Input: [data]";
  CHECK_EQ(out_attrs->size(), 1U);
  const NumpyReshapeParam& param = nnvm::get<NumpyReshapeParam>(attrs.parsed);
  // sanity check
  bool has_unknown_dim_size = false;
  for (int i = 0; i < param.newshape.ndim(); ++i) {
    if (param.newshape[i] < 0) {
      CHECK_EQ(param.newshape[i], -1) << "The shape dimension size to inferred must be -1";
      CHECK(!has_unknown_dim_size) << "Can only specify one unknown dimension";
      has_unknown_dim_size = true;
    }
  }

  mxnet::TShape target_shape = param.newshape;
  bool success = NumpyReshapeInferShape(in_attrs->at(0), &target_shape);
  SHAPE_ASSIGN_CHECK(*out_attrs, 0, target_shape);
  if (!success) {
    success = NumpyReshapeInferShape(out_attrs->at(0), &in_attrs->at(0));
  }
  return success;
}

NNVM_REGISTER_OP(_np_reshape)
.describe(R"code()code" ADD_FILELINE)
.set_num_inputs(1)
.set_num_outputs(1)
.set_attr_parser(ParamParser<NumpyReshapeParam>)
.set_attr<mxnet::FInferShape>("FInferShape", NumpyReshapeShape)
.set_attr<nnvm::FInferType>("FInferType", ElemwiseType<1, 1>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseNone{"_backward_reshape"})
.set_attr<FCompute>("FCompute<cpu>", UnaryOp::IdentityCompute<cpu>)
.set_attr<nnvm::FInplaceOption>("FInplaceOption",
  [](const NodeAttrs& attrs) {
    return std::vector<std::pair<int, int> >{{0, 0}};
  })
.set_attr<nnvm::FInplaceIdentity>("FInplaceIdentity",
  [](const NodeAttrs& attrs){
    return std::vector<bool>{true};
  })
.set_attr<nnvm::FListInputNames>("FListInputNames",
  [](const NodeAttrs& attrs) {
    return std::vector<std::string>{"a"};
  })
.add_argument("a", "NDArray-or-Symbol", "Array to be reshaped.")
.add_arguments(NumpyReshapeParam::__FIELDS__());

NNVM_REGISTER_OP(_npi_stack)
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
.set_attr<mxnet::FInferShape>("FInferShape", StackOpShape)
.set_attr<nnvm::FInferType>("FInferType", ElemwiseType<-1, 1>)
.set_attr<FCompute>("FCompute<cpu>", StackOpForward<cpu>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseNone{"_backward_stack"})
.add_argument("data", "NDArray-or-Symbol[]", "List of arrays to stack")
.add_arguments(StackParam::__FIELDS__());

bool ConcatShape(const nnvm::NodeAttrs& attrs,
                 mxnet::ShapeVector *in_shape,
                 mxnet::ShapeVector *out_shape);

bool ConcatType(const nnvm::NodeAttrs& attrs,
                std::vector<int> *in_type,
                std::vector<int> *out_type);

struct NumpyConcatGrad {
  const char *op_name;
  std::vector<nnvm::NodeEntry> operator()(const nnvm::NodePtr& n,
                                          const std::vector<nnvm::NodeEntry>& ograds) const {
    CHECK_EQ(ograds.size(), 1);
    std::vector<nnvm::NodeEntry> heads(ograds.begin(), ograds.end());
    return MakeGradNode(op_name, n, heads, n->attrs.dict);
  }
};


NNVM_REGISTER_OP(_npi_concatenate)
.describe(R"code(Join a sequence of arrays along an existing axis.)code" ADD_FILELINE)
.set_num_inputs([](const NodeAttrs& attrs) {
  const ConcatParam& params = nnvm::get<ConcatParam>(attrs.parsed);
  return params.num_args;
})
.set_num_outputs(1)
.set_attr_parser(ParamParser<ConcatParam>)
.set_attr<nnvm::FListInputNames>("FListInputNames",
  [](const NodeAttrs& attrs) {
    const ConcatParam& params = nnvm::get<ConcatParam>(attrs.parsed);
    std::vector<std::string> ret;
    for (int i = 0; i < params.num_args; ++i) {
      ret.push_back(std::string("data") + std::to_string(i));
    }
    return ret;
})
.set_attr<nnvm::FListOutputNames>("FListOutputNames",
  [](const NodeAttrs& attrs) {
    return std::vector<std::string>{"out"};
})
.set_attr<std::string>("key_var_num_args", "num_args")
.set_attr<nnvm::FInferType>("FInferType", ConcatType)
.set_attr<mxnet::FInferShape>("FInferShape", ConcatShape)
.set_attr<FCompute>("FCompute<cpu>", ConcatCompute<cpu>)
.set_attr<nnvm::FGradient>("FGradient", NumpyConcatGrad{"_backward_np_concat"})
.add_argument("data", "NDArray-or-Symbol[]", "List of arrays to concatenate")
.add_arguments(ConcatParam::__FIELDS__());

NNVM_REGISTER_OP(_backward_np_concat)
.set_num_outputs([](const NodeAttrs& attrs) {
  const ConcatParam& params = nnvm::get<ConcatParam>(attrs.parsed);
  return params.num_args;
})
.set_attr_parser(ParamParser<ConcatParam>)
.set_attr<nnvm::TIsBackward>("TIsBackward", true)
.set_attr<FCompute>("FCompute<cpu>", ConcatGradCompute<cpu>);

}  // namespace op
}  // namespace mxnet
