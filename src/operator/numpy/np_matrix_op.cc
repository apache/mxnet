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
 * \file np_matrix_op.cc
 * \brief CPU Implementation of numpy matrix operations
 */

#include <vector>
#include <set>
#include "./np_matrix_op-inl.h"
#include "../nn/concat-inl.h"
#if MXNET_USE_ONEDNN == 1
#include "operator/nn/dnnl/dnnl_base-inl.h"
#include "operator/nn/dnnl/dnnl_transpose-inl.h"
#endif
namespace mxnet {
namespace op {

DMLC_REGISTER_PARAMETER(NumpyTransposeParam);
DMLC_REGISTER_PARAMETER(NumpyRollParam);
DMLC_REGISTER_PARAMETER(NumpyMoveaxisParam);
DMLC_REGISTER_PARAMETER(NumpyRollaxisParam);
DMLC_REGISTER_PARAMETER(NumpyRot90Param);
DMLC_REGISTER_PARAMETER(NumpyReshapeParam);
DMLC_REGISTER_PARAMETER(NumpyXReshapeParam);
DMLC_REGISTER_PARAMETER(NumpyDiagParam);
DMLC_REGISTER_PARAMETER(NumpyDiagonalParam);
DMLC_REGISTER_PARAMETER(NumpyDiagflatParam);

#if MXNET_USE_ONEDNN == 1

static void NumpyTransposeComputeExCPU(const nnvm::NodeAttrs& attrs,
                                       const OpContext& ctx,
                                       const std::vector<NDArray>& inputs,
                                       const std::vector<OpReqType>& req,
                                       const std::vector<NDArray>& outputs) {
  if (req[0] == kNullOp) {
    return;
  }
  CHECK(req[0] == kWriteTo || req[0] == kAddTo)
      << "Transpose only supports kNullOp, kWriteTo and kAddTo";
  CHECK_EQ(inputs.size(), 1U);
  CHECK_EQ(outputs.size(), 1U);

  if (SupportDNNL(inputs[0]) && req[0] == kWriteTo) {
    DNNLRun(DNNLTransposeForward<NumpyTransposeParam>, attrs, ctx, inputs[0], req[0], outputs[0]);
    return;
  }
  FallBackCompute(NumpyTranspose<cpu>, attrs, ctx, inputs, req, outputs);
}

inline static bool NumpyTransposeStorageType(const nnvm::NodeAttrs& attrs,
                                             const int dev_mask,
                                             DispatchMode* dispatch_mode,
                                             std::vector<int>* in_attrs,
                                             std::vector<int>* out_attrs) {
  CHECK_EQ(in_attrs->size(), 1U);
  CHECK_EQ(out_attrs->size(), 1U);
  return DNNLStorageType(attrs, dev_mask, true, dispatch_mode, in_attrs, out_attrs);
}
#endif

NNVM_REGISTER_OP(_npi_transpose)
    .set_num_inputs(1)
    .set_num_outputs(1)
    .set_attr_parser(ParamParser<NumpyTransposeParam>)
    .set_attr<mxnet::FInferShape>("FInferShape", NumpyTransposeShape)
    .set_attr<nnvm::FInferType>("FInferType", ElemwiseType<1, 1>)
    .set_attr<nnvm::FGradient>(
        "FGradient",
        [](const nnvm::ObjectPtr& n, const std::vector<nnvm::NodeEntry>& ograds) {
          const NumpyTransposeParam& param = nnvm::get<NumpyTransposeParam>(n->attrs.parsed);
          if (ndim_is_known(param.axes)) {
            mxnet::TShape axes = mxnet::TShape(param.axes.ndim(), -1);
            for (int i = 0; i < axes.ndim(); ++i) {
              int axis = param.axes[i];
              if (axis < 0) {
                axis += param.axes.ndim();
              }
              CHECK(axis >= 0 && axis < param.axes.ndim());
              axes[axis] = i;
            }
            std::ostringstream os;
            os << axes;
            return MakeNonlossGradNode("_npi_transpose", n, ograds, {}, {{"axes", os.str()}});
          } else {
            return MakeNonlossGradNode(
                "_npi_transpose", n, ograds, {}, std::unordered_map<std::string, std::string>());
          }
        })
    .set_attr<FCompute>("FCompute<cpu>", NumpyTranspose<cpu>)
    .set_attr<FResourceRequest>("FResourceRequest",
                                [](const NodeAttrs& attrs) {
                                  return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
                                })
#if MXNET_USE_ONEDNN == 1
    .set_attr<bool>("TIsDNNL", true)
    .set_attr<FComputeEx>("FComputeEx<cpu>", NumpyTransposeComputeExCPU)
    .set_attr<FInferStorageType>("FInferStorageType", NumpyTransposeStorageType)
#endif
    .set_attr<nnvm::FListInputNames>("FListInputNames",
                                     [](const NodeAttrs& attrs) {
                                       return std::vector<std::string>{"a"};
                                     })
    .add_argument("a", "NDArray-or-Symbol", "Source input")
    .add_arguments(NumpyTransposeParam::__FIELDS__());

bool NumpyReshapeInferShape(const mxnet::TShape& src, mxnet::TShape* dst) {
  if (shape_is_known(src) && shape_is_known(*dst)) {
    CHECK_EQ(src.Size(), dst->Size())
        << "Cannot reshape array of size " << src.Size() << " into shape " << *dst;
    return true;
  } else if (!shape_is_known(src) || !ndim_is_known(*dst)) {
    return false;
  } else {
    int unknown_axis          = -1;
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
    CHECK_NE(known_dim_size_prod, 0)
        << "Cannot reshape array of size " << src.Size() << " into shape " << *dst;
    CHECK_EQ(src.Size() % known_dim_size_prod, 0)
        << "Cannot reshape array of size " << src.Size() << " into shape " << *dst;
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
  bool success               = NumpyReshapeInferShape(in_attrs->at(0), &target_shape);
  SHAPE_ASSIGN_CHECK(*out_attrs, 0, target_shape);
  if (!success) {
    success = NumpyReshapeInferShape(out_attrs->at(0), &in_attrs->at(0));
  }
  return success;
}

bool NumpyXReshapeInferShape(const mxnet::TShape& src,
                             const mxnet::TShape& target,
                             mxnet::TShape* output,
                             const std::string& default_error_msg) {
  bool target_shape_is_known = true;
  dim_t target_size          = 1;
  for (int i = 0; i < target.ndim(); ++i) {
    if (target[i] < 0) {
      target_shape_is_known = false;
      target_size           = -1;
      break;
    } else {
      target_size *= target[i];
    }
  }
  if (shape_is_known(src) && target_shape_is_known) {
    CHECK_EQ(src.Size(), target_size) << default_error_msg;
    *output = TShape(target.begin(), target.end());
    return true;
  } else if (!shape_is_known(src) || target.ndim() == -1) {
    return false;
  } else {
    int unknown_axis          = -1;
    dim_t known_dim_size_prod = 1;
    std::vector<dim_t> output_shape_vector;
    int src_inx = 0;
    for (int i = 0; i < target.ndim(); ++i) {
      dim_t proposed_dim = target[i];
      CHECK(proposed_dim >= -6) << "Dimension size must be greater than -6, received "
                                << proposed_dim;
      if (proposed_dim == -1) {
        // infer the known dimension
        CHECK_LT(unknown_axis, 0) << "One and only one dim can be inferred";
        unknown_axis = output_shape_vector.size();
        output_shape_vector.push_back(-1);
        src_inx++;
      } else if (proposed_dim == -2) {
        // copy the dimension from src to output
        CHECK_LT(src_inx, src.ndim()) << "Unmatching dimension of proposed new shape";
        known_dim_size_prod *= src[src_inx];
        output_shape_vector.push_back(src[src_inx++]);
      } else if (proposed_dim == -3) {
        // skip the source dimension if and only if it is one
        CHECK_EQ(src[src_inx], 1) << "-3 index should only be used to skip dimension size 1";
        src_inx++;
      } else if (proposed_dim == -4) {
        // copy all remaining dims from source
        while (src_inx < src.ndim()) {
          known_dim_size_prod *= src[src_inx];
          const dim_t dn = src[src_inx++];
          output_shape_vector.push_back(dn);
        }
      } else if (proposed_dim == -5) {
        // merge two dims from source
        CHECK_LT(src_inx, src.ndim() - 1) << "Not enough dimensions left for the product";
        const dim_t d1 = src[src_inx++];
        const dim_t d2 = src[src_inx++];
        if (!mxnet::dim_size_is_known(d1) || !mxnet::dim_size_is_known(d2)) {
          CHECK_LT(unknown_axis, 0) << "One and only one dim can be inferred";
          unknown_axis = output_shape_vector.size();
          output_shape_vector.push_back(-1);
        } else {
          known_dim_size_prod *= d1 * d2;
          output_shape_vector.push_back(d1 * d2);
        }
      } else if (proposed_dim == -6) {
        // split the source dim s into two dims
        // read the left dim and then the right dim (either can be -1)
        CHECK_LT(i + 2, target.ndim());
        CHECK_LT(src_inx, src.ndim());
        const dim_t d0 = src[src_inx++];
        dim_t d1       = target[++i];
        dim_t d2       = target[++i];
        CHECK(d1 != -1 || d2 != -1) << "Split dims cannot both be -1.";
        if (d1 == -1 && d0 >= 0)
          d1 = d0 / d2;  // d0 must be known to do this
        if (d2 == -1 && d0 >= 0)
          d2 = d0 / d1;  // d0 must be known to do this
        CHECK(d1 * d2 == static_cast<dim_t>(d0) || static_cast<dim_t>(d0) == dim_t(-1))
            << "Split dims " << d1 << ", " << d2 << " do not divide original dim " << d0;
        if (d1 == -1) {
          CHECK_LT(unknown_axis, 0) << "One and only one dim can be inferred";
          unknown_axis = output_shape_vector.size();
        } else if (d2 == -1) {
          CHECK_LT(unknown_axis, 0) << "One and only one dim can be inferred";
          unknown_axis = output_shape_vector.size() + 1;
        }
        known_dim_size_prod *= d0 == -1 ? 1 : d0;
        output_shape_vector.push_back(d1);
        output_shape_vector.push_back(d2);
      } else {
        // greater than 0, new shape
        known_dim_size_prod *= proposed_dim;
        output_shape_vector.push_back(proposed_dim);
        src_inx++;
      }
    }

    if (unknown_axis > -1) {
      // if the input in zero size tensor, the output must be of known shape of zero size
      CHECK_NE(known_dim_size_prod, 0) << default_error_msg;
      CHECK(src.Size() % known_dim_size_prod == 0) << default_error_msg;
      output_shape_vector[unknown_axis] = src.Size() / known_dim_size_prod;
    }

    *output = mxnet::TShape(output_shape_vector.begin(), output_shape_vector.end());
    CHECK_EQ((*output).Size(), src.Size()) << default_error_msg;
    return true;
  }
}

bool NumpyXReshapeShape(const nnvm::NodeAttrs& attrs,
                        mxnet::ShapeVector* in_attrs,
                        mxnet::ShapeVector* out_attrs) {
  CHECK_EQ(in_attrs->size(), 1U) << "Input: [data]";
  CHECK_EQ(out_attrs->size(), 1U);
  const NumpyXReshapeParam& param = nnvm::get<NumpyXReshapeParam>(attrs.parsed);
  // sanity check
  bool has_unknown_dim_size = false;
  for (int i = 0; i < param.newshape.ndim(); ++i) {
    if (param.newshape[i] < 0) {
      CHECK_GE(param.newshape[i], -6) << "Dimension size must be greater than or equal to -6";
      if (param.newshape[i] == -1) {
        CHECK(!has_unknown_dim_size) << "Can only specify one unknown dimension";
        has_unknown_dim_size = true;
      }
    }
  }

  mxnet::TShape output_shape;
  bool success;
  std::stringstream ss;
  ss << "Cannot reshape array of shape " << in_attrs->at(0) << " into shape " << param.newshape
     << " , reverse = " << param.reverse;
  std::string err_msg = ss.str();
  if (!param.reverse) {
    success = NumpyXReshapeInferShape(in_attrs->at(0), param.newshape, &output_shape, err_msg);
  } else {
    mxnet::TShape rev_in_shape = in_attrs->at(0);
    mxnet::TShape rev_newshape = param.newshape;
    std::reverse(rev_in_shape.begin(), rev_in_shape.end());
    std::reverse(rev_newshape.begin(), rev_newshape.end());
    success = NumpyXReshapeInferShape(rev_in_shape, rev_newshape, &output_shape, err_msg);
    std::reverse(output_shape.begin(), output_shape.end());
  }
  SHAPE_ASSIGN_CHECK(*out_attrs, 0, output_shape);
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
                                      [](const NodeAttrs& attrs) {
                                        return std::vector<bool>{true};
                                      })
    .set_attr<nnvm::FListInputNames>("FListInputNames",
                                     [](const NodeAttrs& attrs) {
                                       return std::vector<std::string>{"a"};
                                     })
    .add_argument("a", "NDArray-or-Symbol", "Array to be reshaped.")
    .add_arguments(NumpyReshapeParam::__FIELDS__());

NNVM_REGISTER_OP(_npx_reshape)
    .describe(R"code()code" ADD_FILELINE)
    .add_alias("_npi_reshape")
    .set_num_inputs(1)
    .set_num_outputs(1)
    .set_attr_parser(ParamParser<NumpyXReshapeParam>)
    .set_attr<mxnet::FInferShape>("FInferShape", NumpyXReshapeShape)
    .set_attr<nnvm::FInferType>("FInferType", ElemwiseType<1, 1>)
    .set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseNone{"_backward_reshape"})
    .set_attr<FCompute>("FCompute<cpu>", UnaryOp::IdentityCompute<cpu>)
#if MXNET_USE_ONEDNN == 1
    .set_attr<bool>("TIsDNNL", true)
    .set_attr<FComputeEx>("FComputeEx<cpu>", ReshapeComputeExCPU)
    .set_attr<FInferStorageType>("FInferStorageType", ReshapeStorageType)
    .set_attr<FResourceRequest>("FResourceRequest",
                                [](const NodeAttrs& n) {
                                  return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
                                })
#endif
    .set_attr<nnvm::FInplaceOption>("FInplaceOption",
                                    [](const NodeAttrs& attrs) {
                                      return std::vector<std::pair<int, int> >{{0, 0}};
                                    })
    .set_attr<nnvm::FInplaceIdentity>("FInplaceIdentity",
                                      [](const NodeAttrs& attrs) {
                                        return std::vector<bool>{true};
                                      })
    .set_attr<nnvm::FListInputNames>("FListInputNames",
                                     [](const NodeAttrs& attrs) {
                                       return std::vector<std::string>{"a"};
                                     })
    .add_argument("a", "NDArray-or-Symbol", "Array to be reshaped.")
    .add_arguments(NumpyXReshapeParam::__FIELDS__());

bool NumpySqueezeShape(const nnvm::NodeAttrs& attrs,
                       mxnet::ShapeVector* in_attrs,
                       mxnet::ShapeVector* out_attrs) {
  const SqueezeParam& param = nnvm::get<SqueezeParam>(attrs.parsed);
  CHECK_EQ(in_attrs->size(), 1U) << "Input: [a]";
  CHECK_EQ(out_attrs->size(), 1U);
  const mxnet::TShape& dshape = in_attrs->at(0);
  const int dndim             = dshape.ndim();
  if (!shape_is_known(dshape))
    return false;
  mxnet::TShape oshape = dshape;
  // special case, scalar tensor
  if (dshape.ndim() == 0) {
    if (param.axis.has_value()) {
      mxnet::Tuple<int> axes = param.axis.value();
      CHECK_EQ(axes.ndim(), 1) << "cannot specify more than one axis for a scalar tensor";
      CHECK(axes[0] == 0 || axes[0] == -1)
          << "axis " << axes[0] << " is out of bounds of array of dimension 0";
    }
    SHAPE_ASSIGN_CHECK(*out_attrs, 0, mxnet::TShape(0, -1));
    return true;
  }
  if (param.axis.has_value()) {
    // preprocess axis
    mxnet::Tuple<int> axes = param.axis.value();
    for (int i = 0; i < axes.ndim(); ++i) {
      if (axes[i] < 0) {
        axes[i] += dndim;
        CHECK_GE(axes[i], 0) << "axis " << axes[i] - dndim
                             << " is out of bounds for array of dimension " << dndim;
      }
      CHECK_LT(axes[i], dndim) << "axis " << axes[i] << " is out of bounds for array of dimension "
                               << dndim;
      CHECK_EQ(dshape[axes[i]], 1)
          << "cannot select an axis to squeeze out which has size=" << dshape[axes[i]]
          << " not equal to one";
      CHECK_NE(oshape[axes[i]], 0) << "duplicate value in axis";
      oshape[axes[i]] = -1;
    }
  } else {
    for (int i = 0; i < oshape.ndim(); ++i) {
      if (oshape[i] == 1)
        oshape[i] = -1;
    }
  }
  size_t oshape_size = SqueezeShapeHelper(&oshape);
  SHAPE_ASSIGN_CHECK(*out_attrs, 0, mxnet::TShape(oshape.data(), oshape.data() + oshape_size));
  return true;
}

NNVM_REGISTER_OP(_npi_squeeze)
    .set_num_inputs(1)
    .set_num_outputs(1)
    .set_attr_parser(ParamParser<SqueezeParam>)
    .set_attr<nnvm::FListInputNames>("FListInputNames",
                                     [](const NodeAttrs& attrs) {
                                       return std::vector<std::string>{"a"};
                                     })
    .set_attr<mxnet::FInferShape>("FInferShape", NumpySqueezeShape)
    .set_attr<nnvm::FInferType>("FInferType", ElemwiseType<1, 1>)
    .set_attr<FCompute>("FCompute<cpu>", UnaryOp::IdentityCompute<cpu>)
    .set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseNone{"_backward_squeeze"})
    .add_argument("a", "NDArray-or-Symbol", "data to squeeze")
    .add_arguments(SqueezeParam::__FIELDS__());

bool HStackShape(const nnvm::NodeAttrs& attrs,
                 mxnet::ShapeVector* in_shape,
                 mxnet::ShapeVector* out_shape) {
  using namespace mshadow;
  ConcatParam param_ = nnvm::get<ConcatParam>(attrs.parsed);
  CHECK_EQ(in_shape->size(), static_cast<size_t>(param_.num_args));
  mxnet::TShape dshape;
  dim_t size                = 0;
  bool has_unknown_dim_size = false;
  int axis                  = (*in_shape)[0].ndim() > 1 ? 1 : 0;
  param_.dim                = axis;
  for (int i = 0; i < param_.num_args; ++i) {
    // scalor tensor is treated as one dimensional vector
    if ((*in_shape)[i].ndim() == 0) {
      (*in_shape)[i] = mxnet::TShape(1, 1);
    }
    mxnet::TShape& tmp = (*in_shape)[i];
    if (tmp.ndim() > 0) {
      CheckAxis(axis, tmp.ndim());
      if (!mxnet::dim_size_is_known(tmp, axis)) {
        has_unknown_dim_size = true;
      } else {
        size += tmp[axis];
      }
      tmp[axis] = -1;
      shape_assign(&dshape, tmp);
    }
  }

  mxnet::TShape tmp = (*out_shape)[0];
  if (tmp.ndim() > 0) {
    axis      = CheckAxis(param_.dim.value(), tmp.ndim());
    tmp[axis] = -1;
    shape_assign(&dshape, tmp);
  }

  if (dshape.ndim() == -1)
    return false;
  CHECK_NE(dshape.ndim(), 0) << "zero-dimensional arrays cannot be concatenated";

  for (int i = 0; i < param_.num_args; ++i) {
    CHECK(shape_assign(&(*in_shape)[i], dshape))
        << "Incompatible input shape: expected " << dshape << ", got " << (*in_shape)[i];
  }

  if (!has_unknown_dim_size) {
    dshape[axis] = size;
  }
  CHECK(shape_assign(&(*out_shape)[0], dshape))
      << "Incompatible output shape: expected " << dshape << ", got " << (*out_shape)[0];

  return shape_is_known(dshape);
}

bool DStackShape(const nnvm::NodeAttrs& attrs,
                 mxnet::ShapeVector* in_shape,
                 mxnet::ShapeVector* out_shape) {
  using namespace mshadow;
  ConcatParam param_ = nnvm::get<ConcatParam>(attrs.parsed);
  CHECK_EQ(in_shape->size(), static_cast<size_t>(param_.num_args));
  mxnet::TShape dshape;
  dim_t size                = 0;
  bool has_unknown_dim_size = false;
  int axis                  = 2;
  param_.dim                = axis;
  for (int i = 0; i < param_.num_args; ++i) {
    if ((*in_shape)[i].ndim() == 0) {
      (*in_shape)[i] = mxnet::TShape(3, 1);
    } else if ((*in_shape)[i].ndim() == 1) {
      mxnet::TShape t = mxnet::TShape(3, 1);
      t[1]            = (*in_shape)[i][0];
      (*in_shape)[i]  = t;
    } else if ((*in_shape)[i].ndim() == 2) {
      mxnet::TShape t = mxnet::TShape(3, 1);
      t[0]            = (*in_shape)[i][0];
      t[1]            = (*in_shape)[i][1];
      (*in_shape)[i]  = t;
    }
    mxnet::TShape& tmp = (*in_shape)[i];
    if (tmp.ndim() > 0) {
      CheckAxis(axis, tmp.ndim());
      if (!mxnet::dim_size_is_known(tmp, axis)) {
        has_unknown_dim_size = true;
      } else {
        size += tmp[axis];
      }
      tmp[axis] = -1;
      shape_assign(&dshape, tmp);
    }
  }

  mxnet::TShape tmp = (*out_shape)[0];
  if (tmp.ndim() > 0) {
    axis      = CheckAxis(param_.dim.value(), tmp.ndim());
    tmp[axis] = -1;
    shape_assign(&dshape, tmp);
  }

  if (dshape.ndim() == -1)
    return false;
  CHECK_NE(dshape.ndim(), 0) << "zero-dimensional arrays cannot be concatenated";

  for (int i = 0; i < param_.num_args; ++i) {
    CHECK(shape_assign(&(*in_shape)[i], dshape))
        << "Incompatible input shape: expected " << dshape << ", got " << (*in_shape)[i];
  }

  if (!has_unknown_dim_size) {
    dshape[axis] = size;
  }
  CHECK(shape_assign(&(*out_shape)[0], dshape))
      << "Incompatible output shape: expected " << dshape << ", got " << (*out_shape)[0];

  return shape_is_known(dshape);
}

bool ConcatType(const nnvm::NodeAttrs& attrs,
                std::vector<int>* in_type,
                std::vector<int>* out_type);

struct NumpyConcatGrad {
  const char* op_name;
  std::vector<nnvm::NodeEntry> operator()(const nnvm::ObjectPtr& n,
                                          const std::vector<nnvm::NodeEntry>& ograds) const {
    CHECK_EQ(ograds.size(), 1);
    std::vector<nnvm::NodeEntry> heads(ograds.begin(), ograds.end());
    return MakeGradNode(op_name, n, heads, n->attrs.dict);
  }
};

bool NumpyColumnStackType(const nnvm::NodeAttrs& attrs,
                          std::vector<int>* in_type,
                          std::vector<int>* out_type) {
  const NumpyColumnStackParam& param = nnvm::get<NumpyColumnStackParam>(attrs.parsed);
  CHECK_EQ(in_type->size(), param.num_args);
  CHECK_EQ(out_type->size(), 1);
  int dtype = -1;
  for (int i = 0; i < param.num_args; i++) {
    if (dtype == -1) {
      dtype = in_type->at(i);
    }
  }
  if (dtype == -1) {
    dtype = out_type->at(0);
  }
  for (int i = 0; i < param.num_args; i++) {
    TYPE_ASSIGN_CHECK(*in_type, i, dtype);
  }
  TYPE_ASSIGN_CHECK(*out_type, 0, dtype);
  return dtype != -1;
}

bool NumpyColumnStackShape(const nnvm::NodeAttrs& attrs,
                           mxnet::ShapeVector* in_attrs,
                           mxnet::ShapeVector* out_attrs) {
  CHECK_EQ(out_attrs->size(), 1U);
  const NumpyColumnStackParam& param = nnvm::get<NumpyColumnStackParam>(attrs.parsed);
  CHECK_EQ(in_attrs->size(), param.num_args);
  std::vector<mxnet::TShape> in_attrs_tmp(param.num_args);
  TShape dshape;
  // For each array in the input, reshape to 2D if ndim < 2.
  for (int i = 0; i < param.num_args; i++) {
    if ((*in_attrs)[i].ndim() == 0) {
      in_attrs_tmp[i] = TShape(2, 1);
    } else if ((*in_attrs)[i].ndim() == 1) {
      // Transpose 1D row into a column.
      in_attrs_tmp[i]    = TShape(2, 1);
      in_attrs_tmp[i][0] = (*in_attrs)[i][0];
    } else {
      in_attrs_tmp[i] = (*in_attrs)[i];
    }
    TShape tmp(in_attrs_tmp[i].ndim(), -1);
    shape_assign(&dshape, tmp);
  }
  TShape tmp((*out_attrs)[0].ndim(), -1);
  shape_assign(&dshape, tmp);
  for (int i = 0; i < param.num_args; i++) {
    SHAPE_ASSIGN_CHECK(in_attrs_tmp, i, dshape)
  }
  SHAPE_ASSIGN_CHECK(*out_attrs, 0, dshape)
  if (dshape.ndim() == -1) {
    return false;
  }
  // Accumulate along column axis.
  int cnt = 0, sum = 0, pos = -1;
  for (int i = 0; i < param.num_args; i++) {
    TShape tmp = in_attrs_tmp[i];
    if (!dim_size_is_known(tmp, 1)) {
      cnt++;
      pos = i;
    } else {
      sum += tmp[1];
    }
    tmp[1] = -1;
    shape_assign(&dshape, tmp);
  }
  tmp = out_attrs->at(0);
  if (!dim_size_is_known(tmp, 1)) {
    cnt++;
    pos = -1;
  } else {
    sum += tmp[1];
  }
  tmp[1] = -1;
  shape_assign(&dshape, tmp);
  for (int i = 0; i < param.num_args; i++) {
    SHAPE_ASSIGN_CHECK(in_attrs_tmp, i, dshape)
  }
  SHAPE_ASSIGN_CHECK(*out_attrs, 0, dshape)
  dshape[1] = 0;
  if (!shape_is_known(dshape)) {
    return false;
  }
  dshape[1] = sum;
  if (cnt == 0) {
    SHAPE_ASSIGN_CHECK(*out_attrs, 0, dshape);
  } else if (cnt == 1) {
    // Infer missing dimension if only one column dimension of the input is missing
    if (pos >= 0) {
      in_attrs_tmp[pos][1] = out_attrs->at(0)[1] - sum;
    } else {
      out_attrs->at(0)[1] = sum;
    }
  } else {
    return false;
  }
  for (int i = 0; i < param.num_args; i++) {
    if (in_attrs->at(i).ndim() == 1) {
      in_attrs->at(i)[0] = in_attrs_tmp[i][1];
    } else if (in_attrs->at(i).ndim() >= 2) {
      in_attrs->at(i) = in_attrs_tmp[i];
    }
  }

  return true;
}

DMLC_REGISTER_PARAMETER(NumpyColumnStackParam);

NNVM_REGISTER_OP(_npi_column_stack)
    .describe(R"code()code" ADD_FILELINE)
    .set_attr_parser(ParamParser<NumpyColumnStackParam>)
    .set_num_inputs([](const nnvm::NodeAttrs& attrs) {
      const NumpyColumnStackParam& param = dmlc::get<NumpyColumnStackParam>(attrs.parsed);
      return static_cast<uint32_t>(param.num_args);
    })
    .set_num_outputs(1)
    .set_attr<nnvm::FListInputNames>("FListInputNames",
                                     [](const nnvm::NodeAttrs& attrs) {
                                       int num_args =
                                           dmlc::get<NumpyColumnStackParam>(attrs.parsed).num_args;
                                       std::vector<std::string> ret;
                                       ret.reserve(num_args);
                                       for (int i = 0; i < num_args; ++i) {
                                         ret.push_back(std::string("arg") + std::to_string(i));
                                       }
                                       return ret;
                                     })
    .set_attr<std::string>("key_var_num_args", "num_args")
    .set_attr<mxnet::FInferShape>("FInferShape", NumpyColumnStackShape)
    .set_attr<nnvm::FInferType>("FInferType", NumpyColumnStackType)
    .set_attr<FCompute>("FCompute<cpu>", NumpyColumnStackForward<cpu>)
    .set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseNone{"_backward_np_column_stack"})
    .add_argument("data", "NDArray-or-Symbol[]", "List of arrays to column_stack")
    .add_arguments(NumpyColumnStackParam::__FIELDS__());

NNVM_REGISTER_OP(_backward_np_column_stack)
    .set_attr_parser(ParamParser<NumpyColumnStackParam>)
    .set_num_inputs(1)
    .set_num_outputs([](const nnvm::NodeAttrs& attrs) {
      const NumpyColumnStackParam& param = dmlc::get<NumpyColumnStackParam>(attrs.parsed);
      return static_cast<uint32_t>(param.num_args);
    })
    .set_attr<nnvm::TIsBackward>("TIsBackward", true)
    .set_attr<FCompute>("FCompute<cpu>", NumpyColumnStackBackward<cpu>);

DMLC_REGISTER_PARAMETER(NumpyVstackParam);

bool NumpyVstackType(const nnvm::NodeAttrs& attrs,
                     std::vector<int>* in_type,
                     std::vector<int>* out_type) {
  const NumpyVstackParam& param = nnvm::get<NumpyVstackParam>(attrs.parsed);
  CHECK_EQ(in_type->size(), param.num_args);
  CHECK_EQ(out_type->size(), 1);
  int dtype = -1;
  for (int i = 0; i < param.num_args; i++) {
    if (dtype == -1) {
      dtype = in_type->at(i);
    }
  }
  if (dtype == -1) {
    dtype = out_type->at(0);
  }
  for (int i = 0; i < param.num_args; i++) {
    TYPE_ASSIGN_CHECK(*in_type, i, dtype);
  }
  TYPE_ASSIGN_CHECK(*out_type, 0, dtype);
  return dtype != -1;
}

bool NumpyVstackShape(const nnvm::NodeAttrs& attrs,
                      mxnet::ShapeVector* in_attrs,
                      mxnet::ShapeVector* out_attrs) {
  CHECK_EQ(out_attrs->size(), 1U);
  const NumpyVstackParam& param = nnvm::get<NumpyVstackParam>(attrs.parsed);
  CHECK_EQ(in_attrs->size(), param.num_args);
  std::vector<mxnet::TShape> in_attrs_tmp(param.num_args);
  TShape dshape;
  for (int i = 0; i < param.num_args; i++) {
    if ((*in_attrs)[i].ndim() == 0) {
      in_attrs_tmp[i] = TShape(2, 1);
    } else if ((*in_attrs)[i].ndim() == 1) {
      in_attrs_tmp[i]    = TShape(2, 1);
      in_attrs_tmp[i][1] = (*in_attrs)[i][0];
    } else {
      in_attrs_tmp[i] = (*in_attrs)[i];
    }
    TShape tmp(in_attrs_tmp[i].ndim(), -1);
    shape_assign(&dshape, tmp);
  }
  TShape tmp((*out_attrs)[0].ndim(), -1);
  shape_assign(&dshape, tmp);
  for (int i = 0; i < param.num_args; i++) {
    SHAPE_ASSIGN_CHECK(in_attrs_tmp, i, dshape)
  }
  SHAPE_ASSIGN_CHECK(*out_attrs, 0, dshape)
  if (dshape.ndim() == -1) {
    return false;
  }
  index_t cnt = 0, sum = 0, pos = -1;
  for (int i = 0; i < param.num_args; i++) {
    TShape tmp = in_attrs_tmp[i];
    if (!dim_size_is_known(tmp, 0)) {
      cnt++;
      pos = i;
    } else {
      sum += tmp[0];
    }
    tmp[0] = -1;
    shape_assign(&dshape, tmp);
  }
  tmp = out_attrs->at(0);
  if (!dim_size_is_known(tmp, 0)) {
    cnt++;
    pos = -1;
  } else {
    sum += tmp[0];
  }
  tmp[0] = -1;
  shape_assign(&dshape, tmp);
  for (int i = 0; i < param.num_args; i++) {
    SHAPE_ASSIGN_CHECK(in_attrs_tmp, i, dshape)
  }
  SHAPE_ASSIGN_CHECK(*out_attrs, 0, dshape)
  dshape[0] = 0;
  if (!shape_is_known(dshape)) {
    return false;
  }

  dshape[0] = sum;
  if (cnt == 0) {
    SHAPE_ASSIGN_CHECK(*out_attrs, 0, dshape);
  } else if (cnt == 1) {
    if (pos >= 0) {
      in_attrs_tmp[pos][0] = out_attrs->at(0)[0] - sum;
    } else {
      out_attrs->at(0)[0] = sum;
    }
  } else {
    return false;
  }

  for (int i = 0; i < param.num_args; i++) {
    if (in_attrs->at(i).ndim() == 1) {
      in_attrs->at(i)[0] = in_attrs_tmp[i][1];
    } else if (in_attrs->at(i).ndim() >= 2) {
      in_attrs->at(i) = in_attrs_tmp[i];
    }
  }

  return true;
}

NNVM_REGISTER_OP(_npi_vstack)
    .describe(R"code()code" ADD_FILELINE)
    .set_attr_parser(ParamParser<NumpyVstackParam>)
    .set_num_inputs([](const nnvm::NodeAttrs& attrs) {
      const NumpyVstackParam& param = dmlc::get<NumpyVstackParam>(attrs.parsed);
      return static_cast<uint32_t>(param.num_args);
    })
    .set_num_outputs(1)
    .set_attr<nnvm::FListInputNames>("FListInputNames",
                                     [](const nnvm::NodeAttrs& attrs) {
                                       int num_args =
                                           dmlc::get<NumpyVstackParam>(attrs.parsed).num_args;
                                       std::vector<std::string> ret;
                                       ret.reserve(num_args);
                                       for (int i = 0; i < num_args; i++) {
                                         ret.push_back(std::string("arg") + std::to_string(i));
                                       }
                                       return ret;
                                     })
    .set_attr<std::string>("key_var_num_args", "num_args")
    .set_attr<mxnet::FInferShape>("FInferShape", NumpyVstackShape)
    .set_attr<nnvm::FInferType>("FInferType", NumpyVstackType)
    .set_attr<FCompute>("FCompute<cpu>", NumpyVstackForward<cpu>)
    .set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseNone{"_backward_np_vstack"})
    .add_argument("data", "NDArray-or-Symbol[]", "List of arrays to vstack")
    .add_arguments(NumpyVstackParam::__FIELDS__());

NNVM_REGISTER_OP(_backward_np_vstack)
    .set_attr_parser(ParamParser<NumpyVstackParam>)
    .set_num_inputs(1)
    .set_num_outputs([](const nnvm::NodeAttrs& attrs) {
      const NumpyVstackParam& param = dmlc::get<NumpyVstackParam>(attrs.parsed);
      return static_cast<uint32_t>(param.num_args);
    })
    .set_attr<nnvm::TIsBackward>("TIsBackward", true)
    .set_attr<FCompute>("FCompute<cpu>", NumpyVstackBackward<cpu>);

NNVM_REGISTER_OP(_npi_hstack)
    .describe(R"code(Stack tensors horizontally (in second dimension))code" ADD_FILELINE)
    .set_num_inputs([](const NodeAttrs& attrs) {
      const ConcatParam& params = nnvm::get<ConcatParam>(attrs.parsed);
      return params.num_args;
    })
    .set_num_outputs(1)
    .set_attr_parser(ParamParser<ConcatParam>)
    .set_attr<nnvm::FListInputNames>("FListInputNames",
                                     [](const NodeAttrs& attrs) {
                                       const ConcatParam& params =
                                           nnvm::get<ConcatParam>(attrs.parsed);
                                       std::vector<std::string> ret;
                                       ret.reserve(params.num_args);
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
    .set_attr<mxnet::FInferShape>("FInferShape", HStackShape)
    .set_attr<FCompute>("FCompute<cpu>", HStackCompute<cpu>)
    .set_attr<nnvm::FGradient>("FGradient", NumpyConcatGrad{"_backward_np_hstack"})
    .add_argument("data", "NDArray-or-Symbol[]", "List of arrays to concatenate")
    .add_arguments(ConcatParam::__FIELDS__());

NNVM_REGISTER_OP(_backward_np_hstack)
    .set_num_outputs([](const NodeAttrs& attrs) {
      const ConcatParam& params = nnvm::get<ConcatParam>(attrs.parsed);
      return params.num_args;
    })
    .set_attr_parser(ParamParser<ConcatParam>)
    .set_attr<nnvm::TIsBackward>("TIsBackward", true)
    .set_attr<FCompute>("FCompute<cpu>", HStackGradCompute<cpu>);

NNVM_REGISTER_OP(_npi_dstack)
    .describe(R"code(Stack tensors in sequence depthwise (in third dimension))code" ADD_FILELINE)
    .set_num_inputs([](const NodeAttrs& attrs) {
      const ConcatParam& params = nnvm::get<ConcatParam>(attrs.parsed);
      return params.num_args;
    })
    .set_num_outputs(1)
    .set_attr_parser(ParamParser<ConcatParam>)
    .set_attr<nnvm::FListInputNames>("FListInputNames",
                                     [](const NodeAttrs& attrs) {
                                       const ConcatParam& params =
                                           nnvm::get<ConcatParam>(attrs.parsed);
                                       std::vector<std::string> ret;
                                       ret.reserve(params.num_args);
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
    .set_attr<mxnet::FInferShape>("FInferShape", DStackShape)
    .set_attr<FCompute>("FCompute<cpu>", DStackCompute<cpu>)
    .set_attr<nnvm::FGradient>("FGradient", NumpyConcatGrad{"_backward_np_dstack"})
    .add_argument("data", "NDArray-or-Symbol[]", "List of arrays to concatenate")
    .add_arguments(ConcatParam::__FIELDS__());

NNVM_REGISTER_OP(_backward_np_dstack)
    .set_num_outputs([](const NodeAttrs& attrs) {
      const ConcatParam& params = nnvm::get<ConcatParam>(attrs.parsed);
      return params.num_args;
    })
    .set_attr_parser(ParamParser<ConcatParam>)
    .set_attr<nnvm::TIsBackward>("TIsBackward", true)
    .set_attr<FCompute>("FCompute<cpu>", DStackGradCompute<cpu>);

DMLC_REGISTER_PARAMETER(NumpyTrilindicesParam);

inline bool TrilindicesOpType(const nnvm::NodeAttrs& attrs,
                              std::vector<int>* in_attrs,
                              std::vector<int>* out_attrs) {
  CHECK_EQ(in_attrs->size(), 0U);
  CHECK_EQ(out_attrs->size(), 2U);

  TYPE_ASSIGN_CHECK(*out_attrs, 0, mshadow::kInt64);
  TYPE_ASSIGN_CHECK(*out_attrs, 1, mshadow::kInt64);

  return true;
}

inline bool TrilindicesOpShape(const nnvm::NodeAttrs& attrs,
                               mxnet::ShapeVector* in_attrs,
                               mxnet::ShapeVector* out_attrs) {
  CHECK_EQ(in_attrs->size(), 0U);
  CHECK_EQ(out_attrs->size(), 2U);

  const NumpyTrilindicesParam& param = nnvm::get<NumpyTrilindicesParam>(attrs.parsed);

  index_t n = param.n;
  index_t m = param.m;
  index_t k = param.k;

  index_t length = 0;
  index_t end    = k;
  for (index_t i = 0; i < n; i++) {
    index_t mi = std::min(end, m - 1);
    if (mi >= 0)
      length += mi + 1;
    end++;
  }
  mxnet::TShape oshape;
  oshape = mxnet::TShape(1, length);

  SHAPE_ASSIGN_CHECK(*out_attrs, 0, oshape);
  SHAPE_ASSIGN_CHECK(*out_attrs, 1, oshape);

  return shape_is_known(out_attrs->at(0)) && shape_is_known(out_attrs->at(1));
}

NNVM_REGISTER_OP(_npi_tril_indices)
    .set_attr_parser(ParamParser<NumpyTrilindicesParam>)
    .set_num_inputs(0)
    .set_num_outputs(2)
    .set_attr<mxnet::FInferShape>("FInferShape", TrilindicesOpShape)
    .set_attr<nnvm::FInferType>("FInferType", TrilindicesOpType)
    .set_attr<FCompute>("FCompute<cpu>", TrilindicesOpForward<cpu>)
    .set_attr<FResourceRequest>("FResourceRequest",
                                [](const NodeAttrs& n) {
                                  return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
                                })
    .add_arguments(NumpyTrilindicesParam::__FIELDS__());

inline bool NumpyRollShape(const nnvm::NodeAttrs& attrs,
                           mxnet::ShapeVector* in_attrs,
                           mxnet::ShapeVector* out_attrs) {
  using namespace mshadow;
  const NumpyRollParam& param = nnvm::get<NumpyRollParam>(attrs.parsed);

  if (!param.shift.has_value()) {
    LOG(FATAL) << "roll missing 1 required positional argument: 'shift'.";
  }
  if (param.shift.value().ndim() > 1 && param.axis.has_value() &&
      param.axis.value().ndim() != param.shift.value().ndim()) {
    LOG(FATAL) << "shift and `axis` must be a tuple of the same size.";
  }
  if (!param.axis.has_value() && param.shift.has_value() && param.shift.value().ndim() > 1) {
    LOG(FATAL) << "shift must be an int.";
  }
  if (param.axis.has_value()) {
    mxnet::TShape axes(param.axis.value());
    const index_t ndim = (*in_attrs)[0].ndim();
    for (index_t i = 0; i < axes.ndim(); i++) {
      if (axes[i] < 0) {
        axes[i] += ndim;
      }
    }
    std::sort(axes.begin(), axes.end());
    for (index_t i = 1; i < axes.ndim(); i++) {
      CHECK_LT(axes[i - 1], axes[i]) << "axes have duplicates " << axes;
    }
    CHECK_LT(axes[axes.ndim() - 1], ndim)
        << "axis " << axes[axes.ndim() - 1] << " Exceeds input dimensions " << (*in_attrs)[0];
    CHECK_GE(axes[0], 0) << "Reduction axis " << param.axis.value() << " Exceeds input dimensions "
                         << (*in_attrs)[0];
  }
  return ElemwiseShape<1, 1>(attrs, in_attrs, out_attrs);
}

NNVM_REGISTER_OP(_npi_roll)
    .set_num_inputs(1)
    .set_num_outputs(1)
    .set_attr_parser(ParamParser<NumpyRollParam>)
    .set_attr<nnvm::FListInputNames>("FListInputNames",
                                     [](const NodeAttrs& attrs) {
                                       return std::vector<std::string>{"data"};
                                     })
    .set_attr<mxnet::FInferShape>("FInferShape", NumpyRollShape)
    .set_attr<nnvm::FInferType>("FInferType", ElemwiseType<1, 1>)
    .set_attr<mxnet::FCompute>("FCompute<cpu>", NumpyRollCompute<cpu>)
    .set_attr<nnvm::FGradient>(
        "FGradient",
        [](const nnvm::ObjectPtr& n, const std::vector<nnvm::NodeEntry>& ograds) {
          const NumpyRollParam& param = nnvm::get<NumpyRollParam>(n->attrs.parsed);
          if (!param.shift.has_value()) {
            LOG(FATAL) << "roll missing 1 required positional argument: 'shift'.";
          }
          mxnet::TShape shifts(param.shift.value());
          for (int i = 0; i < shifts.ndim(); ++i) {
            shifts[i] = -shifts[i];
          }
          std::ostringstream os1;
          os1 << dmlc::optional<mxnet::TShape>(shifts);
          std::ostringstream os2;
          os2 << param.axis;
          return MakeNonlossGradNode(
              "_npi_roll", n, ograds, {}, {{"shift", os1.str()}, {"axis", os2.str()}});
        })
    .set_attr<FResourceRequest>("FResourceRequest",
                                [](const NodeAttrs& n) {
                                  return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
                                })
    .add_argument("data", "NDArray-or-Symbol", "Input ndarray")
    .add_arguments(NumpyRollParam::__FIELDS__());

bool NumpyRollaxisShape(const nnvm::NodeAttrs& attrs,
                        mxnet::ShapeVector* in_attrs,
                        mxnet::ShapeVector* out_attrs) {
  const NumpyRollaxisParam& param = nnvm::get<NumpyRollaxisParam>(attrs.parsed);
  // check 1 input, 1 output
  CHECK_EQ(in_attrs->size(), 1U);
  CHECK_EQ(out_attrs->size(), 1U);

  // check transpose dimentions no more than 6
  mxnet::TShape& shp = (*in_attrs)[0];

  // check axis and start range
  CHECK_GE(param.axis, -shp.ndim())
      << "axis must be within the range of " << -shp.ndim() << " and " << shp.ndim() - 1;
  CHECK_LT(param.axis, shp.ndim())
      << "axis must be within the range of " << -shp.ndim() << " and " << shp.ndim() - 1;
  CHECK_GE(param.start, -shp.ndim())
      << "start must be within the range of " << -shp.ndim() << " and " << shp.ndim();
  CHECK_LE(param.start, shp.ndim())
      << "start must be within the range of " << -shp.ndim() << " and " << shp.ndim();

  // generate output shape
  mxnet::TShape ret(shp.ndim(), -1);
  mxnet::TShape axes;

  axes = NumpyRollaxisShapeImpl(param.axis, param.start, shp.ndim());
  for (int i = 0; i < shp.ndim(); ++i) {
    CHECK(axes[i] < static_cast<int64_t>(shp.ndim()));
    ret[i] = shp[axes[i]];
  }
  SHAPE_ASSIGN_CHECK(*out_attrs, 0, ret);
  return shape_is_known(ret);
}

NNVM_REGISTER_OP(_npi_rollaxis)
    .describe(R"code(Roll the specified axis backwards, 
until it lies in a given position.)code" ADD_FILELINE)
    .set_num_inputs(1)
    .set_num_outputs(1)
    .set_attr_parser(ParamParser<NumpyRollaxisParam>)
    .set_attr<nnvm::FListInputNames>("FListInputNames",
                                     [](const NodeAttrs& attrs) {
                                       return std::vector<std::string>{"data"};
                                     })
    .set_attr<mxnet::FInferShape>("FInferShape", NumpyRollaxisShape)
    .set_attr<nnvm::FInferType>("FInferType", ElemwiseType<1, 1>)
    .set_attr<FCompute>("FCompute<cpu>", NumpyRollaxisCompute<cpu>)
    .set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseNone{"_npi_rollaxis_backward"})
    .set_attr<FResourceRequest>("FResourceRequest",
                                [](const NodeAttrs& attrs) {
                                  return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
                                })
    .add_argument("data", "NDArray-or-Symbol", "Input ndarray")
    .add_arguments(NumpyRollaxisParam::__FIELDS__());

NNVM_REGISTER_OP(_npi_rollaxis_backward)
    .set_num_inputs(1)
    .set_num_outputs(1)
    .set_attr_parser(ParamParser<NumpyRollaxisParam>)
    .set_attr<nnvm::TIsBackward>("TIsBackward", true)
    .set_attr<FCompute>("FCompute<cpu>", NumpyRollaxisBackward<cpu>)
    .set_attr<FResourceRequest>("FResourceRequest", [](const NodeAttrs& attrs) {
      return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
    });

template <>
void NumpyFlipForwardImpl<cpu>(const OpContext& ctx,
                               const std::vector<TBlob>& inputs,
                               const std::vector<TBlob>& outputs,
                               const std::vector<index_t>& stride_,
                               const std::vector<index_t>& trailing_,
                               const index_t& flip_index) {
  mshadow::Stream<cpu>* s = ctx.get_stream<cpu>();
  MSHADOW_TYPE_SWITCH(outputs[0].type_flag_, DType, {
    mxnet_op::Kernel<reverse, cpu>::Launch(s,
                                           inputs[0].Size(),
                                           flip_index,
                                           inputs[0].dptr<DType>(),
                                           outputs[0].dptr<DType>(),
                                           stride_.data(),
                                           trailing_.data());
  });
}

DMLC_REGISTER_PARAMETER(FlipParam);

NNVM_REGISTER_OP(_npi_flip)
    .set_num_outputs(1)
    .set_num_inputs(1)
    .set_attr_parser(ParamParser<FlipParam>)
    .set_attr<nnvm::FListInputNames>("FListInputNames",
                                     [](const NodeAttrs& attrs) {
                                       return std::vector<std::string>{"data"};
                                     })
    .set_attr<FResourceRequest>("FResourceRequest",
                                [](const NodeAttrs& attrs) {
                                  return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
                                })
    .set_attr<mxnet::FInferShape>("FInferShape", ElemwiseShape<1, 1>)
    .set_attr<nnvm::FInferType>("FInferType", ElemwiseType<1, 1>)
    .set_attr<FCompute>("FCompute<cpu>", NumpyFlipForward<cpu>)
    .set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseNone{"_backward_npi_flip"})
    .add_argument("data", "NDArray-or-Symbol", "Input data array")
    .add_arguments(FlipParam::__FIELDS__());

NNVM_REGISTER_OP(_backward_npi_flip)
    .set_num_inputs(1)
    .set_num_outputs(1)
    .set_attr_parser(ParamParser<FlipParam>)
    .set_attr<nnvm::TIsBackward>("TIsBackward", true)
    .set_attr<FResourceRequest>("FResourceRequest",
                                [](const NodeAttrs& attrs) {
                                  return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
                                })
    .set_attr<FCompute>("FCompute<cpu>", NumpyFlipForward<cpu>);

bool NumpyMoveaxisShape(const nnvm::NodeAttrs& attrs,
                        mxnet::ShapeVector* in_attrs,
                        mxnet::ShapeVector* out_attrs) {
  const NumpyMoveaxisParam& param = nnvm::get<NumpyMoveaxisParam>(attrs.parsed);
  CHECK_EQ(in_attrs->size(), 1U);
  CHECK_EQ(out_attrs->size(), 1U);
  mxnet::TShape& shp = (*in_attrs)[0];
  CHECK_EQ(param.source.ndim(), param.destination.ndim()) << "source and destination not equal.";
  mxnet::TShape ret(shp.ndim(), -1);
  mxnet::TShape axes;
  axes = NumpyMoveaxisShapeImpl(attrs, shp.ndim());
  for (int i = 0; i < shp.ndim(); ++i) {
    CHECK(axes[i] < static_cast<int64_t>(shp.ndim()));
    ret[i] = shp[axes[i]];
  }
  SHAPE_ASSIGN_CHECK(*out_attrs, 0, ret);
  return shape_is_known(ret);
}

NNVM_REGISTER_OP(_npi_moveaxis)
    .describe(R"code(Move axes of an array to new positions.
Other axes remain in their original order.
)code" ADD_FILELINE)
    .set_num_inputs(1)
    .set_num_outputs(1)
    .set_attr_parser(ParamParser<NumpyMoveaxisParam>)
    .set_attr<mxnet::FInferShape>("FInferShape", NumpyMoveaxisShape)
    .set_attr<nnvm::FInferType>("FInferType", ElemwiseType<1, 1>)
    .set_attr<nnvm::FGradient>(
        "FGradient",
        [](const nnvm::ObjectPtr& n, const std::vector<nnvm::NodeEntry>& ograds) {
          const NumpyMoveaxisParam& param = nnvm::get<NumpyMoveaxisParam>(n->attrs.parsed);
          std::ostringstream os1;
          os1 << param.source;
          std::ostringstream os2;
          os2 << param.destination;
          return MakeNonlossGradNode(
              "_npi_moveaxis", n, ograds, {}, {{"source", os2.str()}, {"destination", os1.str()}});
        })
    .set_attr<FCompute>("FCompute<cpu>", NumpyMoveaxisCompute<cpu>)
    .set_attr<nnvm::FListInputNames>("FListInputNames",
                                     [](const NodeAttrs& attrs) {
                                       return std::vector<std::string>{"a"};
                                     })
    .add_argument("a", "NDArray-or-Symbol", "Source input")
    .add_arguments(NumpyMoveaxisParam::__FIELDS__());

inline bool NumpyRot90Shape(const nnvm::NodeAttrs& attrs,
                            mxnet::ShapeVector* in_attrs,
                            mxnet::ShapeVector* out_attrs) {
  using namespace mshadow;
  const NumpyRot90Param& param = nnvm::get<NumpyRot90Param>(attrs.parsed);
  CHECK_EQ(in_attrs->size(), 1U);
  CHECK_EQ(out_attrs->size(), 1U);
  mxnet::TShape& shp = (*in_attrs)[0];
  if (!param.axes.has_value() || param.axes.value().ndim() != 2) {
    LOG(FATAL) << "The length of axes must be 2.";
  }
  int real_k(param.k);
  real_k = real_k % 4;
  if (real_k < 0) {
    real_k += 4;
  }

  mxnet::TShape res(shp);
  mxnet::TShape real_axes(param.axes.value());
  for (index_t i = 0; i < real_axes.ndim(); i++) {
    if (real_axes[i] < 0) {
      real_axes[i] += shp.ndim();
    }
  }

  CHECK_NE(real_axes[0], real_axes[1]) << "axes have duplicates " << real_axes;
  if (real_axes[0] > shp.ndim() || real_axes[1] > shp.ndim() || real_axes[0] < 0 ||
      real_axes[1] < 0) {
    LOG(FATAL) << "Axes out of range for array of ndim";
  }

  if (real_k % 2 == 0) {
    SHAPE_ASSIGN_CHECK(*out_attrs, 0, res);
    return shape_is_known(res);
  }

  res[real_axes[0]] += res[real_axes[1]];
  res[real_axes[1]] = res[real_axes[0]] - res[real_axes[1]];
  res[real_axes[0]] -= res[real_axes[1]];
  SHAPE_ASSIGN_CHECK(*out_attrs, 0, res);
  return shape_is_known(res);
}

NNVM_REGISTER_OP(_npi_rot90)
    .set_num_inputs(1)
    .set_num_outputs(1)
    .set_attr_parser(ParamParser<NumpyRot90Param>)
    .set_attr<nnvm::FListInputNames>("FListInputNames",
                                     [](const NodeAttrs& attrs) {
                                       return std::vector<std::string>{"data"};
                                     })
    .set_attr<mxnet::FInferShape>("FInferShape", NumpyRot90Shape)
    .set_attr<nnvm::FInferType>("FInferType", ElemwiseType<1, 1>)
    .set_attr<mxnet::FCompute>("FCompute<cpu>", NumpyRot90Compute<cpu>)
    .set_attr<nnvm::FGradient>(
        "FGradient",
        [](const nnvm::ObjectPtr& n, const std::vector<nnvm::NodeEntry>& ograds) {
          const NumpyRot90Param& param = nnvm::get<NumpyRot90Param>(n->attrs.parsed);
          std::ostringstream os1;
          os1 << param.k;
          std::ostringstream os2;
          os2 << param.axes;
          return MakeNonlossGradNode(
              "_npi_rot90", n, ograds, {}, {{"k", os1.str()}, {"axes", os2.str()}});
        })
    .set_attr<FResourceRequest>("FResourceRequest",
                                [](const NodeAttrs& n) {
                                  return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
                                })
    .add_argument("data", "NDArray-or-Symbol", "Input ndarray")
    .add_arguments(NumpyRot90Param::__FIELDS__());

inline bool HSplitOpShape(const nnvm::NodeAttrs& attrs,
                          mxnet::ShapeVector* in_attrs,
                          mxnet::ShapeVector* out_attrs) {
  using namespace mshadow;
  CHECK_EQ(in_attrs->size(), 1U);
  mxnet::TShape dshape = in_attrs->at(split_enum::kData);
  CHECK_GE(dshape.ndim(), 1U) << "ValueError: hsplit only works on arrays of 1 or more dimensions";
  if (!mxnet::ndim_is_known(dshape))
    return false;
  int real_axis;
  if (dshape.ndim() > 1) {
    real_axis = 1;
  } else {
    real_axis = 0;
  }
  return SplitOpShapeImpl(attrs, in_attrs, out_attrs, real_axis);
}

NNVM_REGISTER_OP(_npi_hsplit)
    .set_attr_parser(ParamParser<SplitParam>)
    .set_num_inputs(1)
    .set_num_outputs(SplitNumOutputs)
    .set_attr<nnvm::FListInputNames>("FListInputNames",
                                     [](const NodeAttrs& attrs) {
                                       return std::vector<std::string>{"data"};
                                     })
    .set_attr<mxnet::FInferShape>("FInferShape", HSplitOpShape)
    .set_attr<nnvm::FInferType>("FInferType", SplitOpType)
    .set_attr<FCompute>("FCompute<cpu>", HSplitOpForward<cpu>)
    .set_attr<FResourceRequest>("FResourceRequest",
                                [](const NodeAttrs& n) {
                                  return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
                                })
    .set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseNone{"_npi_hsplit_backward"})
    .add_argument("data", "NDArray-or-Symbol", "The input")
    .add_arguments(SplitParam::__FIELDS__());

NNVM_REGISTER_OP(_npi_hsplit_backward)
    .set_attr_parser(ParamParser<SplitParam>)
    .set_num_inputs(SplitNumOutputs)
    .set_num_outputs(1)
    .set_attr<nnvm::TIsBackward>("TIsBackward", true)
    .set_attr<FResourceRequest>("FResourceRequest",
                                [](const NodeAttrs& n) {
                                  return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
                                })
    .set_attr<FCompute>("FCompute<cpu>", HSplitOpBackward<cpu>);

inline bool DSplitOpShape(const nnvm::NodeAttrs& attrs,
                          mxnet::ShapeVector* in_attrs,
                          mxnet::ShapeVector* out_attrs) {
  using namespace mshadow;
  CHECK_EQ(in_attrs->size(), 1U);
  mxnet::TShape dshape = in_attrs->at(split_enum::kData);
  if (!mxnet::ndim_is_known(dshape))
    return false;
  CHECK(dshape.ndim() >= 3) << "ValueError: dsplit only works on arrays of 3 or more dimensions";
  return SplitOpShapeImpl(attrs, in_attrs, out_attrs, 2);
}

NNVM_REGISTER_OP(_npi_dsplit)
    .set_attr_parser(ParamParser<SplitParam>)
    .set_num_inputs(1)
    .set_num_outputs(SplitNumOutputs)
    .set_attr<nnvm::FListInputNames>("FListInputNames",
                                     [](const NodeAttrs& attrs) {
                                       return std::vector<std::string>{"data"};
                                     })
    .set_attr<mxnet::FInferShape>("FInferShape", DSplitOpShape)
    .set_attr<nnvm::FInferType>("FInferType", SplitOpType)
    .set_attr<FCompute>("FCompute<cpu>", SplitOpForward<cpu>)
    .set_attr<FResourceRequest>("FResourceRequest",
                                [](const NodeAttrs& n) {
                                  return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
                                })
    .set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseNone{"_split_v2_backward"})
    .add_argument("data", "NDArray-or-Symbol", "The input")
    .add_arguments(SplitParam::__FIELDS__());

NNVM_REGISTER_OP(_npi_diag)
    .set_attr_parser(ParamParser<NumpyDiagParam>)
    .set_num_inputs(1)
    .set_num_outputs(1)
    .set_attr<nnvm::FListInputNames>("FListInputNames",
                                     [](const NodeAttrs& attrs) {
                                       return std::vector<std::string>{"data"};
                                     })
    .set_attr<mxnet::FInferShape>("FInferShape", NumpyDiagOpShape)
    .set_attr<nnvm::FInferType>("FInferType", NumpyDiagOpType)
    .set_attr<FCompute>("FCompute<cpu>", NumpyDiagOpForward<cpu>)
    .set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseNone{"_backward_npi_diag"})
    .add_argument("data", "NDArray-or-Symbol", "Input ndarray")
    .add_arguments(NumpyDiagParam::__FIELDS__());

NNVM_REGISTER_OP(_backward_npi_diag)
    .set_attr_parser(ParamParser<NumpyDiagParam>)
    .set_num_inputs(1)
    .set_num_outputs(1)
    .set_attr<nnvm::TIsBackward>("TIsBackward", true)
    .set_attr<FCompute>("FCompute<cpu>", NumpyDiagOpBackward<cpu>);

NNVM_REGISTER_OP(_npi_diagonal)
    .set_attr_parser(ParamParser<NumpyDiagonalParam>)
    .set_num_inputs(1)
    .set_num_outputs(1)
    .set_attr<nnvm::FListInputNames>("FListInputNames",
                                     [](const NodeAttrs& attrs) {
                                       return std::vector<std::string>{"data"};
                                     })
    .set_attr<mxnet::FInferShape>("FInferShape", NumpyDiagonalOpShape)
    .set_attr<nnvm::FInferType>("FInferType", NumpyDiagonalOpType)
    .set_attr<FCompute>("FCompute<cpu>", NumpyDiagonalOpForward<cpu>)
    .set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseNone{"_backward_npi_diagonal"})
    .add_argument("data", "NDArray-or-Symbol", "Input ndarray")
    .add_arguments(NumpyDiagonalParam::__FIELDS__());

NNVM_REGISTER_OP(_backward_npi_diagonal)
    .set_attr_parser(ParamParser<NumpyDiagonalParam>)
    .set_num_inputs(1)
    .set_num_outputs(1)
    .set_attr<nnvm::TIsBackward>("TIsBackward", true)
    .set_attr<FCompute>("FCompute<cpu>", NumpyDiagonalOpBackward<cpu>);

NNVM_REGISTER_OP(_npi_diagflat)
    .set_attr_parser(ParamParser<NumpyDiagflatParam>)
    .set_num_inputs(1)
    .set_num_outputs(1)
    .set_attr<nnvm::FListInputNames>("FListInputNames",
                                     [](const NodeAttrs& attrs) {
                                       return std::vector<std::string>{"data"};
                                     })
    .set_attr<mxnet::FInferShape>("FInferShape", NumpyDiagflatOpShape)
    .set_attr<nnvm::FInferType>("FInferType", NumpyDiagflatOpType)
    .set_attr<FCompute>("FCompute<cpu>", NumpyDiagflatOpForward<cpu>)
    .set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseNone{"_backward_npi_diagflat"})
    .add_argument("data", "NDArray-or-Symbol", "Input ndarray")
    .add_arguments(NumpyDiagflatParam::__FIELDS__());

NNVM_REGISTER_OP(_backward_npi_diagflat)
    .set_attr_parser(ParamParser<NumpyDiagflatParam>)
    .set_num_inputs(1)
    .set_num_outputs(1)
    .set_attr<nnvm::TIsBackward>("TIsBackward", true)
    .set_attr<FCompute>("FCompute<cpu>", NumpyDiagflatOpBackward<cpu>);

bool NumpyDiagIndicesFromShape(const nnvm::NodeAttrs& attrs,
                               mxnet::ShapeVector* in_attrs,
                               mxnet::ShapeVector* out_attrs) {
  CHECK_EQ(in_attrs->size(), 1U);
  CHECK_EQ(out_attrs->size(), 1U);

  const mxnet::TShape& ishape = (*in_attrs)[0];
  if (!mxnet::shape_is_known(ishape))
    return false;
  CHECK_GE(ishape.ndim(), 2) << "ValueError: Input array should be at least 2d";

  int size = ishape[0];
  for (int i = 1; i < ishape.ndim(); i++) {
    CHECK_EQ(ishape[i], size) << "ValueError: All dimensions of "
                                 "input must be of equal length";
  }

  mxnet::TShape oshape(2, -1);
  oshape[0] = ishape.ndim();
  oshape[1] = size;
  SHAPE_ASSIGN_CHECK(*out_attrs, 0, oshape);
  return shape_is_known(out_attrs->at(0));
}

bool NumpyDiagIndicesFromType(const nnvm::NodeAttrs& attrs,
                              std::vector<int>* in_attrs,
                              std::vector<int>* out_attrs) {
  CHECK_EQ(in_attrs->size(), 1U);
  CHECK_EQ(out_attrs->size(), 1U);

  TYPE_ASSIGN_CHECK(*out_attrs, 0, mshadow::kInt64);
  return out_attrs->at(0) != -1 && in_attrs->at(0) != -1;
}

NNVM_REGISTER_OP(_npi_diag_indices_from)
    .set_num_inputs(1)
    .set_num_outputs(1)
    .set_attr<nnvm::FListInputNames>("FListInputNames",
                                     [](const NodeAttrs& attrs) {
                                       return std::vector<std::string>{"data"};
                                     })
    .set_attr<mxnet::FInferShape>("FInferShape", NumpyDiagIndicesFromShape)
    .set_attr<nnvm::FInferType>("FInferType", NumpyDiagIndicesFromType)
    .set_attr<FCompute>("FCompute<cpu>", NumpyDiagIndicesFromForward<cpu>)
    .set_attr<nnvm::FGradient>("FGradient", MakeZeroGradNodes)
    .add_argument("data", "NDArray-or-Symbol", "Input ndarray");

}  // namespace op
}  // namespace mxnet
