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
 * Copyright (c) 2019 by Contributors
 * \file np_norm-.cc
 * \brief CPU registration of np.linalg.norm
 */

#include "./np_norm-inl.h"

namespace mxnet {
namespace op {

DMLC_REGISTER_PARAMETER(NumpyNormParam);

inline bool NumpyLpNormShape(const nnvm::NodeAttrs& attrs,
                             mxnet::ShapeVector *in_attrs,
                             mxnet::ShapeVector *out_attrs) {
  if (!shape_is_known((*in_attrs)[0])) return false;
  const NumpyNormParam& param = nnvm::get<NumpyNormParam>(attrs.parsed);
  const int ndim = (*in_attrs)[0].ndim();
  if ((!param.axis.has_value() && param.flag != 0 && ndim > 2) ||
      (param.axis.has_value() && param.axis.value().ndim() > 2))
    LOG(FATAL) << "Improper number of dimensions to norm.";
  if (!param.axis.has_value()) {
    if ((ndim == 0 && param.flag != 0) ||  // for scalar
        (ndim == 1 && (param.flag == 2)) ||
        (ndim >= 2 && (param.ord == 0 || param.ord > 2 || param.ord < -2))) {
      LOG(FATAL) << "Invalid norm order for inputs.";
    }
  } else {
    if ((param.axis.value().ndim() == 0 && param.flag != 0) ||  // for scalar
        (param.axis.value().ndim() == 1 && (param.flag == 2)) ||
        (param.axis.value().ndim() == 2 && (param.ord == 0 || param.ord > 2 || param.ord < -2))) {
      LOG(FATAL) << "Invalid norm order for inputs.";
    }
  }
  if (!param.keepdims && (*in_attrs)[0].ndim() == 1) {
    SHAPE_ASSIGN_CHECK(*out_attrs, 0, TShape(0, -1));
  } else {
    SHAPE_ASSIGN_CHECK(*out_attrs, 0,
                       ReduceAxesShapeImpl((*in_attrs)[0], param.axis, param.keepdims, false));
  }
  return true;
}

inline bool NumpyMatrixNormShape(const nnvm::NodeAttrs& attrs,
                                 mxnet::ShapeVector *in_attrs,
                                 mxnet::ShapeVector *out_attrs) {
  const NumpyNormParam& param = nnvm::get<NumpyNormParam>(attrs.parsed);
  const int ndim = (*in_attrs)[0].ndim();
  auto shape = swapMatDims((*in_attrs)[0], param.axis.value());
  if (param.axis.value().ndim() == 2) {
    int batch_dim = 1;
    int row_dim = (*in_attrs)[0][param.axis.value()[0]];
    int col_dim = (*in_attrs)[0][param.axis.value()[1]];
    TShape out_shape(ndim - (param.keepdims ? 0 : 2), 1);
    for (int i = 0; i < ndim - 2; ++i) {
      batch_dim *= shape[i];
    }
    if (param.keepdims) {
      out_shape = (*in_attrs)[0];
      out_shape[param.axis.value()[0]] = 1;
      out_shape[param.axis.value()[1]] = 1;
    } else {
      for (int i = 0; i < ndim - 2; ++i) {
        out_shape[i] = shape[i];
      }
    }
    int svd_dim = row_dim < col_dim ? row_dim : col_dim;
    SHAPE_ASSIGN_CHECK(*out_attrs, 0, out_shape);
    if (param.ord == 2 || param.ord == -2) {
      SHAPE_ASSIGN_CHECK(*out_attrs, 1, TShape({ batch_dim, row_dim, row_dim }));  // UT
      SHAPE_ASSIGN_CHECK(*out_attrs, 2, TShape({ batch_dim, svd_dim }));  // L
      SHAPE_ASSIGN_CHECK(*out_attrs, 3, TShape({ batch_dim, row_dim, col_dim }));  // V
    } else {
      TShape sum_shape = (*in_attrs)[0];
      TShape mat_axis = param.axis.value();
      int sum_dim = mat_axis[!(param.ord == 1 || param.ord == -1)];
      TShape small(3, 1);
      sum_shape[sum_dim] = 1;
      small[0] = sum_shape.ProdShape(0, sum_dim);
      small[2] = sum_shape.ProdShape(sum_dim + 1, sum_shape.ndim());
      SHAPE_ASSIGN_CHECK(*out_attrs, 1, small);  // sum
      SHAPE_ASSIGN_CHECK(*out_attrs, 2, TShape({ 0, 0 }));  // L
      SHAPE_ASSIGN_CHECK(*out_attrs, 3, TShape({ 0, 0, 0 }));  // V
    }
  } else {
    LOG(FATAL) << "Invalid norm or ord arguments.";
  }
  return true;
}

inline void assign_svd_empty(mxnet::ShapeVector *out_attrs) {
  SHAPE_ASSIGN_CHECK(*out_attrs, 1, TShape({ 0, 0, 0 }));  // UT
  SHAPE_ASSIGN_CHECK(*out_attrs, 2, TShape({ 0, 0 }));  // L
  SHAPE_ASSIGN_CHECK(*out_attrs, 3, TShape({ 0, 0, 0 }));  // V
}

bool NumpyNormType(const nnvm::NodeAttrs& attrs,
                   std::vector<int>* in_attrs,
                   std::vector<int>* out_attrs) {
  CHECK_EQ(in_attrs->size(), 1U);
  CHECK_EQ(out_attrs->size(), 4U);
  int in_type = in_attrs->at(0);
  int out_type;
  if (!common::is_float(in_type)) {
    out_type = in_type;
    LOG(WARNING) << "WARNING: Integer input to norm. This will result in integer "
                    "output which is different from standard NumPy behavior and "
                    "breaks gradient compute in backward. Please cast the input "
                    "to floating point types first.";
  } else {
    out_type = in_type;
  }
  for (int i = 0; i < 4; ++i) {
    TYPE_ASSIGN_CHECK(*out_attrs, i, out_type);
  }
  return out_attrs->at(0) != -1;
}

bool NumpyNormShape(const nnvm::NodeAttrs& attrs,
                    mxnet::ShapeVector *in_attrs,
                    mxnet::ShapeVector *out_attrs) {
  CHECK_EQ(in_attrs->size(), 1U);
  CHECK_EQ(out_attrs->size(), 4U);  // reduced, UT, S, V
  const NumpyNormParam& param = nnvm::get<NumpyNormParam>(attrs.parsed);
  if (!param.axis.has_value()) {
    if (param.flag == -2) {
      int ndim = param.keepdims ? (*in_attrs)[0].ndim() : 0;
      int sz = param.keepdims ? 1 : -1;
      SHAPE_ASSIGN_CHECK(*out_attrs, 0, TShape(ndim, sz));
      assign_svd_empty(out_attrs);
      return true;
    }
    if ((*in_attrs)[0].ndim() >= 2) {
      TShape axis(2, 0);
      axis[0] = (*in_attrs)[0].ndim() - 2;
      axis[1] = (*in_attrs)[0].ndim() - 1;
      const_cast<NumpyNormParam&>(param).axis = axis;
      return NumpyMatrixNormShape(attrs, in_attrs, out_attrs);
    } else {
      TShape axis(1, (*in_attrs)[0].ndim() - 1);
      const_cast<NumpyNormParam&>(param).axis = axis;
      assign_svd_empty(out_attrs);
      return NumpyLpNormShape(attrs, in_attrs, out_attrs);
    }
  } else {
    TShape axis(param.axis.value().ndim(), 0);
    for (int i = 0; i < param.axis.value().ndim(); ++i) {
      axis[i] = param.axis.value()[i] < 0 ?
                  (*in_attrs)[0].ndim() + param.axis.value()[i] :
                  param.axis.value()[i];
    }
    const_cast<NumpyNormParam&>(param).axis = axis;
    if (param.axis.value().ndim() == 2) {
      return NumpyMatrixNormShape(attrs, in_attrs, out_attrs);
    } else {
      assign_svd_empty(out_attrs);
      return NumpyLpNormShape(attrs, in_attrs, out_attrs);
    }
  }
}

TShape swapMatDims(const TShape &shape, const TShape &axis) {
  TShape ret(shape.ndim(), 1);
  int i, j = 0;
  for (i = 0; i < shape.ndim(); ++i) {
    if (i != axis[0] && i != axis[1]) {
      ret[j++] = shape[i];
    }
  }
  ret[j++] = shape[axis[0]];
  ret[j] = shape[axis[1]];
  return ret;
}

TShape inverseTranspose(const TShape &axes) {
  TShape ret(axes.ndim(), 1);
  for (int i = 0; i < axes.ndim(); ++i) {
    ret[axes[i]] = i;
  }
  return ret;
}

}  // namespace op
}  // namespace mxnet
