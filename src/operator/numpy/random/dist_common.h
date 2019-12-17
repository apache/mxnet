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
 *  Copyright (c) 2015 by Contributors
 * \file dist_common.h
 * \brief Function definition of common functions for distributions
 * \with two parameters.
 */

#ifndef MXNET_OPERATOR_NUMPY_RANDOM_DIST_COMMON_H_
#define MXNET_OPERATOR_NUMPY_RANDOM_DIST_COMMON_H_

#include <mxnet/operator_util.h>
#include <algorithm>
#include <string>
#include <vector>
#include "../../elemwise_op_common.h"
#include "../../mshadow_op.h"
#include "../../mxnet_op.h"
#include "../../operator_common.h"
#include "../../tensor/elemwise_binary_broadcast_op.h"

namespace mxnet {
namespace op {

template <typename xpu>
void _copy(mshadow::Stream<xpu> *s, float *dst, float*src);

template <typename xpu>
void _copy(mshadow::Stream<xpu> *s, double *dst, double*src);


inline int FillShape(const mxnet::TShape &lshape, const mxnet::TShape &rshape,
                     const mxnet::TShape &oshape, mxnet::TShape *new_lshape,
                     mxnet::TShape *new_rshape, mxnet::TShape *new_oshape) {
  const int odim = std::max(oshape.ndim(), broadcast::MAX_DIM);
  *new_lshape = mxnet::TShape(odim, 1);
  *new_rshape = mxnet::TShape(odim, 1);
  *new_oshape = mxnet::TShape(odim, 1);
  int bl = oshape.ndim() - lshape.ndim();
  int br = oshape.ndim() - rshape.ndim();
  int j = 0, lprod = 1, rprod = 1, oprod = 1;
  for (int i = 0; i < oshape.ndim(); ++i) {
    int l = 1;
    int r = 1;
    int o = oshape[i];
    if (i >= bl) l = lshape[i - bl];
    if (i >= br) r = rshape[i - br];
    if ((lprod != rprod || lprod != oprod || l != r || l != o) &&
        (lprod * l > 1 || rprod * r > 1 || oprod * o > 1)) {
      (*new_lshape)[j] = lprod;
      (*new_rshape)[j] = rprod;
      (*new_oshape)[j] = oprod;
      lprod = rprod = oprod = 1;
      ++j;
    }
    lprod *= l;
    rprod *= r;
    oprod *= o;
  }
  if (lprod > 1 || rprod > 1 || oprod > 1) {
    (*new_lshape)[j] = lprod;
    (*new_rshape)[j] = rprod;
    (*new_oshape)[j] = oprod;
    ++j;
  }
  if (j <= broadcast::MAX_DIM) {
    BROADCAST_NDIM_SWITCH(j, NDim, {
      new_lshape->assign(new_lshape->begin(), new_lshape->begin() + NDim);
      new_rshape->assign(new_rshape->begin(), new_rshape->begin() + NDim);
      new_oshape->assign(new_oshape->begin(), new_oshape->begin() + NDim);
    });
  } else {
    LOG(FATAL) << "Too many broadcast dimensions with operands " << lshape
               << " " << rshape;
  }
  return j;
}

inline void CheckBroadcastable(const mxnet::TShape &from,
                               const mxnet::TShape &to) {
  const int bl = to.ndim() - from.ndim();
  const int br = 0;
  for (int i = 0; i < to.ndim(); ++i) {
    int l = 1, r = 1;
    if (i >= bl) l = from[i - bl];
    if (i >= br) r = to[i - br];
    if (!mxnet::dim_size_is_known(l) || !mxnet::dim_size_is_known(r)) continue;
    if (l != r) {
      // Make it compatible with NumPy.
      // For example, (2, 3) cannot broadcast to (2, 0, 3), but (1, 3) can
      // broadcast to (2, 0, 3).
      CHECK(l == 1 || r == 1)
          << "operands could not be broadcast together with shapes " << from
          << " " << to;
    }
  }
}

inline void InferBroadcastShape(const mxnet::TShape &lhs,
                                const mxnet::TShape &rhs,
                                mxnet::TShape *out_ptr) {
  mxnet::TShape &out = (*out_ptr);
  const int bl = out.ndim() - lhs.ndim();
  const int br = out.ndim() - rhs.ndim();
  for (int i = 0; i < out.ndim(); ++i) {
    int l = 1, r = 1;
    if (i >= bl) l = lhs[i - bl];
    if (i >= br) r = rhs[i - br];
    if (!mxnet::dim_size_is_known(l) || !mxnet::dim_size_is_known(r)) continue;
    if (l != r) {
      // Make it compatible with NumPy.
      // For example, (2, 3) cannot broadcast to (2, 0, 3), but (1, 3) can
      // broadcast to (2, 0, 3).
      CHECK(l == 1 || r == 1)
          << "operands could not be broadcast together with shapes " << lhs
          << " " << rhs;
      out[i] = (l == 1 ? r : l);
    } else {
      out[i] = l;
    }
  }
}

template <typename DistParam>
inline bool TwoparamsDistOpShape(const nnvm::NodeAttrs &attrs,
                                 std::vector<TShape> *in_attrs,
                                 std::vector<TShape> *out_attrs) {
  const DistParam &param = nnvm::get<DistParam>(attrs.parsed);
  if (param.size.has_value()) {
    // Size declared.
    std::vector<dim_t> oshape_vec;
    const mxnet::Tuple<int> &size = param.size.value();
    for (int i = 0; i < size.ndim(); ++i) {
      oshape_vec.emplace_back(size[i]);
    }
    SHAPE_ASSIGN_CHECK(*out_attrs, 0, TShape(oshape_vec));
    for (size_t input_idx = 0; input_idx < in_attrs->size(); input_idx++) {
      CheckBroadcastable((*in_attrs)[input_idx], (*out_attrs)[0]);
    }
  } else {
    // Size undeclared.
    if (in_attrs->size() == 2U) {
      // Both params from ndarray.
      mxnet::TShape &low = (*in_attrs)[0];
      mxnet::TShape &high = (*in_attrs)[1];
      mxnet::TShape out(std::max(low.ndim(), high.ndim()), -1);
      InferBroadcastShape(low, high, &out);
      SHAPE_ASSIGN_CHECK(*out_attrs, 0, out);
    } else if (in_attrs->size() == 1U) {
      // One param from ndarray.
      SHAPE_ASSIGN_CHECK(*out_attrs, 0, in_attrs->at(0))
    } else if (in_attrs->size() == 0) {
      // Two scalar case.
      SHAPE_ASSIGN_CHECK(*out_attrs, 0, TShape(0, -1))
    }
  }
  if (out_attrs->size() == 2U) {
    SHAPE_ASSIGN_CHECK(*out_attrs, 1, out_attrs->at(0));
  }
  return true;
}

template <typename DistParam>
inline bool UnaryDistOpShape(const nnvm::NodeAttrs &attrs,
                             std::vector<TShape> *in_attrs,
                             std::vector<TShape> *out_attrs) {
  const DistParam &param = nnvm::get<DistParam>(attrs.parsed);
  if (param.size.has_value()) {
    // Size declared.
    std::vector<dim_t> oshape_vec;
    const mxnet::Tuple<int> &size = param.size.value();
    for (int i = 0; i < size.ndim(); ++i) {
      oshape_vec.emplace_back(size[i]);
    }
    SHAPE_ASSIGN_CHECK(*out_attrs, 0, TShape(oshape_vec));
    for (size_t input_idx = 0; input_idx < in_attrs->size(); input_idx++) {
      CheckBroadcastable((*in_attrs)[input_idx], (*out_attrs)[0]);
    }
  } else {
    if (in_attrs->size() == 1U) {
      // One param from ndarray.
      SHAPE_ASSIGN_CHECK(*out_attrs, 0, in_attrs->at(0))
    } else {
      SHAPE_ASSIGN_CHECK(*out_attrs, 0, TShape(0, -1))
    }
  }
  return shape_is_known(out_attrs->at(0));
}


// Infer Shape function for sample_n Op.
// i.e. output_shape = (shape,) + broadcast(param1.shape, param2.shape)
template <typename DistParam>
inline bool TwoparamsDistOpConcatShape(const nnvm::NodeAttrs &attrs,
                                       std::vector<TShape> *in_attrs,
                                       std::vector<TShape> *out_attrs) {
  const DistParam &param = nnvm::get<DistParam>(attrs.parsed);
  // broadcast(param1.shape, param2.shape).
  mxnet::TShape param_broadcast_shape;
  if (in_attrs->size() == 2U) {
      // Both params from ndarray.
      mxnet::TShape &param1 = (*in_attrs)[0];
      mxnet::TShape &param2 = (*in_attrs)[1];
      mxnet::TShape out(std::max(param1.ndim(), param2.ndim()), -1);
      InferBroadcastShape(param1, param2, &out);
      param_broadcast_shape = out;
    } else if (in_attrs->size() == 1U) {
      // One param from ndarray.
      param_broadcast_shape = in_attrs->at(0);
    } else if (in_attrs->size() == 0) {
      // Two scalar case.
      param_broadcast_shape = TShape(0, -1);
    }
  if (param.size.has_value()) {
    // Size declared.
    std::vector<dim_t> oshape_vec;
    const mxnet::Tuple<int> &size = param.size.value();
    for (int i = 0; i < size.ndim(); ++i) {
      oshape_vec.emplace_back(size[i]);
    }
    for (int i = 0; i < param_broadcast_shape.ndim(); ++i) {
      oshape_vec.emplace_back(param_broadcast_shape[i]);
    }
    SHAPE_ASSIGN_CHECK(*out_attrs, 0, TShape(oshape_vec));
  } else {
    SHAPE_ASSIGN_CHECK(*out_attrs, 0, param_broadcast_shape);
  }
  if (out_attrs->size() == 2U) {
    SHAPE_ASSIGN_CHECK(*out_attrs, 1, out_attrs->at(0));
  }
  return true;
}

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_NUMPY_RANDOM_DIST_COMMON_H_ */
