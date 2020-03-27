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
 * \file np_copyto_op-inl.h
*/

#ifndef MXNET_OPERATOR_NUMPY_NP_COPYTO_OP_INL_H_
#define MXNET_OPERATOR_NUMPY_NP_COPYTO_OP_INL_H_


#include <mxnet/operator_util.h>
#include <cstdio>
#include <algorithm>
#include <cmath>
#include <vector>
#include <string>
#include "../tensor/ordering_op-inl.h"
#include "../tensor/matrix_op-inl.h"
#include "../../common/utils.h"
#include "../mshadow_op.h"
#include "../operator_common.h"
#include "../elemwise_op_common.h"
#include "np_broadcast_reduce_op.h"
#include "../../api/operator/op_utils.h"
#include "random/dist_common.h"
#include "../mxnet_op.h"

namespace mxnet {
namespace op {

namespace copyto_casting_enum {
enum CopytoCastingType {no, equiv, safe, same_kind, unsafe};
}  // copyto_enum

typedef bool where_Type;

struct NumpyCopytoParam : public dmlc::Parameter<NumpyCopytoParam> {
  int casting;
  dmlc::optional<double> src;
  dmlc::optional<bool> where;
  dmlc::optional<int> dtype;
  DMLC_DECLARE_PARAMETER(NumpyCopytoParam) {
    DMLC_DECLARE_FIELD(src);
    DMLC_DECLARE_FIELD(casting).set_default(copyto_casting_enum::same_kind)
      .add_enum("no", copyto_casting_enum::no)
      .add_enum("equiv", copyto_casting_enum::equiv)
      .add_enum("safe", copyto_casting_enum::safe)
      .add_enum("same_kind", copyto_casting_enum::same_kind)
      .add_enum("unsafe", copyto_casting_enum::unsafe)
      .describe("Controls what kind of data casting may occur when copying.");
    DMLC_DECLARE_FIELD(where)
      .set_default(dmlc::optional<bool>(true));
    DMLC_DECLARE_FIELD(dtype)
      .add_enum("float32", mshadow::kFloat32)
      .add_enum("float64", mshadow::kFloat64)
      .add_enum("float16", mshadow::kFloat16)
      .add_enum("bfloat16", mshadow::kBfloat16)
      .add_enum("int64", mshadow::kInt64)
      .add_enum("int32", mshadow::kInt32)
      .add_enum("uint8", mshadow::kUint8)
      .add_enum("int8", mshadow::kInt8)
      .set_default(dmlc::optional<int>());
  }
};

struct tensor_scalar_kernel {
  template <int ndim, typename I1Type, typename I2Type>
  MSHADOW_XINLINE static void Map(index_t i,
                                  const Shape<ndim> &stride,
                                  const Shape<ndim> &lshape,
                                  const Shape<ndim> &rshape,
                                  I2Type *array,
                                  I1Type *out) {
    Shape<ndim> coord = mxnet_op::unravel(i, rshape);
    auto idx = static_cast<index_t>(mxnet_op::dot(coord, stride));
    out[i] = array[idx];
  }
};

struct scalar_tensor_kernel {
  template <int ndim, typename I1Type, typename I2Type, typename SType>
  MSHADOW_XINLINE static void Map(index_t i,
                                  const Shape<ndim> &stride,
                                  const Shape<ndim> &lshape,
                                  const Shape<ndim> &rshape,
                                  SType src,
                                  I1Type *out,
                                  I2Type *array_w) {
    Shape<ndim> coord = mxnet_op::unravel(i, rshape);
    auto idx = static_cast<index_t>(mxnet_op::dot(coord, stride));
    if (array_w[idx]) {
        out[i] = src;
    }
  }
};

struct scalar_scalar_kernel {
  template <typename I1Type, typename SType>
  MSHADOW_XINLINE static void Map(index_t i,
                                  SType src,
                                  I1Type *out) {
    out[i] = src;
  }
};

struct tensor_tensor_kernel {
  template <int ndim, typename I1Type, typename I2Type, typename I3Type>
  MSHADOW_XINLINE static void Map(index_t i,
                                  const Shape<ndim> &stride_r,
                                  const Shape<ndim> &stride_w,
                                  const Shape<ndim> &lshape,
                                  const Shape<ndim> &rshape,
                                  const Shape<ndim> &wshape,
                                  I3Type *array_w,
                                  I2Type *array,
                                  I1Type *out) {
    Shape<ndim> coord1 = mxnet_op::unravel(i, rshape);
    Shape<ndim> coord2 = mxnet_op::unravel(i, wshape);
    auto idx1 = static_cast<index_t>(mxnet_op::dot(coord1, stride_r));
    auto idx2 = static_cast<index_t>(mxnet_op::dot(coord2, stride_w));
    if (array_w[idx2])
      out[i] = array[idx1];
  }
};

template <typename xpu>
void NumpyCopytoForward(const nnvm::NodeAttrs &attrs,
                             const OpContext &ctx,
                             const std::vector<TBlob> &inputs,
                             const std::vector<OpReqType> &req,
                             const std::vector<TBlob> &outputs) {
  using namespace mshadow;
  using namespace mxnet_op;
  const NumpyCopytoParam &param = nnvm::get<NumpyCopytoParam>(attrs.parsed);
  Stream<xpu> *s = ctx.get_stream<xpu>();

  if (!param.src.has_value())
    CheckBroadcastable(inputs[1].shape_, inputs[0].shape_);
  if (!param.where.has_value() && !param.src.has_value())
    CheckBroadcastable(inputs[2].shape_, inputs[0].shape_);
  else if (!param.where.has_value() && param.src.has_value())
    CheckBroadcastable(inputs[1].shape_, inputs[0].shape_);

  if (param.where.has_value() && param.where == dmlc::optional<bool>(false))
    return;
  if (param.where.has_value() && !param.src.has_value()) {
    // tensor scalar
    mxnet::TShape new_lshape, new_rshape;
    int ndim = FillShape(inputs[1].shape_, inputs[0].shape_, inputs[0].shape_,
                         &new_rshape, &new_lshape, &new_lshape);
    MSHADOW_TYPE_SWITCH_WITH_BOOL(inputs[0].type_flag_, I1Type, {
      MSHADOW_TYPE_SWITCH_WITH_BOOL(inputs[1].type_flag_, I2Type, {
        BROADCAST_NDIM_SWITCH(ndim, NDim, {
          Shape<NDim> lshape = new_lshape.get<NDim>();
          Shape<NDim> rshape = new_rshape.get<NDim>();
          Shape<NDim> stride = calc_stride(new_rshape.get<NDim>());
          Kernel<tensor_scalar_kernel, xpu>::Launch(
            s, inputs[0].Size(), stride, lshape, rshape,
            inputs[1].dptr<I2Type>(), inputs[0].dptr<I1Type>());
        });
      });
    });
  } else if (!param.where.has_value() && param.src.has_value()) {
    // scalar tensor
    mxnet::TShape new_lshape, new_rshape;
    int ndim = FillShape(inputs[1].shape_, inputs[0].shape_, inputs[0].shape_,
                         &new_rshape, &new_lshape, &new_lshape);
    MSHADOW_TYPE_SWITCH_WITH_BOOL(inputs[0].type_flag_, I1Type, {
      BROADCAST_NDIM_SWITCH(ndim, NDim, {
        Shape<NDim> lshape = new_lshape.get<NDim>();
        Shape<NDim> rshape = new_rshape.get<NDim>();
        Shape<NDim> stride = calc_stride(new_rshape.get<NDim>());
        Kernel<scalar_tensor_kernel, xpu>::Launch(
          s, inputs[0].Size(), stride, lshape, rshape, param.src.value(),
          inputs[0].dptr<I1Type>(), inputs[1].dptr<where_Type>());
      });
    });
  } else if (param.where.has_value() && param.src.has_value()) {
    // scalar scalar
    MSHADOW_TYPE_SWITCH_WITH_BOOL(inputs[0].type_flag_, I1Type, {
      Kernel<scalar_scalar_kernel, xpu>::Launch(
        s, inputs[0].Size(), param.src.value(),
        inputs[0].dptr<I1Type>());
    });
  } else if (!param.where.has_value()) {
  // tensor tensor
    mxnet::TShape new_lshape, new_rshape, new_wshape;
    int ndim = FillShape(inputs[1].shape_, inputs[2].shape_, inputs[0].shape_,
                         &new_rshape, &new_wshape, &new_lshape);
    MSHADOW_TYPE_SWITCH_WITH_BOOL(inputs[0].type_flag_, I1Type, {
      MSHADOW_TYPE_SWITCH_WITH_BOOL(inputs[1].type_flag_, I2Type, {
        BROADCAST_NDIM_SWITCH(ndim, NDim, {
          Shape<NDim> lshape = new_lshape.get<NDim>();
          Shape<NDim> rshape = new_rshape.get<NDim>();
          Shape<NDim> wshape = new_wshape.get<NDim>();
          Shape<NDim> stride_r = calc_stride(new_rshape.get<NDim>());
          Shape<NDim> stride_w = calc_stride(new_wshape.get<NDim>());
          Kernel<tensor_tensor_kernel, xpu>::Launch(
            s, inputs[0].Size(), stride_r, stride_w,
            lshape, rshape, wshape,
            inputs[2].dptr<where_Type>(),
            inputs[1].dptr<I2Type>(), inputs[0].dptr<I1Type>());
        });
      });
    });
  }
}

}  // namespace op
}  // namespace mxnet

    #endif  // MXNET_OPERATOR_NUMPY_NP_COPYTO_OP_INL_H_
