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
 * Copyright (c) 2020 by Contributors
 * \file np_unpackbits_op-inl.h
 * \brief Function definition of numpy operator unpackbits
 */

#ifndef MXNET_OPERATOR_NUMPY_NP_UNPACKBITS_OP_INL_H_
#define MXNET_OPERATOR_NUMPY_NP_UNPACKBITS_OP_INL_H_

#include <mxnet/base.h>
#include <string>
#include <vector>
#include "../mxnet_op.h"
#include "../mshadow_op.h"
#include "../../common/utils.h"
#include "../operator_common.h"

namespace mxnet {
namespace op {

using namespace mshadow;

struct NumpyUnpackbitsParam : public dmlc::Parameter<NumpyUnpackbitsParam> {
  dmlc::optional<int> axis;
  std::string bitorder;
  DMLC_DECLARE_PARAMETER(NumpyUnpackbitsParam) {
    DMLC_DECLARE_FIELD(axis)
    .set_default(dmlc::optional<int>())
    .describe("The dimension over which bit-unpacking is done. "
              "None implies unpacking the flattened array.");
    DMLC_DECLARE_FIELD(bitorder)
    .set_default("big")
    .describe("The order of the returned bits: {'big', 'little'}");
  }

  void SetAttrDict(std::unordered_map<std::string, std::string>* dict) {
    std::ostringstream axis_s, bitorder_s;
    axis_s << axis;
    bitorder_s << bitorder;
    (*dict)["axis"] = axis_s.str();
    (*dict)["bitorder"] = bitorder_s.str();
  }
};

inline bool NumpyUnpackbitsDType(const nnvm::NodeAttrs& attrs,
                                 std::vector<int>* in_attrs,
                                 std::vector<int>* out_attrs) {
  CHECK_EQ(in_attrs->size(), 1U);
  CHECK_EQ(out_attrs->size(), 1U);
  TYPE_ASSIGN_CHECK(*in_attrs, 0, mshadow::kUint8);
  TYPE_ASSIGN_CHECK(*out_attrs, 0, in_attrs->at(0));
  return out_attrs->at(0) != -1 && in_attrs->at(0) != -1;
}

inline TShape NumpyUnpackbitsShapeImpl(std::vector<TShape>* in_attrs,
                                       dmlc::optional<int> axis) {
  TShape oshape;
  int ndim = in_attrs->at(0).ndim();

  if (!axis.has_value()) {
    /* If param.axis is None, then unpack the flattened array */
    int size = 8 * in_attrs->at(0).Size();
    oshape = TShape(1, size);
  } else if (ndim == 0 && -1 <= axis.value() && axis.value() <= 0) {
    /* Handle 0-d array by converting it to a 1-d array */
    int size = 8 * in_attrs->at(0).Size();
    oshape = TShape(1, size);
  } else if (-ndim <= axis.value() && axis.value() < ndim) {
    oshape = in_attrs->at(0);
    int abs_axis = (axis.value() + ndim) % ndim;
    /* Multiply axis dimension by 8 */
    oshape[abs_axis] *= 8;
  } else {
    LOG(FATAL) << "ValueError: axis " << axis.value()
               << " is out of bounds for array of dimension " << ndim;
  }

  return oshape;
}

inline bool NumpyUnpackbitsShape(const nnvm::NodeAttrs& attrs,
                                 std::vector<TShape>* in_attrs,
                                 std::vector<TShape>* out_attrs) {
  CHECK_EQ(in_attrs->size(), 1U);
  CHECK_EQ(out_attrs->size(), 1U);
  if (!shape_is_known(in_attrs->at(0))) {
    return false;
  }
  const NumpyUnpackbitsParam& param = nnvm::get<NumpyUnpackbitsParam>(attrs.parsed);
  SHAPE_ASSIGN_CHECK(*out_attrs, 0,
                     NumpyUnpackbitsShapeImpl(in_attrs, param.axis));
  return shape_is_known(out_attrs->at(0));
}

template<int req>
struct unpackbits_forward_order_big {
  template <typename DType>
  MSHADOW_XINLINE static void Map(int i,
                                  DType* out_data,
                                  const DType* in_data,
                                  const int M,
                                  const int N,
                                  const size_t bit_stride,
                                  const size_t num_stride) {
    for (int j = 0; j < M; j++) {
      int from = j * N + i;
      for (int k = 7; k >= 0; k--) {
        int to = j * num_stride + i + (7 - k) * bit_stride;
        KERNEL_ASSIGN(out_data[to], req, (in_data[from] & (1 << k)) == (1 << k))
      }
    }
  }
};

template<int req>
struct unpackbits_forward_order_little {
  template <typename DType>
  MSHADOW_XINLINE static void Map(int i,
                                  DType* out_data,
                                  const DType* in_data,
                                  const int M,
                                  const int N,
                                  const size_t bit_stride,
                                  const size_t num_stride) {
    for (int j = 0; j < M; j++) {
      int from = j * N + i;
      for (int k = 0; k < 8; k++) {
        int to = j * num_stride + i + k * bit_stride;
        KERNEL_ASSIGN(out_data[to], req, (in_data[from] & (1 << k)) == (1 << k))
      }
    }
  }
};

template<typename xpu>
inline void NumpyUnpackbitsForward(const nnvm::NodeAttrs& attrs,
                                   const OpContext& ctx,
                                   const std::vector<TBlob>& inputs,
                                   const std::vector<OpReqType>& req,
                                   const std::vector<TBlob>& outputs) {
  CHECK_EQ(inputs.size(), 1U);
  CHECK_EQ(outputs.size(), 1U);
  const NumpyUnpackbitsParam& param = nnvm::get<NumpyUnpackbitsParam>(attrs.parsed);

  if (param.bitorder != "big" && param.bitorder != "little") {
    LOG(FATAL)<< "ValueError: 'bitorder' must be from {'big', 'little'}";
  }

  const TBlob& in_data = inputs[0];
  const TBlob& out_data = outputs[0];
  const mxnet::TShape& ishape = in_data.shape_;
  const mxnet::TShape& oshape = out_data.shape_;
  Stream<xpu> *s = ctx.get_stream<xpu>();

  using namespace mxnet_op;
  /* Handle 0-d array by converting it to a 1-d array */
  dim_t ndim = (ishape.ndim() > 0) ? ishape.ndim() : 1;
  /* Handle parameter axis */
  index_t axis = param.axis.has_value()? param.axis.value(): ndim - 1;
  if (axis < 0) axis += ndim;
  /* Calculate the number of blocks */
  int M = (ishape.ndim() == 0) ? 1 : ishape.ProdShape(0, axis + 1);
  /* Calculate the number of elements in each axis */
  int N = (M == 0) ? 0 : ishape.Size() / M;
  /* Calculate the stride of adjacent bit of each number depending on `axis` */
  size_t bit_stride = oshape.ProdShape(axis + 1, oshape.ndim());
  size_t num_stride = 8 * bit_stride;

  MXNET_ASSIGN_REQ_SWITCH(req[0], req_type, {
    if (param.bitorder == "little") {
      Kernel<unpackbits_forward_order_little<req_type>, xpu>::Launch(
        s, N, out_data.dptr<uint8_t>(), in_data.dptr<uint8_t>(), M, N, bit_stride, num_stride);
    } else {
      Kernel<unpackbits_forward_order_big<req_type>, xpu>::Launch(
        s, N, out_data.dptr<uint8_t>(), in_data.dptr<uint8_t>(), M, N, bit_stride, num_stride);
    }
  });
}

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_NUMPY_NP_UNPACKBITS_OP_INL_H_
