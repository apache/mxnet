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
 * \file np_tril_op-inl.h
 * \brief Function definition of the tril (lower triangle of an array) op
 */

#ifndef MXNET_OPERATOR_NUMPY_NP_FILL_DIAGONAL_OP_INL_H_
#define MXNET_OPERATOR_NUMPY_NP_FILL_DIAGONAL_OP_INL_H_

#include <dmlc/parameter.h>
#include <vector>
#include <string>
#include <algorithm>
#include "../mxnet_op.h"
#include "../operator_common.h"
#include "../elemwise_op_common.h"

namespace mxnet {
namespace op {

struct NumpyFillDiagonalParam : public dmlc::Parameter<NumpyFillDiagonalParam> {
  Tuple<double> val;
  bool wrap;
  DMLC_DECLARE_PARAMETER(NumpyFillDiagonalParam) {
    DMLC_DECLARE_FIELD(val)
      .describe("Value to be written on the diagonal, "
                "its type must be compatible with that of the array a.");
    DMLC_DECLARE_FIELD(wrap)
    .set_default(false)
    .describe("The diagonal “wrapped” after N columns."
              "You can have this behavior with this option. "
              "This affects only tall matrices.");
  }

  void SetAttrDict(std::unordered_map<std::string, std::string>* dict) {
    std::ostringstream val_s, wrap_s;
    val_s << val;
    wrap_s << wrap;
    (*dict)["val"] = val_s.str();
    (*dict)["wrap"] = wrap_s.str();
  }
};

inline bool NumpyFillDiagonalOpShape(const nnvm::NodeAttrs& attrs,
                                      mxnet::ShapeVector* in_attrs,
                                      mxnet::ShapeVector* out_attrs) {
  CHECK_EQ(in_attrs->size(), 1U);
  CHECK_EQ(out_attrs->size(), 1U);

  const mxnet::TShape& ishape = (*in_attrs)[0];
  mxnet::TShape oshape;

  if (!mxnet::ndim_is_known(ishape)) {
    return false;
  }

  CHECK_GE(ishape.ndim(), 2U);

  oshape = ishape;

  if (shape_is_none(oshape)) {
    LOG(FATAL) << "Diagonal does not exist.";
  }
  SHAPE_ASSIGN_CHECK(*out_attrs, 0, oshape);

  return shape_is_known(out_attrs->at(0));
}

template<int req>
struct FillDiagonalOpForwardImpl {
  template<typename DType>
  MSHADOW_XINLINE static void Map(index_t i, DType* out_data, const DType* in_data,
                                  const double* val, int length, int step, int end){
    using namespace mxnet_op;
    if (i < end) {
      if (i % step == 0) {
        KERNEL_ASSIGN(out_data[i], req, val[(i / step) % length]);
      }
    }
  }
};

template<typename xpu>
void NumpyFillDiagonalForward(const nnvm::NodeAttrs& attrs,
                              const OpContext& ctx,
                              const std::vector<TBlob>& inputs,
                              const std::vector<OpReqType>& req,
                              const std::vector<TBlob>& outputs) {
  using namespace mxnet_op;
  using namespace mshadow;
  CHECK_EQ(inputs.size(), 1U);
  CHECK_EQ(outputs.size(), 1U);
  Stream<xpu> *s = ctx.get_stream<xpu>();

  const TBlob& in_data = inputs[0];
  const TBlob& out_data = outputs[0];
  CHECK_GE(in_data.ndim(), 2) << "Input ndim must greater or equal 2.";
  const TShape inshape = in_data.shape_;

  const NumpyFillDiagonalParam& param = nnvm::get<NumpyFillDiagonalParam>(attrs.parsed);
  Tuple<double> val_data = param.val;
  bool wrap = param.wrap;

  size_t total_temp_size = val_data.ndim() * sizeof(double);
  Tensor<xpu, 1, char> temp_space =
    ctx.requested[0].get_space_typed<xpu, 1, char>(Shape1(total_temp_size), s);
  double* val = reinterpret_cast<double*>(temp_space.dptr_);

  // filling in
  if (ctx.run_ctx.ctx.dev_mask() == gpu::kDevMask) {
  #if MXNET_USE_CUDA
    cudaMemcpyAsync(val, val_data.begin(), val_data.ndim() * sizeof(double),
                    cudaMemcpyHostToDevice, Stream<gpu>::GetStream(ctx.get_stream<gpu>()));
  #else
    LOG(FATAL) << "Illegal attempt to use GPU in a CPU-only build";
  #endif
  } else {
    std::memcpy(val, val_data.begin(), val_data.ndim() * sizeof(double));
  }

  // for kernel
  int64_t step = 0;
  int64_t end = 0;
  // wrap only works when ndim is 2
  if (in_data.ndim() == 2) {
    if (!wrap) {
      end = inshape[1] * inshape[1];
    } else {
      end = inshape[0] * inshape[1];
    }
    step = inshape[1] + 1;
  } else {
    // input_data ndim must all equal
    int64_t cum = 1;
    for (int i = 0; i < in_data.ndim() - 1; i++) {
      CHECK_EQ(inshape[i], inshape[i + 1]) << "All dimensions must equal.";
      cum *= inshape[i];
      step += cum;
    }
    step++;
    end = cum * inshape[in_data.ndim() - 1];
  }

  MSHADOW_TYPE_SWITCH(out_data.type_flag_, DType, {
    MXNET_ASSIGN_REQ_SWITCH(req[0], req_type, {
      Kernel<FillDiagonalOpForwardImpl<req_type>, xpu>::Launch(
          s, out_data.Size(), out_data.dptr<DType>(), in_data.dptr<DType>(),
          val, val_data.ndim(), step, end);
    });
  });
}

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_NUMPY_NP_FILL_DIAGONAL_OP_INL_H_
