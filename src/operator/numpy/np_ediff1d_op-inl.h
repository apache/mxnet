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
 * \file np_ediff1d-inl.h
 * \brief Function definition of numpy-compatible ediff1d operator
 */

#ifndef MXNET_OPERATOR_NUMPY_NP_EDIFF1D_OP_INL_H_
#define MXNET_OPERATOR_NUMPY_NP_EDIFF1D_OP_INL_H_

#include <mxnet/base.h>
#include <mxnet/operator_util.h>
#include <vector>
#include "../mxnet_op.h"
#include "../operator_common.h"
#include "../elemwise_op_common.h"

namespace mxnet {
namespace op {

struct EDiff1DParam : public dmlc::Parameter<EDiff1DParam> {
  bool to_begin_arr_given, to_end_arr_given;
  dmlc::optional<double> to_begin_scalar;
  dmlc::optional<double> to_end_scalar;
  DMLC_DECLARE_PARAMETER(EDiff1DParam) {
    DMLC_DECLARE_FIELD(to_begin_arr_given)
      .set_default(false)
      .describe("To determine whether the `to_begin` parameter is an array.");
    DMLC_DECLARE_FIELD(to_end_arr_given)
      .set_default(false)
      .describe("To determine whether the `to_end` parameter is an array.");
    DMLC_DECLARE_FIELD(to_begin_scalar)
      .set_default(dmlc::optional<double>())
      .describe("If the `to_begin`is a scalar, the value of this parameter.");
    DMLC_DECLARE_FIELD(to_end_scalar)
      .set_default(dmlc::optional<double>())
      .describe("If the `to_end`is a scalar, the value of this parameter.");
  }
};

template<typename DType>
struct set_to_val {
  MSHADOW_XINLINE static void Map(index_t i, DType *out, double val) {
    out[i] = DType(val);
  }
};

template <typename DType>
void copyArr(DType* dest, DType* src, size_t count,
             mshadow::Stream<cpu> *s) {
  memcpy(dest, src, count);
}

template <typename DType>
void AssignScalar(DType* dest, index_t idx, double val,
                  mshadow::Stream<cpu> *s) {
  dest[idx] = DType(val);
}

#ifdef __CUDACC__
template <typename DType>
void copyArr(DType* dest, DType* src, size_t count,
             mshadow::Stream<gpu> *s) {
  CUDA_CALL(cudaMemcpyAsync(dest, src, count, cudaMemcpyDeviceToHost,
                            mshadow::Stream<gpu>::GetStream(s)));
}

template <typename DType>
void AssignScalar(DType* dest, index_t idx, double val,
                  mshadow::Stream<gpu> *s) {
  mxnet_op::Kernel<set_to_val<DType>, gpu>::Launch(s, 1, dest + idx, val);
}
#endif

template<int req>
struct ediff1d_forward {
  template <typename DType>
  MSHADOW_XINLINE static void Map(int i,
                                  DType* out_data,
                                  const DType* in_data,
                                  const index_t padding) {
    KERNEL_ASSIGN(out_data[i + padding], req, in_data[i + 1] - in_data[i]);
  }
};

template<typename xpu>
void EDiff1DForward(const nnvm::NodeAttrs& attrs,
                    const OpContext& ctx,
                    const std::vector<TBlob>& inputs,
                    const std::vector<OpReqType>& req,
                    const std::vector<TBlob>& outputs) {
  using namespace mshadow;
  using namespace mxnet_op;
  CHECK_GE(inputs.size(), 1U);
  CHECK_LE(inputs.size(), 3U);
  CHECK_EQ(req.size(), 1U);
  CHECK_EQ(outputs.size(), 1U);
  const TBlob& in_data = inputs[0];
  const TBlob& out_data = outputs[0];
  mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();
  MSHADOW_TYPE_SWITCH(out_data.type_flag_, DType, {
    const EDiff1DParam& param = nnvm::get<EDiff1DParam>(attrs.parsed);
    size_t padding = 0;
    size_t in_size = (in_data.Size() > 0)? in_data.Size() - 1: 0;
    index_t idx = 1;  // used to index the rest of input arrays

    if (param.to_begin_arr_given) {
      // if the `to_begin` parameter is an array, copy its values to the beginning of the out array
      copyArr<DType>(out_data.dptr<DType>(), inputs[idx].dptr<DType>(),
                      inputs[idx].Size() * sizeof(DType), s);
      padding += inputs[idx].Size();
      idx += 1;
    } else if (param.to_begin_scalar.has_value()) {
      // if the `to_begin` parameter is a scalar, directly assign its value
      AssignScalar(out_data.dptr<DType>(), 0, param.to_begin_scalar.value(), s);
      padding += 1;
    }

    if (param.to_end_arr_given) {
      // if the `to_end` parameter is an array, copy its values to the end of the out array
      copyArr<DType>(out_data.dptr<DType>() + padding + in_size,
                     inputs[idx].dptr<DType>(), inputs[idx].Size() * sizeof(DType), s);
    } else if (param.to_end_scalar.has_value()) {
      // if the `to_end` parameter is a scalar, directly assign its value
      AssignScalar(out_data.dptr<DType>(), padding + in_size, param.to_end_scalar.value(), s);
    }

    MXNET_ASSIGN_REQ_SWITCH(req[0], req_type, {
      Kernel<ediff1d_forward<req_type>, xpu>::Launch(
        s, in_size, out_data.dptr<DType>(), in_data.dptr<DType>(), padding);
    });
  });
}

template<int req>
struct ediff1d_backward_arr {
  template <typename DType>
  MSHADOW_XINLINE static void Map(size_t i,
                                  DType* igrad_dptr,
                                  const DType* input_dptr,
                                  const DType* ograd_dptr,
                                  const size_t padding,
                                  const size_t input_size) {
    if (i == 0) {
      KERNEL_ASSIGN(igrad_dptr[i], req, -ograd_dptr[i + padding]);
    } else if (i == input_size - 1) {
      KERNEL_ASSIGN(igrad_dptr[i], req, ograd_dptr[i - 1 + padding]);
    } else {
      KERNEL_ASSIGN(igrad_dptr[i], req, ograd_dptr[i - 1 + padding] - ograd_dptr[i + padding]);
    }
  }
};

template<typename xpu>
void EDiff1DBackward(const nnvm::NodeAttrs& attrs, const OpContext& ctx,
                     const std::vector<TBlob>& inputs,
                     const std::vector<OpReqType>& req,
                     const std::vector<TBlob>& outputs) {
  using namespace mshadow;
  using namespace mxnet_op;
  CHECK_GE(inputs.size(), 2U);
  CHECK_LE(inputs.size(), 4U);
  CHECK_GE(outputs.size(), 1U);
  CHECK_LE(outputs.size(), 3U);
  CHECK_EQ(req.size(), outputs.size());

  mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();
  const EDiff1DParam& param = nnvm::get<EDiff1DParam>(attrs.parsed);

  const TBlob& ograd = inputs[0];
  const TBlob& input = inputs[1];
  const TBlob& igrad = outputs[0];
  size_t in_size = (input.Size() > 0)? input.Size() - 1: 0;

  MSHADOW_REAL_TYPE_SWITCH(ograd.type_flag_, DType, {
    MXNET_ASSIGN_REQ_SWITCH(req[0], req_type, {
      size_t padding = 0;
      index_t idx = 1;  // start from the second argument of `outputs`
      if (param.to_begin_arr_given) {
        copyArr<DType>(outputs[idx].dptr<DType>(),
                       ograd.dptr<DType>(),
                       outputs[idx].Size() * sizeof(DType), s);
        padding += outputs[idx].Size();
        idx += 1;
      } else if (param.to_begin_scalar.has_value()) {
        padding += 1;
      }

      if (param.to_end_arr_given) {
        copyArr<DType>(outputs[idx].dptr<DType>(),
                       ograd.dptr<DType>()+ in_size + padding,
                       outputs[idx].Size() * sizeof(DType), s);
      }

      if (input.Size() == 0) return;
      if (input.Size() == 1) {
        Kernel<set_to_val<DType>, xpu>::Launch(s, 1, igrad.dptr<DType>(), 0);
      } else {
        Kernel<ediff1d_backward_arr<req_type>, xpu>::Launch(
          s, igrad.Size(), igrad.dptr<DType>(),
          input.dptr<DType>(), ograd.dptr<DType>(),
          padding, igrad.Size());
      }
    });
  });
}

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_NUMPY_NP_EDIFF1D_OP_INL_H_
