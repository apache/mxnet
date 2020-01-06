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
 * \file np_diff-inl.h
 * \brief Function definition of numpy-compatible diff operator
 */

#ifndef MXNET_OPERATOR_NUMPY_NP_DIFF_INL_H_
#define MXNET_OPERATOR_NUMPY_NP_DIFF_INL_H_

#include <mxnet/base.h>
#include <mxnet/operator_util.h>
#include <vector>
#include "../mxnet_op.h"
#include "../operator_common.h"
#include "../tensor/broadcast_reduce_op.h"

namespace mxnet {
namespace op {

struct DiffParam : public dmlc::Parameter<DiffParam> {
  int n, axis;
  dmlc::optional<mxnet::TShape> prepend;
  dmlc::optional<mxnet::TShape> append;
  DMLC_DECLARE_PARAMETER(DiffParam) {
    DMLC_DECLARE_FIELD(n).set_default(1).describe(
        "The number of times values are differenced."
        " If zero, the input is returned as-is.");
    DMLC_DECLARE_FIELD(axis).set_default(-1).describe(
        "Axis along which the cumulative sum is computed."
        " The default (None) is to compute the diff over the flattened array.");
  }
};

inline void YanghuiTri(std::vector<int>* buffer, int n) {
  // apply basic yanghui's triangular to calculate the factors
  (*buffer)[0] = 1;
  for (int i = 1; i <= n; ++i) {
    (*buffer)[i] = 1;
    for (int j = i - 1; j > 0; --j) {
      (*buffer)[j] += (*buffer)[j - 1];
    }
  }
}

struct diff_forward {
  template <typename IType, typename OType, int ndim>
  MSHADOW_XINLINE static void Map(int i, int* diffFactor, OType* out,
                                  const IType* in, const int n,
                                  const int stride,
                                  const mshadow::Shape<ndim> oshape,
                                  const mshadow::Shape<ndim> ishape) {
    using namespace broadcast;

    // j represent the memory index of the corresponding input entry
    int j = ravel(unravel(i, oshape), ishape);
    int indicator = 1;
    out[i] = 0;
    for (int k = n; k >= 0; --k) {
      out[i] += in[j + stride * k] * indicator * diffFactor[k];
      indicator *= -1;
    }
  }
};

template <typename xpu>
void DiffForwardImpl(const OpContext& ctx, const TBlob& in, const TBlob& out,
                     const int n, const int axis) {
  using namespace mshadow;
  using namespace mxnet_op;

  // undefined behavior for n < 0
  CHECK_GE(n, 0);
  int axis_checked = CheckAxis(axis, in.ndim());
  // nothing in the output
  if (n >= in.shape_[axis_checked]) return;
  // stride for elements on the given axis, same in input and output
  int stride = 1;
  for (int i = in.ndim() - 1; i > axis_checked; --i) {
    stride *= in.shape_[i];
  }

  Stream<xpu>* s = ctx.get_stream<xpu>();
  std::vector<int> buffer(n+1, 0);
  YanghuiTri(&buffer, n);
  Tensor<xpu, 1, int> diffFactor =
      ctx.requested[0].get_space_typed<xpu, 1, int>(Shape1(n + 1), s);
  Copy(diffFactor, Tensor<cpu, 1, int>(&buffer[0], Shape1(n + 1), 0), s);

  MSHADOW_TYPE_SWITCH(in.type_flag_, IType, {
    MSHADOW_TYPE_SWITCH(out.type_flag_, OType, {
      MXNET_NDIM_SWITCH(in.ndim(), ndim, {
        Kernel<diff_forward, xpu>::Launch(
          s, out.Size(), diffFactor.dptr_,
          out.dptr<OType>(), in.dptr<IType>(),
          n, stride, out.shape_.get<ndim>(),
          in.shape_.get<ndim>());
      });
    });
  });
}

template <typename xpu>
void DiffForward(const nnvm::NodeAttrs& attrs, const OpContext& ctx,
                 const std::vector<TBlob>& inputs,
                 const std::vector<OpReqType>& req,
                 const std::vector<TBlob>& outputs) {
  using namespace mshadow;
  using namespace mxnet_op;
  CHECK_EQ(inputs.size(), 1U);
  CHECK_EQ(req.size(), 1U);
  CHECK_EQ(outputs.size(), 1U);
  const DiffParam& param = nnvm::get<DiffParam>(attrs.parsed);

  DiffForwardImpl<xpu>(ctx, inputs[0], outputs[0], param.n, param.axis);
}

struct diff_backward {
  template <typename IType, typename OType, int ndim>
  MSHADOW_XINLINE static void Map(int i, int* diffFactor, OType* igrad,
                                  const IType* ograd, const int n,
                                  const int stride, const int axis,
                                  const mshadow::Shape<ndim> oshape,
                                  const mshadow::Shape<ndim> ishape) {
    using namespace broadcast;
    if (n == 0) {
      igrad[i] = ograd[i];
      return;
    }

    Shape<ndim> coor = unravel(i, oshape);
    // one head thread for a whole sequence along the axis
    if (coor[axis] != 0) return;
    int j = ravel(coor, ishape);
    // initialize the elements of output array
    for (int k = 0; k < oshape[axis]; ++k) igrad[i + k * stride] = 0;
    for (int k = 0; k < ishape[axis]; ++k) {
      int indicator = 1;
      for (int m = n; m >= 0; --m) {
        igrad[i + (m + k) * stride] +=
            ograd[j + k * stride] * indicator * diffFactor[m];
        indicator *= -1;
      }
    }
  }
};

template <typename xpu>
void DiffBackwardImpl(const OpContext& ctx, const TBlob& ograd,
                      const TBlob& igrad, const int n, const int axis) {
  using namespace mshadow;
  using namespace mxnet_op;

  // undefined behavior for n < 0
  CHECK_GE(n, 0);
  int axis_checked = CheckAxis(axis, igrad.ndim());
  // nothing in the ograd and igrad
  if (n >= igrad.shape_[axis_checked]) return;
  // stride for elements on the given axis, same in input and output
  int stride = 1;
  for (int i = igrad.ndim() - 1; i > axis_checked; --i) {
    stride *= igrad.shape_[i];
  }

  Stream<xpu>* s = ctx.get_stream<xpu>();
  std::vector<int> buffer(n+1, 0);
  YanghuiTri(&buffer, n);
  Tensor<xpu, 1, int> diffFactor =
      ctx.requested[0].get_space_typed<xpu, 1, int>(Shape1(n + 1), s);
  Copy(diffFactor, Tensor<cpu, 1, int>(&buffer[0], Shape1(n + 1), 0), s);

  MSHADOW_TYPE_SWITCH(ograd.type_flag_, IType, {
    MSHADOW_TYPE_SWITCH(igrad.type_flag_, OType, {
      MXNET_NDIM_SWITCH(igrad.ndim(), ndim, {
        Kernel<diff_backward, xpu>::Launch(
          s, igrad.Size(), diffFactor.dptr_,
          igrad.dptr<OType>(), ograd.dptr<IType>(),
          n, stride, axis_checked,
          igrad.shape_.get<ndim>(), ograd.shape_.get<ndim>());
      });
    });
  });
}

template <typename xpu>
void DiffBackward(const nnvm::NodeAttrs& attrs, const OpContext& ctx,
                  const std::vector<TBlob>& inputs,
                  const std::vector<OpReqType>& req,
                  const std::vector<TBlob>& outputs) {
  using namespace mshadow;
  using namespace mxnet_op;
  CHECK_EQ(inputs.size(), 1U);
  CHECK_EQ(req.size(), 1U);
  CHECK_EQ(outputs.size(), 1U);
  const DiffParam& param = nnvm::get<DiffParam>(attrs.parsed);

  DiffBackwardImpl<xpu>(ctx, inputs[0], outputs[0], param.n, param.axis);
}

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_NUMPY_NP_DIFF_INL_H_
