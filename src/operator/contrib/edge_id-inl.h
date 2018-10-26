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
 * \file edge_id-inl.h
 * \brief Operator implementing edge_id function.
 */
#ifndef MXNET_OPERATOR_CONTRIB_EDGE_ID_INL_H_
#define MXNET_OPERATOR_CONTRIB_EDGE_ID_INL_H_

#include <mxnet/operator_util.h>
#include <vector>
#include "../mshadow_op.h"
#include "../mxnet_op.h"
#include "../operator_common.h"
#include "../tensor/init_op.h"

namespace mxnet {
namespace op {

inline bool EdgeIDShape(const nnvm::NodeAttrs& attrs,
                        std::vector<TShape>* in_attrs,
                        std::vector<TShape>* out_attrs) {
  CHECK_EQ(in_attrs->size(), 3U);
  CHECK_EQ(out_attrs->size(), 1U);
  CHECK_EQ(in_attrs->at(1).ndim(), 1U);
  CHECK_EQ(in_attrs->at(2).ndim(), 1U);
  CHECK_EQ(in_attrs->at(1)[0], in_attrs->at(2)[0]);

  SHAPE_ASSIGN_CHECK(*out_attrs, 0, in_attrs->at(1));
  SHAPE_ASSIGN_CHECK(*in_attrs, 1, out_attrs->at(0));
  SHAPE_ASSIGN_CHECK(*in_attrs, 2, out_attrs->at(0));
  return out_attrs->at(0).ndim() != 0U && out_attrs->at(0).Size() != 0U;
}

inline bool EdgeIDType(const nnvm::NodeAttrs& attrs,
                       std::vector<int>* in_attrs,
                       std::vector<int>* out_attrs) {
  CHECK_EQ(in_attrs->size(), 3U);
  CHECK_EQ(out_attrs->size(), 1U);

  TYPE_ASSIGN_CHECK(*out_attrs, 0, in_attrs->at(0));
  TYPE_ASSIGN_CHECK(*in_attrs, 0, out_attrs->at(0));
  return out_attrs->at(0) != -1;
}

inline bool EdgeIDStorageType(const nnvm::NodeAttrs& attrs,
                              const int dev_mask,
                              DispatchMode* dispatch_mode,
                              std::vector<int>* in_attrs,
                              std::vector<int>* out_attrs) {
  CHECK_EQ(in_attrs->size(), 3U) << "Only works for 2d arrays";
  CHECK_EQ(out_attrs->size(), 1U);
  int& in_stype = in_attrs->at(0);
  int& out_stype = out_attrs->at(0);
  bool dispatched = false;
  if (!dispatched && in_stype == kCSRStorage) {
    // csr -> dns
    dispatched = storage_type_assign(&out_stype, kDefaultStorage,
                                     dispatch_mode, DispatchMode::kFComputeEx);
  }
  if (!dispatched) {
    LOG(ERROR) << "Cannot dispatch edge_id storage type, only works for csr matrices";
  }
  return dispatched;
}

struct edge_id_csr_forward {
  template<typename DType, typename IType, typename CType>
  MSHADOW_XINLINE static void Map(int i, DType* out_data, const DType* in_data,
                                  const IType* in_indices, const IType* in_indptr,
                                  const CType* u, const CType* v) {
    const int64_t target_row_id = static_cast<int64_t>(u[i]);
    const IType target_col_id = static_cast<IType>(v[i]);
    auto ptr = std::find(in_indices + in_indptr[target_row_id], in_indices + in_indptr[target_row_id + 1], target_col_id);
    if (ptr == in_indices + in_indptr[target_row_id + 1]) {
      // does not exist in the range
      out_data[i] = DType(-1);
    } else {
      out_data[i] = *(in_data + (ptr - in_indices));
    }
  }
};

template<typename xpu>
void EdgeIDForwardCsrImpl(const OpContext& ctx,
                          const std::vector<NDArray>& inputs,
                          const OpReqType req,
                          const NDArray& output) {
  using namespace mshadow;
  using namespace mxnet_op;
  using namespace csr;
  if (req == kNullOp) return;
  CHECK_EQ(inputs.size(), 3U);
  CHECK_EQ(req, kWriteTo) << "EdgeID with CSR only supports kWriteTo";
  Stream<xpu> *s = ctx.get_stream<xpu>();
  const NDArray& u = inputs[1];
  const nnvm::dim_t out_elems = u.shape().Size();
  if (!inputs[0].storage_initialized()) {
    MSHADOW_TYPE_SWITCH(output.dtype(), DType, {
      Kernel<mxnet_op::op_with_req<mshadow_op::identity, kWriteTo>, xpu>::Launch(
        s, out_elems, output.data().dptr<DType>(), DType(-1));
    });
    return;
  }
  const NDArray& data = inputs[0];
  const TBlob& in_data = data.data();
  const TBlob& in_indices = data.aux_data(kIdx);
  const TBlob& in_indptr = data.aux_data(kIndPtr);
  const NDArray& v = inputs[2];

  CHECK_EQ(data.aux_type(kIdx), data.aux_type(kIndPtr))
    << "The dtypes of indices and indptr don't match";
  MSHADOW_TYPE_SWITCH(data.dtype(), DType, {
    MSHADOW_IDX_TYPE_SWITCH(data.aux_type(kIdx), IType, {
      MSHADOW_TYPE_SWITCH(u.dtype(), CType, {
        Kernel<edge_id_csr_forward, xpu>::Launch(
            s, out_elems, output.data().dptr<DType>(), in_data.dptr<DType>(),
            in_indices.dptr<IType>(), in_indptr.dptr<IType>(),
            u.data().dptr<CType>(), v.data().dptr<CType>());
      });
    });
  });
}

template<typename xpu>
void EdgeIDForwardEx(const nnvm::NodeAttrs& attrs,
                     const OpContext& ctx,
                     const std::vector<NDArray>& inputs,
                     const std::vector<OpReqType>& req,
                     const std::vector<NDArray>& outputs) {
  CHECK_EQ(inputs.size(), 3U);
  CHECK_EQ(outputs.size(), 1U);
  CHECK_EQ(req.size(), 1U);
  const auto in_stype = inputs[0].storage_type();
  const auto out_stype = outputs[0].storage_type();
  if (in_stype == kCSRStorage && out_stype == kDefaultStorage) {
    EdgeIDForwardCsrImpl<xpu>(ctx, inputs, req[0], outputs[0]);
  } else {
    LogUnimplementedOp(attrs, ctx, inputs, req, outputs);
  }
}

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_CONTRIB_EDGE_ID_INL_H_
