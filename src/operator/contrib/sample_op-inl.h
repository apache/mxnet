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
 *  Copyright (c) 2017 by Contributors
 * \file quantize-inl.h
 * \brief implementation of quantize operation
 */
#ifndef MXNET_OPERATOR_CONTRIB_SAMPLE_OP_INL_H_
#define MXNET_OPERATOR_CONTRIB_SAMPLE_OP_INL_H_

#include <mxnet/operator_util.h>
#include <vector>
#include <limits>
#include "../elemwise_op_common.h"
#include "../operator_common.h"
#include "../mshadow_op.h"
#include "../mxnet_op.h"

namespace mxnet {
namespace op {

struct accidental_hit {
  template<typename IType, typename DType>
  MSHADOW_XINLINE static void Map(int i, DType *out_data, IType *out_idx,
                                  const DType* label, const IType* out_indptr,
                                  const std::unordered_map<DType, std::list<IType>> *map) {
    const auto it = map->find(label[i]);
    const DType one = static_cast<DType>(1);
    IType j = out_indptr[i];
    if (it != map->end()) {
      for (const IType idx : it->second) {
        out_data[j] = one;
        out_idx[j++] = idx;
      }
    }
  }
};

// Only works for cpu
template<typename xpu>
void AccidentalHitComputeCsrImpl(mshadow::Stream<xpu> *s,
                                 const TBlob& label,
                                 const TBlob& sample,
                                 const OpReqType req,
                                 const NDArray& output) {
  if (req == kNullOp) return;
  using nnvm::dim_t;
  using namespace csr;
  using namespace mxnet_op;
  dim_t num_sample = sample.shape_.Size();
  dim_t num_label = label.shape_.Size();
  // TODO more types
  MSHADOW_SGL_DBL_TYPE_SWITCH(label.type_flag_, DType, {
    MSHADOW_IDX_TYPE_SWITCH(output.aux_type(kIdx), IType, {
      std::unordered_map<DType, std::list<IType>> sample_map;
      DType *label_data = label.dptr<DType>();
      DType *sample_data = sample.dptr<DType>();
      for (IType i = 0; i < num_sample; i++) {
        sample_map[sample_data[i]].push_back(i);
      }
      output.CheckAndAllocAuxData(kIndPtr, mshadow::Shape1(num_label + 1));
      IType *out_indptr = output.aux_data(kIndPtr).dptr<IType>();
      out_indptr[0] = 0;
      for (dim_t i = 1; i < num_label + 1; i++) {
        IType count = 0;
        const auto it = sample_map.find(label_data[i - 1]);
        // found accidental match
        if (it != sample_map.end()) {
          count = it->second.size();
        }
        out_indptr[i] = out_indptr[i - 1] + count;
      }
      IType nnz = out_indptr[num_label];
      output.CheckAndAllocData(mshadow::Shape1(nnz));
      output.CheckAndAllocAuxData(kIdx, mshadow::Shape1(nnz));
      DType *out_data = output.data().dptr<DType>();
      IType *out_idx = output.aux_data(kIdx).dptr<IType>();
      Kernel<accidental_hit, xpu>::Launch(s, num_label, out_data,
             out_idx, label_data, out_indptr, &sample_map);
    });
  });
}

template<typename xpu>
void AccidentalHitComputeEx(const nnvm::NodeAttrs& attrs,
                            const OpContext& ctx,
                            const std::vector<NDArray>& inputs,
                            const std::vector<OpReqType>& req,
                            const std::vector<NDArray>& outputs) {
  CHECK_EQ(inputs.size(), 2U);
  CHECK_EQ(outputs.size(), 1U);
  CHECK_EQ(req.size(), 1U);
  mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();
  if (common::ContainsOnlyStorage(inputs, kDefaultStorage) &&
      outputs[0].storage_type() == kCSRStorage) {
    AccidentalHitComputeCsrImpl(s, inputs[0].data(), inputs[1].data(), req[0],
                                outputs[0]);
  } else {
    LOG(FATAL) << "Not implemented: " << operator_string(attrs, ctx, inputs, req, outputs);
  }
}

inline bool AccidentalHitShape(const nnvm::NodeAttrs& attrs,
                               std::vector<TShape> *in_attrs,
                               std::vector<TShape> *out_attrs) {
  CHECK_EQ(in_attrs->size(), 2U);
  CHECK_EQ(out_attrs->size(), 1U);
  for (size_t i = 0; i < 2; ++i) {
    CHECK_EQ(in_attrs->at(i).ndim(), 1);
  }
  TShape out_attr{in_attrs->at(0)[0], in_attrs->at(1)[0]};
  SHAPE_ASSIGN_CHECK(*out_attrs, 0, out_attr);
  return true;
}

inline bool AccidentalHitStorageType(const nnvm::NodeAttrs& attrs,
                                     const int dev_mask,
                                     DispatchMode* dispatch_mode,
                                     std::vector<int>* in_attrs,
                                     std::vector<int>* out_attrs) {
  CHECK_EQ(in_attrs->size(), 2U);
  CHECK_EQ(out_attrs->size(), 1U);
  auto& out_stype = out_attrs->at(0);
  bool dispatched = false;
  if (!dispatched && common::ContainsOnlyStorage(*in_attrs, kDefaultStorage)) {
    // dns, dns -> csr
    dispatched = storage_type_assign(&out_stype, kCSRStorage, dispatch_mode,
                                     DispatchMode::kFComputeEx);
  }
  if (!dispatched) {
    LOG(FATAL) << "Not implemented: "
               << operator_stype_string(attrs, dev_mask, *in_attrs, *out_attrs);
  }
  return true;
}

}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_CONTRIB_SAMPLE_OP_INL_H_
