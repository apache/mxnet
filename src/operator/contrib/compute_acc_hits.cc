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
 * \file compute_acc_hits.cc
 * \brief
 */
#include "./compute_acc_hits-inl.h"

namespace mxnet {
namespace op {

/* \brief the kernel to compute accidental hit on CPU
 * \param i i-th        thread
 * \param out_data      the output csr's data
 * \param out_idx       the output csr's column indices
 * \param label         the true classes
 * \param out_indptr    the output csr's indptr
 * \param map           the hash map that stores positions of sampled candidates
 */
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

template<>
void AccidentalHitComputeCsrImpl<cpu>(mshadow::Stream<cpu> *s,
                                      const TBlob& label,
                                      const TBlob& sample,
                                      const OpReqType req,
                                      const NDArray& output) {
  if (req == kNullOp) return;
  CHECK_EQ(req, kWriteTo) << "Unexpected req for compute accidental hits operator";
  using nnvm::dim_t;
  using namespace csr;
  using namespace mxnet_op;
  const dim_t num_sample = sample.shape_.Size();
  const dim_t num_label = label.shape_.Size();
  MSHADOW_TYPE_SWITCH(label.type_flag_, DType, {
    MSHADOW_IDX_TYPE_SWITCH(output.aux_type(kIdx), IType, {
      std::unordered_map<DType, std::list<IType>> sample_map;
      const DType *label_data = label.dptr<DType>();
      const DType *sample_data = sample.dptr<DType>();
      for (IType i = 0; i < num_sample; i++) {
        sample_map[sample_data[i]].push_back(i);
      }
      output.CheckAndAllocAuxData(kIndPtr, mshadow::Shape1(num_label + 1));
      IType *out_indptr = output.aux_data(kIndPtr).dptr<IType>();
      out_indptr[0] = 0;
      // compute the number of matches for each row
      for (dim_t i = 1; i < num_label + 1; i++) {
        IType count = 0;
        const auto it = sample_map.find(label_data[i - 1]);
        // accidental match found
        if (it != sample_map.end()) {
          count = it->second.size();
        }
        out_indptr[i] = out_indptr[i - 1] + count;
      }
      // allocate the memory based on nnz
      const IType nnz = out_indptr[num_label];
      output.CheckAndAllocData(mshadow::Shape1(nnz));
      output.CheckAndAllocAuxData(kIdx, mshadow::Shape1(nnz));
      DType *out_data = output.data().dptr<DType>();
      IType *out_idx = output.aux_data(kIdx).dptr<IType>();
      Kernel<accidental_hit, cpu>::Launch(s, num_label, out_data,
             out_idx, label_data, out_indptr, &sample_map);
    });
  });
}

NNVM_REGISTER_OP(_contrib_compute_accidental_hits)
.describe(R"code(Compute the indices in ``sampled_candidates`` which matches ``true_classes``
and return the mask for the matching positions.

The operator is used for removing sampled classes which happen to match target classes
(i.e. accidental hits) for sampled softmax and sampled logistic. The mask has 0
for non-matching positions and 1 for matching ones.

Both inputs are expected to be 1-D. For example, let's say ``true_classes`` has shape (M,),
and ``sampled_candidates`` has (N,), then the resulting mask will have shape (M, N).
mask[i][j] = 1 iff true_classes[i] == sampled_candidates[j].

Example::

   true = [1,5,11]
   sampled = [5,8,1,5,24]
   compute_accidental_hits(true, sampled) = [[0, 0, 1, 0, 0],
                                             [1, 0, 0, 1, 0],
                                             [0, 0, 0, 0, 0]]

.. note:: `compute_accidental_hits` is only available on CPU and returns a compressed sparse row mask.

)code" ADD_FILELINE)
.set_num_inputs(2)
.set_num_outputs(1)
.set_attr<nnvm::FInferShape>("FInferShape", AccidentalHitShape)
.set_attr<nnvm::FInferType>("FInferType", ElemwiseType<2, 1>)
.set_attr<FInferStorageType>("FInferStorageType", AccidentalHitStorageType)
.set_attr<FComputeEx>("FComputeEx<cpu>", AccidentalHitComputeEx<cpu>)
.add_argument("true_classes", "NDArray-or-Symbol", "True Classes of 1-D shape.")
.add_argument("sampled_candidates", "NDArray-or-Symbol", "Sampled Candidates of 1-D shape.");

}  // namespace op
}  // namespace mxnet
