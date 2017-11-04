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
 * \file indexing_op-inl.h
 * \brief
 * \author Haibin Lin
*/
#ifndef MXNET_OPERATOR_TENSOR_INDEXING_OP_INL_H_
#define MXNET_OPERATOR_TENSOR_INDEXING_OP_INL_H_
#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <mxnet/operator.h>
#include <mxnet/operator_util.h>
#include <algorithm>
#include <vector>
#include <string>
#include <type_traits>
#include "../operator_common.h"
#include "./util/tensor_util-inl.h"
#include "../mxnet_op.h"
#include "./init_op.h"
#include "./matrix_op-inl.h"

namespace mxnet {
namespace op {

namespace embedding {
enum EmbeddingOpInputs {kData, kWeight};
enum EmbeddingOpOutputs {kOut};
enum EmbeddingOpResource {kTempSpace};
}  // namespace embedding

struct AddTakeGradRspKernel {
  /*!
   * \brief Each thread i is responsible for row slices in [segment_start, segment_end)
            of the result gradient
   * \param tid             global thread id
   * \param grad            the gradient to calculate
   * \param prefix_sum      the inclusive prefix sum of row ids of the gradient
   * \param ograd           output gradient
   * \param row_length      the length of the row slices of the gradient
   * \param data_val        the values of input data
   * \param data_size       number of values of input data
   * \param segment_length  the length of row segment to process for each thread
   * \param nnr             total number of non-zero rows of result gradient
   */
  template<typename DType, typename IType>
  MSHADOW_XINLINE static void Map(int tid,
                                  DType* grad,
                                  const nnvm::dim_t* prefix_sum,
                                  const DType* ograd,
                                  const nnvm::dim_t row_length,
                                  const IType* data_val,
                                  const nnvm::dim_t data_size,
                                  const nnvm::dim_t segment_length,
                                  const nnvm::dim_t nnr) {
    using nnvm::dim_t;
    dim_t segment_start = tid * segment_length;
    dim_t segment_end = std::min(nnr, segment_start + segment_length);
    // scan all data
    for (dim_t data_i = 0; data_i < data_size; data_i++) {
      dim_t data = static_cast<dim_t>(data_val[data_i]);
      dim_t grad_row_id = prefix_sum[data] - 1;
      if (grad_row_id < segment_start || grad_row_id >= segment_end) continue;
      // no projection is performed
      dim_t ograd_i = data_i * row_length;
      dim_t grad_i = grad_row_id * row_length;
      for (dim_t offset = 0; offset < row_length; offset++) {
        grad[grad_i + offset] += ograd[ograd_i + offset];
      }
    }
  }
};

inline void SparseEmbeddingOpBackwardRspImpl(const OpContext& ctx,
                                             const cpu& cpu_dev,
                                             const TBlob& ograd,
                                             const TBlob& data,
                                             const OpReqType req,
                                             const NDArray& output) {
  using namespace mshadow;
  using namespace mxnet_op;
  using namespace mshadow::expr;
  using namespace rowsparse;
  using nnvm::dim_t;
  if (req == kNullOp) return;
  CHECK_EQ(req, kWriteTo) << "SparseEmbedding layer doesn't support "
                          << "weight gradient calculation with req != write";

  // Request temporary storage for marking non-zero rows and prefix sum
  Stream<cpu> *s = ctx.get_stream<cpu>();
  dim_t num_rows = output.shape()[0];
  dim_t row_length = output.shape()[1];
  // TODO(haibin) request less storage to save space in the future
  size_t workspace_size = 2 * (num_rows * sizeof(dim_t));
  Tensor<cpu, 1, char> workspace =
    ctx.requested[embedding::kTempSpace].get_space_typed<cpu, 1, char>(
      Shape1(workspace_size), s);
  dim_t* row_flg = reinterpret_cast<dim_t*>(workspace.dptr_);
  dim_t* prefix_sum = row_flg + num_rows;
  dim_t data_size = static_cast<dim_t>(data.shape_.Size());

  MSHADOW_TYPE_SWITCH(data.type_flag_, IType, {
    MSHADOW_TYPE_SWITCH(ograd.type_flag_, DType, {
      MSHADOW_TYPE_SWITCH(output.aux_type(kIdx), RType, {
        // mark row flags
        Fill<false>(s, TBlob(row_flg, mshadow::Shape1(num_rows), cpu::kDevMask), kWriteTo, 0);
        Kernel<MarkRowFlgKernel, cpu>::Launch(s, data_size, row_flg, data.dptr<IType>());
        // calculate inclusive prefix sum
        // TODO(haibin) ideally this is should be done in parallel
        prefix_sum[0] = row_flg[0];
        for (dim_t i = 1; i < num_rows; i++) {
          prefix_sum[i] = prefix_sum[i - 1] + row_flg[i];
        }
        // total number of non-zero rows
        dim_t nnr = prefix_sum[num_rows - 1];
        output.CheckAndAlloc({Shape1(nnr)});
        if (nnr == 0) return;
        RType* grad_row_idx = output.aux_data(kIdx).dptr<RType>();
        // fill row_idx array of output matrix, using the row_flg values
        Kernel<FillRspRowIdxKernel, cpu>::Launch(s, num_rows,
               grad_row_idx, prefix_sum, num_rows);
        // prefill with zeros
        DType* grad_data = output.data().dptr<DType>();
        Kernel<set_zero, cpu>::Launch(s, nnr * row_length, grad_data);
        // add the final gradients
        int num_threads = Engine::Get()->num_omp_threads_per_worker();
        dim_t segment_len = (nnr + num_threads - 1) / num_threads;
        Kernel<AddTakeGradRspKernel, cpu>::Launch(s, num_threads, grad_data, prefix_sum,
                                                  ograd.dptr<DType>(), row_length,
                                                  data.dptr<IType>(), data_size, segment_len,
                                                  num_rows);
      });
    });
  });
}


template<int req>
struct TakeRspKernel {
  /*!
   * \brief
   * \param i           thread id
   * \param data        input data
   * \param out         output
   * \param weight_idx  indices of rsp weight
   * \param weight_data data of rsp weight
   * \param row_length  number of elements per row
   * \param nnr         number of non-zero rows
   */
  template<typename DType, typename IType, typename RType>
  MSHADOW_XINLINE static void Map(int i,
                                  const IType* data,
                                  DType* out,
                                  const RType* weight_idx,
                                  const DType* weight_data,
                                  const nnvm::dim_t row_length,
                                  const nnvm::dim_t nnr) {
    using nnvm::dim_t;
    const dim_t val = static_cast<dim_t>(data[i]);
    const DType zero = 0;
    // Use binary search to find the lower_bound of val in weight_idx array
    // (adapted based on the binary search in dot kernel)
    const RType* first = weight_idx;
    const RType* last = weight_idx + nnr;
    const RType* it;
    dim_t count = last - first, step;
    while (count > 0) {
      it = first;
      step = count / 2;
      it += step;
      if (*it < val) {
        first = ++it;
        count -= step + 1;
      } else {
        count = step;
      }
    }
    // end of binary search
    const dim_t idx_offset = first - weight_idx;
    const dim_t out_offset = i * row_length;
    const dim_t weight_offset = idx_offset * row_length;
    // target idx might be missing in weight.idx. For example,
    // weight.idx = [5,10] and data = [3,7], so binary search fails to
    // find any matching indices in weight_idx.
    if (idx_offset >= nnr || *(weight_idx + idx_offset) > val) {
      // val not found, fill zeros
      for (int j = 0; j < row_length; j++) {
        KERNEL_ASSIGN(out[out_offset + j], req, zero);
      }
    } else {
      for (int j = 0; j < row_length; j++) {
        KERNEL_ASSIGN(out[out_offset + j], req, weight_data[weight_offset + j]);
      }
    }
  }
};

inline void EmbeddingOpForwardRspImpl(mshadow::Stream<mshadow::cpu>* s,
                                      const cpu& cpu_dev,
                                      const TBlob& data,
                                      const NDArray& weight,
                                      const OpReqType req,
                                      const TBlob& output) {
  using namespace mxnet_op;
  using namespace rowsparse;
  MSHADOW_TYPE_SWITCH(output.type_flag_, DType, {
    MSHADOW_TYPE_SWITCH(data.type_flag_, IType, {
      MSHADOW_TYPE_SWITCH(weight.aux_type(kIdx), RType, {
        MXNET_ASSIGN_REQ_SWITCH(req, req_t, {
          size_t data_size = data.shape_.Size();
          // only using the second dim since weight.ndim() == 2
          const nnvm::dim_t row_length = weight.shape()[1];
          Kernel<TakeRspKernel<req_t>, cpu>::Launch(s, data_size, data.dptr<IType>(),
                                                    output.dptr<DType>(),
                                                    weight.aux_data(kIdx).dptr<RType>(),
                                                    weight.data().dptr<DType>(),
                                                    row_length, weight.aux_shape(kIdx)[0]);
        });
      });
    });
  });
}

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_TENSOR_INDEXING_OP_INL_H_
