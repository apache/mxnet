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
 * Copyright (c) 2018 by Contributors
 * \file shuffle_op.cc
 * \brief Operator to shuffle elements of an NDArray
 */
#include <mxnet/operator_util.h>
#include <algorithm>
#include <random>
#include <vector>
#include "../elemwise_op_common.h"
#include "../tensor/init_op.h"

namespace mxnet {
namespace op {

namespace {

struct CopyForShuffle {
  template<typename DType>
  MSHADOW_XINLINE static void Map(int i, const DType* const in, DType* out,
                                  const index_t* indices, const index_t stride) {
    out[i] = in[indices[i / stride] * stride + i % stride];
  }
};

}  // namespace

void ShuffleForwardGPU(const nnvm::NodeAttrs& attrs,
                       const OpContext& ctx,
                       const std::vector<TBlob>& inputs,
                       const std::vector<OpReqType>& req,
                       const std::vector<TBlob>& outputs) {
  using namespace mxnet_op;
  if (req[0] == kNullOp) {
    return;
  }
  CHECK_NE(req[0], kAddTo) << "Shuffle does not support AddTo";
  const TShape& input_shape = inputs[0].shape_;
  const index_t size = inputs[0].Size();
  const index_t first_axis_len = input_shape[0];
  const index_t stride = size / first_axis_len;
  Stream<gpu> *s = ctx.get_stream<gpu>();
  MSHADOW_TYPE_SWITCH(inputs[0].type_flag_, DType, {
    using KeyType = index_t;
    Tensor<gpu, 1, DType> in = inputs[0].get_with_shape<gpu, 1, DType>(Shape1(size), s);
    Tensor<gpu, 1, DType> out = outputs[0].get_with_shape<gpu, 1, DType>(Shape1(size), s);
    Random<gpu, KeyType> *prnd = ctx.requested[0].get_random<gpu, KeyType>(s);
    if (input_shape.ndim() == 1) {
      if (req[0] != kWriteInplace) {
        Copy(out, in, s);
      }
      Tensor<gpu, 1, KeyType> keys =
        ctx.requested[1].get_space_typed<gpu, 1, KeyType>(Shape1(size), s);
      prnd->GetRandInt(keys);
      SortByKey(keys, out, true);
    } else {
      const size_t tmp_space_size = req[0] == kWriteInplace ?
        2 * first_axis_len * sizeof(index_t) + size * sizeof(DType) :
        2 * first_axis_len * sizeof(index_t);
      Tensor<gpu, 1, char> tmp_space =
        ctx.requested[1].get_space_typed<gpu, 1, char>(Shape1(tmp_space_size), s);
      char* tmp_space_ptr = tmp_space.dptr_;
      Tensor<gpu, 1, index_t> indices(reinterpret_cast<index_t*>(tmp_space_ptr),
                                      Shape1(first_axis_len), s);
      tmp_space_ptr += sizeof(index_t) * first_axis_len;
      Kernel<range_fwd, gpu>::Launch(s, first_axis_len, 1, 0U, 1U, kWriteTo, indices.dptr_);
      Tensor<gpu, 1, KeyType> keys(reinterpret_cast<KeyType*>(tmp_space_ptr),
                                   Shape1(first_axis_len), s);
      tmp_space_ptr += sizeof(KeyType) * first_axis_len;
      prnd->GetRandInt(keys);
      SortByKey(keys, indices, true);
      if (req[0] == kWriteInplace) {
        Tensor<gpu, 1, DType> buf(reinterpret_cast<DType*>(tmp_space_ptr), Shape1(size), s);
        Copy(buf, in, s);
        Kernel<CopyForShuffle, gpu>::Launch(s, size, buf.dptr_, out.dptr_, indices.dptr_, stride);
      } else {
        Kernel<CopyForShuffle, gpu>::Launch(s, size, in.dptr_, out.dptr_, indices.dptr_, stride);
      }
    }
  });
}

NNVM_REGISTER_OP(_shuffle)
.set_attr<FCompute>("FCompute<gpu>", ShuffleForwardGPU);

}  // namespace op
}  // namespace mxnet
