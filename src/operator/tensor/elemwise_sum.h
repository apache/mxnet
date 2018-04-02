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
 * Copyright (c) 2015 by Contributors
 * \file elemwise_sum.h
 * \brief elementwise sum
 * \author Bing Xu
*/
#ifndef MXNET_OPERATOR_TENSOR_ELEMWISE_SUM_H_
#define MXNET_OPERATOR_TENSOR_ELEMWISE_SUM_H_

#include <dmlc/logging.h>
#include <cstring>
#include <vector>
#include "../operator_common.h"
#include "../elemwise_op_common.h"
#include "../mshadow_op.h"
#include "../mxnet_op.h"

namespace mxnet {
namespace op {

struct Sum {
  template<typename DType>
  MSHADOW_XINLINE static DType sum(int i, const DType* a) {
    return a[i];
  }
  template<typename DType, typename... DTypes>
  MSHADOW_XINLINE static DType sum(int i, const DType* a, const DTypes... b) {
    return a[i] + sum(i, b...);
  }
  template<typename DType, typename... DTypes>
  MSHADOW_XINLINE static void Map(int i, DType* out, const OpReqType req, const DType* in0,
    const DTypes... ins) {
    KERNEL_ASSIGN(out[i], req, sum(i, in0, ins...));
  }
};

template<typename xpu, typename DType>
void ElementWiseSumCompute_(const nnvm::NodeAttrs& attrs,
                            const OpContext& ctx,
                            const std::vector<TBlob>& in_data,
                            const std::vector<OpReqType>& req,
                            const std::vector<TBlob>& out_data) {
  using namespace mxnet_op;
  if (req[0] == kNullOp) return;
  size_t size = in_data.size();
  Stream<xpu> *s = ctx.get_stream<xpu>();
  DType* out_dptr = out_data[0].dptr<DType>();
  int out_size = static_cast<int>((out_data[0].Size() + DataType<DType>::kLanes - 1)
                                  /DataType<DType>::kLanes);
  switch (size) {
    case 2: {
      DType* in_0_dptr = in_data[0].dptr<DType>();
      DType* in_1_dptr = in_data[1].dptr<DType>();
      Kernel<Sum, xpu>::Launch(s, out_size, out_dptr, req[0], in_0_dptr, in_1_dptr);
      break;
    }
    case 3: {
      DType* in_0_dptr = in_data[0].dptr<DType>();
      DType* in_1_dptr = in_data[1].dptr<DType>();
      DType* in_2_dptr = in_data[2].dptr<DType>();
      Kernel<Sum, xpu>::Launch(s, out_size, out_dptr, req[0], in_0_dptr, in_1_dptr, in_2_dptr);
      break;
    }
    case 4: {
      DType* in_0_dptr = in_data[0].dptr<DType>();
      DType* in_1_dptr = in_data[1].dptr<DType>();
      DType* in_2_dptr = in_data[2].dptr<DType>();
      DType* in_3_dptr = in_data[3].dptr<DType>();
      Kernel<Sum, xpu>::Launch(s, out_size, out_dptr, req[0], in_0_dptr, in_1_dptr, in_2_dptr,
        in_3_dptr);
      break;
    }
    default: {
      DType* in_0_dptr = in_data[0].dptr<DType>();
      Kernel<Sum, xpu>::Launch(s, out_size, out_dptr, req[0], in_0_dptr);
      for (size_t i = 1; i < size; ++i) {
        DType* in_dptr = in_data[i].dptr<DType>();
        Kernel<Sum, xpu>::Launch(s, out_size, out_dptr, req[0], out_dptr, in_dptr);
      }
      break;
    }
  }
}

template<typename xpu>
void ElementWiseSumCompute(const nnvm::NodeAttrs& attrs,
                           const OpContext& ctx,
                           const std::vector<TBlob>& inputs,
                           const std::vector<OpReqType>& req,
                           const std::vector<TBlob>& outputs) {
  CHECK_EQ(outputs.size(), 1U);
  MSHADOW_TYPE_SWITCH(outputs[0].type_flag_, DType, {
      ElementWiseSumCompute_<xpu, DType>(attrs, ctx, inputs, req, outputs);
  });
}

template<typename xpu>
void ElementWiseSumComputeWithHalf2(const nnvm::NodeAttrs& attrs,
                                    const OpContext& ctx,
                                    const std::vector<TBlob>& inputs,
                                    const std::vector<OpReqType>& req,
                                    const std::vector<TBlob>& outputs) {
  CHECK_EQ(outputs.size(), 1U);
  MSHADOW_TYPE_SWITCH_WITH_HALF2(outputs[0].type_flag_, DType, {
      ElementWiseSumCompute_<xpu, DType>(attrs, ctx, inputs, req, outputs);
  });
}

}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_TENSOR_ELEMWISE_SUM_H_
