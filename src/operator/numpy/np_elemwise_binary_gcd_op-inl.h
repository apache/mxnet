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
 *  Copyright (c) 2019 by Contributors
 *
 * \file np_gcd_op-inl.h
 * \brief Function definition of greatest common divisor 
 */
#ifndef MXNET_OPERATOR_NUMPY_NP_GCD_OP_INL_H_
#define MXNET_OPERATOR_NUMPY_NP_GCD_OP_INL_H_

#include <vector>
#include "../tensor/broadcast_reduce_op.h"
#include "../src/operator/mxnet_op.h"

namespace mxnet {
namespace op {

template<int req>
struct gcd_forward {
  template<typename DType>
  MSHADOW_XINLINE static void Map(int i,
                                  DType* out_data,
                                  const DType* in_data_1,
                                  const DType* in_data_2) {
    int a = in_data_1[i];
    int b = in_data_2[i];

    // minus cases.
    if (a < 0) {
       a = -a;
    }

    // minus cases.
    if (b < 0) {
       b = -b;
    }

    // handle zero-valued cases.
    int c;
    if (a == 0 && b != 0) {
        c = b;
    } else if (b == 0 && a != 0) {
        c = a;
    } else if (a == 0 && b == 0) {
        c = 0;
    } else {
       if (a < b) {
          std::swap<int>(a, b);
       }
       while (a % b != 0) {
          a = a % b;
          if (a < b) {
             std::swap<int>(a, b);
          }
       }
       c = b;
    }
    KERNEL_ASSIGN(out_data[i], req, c);
  }
};

template<typename xpu>
void GcdOpForward(const nnvm::NodeAttrs& attrs,
                  const OpContext& ctx,
                  const std::vector<TBlob>& inputs,
                  const std::vector<OpReqType>& req,
                  const std::vector<TBlob>& outputs) {
  CHECK_EQ(inputs.size(), 2U);
  CHECK_EQ(outputs.size(), 1U);
  CHECK_EQ(req.size(), 1U);

  mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();
  const TBlob& in_data_1 = inputs[0];
  const TBlob& in_data_2 = inputs[1];
  const TBlob& out_data = outputs[0];

  // if zero buffered, then just return, scalar cases are handled similarly with tensors.
  if (in_data_1.shape_.ndim() == -1 || in_data_2.shape_.ndim() == -1) {
    return;
  }

  using namespace mxnet_op;
  MXNET_INT_TYPE_SWITCH(out_data.type_flag_, DType, {
    MXNET_ASSIGN_REQ_SWITCH(req[0], req_type, {
      Kernel<gcd_forward<req_type>, xpu>::Launch( s, out_data.Size(), out_data.dptr<DType>(),
            in_data_1.dptr<DType>(), in_data_2.dptr<DType>());
    });
  });
}
}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_NUMPY_NP_GCD_OP_INL_H_
