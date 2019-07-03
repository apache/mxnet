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
 * \file np_around_op.h
*/
#ifndef MXNET_OPERATOR_NUMPY_NP_AROUND_OP_H_
#define MXNET_OPERATOR_NUMPY_NP_AROUND_OP_H_

#include <mxnet/operator_util.h>
#include <vector>
#include <cmath>
#include "../mshadow_op.h"
#include "../mxnet_op.h"
#include "../operator_common.h"
#include "../elemwise_op_common.h"
#include "../tensor/init_op.h"

namespace mxnet {
namespace op {

struct AroundParam : public dmlc::Parameter<AroundParam> {
  int decimals;
  DMLC_DECLARE_PARAMETER(AroundParam) {
    DMLC_DECLARE_FIELD(decimals)
      .set_default(0)
      .describe("Number of decimal places to round to.");
  }
};

template<int req>
struct around_forwardint{
  template<typename DType>
  MSHADOW_XINLINE static void Map(int i, DType* out_data, const DType* in_data,
                                  const int decimals) {
    KERNEL_ASSIGN(out_data[i], req, in_data[i]);
  }
};

template<int req>
struct around_forward {
  template<typename DType>
  MSHADOW_XINLINE static void Map(int i, DType* out_data, const DType* in_data,
                                  const int decimals) {
    int d = 0;
    DType temp = in_data[i];
    DType roundtemp;
    while (d != decimals) {
      if (decimals > 0) {
        d++;
        temp *= 10;
      } else {
        d--;
        temp /= 10;
      }
    }
    roundtemp = (DType)round(static_cast<double>(temp));
    // If temp is x.5 and roundtemp is odd number, decrease or increase roundtemp by 1.
    // For example, in numpy, around(0.5) should be 0 but in c, round(0.5) is 1.
    if (roundtemp - temp == 0.5 && (static_cast<int>(roundtemp)) % 2 != 0) {
      roundtemp -= 1;
    } else if (temp - roundtemp == 0.5 && (static_cast<int>(roundtemp)) % 2 != 0) {
      roundtemp += 1;
    }
    while (d != 0) {
      if (roundtemp == 0) {
        break;
      }
      if (decimals > 0) {
        d--;
        roundtemp /= 10;
      } else {
        d++;
        roundtemp *= 10;
      }
    }
    KERNEL_ASSIGN(out_data[i], req, roundtemp);
  }
};

template<typename xpu>
void AroundOpForward(const nnvm::NodeAttrs& attrs,
                     const OpContext& ctx,
                     const std::vector<TBlob>& inputs,
                     const std::vector<OpReqType>& req,
                     const std::vector<TBlob>& outputs) {
  CHECK_EQ(inputs.size(), 1U);
  CHECK_EQ(outputs.size(), 1U);
  CHECK_EQ(req.size(), 1U);
  mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();
  const TBlob& in_data = inputs[0];
  const TBlob& out_data = outputs[0];
  const AroundParam& param = nnvm::get<AroundParam>(attrs.parsed);
  using namespace mxnet_op;
  // if the type is uint8, int8, int32 or int64 and decimals is greater than 0
  // we simply return the number back.
  if (in_data.type_flag_ >= mshadow::kUint8 && in_data.type_flag_ <= mshadow::kInt64 \
     && param.decimals > 0) {
    MSHADOW_TYPE_SWITCH(out_data.type_flag_, DType, {
      MXNET_ASSIGN_REQ_SWITCH(req[0], req_type, {
        Kernel<around_forwardint<req_type>, xpu>::Launch(
          s, out_data.Size(), out_data.dptr<DType>(), in_data.dptr<DType>(),
          param.decimals);
      });
    });
  } else {
    MSHADOW_TYPE_SWITCH(out_data.type_flag_, DType, {
      MXNET_ASSIGN_REQ_SWITCH(req[0], req_type, {
        Kernel<around_forward<req_type>, xpu>::Launch(
          s, out_data.Size(), out_data.dptr<DType>(), in_data.dptr<DType>(),
          param.decimals);
      });
    });
  }
}

}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_NUMPY_NP_AROUND_OP_H_
