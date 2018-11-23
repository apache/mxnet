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
* \file digitize_op.cu
* \brief GPU Implementation of the digitize op
* \author Contributors
*/

#include "./digitize_op.h"

#include <thrust/binary_search.h>
#include <thrust/distance.h>

namespace mxnet{
namespace op {

      template<>
      void DigitizeOp::ForwardKernel::Map<gpu>(int i,
                                               const OpContext &ctx,
                                               const TBlob &input_data,
                                               const TBlob &bins,
                                               TBlob &out_data,
                                               const bool right){
        using namespace mshadow;

        auto s = ctx.get_stream<gpu>();

        MSHADOW_TYPE_SWITCH(bins.type_flag_, BType, {
          const Tensor<cpu, 1, BType> bins_tensor = bins.FlatTo1D<gpu, BType>(s);

          MSHADOW_TYPE_SWITCH(input_data.type_flag_, OType, {
            const auto *data = input_data.FlatTo1D<gpu, OType>(s).dptr_;

            auto elem = right ? thrust::lower_bound(bins.dptr_, bins.dptr_ + bins.size(0), data[i])
                              : thrust::upper_bound(bins.dptr_, bins.dptr_ + bins.size(0), data[i]);

            out_data[i] = thrust::distance(bins.dptr_, elem);

          });
        });

      }

  NNVM_REGISTER_OP(diag)
      .set_attr<FCompute>("FCompute<gpu>", DigitizeOpForward < gpu > );

}  // namespace op
}  // namespace mxnet