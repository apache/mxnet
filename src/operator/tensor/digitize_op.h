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
 * \file digitize_op.h
 * \brief Quantize operator a la numpy.digitize.
 */
#ifndef MXNET_OPERATOR_TENSOR_DIGITIZE_H_
#define MXNET_OPERATOR_TENSOR_DIGITIZE_H_

#include <mxnet/operator_util.h>
#include <vector>
#include <algorithm>
#include "../mshadow_op.h"
#include "../mxnet_op.h"
#include "../operator_common.h"
#include "../elemwise_op_common.h"

namespace mxnet {
    namespace op {

        struct DigitizeParam : public dmlc::Parameter<DigitizeParam> {
            std::vector<float_t> bins; // Is it really float_t that we want?
            bool right;
            DMLC_DECLARE_PARAMETER(DigitizeParam) {
                DMLC_DECLARE_FIELD(right)
                        .set_default(false)
                        .describe("Whether the intervals include the right or the left bin edge.");
            }
        };

        inline bool DigitizeOpShape(const nnvm::NodeAttrs& attrs,
                                 std::vector<TShape>* in_attrs,
                                 std::vector<TShape>* out_attrs) {
            using namespace mshadow;

            CHECK_EQ(in_attrs->size(), 2); // Size 2: data and bins
            CHECK_EQ(out_attrs->size(), 1); // Only one input and one output

            SHAPE_ASSIGN_CHECK(*out_attrs, 0, (*in_attrs)[0]);
            SHAPE_ASSIGN_CHECK(*in_attrs, 0, (*out_attrs)[0]); // TODO: shape of the bins?

            return true;
        }


        struct DigitizeForward {
            template<typename DType>
            MSHADOW_XINLINE static void Map(int i, const DType* in_data, DType* out_data,
                                            const std::vector<float_t> bins, bool right) {

                DType data = in_data[i];
                elem = right? std::upper_bound(bins.begin(), bins.end(), data)
                            : std::lower_bound(bins.begin(), bins.end(), data);

                out_data[i] = std::distance(data.cbegin(), elem);

            }
        };


        template<typename xpu>
        void DigitizeOpForward(const nnvm::NodeAttrs& attrs,
                          const OpContext& ctx,
                          const std::vector<TBlob>& inputs,
                          const std::vector<OpReqType>& req,
                          const std::vector<TBlob>& outputs) {
            using namespace mshadow;
            Stream<xpu> *s = ctx.get_stream<xpu>();
            std::vector<float_t> bins = inputs[1];
            const bool right = nnvm::get<DigitizeParam>(attrs.parsed).right;

            // Check inputs: verify bins is monotonic and ascending
            CHECK_EQ(std::adjacent_find(bins.begin(), bins.end(), std::greater_equal<int>()), bins.end())
                << "Bins vector must be strictly monotonically increasing";


            MSHADOW_TYPE_SWITCH(outputs[0].type_flag_, DType, {
                mxnet_op::Kernel<DigitizeForward, xpu>::Launch(s, inputs, outputs, bins, right);
            });
        }


    }  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_TENSOR_DIGITIZE_H_