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
#include "../mshadow_op.h"
#include "../mxnet_op.h"
#include "../operator_common.h"
#include "../elemwise_op_common.h"
#include <mxnet/base.h>

namespace mxnet {
namespace op {

struct DigitizeParam : public dmlc::Parameter<DigitizeParam> {
  bool right;

  DMLC_DECLARE_PARAMETER(DigitizeParam) {
    DMLC_DECLARE_FIELD(right)
        .set_default(false)
        .describe("Whether the intervals include the right or the left bin edge.");
  }
};

class DigitizeOp {
public:
  bool InferShape(const nnvm::NodeAttrs &attrs,
                  std::vector<TShape> *in_attrs,
                  std::vector<TShape> *out_attrs) {
    using namespace mshadow;

    CHECK_EQ(in_attrs->size(), 2); // Size 2: data and bins
    CHECK_EQ(out_attrs->size(), 1); // Only one output tensor

    const auto &bin_size = in_attrs->at(1).Size();
    CHECK_LE(bin_size, 2); // Size <= 2 for bins

    auto &input_shape = (*in_attrs)[0];
    auto &output_shape = (*out_attrs)[0];

    SHAPE_ASSIGN_CHECK(*out_attrs, 0, input_shape); // First arg is a shape array
    SHAPE_ASSIGN_CHECK(*in_attrs, 0, output_shape);

    // If bins has two dimensions, the first one corresponds to the batch axis, so we need to verify
    // # batches in X = # batches in bins
    if (bin_size == 2) {
      auto &input_batches = (*in_attrs)[0][0];
      auto &bin_batches = (*in_attrs)[1][0];

      CHECK_EQ(input_batches, bin_batches)
        << "If bins has 2 dimensions, the first one should be the same as that of the input data";
      //TODO: Reword the message above
    }

    return true;
  }

  struct ForwardKernel {
    template<typename xpu>
    MSHADOW_XINLINE static void Map(int i,
                                    DType *in_data,
                                    DType *out_data,
                                    mshadow::Tensor<xpu, 1, BType> bins,
                                    bool right);
  };

  template<class ForwardIterator, typename DType>
  void CheckMonotonic(ForwardIterator begin, ForwardIterator end) {
    // adjacent_find here returns the begin element that's >= than the next one or the last element
    CHECK_EQ(std::adjacent_find(begin, end, std::greater_equal<DType>()), end)
      << "Bins vector must be strictly monotonically increasing";
  }

  // Based on http://www.cplusplus.com/reference/algorithm/is_sorted/
//            template<class ForwardIterator>
//            bool CheckMonotonic(ForwardIterator first, ForwardIterator last) {
//                if (first == last) return true;
//                ForwardIterator next = first;
//                while (++next != last) {
//                    if (*next <= *first)
//                        return false;
//                    ++first;
//                }
//                return true;
//            }

//            template<typename xpu, typename BType>
//            void DigitizeTensor(const nnvm::NodeAttrs &attrs,
//                                const std::vector<TBlob> &inputs,
//                                const std::vector<TBlob> &outputs) {
//                const auto *data = bins_tensor[i].dptr_;
//                CheckMonotonic(data, data + bins_tensor[i].MemSize());
//
//                MSHADOW_TYPE_SWITCH(inputs[0].type_flag_, DType, {
//                    mxnet_op::Kernel<ForwardKernel, xpu>::Launch(s, inputs, outputs, right);
//                });
//            }


  template<typename xpu>
  void Forward(const nnvm::NodeAttrs &attrs,
               const OpContext &ctx,
               const std::vector<TBlob> &inputs,
               const std::vector<OpReqType> &req,
               const std::vector<TBlob> &outputs) {
    using namespace mshadow;

    auto s = ctx.get_stream<xpu>();
    const bool right = nnvm::get<DigitizeParam>(attrs.parsed).right;

    // Verify bins is strictly monotonic
    MSHADOW_TYPE_SWITCH(inputs[1].type_flag_, BType, {
      const auto &bins = inputs[1];
      const auto bin_dims = bins.shape_.Size();

      if (bin_dims == 1) {
        const Tensor<xpu, 1, BType> bins_tensor = bins.FlatTo1D(s);
        const auto *data = bins_tensor.dptr_;
        CheckMonotonic(data, data + bins_tensor.size(0));
      } else {
        const Tensor<xpu, 2, BType> bins_tensor = bins.FlatTo2D(s);

        for (auto i = 0; i < bins_tensor.size(0); ++i) {
          const auto *data = bins_tensor[i].dptr_;
          CheckMonotonic(data, data + bins_tensor[i].size(0));

          MSHADOW_TYPE_SWITCH(inputs[0].type_flag_, DType, {
              mxnet_op::Kernel<ForwardKernel, xpu>::Launch(s, inputs[0], outputs, bins_tensor[i],
                                                           right);
          });
        }
      }

    });
  }
};

template<typename xpu, typename DType, typename BType>
void DigitizeOp::ForwardKernel::Map<xpu>(int i,
                                         DType *in_data,
                                         DType *out_data,
                                         mshadow::Tensor<xpu, 1, BType> bins,
                                         bool right);


}  // namespace op
}// namespace mxnet

#endif  // MXNET_OPERATOR_TENSOR_DIGITIZE_H_