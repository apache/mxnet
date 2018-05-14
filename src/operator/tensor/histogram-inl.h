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

#ifndef MXNET_OPERATOR_TENSOR_HISTOGRAM_INL_H_
#define MXNET_OPERATOR_TENSOR_HISTOGRAM_INL_H_

#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <mxnet/operator.h>
#include <mxnet/operator_util.h>
#include <dmlc/optional.h>
#include <mshadow/tensor.h>
#include <nnvm/op.h>
#include <nnvm/node.h>
#include <nnvm/op_attr_types.h>
#include <vector>
#include <type_traits>
#include "./util/tensor_util-inl.h"
#include "../elemwise_op_common.h"
#include "../mshadow_op.h"
#include "../mxnet_op.h"
#include "../operator_common.h"

namespace mxnet {
namespace op {

struct HistogramParam : public dmlc::Parameter<HistogramParam> {
    dmlc::optional<int> bin_cnt;
    dmlc::optional<nnvm::Tuple<float>> range;
    DMLC_DECLARE_PARAMETER(HistogramParam) {
      DMLC_DECLARE_FIELD(bin_cnt)
        .set_default(dmlc::optional<int>())
        .describe("Number of bins for uniform case");
      DMLC_DECLARE_FIELD(range)
        .set_default(dmlc::optional<nnvm::Tuple<float>>())
        .describe("The lower and upper range of the bins. if not provided, "
                  "range is simply (a.min(), a.max()). values outside the "
                  "range are ignored. the first element of the range must be "
                  "less than or equal to the second. range affects the automatic "
                  "bin computation as well. while bin width is computed to be "
                  "optimal based on the actual data within range, the bin count "
                  "will fill the entire range including portions containing no data.");
    }
};

struct FillBinBoundsKernel {
  template<typename DType>
  static MSHADOW_XINLINE void Map(int i, DType* bin_bounds, int bin_cnt, float min, float max) {
    if (i <= bin_cnt) {
      bin_bounds[i] = DType((i * max + (bin_cnt - i) * min) / bin_cnt);
    }
  }
};

inline bool HistogramOpShape(const nnvm::NodeAttrs& attrs,
                             std::vector<TShape>* in_attrs,
                             std::vector<TShape>* out_attrs) {
  CHECK_EQ(in_attrs->size(), 2U);
  CHECK_EQ(out_attrs->size(), 2U);
  HistogramParam param = nnvm::get<HistogramParam>(attrs.parsed);
  const bool has_cnt = param.bin_cnt.has_value();
  const bool has_range = param.range.has_value();
  const bool legal_param = (has_cnt && has_range) || (!has_cnt && !has_range);
  CHECK(legal_param) << "cnt and range should both or neither specified";
  if (has_cnt) {
    // if cnt is specified, the output histogram has shape (cnt,)
    // while output bins has shape (cnt+1,)
    const int bin_cnt = param.bin_cnt.value();
    SHAPE_ASSIGN_CHECK(*out_attrs, 0, TShape({bin_cnt}));
    SHAPE_ASSIGN_CHECK(*out_attrs, 1, TShape({bin_cnt + 1}));
  } else {
    // if cnt is not specified, the output histogram has shape (bins.Size() - 1)
    // while output bins has same shape as input bins
    TShape oshape = (*in_attrs)[1];

    CHECK_EQ(oshape.ndim(), 1U) << "bins argument should be an 1D vector";
    CHECK_GE(oshape.Size(), 2U) << "number of bounds should be >= 2";

    SHAPE_ASSIGN_CHECK(*out_attrs, 0, TShape({(oshape[0] - 1)}));
    SHAPE_ASSIGN_CHECK(*out_attrs, 1, in_attrs->at(1));
  }

  return out_attrs->at(0).ndim() == 1U && out_attrs->at(0).Size() != 0U &&
         out_attrs->at(1).ndim() == 1U && out_attrs->at(1).Size() != 0U &&
         out_attrs->at(0).Size() == out_attrs->at(1).Size() - 1;
}

inline bool HistogramOpType(const nnvm::NodeAttrs& attrs,
                            std::vector<int>* in_attrs,
                            std::vector<int>* out_attrs) {
  CHECK_EQ(in_attrs->size(), 2U);
  CHECK_EQ(in_attrs->at(0), in_attrs->at(1));
  CHECK_EQ(out_attrs->size(), 2U);

  TYPE_ASSIGN_CHECK(*out_attrs, 0, mshadow::kInt64);
  TYPE_ASSIGN_CHECK(*out_attrs, 1, in_attrs->at(0));
  return out_attrs->at(0) != -1 && out_attrs->at(1) != -1;
}

template<typename xpu>
void HistogramForwardImpl(mshadow::Stream<xpu>* s,
                          const OpContext& ctx,
                          const nnvm::NodeAttrs& attrs,
                          const TBlob& in_data,
                          const TBlob& bin_bounds,
                          const TBlob& out_data,
                          const TBlob& out_bins);

template<typename xpu>
void HistogramOpForward(const nnvm::NodeAttrs& attrs,
                        const OpContext& ctx,
                        const std::vector<TBlob>& inputs,
                        const std::vector<OpReqType>& req,
                        const std::vector<TBlob>& outputs) {
  CHECK_EQ(inputs.size(), 2U);
  CHECK_EQ(outputs.size(), 2U);
  CHECK_EQ(req.size(), 2U);
  CHECK_EQ(req[0], kWriteTo);
  CHECK_EQ(req[1], kWriteTo);

  mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();
  const TBlob& in_data = inputs[0];
  const TBlob& bin_bounds = inputs[1];
  const TBlob& out_data = outputs[0];
  const TBlob& out_bins = outputs[1];

  HistogramForwardImpl<xpu>(s, ctx, attrs, in_data, bin_bounds, out_data, out_bins);
}

template<typename xpu>
void HistogramBackwardImpl(const OpContext& ctx,
                          const nnvm::NodeAttrs& attrs,
                          const TBlob& out_grad,
                          const TBlob& in_data,
                          const TBlob& bin_bounds,
                          const TBlob& out_data,
                          const TBlob& in_grad);

template<typename xpu>
void HistogramOpBackward(const nnvm::NodeAttrs& attrs,
                         const OpContext& ctx,
                         const std::vector<TBlob>& inputs,
                         const std::vector<OpReqType>& req,
                         const std::vector<TBlob>& outputs) {
  CHECK_EQ(inputs.size(), 4U);
  CHECK_EQ(outputs.size(), 1U);
  CHECK_EQ(req.size(), 1U);
  CHECK_EQ(req[0], kWriteTo);

  const TBlob& out_grad = inputs[0];
  const TBlob& in_data = inputs[1];
  const TBlob& bin_bounds = inputs[2];
  const TBlob& out_data = inputs[3];
  const TBlob& in_grad = outputs[0];

  HistogramBackwardImpl<xpu>(ctx, attrs, out_grad, in_data, bin_bounds, out_data, in_grad);
}

}   // namespace op
}   // namespace mxnet

#endif  // MXNET_OPERATOR_TENSOR_HISTOGRAM_INL_H_
