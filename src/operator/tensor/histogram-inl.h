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
 * \file histogram-inl.h
 * \brief Function definition of histogram operator
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
    dmlc::optional<nnvm::Tuple<double>> range;
    DMLC_DECLARE_PARAMETER(HistogramParam) {
      DMLC_DECLARE_FIELD(bin_cnt)
        .set_default(dmlc::optional<int>())
        .describe("Number of bins for uniform case");
      DMLC_DECLARE_FIELD(range)
        .set_default(dmlc::optional<nnvm::Tuple<double>>())
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
  static MSHADOW_XINLINE void Map(int i, DType* bin_bounds, int bin_cnt, double min, double max) {
    if (i <= bin_cnt) {
      bin_bounds[i] = DType((max * i + (bin_cnt - i) * min) / bin_cnt);
    }
  }
};

inline bool HistogramOpShape(const nnvm::NodeAttrs& attrs,
                             std::vector<TShape>* in_attrs,
                             std::vector<TShape>* out_attrs) {
  HistogramParam param = nnvm::get<HistogramParam>(attrs.parsed);
  const bool has_cnt = param.bin_cnt.has_value();
  const bool has_range = param.range.has_value();
  const bool legal_param = (has_cnt && has_range) || (!has_cnt && !has_range);
  CHECK_EQ(in_attrs->size(), has_cnt ? 1U : 2U);
  CHECK_EQ(out_attrs->size(), 2U);
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

  return !shape_is_none(out_attrs->at(0)) && !shape_is_none(out_attrs->at(1)) &&
         out_attrs->at(0).Size() == out_attrs->at(1).Size() - 1;
}

inline bool HistogramOpType(const nnvm::NodeAttrs& attrs,
                            std::vector<int>* in_attrs,
                            std::vector<int>* out_attrs) {
  CHECK_EQ(out_attrs->size(), 2U);

  TYPE_ASSIGN_CHECK(*out_attrs, 0, mshadow::kInt64);
  TYPE_ASSIGN_CHECK(*out_attrs, 1, in_attrs->at(0));
  return !type_is_none(out_attrs->at(0)) && !type_is_none(out_attrs->at(1));
}

template<typename xpu>
void HistogramForwardImpl(const OpContext& ctx,
                          const TBlob& in_data,
                          const TBlob& bin_bounds,
                          const TBlob& out_data,
                          const TBlob& out_bins);

template<typename xpu>
void HistogramForwardImpl(const OpContext& ctx,
                          const TBlob& in_data,
                          const TBlob& out_data,
                          const TBlob& out_bins,
                          const int bin_cnt,
                          const double min,
                          const double max);

template<typename xpu>
void HistogramOpForward(const nnvm::NodeAttrs& attrs,
                        const OpContext& ctx,
                        const std::vector<TBlob>& inputs,
                        const std::vector<OpReqType>& req,
                        const std::vector<TBlob>& outputs) {
  CHECK_EQ(req.size(), 2U);
  CHECK_EQ(req[0], kWriteTo);
  CHECK_EQ(req[1], kWriteTo);
  const HistogramParam& param = nnvm::get<HistogramParam>(attrs.parsed);
  const bool has_cnt = param.bin_cnt.has_value();
  const bool has_range = param.range.has_value();
  const bool legal_params = (has_cnt && has_range) || (!has_cnt && !has_range);
  CHECK(legal_params) << "width and range should both or neither be specified";

  const TBlob& in_data = inputs[0];
  const TBlob& out_data = outputs[0];
  const TBlob& out_bins = outputs[1];

  if (has_cnt) {
    CHECK((param.range.value().ndim() == 2U)) << "range should be a tuple with only 2 elements";
    CHECK(param.range.value()[0] <= param.range.value()[1])
      << "left hand side of range(" << param.range.value()[0]
      << ")should be less than or equal to right hand side(" << param.range.value()[1] << ")";
    double max = param.range.value()[1];
    double min = param.range.value()[0];
    const int bin_cnt = param.bin_cnt.value();
    if (min == max) {
      min -= 0.5f;
      max += 0.5f;
      LOG(INFO) << min << " " << max;
    }
    HistogramForwardImpl<xpu>(ctx, in_data, out_data, out_bins, bin_cnt, min, max);
  } else {
    const TBlob& bin_bounds = inputs[1];
    HistogramForwardImpl<xpu>(ctx, in_data, bin_bounds, out_data, out_bins);
  }
}

}   // namespace op
}   // namespace mxnet

#endif  // MXNET_OPERATOR_TENSOR_HISTOGRAM_INL_H_
