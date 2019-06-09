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
 * \file histogram.cc
 * \brief CPU implementation of histogram operator
*/
#include "./histogram-inl.h"

namespace mxnet {
namespace op {

struct ComputeBinKernel {
  template<typename DType>
  MSHADOW_XINLINE static void Map(int i, const DType* in_data, const DType* bin_bounds,
                                  int* bin_indices, int bin_cnt, double min, double max) {
    DType data = in_data[i];
    int target = -1;
    if (data >= min && data <= max) {
      target = (data - min) * bin_cnt / (max - min);
      target = mshadow_op::minimum::Map(bin_cnt - 1, target);
      target -= (data < bin_bounds[target]) ? 1 : 0;
      target += ((data >= bin_bounds[target + 1]) && (target != bin_cnt - 1)) ? 1 : 0;
    }
    bin_indices[i] = target;
  }

  template<typename DType>
  MSHADOW_XINLINE static void Map(int i, const DType* in_data, int* bin_indices,
                                   const DType* bin_bounds, int num_bins) {
    DType data = in_data[i];
    int target = -1;
    if (data >= bin_bounds[0] && data <= bin_bounds[num_bins]) {
      target = 0;
      while ((data - bin_bounds[target]) >= 0) {
        target += 1;
      }
      target = mshadow_op::minimum::Map(target - 1, num_bins - 1);
    }
    bin_indices[i] = target;
  }
};

template<typename CType>
void ComputeHistogram(const int* bin_indices, CType* out_data, size_t input_size) {
  for (size_t i = 0; i < input_size; ++i) {
    int target = bin_indices[i];
    if (target >= 0) {
      out_data[target] += 1;
    }
  }
}

template<>
void HistogramForwardImpl<cpu>(const OpContext& ctx,
                               const TBlob& in_data,
                               const TBlob& bin_bounds,
                               const TBlob& out_data,
                               const TBlob& out_bins) {
  using namespace mshadow;
  using namespace mxnet_op;
  mshadow::Stream<cpu> *s = ctx.get_stream<cpu>();
  Tensor<cpu, 1, int> bin_indices =
    ctx.requested[0].get_space_typed<cpu, 1, int>(Shape1(in_data.Size()), s);
  const int bin_cnt = out_data.Size();

  MSHADOW_TYPE_SWITCH(in_data.type_flag_, DType, {
    Kernel<ComputeBinKernel, cpu>::Launch(
      s, in_data.Size(), in_data.dptr<DType>(), bin_indices.dptr_, bin_bounds.dptr<DType>(),
      bin_cnt);
    Kernel<op_with_req<mshadow_op::identity, kWriteTo>, cpu>::Launch(
      s, bin_bounds.Size(), out_bins.dptr<DType>(), bin_bounds.dptr<DType>());
  });
  MSHADOW_TYPE_SWITCH(out_data.type_flag_, CType, {
    Kernel<set_zero, cpu>::Launch(s, bin_cnt, out_data.dptr<CType>());
    ComputeHistogram(bin_indices.dptr_, out_data.dptr<CType>(), in_data.Size());
  });
}

template<>
void HistogramForwardImpl<cpu>(const OpContext& ctx,
                               const TBlob& in_data,
                               const TBlob& out_data,
                               const TBlob& out_bins,
                               const int bin_cnt,
                               const double min,
                               const double max) {
  using namespace mshadow;
  using namespace mxnet_op;
  mshadow::Stream<cpu> *s = ctx.get_stream<cpu>();
  Tensor<cpu, 1, int> bin_indices =
    ctx.requested[0].get_space_typed<cpu, 1, int>(Shape1(in_data.Size()), s);

  MSHADOW_TYPE_SWITCH(in_data.type_flag_, DType, {
    Kernel<FillBinBoundsKernel, cpu>::Launch(
      s, bin_cnt+1, out_bins.dptr<DType>(), bin_cnt, min, max);
    Kernel<ComputeBinKernel, cpu>::Launch(
      s, in_data.Size(), in_data.dptr<DType>(), out_bins.dptr<DType>(), bin_indices.dptr_,
      bin_cnt, min, max);
  });
  MSHADOW_TYPE_SWITCH(out_data.type_flag_, CType, {
    Kernel<set_zero, cpu>::Launch(s, bin_cnt, out_data.dptr<CType>());
    ComputeHistogram(bin_indices.dptr_, out_data.dptr<CType>(), in_data.Size());
  });
}

DMLC_REGISTER_PARAMETER(HistogramParam);

NNVM_REGISTER_OP(_histogram)
.describe(R"code(This operators implements the histogram function.

Example::
  x = [[0, 1], [2, 2], [3, 4]]
  histo, bin_edges = histogram(data=x, bin_bounds=[], bin_cnt=5, range=(0,5))
  histo = [1, 1, 2, 1, 1]
  bin_edges = [0., 1., 2., 3., 4.]
  histo, bin_edges = histogram(data=x, bin_bounds=[0., 2.1, 3.])
  histo = [4, 1]

)code" ADD_FILELINE)
.set_attr_parser(ParamParser<HistogramParam>)
.set_num_inputs([](const NodeAttrs& attrs) {
    const HistogramParam& params = nnvm::get<HistogramParam>(attrs.parsed);
    return params.bin_cnt.has_value() ? 1 : 2;
})
.set_num_outputs(2)
.set_attr<nnvm::FListInputNames>("FListInputNames",
  [](const NodeAttrs& attrs) {
    const HistogramParam& params = nnvm::get<HistogramParam>(attrs.parsed);
    return params.bin_cnt.has_value() ?
           std::vector<std::string>{"data"} :
           std::vector<std::string>{"data", "bins"};
  })
.set_attr<FResourceRequest>("FResourceRequest",
  [](const NodeAttrs& attrs) {
    return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
})
.set_attr<nnvm::FInferShape>("FInferShape", HistogramOpShape)
.set_attr<nnvm::FInferType>("FInferType", HistogramOpType)
.set_attr<FCompute>("FCompute<cpu>", HistogramOpForward<cpu>)
.add_argument("data", "NDArray-or-Symbol", "Input ndarray")
.add_argument("bins", "NDArray-or-Symbol", "Input ndarray")
.add_arguments(HistogramParam::__FIELDS__());

}  // namespace op
}  // namespace mxnet

