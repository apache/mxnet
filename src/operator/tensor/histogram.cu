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
 * \file histogram.cu
 * \brief GPU implementation of histogram operator
*/
#include "./histogram-inl.h"
#include "./util/tensor_util-inl.cuh"

namespace mxnet {
namespace op {

struct HistogramFusedKernel {
  template<typename DType, typename CType>
  static MSHADOW_XINLINE void Map(int i, const DType* in_data, const DType* bin_bounds, CType* bins,
                                  const int bin_cnt, const double min, const double max) {
    DType data = in_data[i];
    int target = -1;
    if (data >= min && data <= max) {
      target = mshadow_op::floor::Map((data - min) * bin_cnt / (max - min));
      target = mshadow_op::minimum::Map(bin_cnt - 1, target);
      target -= (data < bin_bounds[target]) ? 1 : 0;
      target += ((data >= bin_bounds[target + 1]) && (target != bin_cnt - 1)) ? 1 : 0;
    }
    if (target >= 0) {
      atomicAdd(&bins[target], CType(1));
    }
  }

  template<typename DType, typename CType>
  static MSHADOW_XINLINE void Map(int i, const DType* in_data, const DType* bin_bounds, CType* bins,
                                  const int bin_cnt) {
    DType data = in_data[i];
    int target = -1;
    if (data >= bin_bounds[0] && data <= bin_bounds[bin_cnt]) {
      target = 0;
      while (data >= bin_bounds[target]) {
        target += 1;
      }
      target = min(target - 1, bin_cnt - 1);
    }
    if (target >= 0) {
      atomicAdd(&bins[target], CType(1));
    }
  }
};

template<>
void HistogramForwardImpl<gpu>(const OpContext& ctx,
                               const TBlob& in_data,
                               const TBlob& bin_bounds,
                               const TBlob& out_data,
                               const TBlob& out_bins) {
  using namespace mshadow;
  using namespace mxnet_op;
  mshadow::Stream<gpu> *s = ctx.get_stream<gpu>();
  MSHADOW_TYPE_SWITCH(in_data.type_flag_, DType, {
    MSHADOW_IDX_TYPE_SWITCH(out_data.type_flag_, CType, {
      int bin_cnt = out_bins.Size() - 1;
      Kernel<set_zero, gpu>::Launch(s, bin_cnt, out_data.dptr<CType>());
      Kernel<HistogramFusedKernel, gpu>::Launch(
        s, in_data.Size(), in_data.dptr<DType>(), bin_bounds.dptr<DType>(),
        out_data.dptr<CType>(), bin_cnt);
      Kernel<op_with_req<mshadow_op::identity, kWriteTo>, gpu>::Launch(
        s, bin_bounds.Size(), out_bins.dptr<DType>(), bin_bounds.dptr<DType>());
    });
  });
}

template<>
void HistogramForwardImpl<gpu>(const OpContext& ctx,
                               const TBlob& in_data,
                               const TBlob& out_data,
                               const TBlob& out_bins,
                               const int bin_cnt,
                               const double min,
                               const double max) {
  using namespace mshadow;
  using namespace mxnet_op;
  mshadow::Stream<gpu> *s = ctx.get_stream<gpu>();
  MSHADOW_TYPE_SWITCH(in_data.type_flag_, DType, {
    MSHADOW_IDX_TYPE_SWITCH(out_data.type_flag_, CType, {
      Kernel<set_zero, gpu>::Launch(s, bin_cnt, out_data.dptr<CType>());
      Kernel<FillBinBoundsKernel, gpu>::Launch(
        s, bin_cnt+1, out_bins.dptr<DType>(), bin_cnt, min, max);
      Kernel<HistogramFusedKernel, gpu>::Launch(
        s, in_data.Size(), in_data.dptr<DType>(), out_bins.dptr<DType>(), out_data.dptr<CType>(),
        bin_cnt, min, max);
    });
  });
}

NNVM_REGISTER_OP(_histogram)
.set_attr<FCompute>("FCompute<gpu>", HistogramOpForward<gpu>);

}   // namespace op
}   // namespace mxnet

