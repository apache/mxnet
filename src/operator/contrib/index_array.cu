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
#include <mshadow/tensor.h>
#include "./index_array-inl.h"

namespace mxnet {
namespace op {

using namespace mshadow::cuda;

void IndexArrayForwardGPU(const nnvm::NodeAttrs &attrs,
                          const OpContext &ctx,
                          const std::vector<TBlob> &inputs,
                          const std::vector<OpReqType> &req,
                          const std::vector<TBlob> &outputs) {
  using namespace mshadow;
  CHECK_EQ(inputs.size(), 1U);
  CHECK_EQ(outputs.size(), 1U);
  CHECK_EQ(req.size(), 1U);
  const TBlob& in_data = inputs[0];
  const TBlob& out_data = outputs[0];

  const IndexArrayParam& param = nnvm::get<IndexArrayParam>(attrs.parsed);

  const TShape inshape = in_data.shape_;
  const int ndim = inshape.ndim();

  Stream<gpu> *s = ctx.get_stream<gpu>();
  cudaStream_t stream = Stream<gpu>::GetStream(s);

  using namespace mxnet_op;

  if (param.axes.has_value()) {
    const mxnet::Tuple<int>& axes = param.axes.value();
    const int naxes = axes.ndim();

    MXNET_IDX_TYPE_SWITCH(param.dtype, DType, {
      std::vector<DType> index_products = IndexArrayComputeIndexProducts<DType>(inshape);

      std::vector<DType> cpu_workspace(2 * naxes);
      IndexArrayBuildSelectedAxesWorkspace(axes, index_products, cpu_workspace.data(), ndim);

      Tensor<gpu, 1, DType> workspace =
          ctx.requested[0].get_space_typed<gpu, 1, DType>(Shape1(2 * naxes), s);

      CUDA_CALL(cudaMemcpyAsync(workspace.dptr_, cpu_workspace.data(), sizeof(DType) * (2 * naxes),
                           cudaMemcpyHostToDevice, stream));

      // Assumes param.target_axis is -1 or 0.
      const ptrdiff_t index_axis_offset = param.target_axis == -1 ? naxes : 1;
      const ptrdiff_t target_axis_offset = param.target_axis == -1 ? 1: in_data.Size();

      MXNET_ASSIGN_REQ_SWITCH(req[0], req_type, {
        Kernel<IndexArrayKernel<req_type>, gpu>::Launch(s, in_data.Size(),
            out_data.dptr<DType>(), naxes, index_axis_offset, target_axis_offset, workspace.dptr_);
      });
    });
  } else {
    Tensor<gpu, 1, dim_t> workspace =
        ctx.requested[0].get_space_typed<gpu, 1, dim_t>(Shape1(ndim), s);

    CUDA_CALL(cudaMemcpyAsync(workspace.dptr_, inshape.data(), sizeof(dim_t) * ndim,
        cudaMemcpyHostToDevice, stream));

    // Assumes param.target_axis is -1 or 0.
    const ptrdiff_t index_axis_offset = param.target_axis == -1 ? ndim : 1;
    const ptrdiff_t target_axis_offset = param.target_axis == -1 ? 1: in_data.Size();

    MXNET_IDX_TYPE_SWITCH(param.dtype, DType, {
      MXNET_ASSIGN_REQ_SWITCH(req[0], req_type, {
        Kernel<IndexArrayDefaultKernel<req_type>, gpu>::Launch(s, in_data.Size(),
                                                               out_data.dptr<DType>(), ndim,
                                                               index_axis_offset,
                                                               target_axis_offset,
                                                               workspace.dptr_);
      });
    });
  }
}

NNVM_REGISTER_OP(_contrib_index_array)
.set_attr<FCompute>("FCompute<gpu>", IndexArrayForwardGPU);

}  // namespace op
}  // namespace mxnet
