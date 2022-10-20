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
 * \file dnnl_dot.cc
 */

#if MXNET_USE_ONEDNN == 1

#include <memory>
#include <unordered_map>

#include "dnnl_dot-inl.h"

namespace mxnet {
namespace op {

// Support for https://oneapi-src.github.io/oneDNN/v2.6/dev_guide_matmul.html
bool SupportDNNLDot(const std::vector<NDArray>& inputs) {
#if MXNET_USE_BLAS_MKL == 1
  return false;
#endif
  // Remove cases where ndim of inputs is equal to 1, because output will be scalar in this case
  return SupportDNNL<2, 12, DNNLTypeMode::FloatTypes>(inputs[DotIn::lhs]) &&
         SupportDNNL<2, 12, DNNLTypeMode::FloatTypes>(inputs[DotIn::rhs]);
}

DNNLDotFwd& DNNLDotFwd::GetCached(const DotParam& param,
                                  const std::vector<NDArray>& inputs,
                                  const std::vector<NDArray>& outputs,
                                  const bool isNumpy) {
  using dot_fwd_map = std::unordered_map<DotSignature, DNNLDotFwd, OpHash>;
#if DMLC_CXX11_THREAD_LOCAL
  static thread_local dot_fwd_map fwds;
#else
  static MX_THREAD_LOCAL dot_fwd_map fwds;
#endif

  DotSignature key(param);
  key.AddSign(inputs[DotIn::lhs]);
  key.AddSign(inputs[DotIn::rhs]);
  key.AddSign(outputs[DotOut::out]);
  key.AddSign(static_cast<int>(isNumpy));

  auto it = fwds.find(key);
  if (it == fwds.end()) {
    const DNNLDotFwd fwd(param, inputs, outputs, isNumpy);
    it = AddToCache(&fwds, key, fwd);
  }
  return it->second;
}

auto GetMemoryDesc(const NDArray& tensor, int firstDim, int secondDim, const bool transpose) {
  return dnnl::memory::desc(
      dnnl::memory::dims{firstDim, secondDim},
      get_dnnl_type(tensor.dtype()),
      transpose ? dnnl::memory::format_tag::ba : dnnl::memory::format_tag::ab);
}

DNNLDotFwd::DNNLDotFwd(const DotParam& param,
                       const std::vector<NDArray>& inputs,
                       const std::vector<NDArray>& outputs,
                       const bool isNumpy) {
  auto shapeLhs = inputs[DotIn::lhs].shape(), shapeRhs = inputs[DotIn::rhs].shape();
  auto ndimLhs = shapeLhs.ndim(), ndimRhs = shapeRhs.ndim();
  dnnl::memory::desc lhs_md, rhs_md, out_md;
  // NumPy expects more than 2 dimensional rhs tensor as Ax...xKxN which is different than NDArray's
  // KxAx...xN format. For NumPy shape in rhs memory descriptor is going to be Kx(A*...*N),
  // similarly to NDArray, but for it to match the actual data there is an additional reorder
  // needed, permuting the last two axes Ax...xKxN -> Ax...xNxK. For this data to match Kx(A*...*N)
  // shape format_tag needs to be set to ba. Reorder described above is implemented in Execute
  // function.
  const bool differentNumpy = isNumpy && ndimRhs > 2;
  const int smallDimLhs     = param.transpose_a ? shapeLhs[0] : shapeLhs[ndimLhs - 1];
  const int bigDimLhs       = shapeLhs.Size() / smallDimLhs;
  const int smallDimRhs     = param.transpose_b ?
                              shapeRhs[ndimRhs - 1] :
                              (differentNumpy ? shapeRhs[ndimRhs - 2] : shapeRhs[0]);
  const int bigDimRhs = shapeRhs.Size() / smallDimRhs;

  lhs_md = GetMemoryDesc(inputs[DotIn::lhs], bigDimLhs, smallDimLhs, param.transpose_a);
  rhs_md = GetMemoryDesc(
      inputs[DotIn::rhs], smallDimRhs, bigDimRhs, param.transpose_b || differentNumpy);
  out_md = dnnl::memory::desc({bigDimLhs, bigDimRhs},
                              get_dnnl_type(outputs[DotOut::out].dtype()),
                              dnnl::memory::format_tag::any);
  dnnl::matmul::desc fwd_desc(lhs_md, rhs_md, out_md);
  fwd_pd = std::make_shared<dot_fwd_pd_t>(fwd_desc, mxnet::CpuEngine::Get()->get_engine());
  fwd    = std::make_shared<dot_fwd_t>(*fwd_pd);
}

void DNNLDotFwd::Execute(const OpContext& ctx,
                         const std::vector<NDArray>& inputs,
                         const std::vector<OpReqType>& req,
                         const std::vector<NDArray>& outputs,
                         const bool isNumpy) {
  auto engine = mxnet::CpuEngine::Get()->get_engine();
  auto lhs    = dnnl::memory(
      fwd_pd->src_desc(), engine, reinterpret_cast<void*>(inputs[DotIn::lhs].data().dptr_));
  auto ndimRhs                = inputs[DotIn::rhs].shape().ndim();
  const bool specialNumpyCase = isNumpy && ndimRhs > 2;
  auto rhsMemPointer =
      specialNumpyCase ?
          reinterpret_cast<void*>(
              ctx.requested[0]
                  .get_space<cpu>(mshadow::Shape1(inputs[DotIn::rhs].shape().Size() *
                                                  GetTypeSize(inputs[DotIn::rhs].dtype())),
                                  ctx.get_stream<cpu>())
                  .dptr_) :
          reinterpret_cast<void*>(inputs[DotIn::rhs].data().dptr_);
  dnnl::memory rhs(fwd_pd->weights_desc(), engine, rhsMemPointer);
  if (specialNumpyCase) {
    // Necessity of this reorder is described in DNNLDotFwd constructor.
    auto tmp_rhs = inputs[DotIn::rhs].GetDNNLData();
    dnnl::memory::desc rhs_md(
        dnnl::memory::dims(inputs[DotIn::rhs].shape().begin(), inputs[DotIn::rhs].shape().end()),
        get_dnnl_type(inputs[DotIn::rhs].dtype()),
        static_cast<dnnl::memory::format_tag>(GetPermutedFormat(ndimRhs)));
    dnnl::memory tmp_rhs_dst(rhs_md, engine, rhs.get_data_handle());
    const auto rhs_reorder_pd = dnnl::reorder::primitive_desc(*tmp_rhs, tmp_rhs_dst);
    DNNLStream::Get()->RegisterPrimArgs(dnnl::reorder(rhs_reorder_pd),
                                        {{DNNL_ARG_FROM, *tmp_rhs}, {DNNL_ARG_TO, tmp_rhs_dst}});
  }
  dnnl_output_t out_mem = CreateDNNLMem(
      outputs[DotOut::out], fwd_pd->dst_desc(), req[DotOut::out], &inputs[DotIn::lhs]);

  dnnl_args_map_t args = {
      {DNNL_ARG_SRC, lhs},
      {DNNL_ARG_WEIGHTS, rhs},
      {DNNL_ARG_DST, *out_mem.second},
  };

  DNNLStream::Get()->RegisterPrimArgs(*fwd, args);
  CommitOutput(outputs[DotOut::out], out_mem);
  DNNLStream::Get()->Submit();
}

}  // namespace op
}  // namespace mxnet
#endif  // MXNET_USE_ONEDNN == 1
