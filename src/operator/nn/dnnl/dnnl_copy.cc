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
 * \file dnnl_copy.cc
 * \brief
 * \author
 */

#include "dnnl_base-inl.h"

#if MXNET_USE_ONEDNN == 1
namespace mxnet {
namespace op {

void DNNLCopy(const nnvm::NodeAttrs& attrs,
              const OpContext& ctx,
              const NDArray& in_data,
              const OpReqType& req,
              const NDArray& out_data) {
  if (req == kNullOp || req == kWriteInplace)
    return;
  TmpMemMgr::Get()->Init(ctx.requested[0]);
  auto in_mem = in_data.GetDNNLData();
  if (req == kAddTo) {
    TmpMemMgr::Get()->Init(ctx.requested[0]);
    // We should try and force the input memory has the same format
    // as the input output. If not, we'll have to reorder memory.
    auto out_mem      = out_data.GetDNNLData();
    auto out_mem_desc = out_mem->get_desc();
    in_mem            = in_data.GetDNNLData(&out_mem_desc);
    if (in_mem == nullptr)
      in_mem = in_data.GetDNNLDataReorder(&out_mem_desc);
    DNNLMemorySum(*out_mem, *in_mem, *out_mem);
  } else {
    const_cast<NDArray&>(out_data).CopyFrom(*in_mem);
  }
  DNNLStream::Get()->Submit();
}

}  // namespace op
}  // namespace mxnet
#endif
