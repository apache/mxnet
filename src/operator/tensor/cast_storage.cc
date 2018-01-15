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
 * \file cast_storage.cc
 * \brief CPU Implementation of cast_storage operator.
 */

#include "./cast_storage-inl.h"
#include "../elemwise_op_common.h"
#include "../tensor/elemwise_unary_op.h"
#include "../nn/mkldnn/mkldnn_base-inl.h"

namespace mxnet {
namespace op {

#if MXNET_USE_MKLDNN == 1

static inline int get_type_size(int dtype) {
  MSHADOW_TYPE_SWITCH(dtype, DType, {return sizeof(DType);});
  return -1;
}

void CastStorageMKLDnsImpl(const OpContext& ctx, const NDArray& src, const NDArray &dst_arr) {
  TBlob dns = dst_arr.data();
  CHECK_EQ(ctx.run_ctx.ctx.dev_mask(), Context::kCPU);
  CHECK(src.shape() == dns.shape_);
  if (src.dtype() != dns.type_flag_) {
    // If the input and output have different data types, we have to convert
    // the source array into the default layout, cast the data type and copy
    // data to the destination array.
    const TBlob &src_blob = src.data();
    CHECK(src.ctx() == dst_arr.ctx());
    ndarray::Copy<cpu, cpu>(src.data(), &dns, src.ctx(), dst_arr.ctx(), ctx.run_ctx);
  } else {
    // This converts the source data to the default format and write the data to
    // the destination directly.
    std::vector<mkldnn::primitive> net;
    auto src_mkldnn = src.GetMKLDNNData();
    auto src_pd = src_mkldnn->get_primitive_desc();
    auto def_format = GetDefaultFormat(src_pd.desc());
    if (def_format != src_pd.desc().data.format) {
      auto dst_pd = GetPrimitiveDesc(src_pd, def_format);
      mkldnn::memory dst_mkldnn(dst_pd, dns.dptr_);
      net.push_back(mkldnn::reorder(*src_mkldnn, dst_mkldnn));
      mkldnn::stream(mkldnn::stream::kind::eager).submit(net).wait();
    } else {
      const TBlob &src_blob = src.data();
      memcpy(dns.dptr_, src_blob.dptr_, src.shape().Size() * get_type_size(dns.type_flag_));
    }
  }
}

#endif

DMLC_REGISTER_PARAMETER(CastStorageParam);
NNVM_REGISTER_OP(cast_storage)
.add_alias("_sparse_cast_storage")
.describe(R"code(Casts tensor storage type to the new type.

When an NDArray with default storage type is cast to csr or row_sparse storage,
the result is compact, which means:

- for csr, zero values will not be retained
- for row_sparse, row slices of all zeros will not be retained

The storage type of ``cast_storage`` output depends on stype parameter:

- cast_storage(csr, 'default') = default
- cast_storage(row_sparse, 'default') = default
- cast_storage(default, 'csr') = csr
- cast_storage(default, 'row_sparse') = row_sparse

Example::

    dense = [[ 0.,  1.,  0.],
             [ 2.,  0.,  3.],
             [ 0.,  0.,  0.],
             [ 0.,  0.,  0.]]

    # cast to row_sparse storage type
    rsp = cast_storage(dense, 'row_sparse')
    rsp.indices = [0, 1]
    rsp.values = [[ 0.,  1.,  0.],
                  [ 2.,  0.,  3.]]

    # cast to csr storage type
    csr = cast_storage(dense, 'csr')
    csr.indices = [1, 0, 2]
    csr.values = [ 1.,  2.,  3.]
    csr.indptr = [0, 1, 3, 3, 3]

)code" ADD_FILELINE)
.set_num_inputs(1)
.set_num_outputs(1)
.set_attr_parser(ParamParser<CastStorageParam>)
.set_attr<nnvm::FInferShape>("FInferShape", ElemwiseShape<1, 1>)
.set_attr<nnvm::FInferType>("FInferType", ElemwiseType<1, 1>)
.set_attr<FInferStorageType>("FInferStorageType", CastStorageInferStorageType)
.set_attr<FResourceRequest>("FResourceRequest",
  [](const NodeAttrs& attrs) {
    return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
  })
.set_attr<FCompute>("FCompute<cpu>", UnaryOp::IdentityCompute<cpu>)
.set_attr<FComputeEx>("FComputeEx<cpu>", CastStorageComputeEx<cpu>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseNone{"_copy"})
.add_argument("data", "NDArray-or-Symbol", "The input.")
.add_arguments(CastStorageParam::__FIELDS__());

}  // namespace op
}  // namespace mxnet
