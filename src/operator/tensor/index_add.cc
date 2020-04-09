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
 * \file index_add-inl.cc
 * \brief CPU implementation of index_add operator
*/
#include <vector>
#include "./index_add-inl.h"

namespace mxnet {
namespace op {

struct IndexAddCPUKernel {
  template<typename DType, int NDim>
  MSHADOW_XINLINE static void Map(size_t i, DType* a,
                                  int64_t* pre, DType* a_tmp,
                                  const mshadow::Shape<NDim>& a_shape) {
    
  };
};

void IndexAddOpCPUForward(const nnvm::NodeAttrs& attrs,
                          const OpContext &ctx,
                          const std::vector<NDArray> &inputs,
                          const std::vector<OpReqType> &req,
                          const std::vector<NDArray> &outputs) {
  using namespace mxnet_op;
  using namespace mshadow;
  printf("IndexAddOpForward-begin\n");
  printf("inputs.size():%d\n", inputs.size());
  printf("req.size():%d\n", req.size());
  CHECK_EQ(inputs.size(), 2U);
  Stream<cpu> *s = ctx.get_stream<cpu>();
  const IndexAddParam& param = nnvm::get<IndexAddParam>(attrs.parsed);
  NDArray a = inputs[0];
  NDArray val = inputs[1];
  CHECK_NE(a.shape().ndim(), 0) << "Please use '+' instead.";
  int a_ndim = a.shape().ndim();
  int val_ndim = val.shape().ndim();
  int ind_ndim = param.ind.ndim();
  CHECK_LE(ind_ndim, a_ndim) << "IndexError: too many indices for array.";
  // ind=(), dim:0, ind[0] is invalid
  // ind=(1), dim:1, ind[0].ndim():1
  // ind=((0,0),(0,1)), dim:2, ind[0].ndim():2
  if (ind_ndim == 0) {
    // TODO: all position add 'val'
    printf("TODO: dim = 0\n");
    return;
  }
  printf("dim:%d\n", ind_ndim);
  printf("ind[0].size:%d\n", param.ind[0].ndim());

  // get the number of 'ind' index
  int ind_num = 0;
  for (int p_dim = 0; p_dim < ind_ndim; ++p_dim) {
    ind_num = (param.ind[p_dim].ndim() > ind_num) ? param.ind[p_dim].ndim() : ind_num;
  }
  // check 'ind' data legality
  for (int p_dim = 0; p_dim < ind_ndim; ++p_dim) {
    // broadcast check
    CHECK((param.ind[p_dim].ndim() == ind_num) || (param.ind[p_dim].ndim() == 1) || (param.ind[p_dim].ndim() == 0))
      << "IndexError: shape mismatch: indexing arrays could not be broadcast together"
      << " with shapes (" << ind_num << ",) (" << param.ind[p_dim].ndim() << ",)";
    if (param.ind[p_dim].ndim() == 0) {
      // nothing changed
      return;
    }
    // bounds check
    for(int p_num = 0; p_num < param.ind[p_dim].ndim(); ++p_num) {
      CHECK_LE(param.ind[p_dim][p_num], a.shape()[p_dim])
        << "IndexError: index " << param.ind[p_dim][p_num]
        << " is out of bounds for axis " << p_dim
        << " with size " << a.shape()[p_dim];
    }
  }
  // check 'val' broadcast legality
  CHECK_LE(val_ndim, a_ndim - ind_ndim + 1) << "violate brocast regulatoins.";
  for (int i = a_ndim - 1, j = val_ndim - 1; j >= 0 ; --i, --j) {
    if (j == 0 && i == ind_ndim - 1) {
      CHECK(val.shape()[j] == ind_num || val.shape()[j] == 1)
        << "can not broadcast from " << val.shape()[j] << " to " << ind_num;
    } else {
      CHECK(val.shape()[j] == a.shape()[i] || val.shape()[j] == 1)
        << "can not broadcast from " << val.shape()[j] << " to " << a.shape()[i]
        << " in axis " << i;
    }
  }

  // broadcast 'ind'
  size_t vec_size = ind_ndim * ind_num;
  std::vector<int>vec_ind(vec_size);
  for (int p_dim = 0; p_dim < ind_ndim; ++p_dim) {
    for(int p_num = 0; p_num < ind_num; ++p_num) {
      vec_ind[p_dim * ind_num + p_num] = param.ind[p_dim].ndim() == 1 ?
                                         param.ind[p_dim][0] :
                                         param.ind[p_dim][p_num];
    }
  }

  MSHADOW_TYPE_SWITCH(a.dtype(), DType, {
    size_t tmp_mem_size = a.shape().Size() * sizeof(DType) + ind_num * sizeof(int64_t);
    Tensor<cpu, 1, char> tmp_mem = ctx.requested[0].get_space_typed<cpu, 1, char>(
                                   Shape1(tmp_mem_size), s);
    // If different indexes point to the same position, the last value will be added.
    // example:
    // before: a = [[0, 0], [0, 0]], ind = ((0, 0), (0, 0)) val = [1, 2]
    // after index_add(a, val, ind) : a = [[0, 2], [0, 0]]
    int64_t* pre_ptr = reinterpret_cast<int64_t*>(tmp_mem.dptr_);  // record the index of added value
    DType* a_tmp_ptr = reinterpret_cast<DType*>(tmp_mem.dptr_ + ind_num * sizeof(int64_t));
    Tensor<cpu, 1, int64_t> pre(pre_ptr, Shape1(ind_num), s);
    Kernel<set_to_int<-1>, cpu>::Launch(s, ind_num, pre_ptr);
    MXNET_NDIM_SWITCH(a_ndim, NDim, {
      Tensor<cpu, NDim, DType> a_tmp(a_tmp_ptr, a.shape().get<NDim>(), s);
      mxnet_op::copy(s, TBlob(a_tmp), TBlob(a.data()));
      Kernel<IndexAddCPUKernel, cpu>::Launch(s, ind_num, a.data().dptr<DType>(), pre_ptr, a_tmp_ptr,
                                             mxnet_op::calc_stride(a.shape().get<NDim>()));
    });
  });         
}

DMLC_REGISTER_PARAMETER(IndexAddParam);

NNVM_REGISTER_OP(_npx_index_add)
.describe(R"code(This operators implements the "+=" function.
Example::
)code" ADD_FILELINE) // TODO
.set_attr_parser(ParamParser<IndexAddParam>)
.set_num_inputs(2)
.set_num_outputs(0)
.set_attr<nnvm::FListInputNames>("FListInputNames",
  [](const NodeAttrs& attrs) {
    return std::vector<std::string>{"a", "val"};
  })
.set_attr<mxnet::FInferShape>("FInferShape", IndexAddOpShape)
.set_attr<nnvm::FInferType>("FInferType", IndexAddOpType)
.set_attr<FComputeEx>("FComputeEx<cpu>", IndexAddOpCPUForward)
.set_attr<FResourceRequest>("FResourceRequest",
  [](const NodeAttrs& attrs) {
    return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
  })
.add_argument("a", "NDArray-or-Symbol", "Input ndarray")
.add_argument("val", "NDArray-or-Symbol", "Input ndarray")
.add_arguments(IndexAddParam::__FIELDS__());

}  // namespace op
}  // namespace mxnet

