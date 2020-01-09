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
 * \file index_copy.cc
 * \brief
 */
#include "./index_copy-inl.h"

namespace mxnet {
namespace op {

struct index_copy_fwd_cpu {
  template<typename DType, typename IType>
  static void Map(index_t i,
                  const DType* new_tensor,
                  const IType* idx,
                  DType* out_tensor,
                  int dim_size) {
    DType* out_ptr = out_tensor + static_cast<index_t>(idx[i]) * dim_size;
    const DType* new_ptr = new_tensor + i * dim_size;
    std::memcpy(out_ptr, new_ptr, sizeof(DType) * dim_size);
  }
};

template<>
void IndexCopyForward<cpu>(const nnvm::NodeAttrs& attrs,
                           const OpContext& ctx,
                           const std::vector<TBlob>& inputs,
                           const std::vector<OpReqType>& req,
                           const std::vector<TBlob>& outputs) {
  using namespace mshadow;
  using namespace mxnet_op;
  CHECK_EQ(inputs.size(), 3U);
  CHECK_EQ(outputs.size(), 1U);
  CHECK_EQ(req.size(), 1U);
  CHECK(req[0] != kAddTo);
  if (req[0] == kNullOp) return;
  mshadow::Stream<cpu> *s = ctx.get_stream<cpu>();
  const TBlob& out = outputs[0];
  const TBlob& original_tensor = inputs[0];
  const TBlob& idx_vector = inputs[1];
  const TBlob& copied_tensor = inputs[2];
  int dim_size = inputs[2].Size() / inputs[1].Size();
  // copy original tensor to output
  copy(s, out, original_tensor);
  // index copy
  MSHADOW_TYPE_SWITCH(out.type_flag_, DType, {
    MSHADOW_TYPE_SWITCH(idx_vector.type_flag_, IType, {
      Kernel<index_copy_fwd_cpu, cpu>::Launch(
        s, idx_vector.Size(), copied_tensor.dptr<DType>(),
        idx_vector.dptr<IType>(), out.dptr<DType>(), dim_size);
    });
  });
}

struct index_copy_bwd_cpu {
  template<typename DType, typename IType>
  static void Map(int i,
                  const DType* out_tensor_grad,
                  DType* orig_tensor_grad,
                  DType* new_tensor_grad,
                  const IType* idx,
                  int dim_size,
                  int idx_size,
                  OpReqType orig_req,
                  OpReqType new_req) {
    const int index = idx[i];
    DType* new_ptr = new_tensor_grad + i * dim_size;
    DType* orig_ptr = orig_tensor_grad + index * dim_size;
    const DType* src_ptr = out_tensor_grad + index * dim_size;
    for (int iter = 0; iter < dim_size; ++iter) {
      KERNEL_ASSIGN(new_ptr[iter], new_req, src_ptr[iter]);
    }
    if (orig_req == kAddTo) {
      for (int iter = 0; iter < dim_size; ++iter) {
        orig_ptr[iter] -= src_ptr[iter];
      }
    } else if (orig_req == kNullOp) {
      return;
    } else {
      std::memset(orig_ptr, 0, sizeof(DType) * dim_size);
    }
  }
};

template<>
void IndexCopyBackward<cpu>(const nnvm::NodeAttrs& attrs,
                            const OpContext& ctx,
                            const std::vector<TBlob>& inputs,
                            const std::vector<OpReqType>& req,
                            const std::vector<TBlob>& outputs) {
  using namespace mshadow;
  using namespace mxnet_op;
  CHECK_EQ(inputs.size(), 4U);
  CHECK_EQ(outputs.size(), 3U);
  Stream<cpu> *s = ctx.get_stream<cpu>();
  const TBlob& out_grad = inputs[0];
  const TBlob& index = inputs[2];
  const TBlob& in_grad_1 = outputs[0];
  const TBlob& in_grad_2 = outputs[2];
  int dim_size = inputs[3].Size() / inputs[2].Size();
  int index_size = inputs[2].Size();
  OpReqType orig_req = req[0];
  OpReqType new_req = req[2];
  // index_copy_backward
  MSHADOW_TYPE_SWITCH(out_grad.type_flag_, DType, {
    MSHADOW_TYPE_SWITCH(index.type_flag_, IType, {
      switch (orig_req) {
        case kNullOp:
          break;
        case kWriteTo:
        case kWriteInplace:
          copy(s, in_grad_1, out_grad);
          break;
        case kAddTo:
          Kernel<op_with_req<op::mshadow_op::plus, kWriteInplace>, cpu>::Launch(
            s, out_grad.Size(), in_grad_1.dptr<DType>(),
            out_grad.dptr<DType>(), in_grad_1.dptr<DType>());
      }
      Kernel<index_copy_bwd_cpu, cpu>::Launch(
        s, index_size, out_grad.dptr<DType>(),
        in_grad_1.dptr<DType>(), in_grad_2.dptr<DType>(),
        index.dptr<IType>(), dim_size, index_size, orig_req, new_req);
    });
  });
}

static bool IndexCopyType(const nnvm::NodeAttrs& attrs,
                          std::vector<int> *in_attrs,
                          std::vector<int> *out_attrs) {
  CHECK_EQ(in_attrs->size(), 3U);
  CHECK_EQ(out_attrs->size(), 1U);
  TYPE_ASSIGN_CHECK(*out_attrs, 0, in_attrs->at(0));
  TYPE_ASSIGN_CHECK(*in_attrs, 0, out_attrs->at(0));
  return out_attrs->at(0) != -1;
}

NNVM_REGISTER_OP(_contrib_index_copy)
.describe(R"code(Copies the elements of a `new_tensor` into the `old_tensor`.

This operator copies the elements by selecting the indices in the order given in `index`.
The output will be a new tensor containing the rest elements of old tensor and
the copied elements of new tensor.
For example, if `index[i] == j`, then the `i` th row of `new_tensor` is copied to the
`j` th row of output.

The `index` must be a vector and it must have the same size with the `0` th dimension of
`new_tensor`. Also, the `0` th dimension of old_tensor must `>=` the `0` th dimension of
`new_tensor`, or an error will be raised.

Examples::

    x = mx.nd.zeros((5,3))
    t = mx.nd.array([[1,2,3],[4,5,6],[7,8,9]])
    index = mx.nd.array([0,4,2])

    mx.nd.contrib.index_copy(x, index, t)

    [[1. 2. 3.]
     [0. 0. 0.]
     [7. 8. 9.]
     [0. 0. 0.]
     [4. 5. 6.]]
    <NDArray 5x3 @cpu(0)>

)code" ADD_FILELINE)
.set_num_inputs(3)
.set_num_outputs(1)
.set_attr<mxnet::FInferShape>("FInferShape", IndexCopyShape)
.set_attr<nnvm::FInferType>("FInferType", IndexCopyType)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseIn{"_contrib_backward_index_copy"})
.set_attr<FCompute>("FCompute<cpu>", IndexCopyForward<cpu>)
.set_attr<nnvm::FListInputNames>("FListInputNames",
  [](const NodeAttrs& attrs) {
    return std::vector<std::string>{"old_tensor", "index_vector", "new_tensor"};
  })
.add_argument("old_tensor", "NDArray-or-Symbol", "Old tensor")
.add_argument("index_vector", "NDArray-or-Symbol", "Index vector")
.add_argument("new_tensor", "NDArray-or-Symbol", "New tensor to be copied");

NNVM_REGISTER_OP(_contrib_backward_index_copy)
.set_num_inputs(4)
.set_num_outputs(3)
.set_attr<nnvm::TIsBackward>("TIsBackward", true)
.set_attr<FCompute>("FCompute<cpu>", IndexCopyBackward<cpu>);

}  // namespace op
}  // namespace mxnet
