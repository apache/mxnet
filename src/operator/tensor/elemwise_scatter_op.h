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
 * \file elementwise_scatter_op.h
 * \brief Function definition of elementwise scatter operators
 */
#ifndef MXNET_OPERATOR_TENSOR_ELEMWISE_SCATTER_OP_H_
#define MXNET_OPERATOR_TENSOR_ELEMWISE_SCATTER_OP_H_

#include "./elemwise_binary_op.h"
#include "./elemwise_binary_scalar_op.h"
#include "sparse_retain-inl.h"
#include "../../../tests/cpp/include/test_util.h"

namespace mxnet {
namespace op {

/*! \brief Execute the supplied function/operation, followed by a sparse retain operation
 * of the lhs argument's rows only (row indices). */
template <typename xpu, typename Function>
inline void ScatterWrap(const nnvm::NodeAttrs &attrs,
                        const OpContext &ctx,
                        const std::vector<NDArray> &inputs,
                        const std::vector<OpReqType> &req,
                        const std::vector<NDArray> &outputs,
                        bool pre_retain,
                        Function function) {
  CHECK_EQ(outputs.size(), 1U);
  if (inputs[0].storage_type() == kRowSparseStorage
      && outputs[0].storage_type() == kRowSparseStorage) {
    if (pre_retain && inputs[1].storage_type() == kRowSparseStorage) {
      // Retain only rhs rows which have same row as lhs input
      NDArray retained_input(outputs[0].storage_type(), outputs[0].shape(), outputs[0].ctx());
      SparseRetainOpForwardEx<xpu>(attrs, ctx,
                                   { inputs[1], inputs[0].aux_ndarray(rowsparse::kIdx) },
                                   req,
                                   {retained_input});
      CHECK(retained_input.storage_initialized());
      // Perform the operation
      function(attrs, ctx, {inputs[0], retained_input}, req, outputs);
      // Sanity check
      DCHECK_LE(outputs[0].aux_shape(rowsparse::kIdx).Size(),
                inputs[0].aux_shape(rowsparse::kIdx).Size());
    } else {
      // Perform the operation as usual
      NDArray temp_out(outputs[0].storage_type(), outputs[0].shape(), outputs[0].ctx());
      function(attrs, ctx, inputs, req, { temp_out });
      CHECK(temp_out.storage_initialized());
      CHECK_EQ(temp_out.storage_type(), kRowSparseStorage);
      // Sparse-retain the output based upon lhs-input sparsity
      const NDArray indices(inputs[0].aux_data(rowsparse::kIdx), inputs[0].ctx().dev_id);
      SparseRetainOpForwardEx<xpu>(attrs, ctx, { temp_out, indices },
                                   req, outputs);
      DCHECK_LE(outputs[0].aux_shape(rowsparse::kIdx).Size(),
                inputs[0].aux_shape(rowsparse::kIdx).Size());
    }
  } else {
    function(attrs, ctx, inputs, req, outputs);
  }
}

/*! \brief Scatter elemwise binary op handlers */
class ElemwiseScatterBinaryOp : public ElemwiseBinaryOp {
  /*! \brief  CPU version, RspRsp knows how to do an efficient scatter,
   * otherwise retain rhs + normal op */
  template<typename OP>
  static void ComputeEx_(mshadow::Stream<cpu> *stream,
                         const nnvm::NodeAttrs &attrs,
                         const OpContext &ctx,
                         const std::vector<NDArray> &inputs,
                         const std::vector<OpReqType> &req,
                         const std::vector<NDArray> &outputs) {
    // row_sparse-op-row_sparse or row_sparse-op-default can call RspRsp
    const NDArrayStorageType input1_stype = inputs[1].storage_type();
    if (inputs[0].storage_type() == kRowSparseStorage
        && (input1_stype == kRowSparseStorage || input1_stype == kDefaultStorage)
        && outputs[0].storage_type() == kRowSparseStorage) {
      mshadow::Stream<cpu> *s = ctx.get_stream<cpu>();
      MSHADOW_TYPE_SWITCH(inputs[0].dtype(), DType, {
        MSHADOW_IDX_TYPE_SWITCH(inputs[0].aux_type(rowsparse::kIdx), IType, {
          RspRspOp<DType, IType, OP>(s, attrs, ctx, inputs[0], inputs[1], req[0], outputs[0],
                                     false, true, false, true);
        });
      });
      CHECK_EQ(inputs[0].aux_shape(rowsparse::kIdx).Size(),
               outputs[0].aux_shape(rowsparse::kIdx).Size());
    } else {
      ScatterWrap<cpu>(attrs, ctx, inputs, req,
                       outputs, true, [](const nnvm::NodeAttrs &attrs,
                                         const OpContext &ctx,
                                         const std::vector<NDArray> &inputs,
                                         const std::vector<OpReqType> &req,
                                         const std::vector<NDArray> &outputs) {
          ElemwiseBinaryOp::ComputeEx<cpu, OP>(attrs, ctx, inputs, req, outputs);
        });
    }
  }

#ifdef __CUDACC__
  /*! \brief GPU version, fallback op + retain */
  template<typename OP>
  static void ComputeEx_(mshadow::Stream<gpu> *stream,
                         const nnvm::NodeAttrs &attrs,
                         const OpContext &ctx,
                         const std::vector<NDArray> &inputs,
                         const std::vector<OpReqType> &req,
                         const std::vector<NDArray> &outputs) {
    ScatterWrap<gpu>(attrs, ctx, inputs, req,
                     outputs, false, [](const nnvm::NodeAttrs &attrs,
                                        const OpContext &ctx,
                                        const std::vector<NDArray> &inputs,
                                        const std::vector<OpReqType> &req,
                                        const std::vector<NDArray> &outputs) {
        FCompExFallback<gpu>(attrs, ctx, inputs, req, outputs, ElemwiseBinaryOp::Compute<gpu, OP>,
                             "ComputeEx_");
      });
  }
#endif  // #ifdef __CUDACC__

 public:
  template<typename xpu, typename OP>
  static void ComputeEx(const nnvm::NodeAttrs &attrs,
                        const OpContext &ctx,
                        const std::vector<NDArray> &inputs,
                        const std::vector<OpReqType> &req,
                        const std::vector<NDArray> &outputs) {
    DCHECK_EQ(inputs.size(), 2U);
    DCHECK_EQ(outputs.size(), 1U);
    ComputeEx_<OP>(ctx.get_stream<xpu>(), attrs, ctx, inputs, req, outputs);
  }
};

/*! \brief Scatter elemwise binary scalar op handlers */
class ElemwiseScatterBinaryScalarOp : public BinaryScalarOp {
  /*! \brief  CPU version, retain rhs + normal op */
  template<typename OP>
  static void ComputeEx_(mshadow::Stream<cpu> *stream,
                         const nnvm::NodeAttrs &attrs,
                         const OpContext &ctx,
                         const std::vector<NDArray> &inputs,
                         const std::vector<OpReqType> &req,
                         const std::vector<NDArray> &outputs) {
    ScatterWrap<cpu>(attrs, ctx, inputs, req,
                     outputs, true, [](const nnvm::NodeAttrs &attrs,
                                        const OpContext &ctx,
                                        const std::vector<NDArray> &inputs,
                                        const std::vector<OpReqType> &req,
                                        const std::vector<NDArray> &outputs) {
        BinaryScalarOp::ComputeEx<cpu, OP>(attrs, ctx, inputs, req, outputs);
    });
  }

#ifdef __CUDACC__
  /*! \brief GPU version, fallback op + retain */
  template<typename OP>
  static void ComputeEx_(mshadow::Stream<gpu> *stream,
                         const nnvm::NodeAttrs &attrs,
                         const OpContext &ctx,
                         const std::vector<NDArray> &inputs,
                         const std::vector<OpReqType> &req,
                         const std::vector<NDArray> &outputs) {
    CHECK_NE(inputs[0].storage_type(), kDefaultStorage);
    if(outputs[0].storage_type() == inputs[0].storage_type()) {
      BinaryScalarOp::ComputeEx<gpu, OP>(attrs, ctx, inputs, req, outputs);
    } else {
      ScatterWrap<cpu>(attrs, ctx, inputs, req,
                       outputs, false, [](const nnvm::NodeAttrs &attrs,
                                          const OpContext &ctx,
                                          const std::vector<NDArray> &inputs,
                                          const std::vector<OpReqType> &req,
                                          const std::vector<NDArray> &outputs) {
          FCompExFallback<gpu>(attrs, ctx, inputs, req, outputs, BinaryScalarOp::Compute<gpu, OP>,
                               "ComputeEx_");
        });
    }
  }
#endif  // __CUDACC__

 public:
  using BinaryScalarOp::Compute;
  template<typename xpu, typename OP>
  static void ComputeEx(const nnvm::NodeAttrs &attrs,
                        const OpContext &ctx,
                        const std::vector<NDArray> &inputs,
                        const std::vector<OpReqType> &req,
                        const std::vector<NDArray> &outputs) {
    DCHECK_EQ(inputs.size(), 1U);
    DCHECK_EQ(outputs.size(), 1U);
    CHECK_NE(inputs[0].storage_type(), kDefaultStorage);
    if (inputs[0].storage_type() == kRowSparseStorage
        && outputs[0].storage_type() == kRowSparseStorage) {
      UnaryOp::MapToFCompute<xpu>(attrs, ctx, inputs, req, outputs, Compute<xpu, OP>);
    } else {
      ComputeEx_<OP>(ctx.get_stream<xpu>(), attrs, ctx, inputs, req, outputs);
    }
  }
};

}  // namespace op
}  // namespace mxnet

#endif // MXNET_OPERATOR_TENSOR_ELEMWISE_SCATTER_OP_H_

