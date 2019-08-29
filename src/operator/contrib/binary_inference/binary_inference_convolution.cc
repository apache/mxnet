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
 * Copyright (c) 2018 by Contributors
 * \file binary_inference_convolution-inl.h
 * \brief
 * \ref: https://arxiv.org/abs/1705.09864
 * \author HPI-DeepLearning
*/

#include "./binary_inference_convolution-inl.h"

using ns = std::chrono::nanoseconds;
using get_time = std::chrono::steady_clock ;

namespace mshadow {
    using namespace mxnet::op::xnor;
    /*
     * m: number of output channels (num_filter) per group
     * n: number of pixels of output images per channel (output dimension)
     * k: number of input channels per group * kernel size
     */
    inline void _BinaryConvolutionForward(int m, int n, int k,
                   BINARY_WORD* binary_weights_row,
                   Tensor<cpu, 1, float> &workspace,
                   const Tensor<cpu, 2, float> &in_col,
                   Tensor<cpu, 2, float> &temp_dst) {
        CHECK_EQ(workspace.shape_.Size() * sizeof(workspace[0]) * CHAR_BIT, n * k);
        BINARY_WORD* binary_col = (BINARY_WORD*) workspace.dptr_;

        get_binary_col_unrolled(in_col.dptr_, binary_col, k, n);
        
        temp_dst = 0;
        
        //auto start = std::chrono::high_resolution_clock::now();

        xnor_gemm(m, n, k/BITS_PER_BINARY_WORD,
                    binary_weights_row, k/BITS_PER_BINARY_WORD,
                    binary_col, n,
                    temp_dst.dptr_, n);

        // auto finish = std::chrono::high_resolution_clock::now();
        // std::chrono::duration<double> elapsed = finish - start;
        // std::cout << "xnor elapsed time: " << elapsed.count() << " s\n";       
    }


    inline void BinaryConvolutionForward(int m, int n, int k,
                  BINARY_WORD* wmat_binarized,
                  Tensor<cpu, 1, float> &workspace,
                  const Tensor<cpu, 2, float> &in_col,
                  Tensor<cpu, 2, float> &temp_dst) {

        _BinaryConvolutionForward(m, n, k, wmat_binarized, workspace, in_col, temp_dst);
    }

  inline void BinaryConvolutionForward(int m, int n, int k,
                  const Tensor<cpu, 2, float> &wmat,
                  Tensor<cpu, 1, float> &workspace,
                  const Tensor<cpu, 2, float> &in_col,
                  Tensor<cpu, 2, float> &temp_dst) {
        BINARY_WORD binary_row[m * k/BITS_PER_BINARY_WORD];
        get_binary_row(wmat.dptr_, &binary_row[0], m*k);
        _BinaryConvolutionForward(m, n, k, binary_row, workspace, in_col, temp_dst);
  }



    template<typename DType>
    inline void BinaryConvolutionForward(int m, int n, int k,
                                    const Tensor<cpu, 2, DType> &wmat,
                  Tensor<cpu, 1, DType> &workspace,
                                    const Tensor<cpu, 2, DType> &in_col,
                                    Tensor<cpu, 2, DType> &temp_dst) {
      CHECK(false) << "only float supported";
    }

    template<typename DType>
    inline void BinaryConvolutionForward(int m, int n, int k,
                  BINARY_WORD* wmat_binarized,
                  Tensor<cpu, 1, DType> &workspace,
                                    const Tensor<cpu, 2, DType> &in_col,
                                    Tensor<cpu, 2, DType> &temp_dst) {
      CHECK(false) << "only float supported";
    }
}

namespace mxnet {
namespace op {


DMLC_REGISTER_PARAMETER(BinaryInferenceConvolutionParam);

template<>
Operator* CreateOp<cpu>(BinaryInferenceConvolutionParam param, int dtype,
                        mxnet::ShapeVector *in_shape,
                        mxnet::ShapeVector *out_shape,
                        Context ctx) {
  Operator *op = NULL;
  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
    op = new BinaryInferenceConvolutionOp<cpu, DType>(param);
  })
  return op;
}

// DO_BIND_DISPATCH comes from operator_common.h
Operator *BinaryInferenceConvolutionProp::CreateOperatorEx(Context ctx,
                                            mxnet::ShapeVector *in_shape,
                                            std::vector<int> *in_type) const {
  mxnet::ShapeVector out_shape, aux_shape;
  std::vector<int> out_type, aux_type;
  CHECK(InferType(in_type, &out_type, &aux_type));
  CHECK(InferShape(in_shape, &out_shape, &aux_shape));
  DO_BIND_DISPATCH(CreateOp, param_, (*in_type)[0], in_shape, &out_shape, ctx);
}

MXNET_REGISTER_OP_PROPERTY(BinaryInferenceConvolution, BinaryInferenceConvolutionProp)
.add_argument("data", "NDArray-or-Symbol", "Input data to the BinaryInferenceConvolutionOp.")
.add_argument("weight", "NDArray-or-Symbol", "Weight matrix.")
.add_argument("bias", "NDArray-or-Symbol", "Bias parameter.")
.add_arguments(BinaryInferenceConvolutionParam::__FIELDS__());

}  // namespace op
}  // namespace mxnet
