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
 * \file npi_tensordot_inplace_op-inl.h
 * \brief Implementation of numpy-compatible tensordot_inplace
 */

#include "npi_tensordot_op-inl.h"

namespace mxnet {
namespace op {

using namespace mxnet;
using namespace mshadow;

struct TensordotInplaceParam : public dmlc::Parameter<TensordotInplaceParam> {
  int axes; 
  DMLC_DECLARE_PARAMETER(TensordotInplaceParam) {
    DMLC_DECLARE_FIELD(axes);
  }
};

/**
 * gets summed axes of a and b from parameter axes.
 */
inline void getSummedAxes(mxnet::Tuple<int>& a_axes_summed, 
                          mxnet::Tuple<int>& b_axes_summed,
                          const int& axes,
                          const mxnet::TShape& a_shape) {
    std::vector<int> a_axes_summed_vector;
    for (int i = 0; i < axes; i++) {
      a_axes_summed_vector.push_back(a_shape.ndim() - axes + i);
    }
    a_axes_summed = mxnet::Tuple<int>(a_axes_summed_vector);
    
    std::vector<int> b_axes_summed_vector;
    for (int i = 0; i < axes; i++) {
      b_axes_summed_vector.push_back(i);
    }
    b_axes_summed = mxnet::Tuple<int>(b_axes_summed_vector);
  }

/**
 * forward function
 */
template<typename xpu>  // cpu and gpu                                                
void TensordotInplaceOpForward( const nnvm::NodeAttrs& attrs,                     
                                const OpContext& ctx,                                
                                const std::vector<TBlob>& inputs,                 
                                const std::vector<OpReqType>& req, 
                                const std::vector<TBlob>& outputs) {                   

  CHECK_EQ(inputs.size(), 2U);                                                 
  CHECK_EQ(outputs.size(), 1U);                                               
  CHECK_EQ(req.size(), 1U);         
  
  if (req[0] == kNullOp) {
    return;
  }

  const TBlob& a = inputs[0];
  const TBlob& b = inputs[1];
  const TBlob& out = outputs[0];
 
  if (out.shape_.Size() == 0U) {
    return;  // zero-size output, no need to launch kernel
  }

  if ((a.shape_.ndim() < 1) || (b.shape_.ndim() < 1)) {
    return;
  }

  const mxnet::TShape& a_shape = a.shape_;
  const mxnet::TShape& b_shape = b.shape_;

  mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();  
  CHECK_EQ(out.type_flag_, a.type_flag_)
      << "Binary function only support input/output with the same type";
  CHECK_EQ(out.type_flag_, b.type_flag_)
      << "Binary function only support input/output with the same type";
  CHECK(out.type_flag_ == kFloat32 || out.type_flag_ == kFloat64 ||
      (out.type_flag_ == kFloat16 && ctx.run_ctx.ctx.dev_mask() == mshadow::gpu::kDevMask))
      << "Tensordot only supports float32/float64 for CPU, and float16/float32/float64 for GPU";    
    
                                                              
  const TensordotInplaceParam& param = nnvm::get<TensordotInplaceParam>(attrs.parsed);  
  const int& axes = param.axes;    

  Tuple<int> a_axes_summed;
  Tuple<int> b_axes_summed;  
  getSummedAxes(a_axes_summed, b_axes_summed, axes, a_shape);

  if (a_axes_summed.ndim() != b_axes_summed.ndim()) {
    return;
  }

  Tuple<int> a_axes_remained;
  Tuple<int> b_axes_remained;
  Tuple<int> a_axes;
  Tuple<int> b_axes;
  getReorderedAxes(a_axes_summed, a_axes_remained, a_axes, b_axes_summed, b_axes_remained, 
    b_axes, a_shape, b_shape);

  // get output shape
  std::vector<int> out_dim;
  for (int i = 0; i < a_axes_remained.ndim(); i++) {
    out_dim.push_back(a_shape[a_axes_remained[i]]);
  }
  for (int i = 0; i < b_axes_remained.ndim(); i++) {
    out_dim.push_back(b_shape[b_axes_remained[i]]);
  }

  int ad1 = 1, ad2 = 1, bd1 = 1, bd2 = 1;
  getMatrixDimensions(ad1, ad2, bd1, bd2, a_axes_remained, a_axes_summed, 
    b_axes_remained, b_axes_summed, a_shape, b_shape);

  MSHADOW_REAL_TYPE_SWITCH(out.type_flag_, DType, {                           
    if (a_shape.Size() == 0U || b_shape.Size() == 0U) { // 0-size input
      if (req[0] != kAddTo) {
        Tensor<xpu, 1, DType> out_data = out.get_with_shape<xpu, 1, DType>(
            Shape1(out.shape_.Size()), s);
        out_data = static_cast<DType>(0);
      }
      return;
    }  

    matrixDot<xpu>(ctx, a, b, out, req[0], ad1, ad2, bd1, bd2);                         
  });                                                                                                                                               
}     

/**
 * backward function.
 */
template<typename xpu>                                                       
void TensordotInplaceOpBackward(const nnvm::NodeAttrs& attrs,                       
                                const OpContext& ctx,                               
                                const std::vector<TBlob>& inputs,                   
                                const std::vector<OpReqType>& req,                  
                                const std::vector<TBlob>& outputs) { 

  CHECK_EQ(inputs.size(), 3U);                                              
  CHECK_EQ(outputs.size(), 2U);                                             
  CHECK_EQ(req.size(), 2U);   
                       
  const TBlob& out_grad = inputs[0];                                         
  const TBlob& a = inputs[1];     
  const TBlob& b = inputs[2];                                      
  const TBlob& grad_a = outputs[0];
  const TBlob& grad_b = outputs[1];    
  
  const mxnet::TShape& a_shape = a.shape_;
  const mxnet::TShape& b_shape = b.shape_;

  const TensordotInplaceParam& param = nnvm::get<TensordotInplaceParam>(attrs.parsed);  
  const int& axes = param.axes;    

  Tuple<int> a_axes_summed;
  Tuple<int> b_axes_summed;  
  getSummedAxes(a_axes_summed, b_axes_summed, axes, a_shape);

  Tuple<int> a_axes_remained;
  Tuple<int> b_axes_remained;
  Tuple<int> a_axes;
  Tuple<int> b_axes;
  getReorderedAxes(a_axes_summed, a_axes_remained, a_axes, b_axes_summed, b_axes_remained, 
    b_axes, a_shape, b_shape);

  int ad1 = 1, ad2 = 1, bd1 = 1, bd2 = 1;
  getMatrixDimensions(ad1, ad2, bd1, bd2, a_axes_remained, a_axes_summed, 
    b_axes_remained, b_axes_summed, a_shape, b_shape);

  MSHADOW_REAL_TYPE_SWITCH(out_grad.type_flag_, DType, {                          
    matrixDot<xpu>(ctx, a, out_grad, grad_b, req[1], ad1, ad2, ad1, bd2, true, false);
    matrixDot<xpu>(ctx, out_grad, b, grad_a, req[0], ad1, bd2, bd1, bd2, false, true);  
  });                                                                                                                                              
}   
}  // namespace op
}  // namespace mxnet