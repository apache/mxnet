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
 * \file npi_tensordot.cc
 * \brief CPU Implementation of numpy-compatible tensordot
 */


#include "np_matrix_op-inl.h"

namespace mxnet {
namespace op {

using namespace mxnet;
using namespace mshadow;

struct TensordotParam : public dmlc::Parameter<TensordotParam> {
  mxnet::Tuple<int> a_axes, a_axes_remained, a_axes_summed, 
    b_axes, b_axes_remained, b_axes_summed; 
  DMLC_DECLARE_PARAMETER(TensordotParam) {
    DMLC_DECLARE_FIELD(a_axes_summed);
    DMLC_DECLARE_FIELD(b_axes_summed);
  }
};

/**
 * Gets matrix dimensions of a and b after transpose and reshape. 
 */
inline void getMatrixDimensions(int& ad1, 
  int& ad2, 
  int& bd1, 
  int& bd2, 
  const mxnet::Tuple<int>& a_axes_remained, 
  const mxnet::Tuple<int>& a_axes_summed, 
  const mxnet::Tuple<int>& b_axes_remained, 
  const mxnet::Tuple<int>& b_axes_summed, 
  const mxnet::TShape& a_shape, 
  const mxnet::TShape& b_shape) {
  ad1 = 1;
  ad2 = 1;
  bd1 = 1;
  bd2 = 1;

  for (int i = 0; i < a_axes_remained.ndim(); i++) {
    ad1 *= a_shape[a_axes_remained[i]];
  } 
  for (int i = 0; i < a_axes_summed.ndim(); i++) {
    ad2 *= a_shape[a_axes_summed[i]];
  } 
  for (int i = 0; i < b_axes_summed.ndim(); i++) {
    bd1 *= b_shape[b_axes_summed[i]];
  } 
  for (int i = 0; i < b_axes_remained.ndim(); i++) {
    bd2 *= b_shape[b_axes_remained[i]];
  } 
}

/**
 * gets new axes of a and b after transpose and reshape.
 */
inline void getReorderedAxes(const mxnet::Tuple<int>& a_axes_summed, 
  mxnet::Tuple<int>& a_axes_remained, 
  mxnet::Tuple<int>& a_axes, 
  const mxnet::Tuple<int>& b_axes_summed,
  mxnet::Tuple<int>& b_axes_remained, 
  mxnet::Tuple<int>& b_axes, 
  const mxnet::TShape& a_shape, 
  const mxnet::TShape& b_shape) {
    std::vector<int> a_axes_remained_vector;
    for (int i = 0; i < a_shape.ndim(); i++) {
      a_axes_remained_vector.push_back(i);
    }
    for(auto& i: a_axes_summed) {
      a_axes_remained_vector.erase(std::find(a_axes_remained_vector.begin(), 
        a_axes_remained_vector.end(), i));
    }
    a_axes_remained = mxnet::Tuple<int>(a_axes_remained_vector);

    std::vector<int> a_axes_vector(a_axes_remained_vector);
    for(auto& i: a_axes_summed) {
      a_axes_vector.push_back(i);
    }
    a_axes = mxnet::Tuple<int>(a_axes_vector);

    std::vector<int> b_axes_remained_vector;
    for (int i = 0; i < b_shape.ndim(); i++) {
      b_axes_remained_vector.push_back(i);
    }
    for(auto& i: b_axes_summed) {
      b_axes_remained_vector.erase(std::find(b_axes_remained_vector.begin(), 
        b_axes_remained_vector.end(), i));
    }
    b_axes_remained = mxnet::Tuple<int>(b_axes_remained_vector);

    std::vector<int> b_axes_vector;
    for(auto& i: b_axes_summed) {
      b_axes_vector.push_back(i);
    }
    for(auto& i: b_axes_remained_vector) {
      b_axes_vector.push_back(i);
    }
    b_axes = mxnet::Tuple<int>(b_axes_vector);
  }

/**
 * gets shapes of a and b after transpose and reshape.
 */
inline mxnet::TShape getReorderedShape(const mxnet::TShape& shape, const mxnet::Tuple<int>& axes) {
  mxnet::TShape newShape(shape);
  for (int i = 0; i < axes.ndim(); i++) {
    newShape[i] = shape[axes[i]];
  }
  return newShape;
}

/**
 * gets matrix dot.
 */
template<typename xpu>
inline void matrixDot (const OpContext& ctx,
                   const TBlob& a,
                   const TBlob& b,
                   const TBlob& out,
                   const OpReqType req,
                   const int ad1,
                   const int ad2,
                   const int bd2) {
  using namespace mshadow;
  using namespace mshadow_op;
  
  Stream<xpu> *s = ctx.get_stream<xpu>();

  MSHADOW_REAL_TYPE_SWITCH(out.type_flag_, DType, {
    Tensor<xpu, 2, DType> a_tensor = a.get_with_shape<xpu, 2, DType>(Shape2(ad1, ad2), s);
    Tensor<xpu, 2, DType> b_tensor = b.get_with_shape<xpu, 2, DType>(Shape2(ad2, bd2), s);
    Tensor<xpu, 2, DType> out_tensor = out.get_with_shape<xpu, 2, DType>(Shape2(ad1, bd2), s);
    ASSIGN_DISPATCH(out_tensor, req, dot(a_tensor, b_tensor));
  });

}

/**
 * forward function
 */
template<typename xpu>  // cpu and gpu                                                
void TensordotOpForward(const nnvm::NodeAttrs& attrs,                     
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
    
                                                              
  const TensordotParam& param = nnvm::get<TensordotParam>(attrs.parsed);      
  const Tuple<int> a_axes_summed& = param.a_axes_summed;
  const Tuple<int> b_axes_summed& = param.b_axes_summed;  

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

  mxnet::TShape a_temp_shape = getReorderedShape(a_shape, a_axes);
  mxnet::TShape b_temp_shape = getReorderedShape(b_shape, b_axes);

  MSHADOW_REAL_TYPE_SWITCH(out.type_flag_, DType, {                           
    if (a_shape.Size() == 0U || b_shape.Size() == 0U) { // 0-size input
      if (req[0] != kAddTo) {
        Tensor<xpu, 1, DType> out_data = out.get_with_shape<xpu, 1, DType>(
            Shape1(out.shape_.Size()), s);
        out_data = static_cast<DType>(0);
      }
      return;
    }  

    Tensor<xpu, 1, DType> workspace = ctx.requested[0].get_space_typed<xpu, 1, DType>
      (Shape1(a.Size() + b.Size()), s); 
    DType* a_ptr = reinterpret_cast<DType*>(workspace.dptr_);
    DType* b_ptr = reinterpret_cast<DType*>(workspace.dptr_ + a.Size());
    TBlob a_res = TBlob(a_ptr, a_temp_shape, xpu::kDevMask); 
    TBlob b_res = TBlob(b_ptr, b_temp_shape, xpu::kDevMask); 
  
    mxnet::op::TransposeImpl<xpu>(ctx.run_ctx, a, a_res, mxnet::TShape(a_axes.begin(), a_axes.end()));     
    mxnet::op::TransposeImpl<xpu>(ctx.run_ctx, b, b_res, mxnet::TShape(b_axes.begin(), b_axes.end())); 

    matrixDot<xpu>(ctx, a_res, b_res, out, req[0], ad1, ad2, bd2);                         
  });                                                                                                                                               
}     

/**
 * gets shapes for inverse transpose.
 */
inline mxnet::TShape getReverseShape(const mxnet::Tuple<int>& shape) {
  mxnet::TShape shape2(shape.begin(), shape.end());
  for (int i = 0; i < shape.ndim(); i++) {
    shape2[shape[i]] = i;
  }
  return shape2;
}

/**
 * backward function.
 */
template<typename xpu>                                                       
void TensordotOpBackward(const nnvm::NodeAttrs& attrs,                       
                         const OpContext& ctx,                               
                         const std::vector<TBlob>& inputs,                   
                         const std::vector<OpReqType>& req,                  
                         const std::vector<TBlob>& outputs) { 

  CHECK_EQ(inputs.size(), 3U);                                              
  CHECK_EQ(outputs.size(), 2U);                                             
  CHECK_EQ(req.size(), 2U);   

  mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();                           
  const TBlob& out_grad = inputs[0];                                         
  const TBlob& a = inputs[1];     
  const TBlob& b = inputs[2];                                      
  const TBlob& grad_a = outputs[0];
  const TBlob& grad_b = outputs[1];    
  
  const mxnet::TShape& a_shape = a.shape_;
  const mxnet::TShape& b_shape = b.shape_;

  const TensordotParam& param = nnvm::get<TensordotParam>(attrs.parsed);      
  const Tuple<int> a_axes_summed& = param.a_axes_summed;
  const Tuple<int> b_axes_summed& = param.b_axes_summed;  

  Tuple<int> a_axes_remained;
  Tuple<int> b_axes_remained;
  Tuple<int> a_axes;
  Tuple<int> b_axes;
  getReorderedAxes(a_axes_summed, a_axes_remained, a_axes, b_axes_summed, b_axes_remained, 
    b_axes, a_shape, b_shape);

  int ad1 = 1, ad2 = 1, bd1 = 1, bd2 = 1;
  getMatrixDimensions(ad1, ad2, bd1, bd2, a_axes_remained, a_axes_summed, 
    b_axes_remained, b_axes_summed, a_shape, b_shape);

  std::vector<int> a_T_axes;
  for (int i = 0; i < a_axes_summed.ndim(); i++) {
    a_T_axes.push_back(a_axes_summed[i]);
  }
  for (int i = 0; i < a_axes_remained.ndim(); i++) {
    a_T_axes.push_back(a_axes_remained[i]);
  }
  mxnet::TShape a_temp_shape(getReorderedShape(a_shape, a_T_axes));

  std::vector<int> b_T_axes;
  for (int i = 0; i < b_axes_remained.ndim(); i++) {
    b_T_axes.push_back(b_axes_remained[i]);
  }
  for (int i = 0; i < b_axes_summed.ndim(); i++) {
    b_T_axes.push_back(b_axes_summed[i]);
  }
  mxnet::TShape b_temp_shape(getReorderedShape(b_shape, b_T_axes));

  MSHADOW_REAL_TYPE_SWITCH(out_grad.type_flag_, DType, {                          
    Tensor<xpu, 1, DType> workspace = ctx.requested[0].get_space_typed<xpu, 1, DType>
      (Shape1(a.Size() + b.Size()), s);
    DType* a_ptr = reinterpret_cast<DType*>(workspace.dptr_);
    DType* b_ptr = reinterpret_cast<DType*>(workspace.dptr_ + a.Size());
    TBlob a_res = TBlob(a_ptr, a_temp_shape, xpu::kDevMask);
    TBlob b_res = TBlob(b_ptr, b_temp_shape, xpu::kDevMask);

    mxnet::op::TransposeImpl<xpu>(ctx.run_ctx, a, grad_a, mxnet::TShape(a_T_axes.begin(), a_T_axes.end()));
    mxnet::op::TransposeImpl<xpu>(ctx.run_ctx, b, grad_b, mxnet::TShape(b_T_axes.begin(), b_T_axes.end())); 

    matrixDot<xpu>(ctx, grad_a, out_grad, b_res, req[1], ad2, ad1, bd2);
    matrixDot<xpu>(ctx, out_grad, grad_b, a_res, req[0], ad1, bd2, bd1);  

    mxnet::op::TransposeImpl<xpu>(ctx.run_ctx, a_res, grad_a, getReverseShape(a_axes)); 
    mxnet::op::TransposeImpl<xpu>(ctx.run_ctx, b_res, grad_b, getReverseShape(b_axes)); 
  });                                                                                                                                              
}   
}  // namespace op
}  // namespace mxnet