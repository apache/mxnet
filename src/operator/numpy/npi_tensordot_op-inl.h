#include "np_matrix_op-inl.h"
#include "np_dot-inl.h"

namespace mxnet {
namespace op {

using namespace mxnet;
using namespace mshadow;

struct TensordotParam : public dmlc::Parameter<TensordotParam> {
  mxnet::Tuple<int> a_axes, a_axes_remained, a_axes_summed, 
    b_axes, b_axes_remained, b_axes_summed; 
  DMLC_DECLARE_PARAMETER(TensordotParam) {
    DMLC_DECLARE_FIELD(a_axes);
    DMLC_DECLARE_FIELD(a_axes_remained);
    DMLC_DECLARE_FIELD(a_axes_summed);
    DMLC_DECLARE_FIELD(b_axes);
    DMLC_DECLARE_FIELD(b_axes_remained);
    DMLC_DECLARE_FIELD(b_axes_summed);
  }
};

inline void getMatrixDimensions(int& ad1, int& ad2, int& bd1, int& bd2, const mxnet::Tuple<int>& a_axes_remained, 
  const mxnet::Tuple<int>& a_axes_summed, const mxnet::Tuple<int>& b_axes_remained, 
  const mxnet::Tuple<int>& b_axes_summed, const mxnet::TShape& a_shape, const mxnet::TShape& b_shape) {
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

inline mxnet::TShape getReorderedShape(const mxnet::TShape& shape, const mxnet::Tuple<int>& axes) {
  mxnet::TShape newShape(shape);
  for (int i = 0; i < axes.ndim(); i++) {
    newShape[i] = shape[axes[i]];
  }
  return newShape;
}

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
// forward function. gets f(x) by multi-thread.
template<typename xpu>  // cpu and gpu                                                
void TensordotOpForward(const nnvm::NodeAttrs& attrs, // a,b,c                    
                        const OpContext& ctx,  // calculation orders in GPU                               
                        const std::vector<TBlob>& inputs, // x                    
                        const std::vector<OpReqType>& req, // write, add, or null. How do new values deal with old values. Mostly write, sometimes add and null for backward function.                   
                        const std::vector<TBlob>& outputs) { // y                  

  CHECK_EQ(inputs.size(), 2U);  // assert # of input = 1 (x)                                               
  CHECK_EQ(outputs.size(), 1U); // assert # of output = 1 (y)                                              
  CHECK_EQ(req.size(), 1U);     // same as # of output     
    
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
  const Tuple<int>& a_axes_summed = param.a_axes_summed;
  const Tuple<int>& a_axes_remained = param.a_axes_remained;
  const Tuple<int>& b_axes_summed = param.b_axes_summed;
  const Tuple<int>& b_axes_remained = param.b_axes_remained;
  const Tuple<int>& a_axes = param.a_axes;
  const Tuple<int>& b_axes = param.b_axes;

  if (a_axes_summed.ndim() != b_axes_summed.ndim()) {
    return;
  }

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
      (Shape1(a.Size() + b.Size()), s); // TODO
    DType* a_ptr = reinterpret_cast<DType*>(workspace.dptr_);
    DType* b_ptr = reinterpret_cast<DType*>(workspace.dptr_ + a.Size() * sizeof(DType));
    TBlob a_res = TBlob(a_ptr, a_temp_shape, xpu::kDevMask); 
    TBlob b_res = TBlob(b_ptr, b_temp_shape, xpu::kDevMask); 
  
    std::cout << a.shape_ << std::endl; //TODO
    std::cout << a_res.shape_ << std::endl; //TODO
    mxnet::op::TransposeImpl<xpu>(ctx.run_ctx, a, a_res, mxnet::TShape(a_axes.begin(), a_axes.end()));     
//    a_res.shape_ = mxnet::TShape({ad1, ad2});
    mxnet::op::TransposeImpl<xpu>(ctx.run_ctx, b, b_res, mxnet::TShape(b_axes.begin(), b_axes.end())); 
//    b_res.shape_ = mxnet::TShape({bd1, bd2});
    std::cout << "\n\n" << ad1 << ' ' << bd2 << ' ' << a_res.shape_ << ' ' << b_res.shape_ << ' ' 
      << "000000000000000000000000000000000000\n\n" << std::endl; //TODO
    matrixDot<xpu>(ctx, a_res, b_res, out, req[0], ad1, ad2, bd2); 
    std::cout << "\n\n11111111111111111111111111111111111\n\n" << std::endl; //TODO
    std::cout << "\n\n2222222222222222222222222222222222222\n\n" << std::endl; //TODO                         
  });                                                                                                                                               
}     

inline mxnet::TShape getReverseShape(const mxnet::Tuple<int>& shape) {
  mxnet::TShape shape2(shape.begin(), shape.end());
  for (int i = 0; i < shape.ndim(); i++) {
    shape2[shape[i]] = i;
  }
  return shape2;
}


// backward function. gets dL/dx = dL/dy * dy/dx.
template<typename xpu>                                                       
void TensordotOpBackward(const nnvm::NodeAttrs& attrs,                       
                         const OpContext& ctx,                               
                         const std::vector<TBlob>& inputs,                   
                         const std::vector<OpReqType>& req,                  
                         const std::vector<TBlob>& outputs) { 

  CHECK_EQ(inputs.size(), 3U); // input = [dy/dx, y]. check # of inputs                                              
  CHECK_EQ(outputs.size(), 2U); // check # of outputs                                             
  CHECK_EQ(req.size(), 2U);   // same as output                                               
  mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();                           
  const TBlob& out_grad = inputs[0];                                         
  const TBlob& a = inputs[1];     
  const TBlob& b = inputs[2];                                      
  const TBlob& grad_a = outputs[0];
  const TBlob& grad_b = outputs[1];    
  
  const mxnet::TShape a_shape = a.shape_.ndim() > 1 ? a.shape_: mxnet::TShape({1, a.shape_[0]});
  const mxnet::TShape b_shape = b.shape_.ndim() > 1 ? b.shape_: mxnet::TShape({b.shape_[0], 1});
  const mxnet::TShape out_grad_shape = out_grad.shape_;

  const TensordotParam& param = nnvm::get<TensordotParam>(attrs.parsed);  
  const Tuple<int>& a_axes_summed = param.a_axes_summed;
  const Tuple<int>& a_axes_remained = param.a_axes_remained;
  const Tuple<int>& b_axes_summed = param.b_axes_summed;
  const Tuple<int>& b_axes_remained = param.b_axes_remained;
  const Tuple<int>& a_axes = param.a_axes;
  const Tuple<int>& b_axes = param.b_axes;

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
    DType* b_ptr = reinterpret_cast<DType*>(workspace.dptr_ + a.Size() * sizeof(DType));
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