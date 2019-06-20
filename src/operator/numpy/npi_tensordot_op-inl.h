#include "np_matrix_op-inl.h"
#include "np_dot-inl.h"

struct TensordotParam : public dmlc::Parameter<TensordotParam> {
  dmlc::optional<mxnet::Tuple<int>> a_axes, axes_remained, a_axes_summed, 
  b_axes, b_axes_remained, b_axes_summed; 
  DMLC_DECLARE_PARAMETER(QTensordotParam) {
    DMLC_DECLARE_FIELD(a_axes);
    DMLC_DECLARE_FIELD(a_axes_remained);
    DMLC_DECLARE_FIELD(a_axes_summed);
    DMLC_DECLARE_FIELD(b_axes);
    DMLC_DECLARE_FIELD(b_axes_remained);
    DMLC_DECLARE_FIELD(b_axes_summed);
  }
};

void getMatrixDimensions(int& ad1, int& ad2, int& bd1, int& bd2, mxnet::Tuple<int>>& a_axes_remained, 
mxnet::Tuple<int>>& a_axes_summed, mxnet::Tuple<int>>& b_axes_remained, 
mxnet::Tuple<int>>& b_axes_summed, mxnet::TShape a_shape, mxnet::TShape b_shape) {
  ad1 = 1;
  ad2 = 1;
  bd1 = 1;
  bd2 = 1;

  for (int i: a_axes_remained) {
    ad1 *= a_shape[i];
  } 
  for (int i: a_axes_summed) {
    ad2 *= a_shape[i];
  } 
  for (int i: b_axes_summed) {
    bd1 *= b_shape[i];
  } 
  for (int i: b_axes_remained) {
    bd2 *= b_shape[i];
  } 
}

// forward function. gets f(x) by multi-thread.
template<typename xpu>  // cpu and gpu                                                
void TensordotOpForward(const nnvm::NodeAttrs& attrs, // a,b,c                    
                        const OpContext& ctx,  // calculation orders in GPU                               
                        const std::vector<TBlob>& inputs, // x                    
                        const std::vector<OpReqType>& req, // write, add, or null. How do new values deal with old values. Mostly write, sometimes add and null for backward function.                   
                        const std::vector<TBlob>& outputs) { // y                  
  using namespace mshadow;
  using namespace mxnet_op;
    
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
  const mxnet::TShape a_shape = a.shape_;
  const mxnet::TShape b_shape = b.shape_;
  if ((a_shape.ndim() < 1) || (b_shape.ndim() < 1)) {
    return;
  }
    
  mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();  
  CHECK_EQ(out.type_flag_, a.type_flag_)
      << "Binary function only support input/output with the same type";
  CHECK_EQ(out.type_flag_, b.type_flag_)
      << "Binary function only support input/output with the same type";
  CHECK(out.type_flag_ == kFloat32 || out.type_flag_ == kFloat64 ||
      (out.type_flag_ == kFloat16 && ctx.run_ctx.ctx.dev_mask() == mshadow::gpu::kDevMask))
      << "dot only supports float32/float64 for CPU, and float16/float32/float64 for GPU";    
    
                                                              
  const TensordotParam& param = nnvm::get<TensordotParam>(attrs.parsed);      
  
  vector<int> out_dim;
  for (int i: param.a_axes_remained) {
      out_dim.push_back(a_shape[i]);
  }
  for (int i: param.b_axes_remained) {
      out_dim.push_back(b_shape[i]);
  }

  int ad1 = 1, ad2 = 1, bd1 = 1, bd2 = 1;
  getMatrixDimensions(ad1, ad2, bd1, bd2, param.a_axes_remained, param.a_axes_summed, 
    param.b_axes_remained, param.b_axes_summed, a_shape, b_shape);

  MSHADOW_REAL_TYPE_SWITCH(out.type_flag_, DType, {                           
    MXNET_ASSIGN_REQ_SWITCH(req[0], req_type, { 
      if (a_shape.Size() == 0U || b_shape.Size() == 0U) { // 0-size input
        if (req[0] != kAddTo) {
          Tensor<xpu, 1, DType> out_data = out.get_with_shape<xpu, 1, DType>(
              Shape1(out.shape_.Size()), s);
          out_data = static_cast<DType>(0);
        }
        return;
      }  
      
      // 1, tshape; 2, when to request; 3, shape infer 2 ways
      Tensor<xpu, 1, DType> workspace = ctx.requested[0].get_space_typed<xpu, 1, DType>
        (Shape1(a.Size() + b.Size()), s);
      TBlob* a_ptr = reinterpret_cast<DType*>(workspace.dptr_);
      TBlob* b_ptr = reinterpret_cast<DType*>(workspace.dptr_ + a.size * sizeof(DType));

      TransposeImpl<xpu>(ctx.run_ctx, a, a_ptr, param.a_axes); 
      TBlob* a_res = a_ptr;
      a_res->reshape(mxnet::TShape(ad1, ad2));
      
      TransposeImpl<xpu>(ctx.run_ctx, b, b_ptr, param.b_axes); 
      TBlob* b_res = b_ptr;
      b_res->reshape(mxnet::TShape(bd1, bd2));

      MMImpl<xpu>(ctx, a_res, b_res, out, req[0]); 
      out.reshape(mxnet::TShape(out_dim));                                      
    });                                                                       
  });                                                                         
}     

mxnet::TShape getReverseShape(mxnet::Tuple<int> shape) {
  mxnet::TShape shape2(shape);
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
  using namespace mshadow;
  using namespace mshadow_op;

  CHECK_EQ(inputs.size(), 3U); // input = [dy/dx, y]. check # of inputs                                              
  CHECK_EQ(outputs.size(), 2U); // check # of outputs                                             
  CHECK_EQ(req.size(), 2U);   // same as output                                               
  mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();                           
  const TBlob& out_grad = inputs[0];                                         
  const TBlob& a = inputs[1];     
  const TBlob& b = inputs[2];                                      
  const TBlob& grad_a = outputs[0];
  const TBlob& grad_b = outputs[1];    
  const mxnet::TShape a_shape = a.shape_;
  const mxnet::TShape b_shape = b.shape_;

  const TensordotParam& param = nnvm::get<TensordotParam>(attrs.parsed);  
  int ad1 = 1, ad2 = 1, bd1 = 1, bd2 = 1;
  getMatrixDimensions(ad1, ad2, bd1, bd2, param.a_axes_remained, param.a_axes_summed, 
    param.b_axes_remained, param.b_axes_summed, a_shape, b_shape);

  using namespace mxnet_op;                                                  
  MSHADOW_REAL_TYPE_SWITCH(out_grad.type_flag_, DType, {                          
    MXNET_ASSIGN_REQ_SWITCH(req[0], req_type, { 
      Tensor<xpu, 1, DType> workspace = ctx.requested[1].get_space_typed<xpu, 1, DType>
        (Shape1(a.Size() + b.Size()), s);
      TBlob* a_ptr = reinterpret_cast<DType*>(workspace.dptr_);
      TBlob* b_ptr = reinterpret_cast<DType*>(workspace.dptr_ + a.size * sizeof(DType));

      TransposeImpl<xpu>(ctx.run_ctx, a, a_ptr, param.a_axes); 
      TBlob* a_res = a_ptr;
      a_res->reshape(mxnet::TShape(ad1, ad2));
      
      TransposeImpl<xpu>(ctx.run_ctx, b, b_ptr, param.b_axes); 
      TBlob* b_res = b_ptr;
      b_res->reshape(mxnet::TShape(bd1, bd2));

      out_grad->reshape(mxnet::TShape(ad1, bd2));

      MMImpl<xpu>(ctx, a, out_grad, grad_b, req[1], true, false);
      MMImpl<xpu>(ctx, out_grad, b, grad_a, req[0], false, true);  

      grad_a->reshape(mxnet::TShape(param.a_axes)); 
      grad_b->reshape(mxnet::TShape(param.b_axes)); 
      TransposeImpl<xpu>(ctx.run_ctx, grad_a, grad_a, getReverseShape(param.a_axes)); 
      TransposeImpl<xpu>(ctx.run_ctx, grad_b, grad_b, getReverseShape(param.b_axes)); 
    });                                                                      
  });                                                                        
}   