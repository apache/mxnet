#ifndef MXNET_OPERATOR_NUMPY_NP_ARCTAN2_OP_H
#define MXNET_OPERATOR_NUMPY_NP_ARCTAN2_OP_H

#include<cmath>
#include <mxnet/operator_util.h>
#include "../mshadow_op.h"
#include "../mxnet_op.h"
#include "../operator_common.h"
#include "../elemwise_op_common.h"
#include "../tensor/init_op.h"
#include "../tensor/elemwise_binary_broadcast_op.h"

namespace mxnet{
namespace op{


inline bool Arctan2OpType(const nnvm::NodeAttrs& attrs,
                          std::vector<int>* in_attrs,
                          std::vector<int>* out_attrs){
  CHECK_EQ(in_attrs->size(), 2U);
  CHECK_EQ(out_attrs->size(), 1U);

  TYPE_ASSIGN_CHECK(*in_attrs, 0, in_attrs->at(1));
  TYPE_ASSIGN_CHECK(*in_attrs, 1, in_attrs->at(0));
  //check if it is float16, float32 or float64
  if(in_attrs->at(0) >=0 && in_attrs->at(0) <= 2){
    TYPE_ASSIGN_CHECK(*out_attrs, 0, in_attrs->at(0));
  }else{
    //assign double to it
    TYPE_ASSIGN_CHECK(*out_attrs, 0, 1);
  }
  return out_attrs->at(0) != -1;
}

template<int req>
struct arctan2_forward {
  template<typename OType, typename IType>
  MSHADOW_XINLINE static void Map(int i, OType* out_data, const IType* in_data_1, const IType* in_data_2){
   KERNEL_ASSIGN(out_data[i], req, atan2(in_data_1[i], in_data_2[i]));
  }
};

template<typename xpu>
void Arctan2OpForward(const nnvm::NodeAttrs& attrs,
                      const OpContext& ctx,
                      const std::vector<TBlob>& inputs,
                      const std::vector<OpReqType>& req,
                      const std::vector<TBlob>& outputs){
  CHECK_EQ(inputs.size(), 2U);
  CHECK_EQ(outputs.size(), 1U);
  CHECK_EQ(req.size(), 1U);
  mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();
  const TBlob& in_data_1 = inputs[0];
  const TBlob& in_data_2 = inputs[1];
  const TBlob& out_data = outputs[0];
  using namespace mxnet_op;
   MSHADOW_TYPE_SWITCH(out_data.type_flag_, OType, {
     MSHADOW_TYPE_SWITCH(in_data_1.type_flag_, IType, {
       MXNET_ASSIGN_REQ_SWITCH(req[0], req_type, {
         Kernel<arctan2_forward<req_type>, xpu>::Launch(
             s, out_data.Size(), out_data.dptr<OType>(), in_data_1.dptr<IType>(),
             in_data_2.dptr<IType>());
       });
     });
   });
}

template<int req>
struct arctan2_backward{
  template<typename IType, typename OType>
  MSHADOW_XINLINE static void Map(int i, IType* in_grad_1, IType* in_grad_2, const OType* out_grad,
                                  IType* in_data_1, IType* in_data_2){
    KERNEL_ASSIGN(in_grad_1[i], req, out_grad[i] * in_data_2[i] / (in_data_1[i] * in_data_1[i] + in_data_2[i] * in_data_2[i]));
    KERNEL_ASSIGN(in_grad_2[i], req, out_grad[i] * (-1) * in_data_1[i] / (in_data_1[i] * in_data_1[i] + in_data_2[i] * in_data_2[i]));
  }
};

template<typename xpu>
void Arctan2OpBackward(const nnvm::NodeAttrs& attrs,
                       const OpContext& ctx,
                       const std::vector<TBlob>& inputs,
                       const std::vector<OpReqType>& req,
                       const std::vector<TBlob>& outputs){
  CHECK_EQ(inputs.size(), 3U);
  CHECK_EQ(outputs.size(), 2U);
  CHECK_EQ(req.size(), 2U);
  mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();
  const TBlob& out_grad = inputs[0];
  const TBlob& in_data_1 = inputs[1];
  const TBlob& in_data_2 = inputs[2];
  const TBlob& in_grad_1 = outputs[0];
  const TBlob& in_grad_2 = outputs[1];
  using namespace mxnet_op;
   MSHADOW_TYPE_SWITCH(out_grad.type_flag_, OType, {
     MSHADOW_TYPE_SWITCH(in_grad_1.type_flag_, IType, {
       MXNET_ASSIGN_REQ_SWITCH(req[0], req_type, {
         Kernel<arctan2_backward<req_type>, xpu>::Launch(
             s, in_grad_1.Size(), in_grad_1.dptr<IType>(), in_grad_2.dptr<IType>(), out_grad.dptr<OType>(),
             in_data_1.dptr<IType>(), in_data_2.dptr<IType>());
       });
     });
   });
}

}//namespace op
}//namespace mxnet

#endif
