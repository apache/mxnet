#ifndef MXNET_OPERATOR_TENSOR_MIN_EX_OP_INL_H_
#define MXNET_OPERATOR_TENSOR_MIN_EX_OP_INL_H_

#include <dmlc/parameter.h>
#include <vector>
#include <algorithm>
#include "operator/mxnet_op.h"
#include "operator/operator_common.h"
#include "operator/elemwise_op_common.h"

namespace mxnet {
namespace op {

template<typename xpu>
void MinExForward(const nnvm::NodeAttrs& attrs,
                  const OpContext& ctx,
                  const std::vector<TBlob>& inputs,
                  const std::vector<OpReqType>& req,
                  const std::vector<TBlob>& outputs) {
  //do nothing                                                                                                                                                                         
}


inline bool MinExOpShape(const nnvm::NodeAttrs& attrs,
                         mxnet::ShapeVector* in_attrs,
                         mxnet::ShapeVector* out_attrs) {
    //do nothing                                                                                                                                                                       
    return true;
}

inline bool MinExOpType(const nnvm::NodeAttrs& attrs,
                        std::vector<int> *in_attrs,
                        std::vector<int> *out_attrs) {
  //do nothing                                                                                                                                                                         
  return true;
}

}  // namespace op                                                                                                                                                                     
}  // namespace mxnet                                                                                                                                                                  

#endif  // MXNET_OPERATOR_TENSOR_MIN_EX_OP_INL_H_
