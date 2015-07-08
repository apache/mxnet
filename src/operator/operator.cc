/*!
 *  Copyright (c) 2015 by Contributors
 * \file operator.cc
 * \brief
 * \author: Bing Xu
 */
#include <dmlc/logging.h>
#include <mxnet/base.h>
#include <mxnet/operator.h>
#include <cstring>
#include "./operator_common.h"

namespace mxnet {
namespace op {
// declare the operator
template<typename xpu>
Operator *CreateOperator(OpType type);


OpType GetOpType(const char *type) {
  if (!strcmp(type, "relu")) return kReLU;
  if (!strcmp(type, "fullc")) return kFullc;
  LOG(FATAL) << "unknown operator type " << type;
  return kReLU;
}
}  // namespace op

// implementing the context
Operator *Operator::Create(const char *type,
                           Context ctx) {
  op::OpType otype = op::GetOpType(type);
  if (ctx.dev_mask == cpu::kDevMask) {
    return op::CreateOperator<cpu>(otype);
  }
  if (ctx.dev_mask == gpu::kDevMask) {
#if MXNET_USE_CUDA
    return op::CreateOperator<gpu>(otype);
#else
    LOG(FATAL) << "GPU is not enabled";
#endif
  }
  return NULL;
}  // namespace op
}  // namespace mxnet
