/*!
 *  Copyright (c) 2016 by Contributors
 * \file op_attr_types.h
 * \brief Additional operator attributes
 *  beside the ones provided by NNVM
 */
#ifndef MXNET_OP_ATTR_TYPES_H_
#define MXNET_OP_ATTR_TYPES_H_


#include <mshadow/tensor.h>
#include <nnvm/op_attr_types.h>

#include <vector>
#include <functional>

#include "./base.h"
#include "./operator.h"

namespace mxnet {

using nnvm::NodeAttrs;
/*!
 * \brief Create a Layer style, forward/backward operator.
 *  This is easy to write code that contains state.
 *
 *  This is not the only way to register an op execution function.
 *  More simpler or specialized operator form can be registered
 *
 *  \note Register under "FCreateLayerOp"
 */
using FCreateLayerOp = std::function<
  Operator* (const NodeAttrs& n,
             Context ctx,
             const std::vector<TShape>& in_shape,
             const std::vector<int>& in_type)>;

/*!
 * \brief The resource request from the operator
 *
 * \note Register under "FResourceRequest"
 */
using FResourceRequest = std::function<
  std::vector<ResourceRequest> (const NodeAttrs& n)>;

}  // namespace mxnet

#endif  // MXNET_OP_ATTR_TYPES_H_
