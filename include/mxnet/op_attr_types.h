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
#include "./ndarray.h"

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
/*!
 * \brief Register an operator called as a NDArray function
 *
 * \note Register under "FNDArrayFunction"
 */
using FNDArrayFunction = std::function<void (const nnvm::NodeAttrs& attrs,
                                             const std::vector<NDArray>& inputs,
                                             std::vector<NDArray>* outputs)>;
/*!
 * \brief Resiger a compute function for simple stateless forward only operator
 *
 * \note Register under "FCompute<cpu>" and "FCompute<gpu>"
 */
using FCompute = std::function<void (const nnvm::NodeAttrs& attrs,
                                     const OpContext& ctx,
                                     const std::vector<TBlob>& inputs,
                                     const std::vector<OpReqType>& req,
                                     const std::vector<TBlob>& outputs)>;
}  // namespace mxnet

#endif  // MXNET_OP_ATTR_TYPES_H_
