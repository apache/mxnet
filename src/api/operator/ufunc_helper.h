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
 * \file ufunc_helper.h
 * \brief ufunc helper
 */
#ifndef MXNET_API_OPERATOR_UFUNC_HELPER_H_
#define MXNET_API_OPERATOR_UFUNC_HELPER_H_
#include <mxnet/runtime/packed_func.h>
namespace mxnet {

/*
 * Ufunc helper for unary operators
 */
void UFuncHelper(runtime::MXNetArgs args,
                 runtime::MXNetRetValue* ret,
                 const nnvm::Op* fn_array);

/*
 * Ufunc helper for binary operators
 */
void UFuncHelper(runtime::MXNetArgs args,
                 runtime::MXNetRetValue* ret,
                 const nnvm::Op* fn_array,
                 const nnvm::Op* lfn_scalar,
                 const nnvm::Op* rfn_scalar);

}  // namespace mxnet

#endif  // MXNET_API_OPERATOR_UFUNC_HELPER_H_
