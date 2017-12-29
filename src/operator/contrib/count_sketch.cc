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
 * Copyright (c) 2015 by Contributors
 * \file count_sketch.cc
 * \brief count_sketch op
 * \author Chen Zhu
*/
#include "./count_sketch-inl.h"
namespace mxnet {
namespace op {

template<>
Operator *CreateOp<cpu>(CountSketchParam param, int dtype) {
    LOG(FATAL) << "CountSketch is only available for GPU.";
    return NULL;
}
Operator *CountSketchProp::CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
                                            std::vector<int> *in_type) const {
    std::vector<TShape> out_shape, aux_shape;
    std::vector<int> out_type, aux_type;
    CHECK(InferType(in_type, &out_type, &aux_type));
    CHECK(InferShape(in_shape, &out_shape, &aux_shape));
    DO_BIND_DISPATCH(CreateOp, param_, (*in_type)[0]);
}

DMLC_REGISTER_PARAMETER(CountSketchParam);
MXNET_REGISTER_OP_PROPERTY(_contrib_count_sketch, CountSketchProp)
.describe(R"code(Apply CountSketch to input: map a d-dimension data to k-dimension data"

.. note:: `count_sketch` is only available on GPU.

Assume input data has shape (N, d), sign hash table s has shape (N, d),
index hash table h has shape (N, d) and mapping dimension out_dim = k,
each element in s is either +1 or -1, each element in h is random integer from 0 to k-1.
Then the operator computs:

.. math::
   out[h[i]] += data[i] * s[i]

Example::

   out_dim = 5
   x = [[1.2, 2.5, 3.4],[3.2, 5.7, 6.6]]
   h = [[0, 3, 4]]
   s = [[1, -1, 1]]
   mx.contrib.ndarray.count_sketch(data=x, h=h, s=s, out_dim = 5) = [[1.2, 0, 0, -2.5, 3.4],
                                                                     [3.2, 0, 0, -5.7, 6.6]]

)code" ADD_FILELINE)
.add_argument("data", "NDArray-or-Symbol", "Input data to the CountSketchOp.")
.add_argument("h", "NDArray-or-Symbol", "The index vector")
.add_argument("s", "NDArray-or-Symbol", "The sign vector")
.add_arguments(CountSketchParam::__FIELDS__());

}  // namespace op
}  // namespace mxnet
