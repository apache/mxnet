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

#if MXNET_USE_ONEDNN == 1

#include "dnnl_bn_relu_property.h"
#include "dnnl_conv_property.h"
#include "dnnl_elemwisemul_post_quantize_property.h"
#include "dnnl_fc_post_quantize_property.h"
#include "dnnl_fc_property.h"
#include "dnnl_post_quantize_align_scale_property.h"
#include "dnnl_post_quantize_property.h"
#include "dnnl_transformer_post_quantize_property.h"
#include "dnnl_transformer_qk_property.h"
#include "dnnl_transformer_valatt_property.h"

namespace mxnet {
namespace op {

MXNET_REGISTER_SUBGRAPH_BACKEND(DNNL)
    .set_attr("enable", DNNLEnvSet())
    .set_attr("context", Context::CPU());

MXNET_REGISTER_SUBGRAPH_PROPERTY(DNNL, SgDNNLConvProperty);
MXNET_REGISTER_SUBGRAPH_PROPERTY(DNNL, SgDNNLFCProperty);
MXNET_REGISTER_SUBGRAPH_PROPERTY(DNNL, SgDNNLBNReLUProperty);
MXNET_REGISTER_SUBGRAPH_PROPERTY(DNNL, SgDNNLTransformerQKProperty);
MXNET_REGISTER_SUBGRAPH_PROPERTY(DNNL, SgDNNLTransformerValAttProperty);

MXNET_REGISTER_SUBGRAPH_BACKEND(DNNL_QUANTIZE).set_attr("context", Context::CPU());

MXNET_REGISTER_SUBGRAPH_PROPERTY(DNNL_QUANTIZE, SgDNNLConvProperty).set_attr("quantize", true);

MXNET_REGISTER_SUBGRAPH_PROPERTY(DNNL_QUANTIZE, SgDNNLFCProperty).set_attr("quantize", true);
MXNET_REGISTER_SUBGRAPH_PROPERTY(DNNL_QUANTIZE, SgDNNLTransformerQKProperty);
MXNET_REGISTER_SUBGRAPH_PROPERTY(DNNL_QUANTIZE, SgDNNLTransformerValAttProperty);

MXNET_REGISTER_SUBGRAPH_PROPERTY(DNNL_QUANTIZE, SgDNNLPostQuantizeProperty);
MXNET_REGISTER_SUBGRAPH_PROPERTY(DNNL_QUANTIZE, SgDNNLFCPostQuantizeProperty);
MXNET_REGISTER_SUBGRAPH_PROPERTY(DNNL_QUANTIZE, ElemwiseMulPostQuantizeProperty);
MXNET_REGISTER_SUBGRAPH_PROPERTY(DNNL_QUANTIZE, SgDNNLPostQuantizeAlignScaleProperty);
MXNET_REGISTER_SUBGRAPH_PROPERTY(DNNL_QUANTIZE, SgDNNLTransformerPostQuantizeProperty)
    .set_attr("quantize", true);

}  // namespace op
}  // namespace mxnet
#endif  // MXNET_USE_ONEDNN == 1