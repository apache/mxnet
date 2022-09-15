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

#include "dnnl_batch_dot_property.h"
#include "dnnl_bn_relu_property.h"
#include "dnnl_conv_property.h"
#include "dnnl_fc_property.h"
#include "dnnl_identity_property.h"
#include "dnnl_post_amp_property.h"
#include "dnnl_post_quantize_align_scale_property.h"
#include "dnnl_post_quantize_property.h"
#include "dnnl_pow_mul_scalar_property.h"
#include "dnnl_transformer_qk_property.h"
#include "dnnl_transformer_valatt_property.h"
#include "dnnl_fc_sum_fuse_property.h"
#include "dnnl_remove_casts_property.h"

namespace mxnet {
namespace op {

MXNET_REGISTER_SUBGRAPH_BACKEND(ONEDNN)
    .set_attr("enable", DNNLEnvSet())
    .set_attr("context", Context::CPU());

MXNET_REGISTER_SUBGRAPH_PROPERTY(ONEDNN, SgDNNLIdentityProperty);
MXNET_REGISTER_SUBGRAPH_PROPERTY(ONEDNN, SgDNNLConvProperty);
MXNET_REGISTER_SUBGRAPH_PROPERTY(ONEDNN, SgDNNLFCProperty);
MXNET_REGISTER_SUBGRAPH_PROPERTY(ONEDNN, SgDNNLBNReLUProperty);
MXNET_REGISTER_SUBGRAPH_PROPERTY(ONEDNN, SgDNNLRemoveCastsProperty);
MXNET_REGISTER_SUBGRAPH_PROPERTY(ONEDNN, SgDNNLTransformerQKSplitProperty);
MXNET_REGISTER_SUBGRAPH_PROPERTY(ONEDNN, SgDNNLTransformerQKProperty);
MXNET_REGISTER_SUBGRAPH_PROPERTY(ONEDNN, SgDNNLTransformerValAttProperty);
MXNET_REGISTER_SUBGRAPH_PROPERTY(ONEDNN, SgDNNLBatchDotProperty);
MXNET_REGISTER_SUBGRAPH_PROPERTY(ONEDNN, SgDNNLPowMulScalarProperty);
MXNET_REGISTER_SUBGRAPH_PROPERTY(ONEDNN, SgDNNLFCSumFuseProperty);

MXNET_REGISTER_SUBGRAPH_BACKEND(ONEDNN_QUANTIZE).set_attr("context", Context::CPU());

MXNET_REGISTER_SUBGRAPH_PROPERTY(ONEDNN_QUANTIZE, SgDNNLIdentityProperty);
MXNET_REGISTER_SUBGRAPH_PROPERTY(ONEDNN_QUANTIZE, SgDNNLConvProperty).set_attr("quantize", true);
MXNET_REGISTER_SUBGRAPH_PROPERTY(ONEDNN_QUANTIZE, SgDNNLFCProperty).set_attr("quantize", true);
MXNET_REGISTER_SUBGRAPH_PROPERTY(ONEDNN_QUANTIZE, SgDNNLTransformerQKSplitProperty);
MXNET_REGISTER_SUBGRAPH_PROPERTY(ONEDNN_QUANTIZE, SgDNNLBNReLUProperty);
MXNET_REGISTER_SUBGRAPH_PROPERTY(ONEDNN_QUANTIZE, SgDNNLTransformerQKProperty);
MXNET_REGISTER_SUBGRAPH_PROPERTY(ONEDNN_QUANTIZE, SgDNNLTransformerValAttProperty);
MXNET_REGISTER_SUBGRAPH_PROPERTY(ONEDNN_QUANTIZE, SgDNNLBatchDotProperty)
    .set_attr("quantize", true);
MXNET_REGISTER_SUBGRAPH_PROPERTY(ONEDNN_QUANTIZE, SgDNNLPostQuantizeProperty);
MXNET_REGISTER_SUBGRAPH_PROPERTY(ONEDNN_QUANTIZE, SgDNNLPostQuantizeAlignScaleProperty);
MXNET_REGISTER_SUBGRAPH_PROPERTY(ONEDNN_QUANTIZE, SgDNNLFCSumFuseProperty)
    .set_attr("quantize", true);

MXNET_REGISTER_SUBGRAPH_BACKEND(ONEDNN_AMP).set_attr("context", Context::CPU());
MXNET_REGISTER_SUBGRAPH_PROPERTY(ONEDNN_AMP, SgDNNLPostAMPProperty);

}  // namespace op
}  // namespace mxnet
#endif  // MXNET_USE_ONEDNN == 1
