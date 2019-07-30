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
 * Copyright (c) 2018 by Contributors
 * \file pdf_op.cu
 * \brief GPU-operators for computing the pdf of random distributions. 
 */

#include "./pdf_op.h"

namespace mxnet {
namespace op {

#define MXNET_OPERATOR_REGISTER_PDF(distr, pdffunc, num_parms, vector_parms) \
  NNVM_REGISTER_OP(_random_pdf_##distr) \
  .set_attr<FCompute>("FCompute<gpu>", PdfOpForward<gpu, pdffunc, num_parms, vector_parms>); \
  NNVM_REGISTER_OP(_backward_pdf_##distr) \
  .set_attr<FCompute>("FCompute<gpu>", PdfOpBackward<gpu, pdffunc##_Grad, num_parms, vector_parms>);

MXNET_OPERATOR_REGISTER_PDF(uniform, PDF_Uniform, 2, false)
MXNET_OPERATOR_REGISTER_PDF(normal, PDF_Normal, 2, false)
MXNET_OPERATOR_REGISTER_PDF(gamma, PDF_Gamma, 2, false)
MXNET_OPERATOR_REGISTER_PDF(exponential, PDF_Exponential, 1, false)
MXNET_OPERATOR_REGISTER_PDF(poisson, PDF_Poisson, 1, false)
MXNET_OPERATOR_REGISTER_PDF(negative_binomial, PDF_NegativeBinomial, 2, false)
MXNET_OPERATOR_REGISTER_PDF(generalized_negative_binomial,
                            PDF_GeneralizedNegativeBinomial, 2, false)
MXNET_OPERATOR_REGISTER_PDF(dirichlet, PDF_Dirichlet, 1, true)

}  // namespace op
}  // namespace mxnet
