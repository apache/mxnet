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
 * \file pdf_op.cc
 * \brief CPU-operators for computing the pdf of random distributions. 
 */

#include "./pdf_op.h"

namespace mxnet {
namespace op {

DMLC_REGISTER_PARAMETER(PdfParam);

#define MXNET_OPERATOR_REGISTER_PDF(distr, pdffunc, num_parms, \
                                    parm_name_1, parm_name_2, \
                                    parm_desc_1, parm_desc_2, \
                                    description, vectorparms) \
  NNVM_REGISTER_OP(_random_pdf_##distr) \
  .add_alias("random_pdf_" #distr) \
  .describe(description()+std::string(ADD_FILELINE)) \
  .set_num_inputs(num_parms+1) \
  .set_num_outputs(1) \
  .set_attr_parser(ParamParser<PdfParam>) \
  .set_attr<nnvm::FListInputNames>("FListInputNames", \
    [](const NodeAttrs& attrs) { \
      std::vector<std::string> v = {"sample", parm_name_1, parm_name_2}; \
      v.resize(num_parms+1); \
      return v; \
    }) \
  .set_attr<mxnet::FInferShape>("FInferShape", PdfOpShape<vectorparms>) \
  .set_attr<nnvm::FInferType>("FInferType", ElemwiseType<num_parms+1, 1>) \
  .set_attr<FCompute>("FCompute<cpu>", PdfOpForward<cpu, pdffunc, num_parms, vectorparms>) \
  .set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseInOut{"_backward_pdf_" #distr}) \
  .add_argument("sample", "NDArray-or-Symbol", "Samples from the distributions.") \
  .add_argument(parm_name_1, "NDArray-or-Symbol", parm_desc_1) \
  .add_arguments(PdfParam::__FIELDS__())

#define MXNET_OPERATOR_REGISTER_PDF_GRAD(distr, pdffunc, num_parms, vectorparms) \
  NNVM_REGISTER_OP(_backward_pdf_##distr) \
  .set_num_inputs(num_parms+3) \
  .set_num_outputs(num_parms+1) \
  .set_attr_parser(ParamParser<PdfParam>) \
  .set_attr<nnvm::FInplaceOption>("FInplaceOption", [](const NodeAttrs& attrs) \
    { std::vector<std::pair<int, int> > v = {{1, 0}, {2, 1}, {3, 2}}; \
    v.resize(num_parms+1); \
    return v; }) \
  .set_attr<FResourceRequest>("FResourceRequest", [](const NodeAttrs& attrs) \
    { return std::vector<ResourceRequest>{ResourceRequest::kTempSpace}; }) \
  .set_attr<nnvm::TIsBackward>("TIsBackward", true) \
  .set_attr<FCompute>("FCompute<cpu>", PdfOpBackward<cpu, pdffunc##_Grad, num_parms, vectorparms>);


#define MXNET_OPERATOR_REGISTER_PDF1(distr, pdffunc, parm_name, parm_desc, \
                                     description, vectorparms) \
    MXNET_OPERATOR_REGISTER_PDF(distr, pdffunc, 1, parm_name, parm_name, \
                                parm_desc, parm_desc, description, vectorparms); \
    MXNET_OPERATOR_REGISTER_PDF_GRAD(distr, pdffunc, 1, vectorparms)

#define MXNET_OPERATOR_REGISTER_PDF2(distr, pdffunc, parm_name_1, parm_name_2, \
                                     parm_desc_1, parm_desc_2, description) \
  MXNET_OPERATOR_REGISTER_PDF(distr, pdffunc, 2, parm_name_1, parm_name_2, \
                                   parm_desc_1, parm_desc_2, description, false) \
  .add_argument(parm_name_2, "NDArray-or-Symbol", parm_desc_2); \
  MXNET_OPERATOR_REGISTER_PDF_GRAD(distr, pdffunc, 2, false)

inline std::string uniform_desc() {
  return std::string(R"code(Computes the value of the PDF of *sample* of
uniform distributions on the intervals given by *[low,high)*.

*low* and *high* must have the same shape, which must match the leftmost subshape
of *sample*.  That is, *sample* can have the same shape as *low* and *high*, in which
case the output contains one density per distribution, or *sample* can be a tensor
of tensors with that shape, in which case the output is a tensor of densities such that
the densities at index *i* in the output are given by the samples at index *i* in *sample*
parameterized by the values of *low* and *high* at index *i*.

Examples::

    random_pdf_uniform(sample=[[1,2,3,4]], low=[0], high=[10]) = [0.1, 0.1, 0.1, 0.1]

    sample = [[[1, 2, 3],
               [1, 2, 3]],
              [[1, 2, 3],
               [1, 2, 3]]]
    low  = [[0, 0],
            [0, 0]]
    high = [[ 5, 10],
            [15, 20]]
    random_pdf_uniform(sample=sample, low=low, high=high) =
        [[[0.2,        0.2,        0.2    ],
          [0.1,        0.1,        0.1    ]],
         [[0.06667,    0.06667,    0.06667],
          [0.05,       0.05,       0.05   ]]]

)code");
}

inline std::string normal_desc() {
  return std::string(R"code(Computes the value of the PDF of *sample* of
normal distributions with parameters *mu* (mean) and *sigma* (standard deviation).

*mu* and *sigma* must have the same shape, which must match the leftmost subshape
of *sample*.  That is, *sample* can have the same shape as *mu* and *sigma*, in which
case the output contains one density per distribution, or *sample* can be a tensor
of tensors with that shape, in which case the output is a tensor of densities such that
the densities at index *i* in the output are given by the samples at index *i* in *sample*
parameterized by the values of *mu* and *sigma* at index *i*.

Examples::

    sample = [[-2, -1, 0, 1, 2]]
    random_pdf_normal(sample=sample, mu=[0], sigma=[1]) =
        [[0.05399097, 0.24197073, 0.3989423, 0.24197073, 0.05399097]]

    random_pdf_normal(sample=sample*2, mu=[0,0], sigma=[1,2]) =
        [[0.05399097, 0.24197073, 0.3989423,  0.24197073, 0.05399097],
         [0.12098537, 0.17603266, 0.19947115, 0.17603266, 0.12098537]]
)code");
}

inline std::string gamma_desc() {
  return std::string(R"code(Computes the value of the PDF of *sample* of
gamma distributions with parameters *alpha* (shape) and *beta* (rate).

*alpha* and *beta* must have the same shape, which must match the leftmost subshape
of *sample*.  That is, *sample* can have the same shape as *alpha* and *beta*, in which
case the output contains one density per distribution, or *sample* can be a tensor
of tensors with that shape, in which case the output is a tensor of densities such that
the densities at index *i* in the output are given by the samples at index *i* in *sample*
parameterized by the values of *alpha* and *beta* at index *i*.

Examples::

  random_pdf_gamma(sample=[[1,2,3,4,5]], alpha=[5], beta=[1]) =
      [[0.01532831, 0.09022352, 0.16803136, 0.19536681, 0.17546739]]

  sample = [[1, 2, 3, 4, 5],
            [2, 3, 4, 5, 6],
            [3, 4, 5, 6, 7]]

  random_pdf_gamma(sample=sample, alpha=[5,6,7], beta=[1,1,1]) =
      [[0.01532831, 0.09022352, 0.16803136, 0.19536681, 0.17546739],
       [0.03608941, 0.10081882, 0.15629345, 0.17546739, 0.16062315],
       [0.05040941, 0.10419563, 0.14622283, 0.16062315, 0.14900276]]
)code");
}

inline std::string exponential_desc() {
  return std::string(R"code(Computes the value of the PDF of *sample* of
exponential distributions with parameters *lam* (rate).

The shape of *lam* must match the leftmost subshape of *sample*.  That is, *sample*
can have the same shape as *lam*, in which case the output contains one density per
distribution, or *sample* can be a tensor of tensors with that shape, in which case
the output is a tensor of densities such that the densities at index *i* in the output
are given by the samples at index *i* in *sample* parameterized by the value of *lam*
at index *i*.

Examples::

  random_pdf_exponential(sample=[[1, 2, 3]], lam=[1]) =
      [[0.36787945, 0.13533528, 0.04978707]]

  sample = [[1,2,3],
            [1,2,3],
            [1,2,3]]

  random_pdf_exponential(sample=sample, lam=[1,0.5,0.25]) =
      [[0.36787945, 0.13533528, 0.04978707],
       [0.30326533, 0.18393973, 0.11156508],
       [0.1947002,  0.15163267, 0.11809164]]
)code");
}

inline std::string poisson_desc() {
  return std::string(R"code(Computes the value of the PDF of *sample* of
Poisson distributions with parameters *lam* (rate).

The shape of *lam* must match the leftmost subshape of *sample*.  That is, *sample*
can have the same shape as *lam*, in which case the output contains one density per
distribution, or *sample* can be a tensor of tensors with that shape, in which case
the output is a tensor of densities such that the densities at index *i* in the output
are given by the samples at index *i* in *sample* parameterized by the value of *lam*
at index *i*.

Examples::

    random_pdf_poisson(sample=[[0,1,2,3]], lam=[1]) =
        [[0.36787945, 0.36787945, 0.18393973, 0.06131324]]

    sample = [[0,1,2,3],
              [0,1,2,3],
              [0,1,2,3]]

    random_pdf_poisson(sample=sample, lam=[1,2,3]) =
        [[0.36787945, 0.36787945, 0.18393973, 0.06131324],
         [0.13533528, 0.27067056, 0.27067056, 0.18044704],
         [0.04978707, 0.14936121, 0.22404182, 0.22404182]]
)code");
}

inline std::string negative_binomial_desc() {
  return std::string(R"code(Computes the value of the PDF of samples of
negative binomial distributions with parameters *k* (failure limit) and *p* (failure probability).

*k* and *p* must have the same shape, which must match the leftmost subshape
of *sample*.  That is, *sample* can have the same shape as *k* and *p*, in which
case the output contains one density per distribution, or *sample* can be a tensor
of tensors with that shape, in which case the output is a tensor of densities such that
the densities at index *i* in the output are given by the samples at index *i* in *sample*
parameterized by the values of *k* and *p* at index *i*.

Examples::

    random_pdf_negative_binomial(sample=[[1,2,3,4]], k=[1], p=a[0.5]) =
        [[0.25, 0.125, 0.0625, 0.03125]]

    # Note that k may be real-valued
    sample = [[1,2,3,4],
              [1,2,3,4]]
    random_pdf_negative_binomial(sample=sample, k=[1, 1.5], p=[0.5, 0.5]) =
        [[0.25,       0.125,      0.0625,     0.03125   ],
         [0.26516506, 0.16572815, 0.09667476, 0.05437956]]
)code");
}

inline std::string generalized_negative_binomial_desc() {
  return std::string(R"code(Computes the value of the PDF of *sample* of
generalized negative binomial distributions with parameters *mu* (mean)
and *alpha* (dispersion).  This can be understood as a reparameterization of
the negative binomial, where *k* = *1 / alpha* and *p* = *1 / (mu \* alpha + 1)*.

*mu* and *alpha* must have the same shape, which must match the leftmost subshape
of *sample*.  That is, *sample* can have the same shape as *mu* and *alpha*, in which
case the output contains one density per distribution, or *sample* can be a tensor
of tensors with that shape, in which case the output is a tensor of densities such that
the densities at index *i* in the output are given by the samples at index *i* in *sample*
parameterized by the values of *mu* and *alpha* at index *i*.

Examples::

    random_pdf_generalized_negative_binomial(sample=[[1, 2, 3, 4]], alpha=[1], mu=[1]) =
        [[0.25, 0.125, 0.0625, 0.03125]]

    sample = [[1,2,3,4],
              [1,2,3,4]]
    random_pdf_generalized_negative_binomial(sample=sample, alpha=[1, 0.6666], mu=[1, 1.5]) =
        [[0.25,       0.125,      0.0625,     0.03125   ],
         [0.26517063, 0.16573331, 0.09667706, 0.05437994]]
)code");
}

inline std::string dirichlet_desc() {
  return std::string(R"code(Computes the value of the PDF of *sample* of
Dirichlet distributions with parameter *alpha*.

The shape of *alpha* must match the leftmost subshape of *sample*.  That is, *sample*
can have the same shape as *alpha*, in which case the output contains one density per
distribution, or *sample* can be a tensor of tensors with that shape, in which case
the output is a tensor of densities such that the densities at index *i* in the output
are given by the samples at index *i* in *sample* parameterized by the value of *alpha*
at index *i*.

Examples::

    random_pdf_dirichlet(sample=[[1,2],[2,3],[3,4]], alpha=[2.5, 2.5]) =
        [38.413498, 199.60245, 564.56085]

    sample = [[[1, 2, 3], [10, 20, 30], [100, 200, 300]],
              [[0.1, 0.2, 0.3], [0.01, 0.02, 0.03], [0.001, 0.002, 0.003]]]

    random_pdf_dirichlet(sample=sample, alpha=[0.1, 0.4, 0.9]) =
        [[2.3257459e-02, 5.8420084e-04, 1.4674458e-05],
         [9.2589635e-01, 3.6860607e+01, 1.4674468e+03]]
)code");
}

MXNET_OPERATOR_REGISTER_PDF2(uniform, PDF_Uniform, "low", "high",
  "Lower bounds of the distributions.", "Upper bounds of the distributions.", uniform_desc)
MXNET_OPERATOR_REGISTER_PDF2(normal, PDF_Normal, "mu", "sigma",
  "Means of the distributions.", "Standard deviations of the distributions.", normal_desc)
MXNET_OPERATOR_REGISTER_PDF2(gamma, PDF_Gamma, "alpha", "beta",
  "Alpha (shape) parameters of the distributions.", "Beta (scale) parameters of the distributions.",
  gamma_desc)
MXNET_OPERATOR_REGISTER_PDF1(exponential, PDF_Exponential, "lam",
  "Lambda (rate) parameters of the distributions.", exponential_desc, false)
MXNET_OPERATOR_REGISTER_PDF1(poisson, PDF_Poisson, "lam",
  "Lambda (rate) parameters of the distributions.", poisson_desc, false)
MXNET_OPERATOR_REGISTER_PDF2(negative_binomial, PDF_NegativeBinomial, "k", "p",
  "Limits of unsuccessful experiments.", "Failure probabilities in each experiment.",
  negative_binomial_desc)
MXNET_OPERATOR_REGISTER_PDF2(generalized_negative_binomial,
  PDF_GeneralizedNegativeBinomial, "mu", "alpha",
  "Means of the distributions.", "Alpha (dispersion) parameters of the distributions.",
  generalized_negative_binomial_desc)
MXNET_OPERATOR_REGISTER_PDF1(dirichlet, PDF_Dirichlet, "alpha",
  "Concentration parameters of the distributions.", dirichlet_desc, true)

}  // namespace op
}  // namespace mxnet
