/*
 * Copyright (c) 2016 Marcin Junczys-Dowmunt, the University of Edinburgh, Adam
 * Mickiewicz University
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 *
 * All or part of this file was contributed by Intel under license:
 *   Copyright (C) 2017-2018 Intel Corporation
 *   SPDX-License-Identifier: MIT
 *
 * Function LayerNormCPUKernel is adapated from Marian
 * https://github.com/marian-nmt/marian-dev/blob/master/src/tensors/cpu/tensor_operators.cpp
 *
 */

#ifndef MXNET_OPERATOR_NN_LAYER_NORM_CPU_H_
#define MXNET_OPERATOR_NN_LAYER_NORM_CPU_H_

namespace mxnet {
namespace op {

/* CPU optimized kernel for LayerNorm assuming axis = -1.
 * Data is the underlying storage data type.
 * Accum is the type to use for accumulation.
 *   Apparently there isn't a reduction operator for half_t and anyway it isn't
 *   efficient to use on the CPU, so use float for reduction of half_t.
 *
 * width is the number of values being summed to compute a mean.
 * instances is how many independent layer normalization problems are packed into the tensors.
 *
 * Inputs:
 * data is instances x width
 * gamma is width
 * beta is width
 *
 * Outputs:
 * out is instances x width, can be same as data
 * mean is instances: means of each problem
 * std is instances: standard deviation of each problem
 *
 */
template <typename Data,
          typename Accum = typename
          /* By default accumulate in float32 for float16.  Otherwise use same type. */
          std::conditional<std::is_same<mshadow::half::half_t, Data>::value, float, Data>::type>
void LayerNormCPUKernel(size_t width,
                        size_t instances,
                        Data eps,
                        const Data* data,
                        const Data* gamma,
                        const Data* beta,
                        Data* out,
                        Data* mean,
                        Data* std) {
  // Parallelize over independent instances to normalize.
  // MSVC says index variable in OpenMP 'for' statement must have signed integral type.
  const mshadow::index_t signed_instances = static_cast<mshadow::index_t>(instances);
#pragma omp parallel for
  for (nnvm::dim_t j = 0; j < signed_instances; ++j) {
    const Data* from = data + j * width;

    // Sum the values to compute mean.
    Accum sum = 0.f;
#pragma omp simd reduction(+ : sum)
    for (size_t i = 0; i < width; ++i) {
      sum += from[i];
    }
    Accum mean_value = sum / width;
    mean[j]          = static_cast<Data>(mean_value);

    // Sum squares from mean to compute stddev.
    Accum squares = 0.f;
#pragma omp simd reduction(+ : squares)
    for (size_t i = 0; i < width; ++i) {
      Accum off = from[i] - mean_value;
      squares += off * off;
    }
    Accum sigma = std::sqrt(squares / width + eps);
    std[j]      = static_cast<Data>(sigma);

    // Write normalized values.
    Data* to = out + j * width;
#pragma omp simd
    for (size_t i = 0; i < width; ++i) {
      to[i] = (from[i] - mean_value) * gamma[i] / sigma + beta[i];
    }
  }
}

}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_NN_LAYER_NORM_CPU_H_
