/*******************************************************************************
* Copyright 2019-2020 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

/// @example cpu_matmul_quantization.cpp
/// > Annotated version: @ref cpu_matmul_quantization_cpp
///
/// @page cpu_matmul_quantization_cpp_short
/// C++ API example demonstrating how one can perform reduced precision
/// matrix-matrix multiplication using [MatMul](@ref dev_guide_matmul) and the
/// accuracy of the result compared to the floating point computations.
///
/// Concepts:
/// - **Static** and **dynamic** quantization
/// - Asymmetric quantization
///   - Run-time output scales: dnnl::primitive_attr::set_output_scales() and
///     #DNNL_RUNTIME_F32_VAL
///   - Run-time zero points: dnnl::primitive_attr::set_zero_points() and
///     #DNNL_RUNTIME_S32_VAL
///
/// @page cpu_matmul_quantization_cpp MatMul Tutorial: Quantization
/// @copydetails cpu_matmul_quantization_cpp_short
///
/// The example is focused around the following computation:
/// \f[
///     C = A \times B
/// \f]
///
/// First, we produce the reference result, having the original matrices
/// \f$A\f$ and \f$B\f$ be in #dnnl::memory::data_type::f32 data type.
///
/// For reduced precision computations, the matrices \f$A\f$ and \f$C\f$ will
/// use #dnnl::memory::data_type::u8 data type and would have the appropriate
/// zero points. For the matrix \f$B\f$, we will use the
/// #dnnl::memory::data_type::s8 data type, assuming that the data is centered
/// around zero (hence, the zero point would be simply 0).
///
/// The quantization formula is:
/// \f[
///     X_{f32}(:) := scale\_X \cdot (X_{int8}(:) - zp\_X),
/// \f]
///
/// where:
/// - \f$X_{f32}(:)\f$  -- original matrix;
///
/// - \f$X_{int8}(:)\f$ -- quantized matrix, where `int8` is either `u8`
///                        (`uint8_t`) for the matrices \f$A\f$ and \f$C\f$, or
///                        `s8` (`int8_t`) for the matrix \f$B\f$;
///
/// - \f$scale\_X\f$    -- `f32` scaling factor. For simplicity we will use a
///                        single scale factor for each matrix, though for
///                        better accuracy it might be a good idea to use
///                        per-N-dimension scaling factor for the matrix B.
///
/// - \f$zp\_X\f$       -- integer quantization parameter "zero point"
///                        (essentially, the representation of the real 0 in
///                        the quantized data type).
///
/// For a given matrix \f$X_{f32}\f$ and `int8` data type (`u8` or `s8`), the
/// process of finding the proper \f$scale\_X\f$ and \f$zp\_X\f$ is a research
/// problem and can be different depending on the domain. For example purposes,
/// we will use the simplest approach by mapping the maximum (minimum)
/// \f$X_{f32}\f$ elements to the maximum (minimum) number in the corresponding
/// integer data type, using the following formulas:
///
/// 1. Since:
///   - \f$max(X_{f32}(:)) = scale\_X \cdot (max_{int8} - zp\_X)\f$
///   - \f$min(X_{f32}(:)) = scale\_X \cdot (min_{int8} - zp\_X)\f$
///
/// 2. Hence:
///   - \f$scale\_X =
///     \frac{max(X_{f32}(:)) - min(X_{f32}(:))}{max_{int8} - min_{int8}}\f$
///   - \f$zp\_X = max_{int8} - \frac{max(X_{f32}(:))}{scale\_X}\f$
///
/// It is worth noting that quantization parameters are not always computed at
/// actual run-time. For example, if we perform MatMul operation for _similar_
/// matrices (in a sense that data distribution is similar between the runs) we
/// can simply _guess_ the proper quantization parameters by collecting some
/// statistics during the early runs. This approach is called **static**
/// quantization. It gives good performance (since no cycles are spent on
/// computing those parameters) and is typically used in reduced precision
/// CNN inference. However, the **static** quantization has an obvious
/// disadvantage -- the _guessed_ parameters might not work well for some
/// particular matrices. For example, that would most likely be the case if we
/// could not guarantee the similarity of the input matrices. In this case, the
/// **dynamic** quantization would be used, i.e. the parameters (re-)computed at
/// runtime. This gives slightly worse performance, but that might be inevitable
/// due to accuracy considerations.
///
/// Both approaches are demonstrated in this example.
///
/// Other details:
/// - For simplicity all matrices will be stored in Row-Major format.
/// - The shapes of the matrices are assumed to be known at creation time.
///   However, for dynamic quantization we would consider q10n parameters
///   (\f$scale\_X\f$ and \f$zp\_X\f$) to be known at run-time only. On the
///   contrary, for the static quantization these parameters are known at
///   creation time as well.
///
/// @include cpu_matmul_quantization.cpp

#include <cassert>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>
#include <type_traits>

#include "oneapi/dnnl/dnnl.hpp"

#include "example_utils.hpp"

using namespace dnnl;

enum class q10n_scheme_t { DYNAMIC, STATIC };

namespace {

void init_vector(std::vector<float> &v, float min_value, float max_value) {
    std::mt19937 gen;
    std::uniform_real_distribution<float> u(min_value, max_value);

    for (auto &e : v)
        e = u(gen);
}

template <typename T>
void find_min_max(const std::vector<T> &v, float &min_value, float &max_value) {
    min_value = max_value = v[0];
    for (auto &e : v) {
        min_value = std::min<float>(min_value, e);
        max_value = std::max<float>(max_value, e);
    }
}

template <typename T>
void compute_q10n_params(const char *message, const std::vector<float> &v,
        float &scale, int32_t &zp) {
    // Find property of T integer type
    // Simple trick to improve accuracy: shrink the range a little bit
    float max_int = (float)std::numeric_limits<T>::max() - 1;
    float min_int = (float)std::numeric_limits<T>::lowest() + 1;

#ifndef OMIT_WORKAROUND_FOR_SKX
    // Read more in CPU / Section 1 here:
    // https://oneapi-src.github.io/oneDNN/dev_guide_int8_computations.html
    if (std::is_same<T, uint8_t>::value) max_int /= 2;
#endif

    // Find min and max value in array
    float min_val = v[0], max_val = v[0];
    find_min_max(v, min_val, max_val);

    // Compute appropriate scale
    scale = (max_val - min_val) / (max_int - min_int);

    // Compute appropriate offset
    if (std::is_same<T, int8_t>::value)
        zp = 0;
    else
        zp = (int32_t)(max_int - max_val / scale);
    printf("\tComputing q10n params for %s\n"
           "\t\tData type: %s\n"
           "\t\tScale:%.3g (inverse scale:%.3g)\n"
           "\t\tZero point:%d\n\n",
            message, std::is_same<T, int8_t>::value ? "int8_t" : "uint8_t",
            scale, 1 / scale, zp);
}

int compare_vectors(const std::vector<float> &v1,
        const std::vector<uint8_t> &v2, float scale_v2, int32_t zp_v2,
        float threshold) {
    double v1_l2 = 0, diff_l2 = 0;
    for (size_t n = 0; n < v1.size(); ++n) {
        float v2_n = scale_v2 * (v2[n] - zp_v2); // deq10n v2
        float diff = v1[n] - v2_n;
        v1_l2 += v1[n] * v1[n];
        diff_l2 += diff * diff;
    }

    v1_l2 = std::sqrt(v1_l2);
    diff_l2 = std::sqrt(diff_l2);
    bool ok = diff_l2 <= threshold * v1_l2;

    printf("\tComparison (using l2-norms)\n"
           "\t\tReference matrix:%g\n\t\tError:%g\n\t\tRelative error:%g\n"
           "\nAccuracy check: %s\n\n",
            v1_l2, diff_l2, diff_l2 / v1_l2, ok ? "OK" : "FAILED");

    return ok ? 0 : 1;
}

} // namespace

engine eng(engine::kind::cpu, 0); // We create a global engine for simplicity

// Quantize float data into X_int_m oneDNN memory using the q10n parameters
//
// Inputs:
// - X_f32 -- source f32 matrix
// - scale_X, zp_X -- quantization parameters
// - q10n_scheme -- dynamic or static, to mimic real-world applications wrt to
//                  how the q10n parameters are passed to reorders
// Outputs:
// - X_int_m -- prepared oneDNN memory that would hold quantized values
void quantize(q10n_scheme_t q10n_scheme, const std::vector<float> &X_f32,
        float scale_X, int32_t zp_X, memory &X_int_m) {
    using dt = memory::data_type;

    // Depending on `q10n_scheme` pretend the values come at run-time (dynamic)
    // or were known at creation time (static).
    float inv_scale_X = 1.f / scale_X;

    const bool is_dynamic_q10n = q10n_scheme == q10n_scheme_t::DYNAMIC;

    stream s(eng);

    memory::desc x_int_md = X_int_m.get_desc();
    const auto &dims = x_int_md.data.dims;

    memory::desc x_f32_md({dims[0], dims[1]}, dt::f32, {dims[1], 1});
    memory X_f32_m(x_f32_md, eng, (void *)X_f32.data());

    primitive_attr q10n_attr;
    q10n_attr.set_output_scales(/* mask */ 0,
            {is_dynamic_q10n ? DNNL_RUNTIME_F32_VAL : inv_scale_X});
    q10n_attr.set_zero_points(DNNL_ARG_DST, /* mask */ 0,
            {is_dynamic_q10n ? DNNL_RUNTIME_S32_VAL : zp_X});

    reorder::primitive_desc q10n_pd(eng, x_f32_md, eng, x_int_md, q10n_attr);
    if (is_dynamic_q10n) {
        memory scale_X_m({{1}, dt::f32, {1}}, eng, &inv_scale_X);
        memory zp_X_m({{1}, dt::s32, {1}}, eng, &zp_X);
        reorder(q10n_pd).execute(s,
                {{DNNL_ARG_SRC, X_f32_m}, {DNNL_ARG_DST, X_int_m},
                        {DNNL_ARG_ATTR_OUTPUT_SCALES, scale_X_m},
                        {DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_DST, zp_X_m}});
    } else {
        reorder(q10n_pd).execute(
                s, {{DNNL_ARG_SRC, X_f32_m}, {DNNL_ARG_DST, X_int_m}});
    }

    s.wait();
}

// Floating point MatMul
// Inputs:
// - Shape: M, N, K
// - Matrices A and B
// Outputs:
// - Matrix C
void f32_matmul_compute(int64_t M, int64_t N, int64_t K,
        const std::vector<float> &A_f32, const std::vector<float> &B_f32,
        std::vector<float> &C_f32) {
    // Initialize memory descriptors that describes matrices in Row-Major format
    memory::desc a_md({M, K}, memory::data_type::f32, {K, 1});
    memory::desc b_md({K, N}, memory::data_type::f32, {N, 1});
    memory::desc c_md({M, N}, memory::data_type::f32, {N, 1});

    // Wrap raw pointers into oneDNN memory objects
    memory A_f32_m(a_md, eng, (void *)A_f32.data());
    memory B_f32_m(b_md, eng, (void *)B_f32.data());
    memory C_f32_m(c_md, eng, (void *)C_f32.data());

    // Create a MatMul primitive
    matmul::desc matmul_d(a_md, b_md, c_md);
    matmul::primitive_desc matmul_pd(matmul_d, eng);
    matmul matmul_p(matmul_pd);

    stream s(eng);
    matmul_p.execute(s,
            {{DNNL_ARG_SRC, A_f32_m}, {DNNL_ARG_WEIGHTS, B_f32_m},
                    {DNNL_ARG_DST, C_f32_m}});
    s.wait();
}

// Reduced precision MatMul with **dynamic** quantization
// Inputs:
// - Shape: M, N, K
// - Matrices A and B in float (would be quantized inside the function)
// Outputs:
// - Matrix C in uint8_t
// - Quantization parameters: scale_C and zp_C
void dynamic_q10n_matmul(int64_t M, int64_t N, int64_t K,
        const std::vector<float> &A_f32, const std::vector<float> &B_f32,
        std::vector<uint8_t> &C_u8, float &scale_C, int32_t &zp_C) {
    stream s(eng);

    float scale_A, scale_B;
    int32_t zp_A, zp_B;

    // We compute q10n parameters here, but in the real world applications for
    // inputs these parameters are transferred from the previous layers
    compute_q10n_params<uint8_t>("A", A_f32, scale_A, zp_A);
    compute_q10n_params<int8_t>("B", B_f32, scale_B, zp_B);
    assert(zp_B == 0 && "for int8 q10n we assume zero point = 0");

    // Quantize matrix A_u8 using reorder primitive
    std::vector<uint8_t> A_u8(M * K, 0);
    memory::desc a_u8_md({M, K}, memory::data_type::u8, {K, 1});
    memory A_u8_m(a_u8_md, eng, (void *)A_u8.data());
    quantize(q10n_scheme_t::DYNAMIC, A_f32, scale_A, zp_A, A_u8_m);

    // Quantize matrix B_s8 using reorder primitive
    std::vector<uint8_t> B_s8(K * N, 0);
    memory::desc b_s8_md({K, N}, memory::data_type::s8, {N, 1});
    memory B_s8_m(b_s8_md, eng, (void *)B_s8.data());
    quantize(q10n_scheme_t::DYNAMIC, B_f32, scale_B, 0, B_s8_m);

    // Compute C_f32. We cannot directly compute C_u8 since we don't know the
    // appropriate quantization parameters.
    //
    // Note: typically the computed data type in this case is int32_t and not
    //       float. But for brevity we are going to embed the scale_A and
    //       scale_B directly in this quantized MatMul, and hence will get the
    //       intermediate computation in floating point anyways, so there is
    //       no sense to convert the result to int32_t.
    //       In theory, we could postpone using the scale_A and scale_B, compute
    //       the exact C_s32 := (A_u8 - zp_A) * B_s8, and then find the
    //       appropriate quantization parameters for matrix C.
    //       Let it be an exercise :)

    std::vector<float> C_f32(M * N, 0);
    memory::desc c_f32_md({M, N}, memory::data_type::f32, {N, 1});
    memory C_f32_m(c_f32_md, eng, (void *)C_f32.data());

    // Create and compute a reduced precision MatMul primitive
    {
        primitive_attr matmul_attr;
        matmul_attr.set_output_scales(/* mask */ 0, {DNNL_RUNTIME_F32_VAL});
        matmul_attr.set_zero_points(
                DNNL_ARG_SRC, /* mask */ 0, {DNNL_RUNTIME_S32_VAL});

        matmul::desc matmul_d(a_u8_md, b_s8_md, c_f32_md);
        matmul::primitive_desc matmul_pd(matmul_d, matmul_attr, eng);
        matmul matmul_p(matmul_pd);

        // Pretend the values come at run-time
        float output_scale = scale_A * scale_B;

        memory output_scales_m(
                {{1}, memory::data_type::f32, {1}}, eng, &output_scale);
        memory zp_A_m({{1}, memory::data_type::s32, {1}}, eng, &zp_A);

        matmul_p.execute(s,
                {{DNNL_ARG_SRC, A_u8_m}, {DNNL_ARG_WEIGHTS, B_s8_m},
                        {DNNL_ARG_DST, C_f32_m},
                        {DNNL_ARG_ATTR_OUTPUT_SCALES, output_scales_m},
                        {DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_SRC, zp_A_m}});
    }

    // Find quantization parameters for matrix C
    compute_q10n_params<uint8_t>("C", C_f32, scale_C, zp_C);

    // Finally quantize the matrix C
    memory::desc c_u8_md({M, N}, memory::data_type::u8, {N, 1});
    memory C_u8_m(c_u8_md, eng, (void *)C_u8.data());
    quantize(q10n_scheme_t::DYNAMIC, C_f32, scale_C, zp_C, C_u8_m);
}

// Reduced precision MatMul with **static** quantization
// Inputs:
// - Shape: M, N, K
// - Matrices A and B in float (would be quantized inside the function using
//   given q10n parameters)
// - Quantization parameters for all 3 matrices:
//   - scale_A, zp_A
//   - scale_B
//   - scale_C, zp_C
// Outputs:
// - Matrix C in uint8_t
void static_q10n_matmul(int64_t M, int64_t N, int64_t K,
        const std::vector<float> &A_f32, const std::vector<float> &B_f32,
        float scale_A, int32_t zp_A, float scale_B, float scale_C, int32_t zp_C,
        std::vector<uint8_t> &C_u8) {
    stream s(eng);

    // Quantize matrix A_u8 using reorder primitive
    std::vector<uint8_t> A_u8(M * K, 0);
    memory::desc a_u8_md({M, K}, memory::data_type::u8, {K, 1});
    memory A_u8_m(a_u8_md, eng, (void *)A_u8.data());
    quantize(q10n_scheme_t::STATIC, A_f32, scale_A, zp_A, A_u8_m);

    // Quantize matrix B_s8 using reorder primitive
    std::vector<uint8_t> B_s8(K * N, 0);
    memory::desc b_s8_md({K, N}, memory::data_type::s8, {N, 1});
    memory B_s8_m(b_s8_md, eng, (void *)B_s8.data());
    quantize(q10n_scheme_t::STATIC, B_f32, scale_B, 0, B_s8_m);

    // Directly compute C_u8, since we know quantization parameters for the
    // matrix C. This is the key difference compare to **dynamic** quantization.
    {
        memory::desc c_u8_md({M, N}, memory::data_type::u8, {N, 1});
        memory C_u8_m(c_u8_md, eng, (void *)C_u8.data());

        primitive_attr matmul_attr;
        matmul_attr.set_output_scales(
                /* mask */ 0, {scale_A * scale_B / scale_C});
        matmul_attr.set_zero_points(DNNL_ARG_SRC, /* mask */ 0, {zp_A});
        matmul_attr.set_zero_points(DNNL_ARG_DST, /* mask */ 0, {zp_C});

        matmul::desc matmul_d(a_u8_md, b_s8_md, c_u8_md);
        matmul::primitive_desc matmul_pd(matmul_d, matmul_attr, eng);
        matmul matmul_p(matmul_pd);

        matmul_p.execute(s,
                {{DNNL_ARG_SRC, A_u8_m}, {DNNL_ARG_WEIGHTS, B_s8_m},
                        {DNNL_ARG_DST, C_u8_m}});
    }
}

void compare_f32_and_quantized_matmuls() {
    // MatMul parameters
    const int64_t M = 10, N = 20, K = 30;

    // Data distribution for matrices A and B
    const float param_A_min_val = -2.f;
    const float param_A_max_val = 1.4f;

    const float param_B_min_val = -1.f;
    const float param_B_max_val = -param_B_min_val; // B is centered around 0

    // Thresholds
    //
    // Ideally the threshold for static quantization should be a little higher
    // than for dynamic quantization. However, we will slightly cheat on the
    // guessed q10n parameters of matrix C (see below), so we will get pretty
    // good accuracy as well.
    const float threshold_dynamic_q10n = 3 * 1e-2f;
    const float threshold_static_q10n = 4 * 1e-2f;

    // Prepare matrices
    std::vector<float> A_f32(M * K), B_f32(K * N), C_f32(M * N, 0);
    init_vector(A_f32, param_A_min_val, param_A_max_val);
    init_vector(B_f32, param_B_min_val, param_B_max_val);

    // Compute _true_ f32 result
    f32_matmul_compute(M, N, K, A_f32, B_f32, C_f32);

    // Compute quantized variant (dynamic)
    {
        printf("# DYNAMIC quantization\n\n");

        std::vector<uint8_t> C_u8_dynamic_q10n(M * N, 0);

        float scale_C_dynamic_q10n; // Q10n parameters we don't know yet
        int zp_C_dynamic_q10n;

        dynamic_q10n_matmul(M, N, K, A_f32, B_f32, C_u8_dynamic_q10n,
                scale_C_dynamic_q10n, zp_C_dynamic_q10n);

        // Compare _true_ f32 result with dynamic q10n
        int rc = compare_vectors(C_f32, C_u8_dynamic_q10n, scale_C_dynamic_q10n,
                zp_C_dynamic_q10n, threshold_dynamic_q10n);
        if (rc) throw std::logic_error("Dynamic quantization accuracy failed.");
    }

    // Compute quantized variant (static)
    {
        printf("# STATIC quantization\n\n");

        std::vector<uint8_t> C_u8_static_q10n(M * N, 0);

        // Let's pretend we know the appropriate q10n parameters (by gathering
        // some statistic or whatnot). For matrix C we will slightly _cheat_
        // and get the appropriate q10n from the actual C_f32 result that we
        // computed earlier. Of course, it is not what one would do in the
        // **static** q10n scheme (just by the definition of the **static**
        // q10n), but solely for the purpose of this example print "passed" in
        // the end :)
        const float scale_A_static_q10n
                = (param_A_max_val - param_A_min_val) / 128;
        const int zp_A_static_q10n
                = (int)(128 - param_A_max_val / scale_A_static_q10n);
        const float scale_B_static_q10n
                = (param_B_max_val - param_B_min_val) / 256;

        float scale_C_static_q10n;
        int zp_C_static_q10n;
        // !!! CHEATING STARTS HERE
        const char *warn_message
                = "C"
                  "\n\t*******************************************************"
                  "\n\t* NOTE: These computation do not happen in real world *"
                  "\n\t*       applications and used here solely to simplify *"
                  "\n\t*       the example.                                  *"
                  "\n\t*       Please refer to the example source code for   *"
                  "\n\t*       more information.                             *"
                  "\n\t*******************************************************";

        compute_q10n_params<uint8_t>(
                warn_message, C_f32, scale_C_static_q10n, zp_C_static_q10n);
        // !!! CHEATING ENDS HERE

        static_q10n_matmul(M, N, K, A_f32, B_f32, scale_A_static_q10n,
                zp_A_static_q10n, scale_B_static_q10n, scale_C_static_q10n,
                zp_C_static_q10n, C_u8_static_q10n);

        // Compare _true_ f32 result with static q10n
        int rc = compare_vectors(C_f32, C_u8_static_q10n, scale_C_static_q10n,
                zp_C_static_q10n, threshold_static_q10n);
        if (rc) throw std::logic_error("Static quantization accuracy failed.");
    }
}

int main(int argc, char **argv) {
    return handle_example_errors(
            {engine::kind::cpu}, compare_f32_and_quantized_matmuls);
}
