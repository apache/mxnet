/*******************************************************************************
* Copyright 2020 Intel Corporation
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

#ifndef GPU_OCL_OCL_ELTWISE_H
#define GPU_OCL_OCL_ELTWISE_H

#if WITH_ELTWISE

#if DT_F16 == 1
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#endif

#ifndef DATA_MAX
#if DT_F16 == 1
#define DATA_MAX HALF_MAX
#elif DT_S8 == 1
#define DATA_MAX CHAR_MAX
#elif DT_U8 == 1
#define DATA_MAX UCHAR_MAX
#else
#define DATA_MAX FLT_MAX
#endif
#endif

#ifndef ELTWISE_ALPHA0
#define ELTWISE_ALPHA0 0
#endif

float relu_fwd(float s, float alpha) {
    return s > 0 ? s : (ELTWISE_ALPHA0 ? 0 : s * alpha);
}
float relu_bwd(float dd, float s, float alpha) {
    return s > 0 ? dd : dd * alpha;
}
float relu_bwd_use_dst(float dd, float d, float alpha) {
    return d > 0 ? dd : dd * alpha;
}

float linear_fwd(float s, float alpha, float beta) {
    return alpha * s + beta;
}
float linear_bwd(float dd, float alpha) {
    return dd * alpha;
}

float bounded_relu_fwd(float s, float alpha) {
    s = s > 0 ? s : 0;
    return s > alpha ? alpha : s;
}
float bounded_relu_bwd(float dd, float s, float alpha) {
    return dd * (0 < s && s <= alpha ? 1 : 0);
}

float soft_relu_fwd(float s) {
    return s < log((float)DATA_MAX) ? log1p(exp(s)) : s;
}
float soft_relu_bwd(float dd, float s) {
    return dd / (1 + exp(-s));
}

float logistic_fwd(float s) {
    return 1.0f / (1.0f + exp(-s));
}
float logistic_bwd(float dd, float s) {
    float v = logistic_fwd(s);
    return dd * v * (1 - v);
}
float logistic_bwd_use_dst(float dd, float d) {
    return dd * d * (1 - d);
}

float square_fwd(float s) {
    return s * s;
}
float square_bwd(float dd, float s) {
    return dd * 2 * s;
}

float sqrt_fwd(float s) {
    return sqrt(s);
}
float sqrt_bwd(float dd, float s) {
    return dd / (2 * sqrt(s));
}
float sqrt_bwd_use_dst(float dd, float d) {
    return dd / (2 * d);
}

float abs_fwd(float s) {
    return s > 0 ? s : -s;
}
float abs_bwd(float dd, float s) {
    return s > 0 ? dd : s < 0 ? -dd : 0;
}

float tanh_fwd(float s) {
    return tanh(s);
}
float tanh_bwd(float dd, float s) {
    float e = tanh_fwd(s);
    return dd * (1 - e) * (1 + e);
}
float tanh_bwd_use_dst(float dd, float d) {
    return dd * (1 - d) * (1 + d);
}

float elu_fwd(float s, float alpha) {
    return s > 0 ? s : alpha * expm1(s);
}
float elu_bwd(float dd, float s, float alpha) {
    return dd * (s > 0 ? 1 : alpha * exp(s));
}
float elu_bwd_use_dst(float dd, float d, float alpha) {
    return dd * (d > 0 ? 1 : d + alpha);
}

float exp_fwd(float s) {
    return exp(s);
}
float exp_bwd(float dd, float s) {
    return dd * exp_fwd(s);
}
float exp_bwd_use_dst(float dd, float d) {
    return dd * d;
}

float gelu_tanh_fwd(float s) {
    const float sqrt_2_over_pi = 0.79788458347320556640625f;
    const float fitting_const = 0.044715f;
    const float g = sqrt_2_over_pi * s * (1.f + fitting_const * s * s);
    return (0.5f * s * (1.f + tanh_fwd(g)));
}
float gelu_tanh_bwd(float dd, float s) {
    const float sqrt_2_over_pi = 0.79788458347320556640625f;
    const float fitting_const = 0.044715f;
    const float g = sqrt_2_over_pi * s * (1.f + fitting_const * s * s);
    const float dg = sqrt_2_over_pi * (1.f + 3.f * fitting_const * s * s);
    const float v = tanh_fwd(g);
    return dd * 0.5f * (1.f + v) * (1.f + s * (1.f - v) * dg);
}

float swish_fwd(float s, float alpha) {
    float w = -alpha * s;
    return s / (1.0f + exp(w));
}
float swish_bwd(float dd, float s, float alpha) {
    float v = logistic_fwd(alpha * s);
    return dd * (v + s * alpha * v * (1.0f - v));
}

float log_fwd(float s) {
    return log(s);
}
float log_bwd(float dd, float s) {
    return dd / s;
}

float clip_fwd(float s, float alpha, float beta) {
    s = s > alpha ? s : alpha;
    return s > beta ? beta : s;
}
float clip_bwd(float dd, float s, float alpha, float beta) {
    return dd * (alpha < s && s <= beta ? 1 : 0);
}

float pow_fwd(float s, float alpha, float beta) {
    return alpha * pow(s, beta);
}
float pow_bwd(float dd, float s, float alpha, float beta) {
    if (beta == 0) return 0;

    float v = pow_fwd(s, alpha * beta, beta - 1);
    return dd * v;
}

float gelu_erf_fwd(float s) {
    const float sqrt_2_over_2 = 0.707106769084930419921875f;
    float v = s * sqrt_2_over_2;
    return 0.5f * s * (1.f + erf(v));
}

float gelu_erf_bwd(float dd, float s) {
    const float two_over_sqrt_pi = 1.12837922573089599609375f;
    const float sqrt_2_over_2 = 0.707106769084930419921875f;
    float v = s * sqrt_2_over_2;
    return dd * 0.5f * (1.f + erf(v) + v * two_over_sqrt_pi * exp(-v * v));
}

float round_fwd(float s) {
    return rint(s);
}

float fwd_eltwise_common(
        int eltwise_alg, float x, float alpha_, float beta_, float scale_) {
    switch (eltwise_alg) {
        case RELU: return scale_ * relu_fwd(x, alpha_); break;
        case LINEAR: return scale_ * linear_fwd(x, alpha_, beta_); break;
        case BOUNDED_RELU: return scale_ * bounded_relu_fwd(x, alpha_); break;
        case SOFT_RELU: return scale_ * soft_relu_fwd(x); break;
        case LOGISTIC: return scale_ * logistic_fwd(x); break;
        case TANH: return scale_ * tanh_fwd(x); break;
        case ELU: return scale_ * elu_fwd(x, alpha_); break;
        case SQUARE: return scale_ * square_fwd(x); break;
        case SQRT: return scale_ * sqrt_fwd(x); break;
        case ABS: return scale_ * abs_fwd(x); break;
        case EXP: return scale_ * exp_fwd(x); break;
        case GELU_TANH: return scale_ * gelu_tanh_fwd(x); break;
        case SWISH: return scale_ * swish_fwd(x, alpha_); break;
        case LOG: return scale_ * log_fwd(x); break;
        case CLIP: return scale_ * clip_fwd(x, alpha_, beta_); break;
        case POW: return scale_ * pow_fwd(x, alpha_, beta_); break;
        case GELU_ERF: return scale_ * gelu_erf_fwd(x); break;
        case ROUND: return scale_ * round_fwd(x); break;

        case RELU_DST: return scale_ * relu_fwd(x, alpha_); break;
        case LOGISTIC_DST: return scale_ * logistic_fwd(x); break;
        case TANH_DST: return scale_ * tanh_fwd(x); break;
        case ELU_DST: return scale_ * elu_fwd(x, alpha_); break;
        case SQRT_DST: return scale_ * sqrt_fwd(x); break;
        case EXP_DST: return scale_ * exp_fwd(x); break;

        default: return x; break;
    }
}

float fwd_eltwise(float x, float alpha_, float beta_, float scale_) {
#ifdef ELTWISE_ALG
    return fwd_eltwise_common(ELTWISE_ALG, x, alpha_, beta_, scale_);
#else
    return x;
#endif
}

float bwd_eltwise(float x, float y, float alpha_, float beta_) {
#ifdef ELTWISE_ALG
    switch (ELTWISE_ALG) {
        case RELU: return relu_bwd(x, y, alpha_); break;
        case LINEAR: return linear_bwd(x, alpha_); break;
        case BOUNDED_RELU: return bounded_relu_bwd(x, y, alpha_); break;
        case SOFT_RELU: return soft_relu_bwd(x, y); break;
        case LOGISTIC: return logistic_bwd(x, y); break;
        case TANH: return tanh_bwd(x, y); break;
        case ELU: return elu_bwd(x, y, alpha_); break;
        case SQUARE: return square_bwd(x, y); break;
        case SQRT: return sqrt_bwd(x, y); break;
        case ABS: return abs_bwd(x, y); break;
        case EXP: return exp_bwd(x, y); break;
        case GELU_TANH: return gelu_tanh_bwd(x, y); break;
        case SWISH: return swish_bwd(x, y, alpha_); break;
        case LOG: return log_bwd(x, y); break;
        case CLIP: return clip_bwd(x, y, alpha_, beta_); break;
        case POW: return pow_bwd(x, y, alpha_, beta_); break;
        case GELU_ERF: return gelu_erf_bwd(x, y); break;

        case RELU_DST: return relu_bwd_use_dst(x, y, alpha_); break;
        case LOGISTIC_DST: return logistic_bwd_use_dst(x, y); break;
        case TANH_DST: return tanh_bwd_use_dst(x, y); break;
        case ELU_DST: return elu_bwd_use_dst(x, y, alpha_); break;
        case SQRT_DST: return sqrt_bwd_use_dst(x, y); break;
        case EXP_DST: return exp_bwd_use_dst(x, y); break;

        default: return x; break;
    }
#else
    return x;
#endif
}

#endif // WITH_ELTWISE

#endif // GPU_OCL_OCL_ELTWISE_H
