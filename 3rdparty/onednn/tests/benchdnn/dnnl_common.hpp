/*******************************************************************************
* Copyright 2017-2020 Intel Corporation
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

#ifndef DNNL_COMMON_HPP
#define DNNL_COMMON_HPP

#include <stddef.h>
#include <stdint.h>

#include <cmath>
#include <vector>

#include "oneapi/dnnl/dnnl.h"
#include "src/common/bfloat16.hpp"
#include "src/common/float16.hpp"
#include "src/common/nstl.hpp"

#include "common.hpp"
#include "dnn_types.hpp"
#include "dnnl_debug.hpp"

#define for_ for

#define DNN_SAFE(f, s) \
    do { \
        dnnl_status_t status = f; \
        if (status != dnnl_success) { \
            if (s == CRIT || s == WARN) { \
                BENCHDNN_PRINT(0, "error [%s:%d]: '%s' -> %s(%d)\n", \
                        __PRETTY_FUNCTION__, __LINE__, #f, status2str(status), \
                        (int)status); \
                fflush(0); \
                if (s == CRIT) exit(2); \
            } \
            return FAIL; \
        } \
    } while (0)

#define DNN_SAFE_V(f) \
    do { \
        dnnl_status_t status = f; \
        if (status != dnnl_success) { \
            BENCHDNN_PRINT(0, "error [%s:%d]: '%s' -> %s(%d)\n", \
                    __PRETTY_FUNCTION__, __LINE__, STRINGIFY(f), \
                    status2str(status), (int)status); \
            fflush(0); \
            exit(2); \
        } \
    } while (0)

#define DNN_SAFE_CLEAN(f, s, clean) \
    do { \
        dnnl_status_t status = f; \
        if (status != dnnl_success) { \
            if (s == CRIT || s == WARN) { \
                BENCHDNN_PRINT(0, "error [%s:%d]: '%s' -> %s(%d)\n", \
                        __PRETTY_FUNCTION__, __LINE__, #f, status2str(status), \
                        (int)status); \
                fflush(0); \
                if (s == CRIT) exit(2); \
            } \
            clean(); \
            return FAIL; \
        } \
    } while (0)

/* aux */
using bfloat16_t = dnnl::impl::bfloat16_t;
using float16_t = dnnl::impl::float16_t;
template <dnnl_data_type_t>
struct prec_traits;
template <>
struct prec_traits<dnnl_bf16> {
    typedef bfloat16_t type;
};
template <>
struct prec_traits<dnnl_f16> {
    typedef float16_t type;
};
template <>
struct prec_traits<dnnl_f32> {
    typedef float type;
};
template <>
struct prec_traits<dnnl_s32> {
    typedef int32_t type;
};
template <>
struct prec_traits<dnnl_s8> {
    typedef int8_t type;
};
template <>
struct prec_traits<dnnl_u8> {
    typedef uint8_t type;
};

#define CASE_ALL(dt) \
    switch (dt) { \
        CASE(dnnl_bf16); \
        CASE(dnnl_f16); \
        CASE(dnnl_f32); \
        CASE(dnnl_s32); \
        CASE(dnnl_s8); \
        CASE(dnnl_u8); \
        default: assert(!"bad data_type"); \
    }

inline size_t sizeof_dt(dnnl_data_type_t dt) {
#define CASE(dt) \
    case dt: return sizeof(typename prec_traits<dt>::type);

    CASE_ALL(dt);

#undef CASE
    return 0;
}

/* std::numeric_limits::digits functionality */
inline int digits_dt(dnnl_data_type_t dt) {
#define CASE(dt) \
    case dt: \
        return dnnl::impl::nstl::numeric_limits< \
                typename prec_traits<dt>::type>::digits;

    CASE_ALL(dt);

#undef CASE
    return 0;
}

inline float epsilon_dt(dnnl_data_type_t dt) {
#define CASE(dt) \
    case dt: \
        return (float)dnnl::impl::nstl::numeric_limits< \
                typename prec_traits<dt>::type>::epsilon();

    CASE_ALL(dt);

#undef CASE

    return 0;
}

inline float lowest_dt(dnnl_data_type_t dt) {
#define CASE(dt) \
    case dt: \
        return (float)dnnl::impl::nstl::numeric_limits< \
                typename prec_traits<dt>::type>::lowest();

    CASE_ALL(dt);

#undef CASE

    return 0;
}

inline float max_dt(dnnl_data_type_t dt) {
#define CASE(dt) \
    case dt: \
        return (float)dnnl::impl::nstl::numeric_limits< \
                typename prec_traits<dt>::type>::max();

    CASE_ALL(dt);

#undef CASE

    return 0;
}

#undef CASE_ALL

template <dnnl_data_type_t dt>
inline float saturate_and_round(float val) {
    const float dt_max = (float)dnnl::impl::nstl::numeric_limits<
            typename prec_traits<dt>::type>::max();
    const float dt_min = (float)dnnl::impl::nstl::numeric_limits<
            typename prec_traits<dt>::type>::lowest();
    if (val > dt_max) val = dt_max;
    if (val < dt_min || (std::isnan(val) && std::signbit(val))) val = dt_min;
    return mxcsr_cvt(val);
}

inline bool is_integral_dt(dnnl_data_type_t dt) {
    return dt == dnnl_s32 || dt == dnnl_s8 || dt == dnnl_u8;
}

inline float maybe_saturate(dnnl_data_type_t dt, float value) {
    if (!is_integral_dt(dt)) return value;

    switch (dt) {
#define CASE(dt) \
    case dt: return saturate_and_round<dt>(value);
        CASE(dnnl_s32);
        CASE(dnnl_s8);
        CASE(dnnl_u8);
#undef CASE
        default: assert(!"bad data_type");
    }
    return 0;
}

float round_to_nearest_representable(dnnl_data_type_t dt, float value);

/* simplification */
extern dnnl_engine_kind_t engine_tgt_kind;

inline const char *query_impl_info(const_dnnl_primitive_desc_t pd) {
    const char *str;
    dnnl_primitive_desc_query(pd, dnnl_query_impl_info_str, 0, &str);
    return str;
}

struct dnn_mem_t;

struct args_t {
    args_t &set(int arg, const dnn_mem_t &mem);
    args_t &set(
            const std::vector<int> &args, const std::vector<dnn_mem_t> &mems);
    void clear() { args_.clear(); }

    int size() const { return (int)args_.size(); }

    int arg(int index) const { return args_[index].first; }
    const dnn_mem_t &dnn_mem(int index) const { return *args_[index].second; }

private:
    std::vector<std::pair<int, const dnn_mem_t *>> args_;
};

// Engine used to run oneDNN primitives for testing.
inline const engine_t &get_test_engine() {
    static const engine_t instance(engine_tgt_kind);
    return instance;
}

// Engine used to run reference implementations (fast-ref-gpu option).
inline const engine_t &get_cpu_engine() {
    static const engine_t instance(dnnl_cpu);
    return instance;
}

template <typename func_t, typename prb_t>
int init_prim(dnnl_primitive_t *prim, const func_t &init_pd_func, prb_t *p,
        res_t *r, dir_t dir = FLAG_FWD,
        const_dnnl_primitive_desc_t hint = nullptr) {
    int status = OK;
    dnnl_primitive_desc_t pd {};
    dnnl_primitive_t return_prim {};

    auto cleanup_pd = [&]() { dnnl_primitive_desc_destroy(pd); };
    auto cleanup_prim = [&]() { dnnl_primitive_destroy(return_prim); };
#ifndef DNNL_DISABLE_PRIMITIVE_CACHE
    // The idea is to create the requested primitive twice using
    // different engines.
    // Rationale:
    // 1. Make sure that the primitive cache is robust for the cases when:
    //   - CPU engine is re-created
    //   - GPU engine is re-created for the same device but different context
    // These 2 cases are commonly used or expected to be used in the frameworks.
    // 2. (for GPU only) Identify context dependent parts in primitive
    // implementations, e.g. if a primitive implementation contains
    // a memory_storage_t (for scales, zero points or buffers), which depends
    // on a particular engine then it should fail at execution time.

    // The first primitive creation using a temporary engine.
    engine_t engine(engine_tgt_kind);
    status = init_pd_func(engine, p, pd, r, dir, hint);
    if (status != OK) return status;
    if (r->state == SKIPPED || r->state == UNIMPLEMENTED) return OK;
    DNN_SAFE_CLEAN(dnnl_primitive_create(&return_prim, pd), WARN, cleanup_pd);
    DNN_SAFE_CLEAN(dnnl_primitive_desc_destroy(pd), WARN, cleanup_prim);
    DNN_SAFE(dnnl_primitive_destroy(return_prim), WARN);

#endif
    // The second (if the cache is enabled) primitive creation using
    // the global test engine.
    status = init_pd_func(get_test_engine(), p, pd, r, dir, hint);
    if (status != OK) return status;
    if (r->state == SKIPPED || r->state == UNIMPLEMENTED) return OK;
    // This primitive is expected to come from the cache.
    DNN_SAFE_CLEAN(dnnl_primitive_create(&return_prim, pd), WARN, cleanup_pd);
    DNN_SAFE_CLEAN(dnnl_primitive_desc_destroy(pd), WARN, cleanup_prim);
    (*prim) = return_prim;
    return OK;
}

int execute_and_wait(dnnl_primitive_t prim, const args_t &args);

int measure_perf(benchdnn_timer_t &t, dnnl_primitive_t prim, args_t &args);

void maybe_prepare_runtime_scales(dnn_mem_t &scales_m, const attr_t &attr,
        int64_t scale_cnt, const float *scales);

void maybe_prepare_runtime_zero_points(dnn_mem_t &zero_points_m,
        const attr_t &attr, int arg, int64_t count, const int32_t *zero_points);
void maybe_prepare_runtime_zero_points(
        dnn_mem_t &zero_points_m, const attr_t &attr, int arg);

bool check_md_consistency_with_tag(
        const dnnl_memory_desc_t &md, const std::string &tag);

void check_known_skipped_case_common(
        const std::vector<dnnl_data_type_t> &v_dt, dir_t dir, res_t *r);

#endif
