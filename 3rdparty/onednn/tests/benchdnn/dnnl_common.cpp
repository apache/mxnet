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

#include <assert.h>
#include "oneapi/dnnl/dnnl.h"

#include "dnnl_common.hpp"
#include "dnnl_memory.hpp"

float round_to_nearest_representable(dnnl_data_type_t dt, float value) {
    switch (dt) {
        case dnnl_f32: break;
        case dnnl_bf16: value = (float)dnnl::impl::bfloat16_t(value); break;
        case dnnl_f16: value = (float)dnnl::impl::float16_t(value); break;
        case dnnl_s32:
        case dnnl_s8:
        case dnnl_u8: value = maybe_saturate(dt, value); break;
        default: SAFE_V(FAIL);
    }

    return value;
}

// Engine kind used to run oneDNN primitives for testing
dnnl_engine_kind_t engine_tgt_kind = dnnl_cpu;

args_t &args_t::set(int arg, const dnn_mem_t &mem) {
    args_.emplace_back(arg, &mem);
    return *this;
}

args_t &args_t::set(
        const std::vector<int> &args, const std::vector<dnn_mem_t> &mems) {
    assert(args.size() == mems.size());
    for (size_t i = 0; i < mems.size(); ++i)
        args_.emplace_back(args[i], &mems[i]);
    return *this;
}

// Unmap before passing the memory to execute
void execute_unmap_args(
        const args_t &args, std::vector<dnnl_exec_arg_t> &dnnl_args) {
    dnnl_args.resize(args.size());
    for (int i = 0; i < args.size(); ++i) {
        if (args.dnn_mem(i).is_mapped()) args.dnn_mem(i).unmap();

        dnnl_args[i].arg = args.arg(i);
        dnnl_args[i].memory = args.dnn_mem(i).m_;
    }
}

// Map the memory back after execute
void execute_map_args(const args_t &args) {
    for (int i = 0; i < args.size(); ++i)
        if (!args.dnn_mem(i).is_mapped()) args.dnn_mem(i).map();
}

int execute_and_wait(dnnl_primitive_t prim, const args_t &args) {
    const_dnnl_primitive_desc_t pd;
    dnnl_engine_t engine;

    DNN_SAFE(dnnl_primitive_get_primitive_desc(prim, &pd), CRIT);

    DNN_SAFE(
            dnnl_primitive_desc_query(pd, dnnl_query_engine, 0, &engine), CRIT);

    stream_t stream(engine);
    std::vector<dnnl_exec_arg_t> dnnl_args;
    execute_unmap_args(args, dnnl_args);

    DNN_SAFE(dnnl_primitive_execute(
                     prim, stream, (int)dnnl_args.size(), dnnl_args.data()),
            CRIT);
    DNN_SAFE(dnnl_stream_wait(stream), CRIT);

    execute_map_args(args);

    if (bench_mode & CORR) {
        for (int i = 0; i < args.size(); ++i) {
            SAFE(check_zero_padding(args.dnn_mem(i), args.arg(i)), WARN);
        }
    }

    return OK;
}

inline bool should_stop(const benchdnn_timer_t &t) {
    const bool stop = false
            || (fix_times_per_prb && t.times() >= fix_times_per_prb)
            || (!fix_times_per_prb && t.total_ms() >= max_ms_per_prb
                    && t.times() >= min_times_per_prb);
    return stop;
}

inline int measure_perf_individual(benchdnn_timer_t &t, dnnl_stream_t stream,
        dnnl_primitive_t prim, std::vector<dnnl_exec_arg_t> &dnnl_args) {
    t.reset();
    while (true) {
        DNN_SAFE(dnnl_primitive_execute(
                         prim, stream, (int)dnnl_args.size(), dnnl_args.data()),
                WARN);
        t.stamp();
        if (should_stop(t)) break;
    }
    return OK;
}

inline int measure_perf_aggregate(benchdnn_timer_t &t, dnnl_stream_t stream,
        dnnl_primitive_t prim, std::vector<dnnl_exec_arg_t> &dnnl_args) {
    const int max_batch_times = 10000;

    // Warm-up run
    t.reset();
    DNN_SAFE(dnnl_primitive_execute(
                     prim, stream, (int)dnnl_args.size(), dnnl_args.data()),
            WARN);
    DNN_SAFE(dnnl_stream_wait(stream), WARN);
    t.stamp();

    int cur_batch_times
            = fix_times_per_prb ? fix_times_per_prb : min_times_per_prb;
    --cur_batch_times;

    while (true) {
        for (int i = 0; i < cur_batch_times; i++) {
            DNN_SAFE(dnnl_primitive_execute(prim, stream, (int)dnnl_args.size(),
                             dnnl_args.data()),
                    WARN);
        }
        DNN_SAFE(dnnl_stream_wait(stream), WARN);
        t.stamp(cur_batch_times);

        if (should_stop(t)) break;

        // Adjust cur_batch_times after the first batch run
        if (t.times() == cur_batch_times + 1) {
            double ms_min = t.ms(benchdnn_timer_t::min);
            // Heuristic: try to use ~5 batch runs for the whole benchmark
            int batch_times_heuristic = (ms_min == 0.0)
                    ? INT_MAX
                    : MAX2(1,
                            (int)((max_ms_per_prb - t.total_ms()) / ms_min
                                    / 5));
            cur_batch_times = MIN2(max_batch_times, batch_times_heuristic);
        }
    }
    return OK;
}

int measure_perf(benchdnn_timer_t &t, dnnl_primitive_t prim, args_t &args) {
    dnnl_engine_kind_t engine_kind;
    DNN_SAFE(dnnl_engine_get_kind(get_test_engine(), &engine_kind), CRIT);

    int ret = OK;
    if (bench_mode & PERF) {
        stream_t stream(get_test_engine());
        std::vector<dnnl_exec_arg_t> dnnl_args;
        execute_unmap_args(args, dnnl_args);

        // For CPU: measure indiividual iterations
        // For GPU: measure iterations in batches to hide driver overhead
        if (engine_kind == dnnl_cpu)
            ret = measure_perf_individual(t, stream, prim, dnnl_args);
        else
            ret = measure_perf_aggregate(t, stream, prim, dnnl_args);

        if (ret == OK) execute_map_args(args);
    }
    return ret;
}

void maybe_prepare_runtime_scales(dnn_mem_t &scales_m, const attr_t &attr,
        int64_t scale_cnt, const float *scales) {
    if (!attr.oscale.runtime) return;

    const int64_t count
            = attr.oscale.policy == policy_t::COMMON ? 1 : scale_cnt;

    scales_m = dnn_mem_t(1, &count, dnnl_f32, tag::x, get_test_engine());
    for (int64_t c = 0; c < count; ++c)
        ((float *)scales_m)[c] = scales[c];
}

void maybe_prepare_runtime_zero_points(dnn_mem_t &zero_points_m,
        const attr_t &attr, int arg, int64_t count,
        const int32_t *zero_points) {
    if (!attr.zero_points.runtime(arg)) return;

    const auto e = attr.zero_points.get(arg);
    const int64_t cnt = e.policy == policy_t::COMMON ? 1 : count;

    zero_points_m = dnn_mem_t(1, &cnt, dnnl_s32, tag::x, get_test_engine());
    for (int64_t c = 0; c < cnt; ++c)
        ((int32_t *)zero_points_m)[c] = zero_points[c];
}

void maybe_prepare_runtime_zero_points(
        dnn_mem_t &zero_points_m, const attr_t &attr, int arg) {
    const auto e = attr.zero_points.get(arg);
    maybe_prepare_runtime_zero_points(zero_points_m, attr, arg, 1, &(e.value));
}

bool check_md_consistency_with_tag(
        const dnnl_memory_desc_t &md, const std::string &tag) {
    dnnl_memory_desc_t md_new_tag;
    SAFE(init_md(&md_new_tag, md.ndims, md.dims, md.data_type, tag), CRIT);
    return dnnl_memory_desc_equal(&md_new_tag, &md);
}

void check_known_skipped_case_common(
        const std::vector<dnnl_data_type_t> &v_dt, dir_t dir, res_t *r) {
    static auto isa = dnnl_get_effective_cpu_isa();
    const bool has_bf16_support
            = (engine_tgt_kind == dnnl_cpu && isa >= dnnl_cpu_isa_avx512_core)
            || engine_tgt_kind == dnnl_gpu;

    // rely on dnnl_cpu_isa_t enum order where AVX512_MIC < AVX512_CORE
    for (const auto &i_dt : v_dt) {
        // bf16 is supported on AVX512-CORE+
        if (!has_bf16_support && i_dt == dnnl_bf16) {
            r->state = SKIPPED, r->reason = DATA_TYPE_NOT_SUPPORTED;
            break;
        }
        // f16 is supported on GPU only
        if (i_dt == dnnl_f16 && engine_tgt_kind != dnnl_gpu) {
            r->state = SKIPPED, r->reason = DATA_TYPE_NOT_SUPPORTED;
            break;
        }
        // f16 is supported for inference only
        if (i_dt == dnnl_f16 && (dir & FLAG_BWD)) {
            r->state = SKIPPED, r->reason = DATA_TYPE_NOT_SUPPORTED;
            break;
        }
    }
}
