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

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>

#include "dnnl_memory.hpp"
#include "oneapi/dnnl/dnnl.h"
#include "parser.hpp"

namespace parser {

bool last_parsed_is_problem = false;

// vector types
bool parse_dir(std::vector<dir_t> &dir, const std::vector<dir_t> &def_dir,
        const char *str, const std::string &option_name /* = "dir"*/) {
    return parse_vector_option(dir, def_dir, str2dir, str, option_name);
}

bool parse_dt(std::vector<dnnl_data_type_t> &dt,
        const std::vector<dnnl_data_type_t> &def_dt, const char *str,
        const std::string &option_name /* = "dt"*/) {
    return parse_vector_option(dt, def_dt, str2dt, str, option_name);
}

bool parse_multi_dt(std::vector<std::vector<dnnl_data_type_t>> &dt,
        const std::vector<std::vector<dnnl_data_type_t>> &def_dt,
        const char *str, const std::string &option_name /* = "sdt"*/) {
    return parse_multivector_option(dt, def_dt, str2dt, str, option_name);
}

bool parse_tag(std::vector<std::string> &tag,
        const std::vector<std::string> &def_tag, const char *str,
        const std::string &option_name /* = "tag"*/) {
    auto ret_string = [](const char *str) { return std::string(str); };
    bool ok = parse_vector_option(tag, def_tag, ret_string, str, option_name);
    if (!ok) return false;

    for (size_t i = 0; i < tag.size(); i++) {
        if (check_tag(tag[i], allow_enum_tags_only) != OK) {
            if (allow_enum_tags_only && check_tag(tag[i]) == OK) {
                fprintf(stderr,
                        "%s driver: ERROR: tag `%s` is valid but not found "
                        "in dnnl::memory::format_tag, use "
                        "--allow-enum-tags-only=0 if you want to test this "
                        "tag.\n",
                        driver_name, tag[i].c_str());
            } else {
                fprintf(stderr,
                        "%s driver: ERROR: unknown tag: `%s`, exiting...\n",
                        driver_name, tag[i].c_str());
            }
            exit(2);
        }
    }
    return true;
}

bool parse_multi_tag(std::vector<std::vector<std::string>> &tag,
        const std::vector<std::vector<std::string>> &def_tag, const char *str,
        const std::string &option_name /* = "stag"*/) {
    auto ret_string = [](const char *str) { return std::string(str); };
    return parse_multivector_option(tag, def_tag, ret_string, str, option_name);
}

bool parse_mb(std::vector<int64_t> &mb, const std::vector<int64_t> &def_mb,
        const char *str, const std::string &option_name /* = "mb"*/) {
    return parse_vector_option(mb, def_mb, atoi, str, option_name);
}

bool parse_attr(attr_t &attr, const char *str,
        const std::string &option_name /* = "attr"*/) {
    const std::string pattern = get_pattern(option_name);
    if (pattern.find(str, 0, pattern.size()) == eol) return false;
    static bool notice_printed = false;
    if (!notice_printed) {
        BENCHDNN_PRINT(0, "%s\n",
                "WARNING (DEPRECATION NOTICE): `--attr` option is deprecated. "
                "Please use one of `--attr-oscale`, `--attr-post-ops`, "
                "`--attr-scales` or `--attr-zero-points` to specify attributes "
                "specific part for a problem. New options support mixing and "
                "will iterate over all possible combinations.");
        notice_printed = true;
    }
    SAFE_V(str2attr(&attr, str + pattern.size()));
    return true;
}

bool parse_attr_oscale(std::vector<attr_t::scale_t> &oscale, const char *str,
        const std::string &option_name /* = "attr-oscale"*/) {
    return parse_subattr(oscale, str, option_name);
}

bool parse_attr_post_ops(std::vector<attr_t::post_ops_t> &po, const char *str,
        const std::string &option_name /* = "attr-post-ops"*/) {
    return parse_subattr(po, str, option_name);
}

bool parse_attr_scales(std::vector<attr_t::arg_scales_t> &scales,
        const char *str, const std::string &option_name /* = "attr-scales"*/) {
    return parse_subattr(scales, str, option_name);
}

bool parse_attr_zero_points(std::vector<attr_t::zero_points_t> &zp,
        const char *str,
        const std::string &option_name /* = "attr-zero-points"*/) {
    return parse_subattr(zp, str, option_name);
}

bool parse_attr_scratchpad_mode(
        std::vector<dnnl_scratchpad_mode_t> &scratchpad_mode,
        const std::vector<dnnl_scratchpad_mode_t> &def_scratchpad_mode,
        const char *str,
        const std::string &option_name /* = "attr-scratchpad"*/) {
    return parse_vector_option(scratchpad_mode, def_scratchpad_mode,
            str2scratchpad_mode, str, option_name);
}

bool parse_axis(std::vector<int> &axis, const std::vector<int> &def_axis,
        const char *str, const std::string &option_name /* = "axis"*/) {
    return parse_vector_option(axis, def_axis, atoi, str, option_name);
}

bool parse_test_pattern_match(const char *&match, const char *str,
        const std::string &option_name /* = "match"*/) {
    const std::string pattern = get_pattern(option_name);
    if (pattern.find(str, 0, pattern.size()) != eol) {
        match = str + pattern.size();
        return true;
    }
    return false;
}

bool parse_inplace(std::vector<bool> &inplace,
        const std::vector<bool> &def_inplace, const char *str,
        const std::string &option_name /* = "inplace"*/) {
    return parse_vector_option(
            inplace, def_inplace, str2bool, str, option_name);
}

bool parse_skip_nonlinear(std::vector<bool> &skip,
        const std::vector<bool> &def_skip, const char *str,
        const std::string &option_name /* = "skip-nonlinear"*/) {
    return parse_vector_option(skip, def_skip, str2bool, str, option_name);
}

bool parse_trivial_strides(std::vector<bool> &ts,
        const std::vector<bool> &def_ts, const char *str,
        const std::string &option_name /* = "trivial-strides"*/) {
    return parse_vector_option(ts, def_ts, str2bool, str, option_name);
}

bool parse_scale_policy(std::vector<policy_t> &policy,
        const std::vector<policy_t> &def_policy, const char *str,
        const std::string &option_name /*= "scaling"*/) {
    return parse_vector_option(
            policy, def_policy, attr_t::str2policy, str, option_name);
}

// plain types
bool parse_perf_template(const char *&pt, const char *pt_def,
        const char *pt_csv, const char *str,
        const std::string &option_name /* = "perf-template"*/) {
    const std::string pattern = get_pattern(option_name);
    if (pattern.find(str, 0, pattern.size()) != eol) {
        const std::string csv_pattern = "csv";
        const std::string def_pattern = "def";
        str += pattern.size();
        if (csv_pattern.find(str, 0, csv_pattern.size()) != eol)
            pt = pt_csv;
        else if (def_pattern.find(str, 0, def_pattern.size()) != eol)
            pt = pt_def;
        else
            pt = str;
        return true;
    }
    return false;
}

bool parse_batch(const bench_f bench, const char *str,
        const std::string &option_name /* = "batch"*/) {
    const std::string pattern = get_pattern(option_name);
    if (pattern.find(str, 0, pattern.size()) != eol) {
        SAFE_V(batch(str + pattern.size(), bench));
        return true;
    }
    return false;
}

// dim_t type
static bool parse_dims_as_desc(dims_t &dims, const char *str) {
    dims.clear();

    auto mstrtol = [](const char *nptr, char **endptr) {
        return strtol(nptr, endptr, 10);
    };

#define CASE_NN(p, c, cvfunc) \
    do { \
        if (!strncmp(p, str, strlen(p))) { \
            ok = 1; \
            str += strlen(p); \
            char *end_s; \
            int64_t c = cvfunc(str, &end_s); \
            str += (end_s - str); \
            if ((c) < 0) return false; \
            dims.push_back(c); \
        } \
    } while (0)
#define CASE_N(c, cvfunc) CASE_NN(#c, c, cvfunc)
    while (*str) {
        int ok = 0;
        CASE_N(mb, mstrtol);
        CASE_N(ic, mstrtol);
        CASE_N(id, mstrtol);
        CASE_N(ih, mstrtol);
        CASE_N(iw, mstrtol);
        if (*str == '_') ++str;
        if (!ok) return false;
    }
#undef CASE_NN
#undef CASE_N

    return true;
}

void parse_dims(dims_t &dims, const char *str) {
    // func below provides fragile compatibility of verbose output for v0
    // eltwise and softmax. Remove once we stop supporting v0.
    if (parse_dims_as_desc(dims, str)) return;
    parse_vector_str(dims, dims_t(), atoi, str, 'x');
}

void parse_multi_dims(std::vector<dims_t> &dims, const char *str) {
    std::vector<dims_t> def {dims_t()};
    parse_multivector_str(dims, def, atoi, str, ':', 'x');
}

// service functions
static bool parse_bench_mode(
        const char *str, const std::string &option_name = "mode") {
    return parse_single_value_option(
            bench_mode, CORR, str2bench_mode, str, option_name);
}

static bool parse_max_ms_per_prb(
        const char *str, const std::string &option_name = "max-ms-per-prb") {
    if (parse_single_value_option(max_ms_per_prb, 3e3, atof, str, option_name))
        return max_ms_per_prb = MAX2(100, MIN2(max_ms_per_prb, 60e3)), true;
    return false;
}

static bool parse_fix_times_per_prb(
        const char *str, const std::string &option_name = "fix-times-per-prb") {
    if (parse_single_value_option(fix_times_per_prb, 0, atoi, str, option_name))
        return fix_times_per_prb = MAX2(0, fix_times_per_prb), true;
    return false;
}

static bool parse_verbose(
        const char *str, const std::string &option_name = "verbose") {
    const std::string pattern("-v"); // check short option first
    if (pattern.find(str, 0, pattern.size()) != eol) {
        verbose = atoi(str + pattern.size());
        return true;
    }
    return parse_single_value_option(verbose, 0, atoi, str, option_name);
}

static bool parse_engine_kind(
        const char *str, const std::string &option_name = "engine") {
    return parse_single_value_option(
            engine_tgt_kind, dnnl_cpu, str2engine_kind, str, option_name);
}

static bool parse_fast_ref_gpu(
        const char *str, const std::string &option_name = "fast-ref-gpu") {
    return parse_single_value_option(
            fast_ref_gpu, true, str2bool, str, option_name);
}

static bool parse_canonical(
        const char *str, const std::string &option_name = "canonical") {
    return parse_single_value_option(
            canonical, false, str2bool, str, option_name);
}

static bool parse_mem_check(
        const char *str, const std::string &option_name = "mem-check") {
    return parse_single_value_option(
            mem_check, true, str2bool, str, option_name);
}

static bool parse_skip_impl(
        const char *str, const std::string &option_name = "skip-impl") {
    const std::string pattern = get_pattern(option_name);
    if (pattern.find(str, 0, pattern.size()) == eol) return false;
    skip_impl = std::string(str + pattern.size());
    return true;
}

static bool parse_allow_enum_tags_only(const char *str,
        const std::string &option_name = "allow-enum-tags-only") {
    return parse_single_value_option(
            allow_enum_tags_only, true, str2bool, str, option_name);
}

bool parse_bench_settings(const char *str) {
    last_parsed_is_problem = false; // if start parsing, expect an option

    return parse_bench_mode(str) || parse_max_ms_per_prb(str)
            || parse_fix_times_per_prb(str) || parse_verbose(str)
            || parse_engine_kind(str) || parse_fast_ref_gpu(str)
            || parse_canonical(str) || parse_mem_check(str)
            || parse_skip_impl(str) || parse_allow_enum_tags_only(str);
}

void catch_unknown_options(const char *str) {
    last_parsed_is_problem = true; // if reached, means problem parsing

    const std::string pattern = "--";
    if (pattern.find(str, 0, pattern.size()) != eol) {
        fprintf(stderr, "%s driver: ERROR: unknown option: `%s`, exiting...\n",
                driver_name, str);
        exit(2);
    }
}

int parse_last_argument() {
    if (!last_parsed_is_problem)
        fprintf(stderr,
                "%s driver: WARNING: No problem found for a given option!\n",
                driver_name);
    return OK;
}
} // namespace parser
