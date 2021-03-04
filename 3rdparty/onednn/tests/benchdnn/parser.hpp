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

#ifndef PARSER_HPP
#define PARSER_HPP

#include <stdio.h>
#include <stdlib.h>
#include <string>

#include "dnnl_memory.hpp"
#include "oneapi/dnnl/dnnl.h"

namespace parser {

extern bool last_parsed_is_problem;
static const auto eol = std::string::npos;

static inline std::string get_pattern(const std::string &option_name) {
    return std::string("--") + option_name + std::string("=");
}

template <typename T, typename F>
static bool parse_vector_str(T &vec, const T &def, F process_func,
        const char *str, char delimeter = ',') {
    const std::string s = str;
    if (s.empty()) return vec = def, true;

    vec.clear();
    for (size_t pos_st = 0, pos_en = s.find_first_of(delimeter, pos_st); true;
            pos_st = pos_en + 1, pos_en = s.find_first_of(delimeter, pos_st)) {
        vec.push_back(process_func(s.substr(pos_st, pos_en - pos_st).c_str()));
        if (pos_en == eol) break;
    }
    return true;
}

template <typename T, typename F>
static bool parse_multivector_str(std::vector<T> &vec,
        const std::vector<T> &def, F process_func, const char *str,
        char vector_delim = ',', char element_delim = ':') {
    auto process_subword = [&](const char *word) {
        T v, empty_def_v; // defualt value is not expected to be set here
        // parse vector elements separated by @p element_delim
        parse_vector_str(v, empty_def_v, process_func, word, element_delim);
        return v;
    };

    // parse full vector separated by @p vector_delim
    return parse_vector_str(vec, def, process_subword, str, vector_delim);
}

template <typename T, typename F>
static bool parse_vector_option(T &vec, const T &def, F process_func,
        const char *str, const std::string &option_name) {
    const std::string pattern = get_pattern(option_name);
    if (pattern.find(str, 0, pattern.size()) == eol) return false;
    return parse_vector_str(vec, def, process_func, str + pattern.size());
}

template <typename T, typename F>
static bool parse_multivector_option(std::vector<T> &vec,
        const std::vector<T> &def, F process_func, const char *str,
        const std::string &option_name) {
    const std::string pattern = get_pattern(option_name);
    if (pattern.find(str, 0, pattern.size()) == eol) return false;
    return parse_multivector_str(vec, def, process_func, str + pattern.size());
}

template <typename T, typename F>
static bool parse_single_value_option(T &val, const T &def_val, F process_func,
        const char *str, const std::string &option_name) {
    const std::string pattern = get_pattern(option_name);
    if (pattern.find(str, 0, pattern.size()) == eol) return false;
    str = str + pattern.size();
    if (*str == '\0') return val = def_val, true;
    return val = process_func(str), true;
}

template <typename T, typename F>
static bool parse_cfg(T &vec, const T &def, F process_func, const char *str,
        const std::string &option_name = "cfg") {
    return parse_vector_option(vec, def, process_func, str, option_name);
}

template <typename T, typename F>
static bool parse_alg(T &vec, const T &def, F process_func, const char *str,
        const std::string &option_name = "alg") {
    return parse_vector_option(vec, def, process_func, str, option_name);
}

template <typename T>
bool parse_subattr(
        std::vector<T> &vec, const char *str, const std::string &option_name) {
    std::vector<T> def {T()};
    auto parse_subattr_func = [](const std::string &s) {
        T v;
        SAFE_V(v.from_str(s));
        return v;
    };
    return parse_vector_option(vec, def, parse_subattr_func, str, option_name);
}

template <typename S>
bool parse_reset(S &settings, const char *str,
        const std::string &option_name = "reset") {
    const std::string pattern = get_pattern(option_name);
    if (pattern.find(str, 0, pattern.size() - 1) == eol) return false;
    settings.reset();
    return true;
}

// vector types
bool parse_dir(std::vector<dir_t> &dir, const std::vector<dir_t> &def_dir,
        const char *str, const std::string &option_name = "dir");

bool parse_dt(std::vector<dnnl_data_type_t> &dt,
        const std::vector<dnnl_data_type_t> &def_dt, const char *str,
        const std::string &option_name = "dt");

bool parse_multi_dt(std::vector<std::vector<dnnl_data_type_t>> &dt,
        const std::vector<std::vector<dnnl_data_type_t>> &def_dt,
        const char *str, const std::string &option_name = "sdt");

bool parse_tag(std::vector<std::string> &tag,
        const std::vector<std::string> &def_tag, const char *str,
        const std::string &option_name = "tag");

bool parse_multi_tag(std::vector<std::vector<std::string>> &tag,
        const std::vector<std::vector<std::string>> &def_tag, const char *str,
        const std::string &option_name = "stag");

bool parse_mb(std::vector<int64_t> &mb, const std::vector<int64_t> &def_mb,
        const char *str, const std::string &option_name = "mb");

bool parse_attr(
        attr_t &attr, const char *str, const std::string &option_name = "attr");

bool parse_attr_oscale(std::vector<attr_t::scale_t> &oscale, const char *str,
        const std::string &option_name = "attr-oscale");

bool parse_attr_post_ops(std::vector<attr_t::post_ops_t> &po, const char *str,
        const std::string &option_name = "attr-post-ops");

bool parse_attr_scales(std::vector<attr_t::arg_scales_t> &scales,
        const char *str, const std::string &option_name = "attr-scales");

bool parse_attr_zero_points(std::vector<attr_t::zero_points_t> &zp,
        const char *str, const std::string &option_name = "attr-zero-points");

bool parse_attr_scratchpad_mode(
        std::vector<dnnl_scratchpad_mode_t> &scratchpad_mode,
        const std::vector<dnnl_scratchpad_mode_t> &def_scratchpad_mode,
        const char *str, const std::string &option_name = "attr-scratchpad");

bool parse_axis(std::vector<int> &axis, const std::vector<int> &def_axis,
        const char *str, const std::string &option_name = "axis");

bool parse_test_pattern_match(const char *&match, const char *str,
        const std::string &option_name = "match");

bool parse_inplace(std::vector<bool> &inplace,
        const std::vector<bool> &def_inplace, const char *str,
        const std::string &option_name = "inplace");

bool parse_skip_nonlinear(std::vector<bool> &skip,
        const std::vector<bool> &def_skip, const char *str,
        const std::string &option_name = "skip-nonlinear");

bool parse_trivial_strides(std::vector<bool> &ts,
        const std::vector<bool> &def_ts, const char *str,
        const std::string &option_name = "trivial-strides");

bool parse_scale_policy(std::vector<policy_t> &policy,
        const std::vector<policy_t> &def_policy, const char *str,
        const std::string &option_name = "scaling");

// plain types
bool parse_perf_template(const char *&pt, const char *pt_def,
        const char *pt_csv, const char *str,
        const std::string &option_name = "perf-template");

bool parse_batch(const bench_f bench, const char *str,
        const std::string &option_name = "batch");

// dim_t type
void parse_dims(dims_t &dims, const char *str);

void parse_multi_dims(std::vector<dims_t> &dims, const char *str);

// service functions
bool parse_bench_settings(const char *str);

void catch_unknown_options(const char *str);

int parse_last_argument();
} // namespace parser

#endif
