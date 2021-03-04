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

#include <string.h>

#include <sstream>

#include "dnnl_common.hpp"
#include "dnnl_memory.hpp"
#include "parser.hpp"

#include "reorder.hpp"

namespace reorder {

void check_correctness(const settings_t &s) {
    for_(const auto &i_sdt : s.sdt)
    for_(const auto &i_ddt : s.ddt)
    for_(const auto &i_stag : s.stag)
    for_(const auto &i_dtag : s.dtag)
    for_(const auto &i_oflag : s.oflag)
    for_(const auto &i_alg : s.alg)
    for_(const auto &i_cross_engine : s.cross_engine)
    for_(const auto &i_oscale : s.oscale)
    for_(const auto &i_zero_points : s.zero_points)
    for_(const auto &i_post_ops : s.post_ops)
    for_(const auto &i_scratchpad_mode : s.scratchpad_mode)
    for (auto i_runtime_dim_mask : s.runtime_dim_mask) {
        reorder_conf_t reorder_conf {s.dims, i_stag, i_dtag};
        dt_conf_t iconf = dt2cfg(i_sdt);
        dt_conf_t oconf = dt2cfg(i_ddt);

        attr_t attr;
        attr.insert(i_oscale);
        attr.insert(i_zero_points);
        attr.insert(i_post_ops);
        attr.insert(i_scratchpad_mode);
        handle_legacy_attr(attr, s.attr);

        if (attr.oscale.policy == policy_t::PER_OC) {
            fprintf(stderr,
                    "ERROR: reorder driver: `per_oc` policy is not supported "
                    "due to potential ambiguity. Please use one of `per_dim_0` "
                    "or `per_dim_1` policies.\n"),
                    fflush(stderr);
            SAFE_V(FAIL);
        }
        if (i_cross_engine != NONE && engine_tgt_kind == dnnl_cpu) {
            fprintf(stderr,
                    "ERROR: reorder driver: `cpu` engine does not support "
                    "other values but `none`.\n"),
                    fflush(stderr);
            SAFE_V(FAIL);
        }

        std::vector<float> attr_scale = {attr.oscale.scale};
        auto &scale = attr.oscale.scale == 0 ? s.def_scale : attr_scale;

        for (const auto &i_scale : scale) {
            uint64_t oflag = FLAG_NONE;
            for (const auto &f : i_oflag) {
                oflag |= f;
            }

            const prb_t prb(reorder_conf, iconf, oconf, attr, i_alg, oflag,
                    i_cross_engine, i_runtime_dim_mask, i_scale);
            std::stringstream ss;
            ss << prb;
            const std::string cpp_pstr = ss.str();
            const char *pstr = cpp_pstr.c_str();
            BENCHDNN_PRINT(1, "run: %s\n", pstr);

            res_t res {};
            int status = doit(&prb, &res);

            bool want_perf_report = false;
            parse_result(res, want_perf_report, status, pstr);

            if (want_perf_report && bench_mode & PERF) {
                perf_report_t pr(s.perf_template);
                pr.report(&prb, &res, pstr);
            }

            benchdnn_stat.tests++;
        }
    }
}

int bench(int argc, char **argv) {
    driver_name = "reorder";
    using namespace parser;
    static settings_t s;
    static const settings_t def {};
    for (; argc > 0; --argc, ++argv) {
        const bool parsed_options = parse_bench_settings(argv[0])
                || parse_batch(bench, argv[0])
                || parse_dt(s.sdt, def.sdt, argv[0], "sdt")
                || parse_dt(s.ddt, def.ddt, argv[0], "ddt")
                || parse_tag(s.stag, def.stag, argv[0], "stag")
                || parse_tag(s.dtag, def.dtag, argv[0], "dtag")
                || parse_multivector_option(
                        s.oflag, def.oflag, str2flag, argv[0], "oflag")
                || parse_vector_option(s.runtime_dim_mask, def.runtime_dim_mask,
                        atoi, argv[0], "runtime-dim-mask")
                || parse_alg(s.alg, def.alg, str2alg, argv[0])
                || parse_vector_option(
                        s.def_scale, def.def_scale, atof, argv[0], "def-scales")
                || parse_vector_option(s.cross_engine, def.cross_engine,
                        str2cross_engine, argv[0], "cross-engine")
                || parse_attr(s.attr, argv[0])
                || parse_attr_oscale(s.oscale, argv[0])
                || parse_attr_zero_points(s.zero_points, argv[0])
                || parse_attr_post_ops(s.post_ops, argv[0])
                || parse_attr_scratchpad_mode(
                        s.scratchpad_mode, def.scratchpad_mode, argv[0])
                || parse_perf_template(s.perf_template, s.perf_template_def,
                        s.perf_template_csv, argv[0])
                || parse_reset(s, argv[0]);
        if (!parsed_options) {
            catch_unknown_options(argv[0]);

            parse_dims(s.dims, argv[0]);
            check_correctness(s);
        }
    }

    return parse_last_argument();
}

} // namespace reorder
