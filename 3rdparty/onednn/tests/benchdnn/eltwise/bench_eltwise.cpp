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

#include <sstream>

#include "dnnl_common.hpp"
#include "dnnl_memory.hpp"
#include "parser.hpp"

#include "eltwise/eltwise.hpp"

namespace eltwise {

void check_correctness(const settings_t &s) {
    for_(const auto &i_dir : s.dir)
    for_(const auto &i_dt : s.dt)
    for_(const auto &i_tag : s.tag)
    for_(const auto &i_alg : s.alg)
    for_(const auto &i_alpha : s.alpha)
    for_(const auto &i_beta : s.beta)
    for_(const auto &i_mb : s.mb)
    for_(const auto &i_post_ops : s.post_ops)
    for_(const auto &i_scratchpad_mode : s.scratchpad_mode)
    for (auto i_inplace : s.inplace) {
        bool ok = i_alg > alg_t::ELTWISE_START && i_alg < alg_t::ELTWISE_END;
        if (!ok) SAFE_V(FAIL);

        // iterator over alpha and beta (alphabetic order!)
        switch (i_alg) {
            case alg_t::ABS:
            case alg_t::EXP:
            case alg_t::EXP_DST:
            case alg_t::GELU_ERF:
            case alg_t::GELU_TANH:
            case alg_t::LOG:
            case alg_t::LOGISTIC:
            case alg_t::LOGISTIC_DST:
            case alg_t::SQRT:
            case alg_t::SQRT_DST:
            case alg_t::SQUARE:
            case alg_t::SRELU:
            case alg_t::TANH:
            case alg_t::TANH_DST:
                if (i_alpha != 0)
                    BENCHDNN_PRINT(2, "%s\n",
                            "WARNING: non-zero alpha is ignored. "
                            "Consider adding --alpha=0 to a command line.");
                if (i_beta != 0)
                    BENCHDNN_PRINT(2, "%s\n",
                            "WARNING: non-zero beta is ignored. "
                            "Consider adding --beta=0 to a command line.");
                break;
            case alg_t::BRELU:
            case alg_t::ELU:
            case alg_t::ELU_DST:
            case alg_t::RELU:
            case alg_t::RELU_DST:
            case alg_t::SWISH:
                if (i_beta != 0)
                    BENCHDNN_PRINT(2, "%s\n",
                            "WARNING: non-zero beta is ignored. "
                            "Consider adding --beta=0 to a command line.");
                break;
            default:;
        };

        attr_t attr;
        attr.insert(i_post_ops);
        attr.insert(i_scratchpad_mode);

        const prb_t prb(s.dims, i_dir, i_dt, i_tag, i_alg, i_alpha, i_beta,
                i_inplace, attr, i_mb);
        std::stringstream ss;
        ss << prb;
        const std::string cpp_pstr = ss.str();
        const char *pstr = cpp_pstr.c_str();
        BENCHDNN_PRINT(1, "run: %s\n", pstr);

        res_t res {};
        const int status = doit(&prb, &res);

        bool want_perf_report = false;
        parse_result(res, want_perf_report, status, pstr);

        if (want_perf_report && bench_mode & PERF) {
            perf_report_t pr(s.perf_template);
            pr.report(&prb, &res, pstr);
        }

        benchdnn_stat.tests++;
    }
}

int bench(int argc, char **argv) {
    driver_name = "eltwise";
    using namespace parser;
    static settings_t s;
    static const settings_t def {};
    for (; argc > 0; --argc, ++argv) {
        const bool parsed_options = parse_bench_settings(argv[0])
                || parse_batch(bench, argv[0])
                || parse_dir(s.dir, def.dir, argv[0])
                || parse_dt(s.dt, def.dt, argv[0])
                || parse_tag(s.tag, def.tag, argv[0])
                || parse_vector_option(
                        s.alpha, def.alpha, atof, argv[0], "alpha")
                || parse_vector_option(s.beta, def.beta, atof, argv[0], "beta")
                || parse_alg(
                        s.alg, def.alg, attr_t::post_ops_t::str2kind, argv[0])
                || parse_inplace(s.inplace, def.inplace, argv[0])
                || parse_mb(s.mb, def.mb, argv[0])
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
} // namespace eltwise
