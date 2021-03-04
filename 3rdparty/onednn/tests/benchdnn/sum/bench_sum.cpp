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

#include "sum/sum.hpp"

namespace sum {

void check_correctness(const settings_t &s) {
    for_(const auto &i_sdt : s.sdt)
    for_(const auto &i_ddt : s.ddt)
    for_(const auto &i_stag_ : s.stag)
    for_(const auto &i_dtag : s.dtag)
    for (const auto &i_scratchpad_mode : s.scratchpad_mode) {
        // broadcast tag if needed
        auto i_stag = i_stag_;
        if (i_stag.size() == 1) i_stag.assign(i_sdt.size(), i_stag[0]);

        if (i_sdt.size() != i_stag.size()) // expect 1:1 match of dt and tag
            SAFE_V(FAIL);

        attr_t attr;
        attr.insert(i_scratchpad_mode);

        for (const auto &i_scales : s.scales) {
            // expect either single scale value, or 1:1 match of dt and scale
            if (i_scales.size() != 1 && i_scales.size() != i_sdt.size())
                SAFE_V(FAIL);

            const prb_t prb(
                    s.dims, i_sdt, i_ddt, i_stag, i_dtag, i_scales, attr);
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
    driver_name = "sum";
    using namespace parser;
    static settings_t s;
    static const settings_t def {};
    for (; argc > 0; --argc, ++argv) {
        const bool parsed_options = parse_bench_settings(argv[0])
                || parse_batch(bench, argv[0])
                || parse_multi_dt(s.sdt, def.sdt, argv[0])
                || parse_dt(s.ddt, def.ddt, argv[0], "ddt")
                || parse_multi_tag(s.stag, def.stag, argv[0])
                || parse_tag(s.dtag, def.dtag, argv[0], "dtag")
                || parse_multivector_option(
                        s.scales, def.scales, atof, argv[0], "scales")
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

} // namespace sum
