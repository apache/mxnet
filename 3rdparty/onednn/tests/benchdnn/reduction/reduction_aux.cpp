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

#include "reduction.hpp"

namespace reduction {

alg_t str2alg(const char *str) {
#define CASE(_alg) \
    if (!strcasecmp(STRINGIFY(_alg), str)) return _alg
    CASE(MAX);
    CASE(MIN);
    CASE(SUM);
    CASE(MUL);
    CASE(MEAN);
    CASE(NORM_LP_MAX);
    CASE(NORM_LP_SUM);
    CASE(NORM_LP_POWER_P_MAX);
    CASE(NORM_LP_POWER_P_SUM);

#undef CASE
    assert(!"unknown algorithm");
    return UNDEF;
}

const char *alg2str(alg_t alg) {
    if (alg == MAX) return "MAX";
    if (alg == MIN) return "MIN";
    if (alg == SUM) return "SUM";
    if (alg == MUL) return "MUL";
    if (alg == MEAN) return "MEAN";
    if (alg == NORM_LP_MAX) return "NORM_LP_MAX";
    if (alg == NORM_LP_SUM) return "NORM_LP_SUM";
    if (alg == NORM_LP_POWER_P_MAX) return "NORM_LP_POWER_P_MAX";
    if (alg == NORM_LP_POWER_P_SUM) return "NORM_LP_POWER_P_SUM";
    assert(!"unknown algorithm");
    return "UNDEF";
}

dnnl_alg_kind_t alg2alg_kind(alg_t alg) {
    if (alg == MAX) return dnnl_reduction_max;
    if (alg == MIN) return dnnl_reduction_min;
    if (alg == SUM) return dnnl_reduction_sum;
    if (alg == MUL) return dnnl_reduction_mul;
    if (alg == MEAN) return dnnl_reduction_mean;
    if (alg == NORM_LP_MAX) return dnnl_reduction_norm_lp_max;
    if (alg == NORM_LP_SUM) return dnnl_reduction_norm_lp_sum;
    if (alg == NORM_LP_POWER_P_MAX) return dnnl_reduction_norm_lp_power_p_max;
    if (alg == NORM_LP_POWER_P_SUM) return dnnl_reduction_norm_lp_power_p_sum;
    assert(!"unknown algorithm");
    return dnnl_alg_kind_undef;
}

std::ostream &operator<<(std::ostream &s, const prb_t &prb) {
    dump_global_params(s);
    settings_t def;

    if (canonical || prb.sdt != def.sdt[0]) s << "--sdt=" << prb.sdt << " ";
    if (canonical || prb.ddt != def.ddt[0]) s << "--ddt=" << prb.ddt << " ";
    if (canonical || prb.stag != def.stag[0]) s << "--stag=" << prb.stag << " ";
    if (canonical || prb.dtag != def.dtag[0]) s << "--dtag=" << prb.dtag << " ";
    if (canonical || prb.alg != def.alg[0])
        s << "--alg=" << alg2str(prb.alg) << " ";
    if (canonical || prb.p != def.p[0]) s << "--p=" << prb.p << " ";
    if (canonical || prb.eps != def.eps[0]) s << "--eps=" << prb.eps << " ";

    s << prb.attr;
    s << prb.src_dims << ":" << prb.dst_dims;

    return s;
}

} // namespace reduction
