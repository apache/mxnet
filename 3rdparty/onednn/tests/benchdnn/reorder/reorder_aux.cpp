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

#include <sstream>

#include "dnnl_debug.hpp"

#include "reorder/reorder.hpp"

namespace reorder {

alg_t str2alg(const char *str) {
    if (!strcasecmp("bootstrap", str)) return ALG_BOOT;
    if (!strcasecmp("reference", str)) return ALG_REF;
    assert(!"unknown algorithm");
    return ALG_REF;
}

const char *alg2str(alg_t alg) {
    switch (alg) {
        case ALG_REF: return "reference";
        case ALG_BOOT: return "bootstrap";
        default: assert(!"unknown algorithm"); return "unknown algorithm";
    }
}

uint64_t str2flag(const char *str) {
    uint64_t flag = FLAG_NONE;
    if (!strcasecmp("conv_s8s8", str)) flag |= FLAG_CONV_S8S8;
    if (!strcasecmp("gconv_s8s8", str)) flag |= FLAG_GCONV_S8S8;
    if (!strcasecmp("conv_zp_comp", str)) flag |= FLAG_CONV_ZP_COMP;
    if (!strcasecmp("gconv_zp_comp", str)) flag |= FLAG_GCONV_ZP_COMP;
    if (strcasecmp("none", str) && flag == FLAG_NONE) assert(!"unknown flag");
    return flag;
}

std::string flag2str(uint64_t flag) {
    std::stringstream s;
    bool mult_entry = false;

    if (!flag) return "none";

#define CASE(_f, _l) \
    do { \
        if (flag & (_f)) { \
            s << (mult_entry ? ":" : "") << #_l; \
            mult_entry = true; \
        } \
    } while (0)
    CASE(FLAG_CONV_S8S8, conv_s8s8);
    CASE(FLAG_GCONV_S8S8, gconv_s8s8);
    CASE(FLAG_CONV_ZP_COMP, conv_zp_comp);
    CASE(FLAG_GCONV_ZP_COMP, gconv_zp_comp);
#undef CASE

    return s.str();
}

cross_engine_t str2cross_engine(const char *str) {
    if (!strcasecmp("none", str)) return NONE;
    if (!strcasecmp("cpu2gpu", str)) return CPU2GPU;
    if (!strcasecmp("gpu2cpu", str)) return GPU2CPU;
    assert(!"unknown cross engine");
    return NONE;
}

const char *cross_engine2str(cross_engine_t cross_engine) {
    switch (cross_engine) {
        case NONE: return "none";
        case CPU2GPU: return "cpu2gpu";
        case GPU2CPU: return "gpu2cpu";
        default: assert(!"unknown cross engine"); return "unknown cross engine";
    }
}

float *prb_t::generate_oscales() {
    const attr_t::scale_t &oscale = this->attr.oscale;
    const int mask = attr_t::get_default_mask(oscale.policy);

    int64_t uniq_scales = 1;
    for (int d = 0; d < this->ndims; ++d)
        if (mask & (1 << d)) uniq_scales *= this->reorder.dims[d];

    float *scales = (float *)zmalloc(sizeof(float) * uniq_scales, 64);
    SAFE_V(scales != nullptr ? OK : FAIL);
    for (int d = 0; d < uniq_scales; ++d)
        scales[d] = oscale.scale;
    if (uniq_scales > 1) scales[uniq_scales - 1] /= 2.f;
    return scales;
}

int32_t *prb_t::generate_zero_points(int arg) {
    const attr_t::zero_points_t &zero_points = this->attr.zero_points;
    if (zero_points.is_def(arg)) return nullptr;

    const auto &e = zero_points.get(arg);
    assert(e.policy == policy_t::COMMON);

    int32_t *zp = (int32_t *)zmalloc(sizeof(int32_t), 4);
    SAFE_V(zp != nullptr ? OK : FAIL);
    zp[0] = e.value;
    return zp;
}

std::ostream &operator<<(std::ostream &s, const prb_t &prb) {
    dump_global_params(s);
    settings_t def;

    s << "--sdt=" << cfg2dt(prb.conf_in) << " ";
    s << "--ddt=" << cfg2dt(prb.conf_out) << " ";
    s << "--stag=" << prb.reorder.tag_in << " ";
    s << "--dtag=" << prb.reorder.tag_out << " ";

    if (canonical || prb.alg != def.alg[0])
        s << "--alg=" << alg2str(prb.alg) << " ";
    if (canonical || prb.oflag != def.oflag[0][0])
        s << "--oflag=" << flag2str(prb.oflag) << " ";
    if (canonical || prb.cross_engine != def.cross_engine[0])
        s << "--cross-engine=" << cross_engine2str(prb.cross_engine) << " ";
    if (canonical || prb.runtime_dim_mask != def.runtime_dim_mask[0])
        s << "--runtime-dim-mask=" << prb.runtime_dim_mask << " ";

    s << prb.attr;
    s << prb.reorder.dims;

    return s;
}

} // namespace reorder
