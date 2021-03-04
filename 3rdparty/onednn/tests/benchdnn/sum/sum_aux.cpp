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

#include "dnnl_debug.hpp"
#include "sum/sum.hpp"

namespace sum {

std::ostream &operator<<(std::ostream &s, const std::vector<float> &scales) {
    bool has_single_scale = true;
    for (size_t d = 0; d < scales.size() - 1; ++d)
        has_single_scale = has_single_scale && scales[d] == scales[d + 1];

    s << scales[0];
    if (!has_single_scale)
        for (size_t d = 1; d < scales.size(); ++d)
            s << ":" << scales[d];
    return s;
}

std::ostream &operator<<(std::ostream &s, const prb_t &prb) {
    using ::operator<<;
    using sum::operator<<;

    dump_global_params(s);
    settings_t def;

    bool has_default_tags = true;
    for (const auto &i_stag : prb.stag)
        has_default_tags = has_default_tags && i_stag == tag::abx;

    if (canonical || prb.sdt != def.sdt[0]) s << "--sdt=" << prb.sdt << " ";
    if (canonical || prb.ddt != def.ddt[0]) s << "--ddt=" << prb.ddt << " ";
    if (canonical || !has_default_tags) s << "--stag=" << prb.stag << " ";
    if (canonical || prb.dtag != def.dtag[0]) s << "--dtag=" << prb.dtag << " ";
    s << "--scales=" << prb.scales << " ";

    s << prb.attr;
    s << prb.dims;

    return s;
}

} // namespace sum
