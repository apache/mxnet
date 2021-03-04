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

#include <assert.h>
#include <stdlib.h>
#include "lnorm/lnorm.hpp"

namespace lnorm {

flags_t str2flags(const char *str) {
    flags_t flags = bnorm::str2flags(str);
    assert(flags <= (GLOB_STATS | USE_SCALESHIFT));
    return flags;
}

std::ostream &operator<<(std::ostream &s, const prb_t &prb) {
    dump_global_params(s);
    settings_t def;

    if (canonical || prb.dir != def.dir[0]) s << "--dir=" << prb.dir << " ";
    if (canonical || prb.dt != def.dt[0]) s << "--dt=" << prb.dt << " ";
    if (canonical || prb.tag != def.tag[0]) s << "--tag=" << prb.tag << " ";
    if (canonical || prb.stat_tag != def.stat_tag[0])
        s << "--stat_tag=" << prb.stat_tag << " ";
    if (canonical || prb.flags != def.flags[0])
        s << "--flags=" << flags2str(prb.flags) << " ";
    if (canonical || prb.inplace != def.inplace[0])
        s << "--inplace=" << bool2str(prb.inplace) << " ";

    s << prb.attr;
    s << prb.dims;

    return s;
}
} // namespace lnorm
