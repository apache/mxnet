/*******************************************************************************
* Copyright 2018-2020 Intel Corporation
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

#include "dnnl_common.hpp"
#include "dnnl_debug.hpp"

#include "shuffle/shuffle.hpp"

namespace shuffle {

std::ostream &operator<<(std::ostream &s, const prb_t &prb) {
    dump_global_params(s);
    settings_t def;

    if (canonical || prb.dir != def.dir[0]) s << "--dir=" << prb.dir << " ";
    if (canonical || prb.dt != def.dt[0]) s << "--dt=" << prb.dt << " ";
    if (canonical || prb.tag != def.tag[0]) s << "--tag=" << prb.tag << " ";
    if (canonical || prb.group != def.group[0])
        s << "--group=" << prb.group << " ";
    if (canonical || prb.axis != def.axis[0]) s << "--axis=" << prb.axis << " ";

    s << prb.attr;
    s << prb.dims;

    return s;
}

} // namespace shuffle
