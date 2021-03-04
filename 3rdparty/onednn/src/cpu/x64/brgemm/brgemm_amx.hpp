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

#ifndef CPU_X64_BRGEMM_BRGEMM_AMX_HPP
#define CPU_X64_BRGEMM_BRGEMM_AMX_HPP

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {
namespace brgemm_amx {

const static int max_m_block2 = 2;
const static int max_n_block2 = 2;

// Tile register decomposition
inline int get_C_tensor(int m, int n) {
    return (m * max_m_block2 + n);
}
inline int get_A_tensor(int m) {
    return (max_m_block2 * max_n_block2 + m);
}
inline int get_B_tensor(int n) {
    return (max_m_block2 * max_n_block2 + max_m_block2 + n);
}

} // namespace brgemm_amx
} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif

//vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s