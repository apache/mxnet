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

#ifndef COMMON_BIT_CAST_HPP
#define COMMON_BIT_CAST_HPP

#include <cstring>
#include <type_traits>

namespace dnnl {
namespace impl {
namespace utils {

// Returns a value of type T by reinterpretting the representation of the input
// value (part of C++20).
//
// Provides a safe implementation of type punning.
//
// Constraints:
// - U and T must have the same size
// - U and T must be trivially copyable
template <typename T, typename U>
inline T bit_cast(const U &u) {
    static_assert(sizeof(T) == sizeof(U), "Bit-casting must preserve size.");
    // Use std::is_pod as older GNU versions do not support
    // std::is_trivially_copyable.
    static_assert(std::is_pod<T>::value, "T must be trivially copyable.");
    static_assert(std::is_pod<U>::value, "U must be trivially copyable.");

    T t;
    std::memcpy(&t, &u, sizeof(U));
    return t;
}

} // namespace utils
} // namespace impl
} // namespace dnnl

#endif
