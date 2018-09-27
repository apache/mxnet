/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

#pragma once

#include <limits>
#include <algorithm>
#include <cmath>

#include "hostdevice.h"

typedef enum {
    CTC_STATUS_SUCCESS = 0,
    CTC_STATUS_MEMOPS_FAILED = 1,
    CTC_STATUS_INVALID_VALUE = 2,
    CTC_STATUS_EXECUTION_FAILED = 3,
    CTC_STATUS_UNKNOWN_ERROR = 4
} ctcStatus_t;

typedef enum {
    CTC_CPU = 0,
    CTC_GPU = 1
} ctcComputeLocation;

namespace ctc_helper {

static const float threshold = 1e-1;

template<typename T>
HOSTDEVICE
T neg_inf() { return -T(INFINITY); }

inline int div_up(int x, int y) {
    return (x + y - 1) / y;
}

template <typename Arg, typename Res = Arg> struct maximum {
    HOSTDEVICE
    Res operator()(const Arg& x, const Arg& y) const {
        return x < y ? y : x;
    }
};

template <typename Arg, typename Res = Arg> struct add {
    HOSTDEVICE
    Res operator()(const Arg& x, const Arg& y) const {
        return x + y;
    }
};

template <typename Arg, typename Res = Arg> struct identity {
    HOSTDEVICE Res operator()(const Arg& x) const {return Res(x);}
};

template <typename Arg, typename Res = Arg> struct negate {
    HOSTDEVICE Res operator()(const Arg& x) const {return Res(-x);}
};

template <typename Arg, typename Res = Arg> struct exponential {
    HOSTDEVICE Res operator()(const Arg& x) const {return std::exp(x);}
};

template<typename Arg1, typename Arg2 = Arg1, typename Res=Arg1>
struct log_plus {
    typedef Res result_type;
    HOSTDEVICE
    Res operator()(const Arg1& p1, const Arg2& p2) {
        if (p1 == neg_inf<Arg1>())
            return p2;
        if (p2 == neg_inf<Arg2>())
            return p1;
        Res result = log1p(exp(-fabs(p1 - p2))) + maximum<Res>()(p1, p2);
        return result;
    }
};

}
