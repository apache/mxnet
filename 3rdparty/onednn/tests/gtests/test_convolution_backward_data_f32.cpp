/*******************************************************************************
* Copyright 2016-2020 Intel Corporation
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

#include "dnnl_test_common.hpp"
#include "gtest/gtest.h"

#include "oneapi/dnnl/dnnl.hpp"
#include "test_convolution_backward_data_common.hpp"

namespace dnnl {

using convolution_test
        = convolution_backward_data_test<float, float, float, float>;

TEST_P(convolution_test, TestConvolution) {}

#define FP32
#define DIRECTION_BACKWARD_DATA
#include "convolution_common.h"

} // namespace dnnl
