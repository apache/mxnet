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

CPU_INST_TEST_CASE(SimpleSmall_Blocked_Attributes,
        PARAMS_ATTR(nhwc, FMT_WEIGHTS_BLOCKED16, FMT_BIAS, nhwc, 0.3f, COMMON,
                2, 1, 32, 13, 13, 32, 12, 12, 3, 3, 0, 0, 1, 1),
        PARAMS_ATTR(nhwc, FMT_WEIGHTS_BLOCKED16, FMT_BIAS, nhwc, 0.3f, COMMON,
                2, 1, 32, 13, 13, 32, 12, 12, 3, 3, 0, 0, 1, 1),
        PARAMS_ATTR(nhwc, FMT_WEIGHTS_BLOCKED16, FMT_BIAS, nhwc, 0.5f, COMMON,
                2, 1, 32, 13, 13, 32, 12, 12, 3, 3, 0, 0, 1, 1),
        PARAMS_ATTR(nhwc, FMT_WEIGHTS_BLOCKED16, FMT_BIAS, nhwc, 0.5f, COMMON,
                2, 1, 32, 13, 13, 32, 12, 12, 3, 3, 0, 0, 1, 1));

GPU_INST_TEST_CASE(SimpleSmall_Plain_Attributes,
        PARAMS_ATTR(nhwc, oihw, FMT_NO_BIAS, nchw, 0.3f, COMMON, 2, 1, 2, 1, 1,
                2, 1, 1, 1, 1, 0, 0, 1, 1),
        PARAMS_ATTR(nhwc, oihw, FMT_BIAS, nchw, 0.3f, COMMON, 2, 1, 2, 1, 1, 2,
                1, 1, 1, 1, 0, 0, 1, 1));
